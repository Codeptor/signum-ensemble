"""Training pipeline for alpha models with MLflow tracking.

Integrates:
  - Purged walk-forward cross-validation (eliminates lookahead bias)
  - ModelEnsemble (LightGBM + CatBoost + RF with Ridge meta-learner)
  - SHAP explainability per fold (feature stability tracking)
  - Alpha decay analysis (IC at 1d/5d/10d/20d horizons)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from python.alpha.ensemble import ModelEnsemble

from python.alpha.features import (
    compute_alpha_features,
    compute_cross_sectional_features,
    compute_forward_returns,
    compute_residual_target,
    compute_winsorize_bounds,
    merge_macro_features,
    save_winsorize_bounds,
    winsorize,
)
from python.alpha.model import CrossSectionalModel
from python.data.ingestion import reshape_ohlcv_wide_to_long

logger = logging.getLogger(__name__)

# Phase 3: Reduced feature set — 8 orthogonal, interpretable features.
# Rationale (see docs/IMPROVEMENT_PLAN.md §2.3.2):
#   - Removed highly correlated features (ma_ratio_5/10/20/60 all similar)
#   - Removed duplicate info (bid_ask_proxy ≈ vol proxy)
#   - Removed noisy microstructure features (amihud_illiq, dollar_volume_20d)
#   - Kept only features with distinct predictive signals
FEATURE_COLS = [
    # Momentum (3) — short, medium-term, and Jegadeesh-Titman 12-1
    "ret_5d",
    "ret_20d",
    "mom_12_1",
    # Mean reversion (2) — RSI + Bollinger
    "rsi_14",
    "bb_position",
    # Volatility (2) — close-to-close + Yang-Zhang (OHLC, 8x more efficient)
    "vol_20d",
    "vol_yz_20d",
    # Cross-sectional (2) — relative strength + sector-relative momentum
    "cs_ret_rank_5d",
    "sector_rel_mom",
    # Macro regime (2) — VIX ratio + term spread give the model regime context
    # so it can learn that momentum works in low-vol and reversion in high-vol
    "vix_ma_ratio",
    "term_spread",
]

# Full feature set preserved for comparison / ablation studies
FEATURE_COLS_FULL = [
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "mom_12_1",
    "ma_ratio_5",
    "ma_ratio_10",
    "ma_ratio_20",
    "ma_ratio_60",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
    "mr_zscore_60",
    "volume_ratio",
    "dollar_volume_20d",
    "amihud_illiq",
    "bid_ask_proxy",
    "vol_yz_20d",
    "vol_park_20d",
    # Cross-sectional features
    "cs_ret_rank_5d",
    "cs_ret_rank_20d",
    "cs_vol_rank_20d",
    "cs_volume_rank",
    "sector_rel_mom",
    # Macro regime features
    "vix",
    "vix_ma_ratio",
    "term_spread",
    "term_spread_change_20d",
]

# Alpha decay horizons (trading days)
DECAY_HORIZONS = [1, 5, 10, 20]


def _purged_walk_forward_cv(
    labeled: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_5d",
    n_splits: int = 5,
    embargo_days: int = 22,
    label_horizon: int = 5,
    use_ensemble: bool = True,
) -> dict:
    """Purged walk-forward cross-validation with SHAP and alpha decay.

    Replaces the simple date split with expanding-window walk-forward
    validation. Each fold has:
      - Purge: removes training samples whose labels (forward returns)
        overlap with the test window's time range.
      - Embargo: additional buffer after purge to prevent serial correlation
        leakage from autocorrelated features.

    Returns a dict with:
      - model: the final trained model (ensemble or LightGBM)
      - fold_ics: list of per-fold OOS ICs
      - mean_ic: mean OOS IC across folds
      - shap_stability: SHAP feature stability metrics
      - alpha_decay: IC decay across holding periods
      - bounds: winsorize bounds from the final training split
    """
    dates = labeled.index.get_level_values(0).unique().sort_values()
    n_dates = len(dates)

    # Walk-forward: expanding train window, fixed-size test window
    min_train_dates = int(n_dates * 0.4)  # At least 40% for first fold
    test_size = max(20, (n_dates - min_train_dates) // n_splits)
    embargo_offset = pd.tseries.offsets.BDay(embargo_days)
    purge_offset = pd.tseries.offsets.BDay(label_horizon)

    fold_ics = []
    fold_shap_dfs = []
    fold_predictions = []  # For alpha decay analysis

    for fold in range(n_splits):
        # Expanding window: train on all data up to split point
        train_end_idx = min_train_dates + fold * test_size
        if train_end_idx >= n_dates:
            break

        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size, n_dates)

        if test_start_idx >= n_dates:
            break

        train_end_date = dates[train_end_idx - 1]

        # PURGE: Remove training samples whose forward-return labels
        # overlap with the test window. A sample at date T has a label
        # computed from prices at T+1..T+horizon. If T+horizon falls
        # within the test window, that sample is contaminated.
        purge_cutoff = train_end_date - purge_offset
        train_fold = labeled.loc[labeled.index.get_level_values(0) <= purge_cutoff]

        # EMBARGO: additional buffer after purge for serial correlation
        embargo_date = train_end_date + embargo_offset
        test_start_date = dates[min(test_start_idx, n_dates - 1)]
        test_end_date = dates[min(test_end_idx - 1, n_dates - 1)]

        # Ensure test starts after embargo
        if test_start_date < embargo_date:
            post_embargo = dates[dates >= embargo_date]
            if len(post_embargo) == 0:
                logger.warning(f"Fold {fold}: no dates after embargo, skipping")
                continue
            test_start_date = post_embargo[0]
            # Extend test_end to maintain target test_size after embargo shift
            test_end_candidates = dates[dates >= test_start_date]
            if len(test_end_candidates) >= test_size:
                test_end_date = test_end_candidates[test_size - 1]
            elif len(test_end_candidates) > 0:
                test_end_date = test_end_candidates[-1]

        test_fold = labeled.loc[
            (labeled.index.get_level_values(0) >= test_start_date)
            & (labeled.index.get_level_values(0) <= test_end_date)
        ]

        n_purged = len(
            labeled.loc[
                (labeled.index.get_level_values(0) > purge_cutoff)
                & (labeled.index.get_level_values(0) <= train_end_date)
            ]
        )
        if n_purged > 0:
            logger.info(f"Fold {fold}: purged {n_purged} rows with overlapping labels")

        if len(train_fold) < 100 or len(test_fold) < 10:
            logger.warning(
                f"Fold {fold}: insufficient data "
                f"(train={len(train_fold)}, test={len(test_fold)}), skipping"
            )
            continue

        # Winsorize using train-only bounds
        bounds = compute_winsorize_bounds(train_fold)
        train_fold = winsorize(train_fold, bounds=bounds)
        test_fold = winsorize(test_fold, bounds=bounds)

        # Train model
        if use_ensemble:
            from python.alpha.ensemble import ModelEnsemble

            model = ModelEnsemble(feature_cols=feature_cols)
        else:
            model = CrossSectionalModel(model_type="lightgbm", feature_cols=feature_cols)

        # Split val from train end for early stopping (last 15% of train)
        val_split_date = dates[int(train_end_idx * 0.85)]
        val_fold = train_fold.loc[train_fold.index.get_level_values(0) > val_split_date]
        train_actual = train_fold.loc[train_fold.index.get_level_values(0) <= val_split_date]

        if len(val_fold) > 30:
            model.fit(train_actual, target_col=target_col, val_df=val_fold)
        else:
            model.fit(train_fold, target_col=target_col)

        # OOS predictions
        test_preds = model.predict(test_fold)
        ic, _ = spearmanr(test_preds, test_fold[target_col].values)
        ic = float(ic) if not np.isnan(ic) else 0.0
        fold_ics.append(ic)

        logger.info(
            f"Fold {fold}: train={len(train_fold)}, test={len(test_fold)}, "
            f"IC={ic:.4f} "
            f"(train: {train_fold.index.get_level_values(0).min().date()} to {train_end_date.date()}, "
            f"test: {test_start_date.date()} to {test_end_date.date()})"
        )

        # SHAP per fold (best-effort)
        # P1-20 fix: when using an ensemble, compute SHAP for all tree-based
        # sub-models and average the importance.  Previously only LightGBM was
        # explained, giving a misleading picture of what drives the ensemble.
        try:
            from python.alpha.explainability import compute_shap_importance

            if hasattr(model, "lgbm") and hasattr(model, "rf"):
                # Ensemble: aggregate SHAP from all tree-based sub-models
                sub_shap_dfs = []
                # LightGBM (weight 0.45)
                lgbm_model = model.lgbm.model
                if lgbm_model is not None:
                    sub_shap_dfs.append(
                        compute_shap_importance(lgbm_model, test_fold[feature_cols], feature_cols)
                    )
                # Random Forest (weight 0.25)
                if model.rf is not None:
                    try:
                        sub_shap_dfs.append(
                            compute_shap_importance(model.rf, test_fold[feature_cols], feature_cols)
                        )
                    except Exception:
                        pass  # RF SHAP can fail on large forests
                if sub_shap_dfs:
                    # Average importance across sub-models
                    combined = pd.concat(sub_shap_dfs)
                    shap_df = combined.groupby(combined.index).mean()
                    fold_shap_dfs.append(shap_df)
            else:
                # Single model
                shap_model = model.model if hasattr(model, "model") else None
                if shap_model is not None:
                    shap_df = compute_shap_importance(
                        shap_model, test_fold[feature_cols], feature_cols
                    )
                    fold_shap_dfs.append(shap_df)
        except Exception as e:
            logger.debug(f"SHAP failed for fold {fold}: {e}")

        # Store predictions for alpha decay analysis
        if "ticker" in test_fold.columns:
            ticker_labels = test_fold["ticker"].values
        elif hasattr(test_fold.index, "get_level_values") and test_fold.index.nlevels > 1:
            ticker_labels = test_fold.index.get_level_values(-1).values
        else:
            ticker_labels = test_fold.index.values

        pred_series = pd.Series(
            test_preds,
            index=pd.MultiIndex.from_arrays(
                [test_fold.index.get_level_values(0), ticker_labels],
                names=["date", "ticker"],
            ),
        )
        fold_predictions.append(pred_series)

    # Aggregate results
    mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
    std_ic = float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0

    logger.info(
        f"Purged walk-forward CV: {len(fold_ics)} folds, mean IC={mean_ic:.4f} ± {std_ic:.4f}"
    )

    # SHAP stability across folds
    shap_stability = {}
    if len(fold_shap_dfs) >= 2:
        try:
            from python.alpha.explainability import shap_stability_across_folds

            shap_stability = shap_stability_across_folds(fold_shap_dfs, top_k=5)
            logger.info(
                f"SHAP stability: top-5 overlap={shap_stability.get('top_k_overlap', 0):.2f}, "
                f"rank corr={shap_stability.get('rank_correlation', 0):.2f}, "
                f"always top-5: {shap_stability.get('always_top_k', [])}"
            )
        except Exception as e:
            logger.debug(f"SHAP stability analysis failed: {e}")

    # Alpha decay from OOS predictions
    alpha_decay = {}
    if fold_predictions:
        try:
            all_preds = pd.concat(fold_predictions)
            alpha_decay = _compute_alpha_decay(all_preds, labeled, feature_cols)
        except Exception as e:
            logger.debug(f"Alpha decay analysis failed: {e}")

    # Train final model on ALL data for deployment
    logger.info("Training final model on full dataset for deployment...")
    final_bounds = compute_winsorize_bounds(labeled)
    save_winsorize_bounds(final_bounds)
    labeled_w = winsorize(labeled, bounds=final_bounds)

    # Final train/val split (80/20 with embargo for early stopping)
    final_split_date = dates[int(len(dates) * 0.85)]
    final_embargo_date = final_split_date + embargo_offset
    final_train = labeled_w.loc[labeled_w.index.get_level_values(0) <= final_split_date]
    final_val = labeled_w.loc[labeled_w.index.get_level_values(0) >= final_embargo_date]

    if use_ensemble:
        from python.alpha.ensemble import ModelEnsemble

        final_model = ModelEnsemble(feature_cols=feature_cols)
    else:
        final_model = CrossSectionalModel(model_type="lightgbm", feature_cols=feature_cols)

    if len(final_val) > 30:
        final_model.fit(final_train, target_col=target_col, val_df=final_val)
    else:
        final_model.fit(labeled_w, target_col=target_col)

    # Attach CV metrics to model.
    # Prefer the model's own validation IC (from fit/meta-learner on held-out data)
    # over the CV mean IC. Only fall back to CV mean if the model didn't compute one
    # (e.g., insufficient val data for 3-way split, or single-model mode).
    if getattr(final_model, "validation_ic", 0.0) < 1e-10:
        final_model.validation_ic = mean_ic
    final_model.cv_mean_ic = mean_ic  # Always store CV mean for reference
    final_model.cv_fold_ics = fold_ics
    final_model.shap_stability = shap_stability
    final_model.alpha_decay = alpha_decay

    return {
        "model": final_model,
        "fold_ics": fold_ics,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "shap_stability": shap_stability,
        "alpha_decay": alpha_decay,
        "fold_shap_dfs": fold_shap_dfs,
        "bounds": final_bounds,
    }


def _compute_alpha_decay(
    predictions: pd.Series,
    labeled: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Compute IC at multiple holding periods to measure signal half-life.

    Returns dict with horizon -> IC mapping and estimated half-life.
    """
    # Need close prices to compute forward returns at different horizons
    # Use the labeled data's close prices
    if "close" not in labeled.columns:
        logger.debug("No 'close' column for alpha decay — skipping")
        return {}

    # Build returns DataFrame (tickers as columns)
    try:
        close_pivot = labeled["close"].unstack(level=-1)
    except Exception:
        logger.debug("Cannot pivot close prices for alpha decay")
        return {}

    daily_returns = close_pivot.pct_change()

    decay_results = {}
    for horizon in DECAY_HORIZONS:
        # Compute forward returns at this horizon
        fwd_returns = close_pivot.pct_change(horizon).shift(-horizon)

        ics = []
        pred_dates = predictions.index.get_level_values(0).unique()
        for dt in pred_dates:
            if dt not in fwd_returns.index:
                continue

            try:
                date_preds = predictions.loc[dt]
                if isinstance(date_preds, pd.Series):
                    tickers = date_preds.index
                else:
                    continue

                date_returns = fwd_returns.loc[dt]
                common = tickers.intersection(date_returns.index)
                if len(common) < 5:
                    continue

                p = date_preds.loc[common].values.astype(float)
                r = date_returns.loc[common].values.astype(float)

                valid = ~(np.isnan(p) | np.isnan(r))
                if valid.sum() < 5:
                    continue

                ic_val, _ = spearmanr(p[valid], r[valid])
                if not np.isnan(ic_val):
                    ics.append(ic_val)
            except Exception:
                continue

        if ics:
            decay_results[horizon] = {
                "ic": float(np.mean(ics)),
                "ic_std": float(np.std(ics)),
                "ir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
                "n_dates": len(ics),
            }

    if decay_results:
        logger.info("Alpha decay profile:")
        for h, d in sorted(decay_results.items()):
            logger.info(f"  {h}d: IC={d['ic']:.4f} ± {d['ic_std']:.4f}, IR={d['ir']:.2f}")

        # Estimate half-life: find horizon where IC drops below 50% of peak
        peak_ic = max(d["ic"] for d in decay_results.values())
        half_life = None
        for h in sorted(decay_results.keys()):
            if decay_results[h]["ic"] < peak_ic * 0.5:
                half_life = h
                break
        if half_life:
            logger.info(f"  Estimated signal half-life: ~{half_life} trading days")
            decay_results["half_life"] = half_life
        else:
            logger.info("  Signal half-life: >20 trading days (slow decay)")
            decay_results["half_life"] = None

    return decay_results


def run_training(
    data_path: str = "data/raw/sp500_ohlcv.parquet",
    use_ensemble: bool = True,
    n_cv_splits: int = 5,
) -> "CrossSectionalModel | ModelEnsemble":
    """Full training pipeline with purged walk-forward CV.

    Pipeline:
      1. Load and feature-engineer data
      2. Run purged walk-forward CV (5 folds, 22-day embargo)
      3. Compute SHAP importance per fold + stability
      4. Compute alpha decay profile (IC at 1d/5d/10d/20d)
      5. Train final ensemble on full data for deployment
      6. Log all metrics to MLflow
    """
    raw = pd.read_parquet(data_path)
    raw = reshape_ohlcv_wide_to_long(raw)
    # P0-6 fix: skip winsorization here — the CV loop and final training
    # manage their own per-fold winsorization with train-only bounds.
    # Previously, features were winsorized twice (here with data quantiles,
    # then again in CV/final with train-only bounds), causing train/serve
    # skew because inference only winsorizes once.
    featured = compute_alpha_features(raw, skip_winsorize=True)
    featured = compute_cross_sectional_features(featured)
    featured = merge_macro_features(featured)
    labeled = compute_forward_returns(featured, horizon=5)
    labeled = compute_residual_target(labeled, horizon=5)
    labeled = labeled.dropna(subset=FEATURE_COLS + ["target_5d"])

    # Run purged walk-forward CV with ensemble, SHAP, and alpha decay
    cv_result = _purged_walk_forward_cv(
        labeled,
        feature_cols=FEATURE_COLS,
        n_splits=n_cv_splits,
        use_ensemble=use_ensemble,
    )

    model = cv_result["model"]
    fold_ics = cv_result["fold_ics"]
    mean_ic = cv_result["mean_ic"]
    std_ic = cv_result["std_ic"]
    shap_stability = cv_result["shap_stability"]
    alpha_decay = cv_result["alpha_decay"]

    with mlflow.start_run(run_name="ensemble_purged_cv" if use_ensemble else "lgbm_purged_cv"):
        mlflow.log_params(model.params if hasattr(model, "params") else {})
        mlflow.log_metric("cv_mean_ic", mean_ic)
        mlflow.log_metric("cv_std_ic", std_ic)
        mlflow.log_metric("cv_n_folds", len(fold_ics))
        mlflow.log_metric("cv_sharpe_of_ic", mean_ic / std_ic if std_ic > 0 else 0.0)

        for i, ic in enumerate(fold_ics):
            mlflow.log_metric(f"fold_{i}_ic", ic)

        # SHAP stability metrics
        if shap_stability:
            mlflow.log_metric("shap_top5_overlap", shap_stability.get("top_k_overlap", 0))
            mlflow.log_metric("shap_rank_corr", shap_stability.get("rank_correlation", 0))
            always = shap_stability.get("always_top_k", [])
            if always:
                mlflow.log_param("shap_stable_features", ",".join(always))

        # Alpha decay metrics
        for horizon, decay_info in alpha_decay.items():
            if isinstance(horizon, int):
                mlflow.log_metric(f"ic_{horizon}d", decay_info["ic"])
                mlflow.log_metric(f"ir_{horizon}d", decay_info["ir"])
        if "half_life" in alpha_decay and alpha_decay["half_life"]:
            mlflow.log_metric("signal_half_life_days", alpha_decay["half_life"])

        importance = model.feature_importance()
        logger.info(f"Top features:\n{importance.head(10)}")
        logger.info(f"CV mean IC (Spearman): {mean_ic:.4f} ± {std_ic:.4f}")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
