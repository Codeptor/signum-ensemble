"""SHAP-based model explainability and alpha decay analysis.

Provides per-cycle feature importance via TreeSHAP, cross-cycle stability
tracking, and alpha decay analysis (IC as a function of holding period).

References:
  - SHAP: Lundberg & Lee, 2017
  - Alpha decay: standard quant practice — plot rank IC at horizons 1..20d
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
    max_samples: int = 5000,
) -> pd.DataFrame:
    """Compute SHAP feature importance for a tree-based model.

    Args:
        model: A fitted tree model (LightGBM, CatBoost, RF, or XGBoost).
        X: Feature matrix (DataFrame or array).
        feature_names: Optional feature names (inferred from X if DataFrame).
        max_samples: Cap on samples for speed (TreeSHAP is O(TLD) per sample).

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank.
    """
    import shap

    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X_arr = X.values
    else:
        X_arr = X
        feature_names = feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

    # Subsample for speed
    if len(X_arr) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_arr), max_samples, replace=False)
        X_arr = X_arr[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)

    mean_abs = np.abs(shap_values).mean(axis=0)
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    })
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def shap_stability_across_folds(
    fold_shap_dfs: list[pd.DataFrame],
    top_k: int = 5,
) -> dict:
    """Analyze SHAP feature importance stability across walk-forward folds.

    If the top-5 features change significantly across folds, the model
    may be regime-dependent.

    Args:
        fold_shap_dfs: List of DataFrames from compute_shap_importance, one per fold.
        top_k: Number of top features to track.

    Returns:
        Dict with stability metrics:
          - top_k_overlap: Jaccard similarity of top-k features across fold pairs
          - always_top_k: Features in top-k of every fold
          - never_top_k: Features never in top-k of any fold
          - rank_correlation: Mean Spearman rank correlation across fold pairs
    """
    from scipy.stats import spearmanr

    if not fold_shap_dfs:
        return {"top_k_overlap": 0.0, "always_top_k": [], "never_top_k": []}

    all_features = set(fold_shap_dfs[0]["feature"])
    top_k_sets = []

    for df in fold_shap_dfs:
        top_features = set(df.nlargest(top_k, "mean_abs_shap")["feature"])
        top_k_sets.append(top_features)

    # Jaccard overlap across all pairs
    overlaps = []
    rank_corrs = []
    for i in range(len(top_k_sets)):
        for j in range(i + 1, len(top_k_sets)):
            intersection = top_k_sets[i] & top_k_sets[j]
            union = top_k_sets[i] | top_k_sets[j]
            overlaps.append(len(intersection) / len(union) if union else 0)

            # Rank correlation between full importance rankings
            df_i = fold_shap_dfs[i].set_index("feature")["mean_abs_shap"]
            df_j = fold_shap_dfs[j].set_index("feature")["mean_abs_shap"]
            common = df_i.index.intersection(df_j.index)
            if len(common) > 2:
                rc, _ = spearmanr(df_i.loc[common], df_j.loc[common])
                if not np.isnan(rc):
                    rank_corrs.append(rc)

    always = set.intersection(*top_k_sets) if top_k_sets else set()
    ever = set.union(*top_k_sets) if top_k_sets else set()
    never = all_features - ever

    return {
        "top_k_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "rank_correlation": float(np.mean(rank_corrs)) if rank_corrs else 0.0,
        "always_top_k": sorted(always),
        "never_top_k": sorted(never),
        "n_folds": len(fold_shap_dfs),
    }


def alpha_decay_curve(
    predictions: pd.Series,
    returns_df: pd.DataFrame,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Compute rank IC as a function of holding period (alpha decay curve).

    Shows how quickly the signal decays — a steep decline suggests the model
    captures short-lived momentum; a flat curve suggests structural alpha.

    Args:
        predictions: Series of model predictions indexed by (date, ticker).
        returns_df: DataFrame of daily returns, columns = tickers.
        horizons: List of forward return horizons in trading days.
            Default: [1, 2, 3, 5, 10, 15, 20].

    Returns:
        DataFrame with columns: horizon, ic, ic_std, ir (IC / std).
    """
    from scipy.stats import spearmanr

    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 15, 20]

    results = []

    for h in horizons:
        # Compute forward returns at this horizon
        fwd_returns = returns_df.pct_change(h).shift(-h)

        ics = []
        # Compute IC per date
        pred_dates = predictions.index.get_level_values(0).unique()
        for date in pred_dates:
            if date not in fwd_returns.index:
                continue

            date_preds = predictions.loc[date]
            if isinstance(date_preds, pd.Series):
                tickers = date_preds.index
            else:
                continue

            date_returns = fwd_returns.loc[date]
            common_tickers = tickers.intersection(date_returns.index)
            if len(common_tickers) < 5:
                continue

            p = date_preds.loc[common_tickers].values
            r = date_returns.loc[common_tickers].values

            valid = ~(np.isnan(p) | np.isnan(r))
            if valid.sum() < 5:
                continue

            ic, _ = spearmanr(p[valid], r[valid])
            if not np.isnan(ic):
                ics.append(ic)

        if ics:
            results.append({
                "horizon": h,
                "ic": float(np.mean(ics)),
                "ic_std": float(np.std(ics)),
                "ir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
                "n_dates": len(ics),
            })

    return pd.DataFrame(results)
