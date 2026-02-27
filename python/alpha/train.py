"""Training pipeline for alpha models with MLflow tracking."""

import logging

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    # Mean reversion (3) — RSI, Bollinger, z-score
    "rsi_14",
    "bb_position",
    "mr_zscore_60",
    # Volatility (2) — close-to-close + Yang-Zhang (OHLC, 8x more efficient)
    "vol_20d",
    "vol_yz_20d",
    # Volume (1) — confirm momentum with volume
    "volume_ratio",
    # Cross-sectional (2) — relative strength + sector-relative momentum
    "cs_ret_rank_5d",
    "sector_rel_mom",
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


def run_training(data_path: str = "data/raw/sp500_ohlcv.parquet") -> "CrossSectionalModel":
    """Full training pipeline: load data -> features -> train -> log to MLflow."""
    raw = pd.read_parquet(data_path)
    raw = reshape_ohlcv_wide_to_long(raw)
    featured = compute_alpha_features(raw)
    featured = compute_cross_sectional_features(featured)
    featured = merge_macro_features(featured)
    labeled = compute_forward_returns(featured, horizon=5)
    labeled = compute_residual_target(labeled, horizon=5)
    labeled = labeled.dropna(subset=FEATURE_COLS + ["target_5d"])

    # Date-based split with embargo gap to prevent look-ahead bias.
    # iloc split is wrong for panel data: rows from the same date can land
    # in both train and val sets. Instead, sort dates and split by date.
    dates = labeled.index.get_level_values(0).unique().sort_values()
    split_date = dates[int(len(dates) * 0.8)]
    # 22-day embargo gap to cover the longest feature lookback window
    # (20-day returns, volatility, Bollinger) and prevent information leakage
    # from features that straddle the train/val boundary.
    embargo_offset = pd.tseries.offsets.BDay(22)
    embargo_date = split_date + embargo_offset

    train = labeled.loc[labeled.index.get_level_values(0) <= split_date]
    val = labeled.loc[labeled.index.get_level_values(0) >= embargo_date]

    # Compute and save winsorize bounds from TRAINING split only (no val leakage).
    # Previously this was missing entirely — models trained via run_training()
    # had no saved bounds file, causing inference to use stale/wrong bounds.
    bounds = compute_winsorize_bounds(train)
    save_winsorize_bounds(bounds)
    logger.info(f"Saved winsorize bounds ({len(bounds)} cols) from training split")

    # Re-winsorize both splits using training-only bounds
    train = winsorize(train, bounds=bounds)
    if len(val) > 0:
        val = winsorize(val, bounds=bounds)

    logger.info(
        f"Train: {len(train)} rows up to {split_date.date()}, "
        f"Val: {len(val)} rows from {embargo_date.date()} "
        f"(22-day embargo)"
    )

    with mlflow.start_run(run_name="lgbm_alpha158"):
        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col="target_5d", val_df=val)

        # Use Spearman rank IC (not Pearson) — standard for cross-sectional
        # equity models, invariant to monotonic transforms and robust to outliers.
        val_preds = model.predict(val)
        ic, _ = spearmanr(val_preds, val["target_5d"].values)
        ic = float(ic) if not np.isnan(ic) else 0.0

        mlflow.log_params(model.params)
        mlflow.log_metric("information_coefficient", ic)
        mlflow.log_metric("train_size", len(train))
        mlflow.log_metric("val_size", len(val))

        importance = model.feature_importance()
        logger.info(f"Top features:\n{importance.head(10)}")
        logger.info(f"Validation IC (Spearman): {ic:.4f}")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
