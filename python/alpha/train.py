"""Training pipeline for alpha models with MLflow tracking."""

import logging

import mlflow
import pandas as pd

from python.alpha.features import (
    compute_alpha_features,
    compute_cross_sectional_features,
    compute_forward_returns,
    compute_residual_target,
    merge_macro_features,
)
from python.alpha.model import CrossSectionalModel
from python.data.ingestion import reshape_ohlcv_wide_to_long

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_5d",
    "ret_10d",
    "ret_20d",
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
    "volume_ratio",
    "hl_range",
    "dollar_volume_20d",
    "amihud_illiq",
    "bid_ask_proxy",
    # Cross-sectional features
    "cs_ret_rank_5d",
    "cs_ret_rank_20d",
    "cs_vol_rank_20d",
    "cs_volume_rank",
]


def run_training(data_path: str = "data/raw/sp500_ohlcv.parquet"):
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
    # 5-day embargo gap to avoid target leakage across the boundary
    embargo_offset = pd.tseries.offsets.BDay(5)
    embargo_date = split_date + embargo_offset

    train = labeled.loc[labeled.index.get_level_values(0) <= split_date]
    val = labeled.loc[labeled.index.get_level_values(0) >= embargo_date]

    logger.info(
        f"Train: {len(train)} rows up to {split_date.date()}, "
        f"Val: {len(val)} rows from {embargo_date.date()} "
        f"(5-day embargo)"
    )

    with mlflow.start_run(run_name="lgbm_alpha158"):
        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col="target_5d", val_df=val)

        val_preds = model.predict(val)
        ic = pd.Series(val_preds, index=val.index).corr(val["target_5d"])

        mlflow.log_params(model.params)
        mlflow.log_metric("information_coefficient", ic)
        mlflow.log_metric("train_size", len(train))
        mlflow.log_metric("val_size", len(val))

        importance = model.feature_importance()
        logger.info(f"Top features:\n{importance.head(10)}")
        logger.info(f"Validation IC: {ic:.4f}")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
