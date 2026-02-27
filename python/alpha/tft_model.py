"""Temporal Fusion Transformer wrapper for equity return prediction.

Requires optional dependencies: torch, lightning, pytorch-forecasting.
Install with: pip install 'quant-platform[ml]'
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightning as pl_lightning
    import pytorch_forecasting as pf
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    HAS_TFT_DEPS = True
except ImportError:
    HAS_TFT_DEPS = False
    logger.info("TFT dependencies not available. Install with: pip install 'quant-platform[ml]'")


def _check_deps():
    if not HAS_TFT_DEPS:
        raise ImportError(
            "TFT requires pytorch-forecasting, torch, and lightning. "
            "Install with: pip install 'quant-platform[ml]'"
        )


class TFTAlphaModel:
    """Temporal Fusion Transformer for multi-horizon equity return prediction."""

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str = "target_5d",
        max_encoder_length: int = 60,
        max_prediction_length: int = 5,
        hidden_size: int = 32,
        attention_head_size: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
    ):
        _check_deps()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model: Optional["TemporalFusionTransformer"] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        val_frac: float = 0.2,
    ) -> tuple:
        """Prepare TimeSeriesDataSets for training and validation.

        Args:
            df: DataFrame with DatetimeIndex, columns: ticker, feature_cols, target_col.
            val_frac: Fraction of data to use for validation.

        Returns:
            (train_dataloader, val_dataloader)
        """
        data = df.copy()
        data["ticker"] = data["ticker"].astype(str)

        # Use global date rank as time index so all tickers share the same
        # calendar-aligned axis. Per-ticker cumcount causes misaligned splits
        # where the same time_idx maps to different calendar dates across tickers.
        if "date" not in data.columns:
            # Extract date from index (could be DatetimeIndex or MultiIndex)
            if hasattr(data.index, "get_level_values"):
                date_values = data.index.get_level_values(0)
            else:
                date_values = data.index
            data = data.reset_index(drop=True)
            data["_date"] = date_values
        else:
            data["_date"] = data["date"]
        unique_dates = sorted(data["_date"].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        data["time_idx"] = data["_date"].map(date_to_idx)
        data = data.drop(columns=["_date"])

        # Drop rows with NaN targets (filling with 0.0 fabricates zero-return labels)
        data = data.dropna(subset=[self.target_col])
        # Fill NaN in features only (tree models handle NaN, but TFT needs clean inputs)
        for col in self.feature_cols:
            data[col] = data[col].fillna(0.0)

        # pytorch-forecasting requires a unique index
        data = data.reset_index(drop=True)

        # Split by time
        max_time = data["time_idx"].max()
        train_cutoff = int(max_time * (1 - val_frac))

        training = TimeSeriesDataSet(
            data[data["time_idx"] <= train_cutoff],
            time_idx="time_idx",
            target=self.target_col,
            group_ids=["ticker"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=self.feature_cols + [self.target_col],
            static_categoricals=["ticker"],
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            data[data["time_idx"] > train_cutoff],
            stop_randomization=True,
        )

        train_dl = training.to_dataloader(train=True, batch_size=32, num_workers=0)
        val_dl = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

        return train_dl, val_dl

    def fit(
        self,
        train_dl,
        val_dl,
        max_epochs: int = 30,
    ) -> None:
        """Train the TFT model."""
        self.model = TemporalFusionTransformer.from_dataset(
            train_dl.dataset,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            loss=pf.metrics.QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = pl_lightning.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            enable_progress_bar=True,
            gradient_clip_val=0.1,
        )
        trainer.fit(self.model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    def predict(self, dataloader) -> np.ndarray:
        """Generate predictions from a dataloader."""
        predictions = self.model.predict(dataloader, return_x=False)
        return predictions.numpy()

    def get_attention_weights(self, dataloader) -> dict:
        """Extract interpretable attention weights for analysis."""
        interpretation = self.model.interpret_output(
            self.model.predict(dataloader, return_x=True, mode="raw"),
            reduction="sum",
        )
        return {
            "attention": interpretation["attention"],
            "static_variables": interpretation.get("static_variables"),
            "encoder_variables": interpretation.get("encoder_variables"),
            "decoder_variables": interpretation.get("decoder_variables"),
        }
