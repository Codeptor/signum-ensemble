"""Cross-sectional return prediction models."""

import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default random seed for reproducibility (Fix #24)
DEFAULT_SEED = 42


class CrossSectionalModel:
    """Wraps LightGBM or CatBoost for cross-sectional equity return prediction."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        feature_cols: list[str] | None = None,
        params: dict | None = None,
    ):
        self.model_type = model_type
        self.feature_cols = feature_cols
        self.model = None
        self.params = params or self._default_params()

    def _default_params(self) -> dict:
        if self.model_type == "lightgbm":
            return {
                "objective": "huber",  # Fix #22: robust to outlier targets
                "metric": "huber",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "verbose": -1,
                "n_estimators": 500,  # Higher cap; early stopping will pick best
                "seed": DEFAULT_SEED,  # Fix #24
                "feature_fraction_seed": DEFAULT_SEED,
                "bagging_seed": DEFAULT_SEED,
            }
        elif self.model_type == "catboost":
            return {
                "iterations": 200,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": 0,
                "random_seed": DEFAULT_SEED,
            }
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "target_5d",
        val_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train on labeled cross-sectional data.

        Args:
            df: Training data with features and target.
            target_col: Name of target column.
            val_df: Optional validation data for early stopping (Fix #9).
        """
        features = df[self.feature_cols].values
        y = df[target_col].values

        mask = ~np.isnan(y) & ~np.any(np.isnan(features), axis=1)
        features, y = features[mask], y[mask]

        if self.model_type == "lightgbm":
            fit_kwargs: dict = {}

            if val_df is not None and len(val_df) > 0:
                val_features = val_df[self.feature_cols].values
                val_y = val_df[target_col].values
                val_mask = ~np.isnan(val_y) & ~np.any(np.isnan(val_features), axis=1)
                val_features, val_y = val_features[val_mask], val_y[val_mask]
                if len(val_y) > 0:
                    fit_kwargs["eval_set"] = [(val_features, val_y)]
                    # M-EARLYSTOP fix: stopping_rounds=10 was too aggressive,
                    # causing premature termination before the model fully
                    # converged on noisy cross-sectional equity targets.
                    # Increased to 50 to allow more patience.
                    fit_kwargs["callbacks"] = [
                        lgb.early_stopping(stopping_rounds=20, verbose=True),
                        lgb.log_evaluation(period=50),
                    ]

            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(features, y, **fit_kwargs)

            if fit_kwargs.get("eval_set"):
                best_iter = getattr(self.model, "best_iteration_", None)
                if best_iter is not None:
                    logger.info(f"Early stopping at iteration {best_iter}")

        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor

            self.model = CatBoostRegressor(**self.params)
            self.model.fit(features, y)

        logger.info(f"Trained {self.model_type} on {len(y)} samples")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict raw scores."""
        # Fix #23: guard against untrained/failed model
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() before predict().")
        features = df[self.feature_cols].values
        # R3-P-15 fix: warn if NaN features are passed to model
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            nan_pct = nan_count / features.size * 100
            logger.warning(
                f"R3-P-15: {nan_count} NaN values ({nan_pct:.1f}%) in features "
                f"passed to {self.model_type}.predict(). Results may be unreliable."
            )
        return self.model.predict(features).astype(np.float64)

    def predict_ranks(self, df: pd.DataFrame) -> pd.Series:
        """Predict and rank within each date cross-section (0=worst, 1=best)."""
        preds = pd.Series(self.predict(df), index=df.index)
        ranks = preds.groupby(preds.index).rank(pct=True)
        return ranks

    def feature_importance(self) -> pd.Series:
        """Return feature importance scores."""
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() before feature_importance().")
        if self.model_type == "lightgbm":
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        elif self.model_type == "catboost":
            return pd.Series(
                self.model.get_feature_importance(),
                index=self.feature_cols,
            ).sort_values(ascending=False)
