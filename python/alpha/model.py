"""Cross-sectional return prediction models."""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 250,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "verbose": -1,
                "n_estimators": 150,
            }
        elif self.model_type == "catboost":
            return {
                "iterations": 200,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": 0,
            }
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, df: pd.DataFrame, target_col: str = "target_5d") -> None:
        """Train on labeled cross-sectional data."""
        features = df[self.feature_cols].values
        y = df[target_col].values

        mask = ~np.isnan(y) & ~np.any(np.isnan(features), axis=1)
        features, y = features[mask], y[mask]

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(features, y)
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor

            self.model = CatBoostRegressor(**self.params)
            self.model.fit(features, y)

        logger.info(f"Trained {self.model_type} on {len(y)} samples")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict raw scores."""
        features = df[self.feature_cols].values
        return self.model.predict(features).astype(np.float64)

    def predict_ranks(self, df: pd.DataFrame) -> pd.Series:
        """Predict and rank within each date cross-section (0=worst, 1=best)."""
        preds = pd.Series(self.predict(df), index=df.index)
        ranks = preds.groupby(preds.index).rank(pct=True)
        return ranks

    def feature_importance(self) -> pd.Series:
        """Return feature importance scores."""
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
