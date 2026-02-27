"""Model ensemble for robust cross-sectional return predictions.

**Status: Research-only.**  This module is NOT wired into the live pipeline.
``predict.py`` and ``train.py`` use ``CrossSectionalModel`` (single LightGBM)
directly.  The ensemble is available for offline experiments but requires
explicit integration to use in production.

Combines LightGBM and Random Forest with IC-weighted averaging
to reduce variance and overfitting (Phase 3, §2.3.3).

Each sub-model captures different aspects of the signal:
  - LightGBM: Non-linear interactions (gradient boosting)
  - Random Forest: Robust to outliers (bagging)

Note: Elastic Net was removed because features span ~12 orders of
magnitude (RSI: 0-100, amihud_illiq: ~1e-12) and ElasticNet's L1/L2
penalty is scale-sensitive. Without per-feature standardization the
Elastic Net produced near-zero or garbage predictions. Since LightGBM
and RF are scale-invariant, they don't need this fix. The ensemble
is cleaner and more reliable as a 2-model combination.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from python.alpha.model import DEFAULT_SEED, CrossSectionalModel

logger = logging.getLogger(__name__)


def _safe_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute rank Information Coefficient with NaN/zero-variance guard.

    C-CORR fix: ``np.corrcoef`` returns NaN when either array has zero
    variance, and ``max(0.0, nan)`` propagates nondeterministically in
    Python.  This helper returns 0.0 in all degenerate cases.

    H-PEARSON fix: uses Spearman rank correlation instead of Pearson.
    Rank IC is the standard metric in cross-sectional equity models
    because it is invariant to monotonic transformations of the signal
    and robust to outliers.
    """
    from scipy.stats import spearmanr

    if len(pred) < 2 or len(actual) < 2:
        return 0.0
    if np.std(pred) < 1e-10 or np.std(actual) < 1e-10:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    ic = float(ic)
    return 0.0 if (np.isnan(ic) or ic < 0) else ic


# Default ensemble weights (prior to IC-based calibration)
DEFAULT_WEIGHTS = {
    "lightgbm": 0.60,
    "random_forest": 0.40,
}


class ModelEnsemble:
    """Ensemble of LightGBM + Random Forest.

    IC-based weights from a validation set.

    Compatible with CrossSectionalModel interface (fit/predict on DataFrames).
    """

    def __init__(
        self,
        feature_cols: list[str],
        weights: Optional[dict[str, float]] = None,
    ):
        """Initialize ensemble sub-models.

        Args:
            feature_cols: Feature column names used by all sub-models.
            weights: Optional dict of model_name -> weight. Defaults to
                DEFAULT_WEIGHTS (60/40 split for lgbm/rf).
        """
        self.feature_cols = feature_cols
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self._fitted = False

        # Sub-models
        self.lgbm = CrossSectionalModel(
            model_type="lightgbm",
            feature_cols=feature_cols,
        )
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=100,
            n_jobs=-1,
            random_state=DEFAULT_SEED,
        )

    @property
    def models(self) -> dict[str, object]:
        """Return sub-models as a dict for iteration."""
        return {
            "lightgbm": self.lgbm,
            "random_forest": self.rf,
        }

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "target_5d",
        val_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train all sub-models on the same data.

        Args:
            df: Training DataFrame with feature columns and target.
            target_col: Name of the target column.
            val_df: Optional validation set (used by LightGBM for early
                stopping, and for IC-based weight calibration).
        """
        X_train = df[self.feature_cols].values
        y_train = df[target_col].values

        # Clean NaN rows
        mask = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
        X_clean, y_clean = X_train[mask], y_train[mask]

        logger.info(
            f"Training ensemble on {len(y_clean)} samples, {len(self.feature_cols)} features"
        )

        # 1. LightGBM (uses CrossSectionalModel interface — pass DataFrames)
        logger.info("  Training LightGBM...")

        # H-ICVAL fix: split val_df into two disjoint halves — one for
        # LightGBM early-stopping, one for IC-based weight calibration.
        # Using the same data for both causes the ensemble to overfit
        # to the early-stopping holdout.
        early_stop_df: Optional[pd.DataFrame] = None
        ic_cal_df: Optional[pd.DataFrame] = None
        if val_df is not None and len(val_df) > 20:
            # Date-based split (not iloc) — iloc is wrong for panel data
            # where multiple tickers share the same date.
            val_dates = val_df.index.get_level_values(0).unique().sort_values()
            mid_date = val_dates[len(val_dates) // 2]
            early_stop_df = val_df.loc[val_df.index.get_level_values(0) < mid_date]
            ic_cal_df = val_df.loc[val_df.index.get_level_values(0) >= mid_date]
            logger.info(
                f"  H-ICVAL: split val into early-stop ({len(early_stop_df)}) "
                f"and IC-cal ({len(ic_cal_df)})"
            )
        elif val_df is not None:
            # Too small to split — use for early stopping only, skip IC cal
            early_stop_df = val_df

        self.lgbm.fit(df, target_col=target_col, val_df=early_stop_df)

        # 2. Random Forest (sklearn interface — needs arrays)
        logger.info("  Training Random Forest...")
        self.rf.fit(X_clean, y_clean)

        self._fitted = True
        logger.info("  Ensemble training complete")

        # If IC-calibration holdout available, calibrate weights
        if ic_cal_df is not None and len(ic_cal_df) > 10:
            self.calibrate_weights(ic_cal_df, target_col=target_col)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Weighted average prediction from all sub-models.

        Args:
            df: DataFrame with feature columns.

        Returns:
            1-D array of ensemble predictions.

        Raises:
            ValueError: If the ensemble has not been fitted.
        """
        if not self._fitted:
            raise ValueError("Ensemble is not trained. Call fit() first.")

        X = df[self.feature_cols].values

        preds = {
            "lightgbm": self.lgbm.predict(df),
            "random_forest": self.rf.predict(X),
        }

        ensemble_pred = sum(preds[name] * weight for name, weight in self.weights.items())
        return np.asarray(ensemble_pred, dtype=np.float64)

    def predict_individual(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return predictions from each sub-model separately.

        Useful for diagnostics and weight calibration.
        """
        if not self._fitted:
            raise ValueError("Ensemble is not trained. Call fit() first.")

        X = df[self.feature_cols].values
        return {
            "lightgbm": self.lgbm.predict(df),
            "random_forest": self.rf.predict(X),
        }

    def calibrate_weights(
        self,
        val_df: pd.DataFrame,
        target_col: str = "target_5d",
    ) -> dict[str, float]:
        """Dynamically weight models by validation Information Coefficient.

        Models with negative IC are zeroed out. Weights are normalized
        so they sum to 1.0.

        Args:
            val_df: Validation DataFrame with features and target.
            target_col: Target column name.

        Returns:
            Updated weights dict.
        """
        X_val = val_df[self.feature_cols].values
        y_val = val_df[target_col].values

        # Clean NaN rows
        mask = ~np.isnan(y_val) & ~np.any(np.isnan(X_val), axis=1)
        if mask.sum() < 10:
            logger.warning("Not enough valid validation samples for IC calibration")
            return self.weights

        val_clean = val_df.iloc[mask.nonzero()[0]]
        y_clean = y_val[mask]

        ics: dict[str, float] = {}

        # C-CORR fix: use _safe_ic instead of raw np.corrcoef to handle
        # zero-variance predictions.
        lgbm_pred = self.lgbm.predict(val_clean)
        ics["lightgbm"] = _safe_ic(lgbm_pred, y_clean)

        X_clean = X_val[mask]
        ics["random_forest"] = _safe_ic(self.rf.predict(X_clean), y_clean)

        logger.info(f"  Individual model ICs: {ics}")

        # C-CORR fix: _safe_ic already returns 0.0 for negative/NaN ICs,
        # so max(0, ic) is redundant but kept for clarity.
        positive_ics = {name: max(0.0, ic) for name, ic in ics.items()}
        total_ic = sum(positive_ics.values())

        if total_ic > 0:
            self.weights = {name: ic / total_ic for name, ic in positive_ics.items()}
        else:
            logger.warning("All models have non-positive IC — keeping default weights")

        logger.info(f"  Calibrated ensemble weights: {self.weights}")
        return self.weights

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from each sub-model.

        Returns a DataFrame with columns per model, index = feature names.
        """
        result = pd.DataFrame(index=self.feature_cols)

        # LightGBM
        if self.lgbm.model is not None:
            result["lightgbm"] = self.lgbm.feature_importance().reindex(self.feature_cols).values

        # Random Forest (R3-P-11 fix: guard against unfitted model)
        if hasattr(self.rf, "feature_importances_"):
            result["random_forest"] = self.rf.feature_importances_

        # Weighted average importance
        result["ensemble"] = sum(
            result[name] / result[name].sum() * self.weights[name]
            for name in self.weights
            if name in result.columns and result[name].sum() > 0
        )

        return result.sort_values("ensemble", ascending=False)

    def save(self, path: str | Path) -> None:
        """Serialize ensemble to disk via joblib.

        Saves all sub-models, weights, feature_cols, and fitted state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_cols": self.feature_cols,
            "weights": self.weights,
            "fitted": self._fitted,
            "lgbm_model": self.lgbm.model,
            "rf_model": self.rf,
        }
        joblib.dump(state, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ModelEnsemble":
        """Load a serialized ensemble from disk.

        Returns a fitted ModelEnsemble ready for predict().
        """
        state = joblib.load(path)
        ensemble = cls(feature_cols=state["feature_cols"], weights=state["weights"])
        ensemble.lgbm.model = state["lgbm_model"]
        ensemble.rf = state["rf_model"]
        ensemble._fitted = state["fitted"]
        logger.info(f"Ensemble loaded from {path}")
        return ensemble
