"""Model ensemble with stacking meta-learner for cross-sectional return predictions.

Combines LightGBM, CatBoost, and Random Forest with a Ridge stacking
meta-learner for robust cross-sectional predictions.

Each base model captures different aspects of the signal:
  - LightGBM: Non-linear interactions (gradient boosting, leaf-wise)
  - CatBoost: Ordered boosting with native overfitting resistance
  - Random Forest: Robust to outliers (bagging)

The stacking meta-learner (Ridge regression) learns optimal weights
from out-of-sample base model predictions, capturing complementary
signal that simple averaging misses. CFA Institute 2025 research shows
stacking XGB+LGBM+CatBoost yields R²=0.977 vs individual models.

CRITICAL: The meta-learner must be trained on OUT-OF-SAMPLE predictions
only. Using in-sample predictions causes catastrophic overfitting.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from python.alpha.model import DEFAULT_SEED, CrossSectionalModel

logger = logging.getLogger(__name__)


def _safe_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute rank Information Coefficient with NaN/zero-variance guard.

    Uses Spearman rank correlation (standard for cross-sectional equity models).
    Returns 0.0 for degenerate inputs (NaN, zero variance).

    P1-17 fix: negative IC is preserved instead of being clamped to 0.
    A negative IC means the model has predictive power but with inverted
    sign — the IC-weighting logic in the ensemble uses abs(IC) for weight
    magnitude and the sign to decide whether to flip predictions.  Clamping
    negative IC to 0 discarded useful signal and gave equal weight to
    genuinely uninformative models (IC ≈ 0) and inversely predictive ones.
    """
    from scipy.stats import spearmanr

    if len(pred) < 2 or len(actual) < 2:
        return 0.0
    if np.std(pred) < 1e-10 or np.std(actual) < 1e-10:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    ic = float(ic)
    return 0.0 if np.isnan(ic) else ic


# Default ensemble weights (prior to IC-based calibration / stacking)
DEFAULT_WEIGHTS = {
    "lightgbm": 0.45,
    "catboost": 0.30,
    "random_forest": 0.25,
}


class ModelEnsemble:
    """Ensemble of LightGBM + CatBoost + Random Forest with stacking meta-learner.

    Two prediction modes:
      1. **Stacking** (default when meta-learner is trained): Ridge regression
         over OOS base model predictions. Captures complementary signal.
      2. **IC-weighted averaging** (fallback): Weights calibrated on validation IC.

    Compatible with CrossSectionalModel interface (fit/predict on DataFrames).
    """

    def __init__(
        self,
        feature_cols: list[str],
        weights: Optional[dict[str, float]] = None,
        use_stacking: bool = True,
    ):
        """Initialize ensemble sub-models.

        Args:
            feature_cols: Feature column names used by all sub-models.
            weights: Optional dict of model_name -> weight for averaging fallback.
            use_stacking: If True, train a Ridge meta-learner on OOS predictions.
        """
        self.feature_cols = feature_cols
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.use_stacking = use_stacking
        self._fitted = False
        self.validation_ic: float = 0.0  # Compatibility with predict.py quality gate

        # Base models
        self.lgbm = CrossSectionalModel(
            model_type="lightgbm",
            feature_cols=feature_cols,
        )
        # P1-16 fix: use Huber loss instead of RMSE for consistency with
        # the LightGBM sub-model (which uses Huber).  RMSE is sensitive to
        # return outliers (earnings gaps, halts) — Huber's bounded gradient
        # prevents single observations from dominating the ensemble.
        self.catboost = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=DEFAULT_SEED,
            verbose=0,
            loss_function="Huber:delta=1.0",
        )
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=100,
            n_jobs=-1,
            random_state=DEFAULT_SEED,
        )

        # Stacking meta-learner
        self.meta_learner: Optional[Ridge] = None

    @property
    def model(self):
        """Compatibility shim: return the LightGBM model for code that accesses model.model."""
        return self.lgbm.model

    @property
    def params(self) -> dict:
        """Compatibility shim: return ensemble config as params dict."""
        return {
            "ensemble_type": "stacking" if self.meta_learner is not None else "weighted_avg",
            "base_models": list(self.weights.keys()),
            "weights": self.weights,
            "n_features": len(self.feature_cols),
        }

    @property
    def base_models(self) -> dict[str, object]:
        """Return base models as a dict for iteration."""
        return {
            "lightgbm": self.lgbm,
            "catboost": self.catboost,
            "random_forest": self.rf,
        }

    def _clean_arrays(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature/target arrays and a valid-row mask."""
        X = df[self.feature_cols].values
        y = df[target_col].values
        mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        return X, y, mask

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "target_5d",
        val_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train all base models and optionally a stacking meta-learner.

        When val_df is provided, it's split into three disjoint date-based
        segments: early-stopping, OOS prediction generation, and IC calibration.
        """
        X_train, y_train, mask = self._clean_arrays(df, target_col)
        X_clean, y_clean = X_train[mask], y_train[mask]

        logger.info(
            f"Training ensemble on {len(y_clean)} samples, {len(self.feature_cols)} features"
        )

        # Split val_df into disjoint segments
        early_stop_df: Optional[pd.DataFrame] = None
        oos_df: Optional[pd.DataFrame] = None
        ic_cal_df: Optional[pd.DataFrame] = None

        if val_df is not None and len(val_df) > 30:
            val_dates = val_df.index.get_level_values(0).unique().sort_values()
            n_val_dates = len(val_dates)
            # 3-way split: 33% early-stop, 33% OOS for meta-learner, 33% IC-cal
            d1 = val_dates[n_val_dates // 3]
            d2 = val_dates[2 * n_val_dates // 3]
            early_stop_df = val_df.loc[val_df.index.get_level_values(0) < d1]
            oos_df = val_df.loc[
                (val_df.index.get_level_values(0) >= d1) & (val_df.index.get_level_values(0) < d2)
            ]
            ic_cal_df = val_df.loc[val_df.index.get_level_values(0) >= d2]
            logger.info(
                f"  Val split: early-stop={len(early_stop_df)}, "
                f"OOS={len(oos_df)}, IC-cal={len(ic_cal_df)}"
            )
        elif val_df is not None:
            early_stop_df = val_df

        # 1. LightGBM
        logger.info("  Training LightGBM...")
        self.lgbm.fit(df, target_col=target_col, val_df=early_stop_df)

        # 2. CatBoost
        logger.info("  Training CatBoost...")
        if early_stop_df is not None and len(early_stop_df) > 10:
            X_es, y_es, es_mask = self._clean_arrays(early_stop_df, target_col)
            self.catboost.fit(
                X_clean,
                y_clean,
                eval_set=(X_es[es_mask], y_es[es_mask]),
                early_stopping_rounds=20,
            )
        else:
            self.catboost.fit(X_clean, y_clean)

        # 3. Random Forest
        logger.info("  Training Random Forest...")
        self.rf.fit(X_clean, y_clean)

        self._fitted = True

        # 4. Stacking meta-learner (on OOS predictions only)
        #    Evaluate IC on ic_cal_df (held-out) to avoid in-sample evaluation.
        if self.use_stacking and oos_df is not None and len(oos_df) > 10:
            self._fit_meta_learner(oos_df, target_col, ic_eval_df=ic_cal_df)

        # 5. IC-based weight calibration (fallback or for diagnostics)
        if ic_cal_df is not None and len(ic_cal_df) > 10:
            self.calibrate_weights(ic_cal_df, target_col=target_col)

        logger.info("  Ensemble training complete")

    def _fit_meta_learner(
        self,
        oos_df: pd.DataFrame,
        target_col: str,
        ic_eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Train Ridge meta-learner on out-of-sample base model predictions.

        Args:
            oos_df: Out-of-sample data for training the meta-learner.
            target_col: Target column name.
            ic_eval_df: Separate held-out data for evaluating stacking IC.
                If None, IC is not computed (avoids in-sample evaluation).
        """
        X_oos, y_oos, mask = self._clean_arrays(oos_df, target_col)
        if mask.sum() < 10:
            logger.warning("Not enough OOS samples for meta-learner training")
            return

        oos_clean = oos_df.iloc[mask.nonzero()[0]]
        y_clean = y_oos[mask]
        X_clean = X_oos[mask]

        # Collect OOS predictions from each base model
        base_preds = np.column_stack(
            [
                self.lgbm.predict(oos_clean),
                self.catboost.predict(X_clean),
                self.rf.predict(X_clean),
            ]
        )

        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(base_preds, y_clean)

        # Evaluate stacking IC on held-out ic_eval_df (NOT training data)
        stacking_ic = 0.0
        if ic_eval_df is not None and len(ic_eval_df) > 10:
            X_eval, y_eval, eval_mask = self._clean_arrays(ic_eval_df, target_col)
            if eval_mask.sum() >= 10:
                eval_clean = ic_eval_df.iloc[eval_mask.nonzero()[0]]
                y_eval_clean = y_eval[eval_mask]
                X_eval_clean = X_eval[eval_mask]
                eval_base_preds = np.column_stack(
                    [
                        self.lgbm.predict(eval_clean),
                        self.catboost.predict(X_eval_clean),
                        self.rf.predict(X_eval_clean),
                    ]
                )
                eval_meta_pred = self.meta_learner.predict(eval_base_preds)
                stacking_ic = _safe_ic(eval_meta_pred, y_eval_clean)

        self.validation_ic = stacking_ic
        logger.info(
            f"  Meta-learner trained on {len(y_clean)} OOS samples. "
            f"Stacking IC (held-out)={stacking_ic:.4f}, "
            f"Ridge coefs={dict(zip(['lgbm', 'catboost', 'rf'], self.meta_learner.coef_.round(3)))}"
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction — stacking if meta-learner available, else weighted average.

        Args:
            df: DataFrame with feature columns.

        Returns:
            1-D array of ensemble predictions.
        """
        if not self._fitted:
            raise ValueError("Ensemble is not trained. Call fit() first.")

        X = df[self.feature_cols].values

        base_preds = {
            "lightgbm": self.lgbm.predict(df),
            "catboost": self.catboost.predict(X),
            "random_forest": self.rf.predict(X),
        }

        if self.meta_learner is not None:
            # Stacking: Ridge meta-learner over base predictions
            stacked = np.column_stack(
                [
                    base_preds["lightgbm"],
                    base_preds["catboost"],
                    base_preds["random_forest"],
                ]
            )
            return self.meta_learner.predict(stacked)
        else:
            # Fallback: IC-weighted average
            ensemble_pred = sum(base_preds[name] * weight for name, weight in self.weights.items())
            return np.asarray(ensemble_pred, dtype=np.float64)

    def predict_individual(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return predictions from each base model separately."""
        if not self._fitted:
            raise ValueError("Ensemble is not trained. Call fit() first.")

        X = df[self.feature_cols].values
        return {
            "lightgbm": self.lgbm.predict(df),
            "catboost": self.catboost.predict(X),
            "random_forest": self.rf.predict(X),
        }

    def calibrate_weights(
        self,
        val_df: pd.DataFrame,
        target_col: str = "target_5d",
    ) -> dict[str, float]:
        """Dynamically weight base models by validation IC.

        Models with negative IC are zeroed out. Weights sum to 1.0.
        """
        X_val, y_val, mask = self._clean_arrays(val_df, target_col)
        if mask.sum() < 10:
            logger.warning("Not enough valid validation samples for IC calibration")
            return self.weights

        val_clean = val_df.iloc[mask.nonzero()[0]]
        y_clean = y_val[mask]
        X_clean = X_val[mask]

        ics = {
            "lightgbm": _safe_ic(self.lgbm.predict(val_clean), y_clean),
            "catboost": _safe_ic(self.catboost.predict(X_clean), y_clean),
            "random_forest": _safe_ic(self.rf.predict(X_clean), y_clean),
        }

        logger.info(f"  Individual model ICs: {ics}")

        positive_ics = {name: max(0.0, ic) for name, ic in ics.items()}
        total_ic = sum(positive_ics.values())

        if total_ic > 0:
            self.weights = {name: ic / total_ic for name, ic in positive_ics.items()}
        else:
            logger.warning("All models have non-positive IC — keeping default weights")

        # Only set validation_ic if meta-learner hasn't already set it.
        # Stacking IC (from _fit_meta_learner) is a more accurate measure
        # of ensemble quality than the best individual base model IC.
        if self.meta_learner is None:
            self.validation_ic = max(ics.values()) if ics else 0.0
        logger.info(f"  Calibrated ensemble weights: {self.weights}")
        return self.weights

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from each base model.

        Returns a DataFrame with columns per model, index = feature names.
        """
        result = pd.DataFrame(index=self.feature_cols)

        # LightGBM
        if self.lgbm.model is not None:
            result["lightgbm"] = self.lgbm.feature_importance().reindex(self.feature_cols).values

        # CatBoost
        if hasattr(self.catboost, "feature_importances_"):
            result["catboost"] = self.catboost.feature_importances_

        # Random Forest
        if hasattr(self.rf, "feature_importances_"):
            result["random_forest"] = self.rf.feature_importances_

        # Weighted average importance
        result["ensemble"] = sum(
            result[name] / result[name].sum() * self.weights.get(name, 0)
            for name in ["lightgbm", "catboost", "random_forest"]
            if name in result.columns and result[name].sum() > 0
        )

        return result.sort_values("ensemble", ascending=False)

    def save(self, path: str | Path) -> None:
        """Serialize ensemble to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_cols": self.feature_cols,
            "weights": self.weights,
            "fitted": self._fitted,
            "lgbm_model": self.lgbm.model,
            "catboost_model": self.catboost,
            "rf_model": self.rf,
            "meta_learner": self.meta_learner,
            "validation_ic": self.validation_ic,
        }
        joblib.dump(state, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ModelEnsemble":
        """Load a serialized ensemble from disk."""
        state = joblib.load(path)
        ensemble = cls(feature_cols=state["feature_cols"], weights=state["weights"])
        ensemble.lgbm.model = state["lgbm_model"]
        ensemble.catboost = state["catboost_model"]
        ensemble.rf = state["rf_model"]
        ensemble.meta_learner = state.get("meta_learner")
        ensemble.validation_ic = state.get("validation_ic", 0.0)
        ensemble._fitted = state["fitted"]
        logger.info(f"Ensemble loaded from {path}")
        return ensemble
