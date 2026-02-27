"""Conformal prediction for uncertainty-aware position sizing.

Uses MAPIE (Model Agnostic Prediction Interval Estimator) to produce
distribution-free prediction intervals around cross-sectional return
forecasts. Position sizes are scaled inversely to interval width:
narrow intervals (high confidence) get larger positions.

Applications:
  1. Position sizing: scale inversely to prediction interval width
  2. Trade filtering: skip trades with very wide intervals
  3. Coverage validation: ensure intervals achieve target coverage

References:
  - Vovk, Gammerman & Shafer, 2005 — "Algorithmic Learning in a Random World"
  - MAPIE: scikit-learn-contrib/MAPIE
  - ICML 2025 — "Conformal Prediction for Stock Selection"
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConformalPositionSizer:
    """Conformal prediction intervals for position sizing.

    Wraps a base estimator with MAPIE's SplitConformalRegressor to produce
    prediction intervals, then sizes positions inversely to interval width.

    Two-step workflow:
      1. fit(X_train, y_train) — trains the base estimator
      2. conformalize(X_cal, y_cal) — calibrates prediction intervals on
         a held-out calibration set (MUST be separate from training data)

    Typical usage::

        sizer = ConformalPositionSizer(base_model)
        sizer.fit(X_train, y_train)
        sizer.conformalize(X_cal, y_cal)
        sizes = sizer.position_sizes(X_test, base_size=0.05)

    Args:
        base_estimator: A scikit-learn compatible regressor.
        confidence_levels: Confidence levels for prediction intervals.
            Default [0.90] produces 90% intervals.
    """

    def __init__(
        self,
        base_estimator,
        confidence_levels: Optional[list[float]] = None,
    ):
        self.base_estimator = base_estimator
        self.confidence_levels = confidence_levels or [0.90]
        self._mapie = None
        self._fitted = False
        self._conformalized = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalPositionSizer":
        """Train the base estimator.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target array (n_samples,).

        Returns:
            self (for chaining).
        """
        self.base_estimator.fit(X, y)
        self._fitted = True
        logger.info("ConformalPositionSizer: base estimator fitted")
        return self

    def conformalize(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> "ConformalPositionSizer":
        """Calibrate prediction intervals on a held-out calibration set.

        CRITICAL: X_cal/y_cal MUST be separate from training data.
        Using training data will produce intervals that are too narrow.

        Args:
            X_cal: Calibration feature matrix.
            y_cal: Calibration target array.

        Returns:
            self (for chaining).
        """
        if not self._fitted:
            raise ValueError("Base estimator not fitted. Call fit() first.")

        from mapie.regression import SplitConformalRegressor

        self._mapie = SplitConformalRegressor(
            estimator=self.base_estimator,
            confidence_level=self.confidence_levels,
            prefit=True,
        )
        self._mapie.conformalize(X_cal, y_cal)
        self._conformalized = True
        logger.info(
            f"Conformal intervals calibrated on {len(y_cal)} samples "
            f"(confidence={self.confidence_levels})"
        )
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with conformal intervals.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Tuple of (point_predictions, intervals) where intervals has
            shape (n_samples, 2, n_levels) with [lower, upper] bounds.
        """
        if not self._conformalized:
            raise ValueError("Not conformalized. Call conformalize() first.")

        y_pred, y_pis = self._mapie.predict_interval(X)
        return y_pred, y_pis

    def interval_widths(self, X: np.ndarray, level_idx: int = 0) -> np.ndarray:
        """Compute prediction interval widths.

        Args:
            X: Feature matrix.
            level_idx: Which confidence level to use (index into confidence_levels).

        Returns:
            1-D array of interval widths.
        """
        _, y_pis = self.predict(X)
        lower = y_pis[:, 0, level_idx]
        upper = y_pis[:, 1, level_idx]
        return upper - lower

    def position_sizes(
        self,
        X: np.ndarray,
        base_size: float = 1.0,
        min_size: float = 0.0,
        max_size: float = 1.0,
        level_idx: int = 0,
    ) -> np.ndarray:
        """Compute uncertainty-adjusted position sizes.

        Sizes are scaled inversely to interval width:
            size = base_size / (1 + width / median_width)

        This ensures:
          - Narrow intervals (high confidence) -> larger positions
          - Wide intervals (low confidence) -> smaller positions
          - Median-width interval -> approximately base_size / 2

        Args:
            X: Feature matrix.
            base_size: Base position size (before scaling).
            min_size: Floor for position sizes.
            max_size: Cap for position sizes.
            level_idx: Which confidence level to use.

        Returns:
            1-D array of position sizes, clipped to [min_size, max_size].
        """
        widths = self.interval_widths(X, level_idx=level_idx)

        # Avoid division by zero
        median_width = np.median(widths)
        if median_width <= 0:
            return np.full(len(widths), base_size)

        sizes = base_size / (1 + widths / median_width)
        return np.clip(sizes, min_size, max_size)

    def validate_coverage(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """Validate empirical coverage of prediction intervals.

        CRITICAL: Financial returns violate exchangeability (ARCH effects),
        so conformal guarantees may not hold. Always validate empirically.

        Args:
            X: Feature matrix.
            y_true: True target values.

        Returns:
            Dict with target_coverage, empirical_coverage per confidence level,
            and coverage_gap.
        """
        if not self._conformalized:
            raise ValueError("Not conformalized. Call conformalize() first.")

        _, y_pis = self.predict(X)
        results = {}

        for i, level in enumerate(self.confidence_levels):
            lower = y_pis[:, 0, i]
            upper = y_pis[:, 1, i]
            covered = ((y_true >= lower) & (y_true <= upper)).mean()
            gap = covered - level

            results[f"level_{level}"] = {
                "target_coverage": float(level),
                "empirical_coverage": float(covered),
                "coverage_gap": float(gap),
                "is_valid": abs(gap) < 0.05,  # Within 5pp of target
            }
            logger.info(
                f"Coverage @ {level:.0%}: actual={covered:.1%}, gap={gap:+.1%}"
            )

        return results

    def to_json(self) -> dict:
        """Export state as JSON-serializable dict."""
        return {
            "fitted": self._fitted,
            "conformalized": self._conformalized,
            "confidence_levels": self.confidence_levels,
        }
