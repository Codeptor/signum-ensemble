"""Online learning and concept drift adaptation for batch model correction.

Wraps the batch-trained ensemble with an adaptive layer that monitors
prediction quality, detects distributional drift, and applies real-time
corrections without full retraining.

Components:
  - ``ConceptDriftMonitor``: Multi-signal drift detection (ADWIN-style
    change detection on residuals + PSI on features + CUSUM on IC).
  - ``AdaptiveFeatureWeighter``: EWM-tracked feature IC for dynamic
    feature importance adjustment.
  - ``ModelConfidenceTracker``: Exponential decay of model trust since
    last training, with accuracy feedback.
  - ``OnlineResidualCorrector``: SGD Ridge regression that learns the
    batch model's errors and applies bounded corrections.
  - ``OnlineLearningModule``: Orchestrator combining all four components.

Usage::

    module = OnlineLearningModule(feature_cols=["mom_20", "vol_60", ...])
    module.on_model_trained(datetime.now())

    # At prediction time:
    corrected = module.correct_predictions(raw_preds, features_dict)

    # When realized returns arrive:
    drift_report = module.update_with_realized(realized, features_dict)

References:
  - Bifet & Gavalda (2007), "Learning from Time-Changing Data with
    Adaptive Windowing" (ADWIN)
  - Wen & Keyes (2019), "Population Stability Index"
"""

import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnlineLearningConfig:
    """Hyperparameters for online learning module.

    Uses half-life parameterization — interpretable and auditable.
    """

    # Drift detection
    residual_window: int = 60
    psi_threshold: float = 0.15
    psi_window: int = 60
    cusum_threshold: float = 3.0
    drift_cooldown_days: int = 5

    # Adaptive feature weighting
    feature_hl_days: int = 21
    min_feature_weight: float = 0.1
    max_feature_weight: float = 3.0

    # Model confidence decay
    confidence_hl_days: int = 10
    min_confidence: float = 0.3

    # Residual correction
    correction_lr: float = 0.005
    correction_l2: float = 1.0
    correction_max_weight: float = 0.3
    min_samples_for_correction: int = 30

    def decay_factor(self, half_life_days: int) -> float:
        return 0.5 ** (1.0 / max(half_life_days, 1))


# ---------------------------------------------------------------------------
# Drift severity
# ---------------------------------------------------------------------------


class DriftSeverity(enum.IntEnum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


@dataclass
class DriftSignal:
    """A single drift detection event."""

    timestamp: datetime
    source: str
    feature: str
    severity: float
    details: dict = field(default_factory=dict)


@dataclass
class DriftReport:
    """Aggregated drift detection report."""

    severity: DriftSeverity
    signals: list[DriftSignal]
    feature_psi: dict[str, float]
    residual_mean_shift: float
    ic_cusum: float
    exposure_multiplier: float

    @property
    def recommended_action(self) -> str:
        actions = {
            DriftSeverity.NONE: "No action needed",
            DriftSeverity.MILD: "Monitor closely",
            DriftSeverity.MODERATE: "Reduce position sizes, consider retraining",
            DriftSeverity.SEVERE: "Halt new positions, retrain model",
        }
        return actions[self.severity]


# ---------------------------------------------------------------------------
# Concept Drift Monitor
# ---------------------------------------------------------------------------


class ConceptDriftMonitor:
    """Multi-signal concept drift detection.

    Runs three complementary detectors:
      1. Residual mean shift: detects change in prediction error distribution
      2. Rolling PSI per feature: detects distributional shift in inputs
      3. CUSUM on IC: detects slow decay in predictive power

    Parameters
    ----------
    config : OnlineLearningConfig
    feature_cols : list[str]
    """

    def __init__(self, config: OnlineLearningConfig, feature_cols: list[str]):
        self.config = config
        self.feature_cols = feature_cols

        # Residual tracking (ADWIN-style: compare two windows)
        self._residuals: deque[float] = deque(maxlen=config.residual_window * 2)

        # PSI: reference and current distributions per feature
        self._ref_buffers: dict[str, deque[float]] = {
            f: deque(maxlen=config.psi_window) for f in feature_cols
        }
        self._cur_buffers: dict[str, deque[float]] = {
            f: deque(maxlen=config.psi_window) for f in feature_cols
        }
        self._psi_ref_filled: bool = False

        # CUSUM on IC
        self._ic_history: deque[float] = deque(maxlen=252)
        self._cusum_pos: float = 0.0
        self._cusum_target_ic: float = 0.04  # expected IC

        self._signals: list[DriftSignal] = []
        self._last_drift_time: datetime | None = None

    def update(
        self,
        timestamp: datetime,
        features: dict[str, float],
        prediction: float,
        actual: float | None = None,
    ) -> DriftSeverity:
        """Process one observation. Returns current drift severity."""
        self._signals.clear()

        # Update feature PSI buffers
        for feat in self.feature_cols:
            val = features.get(feat)
            if val is None or np.isnan(val):
                continue
            if not self._psi_ref_filled:
                self._ref_buffers[feat].append(val)
            else:
                self._cur_buffers[feat].append(val)

        if not self._psi_ref_filled and all(
            len(b) >= self.config.psi_window for b in self._ref_buffers.values()
        ):
            self._psi_ref_filled = True

        # Compute PSI for each feature
        for feat in self.feature_cols:
            psi = self._compute_psi(feat)
            if psi > self.config.psi_threshold:
                self._signals.append(DriftSignal(
                    timestamp=timestamp,
                    source="psi",
                    feature=feat,
                    severity=min(psi / self.config.psi_threshold, 1.0),
                    details={"psi": psi},
                ))

        # Update residuals if actual is available
        if actual is not None:
            residual = actual - prediction
            self._residuals.append(residual)

            # Residual mean shift detection
            if len(self._residuals) >= self.config.residual_window:
                mid = len(self._residuals) // 2
                resids = list(self._residuals)
                old_mean = np.mean(resids[:mid])
                new_mean = np.mean(resids[mid:])
                old_std = max(np.std(resids[:mid]), 1e-8)
                z_score = abs(new_mean - old_mean) / old_std

                if z_score > 2.0:
                    self._signals.append(DriftSignal(
                        timestamp=timestamp,
                        source="residual_shift",
                        feature="prediction_error",
                        severity=min(z_score / 4.0, 1.0),
                        details={"z_score": z_score, "old_mean": old_mean, "new_mean": new_mean},
                    ))

            # CUSUM on IC (approximate: use -abs(residual) as inverse quality)
            ic_proxy = -abs(residual)
            self._ic_history.append(ic_proxy)
            if len(self._ic_history) >= 20:
                ic_std = max(np.std(list(self._ic_history)), 1e-8)
                slack = 0.5 * ic_std
                self._cusum_pos = max(0, self._cusum_pos + (-ic_proxy - self._cusum_target_ic) - slack)

                if self._cusum_pos > self.config.cusum_threshold * ic_std:
                    self._signals.append(DriftSignal(
                        timestamp=timestamp,
                        source="cusum",
                        feature="ic_degradation",
                        severity=min(self._cusum_pos / (self.config.cusum_threshold * ic_std * 2), 1.0),
                        details={"cusum": self._cusum_pos},
                    ))

        return self._compute_severity()

    def _compute_psi(self, feature: str) -> float:
        """Population Stability Index between reference and current windows."""
        ref = list(self._ref_buffers[feature])
        cur = list(self._cur_buffers[feature])
        if len(ref) < 10 or len(cur) < 10:
            return 0.0

        n_bins = 10
        combined = ref + cur
        bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts = np.histogram(ref, bins=bins)[0].astype(float)
        cur_counts = np.histogram(cur, bins=bins)[0].astype(float)

        # Add small constant to avoid log(0)
        ref_pct = (ref_counts + 0.5) / (len(ref) + n_bins * 0.5)
        cur_pct = (cur_counts + 0.5) / (len(cur) + n_bins * 0.5)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)

    def _compute_severity(self) -> DriftSeverity:
        n = len(self._signals)
        has_cusum = any(s.source == "cusum" for s in self._signals)
        has_residual = any(s.source == "residual_shift" for s in self._signals)
        n_psi = sum(1 for s in self._signals if s.source == "psi")

        if has_cusum and has_residual and n_psi >= 2:
            return DriftSeverity.SEVERE
        if has_cusum or (has_residual and n_psi >= 1):
            return DriftSeverity.MODERATE
        if n >= 1:
            return DriftSeverity.MILD
        return DriftSeverity.NONE

    def get_report(self) -> DriftReport:
        severity = self._compute_severity()
        feature_psi = {f: self._compute_psi(f) for f in self.feature_cols}
        resids = list(self._residuals)
        shift = 0.0
        if len(resids) >= 20:
            mid = len(resids) // 2
            shift = float(np.mean(resids[mid:]) - np.mean(resids[:mid]))

        exposure_map = {
            DriftSeverity.NONE: 1.0,
            DriftSeverity.MILD: 0.9,
            DriftSeverity.MODERATE: 0.7,
            DriftSeverity.SEVERE: 0.3,
        }

        return DriftReport(
            severity=severity,
            signals=list(self._signals),
            feature_psi=feature_psi,
            residual_mean_shift=shift,
            ic_cusum=self._cusum_pos,
            exposure_multiplier=exposure_map[severity],
        )

    def reset(self) -> None:
        self._residuals.clear()
        self._cusum_pos = 0.0
        self._ic_history.clear()
        self._signals.clear()
        self._psi_ref_filled = False
        for f in self.feature_cols:
            self._ref_buffers[f].clear()
            self._cur_buffers[f].clear()


# ---------------------------------------------------------------------------
# Adaptive Feature Weighter
# ---------------------------------------------------------------------------


class AdaptiveFeatureWeighter:
    """Track per-feature predictive value with exponential decay.

    Maintains rolling EWM of Spearman IC per feature. Features with
    declining IC get down-weighted; strengthening IC get up-weighted.

    Parameters
    ----------
    feature_cols : list[str]
    config : OnlineLearningConfig
    """

    def __init__(self, feature_cols: list[str], config: OnlineLearningConfig):
        self.feature_cols = feature_cols
        self.config = config
        self._alpha = 1.0 - config.decay_factor(config.feature_hl_days)
        self._ewm_ic: dict[str, float] = {f: 0.0 for f in feature_cols}
        self._feature_buffer: deque[dict[str, float]] = deque(maxlen=252)
        self._return_buffer: deque[float] = deque(maxlen=252)
        self._n_updates: int = 0

    def update(self, features: dict[str, float], actual_return: float) -> None:
        self._feature_buffer.append(features)
        self._return_buffer.append(actual_return)
        self._n_updates += 1

        if self._n_updates >= 30 and self._n_updates % 5 == 0:
            batch_ic = self._compute_batch_ic()
            for feat, ic in batch_ic.items():
                old = self._ewm_ic.get(feat, 0.0)
                self._ewm_ic[feat] = self._alpha * ic + (1 - self._alpha) * old

    def _compute_batch_ic(self) -> dict[str, float]:
        returns = np.array(self._return_buffer)
        result = {}
        for feat in self.feature_cols:
            values = np.array([fb.get(feat, np.nan) for fb in self._feature_buffer])
            mask = ~(np.isnan(values) | np.isnan(returns))
            if mask.sum() < 10:
                result[feat] = 0.0
                continue
            rho, _ = spearmanr(values[mask], returns[mask])
            result[feat] = float(rho) if not np.isnan(rho) else 0.0
        return result

    def get_weights(self) -> dict[str, float]:
        """Return adaptive feature weights (mean=1.0, clipped to [min, max])."""
        if self._n_updates < 30:
            return {f: 1.0 for f in self.feature_cols}

        ics = np.array([abs(self._ewm_ic.get(f, 0.0)) for f in self.feature_cols])
        median_ic = max(np.median(ics), 1e-8)

        weights = {}
        for f in self.feature_cols:
            raw = abs(self._ewm_ic.get(f, 0.0)) / median_ic
            weights[f] = float(np.clip(raw, self.config.min_feature_weight, self.config.max_feature_weight))
        return weights


# ---------------------------------------------------------------------------
# Model Confidence Tracker
# ---------------------------------------------------------------------------


class ModelConfidenceTracker:
    """Exponential decay of model trust since last training.

    Confidence decays with time and adjusts based on accuracy feedback.

    Parameters
    ----------
    config : OnlineLearningConfig
    """

    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self._last_train_time: datetime | None = None
        self._decay = config.decay_factor(config.confidence_hl_days)
        self._accuracy_ewm: float = 1.0
        self._accuracy_alpha: float = 1.0 - config.decay_factor(config.confidence_hl_days)
        self._drift_adj: float = 1.0
        self._n_updates: int = 0

    def on_model_retrained(self, timestamp: datetime) -> None:
        self._last_train_time = timestamp
        self._accuracy_ewm = 1.0
        self._drift_adj = 1.0

    def update(
        self,
        timestamp: datetime,
        prediction: float | None = None,
        actual: float | None = None,
        drift_severity: DriftSeverity = DriftSeverity.NONE,
    ) -> float:
        self._n_updates += 1

        # Accuracy feedback
        if prediction is not None and actual is not None:
            error = abs(prediction - actual)
            accuracy_signal = max(0.0, 1.0 - error * 50)  # Scale: 0.02 error → 0
            self._accuracy_ewm = (
                self._accuracy_alpha * accuracy_signal
                + (1 - self._accuracy_alpha) * self._accuracy_ewm
            )

        # Drift adjustment
        drift_penalties = {
            DriftSeverity.NONE: 1.0,
            DriftSeverity.MILD: 0.95,
            DriftSeverity.MODERATE: 0.8,
            DriftSeverity.SEVERE: 0.5,
        }
        self._drift_adj = drift_penalties.get(drift_severity, 1.0)

        return self.confidence

    @property
    def confidence(self) -> float:
        time_decay = self._decay ** self.days_since_training
        raw = time_decay * self._accuracy_ewm * self._drift_adj
        return max(raw, self.config.min_confidence)

    @property
    def should_retrain(self) -> bool:
        return self.confidence < self.config.min_confidence * 1.5

    @property
    def days_since_training(self) -> float:
        if self._last_train_time is None:
            return 0.0
        delta = datetime.now() - self._last_train_time
        return max(delta.days, 0)


# ---------------------------------------------------------------------------
# Online Residual Corrector
# ---------------------------------------------------------------------------


class OnlineResidualCorrector:
    """SGD Ridge regression that learns the batch model's prediction errors.

    Applies bounded corrections: corrected = raw - clip(predicted_error).

    Parameters
    ----------
    feature_cols : list[str]
    config : OnlineLearningConfig
    """

    def __init__(self, feature_cols: list[str], config: OnlineLearningConfig):
        self.feature_cols = feature_cols
        self.config = config
        self._meta_features = ["days_since_train", "regime_id", "drift_severity"]
        self._all_features = feature_cols + self._meta_features
        self._n_features = len(self._all_features)

        self._weights = np.zeros(self._n_features)
        self._bias: float = 0.0
        self._feature_means = np.zeros(self._n_features)
        self._feature_vars = np.ones(self._n_features)
        self._n_updates: int = 0
        self._ready: bool = False

    def update(
        self,
        features: dict[str, float],
        meta_features: dict[str, float],
        prediction: float,
        actual: float,
    ) -> None:
        """Learn from one (prediction, actual) pair."""
        residual = actual - prediction
        x_raw = self._to_array(features, meta_features)
        self._update_running_stats(x_raw)
        x = self._standardize(x_raw)

        self._sgd_step(x, residual)
        self._n_updates += 1

        if self._n_updates >= self.config.min_samples_for_correction:
            self._ready = True

    def correct(
        self,
        raw_prediction: float,
        features: dict[str, float],
        meta_features: dict[str, float],
    ) -> float:
        """Apply correction to a batch model prediction."""
        if not self._ready:
            return raw_prediction

        x = self._standardize(self._to_array(features, meta_features))
        predicted_error = float(np.dot(self._weights, x) + self._bias)

        max_adj = abs(raw_prediction) * self.config.correction_max_weight
        correction = np.clip(predicted_error, -max_adj, max_adj)
        return raw_prediction - correction

    def _sgd_step(self, x: np.ndarray, target_residual: float) -> None:
        predicted = np.dot(self._weights, x) + self._bias
        error = predicted - target_residual
        lr = self.config.correction_lr

        self._weights -= lr * (error * x + self.config.correction_l2 * self._weights)
        self._bias -= lr * error

    def _update_running_stats(self, x_raw: np.ndarray) -> None:
        """Welford's online algorithm for running mean/variance."""
        n = self._n_updates + 1
        delta = x_raw - self._feature_means
        self._feature_means += delta / n
        delta2 = x_raw - self._feature_means
        self._feature_vars += (delta * delta2 - self._feature_vars) / n

    def _standardize(self, x_raw: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self._feature_vars, 1e-8))
        return (x_raw - self._feature_means) / std

    def _to_array(
        self,
        features: dict[str, float],
        meta_features: dict[str, float],
    ) -> np.ndarray:
        combined = {**features, **meta_features}
        return np.array([combined.get(f, 0.0) for f in self._all_features])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class OnlineLearningModule:
    """Top-level orchestrator for the online learning subsystem.

    Parameters
    ----------
    feature_cols : list[str]
        Feature column names used by the batch model.
    config : OnlineLearningConfig, optional
    """

    def __init__(
        self,
        feature_cols: list[str],
        config: OnlineLearningConfig | None = None,
    ):
        self.config = config or OnlineLearningConfig()
        self.feature_cols = feature_cols

        self.drift_monitor = ConceptDriftMonitor(self.config, feature_cols)
        self.feature_weighter = AdaptiveFeatureWeighter(feature_cols, self.config)
        self.confidence_tracker = ModelConfidenceTracker(self.config)
        self.residual_corrector = OnlineResidualCorrector(feature_cols, self.config)

        self._last_predictions: dict[str, float] = {}

    def on_model_trained(self, timestamp: datetime) -> None:
        """Notify all components that the batch model was retrained."""
        self.confidence_tracker.on_model_retrained(timestamp)
        self.drift_monitor.reset()
        logger.info(f"OnlineLearningModule: model retrained at {timestamp}")

    def correct_predictions(
        self,
        predictions: dict[str, float],
        features_per_ticker: dict[str, dict[str, float]],
        regime_id: int = 1,
    ) -> dict[str, float]:
        """Apply online corrections to batch model predictions."""
        meta = {
            "days_since_train": self.confidence_tracker.days_since_training,
            "regime_id": float(regime_id),
            "drift_severity": float(self.drift_monitor.get_report().severity),
        }

        confidence = self.confidence_tracker.confidence
        corrected = {}

        for ticker, raw_pred in predictions.items():
            features = features_per_ticker.get(ticker, {})
            adj_pred = self.residual_corrector.correct(raw_pred, features, meta)
            corrected[ticker] = adj_pred * confidence

        self._last_predictions = corrected
        return corrected

    def update_with_realized(
        self,
        realized_returns: dict[str, float],
        features_per_ticker: dict[str, dict[str, float]],
        regime_id: int = 1,
        timestamp: datetime | None = None,
    ) -> DriftReport:
        """Feed realized returns back into all online components."""
        timestamp = timestamp or datetime.now()
        meta = {
            "days_since_train": self.confidence_tracker.days_since_training,
            "regime_id": float(regime_id),
            "drift_severity": float(self.drift_monitor.get_report().severity),
        }

        for ticker, actual in realized_returns.items():
            prediction = self._last_predictions.get(ticker)
            features = features_per_ticker.get(ticker, {})
            if prediction is None:
                continue

            self.drift_monitor.update(
                timestamp=timestamp,
                features=features,
                prediction=prediction,
                actual=actual,
            )
            self.feature_weighter.update(features, actual)
            self.residual_corrector.update(features, meta, prediction, actual)

        drift_report = self.drift_monitor.get_report()
        self.confidence_tracker.update(
            timestamp=timestamp,
            drift_severity=drift_report.severity,
        )
        return drift_report

    def get_exposure_multiplier(self) -> float:
        """Combined exposure multiplier from confidence + drift."""
        confidence = self.confidence_tracker.confidence
        drift_mult = self.drift_monitor.get_report().exposure_multiplier
        return confidence * drift_mult

    def get_status(self) -> dict:
        """Full status for dashboard/monitoring."""
        return {
            "confidence": self.confidence_tracker.confidence,
            "days_since_training": self.confidence_tracker.days_since_training,
            "should_retrain": self.confidence_tracker.should_retrain,
            "drift_severity": self.drift_monitor.get_report().severity.name,
            "feature_weights": self.feature_weighter.get_weights(),
            "corrector_ready": self.residual_corrector._ready,
            "corrector_updates": self.residual_corrector._n_updates,
            "exposure_multiplier": self.get_exposure_multiplier(),
        }
