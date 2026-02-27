"""Tests for online learning and concept drift adaptation."""

import numpy as np
import pytest
from datetime import datetime, timedelta

from python.alpha.online_learning import (
    AdaptiveFeatureWeighter,
    ConceptDriftMonitor,
    DriftReport,
    DriftSeverity,
    ModelConfidenceTracker,
    OnlineLearningConfig,
    OnlineLearningModule,
    OnlineResidualCorrector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURES = ["mom_20", "vol_60", "size"]


def _make_config(**kwargs):
    return OnlineLearningConfig(**kwargs)


def _make_features(rng, n=1, feat_cols=FEATURES):
    """Generate random feature dicts."""
    if n == 1:
        return {f: float(rng.normal()) for f in feat_cols}
    return [{f: float(rng.normal()) for f in feat_cols} for _ in range(n)]


# ---------------------------------------------------------------------------
# OnlineLearningConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        c = OnlineLearningConfig()
        assert c.psi_threshold == 0.15
        assert c.min_confidence == 0.3

    def test_decay_factor(self):
        c = OnlineLearningConfig()
        d = c.decay_factor(10)
        assert 0 < d < 1
        # After 10 days, should be 0.5
        assert d**10 == pytest.approx(0.5, abs=0.01)

    def test_frozen(self):
        c = OnlineLearningConfig()
        with pytest.raises(AttributeError):
            c.psi_threshold = 0.5


# ---------------------------------------------------------------------------
# ConceptDriftMonitor
# ---------------------------------------------------------------------------


class TestDriftMonitor:
    def test_no_drift_stable_data(self):
        config = _make_config(residual_window=20, psi_window=15)
        monitor = ConceptDriftMonitor(config, FEATURES)
        rng = np.random.default_rng(42)
        ts = datetime(2024, 1, 1)

        for i in range(40):
            features = _make_features(rng)
            pred = float(rng.normal(0.01, 0.005))
            actual = pred + float(rng.normal(0, 0.001))
            sev = monitor.update(ts + timedelta(days=i), features, pred, actual)

        report = monitor.get_report()
        assert report.severity in (DriftSeverity.NONE, DriftSeverity.MILD)

    def test_drift_on_distribution_shift(self):
        config = _make_config(residual_window=20, psi_window=15, psi_threshold=0.1)
        monitor = ConceptDriftMonitor(config, FEATURES)
        rng = np.random.default_rng(42)
        ts = datetime(2024, 1, 1)

        # Fill reference window with normal data
        for i in range(20):
            features = {f: float(rng.normal(0, 1)) for f in FEATURES}
            monitor.update(ts + timedelta(days=i), features, 0.01, 0.01)

        # Shift feature distributions
        for i in range(20, 50):
            features = {f: float(rng.normal(5, 1)) for f in FEATURES}  # big shift
            monitor.update(ts + timedelta(days=i), features, 0.01, 0.01)

        report = monitor.get_report()
        # Should detect at least mild drift
        assert report.severity >= DriftSeverity.MILD

    def test_drift_on_residual_shift(self):
        config = _make_config(residual_window=20)
        monitor = ConceptDriftMonitor(config, FEATURES)
        rng = np.random.default_rng(42)
        ts = datetime(2024, 1, 1)

        # Stable residuals
        for i in range(20):
            monitor.update(ts + timedelta(days=i), _make_features(rng), 0.01, 0.011)

        # Big residual shift
        for i in range(20, 40):
            monitor.update(ts + timedelta(days=i), _make_features(rng), 0.01, 0.1)

        report = monitor.get_report()
        assert report.severity >= DriftSeverity.MILD

    def test_reset_clears_state(self):
        config = _make_config()
        monitor = ConceptDriftMonitor(config, FEATURES)
        rng = np.random.default_rng(42)

        for i in range(10):
            monitor.update(datetime(2024, 1, i + 1), _make_features(rng), 0.01, 0.1)

        monitor.reset()
        report = monitor.get_report()
        assert report.severity == DriftSeverity.NONE

    def test_psi_computation(self):
        config = _make_config(psi_window=20)
        monitor = ConceptDriftMonitor(config, ["feat"])
        rng = np.random.default_rng(42)
        ts = datetime(2024, 1, 1)

        # Fill reference
        for i in range(20):
            monitor.update(ts + timedelta(days=i), {"feat": float(rng.normal(0, 1))}, 0.0)

        # Same distribution → PSI should be low
        psi = monitor._compute_psi("feat")
        assert psi < 0.3  # Not shifted yet

    def test_report_exposure_multiplier(self):
        config = _make_config()
        monitor = ConceptDriftMonitor(config, FEATURES)
        report = monitor.get_report()
        assert report.exposure_multiplier == 1.0  # No drift


# ---------------------------------------------------------------------------
# AdaptiveFeatureWeighter
# ---------------------------------------------------------------------------


class TestFeatureWeighter:
    def test_default_weights_before_warmup(self):
        config = _make_config()
        weighter = AdaptiveFeatureWeighter(FEATURES, config)
        weights = weighter.get_weights()
        assert all(w == 1.0 for w in weights.values())

    def test_weights_update_after_observations(self):
        config = _make_config(feature_hl_days=5)
        weighter = AdaptiveFeatureWeighter(FEATURES, config)
        rng = np.random.default_rng(42)

        for i in range(50):
            features = _make_features(rng)
            ret = float(rng.normal(0, 0.01))
            weighter.update(features, ret)

        weights = weighter.get_weights()
        assert len(weights) == len(FEATURES)
        # At least some weights should differ from 1.0
        assert any(w != 1.0 for w in weights.values())

    def test_weights_clipped(self):
        config = _make_config(min_feature_weight=0.2, max_feature_weight=2.0)
        weighter = AdaptiveFeatureWeighter(FEATURES, config)
        rng = np.random.default_rng(42)

        for i in range(50):
            weighter.update(_make_features(rng), float(rng.normal()))

        weights = weighter.get_weights()
        for w in weights.values():
            assert w >= 0.2 - 0.01
            assert w <= 2.0 + 0.01


# ---------------------------------------------------------------------------
# ModelConfidenceTracker
# ---------------------------------------------------------------------------


class TestConfidenceTracker:
    def test_initial_confidence(self):
        tracker = ModelConfidenceTracker(_make_config())
        tracker.on_model_retrained(datetime.now())
        assert tracker.confidence >= 0.9

    def test_confidence_decays(self):
        config = _make_config(confidence_hl_days=5)
        tracker = ModelConfidenceTracker(config)
        tracker.on_model_retrained(datetime.now() - timedelta(days=20))
        assert tracker.confidence < 1.0

    def test_confidence_has_floor(self):
        config = _make_config(min_confidence=0.3, confidence_hl_days=2)
        tracker = ModelConfidenceTracker(config)
        tracker.on_model_retrained(datetime.now() - timedelta(days=100))
        assert tracker.confidence >= 0.3

    def test_drift_reduces_confidence(self):
        config = _make_config()
        tracker = ModelConfidenceTracker(config)
        tracker.on_model_retrained(datetime.now())

        c_before = tracker.confidence
        tracker.update(datetime.now(), drift_severity=DriftSeverity.SEVERE)
        c_after = tracker.confidence
        assert c_after < c_before

    def test_should_retrain(self):
        config = _make_config(min_confidence=0.3, confidence_hl_days=2)
        tracker = ModelConfidenceTracker(config)
        tracker.on_model_retrained(datetime.now() - timedelta(days=50))
        assert tracker.should_retrain

    def test_days_since_training(self):
        tracker = ModelConfidenceTracker(_make_config())
        tracker.on_model_retrained(datetime.now() - timedelta(days=5))
        assert 4 <= tracker.days_since_training <= 6

    def test_no_training_zero_days(self):
        tracker = ModelConfidenceTracker(_make_config())
        assert tracker.days_since_training == 0.0


# ---------------------------------------------------------------------------
# OnlineResidualCorrector
# ---------------------------------------------------------------------------


class TestResidualCorrector:
    def test_no_correction_before_warmup(self):
        config = _make_config(min_samples_for_correction=10)
        corrector = OnlineResidualCorrector(FEATURES, config)
        meta = {"days_since_train": 0.0, "regime_id": 1.0, "drift_severity": 0.0}
        raw = 0.05
        assert corrector.correct(raw, {"mom_20": 0.1, "vol_60": 0.2, "size": 0.5}, meta) == raw

    def test_correction_after_warmup(self):
        config = _make_config(min_samples_for_correction=10, correction_lr=0.01)
        corrector = OnlineResidualCorrector(FEATURES, config)
        meta = {"days_since_train": 1.0, "regime_id": 1.0, "drift_severity": 0.0}
        rng = np.random.default_rng(42)

        # Train with systematic positive residual
        for _ in range(20):
            features = {f: float(rng.normal()) for f in FEATURES}
            pred = 0.05
            actual = 0.07  # Model consistently under-predicts
            corrector.update(features, meta, pred, actual)

        # After learning, correction should nudge prediction up
        test_features = {f: 0.0 for f in FEATURES}
        corrected = corrector.correct(0.05, test_features, meta)
        # The correction should push it toward 0.07 (upward)
        assert corrected != 0.05  # At least some correction applied

    def test_correction_bounded(self):
        config = _make_config(correction_max_weight=0.3)
        corrector = OnlineResidualCorrector(FEATURES, config)
        meta = {"days_since_train": 0.0, "regime_id": 1.0, "drift_severity": 0.0}
        rng = np.random.default_rng(42)

        # Force large weights
        for _ in range(50):
            features = {f: float(rng.normal()) for f in FEATURES}
            corrector.update(features, meta, 0.01, 0.5)

        # Correction should be bounded
        corrected = corrector.correct(0.10, {f: 0.0 for f in FEATURES}, meta)
        max_change = 0.10 * config.correction_max_weight
        assert abs(corrected - 0.10) <= max_change + 0.01


# ---------------------------------------------------------------------------
# OnlineLearningModule (integration)
# ---------------------------------------------------------------------------


class TestOnlineLearningModule:
    def test_basic_flow(self):
        module = OnlineLearningModule(FEATURES)
        module.on_model_trained(datetime.now())

        rng = np.random.default_rng(42)
        preds = {"AAPL": 0.05, "MSFT": 0.03}
        feats = {
            t: {f: float(rng.normal()) for f in FEATURES}
            for t in preds
        }

        corrected = module.correct_predictions(preds, feats)
        assert set(corrected.keys()) == {"AAPL", "MSFT"}

    def test_update_with_realized(self):
        module = OnlineLearningModule(FEATURES)
        module.on_model_trained(datetime.now())

        rng = np.random.default_rng(42)
        preds = {"AAPL": 0.05}
        feats = {"AAPL": {f: float(rng.normal()) for f in FEATURES}}

        module.correct_predictions(preds, feats)

        realized = {"AAPL": 0.04}
        report = module.update_with_realized(realized, feats)
        assert isinstance(report, DriftReport)

    def test_exposure_multiplier(self):
        module = OnlineLearningModule(FEATURES)
        module.on_model_trained(datetime.now())
        mult = module.get_exposure_multiplier()
        assert 0 < mult <= 1.0

    def test_status_dict(self):
        module = OnlineLearningModule(FEATURES)
        module.on_model_trained(datetime.now())
        status = module.get_status()
        assert "confidence" in status
        assert "drift_severity" in status
        assert "exposure_multiplier" in status

    def test_retrain_resets_drift(self):
        module = OnlineLearningModule(FEATURES)
        module.on_model_trained(datetime.now())

        # Inject some drift
        rng = np.random.default_rng(42)
        for i in range(30):
            module.drift_monitor.update(
                datetime.now(), _make_features(rng), 0.01, 0.1
            )

        module.on_model_trained(datetime.now())
        report = module.drift_monitor.get_report()
        assert report.severity == DriftSeverity.NONE

    def test_confidence_scales_predictions(self):
        config = _make_config(confidence_hl_days=2, min_confidence=0.3)
        module = OnlineLearningModule(FEATURES, config)
        # Train a long time ago — confidence should be low
        module.on_model_trained(datetime.now() - timedelta(days=30))

        feats = {"AAPL": {f: 0.0 for f in FEATURES}}
        corrected = module.correct_predictions({"AAPL": 0.10}, feats)
        # Low confidence should scale prediction down significantly
        assert abs(corrected["AAPL"]) < 0.10
