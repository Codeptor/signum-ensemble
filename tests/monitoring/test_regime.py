"""Tests for market regime detection (Phase 2.2)."""

import pytest

from python.monitoring.regime import RegimeDetector, RegimeState


@pytest.fixture
def detector():
    return RegimeDetector()


class TestRegimeDetector:
    def test_normal_regime(self, detector):
        assert detector.get_regime(vix=15.0, spy_drawdown=0.03) == "normal"

    def test_caution_regime_vix(self, detector):
        assert detector.get_regime(vix=35.0, spy_drawdown=0.05) == "caution"

    def test_caution_regime_drawdown(self, detector):
        assert detector.get_regime(vix=20.0, spy_drawdown=0.12) == "caution"

    def test_halt_regime_vix(self, detector):
        assert detector.get_regime(vix=45.0, spy_drawdown=0.05) == "halt"

    def test_halt_regime_drawdown(self, detector):
        assert detector.get_regime(vix=20.0, spy_drawdown=0.18) == "halt"

    def test_halt_both_extreme(self, detector):
        assert detector.get_regime(vix=50.0, spy_drawdown=0.20) == "halt"

    def test_boundary_vix_caution(self, detector):
        """VIX exactly at caution threshold."""
        assert detector.get_regime(vix=30.0, spy_drawdown=0.05) == "normal"
        assert detector.get_regime(vix=30.1, spy_drawdown=0.05) == "caution"

    def test_boundary_vix_halt(self, detector):
        """VIX exactly at halt threshold."""
        assert detector.get_regime(vix=40.0, spy_drawdown=0.05) == "caution"
        assert detector.get_regime(vix=40.1, spy_drawdown=0.05) == "halt"

    def test_boundary_drawdown_caution(self, detector):
        """Drawdown exactly at caution threshold."""
        assert detector.get_regime(vix=15.0, spy_drawdown=0.10) == "normal"
        assert detector.get_regime(vix=15.0, spy_drawdown=0.101) == "caution"

    def test_boundary_drawdown_halt(self, detector):
        """Drawdown exactly at halt threshold."""
        assert detector.get_regime(vix=15.0, spy_drawdown=0.15) == "caution"
        assert detector.get_regime(vix=15.0, spy_drawdown=0.151) == "halt"

    def test_negative_drawdown_treated_as_positive(self, detector):
        """Negative drawdown value should be treated as its absolute value."""
        assert detector.get_regime(vix=15.0, spy_drawdown=-0.12) == "caution"


class TestRegimeState:
    def test_normal_state(self, detector):
        state = detector.get_regime_state(vix=15.0, spy_drawdown=0.03)
        assert state.regime == "normal"
        assert state.exposure_multiplier == 1.0
        assert state.vix == 15.0
        assert state.spy_drawdown == 0.03

    def test_caution_state(self, detector):
        state = detector.get_regime_state(vix=35.0, spy_drawdown=0.05)
        assert state.regime == "caution"
        assert state.exposure_multiplier == 0.5
        assert "CAUTION" in state.message

    def test_halt_state(self, detector):
        state = detector.get_regime_state(vix=45.0, spy_drawdown=0.05)
        assert state.regime == "halt"
        assert state.exposure_multiplier == 0.0
        assert "HALT" in state.message


class TestAdjustWeights:
    def test_normal_no_change(self, detector):
        weights = {"AAPL": 0.3, "MSFT": 0.3, "JPM": 0.4}
        adjusted = detector.adjust_weights(weights, "normal")
        assert adjusted == weights

    def test_caution_halves_weights(self, detector):
        weights = {"AAPL": 0.3, "MSFT": 0.3, "JPM": 0.4}
        adjusted = detector.adjust_weights(weights, "caution")
        assert adjusted["AAPL"] == pytest.approx(0.15)
        assert adjusted["MSFT"] == pytest.approx(0.15)
        assert adjusted["JPM"] == pytest.approx(0.20)

    def test_halt_returns_empty(self, detector):
        weights = {"AAPL": 0.3, "MSFT": 0.3, "JPM": 0.4}
        adjusted = detector.adjust_weights(weights, "halt")
        assert adjusted == {}

    def test_empty_weights_normal(self, detector):
        assert detector.adjust_weights({}, "normal") == {}

    def test_empty_weights_halt(self, detector):
        assert detector.adjust_weights({}, "halt") == {}


class TestCustomThresholds:
    def test_custom_vix_thresholds(self):
        detector = RegimeDetector(vix_caution=20.0, vix_halt=30.0)
        assert detector.get_regime(vix=25.0, spy_drawdown=0.0) == "caution"
        assert detector.get_regime(vix=35.0, spy_drawdown=0.0) == "halt"

    def test_custom_drawdown_thresholds(self):
        detector = RegimeDetector(drawdown_caution=0.05, drawdown_halt=0.10)
        assert detector.get_regime(vix=15.0, spy_drawdown=0.07) == "caution"
        assert detector.get_regime(vix=15.0, spy_drawdown=0.12) == "halt"

    def test_custom_caution_multiplier(self):
        detector = RegimeDetector(caution_multiplier=0.3)
        weights = {"AAPL": 0.5, "JPM": 0.5}
        adjusted = detector.adjust_weights(weights, "caution")
        assert adjusted["AAPL"] == pytest.approx(0.15)
        assert adjusted["JPM"] == pytest.approx(0.15)
