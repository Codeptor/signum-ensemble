"""Tests for market regime detection (Phase 2.2)."""

import pandas as pd
import pytest

from python.monitoring.regime import (
    RegimeDetector,
    _extract_close_series,
)


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


# =====================================================================
# Tests for audit fixes: H8, H9, M14
# =====================================================================


class TestExtractCloseSeries:
    """H8: _extract_close_series handles both flat and MultiIndex columns."""

    def test_flat_columns(self):
        """Standard DataFrame with 'Close' column returns Series."""
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        result = _extract_close_series(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_multiindex_columns(self):
        """yfinance MultiIndex columns (e.g. ('Close', '^VIX')) are squeezed."""
        arrays = [["Close", "Open"], ["^VIX", "^VIX"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[100.0, 99.0], [101.0, 100.0]], columns=index)
        result = _extract_close_series(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_single_ticker_multiindex(self):
        """Single-ticker download returns MultiIndex even for one ticker.

        yfinance >= 0.2.31 returns columns like MultiIndex([('Close', '^VIX'),
        ('Open', '^VIX')]).  df["Close"] then yields a *DataFrame* (one column
        named '^VIX'), not a Series.  _extract_close_series must squeeze it.
        """
        mi = pd.MultiIndex.from_tuples(
            [("Close", "^VIX"), ("Open", "^VIX")],
            names=["Price", "Ticker"],
        )
        df = pd.DataFrame(
            [[25.0, 24.0], [26.0, 25.0], [27.0, 26.0]],
            columns=mi,
        )
        result = _extract_close_series(df)
        assert isinstance(result, pd.Series)
        assert float(result.iloc[-1]) == pytest.approx(27.0)


class TestHysteresis:
    """H9: regime de-escalation requires crossing threshold minus hysteresis band."""

    def test_escalation_is_immediate(self):
        """Escalation from normal→caution uses primary threshold (no delay)."""
        d = RegimeDetector()
        assert d.get_regime(vix=31.0, spy_drawdown=0.0) == "caution"

    def test_deescalation_requires_band_crossing(self):
        """De-escalation from caution→normal requires VIX < caution - hysteresis."""
        d = RegimeDetector()  # vix_caution=30, vix_hysteresis=2
        # Escalate to caution
        d.get_regime(vix=31.0, spy_drawdown=0.0)
        # VIX drops to 29 — still within hysteresis band (30-2=28), stays caution
        assert d.get_regime(vix=29.0, spy_drawdown=0.0) == "caution"
        # VIX drops below 28 — clears hysteresis band, back to normal
        assert d.get_regime(vix=27.5, spy_drawdown=0.0) == "normal"

    def test_halt_deescalation_requires_band_crossing(self):
        """De-escalation from halt: OR logic allows partial de-escalation.

        M-HYSTERESIS fix: changed from AND (both VIX AND drawdown must clear)
        to OR (either VIX OR drawdown clearing allows de-escalation to caution).
        This prevents the strategy from being locked in halt for months during
        prolonged drawdowns when VIX has already normalized.
        """
        d = RegimeDetector()  # vix_halt=40, vix_hysteresis=2
        # Escalate to halt via VIX only (drawdown is 0)
        d.get_regime(vix=41.0, spy_drawdown=0.0)
        # VIX drops to 39 — within halt hysteresis band, but drawdown (0.0)
        # clears dd_clear (0.0 <= 0.13), so OR passes → de-escalate to caution
        assert d.get_regime(vix=39.0, spy_drawdown=0.0) == "caution"

        # Now test the case where BOTH signals are bad and neither clears
        d2 = RegimeDetector()
        d2.get_regime(vix=41.0, spy_drawdown=0.16)
        # Both VIX (39 > 38) and drawdown (0.14 > 0.13) are within bands
        assert d2.get_regime(vix=39.0, spy_drawdown=0.14) == "halt"
        # VIX clears band (37 <= 38), drawdown still bad → OR passes → caution
        assert d2.get_regime(vix=37.0, spy_drawdown=0.14) == "caution"

    def test_drawdown_hysteresis(self):
        """Drawdown de-escalation also uses hysteresis band."""
        d = RegimeDetector()  # drawdown_caution=0.10, drawdown_hysteresis=0.02
        # Escalate via drawdown
        d.get_regime(vix=15.0, spy_drawdown=0.11)
        # Drawdown improves to 0.09 — within band (0.10 - 0.02 = 0.08)
        assert d.get_regime(vix=15.0, spy_drawdown=0.09) == "caution"
        # Drawdown improves below 0.08
        assert d.get_regime(vix=15.0, spy_drawdown=0.07) == "normal"

    def test_rapid_oscillation_stays_sticky(self):
        """VIX oscillating around 30 should not cause rapid regime changes."""
        d = RegimeDetector()
        regimes = []
        # Simulate VIX oscillating: 29, 31, 29, 31, 29, 31
        for vix in [29.0, 31.0, 29.0, 31.0, 29.0, 31.0]:
            regimes.append(d.get_regime(vix=vix, spy_drawdown=0.0))
        # After first escalation at VIX=31, should stay caution
        # (29 > 28=threshold-band, so no de-escalation)
        assert regimes == ["normal", "caution", "caution", "caution", "caution", "caution"]

    def test_custom_hysteresis_bands(self):
        """Custom hysteresis values are respected."""
        d = RegimeDetector(vix_caution=30, vix_hysteresis=5)
        d.get_regime(vix=31.0, spy_drawdown=0.0)
        # With 5-point band, need VIX < 25 to de-escalate
        assert d.get_regime(vix=26.0, spy_drawdown=0.0) == "caution"
        assert d.get_regime(vix=24.0, spy_drawdown=0.0) == "normal"


class TestIsHalt:
    """M14: is_halt() provides unambiguous halt detection."""

    def test_is_halt_true(self):
        assert RegimeDetector.is_halt("halt") is True

    def test_is_halt_false_caution(self):
        assert RegimeDetector.is_halt("caution") is False

    def test_is_halt_false_normal(self):
        assert RegimeDetector.is_halt("normal") is False
