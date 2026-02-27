"""Tests for HMM-based regime detection."""

import numpy as np
import pandas as pd
import pytest

from python.monitoring.hmm_regime import HMMRegimeDetector, HMMRegimeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n=500, seed=42, regime_switch=True):
    """Generate synthetic returns with regime structure.

    First half: low-vol regime (mu=0.05%/day, sigma=0.8%/day)
    Second half: high-vol regime (mu=-0.02%/day, sigma=2.5%/day)
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)

    if regime_switch:
        mid = n // 2
        low_vol = rng.normal(0.0005, 0.008, mid)
        high_vol = rng.normal(-0.0002, 0.025, n - mid)
        returns = np.concatenate([low_vol, high_vol])
    else:
        returns = rng.normal(0.0003, 0.01, n)

    return pd.Series(returns, index=dates, name="returns")


# ---------------------------------------------------------------------------
# Tests: Fitting
# ---------------------------------------------------------------------------


class TestHMMFit:
    def test_fit_succeeds(self):
        returns = _make_returns(300)
        detector = HMMRegimeDetector()
        detector.fit(returns)
        assert detector._fitted is True

    def test_fit_too_few_observations(self):
        returns = _make_returns(30)
        detector = HMMRegimeDetector()
        with pytest.raises(ValueError, match="at least 60"):
            detector.fit(returns)

    def test_state_order_sorted_by_volatility(self):
        returns = _make_returns(500, regime_switch=True)
        detector = HMMRegimeDetector()
        detector.fit(returns)
        # State order should be sorted: low-vol state first
        mean_vols = detector._model.means_[:, 2]
        ordered = [mean_vols[s] for s in detector._state_order]
        assert all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))

    def test_fit_returns_self(self):
        returns = _make_returns(200)
        detector = HMMRegimeDetector()
        result = detector.fit(returns)
        assert result is detector


# ---------------------------------------------------------------------------
# Tests: Prediction
# ---------------------------------------------------------------------------


class TestHMMPredict:
    @pytest.fixture
    def fitted_detector(self):
        returns = _make_returns(500, regime_switch=True)
        detector = HMMRegimeDetector()
        detector.fit(returns)
        return detector, returns

    def test_predict_regime_returns_state(self, fitted_detector):
        detector, returns = fitted_detector
        state = detector.predict_regime(returns.tail(60))
        assert isinstance(state, HMMRegimeState)
        assert state.regime in {"low_vol", "normal", "high_vol"}

    def test_predict_regime_probabilities_sum_to_one(self, fitted_detector):
        detector, returns = fitted_detector
        state = detector.predict_regime(returns.tail(60))
        total_prob = sum(state.probabilities.values())
        assert total_prob == pytest.approx(1.0, abs=1e-6)

    def test_predict_regime_exposure_in_range(self, fitted_detector):
        detector, returns = fitted_detector
        state = detector.predict_regime(returns.tail(60))
        assert 0.0 <= state.exposure_multiplier <= 1.0

    def test_predict_states_dataframe(self, fitted_detector):
        detector, returns = fitted_detector
        df = detector.predict_states(returns)
        assert isinstance(df, pd.DataFrame)
        assert "regime" in df.columns
        assert "regime_id" in df.columns
        assert "date" in df.columns
        assert len(df) > 0

    def test_predict_states_all_regimes_valid(self, fitted_detector):
        detector, returns = fitted_detector
        df = detector.predict_states(returns)
        valid_regimes = {"low_vol", "normal", "high_vol"}
        assert set(df["regime"].unique()).issubset(valid_regimes)

    def test_predict_not_fitted_raises(self):
        detector = HMMRegimeDetector()
        returns = _make_returns(100)
        with pytest.raises(ValueError, match="not fitted"):
            detector.predict_regime(returns)

    def test_predict_insufficient_data(self, fitted_detector):
        """With very short series, should return safe default."""
        detector, _ = fitted_detector
        short = pd.Series([0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=2))
        state = detector.predict_regime(short)
        assert state.regime == "normal"
        assert "Insufficient" in state.message


# ---------------------------------------------------------------------------
# Tests: Regime structure detection
# ---------------------------------------------------------------------------


class TestRegimeDetection:
    def test_detects_volatility_shift(self):
        """HMM should detect the shift from low-vol to high-vol."""
        returns = _make_returns(500, regime_switch=True, seed=42)
        detector = HMMRegimeDetector(random_state=42)
        detector.fit(returns)

        df = detector.predict_states(returns)

        # First quarter should be mostly low-vol or normal
        first_q = df.iloc[: len(df) // 4]
        # Last quarter should be mostly high-vol or normal
        last_q = df.iloc[3 * len(df) // 4 :]

        # The average regime_id should be higher in the last quarter
        assert last_q["regime_id"].mean() > first_q["regime_id"].mean()

    def test_stable_returns_no_dominant_high_vol(self):
        """Constant-volatility returns should not be mostly high_vol."""
        returns = _make_returns(400, regime_switch=False, seed=42)
        detector = HMMRegimeDetector(random_state=42)
        detector.fit(returns)

        df = detector.predict_states(returns)
        # With no regime shift, high_vol should not dominate
        high_vol_frac = (df["regime"] == "high_vol").mean()
        assert high_vol_frac < 0.6


# ---------------------------------------------------------------------------
# Tests: Custom exposure map
# ---------------------------------------------------------------------------


class TestExposureMap:
    def test_custom_exposure(self):
        returns = _make_returns(300)
        custom = {"low_vol": 1.0, "normal": 0.5, "high_vol": 0.0}
        detector = HMMRegimeDetector(exposure_map=custom)
        detector.fit(returns)

        state = detector.predict_regime(returns.tail(60))
        assert state.exposure_multiplier in {1.0, 0.5, 0.0}


# ---------------------------------------------------------------------------
# Tests: JSON export
# ---------------------------------------------------------------------------


class TestToJson:
    def test_json_not_fitted(self):
        detector = HMMRegimeDetector()
        result = detector.to_json()
        assert result["fitted"] is False

    def test_json_fitted(self):
        returns = _make_returns(300)
        detector = HMMRegimeDetector()
        detector.fit(returns)

        result = detector.to_json()
        assert result["fitted"] is True
        assert result["n_states"] == 3
        assert "state_means" in result
        assert "low_vol" in result["state_means"]
        assert "vol_20d" in result["state_means"]["low_vol"]


# ---------------------------------------------------------------------------
# Tests: Comparison with threshold-based
# ---------------------------------------------------------------------------


class TestComparison:
    def test_compare_with_threshold(self):
        returns = _make_returns(300, seed=42)
        detector = HMMRegimeDetector()
        detector.fit(returns)

        vix = pd.Series(20.0, index=returns.index)
        spy_dd = pd.Series(0.02, index=returns.index)

        comparison = detector.compare_with_threshold(returns, vix, spy_dd)
        assert "threshold_regime" in comparison.columns
        assert "agreement" in comparison.columns
        assert len(comparison) > 0
