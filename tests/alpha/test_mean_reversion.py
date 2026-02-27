"""Tests for OU estimation and mean-reversion signals."""

import numpy as np
import pytest

from python.alpha.mean_reversion import (
    KalmanHedgeRatio,
    OUEstimator,
    OUParams,
    OptimalThresholds,
    optimal_ou_thresholds,
    pca_spreads,
    zscore_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_ou(kappa=0.1, mu=0.0, sigma=0.05, n=1000, dt=1.0, seed=42):
    """Simulate Ornstein-Uhlenbeck process."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return x


# ---------------------------------------------------------------------------
# OUEstimator — OLS
# ---------------------------------------------------------------------------


class TestOUEstimatorOLS:
    def test_returns_params(self):
        series = _simulate_ou()
        est = OUEstimator(method="ols")
        params = est.fit(series)
        assert isinstance(params, OUParams)

    def test_kappa_positive(self):
        series = _simulate_ou(kappa=0.1)
        params = OUEstimator(method="ols").fit(series)
        assert params.kappa > 0

    def test_recovers_kappa(self):
        series = _simulate_ou(kappa=0.1, n=5000)
        params = OUEstimator(method="ols").fit(series)
        assert abs(params.kappa - 0.1) < 0.05

    def test_recovers_mu(self):
        series = _simulate_ou(kappa=0.1, mu=1.0, n=5000)
        params = OUEstimator(method="ols").fit(series)
        assert abs(params.mu - 1.0) < 0.5

    def test_half_life_consistent(self):
        series = _simulate_ou(kappa=0.1)
        params = OUEstimator(method="ols").fit(series)
        expected_hl = np.log(2) / params.kappa if params.kappa > 0 else float("inf")
        assert params.half_life == pytest.approx(expected_hl, rel=0.01)

    def test_is_mean_reverting(self):
        series = _simulate_ou(kappa=0.1)
        params = OUEstimator(method="ols").fit(series)
        assert params.is_mean_reverting

    def test_random_walk_not_mean_reverting(self):
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.standard_normal(1000))
        params = OUEstimator(method="ols").fit(series)
        # Should have kappa ≈ 0 or infinite half-life
        assert params.half_life > 100 or not params.is_mean_reverting

    def test_short_series(self):
        params = OUEstimator(method="ols").fit(np.array([1.0, 2.0, 1.5]))
        assert isinstance(params, OUParams)


# ---------------------------------------------------------------------------
# OUEstimator — MLE
# ---------------------------------------------------------------------------


class TestOUEstimatorMLE:
    def test_returns_params(self):
        series = _simulate_ou()
        params = OUEstimator(method="mle").fit(series)
        assert isinstance(params, OUParams)

    def test_kappa_positive(self):
        series = _simulate_ou(kappa=0.1, n=2000)
        params = OUEstimator(method="mle").fit(series)
        assert params.kappa > 0

    def test_log_likelihood_finite(self):
        series = _simulate_ou(kappa=0.1, n=2000)
        params = OUEstimator(method="mle").fit(series)
        assert np.isfinite(params.log_likelihood)

    def test_mean_reversion_speed_labels(self):
        fast = OUParams(kappa=0.5, mu=0.0, sigma=0.05, half_life=1.4, log_likelihood=0.0)
        slow = OUParams(kappa=0.01, mu=0.0, sigma=0.05, half_life=69.3, log_likelihood=0.0)
        assert fast.mean_reversion_speed == "very_fast"
        assert slow.mean_reversion_speed == "very_slow"


# ---------------------------------------------------------------------------
# Kalman Hedge Ratio
# ---------------------------------------------------------------------------


class TestKalmanHedgeRatio:
    def test_returns_arrays(self):
        rng = np.random.default_rng(42)
        n = 500
        x = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        y = 1.5 * x + rng.standard_normal(n) * 2
        kf = KalmanHedgeRatio(delta=1e-4)
        betas, spreads = kf.filter(y, x)
        assert len(betas) == n
        assert len(spreads) == n

    def test_converges_to_true_ratio(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        true_beta = 1.5
        y = true_beta * x + rng.standard_normal(n) * 2
        kf = KalmanHedgeRatio(delta=1e-4, initial_beta=1.0)
        betas, _ = kf.filter(y, x)
        # Should converge near true beta
        assert abs(betas[-1] - true_beta) < 0.2

    def test_adapts_to_change(self):
        """Beta should adapt when true hedge ratio changes."""
        rng = np.random.default_rng(42)
        n = 1000
        x = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        # Beta changes from 1.0 to 2.0 at midpoint
        y = np.zeros(n)
        y[:500] = 1.0 * x[:500] + rng.standard_normal(500)
        y[500:] = 2.0 * x[500:] + rng.standard_normal(500)
        kf = KalmanHedgeRatio(delta=1e-3)
        betas, _ = kf.filter(y, x)
        # First half should be near 1, second half near 2
        assert abs(betas[400] - 1.0) < 0.5
        assert abs(betas[-1] - 2.0) < 0.5

    def test_spread_is_residual(self):
        rng = np.random.default_rng(42)
        n = 500
        x = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        y = 1.5 * x + rng.standard_normal(n) * 2
        kf = KalmanHedgeRatio()
        _, spreads = kf.filter(y, x)
        # Spread should be roughly zero-mean
        assert abs(np.mean(spreads[100:])) < 5


# ---------------------------------------------------------------------------
# Optimal Thresholds
# ---------------------------------------------------------------------------


class TestOptimalThresholds:
    def test_returns_thresholds(self):
        params = OUParams(kappa=0.1, mu=0.0, sigma=0.05, half_life=6.93, log_likelihood=0.0)
        thresholds = optimal_ou_thresholds(params)
        assert isinstance(thresholds, OptimalThresholds)

    def test_symmetric(self):
        params = OUParams(kappa=0.1, mu=0.0, sigma=0.05, half_life=6.93, log_likelihood=0.0)
        thresholds = optimal_ou_thresholds(params)
        assert thresholds.entry_long == -thresholds.entry_short
        assert thresholds.exit_long == -thresholds.exit_short

    def test_entry_beyond_exit(self):
        params = OUParams(kappa=0.1, mu=0.0, sigma=0.05, half_life=6.93, log_likelihood=0.0)
        thresholds = optimal_ou_thresholds(params)
        assert thresholds.entry_short > thresholds.exit_short

    def test_non_reverting_defaults(self):
        params = OUParams(kappa=0.0, mu=0.0, sigma=0.05, half_life=float("inf"), log_likelihood=0.0)
        thresholds = optimal_ou_thresholds(params)
        assert thresholds.expected_trades_per_year == 0.0


# ---------------------------------------------------------------------------
# PCA Spreads
# ---------------------------------------------------------------------------


class TestPCASpreads:
    def test_returns_arrays(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal((500, 5)) * 0.5, axis=0)
        spreads, weights = pca_spreads(prices, n_spreads=2)
        assert spreads.shape == (500, 2)
        assert weights.shape == (2, 5)

    def test_spreads_mean_reverting(self):
        """PCA spreads from smallest eigenvalues should be more mean-reverting."""
        rng = np.random.default_rng(42)
        # Create correlated price series
        factor = np.cumsum(rng.standard_normal(500) * 0.5)
        prices = np.column_stack([
            100 + factor + rng.standard_normal(500) * 0.1 * (i + 1)
            for i in range(5)
        ])
        spreads, _ = pca_spreads(prices, n_spreads=1)
        # The spread should be roughly stationary
        adf_proxy = np.corrcoef(spreads[:-1, 0], np.diff(spreads[:, 0]))[0, 1]
        # Negative autocorrelation of changes → mean-reverting
        assert adf_proxy < 0


# ---------------------------------------------------------------------------
# Z-Score Signal
# ---------------------------------------------------------------------------


class TestZScoreSignal:
    def test_output_shape(self):
        spread = _simulate_ou(kappa=0.1, n=500)
        signals = zscore_signal(spread, lookback=60)
        assert len(signals) == 500

    def test_signal_values(self):
        spread = _simulate_ou(kappa=0.1, n=500)
        signals = zscore_signal(spread)
        assert set(np.unique(signals)) <= {-1, 0, 1}

    def test_no_signal_in_lookback(self):
        spread = _simulate_ou(kappa=0.1, n=500)
        signals = zscore_signal(spread, lookback=60)
        assert all(signals[:60] == 0)

    def test_generates_trades(self):
        """A mean-reverting series should generate some trades."""
        spread = _simulate_ou(kappa=0.15, sigma=0.08, n=1000)
        signals = zscore_signal(spread, lookback=60, entry_z=1.5, exit_z=0.3)
        n_trades = np.sum(np.abs(np.diff(signals)) > 0)
        assert n_trades > 0
