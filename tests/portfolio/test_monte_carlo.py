"""Tests for Monte Carlo portfolio simulation."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.monte_carlo import (
    MonteCarloSimulator,
    RiskMetrics,
    _max_drawdown,
)


@pytest.fixture
def returns_data():
    """Synthetic daily returns for 5 assets."""
    np.random.seed(42)
    n_days = 500
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    # Correlated returns
    mu = np.array([0.0003, 0.0002, 0.0004, 0.0003, 0.0002])
    cov = np.array([
        [0.0004, 0.0001, 0.00015, 0.0001, 0.0001],
        [0.0001, 0.0003, 0.0001, 0.0001, 0.00008],
        [0.00015, 0.0001, 0.0005, 0.00012, 0.0001],
        [0.0001, 0.0001, 0.00012, 0.0004, 0.00009],
        [0.0001, 0.00008, 0.0001, 0.00009, 0.00035],
    ])
    returns = np.random.multivariate_normal(mu, cov, n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def weights():
    return pd.Series(
        [0.2, 0.2, 0.2, 0.2, 0.2],
        index=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
    )


# ---------------------------------------------------------------------------
# Return simulation
# ---------------------------------------------------------------------------


class TestSimulateReturns:
    def test_normal_shape(self, returns_data):
        sim = MonteCarloSimulator(returns_data, n_scenarios=100, horizon=60, seed=42)
        scenarios = sim.simulate_returns()
        assert scenarios.shape == (100, 60, 5)

    def test_t_distribution(self, returns_data):
        sim = MonteCarloSimulator(returns_data, n_scenarios=100, horizon=60, method="t", seed=42)
        scenarios = sim.simulate_returns()
        assert scenarios.shape == (100, 60, 5)

    def test_bootstrap(self, returns_data):
        sim = MonteCarloSimulator(
            returns_data, n_scenarios=100, horizon=60, method="bootstrap", seed=42
        )
        scenarios = sim.simulate_returns()
        assert scenarios.shape == (100, 60, 5)

    def test_unknown_method_raises(self, returns_data):
        sim = MonteCarloSimulator(returns_data, method="unknown")
        with pytest.raises(ValueError, match="Unknown method"):
            sim.simulate_returns()

    def test_reproducible(self, returns_data):
        sim1 = MonteCarloSimulator(returns_data, n_scenarios=50, horizon=30, seed=123)
        sim2 = MonteCarloSimulator(returns_data, n_scenarios=50, horizon=30, seed=123)
        s1 = sim1.simulate_returns()
        s2 = sim2.simulate_returns()
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


class TestSimulatePortfolio:
    def test_paths_shape(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=100, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights)
        assert paths.shape == (100, 61)  # horizon + 1

    def test_starts_at_initial(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=100, horizon=30, seed=42)
        paths = sim.simulate_portfolio(weights, initial_value=100.0)
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_positive_values(self, returns_data, weights):
        """Wealth should stay positive (for reasonable horizon)."""
        sim = MonteCarloSimulator(returns_data, n_scenarios=100, horizon=30, seed=42)
        paths = sim.simulate_portfolio(weights)
        assert (paths > 0).all()

    def test_with_rebalancing(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=50, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights, rebalance_freq=21)
        assert paths.shape == (50, 61)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_metrics_type(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=1000, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights)
        metrics = sim.compute_risk_metrics(paths)
        assert isinstance(metrics, RiskMetrics)

    def test_var_ordering(self, returns_data, weights):
        """VaR 99 should be more negative than VaR 95."""
        sim = MonteCarloSimulator(returns_data, n_scenarios=5000, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights)
        metrics = sim.compute_risk_metrics(paths)
        assert metrics.var_99 <= metrics.var_95

    def test_cvar_more_extreme(self, returns_data, weights):
        """CVaR should be more negative than VaR."""
        sim = MonteCarloSimulator(returns_data, n_scenarios=5000, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights)
        metrics = sim.compute_risk_metrics(paths)
        assert metrics.cvar_95 <= metrics.var_95
        assert metrics.cvar_99 <= metrics.var_99

    def test_drawdown_negative(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=1000, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights)
        metrics = sim.compute_risk_metrics(paths)
        assert metrics.max_dd_median <= 0
        assert metrics.max_dd_95 <= 0

    def test_prob_loss_bounded(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=1000, horizon=60, seed=42)
        paths = sim.simulate_portfolio(weights)
        metrics = sim.compute_risk_metrics(paths)
        assert 0 <= metrics.prob_loss_5pct <= 1
        assert 0 <= metrics.prob_loss_10pct <= 1
        assert 0 <= metrics.prob_loss_20pct <= 1

    def test_to_dict(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=100, horizon=30, seed=42)
        paths = sim.simulate_portfolio(weights)
        metrics = sim.compute_risk_metrics(paths)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "var_95" in d
        assert "cvar_99" in d


# ---------------------------------------------------------------------------
# Tail risk analysis
# ---------------------------------------------------------------------------


class TestTailRisk:
    def test_structure(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=1000, horizon=60, seed=42)
        result = sim.tail_risk_analysis(weights)
        assert isinstance(result, dict)
        assert "loss_5%" in result
        assert "probability" in result["loss_5%"]

    def test_probabilities_bounded(self, returns_data, weights):
        sim = MonteCarloSimulator(returns_data, n_scenarios=1000, horizon=60, seed=42)
        result = sim.tail_risk_analysis(weights)
        for key, data in result.items():
            assert 0 <= data["probability"] <= 1


# ---------------------------------------------------------------------------
# Helper: max drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_drawdown(self):
        path = np.array([1.0, 1.1, 1.2, 1.3])
        assert _max_drawdown(path) == pytest.approx(0.0)

    def test_drawdown(self):
        path = np.array([1.0, 1.2, 0.9, 1.1])
        dd = _max_drawdown(path)
        assert dd < 0
        assert dd == pytest.approx(-0.25, abs=0.01)

    def test_full_loss(self):
        path = np.array([1.0, 0.5, 0.1])
        dd = _max_drawdown(path)
        assert dd == pytest.approx(-0.9, abs=0.01)
