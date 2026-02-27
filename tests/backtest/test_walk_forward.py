"""Tests for walk-forward optimization engine."""

import numpy as np
import pandas as pd
import pytest

from python.backtest.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WindowResult,
    _calmar,
    _sharpe,
    _sortino,
    _total_return,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n_days=500, n_assets=3, seed=42):
    """Synthetic daily returns with slight drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    mu = rng.uniform(0.0001, 0.0005, n_assets)
    data = mu + rng.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _momentum_strategy(returns: pd.DataFrame, params: dict) -> pd.Series:
    """Simple momentum strategy for testing: equal-weight if mean return > threshold."""
    lookback = params.get("lookback", 20)
    threshold = params.get("threshold", 0.0)

    # Can't compute lookback on very short data
    if len(returns) < lookback + 1:
        return pd.Series(0.0, index=returns.index)

    signal = returns.rolling(lookback).mean().mean(axis=1)
    weights = (signal > threshold).astype(float) / max(returns.shape[1], 1)
    port_returns = (returns * weights.values.reshape(-1, 1)).sum(axis=1)
    return port_returns


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


class TestMetricFunctions:
    def test_sharpe_positive(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.01, 252))
        s = _sharpe(r)
        assert s > 0

    def test_sharpe_zero_std(self):
        r = pd.Series([0.0, 0.0, 0.0])
        assert _sharpe(r) == 0.0

    def test_sortino_positive(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.01, 252))
        assert _sortino(r) > 0

    def test_calmar_positive(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.005, 252))
        assert _calmar(r) > 0

    def test_total_return(self):
        r = pd.Series([0.01, 0.02, -0.01])
        expected = (1.01 * 1.02 * 0.99) - 1
        assert _total_return(r) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


class TestWindowGeneration:
    def test_rolling_produces_windows(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=3, anchored=False,
        )
        windows = wfo._generate_windows()
        assert len(windows) == 3

    def test_anchored_produces_windows(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=3, anchored=True,
        )
        windows = wfo._generate_windows()
        assert len(windows) >= 2

    def test_windows_non_overlapping_test(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=3, anchored=False,
        )
        windows = wfo._generate_windows()
        for i in range(len(windows) - 1):
            _, _, _, test_end_i = windows[i]
            _, _, test_start_j, _ = windows[i + 1]
            assert test_end_i < test_start_j

    def test_train_before_test(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=3,
        )
        windows = wfo._generate_windows()
        for train_start, train_end, test_start, test_end in windows:
            assert train_end < test_start
            assert test_start <= test_end

    def test_purge_gap(self):
        returns = _make_returns(n_days=500)
        purge = 10
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=3, purge_days=purge,
        )
        windows = wfo._generate_windows()
        for _, train_end, test_start, _ in windows:
            gap = (test_start - train_end).days
            # Business days gap should be at least purge_days
            assert gap >= purge - 3  # allow for weekends

    def test_too_few_data_raises(self):
        returns = _make_returns(n_days=30)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=5, min_train_days=60,
        )
        with pytest.raises(ValueError):
            wfo._generate_windows()


# ---------------------------------------------------------------------------
# Basic run
# ---------------------------------------------------------------------------


class TestBasicRun:
    def test_returns_result(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20], "threshold": [0.0]},
            n_windows=3,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        assert isinstance(result, WalkForwardResult)

    def test_oos_returns_not_empty(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20], "threshold": [0.0]},
            n_windows=3,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        assert len(result.oos_returns) > 0

    def test_n_windows_matches(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20]},
            n_windows=3,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        assert result.n_windows == 3

    def test_each_window_has_params(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20, 40], "threshold": [0.0, 0.0001]},
            n_windows=3,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        for w in result.windows:
            assert "lookback" in w.best_params
            assert "threshold" in w.best_params


# ---------------------------------------------------------------------------
# Metrics and properties
# ---------------------------------------------------------------------------


class TestResultProperties:
    @pytest.fixture
    def result(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20], "threshold": [0.0]},
            n_windows=3,
            min_train_days=30,
        )
        return wfo.run(_momentum_strategy)

    def test_oos_sharpe_finite(self, result):
        assert np.isfinite(result.oos_sharpe)

    def test_oos_total_return_finite(self, result):
        assert np.isfinite(result.oos_total_return)

    def test_oos_max_drawdown_non_positive(self, result):
        assert result.oos_max_drawdown <= 0.01  # allow tiny numerical error

    def test_efficiency_ratio_finite(self, result):
        assert np.isfinite(result.efficiency_ratio)

    def test_degradation_slope_finite(self, result):
        assert np.isfinite(result.degradation_slope)

    def test_summary_string(self, result):
        s = result.summary()
        assert "Walk-Forward" in s
        assert "OOS" in s or "Sharpe" in s

    def test_total_oos_days(self, result):
        assert result.total_oos_days > 0


# ---------------------------------------------------------------------------
# Anchored vs rolling
# ---------------------------------------------------------------------------


class TestAnchoredVsRolling:
    def test_anchored_runs(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20]},
            n_windows=3,
            anchored=True,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        assert result.n_windows >= 2

    def test_anchored_expanding_train(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20]},
            n_windows=3,
            anchored=True,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        # All anchored windows should start from the same point
        starts = [w.train_start for w in result.windows]
        assert len(set(starts)) == 1


# ---------------------------------------------------------------------------
# Custom metric
# ---------------------------------------------------------------------------


class TestCustomMetric:
    def test_custom_metric_fn(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20]},
            n_windows=3,
            min_train_days=30,
        )

        def my_metric(r):
            return float(r.mean() * 1000)

        result = wfo.run(_momentum_strategy, custom_metric_fn=my_metric)
        assert result.n_windows >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_param_combo(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [20]},
            n_windows=3,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        for w in result.windows:
            assert w.best_params == {"lookback": 20}

    def test_unknown_metric_raises(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [20]}, n_windows=3, min_train_days=30,
        )
        with pytest.raises(ValueError, match="Unknown metric"):
            wfo.run(_momentum_strategy, metric="invalid")

    def test_empty_param_grid_raises(self):
        returns = _make_returns(n_days=500)
        with pytest.raises(ValueError, match="no parameter combinations"):
            WalkForwardOptimizer(returns, {}, n_windows=3)

    def test_param_stability_computed(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns,
            {"lookback": [10, 20, 40]},
            n_windows=3,
            min_train_days=30,
        )
        result = wfo.run(_momentum_strategy)
        assert "lookback" in result.param_stability

    def test_different_metrics(self):
        returns = _make_returns(n_days=500)
        wfo = WalkForwardOptimizer(
            returns, {"lookback": [10, 20]}, n_windows=3, min_train_days=30,
        )
        for m in ["sharpe", "sortino", "calmar", "total_return"]:
            result = wfo.run(_momentum_strategy, metric=m)
            assert result.n_windows >= 2
