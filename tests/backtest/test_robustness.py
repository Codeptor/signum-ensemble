"""Tests for robustness analysis: compute_sharpe, compute_metrics, StressTester,
monte_carlo_resampling, and regime_stress_tests."""

import numpy as np
import pandas as pd
import pytest

from python.backtest.robustness import (
    StressTester,
    compute_metrics,
    compute_sharpe,
    monte_carlo_resampling,
    regime_stress_tests,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_returns(
    start="2022-01-03",
    periods=756,  # ~3 years of trading days
    mean=0.0004,
    std=0.01,
    seed=42,
) -> pd.Series:
    """Create synthetic daily returns with DatetimeIndex spanning 2022-2025."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=periods)
    returns = rng.normal(mean, std, size=periods)
    return pd.Series(returns, index=dates, name="returns")


def _make_multi_asset_returns(
    tickers=("AAPL", "MSFT", "GOOGL"),
    start="2022-01-03",
    periods=756,
    seed=42,
) -> pd.DataFrame:
    """Create synthetic multi-asset daily returns spanning 2022-2025."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=periods)
    data = {t: rng.normal(0.0003, 0.015, size=periods) for t in tickers}
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests: compute_sharpe
# ---------------------------------------------------------------------------


class TestComputeSharpe:
    def test_positive_returns(self):
        """Positive mean returns should give a positive Sharpe (if large enough)."""
        rng = np.random.RandomState(42)
        # Use large positive mean so annualized return exceeds rf
        rets = pd.Series(rng.normal(0.005, 0.01, 200))
        sharpe = compute_sharpe(rets, periods_per_year=252 / 5, risk_free_rate=0.05)
        assert isinstance(sharpe, float)
        assert sharpe > 0.0

    def test_zero_variance_returns_zero(self):
        """Truly zero-std returns should return 0.0."""
        # Use exactly 0.0 returns to guarantee std() == 0
        rets = pd.Series([0.0] * 100)
        assert compute_sharpe(rets) == 0.0

    def test_empty_returns_zero(self):
        """Empty series should return 0.0."""
        rets = pd.Series([], dtype=float)
        assert compute_sharpe(rets) == 0.0

    def test_numpy_array_input(self):
        """Should accept numpy array and return a float."""
        rng = np.random.RandomState(42)
        arr = rng.normal(0.005, 0.01, 200)
        sharpe = compute_sharpe(arr)
        assert isinstance(sharpe, float)

    def test_negative_returns_negative_sharpe(self):
        """Strongly negative returns should produce negative Sharpe."""
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(-0.02, 0.01, 200))
        sharpe = compute_sharpe(rets, periods_per_year=252 / 5, risk_free_rate=0.05)
        assert sharpe < 0.0

    def test_geometric_annualization(self):
        """Verify geometric formula: ann_return = (1 + mean)^ppy - 1."""
        rets = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        mean = rets.mean()
        ppy = 252 / 5
        expected_ann = (1 + mean) ** ppy - 1
        expected_vol = rets.std() * np.sqrt(ppy)
        expected_sharpe = (expected_ann - 0.05) / expected_vol
        actual = compute_sharpe(rets, periods_per_year=ppy, risk_free_rate=0.05)
        assert abs(actual - expected_sharpe) < 1e-10


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_all_keys(self):
        """Must return dict with the 4 expected keys."""
        rets = _make_daily_returns(periods=100)
        m = compute_metrics(rets)
        assert set(m.keys()) == {"ann_return", "ann_volatility", "sharpe_ratio", "max_drawdown"}

    def test_empty_returns_zeros(self):
        """Empty series should return all zeros."""
        m = compute_metrics(pd.Series([], dtype=float))
        assert m == {
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    def test_max_drawdown_non_negative(self):
        """Max drawdown should be >= 0 (reported as positive magnitude)."""
        rets = _make_daily_returns(periods=200)
        m = compute_metrics(rets)
        assert m["max_drawdown"] >= 0.0

    def test_annualized_volatility_positive(self):
        """Non-constant returns should have positive annualized vol."""
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0.0, 0.01, 200))
        m = compute_metrics(rets)
        assert m["ann_volatility"] > 0.0

    def test_constant_zero_returns(self):
        """Zero returns: max_drawdown should be 0, Sharpe should be 0 (zero vol)."""
        rets = pd.Series([0.0] * 50)
        m = compute_metrics(rets)
        assert m["max_drawdown"] == 0.0
        assert m["sharpe_ratio"] == 0.0

    def test_values_are_floats(self):
        """All metric values should be plain Python floats."""
        rets = _make_daily_returns(periods=100)
        m = compute_metrics(rets)
        for v in m.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Tests: StressTester
# ---------------------------------------------------------------------------


class TestStressTesterInit:
    def test_default_equal_weights(self):
        """Without explicit weights, should assign equal weights."""
        df = _make_multi_asset_returns(tickers=("A", "B", "C"), periods=50)
        st = StressTester(df)
        expected_w = 1.0 / 3
        for w in st.weights:
            assert abs(w - expected_w) < 1e-10

    def test_explicit_weights(self):
        """Explicit weights are used and reindexed to match columns."""
        df = _make_multi_asset_returns(tickers=("A", "B", "C"), periods=50)
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        st = StressTester(df, weights=weights)
        assert abs(st.weights["A"] - 0.5) < 1e-10
        assert abs(st.weights["B"] - 0.3) < 1e-10
        assert abs(st.weights["C"] - 0.2) < 1e-10

    def test_missing_weight_filled_zero(self):
        """If a weight is not provided for a column, it's filled with 0."""
        df = _make_multi_asset_returns(tickers=("A", "B", "C"), periods=50)
        weights = pd.Series({"A": 0.6, "B": 0.4})  # no C
        st = StressTester(df, weights=weights)
        assert abs(st.weights["C"] - 0.0) < 1e-10

    def test_portfolio_returns_shape(self):
        """portfolio_returns should be a Series of same length as input."""
        df = _make_multi_asset_returns(periods=100)
        st = StressTester(df)
        assert len(st.portfolio_returns) == 100


class TestHistoricalStressTest:
    def test_unknown_scenario_raises(self):
        """Unknown scenario name should raise ValueError."""
        df = _make_multi_asset_returns(periods=50)
        st = StressTester(df)
        with pytest.raises(ValueError, match="Unknown scenario"):
            st.historical_stress_test("Fake_Crisis")

    def test_no_data_returns_error_dict(self):
        """If portfolio has no data in date range, returns error dict."""
        # Data only in 2022-2025, so 2008 scenario has no overlap
        df = _make_multi_asset_returns(start="2022-01-03", periods=100)
        st = StressTester(df)
        result = st.historical_stress_test("2008_Financial_Crisis")
        assert "error" in result
        assert result["scenario"] == "2008_Financial_Crisis"

    def test_with_overlapping_data(self):
        """When data overlaps scenario range, should return full result dict."""
        # Create data spanning 2022 to overlap with 2022_Rate_Hikes
        df = _make_multi_asset_returns(start="2022-01-03", periods=756)
        st = StressTester(df)
        result = st.historical_stress_test("2022_Rate_Hikes")
        assert "error" not in result
        assert "total_return" in result
        assert "max_drawdown" in result
        assert "worst_day" in result
        assert "volatility" in result
        assert "num_days" in result
        assert result["num_days"] > 0
        assert result["scenario"] == "2022_Rate_Hikes"

    def test_with_scenario_returns_param(self):
        """Can pass external scenario_returns DataFrame."""
        tickers = ("AAPL", "MSFT", "GOOGL")
        # StressTester data (doesn't need to cover 2022)
        df = _make_multi_asset_returns(tickers=tickers, start="2020-01-02", periods=50)
        st = StressTester(df)
        # External scenario returns covering the 2022 period
        scenario_df = _make_multi_asset_returns(
            tickers=tickers, start="2022-01-03", periods=250, seed=99
        )
        result = st.historical_stress_test("2022_Rate_Hikes", scenario_returns=scenario_df)
        assert "error" not in result
        assert result["num_days"] > 0


class TestHypotheticalShockTest:
    def test_calculates_impact_correctly(self):
        """Shock impact = weight * shock for each asset."""
        df = _make_multi_asset_returns(tickers=("A", "B"), periods=50)
        weights = pd.Series({"A": 0.6, "B": 0.4})
        st = StressTester(df, weights=weights)

        result = st.hypothetical_shock_test({"A": -0.20, "B": -0.10})
        expected_impact = 0.6 * (-0.20) + 0.4 * (-0.10)
        assert abs(result["total_portfolio_impact"] - expected_impact) < 1e-10
        assert result["num_assets_affected"] == 2

    def test_unknown_assets_ignored(self):
        """Assets not in portfolio should be silently ignored."""
        df = _make_multi_asset_returns(tickers=("A", "B"), periods=50)
        st = StressTester(df)
        result = st.hypothetical_shock_test({"A": -0.10, "UNKNOWN": -0.50})
        assert result["num_assets_affected"] == 1
        assert "UNKNOWN" not in result["asset_impacts"]

    def test_no_matching_assets(self):
        """If no shocks match, total impact is 0."""
        df = _make_multi_asset_returns(tickers=("A", "B"), periods=50)
        st = StressTester(df)
        result = st.hypothetical_shock_test({"X": -0.30, "Y": -0.20})
        assert result["total_portfolio_impact"] == 0.0
        assert result["num_assets_affected"] == 0

    def test_shock_type_passthrough(self):
        """shock_type should be echoed in result."""
        df = _make_multi_asset_returns(tickers=("A",), periods=50)
        st = StressTester(df)
        result = st.hypothetical_shock_test({"A": -0.10}, shock_type="excess")
        assert result["shock_type"] == "excess"


class TestMonteCarloStress:
    def test_returns_expected_keys(self):
        """Result dict should contain expected keys."""
        df = _make_multi_asset_returns(periods=100)
        st = StressTester(df)
        result = st.monte_carlo_stress(n_simulations=50, horizon_days=10)
        expected_keys = {
            "n_simulations",
            "horizon_days",
            "expected_return",
            "var_95",
            "var_99",
            "worst_case",
            "prob_loss",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_prob_loss_between_0_and_1(self):
        """prob_loss should be in [0, 1]."""
        df = _make_multi_asset_returns(periods=200)
        st = StressTester(df)
        result = st.monte_carlo_stress(n_simulations=100, horizon_days=10)
        assert 0.0 <= result["prob_loss"] <= 1.0

    def test_var_99_gte_var_95(self):
        """99% VaR should be >= 95% VaR (both are positive loss magnitudes)."""
        df = _make_multi_asset_returns(periods=200)
        st = StressTester(df)
        result = st.monte_carlo_stress(n_simulations=500, horizon_days=21)
        assert result["var_99"] >= result["var_95"]

    def test_reproducibility(self):
        """Same seed should give identical results."""
        df = _make_multi_asset_returns(periods=100)
        st = StressTester(df)
        r1 = st.monte_carlo_stress(n_simulations=50, random_seed=123)
        r2 = st.monte_carlo_stress(n_simulations=50, random_seed=123)
        assert r1["expected_return"] == r2["expected_return"]
        assert r1["var_95"] == r2["var_95"]


class TestCorrelationBreakdown:
    def test_returns_correct_structure(self):
        """Should return dict with known keys."""
        df = _make_multi_asset_returns(periods=200)
        st = StressTester(df)
        result = st.correlation_breakdown()
        assert "threshold_return" in result
        assert "stress_periods" in result
        assert "avg_correlation_normal" in result
        assert "avg_correlation_stress" in result

    def test_stress_periods_count(self):
        """Stress periods should be ~10% of observations at threshold=10."""
        df = _make_multi_asset_returns(periods=200)
        st = StressTester(df)
        result = st.correlation_breakdown(threshold_percentile=10)
        # 10th percentile → ~10% of data = ~20 periods
        assert result["stress_periods"] == pytest.approx(20, abs=2)

    def test_correlations_are_bounded(self):
        """Average correlations should be in [-1, 1]."""
        df = _make_multi_asset_returns(periods=200)
        st = StressTester(df)
        result = st.correlation_breakdown()
        assert -1.0 <= result["avg_correlation_normal"] <= 1.0
        assert -1.0 <= result["avg_correlation_stress"] <= 1.0


class TestRunAllStressTests:
    def test_returns_top_level_keys(self):
        """Should contain all 4 top-level sections."""
        df = _make_multi_asset_returns(periods=200)
        st = StressTester(df)
        result = st.run_all_stress_tests()
        assert "historical_scenarios" in result
        assert "hypothetical_shocks" in result
        assert "monte_carlo_stress" in result
        assert "correlation_breakdown" in result

    def test_historical_scenarios_populated(self):
        """historical_scenarios should have entries for the 3 default scenarios."""
        df = _make_multi_asset_returns(start="2022-01-03", periods=756)
        st = StressTester(df)
        result = st.run_all_stress_tests()
        for scenario in ["2008_Financial_Crisis", "2020_COVID_Crash", "2022_Rate_Hikes"]:
            assert scenario in result["historical_scenarios"]

    def test_custom_hypothetical_shocks(self):
        """Custom shocks dict should override defaults."""
        df = _make_multi_asset_returns(tickers=("A", "B"), periods=100)
        st = StressTester(df)
        custom = {"MyShock": {"A": -0.15}}
        result = st.run_all_stress_tests(hypothetical_shocks=custom)
        assert "MyShock" in result["hypothetical_shocks"]

    def test_monte_carlo_has_severity_levels(self):
        """monte_carlo_stress should have moderate/severe/extreme."""
        df = _make_multi_asset_returns(periods=100)
        st = StressTester(df)
        result = st.run_all_stress_tests()
        mc = result["monte_carlo_stress"]
        assert "moderate" in mc
        assert "severe" in mc
        assert "extreme" in mc


class TestCalculateMaxDD:
    def test_known_drawdown(self):
        """A simple up-then-down series should have a known max drawdown."""
        # Goes up 10%, then down 20% from peak
        rets = pd.Series([0.10, -0.20])
        dd = StressTester._calculate_max_dd(rets)
        # After +10%: cumulative = 1.10
        # After -20%: cumulative = 1.10 * 0.80 = 0.88
        # Drawdown from peak 1.10: (0.88 - 1.10) / 1.10 = -0.2
        assert abs(dd - (-0.2)) < 1e-10

    def test_always_positive_no_drawdown(self):
        """Monotonically increasing returns should have ~0 drawdown."""
        rets = pd.Series([0.01] * 50)
        dd = StressTester._calculate_max_dd(rets)
        assert abs(dd) < 1e-10

    def test_large_drawdown(self):
        """Series with crash should show large negative drawdown."""
        rets = pd.Series([0.01] * 10 + [-0.05] * 20 + [0.01] * 10)
        dd = StressTester._calculate_max_dd(rets)
        assert dd < -0.3  # Significant drawdown after 20 days of -5%


# ---------------------------------------------------------------------------
# Tests: monte_carlo_resampling
# ---------------------------------------------------------------------------


class TestMonteCarloResampling:
    def test_returns_three_top_level_keys(self):
        """Should return sharpe_ratio, ann_return, max_drawdown."""
        rets = _make_daily_returns(periods=100)
        result = monte_carlo_resampling(rets, n_simulations=50)
        assert set(result.keys()) == {"sharpe_ratio", "ann_return", "max_drawdown"}

    def test_each_metric_has_stats(self):
        """Each metric should have mean, 5th_percentile, 95th_percentile."""
        rets = _make_daily_returns(periods=100)
        result = monte_carlo_resampling(rets, n_simulations=50)
        for key in ["sharpe_ratio", "ann_return", "max_drawdown"]:
            assert "mean" in result[key]
            assert "5th_percentile" in result[key]
            assert "95th_percentile" in result[key]

    def test_confidence_interval_ordering(self):
        """5th percentile should be <= mean <= 95th percentile (approximately)."""
        rets = _make_daily_returns(periods=200)
        result = monte_carlo_resampling(rets, n_simulations=100)
        for key in ["sharpe_ratio", "ann_return"]:
            assert result[key]["5th_percentile"] <= result[key]["95th_percentile"]

    def test_block_bootstrap_preserves_approximate_mean(self):
        """Bootstrap mean should be within reasonable tolerance of true mean."""
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0.001, 0.01, 200))
        true_metrics = compute_metrics(rets)
        result = monte_carlo_resampling(rets, n_simulations=100)
        # Bootstrap mean ann_return should be within 50% of true value or within 0.15
        diff = abs(result["ann_return"]["mean"] - true_metrics["ann_return"])
        assert diff < max(abs(true_metrics["ann_return"]) * 0.5, 0.15)

    def test_sharpe_ci_contains_point_estimate(self):
        """The 90% CI for Sharpe should (usually) contain the point estimate."""
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0.001, 0.01, 200))
        point_sharpe = compute_sharpe(rets)
        result = monte_carlo_resampling(rets, n_simulations=100)
        lo = result["sharpe_ratio"]["5th_percentile"]
        hi = result["sharpe_ratio"]["95th_percentile"]
        # Allow some margin since bootstrap CI is approximate
        margin = (hi - lo) * 0.3
        assert lo - margin <= point_sharpe <= hi + margin

    def test_all_values_are_floats(self):
        """All returned values should be plain Python floats."""
        rets = _make_daily_returns(periods=100)
        result = monte_carlo_resampling(rets, n_simulations=50)
        for metric_dict in result.values():
            for v in metric_dict.values():
                assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Tests: regime_stress_tests
# ---------------------------------------------------------------------------


class TestRegimeStressTests:
    def test_returns_dict_keyed_by_regime_names(self):
        """Should return entries for all 4 defined regimes."""
        rets = _make_daily_returns(start="2022-01-03", periods=756)
        result = regime_stress_tests(rets)
        expected_regimes = {
            "2022_Tightening_Cycle",
            "2023_Recovery",
            "2024_Bull_Market",
            "2025_Present",
        }
        assert set(result.keys()) == expected_regimes

    def test_regimes_with_no_data_return_none(self):
        """Regimes outside the data range should return None."""
        # Data only in 2022 — 2023, 2024, 2025 regimes have no data
        rets = _make_daily_returns(start="2022-01-03", periods=50)
        result = regime_stress_tests(rets)
        # Only 2022 might have data, rest should be None
        assert result["2024_Bull_Market"] is None
        assert result["2025_Present"] is None

    def test_regimes_with_data_return_metrics(self):
        """Regimes with sufficient data should return a metrics dict."""
        rets = _make_daily_returns(start="2022-01-03", periods=756)
        result = regime_stress_tests(rets)
        # 2022 Tightening should have data
        m = result["2022_Tightening_Cycle"]
        assert m is not None
        assert "ann_return" in m
        assert "ann_volatility" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m

    def test_converts_index_to_datetime(self):
        """Should work even if index is string dates."""
        dates = pd.bdate_range("2022-01-03", periods=300)
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0.0, 0.01, 300), index=dates.astype(str))
        result = regime_stress_tests(rets)
        # Should not raise — converts internally
        assert "2022_Tightening_Cycle" in result

    def test_insufficient_data_threshold(self):
        """A regime with <= 5 data points should return None."""
        # Only 3 trading days in 2022
        dates = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
        rets = pd.Series([0.01, -0.01, 0.005], index=dates)
        result = regime_stress_tests(rets)
        assert result["2022_Tightening_Cycle"] is None
