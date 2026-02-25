"""Tests for Brinson attribution model."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.brinson_attribution import (
    BrinsonAttribution,
    calculate_brinson_attribution,
)


@pytest.fixture
def sample_sectors():
    """Sample sector data for testing."""
    return ["Tech", "Finance", "Health", "Energy"]


@pytest.fixture
def portfolio_weights(sample_sectors):
    """Sample portfolio weights."""
    return pd.Series({"Tech": 0.40, "Finance": 0.30, "Health": 0.20, "Energy": 0.10})


@pytest.fixture
def benchmark_weights(sample_sectors):
    """Sample benchmark weights (equal weighted)."""
    return pd.Series({"Tech": 0.25, "Finance": 0.25, "Health": 0.25, "Energy": 0.25})


@pytest.fixture
def portfolio_returns(sample_sectors):
    """Sample portfolio returns by sector."""
    return pd.Series({"Tech": 0.15, "Finance": 0.08, "Health": 0.10, "Energy": 0.05})


@pytest.fixture
def benchmark_returns(sample_sectors):
    """Sample benchmark returns by sector."""
    return pd.Series({"Tech": 0.12, "Finance": 0.06, "Health": 0.08, "Energy": 0.04})


class TestBrinsonAttributionInitialization:
    """Test BrinsonAttribution initialization."""

    def test_init(self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
        """Test initialization."""
        model = BrinsonAttribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        assert model.portfolio_weights is portfolio_weights
        assert model.benchmark_weights is benchmark_weights
        assert model.portfolio_returns is portfolio_returns
        assert model.benchmark_returns is benchmark_returns


class TestAttributionCalculation:
    """Test attribution calculations."""

    def test_total_return_calculation(
        self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    ):
        """Test total return calculation."""
        model = BrinsonAttribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        port_ret = model._calculate_total_return(portfolio_weights, portfolio_returns)
        bench_ret = model._calculate_total_return(benchmark_weights, benchmark_returns)

        # Portfolio: 0.4*0.15 + 0.3*0.08 + 0.2*0.10 + 0.1*0.05 = 0.11
        assert port_ret == pytest.approx(0.11, abs=0.002)
        # Benchmark: 0.25*0.12 + 0.25*0.06 + 0.25*0.08 + 0.25*0.04 = 0.075
        assert bench_ret == pytest.approx(0.075, abs=0.002)

    def test_attribution_structure(
        self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    ):
        """Test attribution result structure."""
        model = BrinsonAttribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        result = model.attribution()

        assert "allocation_effect" in result
        assert "selection_effect" in result
        assert "interaction_effect" in result
        assert "total_excess_return" in result
        assert "portfolio_return" in result
        assert "benchmark_return" in result

    def test_attribution_check(
        self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    ):
        """Test that attribution effects sum to total excess return."""
        model = BrinsonAttribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        result = model.attribution()

        attribution_sum = (
            result["allocation_effect"] + result["selection_effect"] + result["interaction_effect"]
        )

        assert attribution_sum == pytest.approx(result["total_excess_return"], abs=1e-10)
        assert result["attribution_check"] == pytest.approx(
            result["total_excess_return"], abs=1e-10
        )

    def test_attribution_values(
        self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    ):
        """Test specific attribution values."""
        model = BrinsonAttribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        result = model.attribution()

        # Portfolio overweights Tech (40% vs 25%) which outperforms
        # So allocation effect should be positive
        assert result["allocation_effect"] > 0

        # Portfolio returns exceed benchmark in each sector
        # So selection effect should be positive
        assert result["selection_effect"] > 0

        # Total excess return should be positive
        assert result["total_excess_return"] > 0


class TestAttributionReport:
    """Test attribution reporting."""

    def test_report_generation(
        self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    ):
        """Test report generation."""
        model = BrinsonAttribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        report = model.attribution_report()

        assert "BRINSON PERFORMANCE ATTRIBUTION" in report
        assert "Portfolio Return:" in report
        assert "Benchmark Return:" in report
        assert "Allocation Effect:" in report
        assert "Selection Effect:" in report
        assert "Interaction Effect:" in report


class TestConvenienceFunction:
    """Test convenience function."""

    def test_calculate_brinson_attribution(
        self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    ):
        """Test convenience function."""
        result = calculate_brinson_attribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        assert isinstance(result, dict)
        assert "allocation_effect" in result
        assert "selection_effect" in result
        assert "interaction_effect" in result


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_sectors(self):
        """Test with empty sector lists."""
        pw = pd.Series(dtype=float)
        bw = pd.Series(dtype=float)
        pr = pd.Series(dtype=float)
        br = pd.Series(dtype=float)

        model = BrinsonAttribution(pw, bw, pr, br)
        result = model.attribution()

        assert result["portfolio_return"] == 0
        assert result["benchmark_return"] == 0
        assert result["total_excess_return"] == 0

    def test_zero_returns(self, portfolio_weights, benchmark_weights):
        """Test with zero returns."""
        pr = pd.Series({"Tech": 0.0, "Finance": 0.0, "Health": 0.0, "Energy": 0.0})
        br = pd.Series({"Tech": 0.0, "Finance": 0.0, "Health": 0.0, "Energy": 0.0})

        model = BrinsonAttribution(portfolio_weights, benchmark_weights, pr, br)
        result = model.attribution()

        assert result["portfolio_return"] == 0
        assert result["benchmark_return"] == 0
        assert result["total_excess_return"] == 0

    def test_identical_weights(self, portfolio_returns, benchmark_returns):
        """Test with identical portfolio and benchmark weights."""
        weights = pd.Series({"Tech": 0.25, "Finance": 0.25, "Health": 0.25, "Energy": 0.25})

        model = BrinsonAttribution(weights, weights, portfolio_returns, benchmark_returns)
        result = model.attribution()

        # With identical weights, allocation effect should be 0
        assert result["allocation_effect"] == pytest.approx(0, abs=1e-10)

    def test_negative_excess_return(self):
        """Test with underperforming portfolio."""
        pw = pd.Series({"Tech": 0.40, "Finance": 0.30, "Health": 0.20, "Energy": 0.10})
        bw = pd.Series({"Tech": 0.25, "Finance": 0.25, "Health": 0.25, "Energy": 0.25})
        # Portfolio underperforms in all sectors
        pr = pd.Series({"Tech": 0.05, "Finance": 0.03, "Health": 0.04, "Energy": 0.02})
        br = pd.Series({"Tech": 0.15, "Finance": 0.08, "Health": 0.10, "Energy": 0.05})

        model = BrinsonAttribution(pw, bw, pr, br)
        result = model.attribution()

        assert result["total_excess_return"] < 0
