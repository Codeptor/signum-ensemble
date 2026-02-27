"""Tests for Barra-style factor risk model."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.factor_risk import (
    STRESS_SCENARIOS,
    FactorRiskModel,
    _zscore_dict,
)


@pytest.fixture
def price_data():
    """Synthetic price data for 5 assets over 300 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=300)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    # Generate correlated random walks
    n_days = len(dates)
    n_assets = len(tickers)
    # Factor-based returns: market + idio
    market = np.random.randn(n_days) * 0.01
    returns = np.zeros((n_days, n_assets))
    betas = [1.2, 0.9, 1.1, 1.3, 1.0]
    for i in range(n_assets):
        idio = np.random.randn(n_days) * 0.008
        returns[:, i] = betas[i] * market + idio

    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=tickers,
    )
    return prices


@pytest.fixture
def equal_weights(price_data):
    return pd.Series(np.ones(5) / 5, index=price_data.columns)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_succeeds(self, price_data):
        model = FactorRiskModel(price_data)
        model.fit()
        assert model._fitted is True

    def test_factor_exposures_shape(self, price_data):
        model = FactorRiskModel(price_data).fit()
        assert model._factor_exposures.shape == (5, 4)

    def test_factor_returns_shape(self, price_data):
        model = FactorRiskModel(price_data).fit()
        assert model._factor_returns.shape[1] == 4
        assert len(model._factor_returns) == len(model.returns)

    def test_factor_covariance_symmetric(self, price_data):
        model = FactorRiskModel(price_data).fit()
        cov = model._factor_covariance.values
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_factor_covariance_positive_diagonal(self, price_data):
        model = FactorRiskModel(price_data).fit()
        assert (np.diag(model._factor_covariance.values) >= 0).all()

    def test_idio_variance_positive(self, price_data):
        model = FactorRiskModel(price_data).fit()
        assert (model._idio_variance >= 0).all()

    def test_fit_returns_self(self, price_data):
        model = FactorRiskModel(price_data)
        result = model.fit()
        assert result is model

    def test_with_market_caps(self, price_data):
        """Size factor should use market caps when provided."""
        caps = price_data * 1e6  # Proxy market caps
        model = FactorRiskModel(price_data, market_caps=caps).fit()
        assert model._fitted


# ---------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------


class TestDecomposeRisk:
    def test_decomposition_keys(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        risk = model.decompose_risk(equal_weights)
        assert "total_vol" in risk
        assert "systematic_vol" in risk
        assert "idio_vol" in risk
        assert "factor_contributions" in risk
        assert "factor_exposures" in risk

    def test_total_vol_positive(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        risk = model.decompose_risk(equal_weights)
        assert risk["total_vol"] > 0

    def test_systematic_plus_idio(self, price_data, equal_weights):
        """total_var = sys_var + idio_var."""
        model = FactorRiskModel(price_data).fit()
        risk = model.decompose_risk(equal_weights)
        total_var = risk["total_vol"] ** 2
        sys_var = risk["systematic_vol"] ** 2
        idio_var = risk["idio_vol"] ** 2
        assert total_var == pytest.approx(sys_var + idio_var, rel=0.01)

    def test_pct_systematic_bounded(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        risk = model.decompose_risk(equal_weights)
        assert 0 <= risk["pct_systematic"] <= 1

    def test_factor_contributions_have_all_factors(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        risk = model.decompose_risk(equal_weights)
        for factor in FactorRiskModel.FACTOR_NAMES:
            assert factor in risk["factor_contributions"]

    def test_not_fitted_raises(self, price_data, equal_weights):
        model = FactorRiskModel(price_data)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.decompose_risk(equal_weights)


# ---------------------------------------------------------------------------
# Portfolio factor exposure
# ---------------------------------------------------------------------------


class TestPortfolioExposure:
    def test_returns_series(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        exp = model.portfolio_factor_exposure(equal_weights)
        assert isinstance(exp, pd.Series)
        assert len(exp) == 4


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------


class TestStressTest:
    def test_known_scenario(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        result = model.stress_test(equal_weights, scenario="gfc_2008")
        assert "portfolio_return" in result
        assert result["portfolio_return"] < 0  # GFC should be negative

    def test_all_predefined_scenarios(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        for scenario in STRESS_SCENARIOS:
            result = model.stress_test(equal_weights, scenario=scenario)
            assert isinstance(result["portfolio_return"], float)

    def test_custom_shocks(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        result = model.stress_test(
            equal_weights,
            custom_shocks={"market": -0.10, "momentum": 0.05},
        )
        assert result["scenario"] == "custom"
        assert isinstance(result["portfolio_return"], float)

    def test_unknown_scenario_raises(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        with pytest.raises(ValueError, match="Unknown scenario"):
            model.stress_test(equal_weights, scenario="doesnt_exist")

    def test_no_scenario_or_shocks_raises(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        with pytest.raises(ValueError, match="Provide either"):
            model.stress_test(equal_weights)

    def test_stress_test_all(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        df = model.stress_test_all(equal_weights)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(STRESS_SCENARIOS)
        assert "portfolio_return" in df.columns


# ---------------------------------------------------------------------------
# Covariance / correlation matrix
# ---------------------------------------------------------------------------


class TestCovarianceMatrix:
    def test_shape(self, price_data):
        model = FactorRiskModel(price_data).fit()
        cov = model.covariance_matrix()
        assert cov.shape == (5, 5)

    def test_symmetric(self, price_data):
        model = FactorRiskModel(price_data).fit()
        cov = model.covariance_matrix().values
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_positive_diagonal(self, price_data):
        model = FactorRiskModel(price_data).fit()
        cov = model.covariance_matrix()
        assert (np.diag(cov.values) > 0).all()

    def test_correlation_diagonal_ones(self, price_data):
        model = FactorRiskModel(price_data).fit()
        corr = model.correlation_matrix()
        np.testing.assert_array_almost_equal(np.diag(corr.values), 1.0)

    def test_correlation_bounded(self, price_data):
        model = FactorRiskModel(price_data).fit()
        corr = model.correlation_matrix().values
        assert (corr >= -1.0 - 1e-6).all()
        assert (corr <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_json_without_weights(self, price_data):
        model = FactorRiskModel(price_data).fit()
        result = model.to_json()
        assert "n_assets" in result
        assert "n_factors" in result
        assert result["n_assets"] == 5
        assert result["n_factors"] == 4

    def test_to_json_with_weights(self, price_data, equal_weights):
        model = FactorRiskModel(price_data).fit()
        result = model.to_json(weights=equal_weights)
        assert "risk_decomposition" in result
        assert result["risk_decomposition"]["total_vol"] > 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class TestZscore:
    def test_zscore_normalizes(self):
        d = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = _zscore_dict(d)
        vals = list(result.values())
        assert abs(np.mean(vals)) < 1e-10
        assert abs(np.std(vals) - 1.0) < 1e-10

    def test_zscore_handles_nan(self):
        d = {"a": 1.0, "b": float("nan"), "c": 3.0}
        result = _zscore_dict(d)
        assert result["b"] == 0.0

    def test_zscore_constant_returns_zeros(self):
        d = {"a": 5.0, "b": 5.0, "c": 5.0}
        result = _zscore_dict(d)
        assert all(v == 0.0 for v in result.values())
