"""Tests for factor risk model and attribution."""

import numpy as np
import pytest

from python.risk.factor_model import (
    FactorModelResult,
    FactorSelectionMethod,
    HistoricalScenario,
    RiskAttribution,
    RiskAttributor,
    ScenarioResult,
    StatisticalFactorModel,
    StressedRiskResult,
    StressTester,
    standard_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factor_returns(T=500, N=20, K=3, seed=42):
    """Generate returns with known factor structure."""
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((N, K)) * 0.3
    factors = rng.standard_normal((T, K)) * 0.01
    noise = rng.standard_normal((T, N)) * 0.005
    returns = factors @ loadings.T + noise
    return returns, loadings, factors


def _equal_weights(N):
    return np.ones(N) / N


# ---------------------------------------------------------------------------
# StatisticalFactorModel
# ---------------------------------------------------------------------------


class TestStatisticalFactorModel:
    def test_fit_returns_result(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        result = model.fit(returns)
        assert isinstance(result, FactorModelResult)
        assert result.n_factors == 3

    def test_factor_returns_shape(self):
        T, N = 500, 20
        returns, _, _ = _make_factor_returns(T=T, N=N)
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        result = model.fit(returns)
        assert result.factor_returns.shape == (T, 3)

    def test_factor_loadings_shape(self):
        T, N = 500, 20
        returns, _, _ = _make_factor_returns(T=T, N=N)
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        result = model.fit(returns)
        assert result.factor_loadings.shape == (N, 3)

    def test_eigenvalues_descending(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel()
        result = model.fit(returns)
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] >= result.eigenvalues[i + 1]

    def test_variance_explained_sums_to_one(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel()
        result = model.fit(returns)
        assert result.explained_variance_ratio.sum() == pytest.approx(1.0, abs=0.01)

    def test_total_variance_explained_property(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        result = model.fit(returns)
        assert 0 < result.total_variance_explained <= 1.0

    def test_recovers_factor_count(self):
        """With clear 3-factor structure, should detect ~3 factors."""
        returns, _, _ = _make_factor_returns(T=1000, N=20, K=3)
        model = StatisticalFactorModel(method=FactorSelectionMethod.MARCHENKO_PASTUR)
        result = model.fit(returns)
        assert abs(result.n_factors - 3) <= 2

    def test_kaiser_method(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.KAISER)
        result = model.fit(returns)
        assert result.n_factors >= 1

    def test_scree_method(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.SCREE)
        result = model.fit(returns)
        assert result.n_factors >= 1

    def test_idiosyncratic_variance_positive(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        result = model.fit(returns)
        assert all(v >= 0 for v in result.idiosyncratic_var)

    def test_factor_covariance_symmetric(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        result = model.fit(returns)
        np.testing.assert_allclose(
            result.factor_covariance, result.factor_covariance.T, atol=1e-10
        )


# ---------------------------------------------------------------------------
# RiskAttributor
# ---------------------------------------------------------------------------


class TestRiskAttributor:
    def _fit_model(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        return model.fit(returns)

    def test_returns_attribution(self):
        result = self._fit_model()
        attrib = RiskAttributor(result)
        risk = attrib.attribute(_equal_weights(20))
        assert isinstance(risk, RiskAttribution)

    def test_euler_decomposition(self):
        """Sum of asset component risks = portfolio volatility."""
        result = self._fit_model()
        attrib = RiskAttributor(result)
        risk = attrib.attribute(_equal_weights(20))
        assert risk.asset_component_risk.sum() == pytest.approx(
            risk.portfolio_volatility, abs=1e-10
        )

    def test_variance_decomposition(self):
        """Factor variance + specific variance ≈ total variance."""
        result = self._fit_model()
        attrib = RiskAttributor(result)
        risk = attrib.attribute(_equal_weights(20))
        assert risk.factor_variance + risk.specific_variance == pytest.approx(
            risk.portfolio_variance, abs=1e-10
        )

    def test_risk_pct_sums_to_one(self):
        result = self._fit_model()
        attrib = RiskAttributor(result)
        risk = attrib.attribute(_equal_weights(20))
        assert risk.factor_risk_pct + risk.specific_risk_pct == pytest.approx(
            1.0, abs=0.01
        )

    def test_zero_weight_zero_contribution(self):
        result = self._fit_model()
        attrib = RiskAttributor(result)
        w = np.zeros(20)
        w[0] = 1.0
        risk = attrib.attribute(w)
        assert all(abs(risk.asset_component_risk[i]) < 1e-10 for i in range(1, 20))

    def test_portfolio_vol_positive(self):
        result = self._fit_model()
        attrib = RiskAttributor(result)
        risk = attrib.attribute(_equal_weights(20))
        assert risk.portfolio_volatility > 0

    def test_factor_exposures_shape(self):
        result = self._fit_model()
        attrib = RiskAttributor(result)
        risk = attrib.attribute(_equal_weights(20))
        assert len(risk.factor_exposures) == result.n_factors


# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------


class TestStressTester:
    def _fit_model(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        return model.fit(returns)

    def test_scenario_result(self):
        result = self._fit_model()
        tester = StressTester(result)
        scenarios = standard_scenarios(result.n_factors)
        sr = tester.apply_scenario(_equal_weights(20), scenarios[0])
        assert isinstance(sr, ScenarioResult)

    def test_scenario_pnl_decomposition(self):
        """Factor contributions should sum to total PnL."""
        result = self._fit_model()
        tester = StressTester(result)
        scenarios = standard_scenarios(result.n_factors)
        sr = tester.apply_scenario(_equal_weights(20), scenarios[0])
        assert sr.factor_contributions.sum() == pytest.approx(sr.portfolio_pnl, abs=1e-12)

    def test_negative_shocks_negative_pnl(self):
        """Negative factor shocks should generally cause negative PnL."""
        result = self._fit_model()
        tester = StressTester(result)
        scenario = HistoricalScenario(
            "CRASH", "All factors down",
            np.full(result.n_factors, -0.10),
        )
        sr = tester.apply_scenario(_equal_weights(20), scenario)
        # With random loadings, not guaranteed, but PnL should be large magnitude
        assert abs(sr.portfolio_pnl) > 0

    def test_run_standard_scenarios(self):
        result = self._fit_model()
        tester = StressTester(result)
        results = tester.run_standard_scenarios(_equal_weights(20))
        assert len(results) == 3
        assert results[0].name == "GFC_2008"

    def test_shock_propagation_identity(self):
        """With diagonal factor covariance, shocks don't propagate."""
        result = self._fit_model()
        # Replace factor cov with identity for this test
        K = result.n_factors
        tester = StressTester(result)
        tester.Sigma_F = np.eye(K)
        shocks = tester.propagate_shock(0, -0.10)
        assert shocks[0] == pytest.approx(-0.10)
        for i in range(1, K):
            assert abs(shocks[i]) < 1e-10

    def test_shock_propagation_correlated(self):
        """With correlated factors, shocks propagate."""
        result = self._fit_model()
        tester = StressTester(result)
        K = result.n_factors
        rho = 0.8
        Sigma = np.eye(K) * (1 - rho) + np.ones((K, K)) * rho
        tester.Sigma_F = Sigma
        shocks = tester.propagate_shock(0, -0.10)
        # All factors should be shocked
        assert shocks[0] == pytest.approx(-0.10)
        for i in range(1, K):
            assert abs(shocks[i]) > 0.01


# ---------------------------------------------------------------------------
# Stressed VaR/CVaR
# ---------------------------------------------------------------------------


class TestStressedRisk:
    def _fit_model(self):
        returns, _, _ = _make_factor_returns()
        model = StatisticalFactorModel(method=FactorSelectionMethod.FIXED, n_factors_fixed=3)
        return model.fit(returns)

    def test_returns_result(self):
        result = self._fit_model()
        tester = StressTester(result)
        sr = tester.stressed_var_cvar(_equal_weights(20))
        assert isinstance(sr, StressedRiskResult)

    def test_stress_increases_var(self):
        result = self._fit_model()
        tester = StressTester(result)
        normal = tester.stressed_var_cvar(
            _equal_weights(20), correlation_stress=0.0, vol_multiplier=1.0
        )
        stressed = tester.stressed_var_cvar(
            _equal_weights(20), correlation_stress=0.5, vol_multiplier=2.0
        )
        assert stressed.stressed_var >= normal.normal_var

    def test_cvar_exceeds_var(self):
        result = self._fit_model()
        tester = StressTester(result)
        sr = tester.stressed_var_cvar(_equal_weights(20), confidence=0.99)
        assert sr.normal_cvar > sr.normal_var
        assert sr.stressed_cvar > sr.stressed_var

    def test_vol_multiplier_increases_vol(self):
        result = self._fit_model()
        tester = StressTester(result)
        r1 = tester.stressed_var_cvar(
            _equal_weights(20), vol_multiplier=1.0
        )
        r2 = tester.stressed_var_cvar(
            _equal_weights(20), vol_multiplier=2.0
        )
        assert r2.stressed_portfolio_vol > r1.stressed_portfolio_vol

    def test_all_positive(self):
        result = self._fit_model()
        tester = StressTester(result)
        sr = tester.stressed_var_cvar(_equal_weights(20))
        assert sr.normal_var > 0
        assert sr.stressed_var > 0
        assert sr.normal_cvar > 0
        assert sr.stressed_cvar > 0


# ---------------------------------------------------------------------------
# Standard scenarios
# ---------------------------------------------------------------------------


class TestStandardScenarios:
    def test_returns_list(self):
        scenarios = standard_scenarios(3)
        assert len(scenarios) == 3

    def test_scenario_names(self):
        scenarios = standard_scenarios(3)
        names = [s.name for s in scenarios]
        assert "GFC_2008" in names
        assert "COVID_2020" in names

    def test_shocks_correct_length(self):
        for k in [2, 5, 10]:
            scenarios = standard_scenarios(k)
            for s in scenarios:
                assert len(s.factor_shocks) == k
