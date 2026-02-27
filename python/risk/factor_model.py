"""Statistical factor risk model and attribution.

Implements PCA-based factor extraction, risk decomposition, and stress testing:
  1. PCA factor extraction with Marchenko-Pastur, Kaiser, or scree selection.
  2. Euler risk decomposition into factor + specific components.
  3. Marginal / component / percentage risk contributions.
  4. Historical scenario replay and hypothetical factor shock propagation.
  5. Stressed VaR/CVaR under elevated correlation/volatility regimes.

Usage::

    model = StatisticalFactorModel()
    result = model.fit(returns_matrix)  # T x N

    attrib = RiskAttributor(result)
    risk = attrib.attribute(weights)

    tester = StressTester(result)
    scenario_results = tester.run_standard_scenarios(weights)

References:
  - Menchero et al. (2011), "Barra Risk Model Handbook"
  - Kritzman et al. (2011), "Principal Components as a Measure of Systemic Risk"
  - Marchenko & Pastur (1967), "Distribution of Eigenvalues"
  - Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor selection methods
# ---------------------------------------------------------------------------


class FactorSelectionMethod(Enum):
    MARCHENKO_PASTUR = auto()
    KAISER = auto()
    SCREE = auto()
    FIXED = auto()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FactorModelResult:
    """Result from PCA factor model fitting."""

    factor_returns: np.ndarray  # (T, K)
    factor_loadings: np.ndarray  # (N, K)
    eigenvalues: np.ndarray  # (min(T,N),) sorted descending
    explained_variance_ratio: np.ndarray  # (min(T,N),)
    n_factors: int
    idiosyncratic_var: np.ndarray  # (N,) per-asset residual variance
    factor_covariance: np.ndarray  # (K, K)

    @property
    def total_variance_explained(self) -> float:
        return float(np.sum(self.explained_variance_ratio[: self.n_factors]))


@dataclass
class RiskAttribution:
    """Risk decomposition for a portfolio."""

    portfolio_volatility: float
    portfolio_variance: float

    # Factor vs specific
    factor_variance: float
    specific_variance: float
    factor_risk_pct: float
    specific_risk_pct: float

    # Factor-level
    factor_exposures: np.ndarray  # (K,)
    factor_marginal_risk: np.ndarray  # (K,)
    factor_component_risk: np.ndarray  # (K,)

    # Asset-level
    asset_marginal_risk: np.ndarray  # (N,)
    asset_component_risk: np.ndarray  # (N,)


@dataclass
class ScenarioResult:
    """Result from a stress scenario."""

    name: str
    portfolio_pnl: float
    factor_contributions: np.ndarray  # (K,)


@dataclass
class StressedRiskResult:
    """VaR/CVaR under normal and stressed conditions."""

    normal_var: float
    stressed_var: float
    normal_cvar: float
    stressed_cvar: float
    stressed_portfolio_vol: float


# ---------------------------------------------------------------------------
# Statistical Factor Model
# ---------------------------------------------------------------------------


class StatisticalFactorModel:
    """PCA-based factor model for asset returns.

    Parameters
    ----------
    method : FactorSelectionMethod
        How to select the number of factors.
    n_factors_fixed : int
        Number of factors if method is FIXED.
    min_factors : int
        Floor on factor count.
    max_factors : int
        Cap on factor count.
    """

    def __init__(
        self,
        method: FactorSelectionMethod = FactorSelectionMethod.MARCHENKO_PASTUR,
        n_factors_fixed: int = 5,
        min_factors: int = 1,
        max_factors: int = 20,
    ):
        self.method = method
        self.n_factors_fixed = n_factors_fixed
        self.min_factors = min_factors
        self.max_factors = max_factors

    def fit(self, returns: np.ndarray) -> FactorModelResult:
        """Fit factor model via eigendecomposition.

        Parameters
        ----------
        returns : np.ndarray
            (T, N) matrix of asset returns.

        Returns
        -------
        FactorModelResult
        """
        T, N = returns.shape
        logger.info(f"Fitting PCA factor model: T={T}, N={N}")

        # Demean
        X = returns - returns.mean(axis=0, keepdims=True)

        # Covariance matrix
        cov = (X.T @ X) / (T - 1)

        # Eigendecomposition (ascending order from eigh)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 0.0)
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        explained = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)

        # Select number of factors
        n_factors = self._select_n_factors(eigenvalues, T, N)
        n_factors = int(np.clip(n_factors, self.min_factors, min(self.max_factors, min(T, N))))

        logger.info(
            f"Selected {n_factors} factors, "
            f"variance explained: {explained[:n_factors].sum():.2%}"
        )

        # Extract loadings and factor returns
        B = eigenvectors[:, :n_factors]  # (N, K)
        F = X @ B  # (T, K)

        # Residuals and idiosyncratic variance
        residuals = X - F @ B.T
        idio_var = np.var(residuals, axis=0, ddof=1)

        # Factor covariance
        factor_cov = np.cov(F, rowvar=False, ddof=1)
        if factor_cov.ndim == 0:
            factor_cov = factor_cov.reshape(1, 1)

        return FactorModelResult(
            factor_returns=F,
            factor_loadings=B,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained,
            n_factors=n_factors,
            idiosyncratic_var=idio_var,
            factor_covariance=factor_cov,
        )

    def _select_n_factors(
        self, eigenvalues: np.ndarray, T: int, N: int
    ) -> int:
        if self.method == FactorSelectionMethod.FIXED:
            return self.n_factors_fixed

        if self.method == FactorSelectionMethod.KAISER:
            threshold = np.mean(eigenvalues)
            return max(int(np.sum(eigenvalues > threshold)), 1)

        if self.method == FactorSelectionMethod.SCREE:
            if len(eigenvalues) < 3:
                return 1
            diffs = np.diff(eigenvalues)
            second_diffs = np.diff(diffs)
            return max(int(np.argmax(second_diffs)) + 1, 1)

        if self.method == FactorSelectionMethod.MARCHENKO_PASTUR:
            q = N / T
            sigma2 = float(np.median(eigenvalues))
            mp_upper = sigma2 * (1 + np.sqrt(q)) ** 2
            n = int(np.sum(eigenvalues > mp_upper))
            logger.info(
                f"Marchenko-Pastur: q={q:.3f}, mp_upper={mp_upper:.6f}, "
                f"n_signal={n}"
            )
            return max(n, 1)

        return self.n_factors_fixed


# ---------------------------------------------------------------------------
# Risk Attribution
# ---------------------------------------------------------------------------


class RiskAttributor:
    """Decompose portfolio risk into factor and specific components.

    Uses Euler decomposition: sum of component contributions = total vol.

    Parameters
    ----------
    model_result : FactorModelResult
        Fitted factor model.
    """

    def __init__(self, model_result: FactorModelResult):
        self.B = model_result.factor_loadings  # (N, K)
        self.Sigma_F = model_result.factor_covariance  # (K, K)
        self.D = np.diag(model_result.idiosyncratic_var)  # (N, N)
        self.N, self.K = self.B.shape

        # Full covariance: B @ Sigma_F @ B' + D
        self.Sigma = self.B @ self.Sigma_F @ self.B.T + self.D

    def attribute(self, weights: np.ndarray) -> RiskAttribution:
        """Compute full risk attribution.

        Parameters
        ----------
        weights : np.ndarray
            (N,) portfolio weight vector.

        Returns
        -------
        RiskAttribution
        """
        w = np.asarray(weights, dtype=float).ravel()

        port_var = float(w @ self.Sigma @ w)
        port_vol = np.sqrt(max(port_var, 0.0))

        # Factor exposures: h = B' @ w
        h = self.B.T @ w

        # Factor variance and specific variance
        factor_var = float(h @ self.Sigma_F @ h)
        specific_var = float(w @ self.D @ w)

        factor_pct = factor_var / port_var if port_var > 1e-16 else 0.0
        specific_pct = specific_var / port_var if port_var > 1e-16 else 0.0

        # Factor marginal and component risk
        if port_vol > 1e-16:
            factor_mctr = (self.Sigma_F @ h) / port_vol
            asset_mctr = (self.Sigma @ w) / port_vol
        else:
            factor_mctr = np.zeros(self.K)
            asset_mctr = np.zeros(self.N)

        factor_cctr = h * factor_mctr
        asset_cctr = w * asset_mctr

        return RiskAttribution(
            portfolio_volatility=port_vol,
            portfolio_variance=port_var,
            factor_variance=factor_var,
            specific_variance=specific_var,
            factor_risk_pct=factor_pct,
            specific_risk_pct=specific_pct,
            factor_exposures=h,
            factor_marginal_risk=factor_mctr,
            factor_component_risk=factor_cctr,
            asset_marginal_risk=asset_mctr,
            asset_component_risk=asset_cctr,
        )


# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------


@dataclass
class HistoricalScenario:
    """A named stress scenario with factor shocks."""

    name: str
    description: str
    factor_shocks: np.ndarray  # (K,) factor return vector


def standard_scenarios(n_factors: int) -> list[HistoricalScenario]:
    """Built-in stress scenarios."""
    scenarios = []

    shocks = np.zeros(n_factors)
    if n_factors >= 1:
        shocks[0] = -0.20
    if n_factors >= 2:
        shocks[1] = -0.15
    scenarios.append(HistoricalScenario(
        "GFC_2008", "Global Financial Crisis", shocks.copy(),
    ))

    shocks = np.zeros(n_factors)
    if n_factors >= 1:
        shocks[0] = -0.15
    if n_factors >= 2:
        shocks[1] = -0.10
    scenarios.append(HistoricalScenario(
        "COVID_2020", "COVID-19 market crash", shocks.copy(),
    ))

    shocks = np.zeros(n_factors)
    if n_factors >= 1:
        shocks[0] = -0.05
    if n_factors >= 2:
        shocks[1] = -0.25
    scenarios.append(HistoricalScenario(
        "QUANT_QUAKE_2007", "Quant factor unwind", shocks.copy(),
    ))

    return scenarios


class StressTester:
    """Stress test portfolios with scenarios and stressed risk measures.

    Parameters
    ----------
    model_result : FactorModelResult
        Fitted factor model.
    """

    def __init__(self, model_result: FactorModelResult):
        self.B = model_result.factor_loadings
        self.Sigma_F = model_result.factor_covariance
        self.D = np.diag(model_result.idiosyncratic_var)
        self.N, self.K = self.B.shape
        self.Sigma = self.B @ self.Sigma_F @ self.B.T + self.D

    def apply_scenario(
        self, weights: np.ndarray, scenario: HistoricalScenario
    ) -> ScenarioResult:
        """Apply a stress scenario to the portfolio."""
        w = np.asarray(weights, dtype=float).ravel()
        shocks = scenario.factor_shocks[: self.K]
        if len(shocks) < self.K:
            shocks = np.pad(shocks, (0, self.K - len(shocks)))

        h = self.B.T @ w
        factor_pnl = h * shocks

        return ScenarioResult(
            name=scenario.name,
            portfolio_pnl=float(factor_pnl.sum()),
            factor_contributions=factor_pnl,
        )

    def propagate_shock(
        self, factor_idx: int, magnitude: float
    ) -> np.ndarray:
        """Propagate a single-factor shock via conditional expectation.

        Returns (K,) vector of correlated shocks.
        """
        var_k = self.Sigma_F[factor_idx, factor_idx]
        if var_k < 1e-16:
            shocks = np.zeros(self.K)
            shocks[factor_idx] = magnitude
            return shocks
        return self.Sigma_F[:, factor_idx] / var_k * magnitude

    def run_standard_scenarios(
        self, weights: np.ndarray
    ) -> list[ScenarioResult]:
        """Run all built-in historical stress scenarios."""
        scenarios = standard_scenarios(self.K)
        return [self.apply_scenario(weights, s) for s in scenarios]

    def stressed_var_cvar(
        self,
        weights: np.ndarray,
        confidence: float = 0.99,
        correlation_stress: float = 0.0,
        vol_multiplier: float = 1.0,
    ) -> StressedRiskResult:
        """Compute VaR/CVaR under stressed conditions.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights.
        confidence : float
            VaR confidence level.
        correlation_stress : float
            Blend factor s in [0, 1] toward perfect correlation.
        vol_multiplier : float
            Multiply all volatilities by this factor.
        """
        w = np.asarray(weights, dtype=float).ravel()
        s = np.clip(correlation_stress, 0.0, 1.0)
        vm = max(vol_multiplier, 1.0)

        # Normal regime
        port_var = float(w @ self.Sigma @ w)
        port_vol = np.sqrt(max(port_var, 0.0))

        z = sp_stats.norm.ppf(confidence)
        phi_z = sp_stats.norm.pdf(z)

        var_normal = z * port_vol
        cvar_normal = port_vol * phi_z / (1 - confidence)

        # Stressed regime
        factor_vols = np.sqrt(np.diag(self.Sigma_F))
        factor_vols_stressed = factor_vols * vm

        D_inv = np.diag(1.0 / np.maximum(factor_vols, 1e-16))
        C_factor = D_inv @ self.Sigma_F @ D_inv

        # Blend toward perfect correlation
        ones_mat = np.ones_like(C_factor)
        C_stressed = (1 - s) * C_factor + s * ones_mat

        # Ensure valid correlation matrix
        eigvals, eigvecs = np.linalg.eigh(C_stressed)
        eigvals = np.maximum(eigvals, 1e-8)
        C_stressed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(C_stressed))
        C_stressed = C_stressed / np.outer(d, d)

        D_stressed = np.diag(factor_vols_stressed)
        Sigma_F_stressed = D_stressed @ C_stressed @ D_stressed

        D_idio_stressed = self.D * (vm**2)
        Sigma_stressed = self.B @ Sigma_F_stressed @ self.B.T + D_idio_stressed

        port_var_stressed = float(w @ Sigma_stressed @ w)
        port_vol_stressed = np.sqrt(max(port_var_stressed, 0.0))

        var_stressed = z * port_vol_stressed
        cvar_stressed = port_vol_stressed * phi_z / (1 - confidence)

        return StressedRiskResult(
            normal_var=var_normal,
            stressed_var=var_stressed,
            normal_cvar=cvar_normal,
            stressed_cvar=cvar_stressed,
            stressed_portfolio_vol=port_vol_stressed,
        )
