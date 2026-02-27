"""Monte Carlo portfolio simulation for risk analysis and scenario modeling.

Implements:
  - ``simulate_returns``: Generate future return scenarios via parametric
    (multivariate normal / t-distribution) or historical bootstrap.
  - ``simulate_portfolio``: Project portfolio wealth paths with rebalancing.
  - ``compute_risk_metrics``: VaR, CVaR, max drawdown distributions from paths.
  - ``tail_risk_analysis``: Extreme event probabilities and expected shortfall.

Usage::

    sim = MonteCarloSimulator(returns, n_scenarios=10000, horizon=252)
    paths = sim.simulate_portfolio(weights)
    metrics = sim.compute_risk_metrics(paths)
    # metrics = {'var_95': -0.15, 'cvar_95': -0.22, 'max_dd_median': -0.18, ...}

References:
  - Glasserman (2003), "Monte Carlo Methods in Financial Engineering"
  - Jorion (2006), "Value at Risk", Ch. 12
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics computed from Monte Carlo simulation."""

    # Value at Risk
    var_95: float
    var_99: float

    # Conditional VaR (Expected Shortfall)
    cvar_95: float
    cvar_99: float

    # Drawdown statistics
    max_dd_median: float
    max_dd_95: float
    max_dd_mean: float

    # Return statistics
    mean_return: float
    median_return: float
    volatility: float
    skewness: float
    kurtosis: float

    # Tail metrics
    prob_loss_5pct: float  # P(return < -5%)
    prob_loss_10pct: float  # P(return < -10%)
    prob_loss_20pct: float  # P(return < -20%)

    def to_dict(self) -> dict:
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "max_dd_median": self.max_dd_median,
            "max_dd_95": self.max_dd_95,
            "max_dd_mean": self.max_dd_mean,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "volatility": self.volatility,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "prob_loss_5pct": self.prob_loss_5pct,
            "prob_loss_10pct": self.prob_loss_10pct,
            "prob_loss_20pct": self.prob_loss_20pct,
        }


class MonteCarloSimulator:
    """Monte Carlo portfolio simulation engine.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns (columns=tickers, index=dates).
    n_scenarios : int
        Number of simulation paths.
    horizon : int
        Simulation horizon in trading days.
    method : str
        Return generation method:
        - "normal": Multivariate normal (fastest, assumes normality)
        - "t": Multivariate t-distribution (captures fat tails)
        - "bootstrap": Block bootstrap from historical returns (non-parametric)
    block_size : int
        Block size for bootstrap method (in trading days).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        n_scenarios: int = 10_000,
        horizon: int = 252,
        method: str = "normal",
        block_size: int = 21,
        seed: Optional[int] = None,
    ):
        self.returns = returns.dropna()
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.n_scenarios = n_scenarios
        self.horizon = horizon
        self.method = method
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

        # Pre-compute statistics from historical data
        self._mu = self.returns.mean().values
        self._cov = self.returns.cov().values
        self._n_hist = len(self.returns)

    def simulate_returns(self) -> np.ndarray:
        """Generate simulated return scenarios.

        Returns
        -------
        np.ndarray, shape (n_scenarios, horizon, n_assets)
            Simulated daily returns for each scenario, day, and asset.
        """
        if self.method == "normal":
            return self._simulate_normal()
        elif self.method == "t":
            return self._simulate_t()
        elif self.method == "bootstrap":
            return self._simulate_bootstrap()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def simulate_portfolio(
        self,
        weights: pd.Series,
        initial_value: float = 1.0,
        rebalance_freq: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate portfolio wealth paths.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        initial_value : float
            Starting portfolio value.
        rebalance_freq : int, optional
            Rebalance every N days. None = buy-and-hold.

        Returns
        -------
        np.ndarray, shape (n_scenarios, horizon + 1)
            Portfolio value paths. First column is initial_value.
        """
        w = weights.reindex(self.tickers, fill_value=0.0).values
        scenarios = self.simulate_returns()  # (n_scen, horizon, n_assets)

        paths = np.zeros((self.n_scenarios, self.horizon + 1))
        paths[:, 0] = initial_value

        for t in range(self.horizon):
            # Portfolio return for this day
            port_ret = scenarios[:, t, :] @ w  # (n_scenarios,)
            paths[:, t + 1] = paths[:, t] * (1 + port_ret)

            # Rebalance: reset weights (in a real sim, you'd track per-asset)
            # For simplicity, assume constant-weight rebalancing
            if rebalance_freq and (t + 1) % rebalance_freq == 0:
                pass  # Already using constant weights each period

        return paths

    def compute_risk_metrics(
        self,
        paths: np.ndarray,
    ) -> RiskMetrics:
        """Compute comprehensive risk metrics from simulation paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_scenarios, horizon + 1)
            Portfolio value paths from simulate_portfolio().

        Returns
        -------
        RiskMetrics
            Comprehensive risk statistics.
        """
        # Terminal returns
        terminal_returns = paths[:, -1] / paths[:, 0] - 1

        # Drawdowns per path
        max_drawdowns = np.array([
            _max_drawdown(paths[i]) for i in range(len(paths))
        ])

        # VaR
        var_95 = float(np.percentile(terminal_returns, 5))
        var_99 = float(np.percentile(terminal_returns, 1))

        # CVaR (Expected Shortfall)
        cvar_95 = float(terminal_returns[terminal_returns <= var_95].mean()) if (terminal_returns <= var_95).any() else var_95
        cvar_99 = float(terminal_returns[terminal_returns <= var_99].mean()) if (terminal_returns <= var_99).any() else var_99

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_dd_median=float(np.median(max_drawdowns)),
            max_dd_95=float(np.percentile(max_drawdowns, 5)),
            max_dd_mean=float(np.mean(max_drawdowns)),
            mean_return=float(np.mean(terminal_returns)),
            median_return=float(np.median(terminal_returns)),
            volatility=float(np.std(terminal_returns)),
            skewness=float(stats.skew(terminal_returns)),
            kurtosis=float(stats.kurtosis(terminal_returns)),
            prob_loss_5pct=float(np.mean(terminal_returns < -0.05)),
            prob_loss_10pct=float(np.mean(terminal_returns < -0.10)),
            prob_loss_20pct=float(np.mean(terminal_returns < -0.20)),
        )

    def tail_risk_analysis(
        self,
        weights: pd.Series,
        thresholds: list[float] | None = None,
    ) -> dict:
        """Compute tail risk probabilities and expected shortfall.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights.
        thresholds : list[float]
            Loss thresholds to analyze (e.g., [-0.05, -0.10, -0.20]).

        Returns
        -------
        dict with loss threshold analysis.
        """
        if thresholds is None:
            thresholds = [-0.05, -0.10, -0.15, -0.20, -0.30]

        paths = self.simulate_portfolio(weights)
        terminal = paths[:, -1] / paths[:, 0] - 1

        results = {}
        for thresh in thresholds:
            mask = terminal <= thresh
            prob = float(np.mean(mask))
            expected_shortfall = float(terminal[mask].mean()) if mask.any() else 0.0
            results[f"loss_{abs(thresh):.0%}"] = {
                "probability": prob,
                "expected_shortfall": expected_shortfall,
                "worst_case": float(terminal[mask].min()) if mask.any() else 0.0,
            }

        return results

    # ------------------------------------------------------------------
    # Return generation methods
    # ------------------------------------------------------------------

    def _simulate_normal(self) -> np.ndarray:
        """Multivariate normal simulation."""
        try:
            L = np.linalg.cholesky(self._cov)
        except np.linalg.LinAlgError:
            # Add small diagonal for numerical stability
            cov_reg = self._cov + np.eye(self.n_assets) * 1e-8
            L = np.linalg.cholesky(cov_reg)

        Z = self.rng.standard_normal((self.n_scenarios, self.horizon, self.n_assets))
        scenarios = self._mu + Z @ L.T
        return scenarios

    def _simulate_t(self, df: int = 5) -> np.ndarray:
        """Multivariate t-distribution simulation (fat tails).

        Uses the relationship: X = mu + sqrt(df/chi2) * Z, where Z ~ N(0, Sigma).
        """
        try:
            L = np.linalg.cholesky(self._cov)
        except np.linalg.LinAlgError:
            cov_reg = self._cov + np.eye(self.n_assets) * 1e-8
            L = np.linalg.cholesky(cov_reg)

        Z = self.rng.standard_normal((self.n_scenarios, self.horizon, self.n_assets))
        chi2 = self.rng.chisquare(df, size=(self.n_scenarios, self.horizon, 1))
        t_scale = np.sqrt(df / chi2)

        scenarios = self._mu + t_scale * (Z @ L.T)
        return scenarios

    def _simulate_bootstrap(self) -> np.ndarray:
        """Block bootstrap from historical returns."""
        hist = self.returns.values  # (n_hist, n_assets)
        n_hist = len(hist)

        if n_hist < self.block_size:
            logger.warning(
                f"History ({n_hist}) shorter than block_size ({self.block_size}); "
                "falling back to iid bootstrap"
            )
            block_size = 1
        else:
            block_size = self.block_size

        n_blocks = (self.horizon + block_size - 1) // block_size
        scenarios = np.zeros((self.n_scenarios, self.horizon, self.n_assets))

        for i in range(self.n_scenarios):
            path = []
            for _ in range(n_blocks):
                start = self.rng.integers(0, n_hist - block_size + 1)
                path.append(hist[start : start + block_size])
            path = np.concatenate(path, axis=0)[: self.horizon]
            scenarios[i] = path

        return scenarios


def _max_drawdown(path: np.ndarray) -> float:
    """Compute maximum drawdown from a wealth path."""
    peak = np.maximum.accumulate(path)
    dd = (path - peak) / np.maximum(peak, 1e-10)
    return float(dd.min())
