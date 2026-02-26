"""Risk attribution: decomposition, marginal contribution, and risk parity."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskAttribution:
    """
    Portfolio risk decomposition and attribution.

    Calculates how much each asset contributes to total portfolio risk
    and provides risk parity optimization.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[pd.Series] = None,
        ann_factor: int = 252,
    ):
        """
        Args:
            returns: DataFrame with asset returns (columns = assets)
            weights: Portfolio weights (default: equal weight)
            ann_factor: Annualization factor (default: 252 trading days)
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.ann_factor = ann_factor

        if weights is not None:
            self.weights = weights.reindex(self.tickers).fillna(0.0)
        else:
            self.weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)

        # Calculate covariance matrix
        self.cov_matrix = returns.cov() * ann_factor
        self.port_vol = self._portfolio_volatility()

    def _portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from weights and covariance."""
        port_var = self.weights @ self.cov_matrix @ self.weights
        return float(np.sqrt(port_var))

    def marginal_risk_contribution(self) -> pd.Series:
        """
        Calculate marginal contribution to risk for each asset.

        MRC_i = (cov_matrix @ weights)_i / portfolio_volatility

        Returns:
            Series with MRC for each asset
        """
        if self.port_vol == 0:
            return pd.Series(0.0, index=self.tickers)

        mrc = (self.cov_matrix @ self.weights) / self.port_vol
        return pd.Series(mrc, index=self.tickers)

    def component_risk(self) -> pd.Series:
        """
        Calculate component contribution to total risk.

        Component_i = weight_i × MRC_i

        Sum of all components equals total portfolio volatility.

        Returns:
            Series with component risk for each asset
        """
        mrc = self.marginal_risk_contribution()
        return self.weights * mrc

    def risk_contribution_pct(self) -> pd.Series:
        """
        Calculate percentage contribution to total risk.

        Returns:
            Series with percentage contribution for each asset
        """
        if self.port_vol == 0:
            return pd.Series(0.0, index=self.tickers)

        component = self.component_risk()
        return component / self.port_vol

    def risk_parity_weights(
        self,
        target_vol: Optional[float] = None,
        max_weight: float = 0.5,
        min_weight: float = 0.0,
        tolerance: float = 1e-10,
    ) -> pd.Series:
        """
        Calculate risk parity weights (equal risk contribution).

        Optimizes for equal risk contribution from all assets:
        weight_i × MRC_i ≈ portfolio_vol / n for all i

        Args:
            target_vol: Target portfolio volatility (optional)
            max_weight: Maximum weight for any asset
            min_weight: Minimum weight for any asset
            tolerance: Optimization tolerance

        Returns:
            Series with optimized risk parity weights
        """
        n = len(self.tickers)
        cov_matrix = self.cov_matrix.values

        def risk_budget_objective(weights: np.ndarray) -> float:
            """Objective: minimize sum of squared deviations from equal risk contribution."""
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return 0.0

            # Marginal risk contribution
            mrc = (cov_matrix @ weights) / port_vol
            # Component risk contribution
            rc = weights * mrc

            if target_vol is not None:
                # Target-specific risk contribution
                target_rc = target_vol / n
            else:
                # Equal risk contribution based on achieved vol
                target_rc = port_vol / n

            # Sum of squared deviations
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]

        # Bounds for each weight
        bounds = [(min_weight, max_weight) for _ in range(n)]

        # Initial guess: equal weights
        x0 = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(
            risk_budget_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": 1000},
        )

        if not result.success:
            logger.error(f"Risk parity optimization failed: {result.message}")
            logger.info("Falling back to equal weights")
            return pd.Series([1.0 / n] * n, index=self.tickers)

        return pd.Series(result.x, index=self.tickers)

    def diversification_ratio(self) -> float:
        """
        Calculate diversification ratio.

        DR = weighted_average_volatility / portfolio_volatility

        Higher is better (DR > 1 indicates diversification benefit).

        Returns:
            Diversification ratio
        """
        asset_vols = self.returns.std() * np.sqrt(self.ann_factor)
        weighted_vol = (self.weights * asset_vols).sum()

        if self.port_vol == 0:
            return 1.0

        return weighted_vol / self.port_vol

    def tracking_error(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error vs benchmark.

        Args:
            benchmark_returns: Benchmark return series

        Returns:
            Annualized tracking error
        """
        port_returns = (self.returns * self.weights).sum(axis=1)
        active_returns = port_returns - benchmark_returns
        return active_returns.std() * np.sqrt(self.ann_factor)

    def conditional_correlation(
        self,
        threshold_percentile: float = 10,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix during stress periods.

        Args:
            threshold_percentile: Percentile for stress threshold (default: 10%)

        Returns:
            Correlation matrix during stress periods
        """
        port_returns = (self.returns * self.weights).sum(axis=1)
        threshold = np.percentile(port_returns, threshold_percentile)
        stress_mask = port_returns <= threshold

        if stress_mask.sum() < 2:
            return self.returns.corr()

        return self.returns[stress_mask].corr()

    def risk_report(self) -> Dict:
        """
        Generate comprehensive risk attribution report.

        Returns:
            Dictionary with risk metrics and attribution
        """
        mrc = self.marginal_risk_contribution()
        component = self.component_risk()
        pct = self.risk_contribution_pct()

        # Sort by contribution
        top_contributors = pct.abs().sort_values(ascending=False)

        return {
            "portfolio_volatility": self.port_vol,
            "portfolio_volatility_pct": self.port_vol * 100,
            "diversification_ratio": self.diversification_ratio(),
            "marginal_risk_contribution": mrc.to_dict(),
            "component_risk": component.to_dict(),
            "risk_contribution_pct": pct.to_dict(),
            "top_risk_contributors": top_contributors.head(5).to_dict(),
            "concentration_risk": pct.std(),  # Standard deviation of contributions
        }

    def get_optimal_weights_comparison(
        self,
        methods: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Compare different weighting schemes.

        Args:
            methods: List of methods to compare

        Returns:
            DataFrame comparing weights across methods
        """
        if methods is None:
            methods = ["current", "equal", "risk_parity"]

        results = {}

        if "current" in methods:
            results["current"] = self.weights

        if "equal" in methods:
            n = len(self.tickers)
            results["equal"] = pd.Series(1.0 / n, index=self.tickers)

        if "risk_parity" in methods:
            try:
                results["risk_parity"] = self.risk_parity_weights()
            except RuntimeError as e:
                logger.error(f"Risk parity optimization failed: {e}")
                results["risk_parity"] = pd.Series(np.nan, index=self.tickers)

        return pd.DataFrame(results)


def calculate_risk_parity_allocation(
    returns: pd.DataFrame,
    max_weight: float = 0.5,
    min_weight: float = 0.0,
) -> pd.Series:
    """
    Convenience function to calculate risk parity weights.

    Args:
        returns: DataFrame with asset returns
        max_weight: Maximum weight for any asset
        min_weight: Minimum weight for any asset

    Returns:
        Series with risk parity weights
    """
    attribution = RiskAttribution(returns)
    return attribution.risk_parity_weights(
        max_weight=max_weight,
        min_weight=min_weight,
    )
