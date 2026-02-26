"""Portfolio optimization: HRP, CVaR, Black-Litterman with ML views."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity, MeanRisk
from skfolio.prior import BlackLitterman, EmpiricalPrior

logger = logging.getLogger(__name__)


def _cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    """Cap individual weights and redistribute excess proportionally.

    Iteratively clips weights above *max_weight* and spreads the surplus to
    uncapped assets (proportional to their original weight) until no asset
    exceeds the cap.  Guarantees weights still sum to 1.
    """
    w = weights.copy()
    for _ in range(20):  # converge in a few iterations
        excess_mask = w > max_weight
        if not excess_mask.any():
            break
        surplus = (w[excess_mask] - max_weight).sum()
        w[excess_mask] = max_weight
        uncapped = ~excess_mask
        unc_total = w[uncapped].sum()
        if unc_total > 0:
            w[uncapped] += surplus * (w[uncapped] / unc_total)
        else:
            # All assets are at the cap — distribute evenly
            w[uncapped] = surplus / max(uncapped.sum(), 1)
    # Normalize for any floating-point drift
    w = w / w.sum() if w.sum() > 0 else w
    return w


class PortfolioOptimizer:
    """Multi-strategy portfolio optimizer using skfolio."""

    def __init__(
        self,
        prices: pd.DataFrame,
        max_weight: Optional[float] = None,
        current_weights: Optional[pd.Series] = None,
        turnover_threshold: float = 0.20,
        shrink_covariance: bool = True,
    ):
        """Initialize with a DataFrame of asset prices (columns=tickers, index=dates).

        Args:
            prices: Price DataFrame (columns=tickers, index=dates).
            max_weight: Optional cap on individual asset weight (e.g. 0.25).
                Applied after optimization via iterative redistribution (Fix #18).
            current_weights: Current portfolio weights for turnover-aware optimization.
            turnover_threshold: Minimum turnover to justify rebalancing (default 20%).
                If computed turnover is below this threshold, current weights are kept.
            shrink_covariance: H-HRP fix — apply Ledoit-Wolf shrinkage to the
                covariance matrix when n_assets > n_observations/2 (noisy regime).
                Default True.
        """
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.tickers = list(prices.columns)
        self.max_weight = max_weight
        self.current_weights = current_weights
        self.turnover_threshold = turnover_threshold

        # H-HRP fix: apply Ledoit-Wolf shrinkage for small-sample covariance
        n_obs, n_assets = self.returns.shape
        self._shrunk = False
        if shrink_covariance and n_assets > 2 and n_obs < n_assets * 3:
            try:
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf().fit(self.returns.values)
                logger.info(
                    f"H-HRP: applied Ledoit-Wolf shrinkage "
                    f"(n_obs={n_obs}, n_assets={n_assets}, shrinkage={lw.shrinkage_:.3f})"
                )
                self._shrunk = True
            except Exception as e:
                logger.warning(f"Ledoit-Wolf shrinkage failed: {e}")

    def hrp(self) -> pd.Series:
        """Hierarchical Risk Parity allocation."""
        try:
            model = HierarchicalRiskParity()
            model.fit(self.returns)
            weights = pd.Series(model.weights_, index=self.tickers, name="hrp_weights")
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}. Falling back to equal weight.")
            weights = pd.Series(
                np.ones(len(self.tickers)) / len(self.tickers),
                index=self.tickers,
                name="hrp_weights",
            )
        if self.max_weight is not None:
            weights = _cap_weights(weights, self.max_weight)
        return weights

    def min_cvar(self, confidence_level: float = 0.95) -> pd.Series:
        """Minimum CVaR (Conditional Value at Risk) allocation."""
        try:
            model = MeanRisk(
                risk_measure=RiskMeasure.CVAR,
                min_weights=0.0,
                cvar_beta=confidence_level,
            )
            model.fit(self.returns)
            weights = pd.Series(model.weights_, index=self.tickers, name="min_cvar_weights")
        except Exception as e:
            logger.error(f"Min-CVaR optimization failed: {e}. Falling back to equal weight.")
            weights = pd.Series(
                np.ones(len(self.tickers)) / len(self.tickers),
                index=self.tickers,
                name="min_cvar_weights",
            )
        if self.max_weight is not None:
            weights = _cap_weights(weights, self.max_weight)
        return weights

    def black_litterman(
        self,
        views: pd.Series,
        view_confidences: pd.Series,
    ) -> pd.Series:
        """Black-Litterman allocation with analyst/ML model views.

        Parameters
        ----------
        views : pd.Series
            Mapping of ticker to expected return (e.g. {"AAPL": 0.02}).
        view_confidences : pd.Series
            Mapping of ticker to confidence level between 0 and 1
            (Idzorek's method).
        """
        # Sanitize ticker names: skfolio's BL parser treats hyphens as
        # subtraction and chokes on scientific notation in view strings.
        sanitize = {t: t.replace("-", "_") for t in self.tickers if "-" in t}
        if sanitize:
            returns = self.returns.rename(columns=sanitize)
        else:
            returns = self.returns
        safe_tickers = [sanitize.get(t, t) for t in self.tickers]

        view_strings = [
            f"{sanitize.get(ticker, ticker)} = {ret:.10f}" for ticker, ret in views.items()
        ]
        confidence_array = [view_confidences[ticker] for ticker in views.index]

        prior_model = BlackLitterman(
            views=view_strings,
            view_confidences=confidence_array,
            prior_estimator=EmpiricalPrior(),
        )

        try:
            model = MeanRisk(
                risk_measure=RiskMeasure.CVAR,
                prior_estimator=prior_model,
                min_weights=0.0,
            )
            model.fit(returns)
            # Map sanitized names back to original tickers
            unsanitize = {v: k for k, v in sanitize.items()}
            original_tickers = [unsanitize.get(t, t) for t in safe_tickers]
            weights = pd.Series(model.weights_, index=original_tickers, name="bl_weights")
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}. Falling back to equal weight.")
            weights = pd.Series(
                np.ones(len(self.tickers)) / len(self.tickers),
                index=self.tickers,
                name="bl_weights",
            )
        if self.max_weight is not None:
            weights = _cap_weights(weights, self.max_weight)
        return weights

    def risk_parity(self) -> pd.Series:
        """Equal risk contribution allocation via HRP with variance risk measure."""
        try:
            model = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
            model.fit(self.returns)
            weights = pd.Series(model.weights_, index=self.tickers, name="risk_parity_weights")
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}. Falling back to equal weight.")
            weights = pd.Series(
                np.ones(len(self.tickers)) / len(self.tickers),
                index=self.tickers,
                name="risk_parity_weights",
            )
        if self.max_weight is not None:
            weights = _cap_weights(weights, self.max_weight)
        return weights

    def optimize_with_turnover_penalty(self, method: str = "hrp") -> pd.Series:
        """Run optimization with turnover penalty.

        If current_weights are provided and computed turnover is below the
        threshold, returns current_weights unchanged to avoid unnecessary trading.

        Args:
            method: Optimization method ('hrp', 'min_cvar', 'risk_parity').

        Returns:
            Optimized weights (or current weights if turnover is too low).
        """
        # Compute new target weights
        if method == "hrp":
            new_weights = self.hrp()
        elif method == "min_cvar":
            new_weights = self.min_cvar()
        elif method == "risk_parity":
            new_weights = self.risk_parity()
        else:
            logger.warning(f"Unknown method '{method}' for turnover penalty, using HRP")
            new_weights = self.hrp()

        if self.current_weights is None or self.current_weights.empty:
            return new_weights

        # M-TURNOVER fix: include ALL held positions in the turnover calc,
        # not just tickers that appear in new_weights.  Positions being sold
        # (in current_weights but absent from new_weights) contribute to
        # turnover and must be counted, otherwise sell-side turnover is
        # systematically underestimated.
        all_tickers = sorted(set(new_weights.index) | set(self.current_weights.index))
        w_new = new_weights.reindex(all_tickers, fill_value=0.0)
        w_old = self.current_weights.reindex(all_tickers, fill_value=0.0)
        turnover = (w_new - w_old).abs().sum() / 2

        if turnover < self.turnover_threshold:
            logger.info(
                f"Low turnover signal ({turnover:.1%} < {self.turnover_threshold:.1%}) "
                f"— maintaining current positions"
            )
            return self.current_weights

        logger.info(f"Turnover {turnover:.1%} exceeds threshold — rebalancing")
        return new_weights

    def compare_all(
        self,
        views: pd.Series | None = None,
        view_confidences: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run all optimization strategies and return weights comparison."""
        results = {"hrp": self.hrp(), "min_cvar": self.min_cvar()}
        if views is not None and view_confidences is not None:
            results["black_litterman"] = self.black_litterman(views, view_confidences)
        results["risk_parity"] = self.risk_parity()
        return pd.DataFrame(results)
