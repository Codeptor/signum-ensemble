"""Regime-specific backtesting (Phase 4, §2.4.2).

Evaluates strategy performance across distinct market regimes to understand
failure modes and set realistic expectations for live trading.

Default regimes cover:
  - Bull markets (strong uptrend)
  - Bear / correction periods
  - COVID crash (extreme vol event)
  - COVID recovery (V-shaped bounce)
  - Inflation regime (rising rates, sector rotation)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default market regime date ranges
DEFAULT_REGIMES: dict[str, tuple[str, str]] = {
    "bull_2017": ("2017-01-01", "2018-01-26"),
    "correction_2018": ("2018-01-27", "2018-12-24"),
    "recovery_2019": ("2018-12-25", "2020-02-19"),
    "covid_crash": ("2020-02-20", "2020-03-23"),
    "covid_recovery": ("2020-03-24", "2021-01-01"),
    "inflation_2022": ("2022-01-01", "2022-12-31"),
    "recovery_2023": ("2023-01-01", "2023-12-31"),
}


@dataclass
class RegimeResult:
    """Performance metrics for a single regime period."""

    regime: str
    start_date: str
    end_date: str
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    volatility: float
    n_days: int

    @property
    def is_profitable(self) -> bool:
        return self.total_return > 0


@dataclass
class RegimeAnalysis:
    """Aggregated results across all regime periods."""

    results: dict[str, RegimeResult] = field(default_factory=dict)

    @property
    def profitable_regimes(self) -> int:
        return sum(1 for r in self.results.values() if r.is_profitable)

    @property
    def total_regimes(self) -> int:
        return len(self.results)

    @property
    def mean_sharpe(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.sharpe for r in self.results.values()]))

    @property
    def worst_drawdown(self) -> float:
        if not self.results:
            return 0.0
        return float(min(r.max_drawdown for r in self.results.values()))

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all regime results."""
        rows = []
        for name, r in self.results.items():
            rows.append(
                {
                    "regime": name,
                    "start": r.start_date,
                    "end": r.end_date,
                    "total_return": r.total_return,
                    "annual_return": r.annual_return,
                    "sharpe": r.sharpe,
                    "max_drawdown": r.max_drawdown,
                    "volatility": r.volatility,
                    "n_days": r.n_days,
                    "profitable": r.is_profitable,
                }
            )
        return pd.DataFrame(rows)


def compute_regime_metrics(
    returns: pd.Series,
    regime: str,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.05,
) -> RegimeResult:
    """Compute performance metrics for a returns series within a regime.

    Args:
        returns: Daily returns series (not cumulative).
        regime: Regime name/label.
        start_date: Regime start (for metadata).
        end_date: Regime end (for metadata).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        RegimeResult with computed metrics.
    """
    if len(returns) == 0:
        return RegimeResult(
            regime=regime,
            start_date=start_date,
            end_date=end_date,
            total_return=0.0,
            annual_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            n_days=0,
        )

    n_days = len(returns)
    total_return = float((1 + returns).prod() - 1)

    # Annualize
    trading_days = 252
    years = n_days / trading_days
    annual_return = float((1 + total_return) ** (1 / max(years, 0.01)) - 1) if years > 0 else 0.0

    vol = float(returns.std() * np.sqrt(trading_days))
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    # M-REGIME-SHARPE fix: denominator must be excess.std(), not returns.std().
    # Using returns.std() ignores the variance of the risk-free subtraction
    # (minor for constant rf, but inconsistent with standard Sharpe definition).
    excess_std = excess.std()
    sharpe = float(excess.mean() / excess_std * np.sqrt(trading_days)) if excess_std > 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(drawdowns.min())

    return RegimeResult(
        regime=regime,
        start_date=start_date,
        end_date=end_date,
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        volatility=vol,
        n_days=n_days,
    )


def backtest_by_regime(
    strategy_returns: pd.Series,
    regimes: Optional[dict[str, tuple[str, str]]] = None,
    risk_free_rate: float = 0.05,
) -> RegimeAnalysis:
    """Evaluate strategy returns across market regimes.

    This function slices a full strategy returns series into regime-specific
    windows and computes performance metrics for each.

    Args:
        strategy_returns: Daily returns series with a DatetimeIndex.
        regimes: Dict mapping regime_name -> (start_date, end_date).
            Defaults to ``DEFAULT_REGIMES``.
        risk_free_rate: Annual risk-free rate for Sharpe.

    Returns:
        RegimeAnalysis with per-regime results.
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    analysis = RegimeAnalysis()

    for regime_name, (start, end) in regimes.items():
        mask = (strategy_returns.index >= start) & (strategy_returns.index <= end)
        regime_returns = strategy_returns[mask]

        if len(regime_returns) == 0:
            logger.warning(
                f"Regime '{regime_name}' ({start} to {end}): no data in returns series, skipping"
            )
            continue

        result = compute_regime_metrics(
            regime_returns,
            regime_name,
            start,
            end,
            risk_free_rate,
        )
        analysis.results[regime_name] = result

        logger.info(
            f"Regime '{regime_name}' ({start} to {end}): "
            f"return={result.total_return:.2%}, sharpe={result.sharpe:.2f}, "
            f"maxDD={result.max_drawdown:.2%}, n_days={result.n_days}"
        )

    logger.info(
        f"Regime analysis: {analysis.profitable_regimes}/{analysis.total_regimes} "
        f"profitable, mean Sharpe={analysis.mean_sharpe:.2f}, "
        f"worst DD={analysis.worst_drawdown:.2%}"
    )

    return analysis
