"""Risk management: real-time monitoring and enforcement."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from python.portfolio.risk import RiskEngine


@dataclass
class RiskCheck:
    """Result of a risk check."""

    passed: bool
    rule: str
    message: str
    severity: str = "info"  # info, warning, critical
    metric_value: Optional[float] = None
    limit_value: Optional[float] = None


@dataclass
class RiskLimits:
    """Configuration for risk limits."""

    # Position limits
    max_position_weight: float = 0.25  # Max 25% per position
    max_sector_weight: float = 0.50  # Max 50% per sector
    min_position_weight: float = 0.01  # Min 1% (avoid dust positions)

    # Portfolio limits
    max_portfolio_var_95: float = 0.05  # Max 5% daily VaR
    max_drawdown_limit: float = 0.20  # Max 20% drawdown
    min_sharpe_ratio: float = 0.0  # Minimum Sharpe

    # Trading limits
    max_daily_trades: int = 20  # Max trades per day
    max_daily_turnover: float = 1.0  # Max 100% portfolio turnover per day
    max_single_trade_size: float = 0.15  # Max 15% in single trade

    # Risk/Reward
    min_risk_reward_ratio: float = 2.0  # Minimum 2:1
    max_leverage: float = 1.0  # No leverage by default

    # Volatility regime
    reduce_size_high_vol: bool = True  # Reduce positions in high vol
    high_vol_threshold: float = 0.25  # 25% annualized vol


class RiskManager:
    """
    Real-time risk monitoring and enforcement for trading.

    Implements risk checks from the risk-management skill:
    - Position size limits (max 25% per position)
    - Trade frequency adaptation
    - Risk validation before entry
    - Drawdown limits with stop-losses
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        risk_free_rate: float = 0.02,
    ):
        """
        Args:
            limits: Risk limits configuration
            risk_free_rate: Annual risk-free rate
        """
        self.limits = limits or RiskLimits()
        self.risk_free_rate = risk_free_rate

        # Tracking
        self.daily_trades: Dict[str, int] = field(default_factory=dict)
        self.daily_turnover: Dict[str, float] = field(default_factory=dict)
        self.risk_engine: Optional[RiskEngine] = None
        self.current_weights: Optional[pd.Series] = None

    def initialize_portfolio_risk(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """Initialize risk engine with portfolio data."""
        self.risk_engine = RiskEngine(
            returns=returns,
            weights=weights,
            rf_rate=self.risk_free_rate,
            benchmark_returns=benchmark_returns,
        )
        self.current_weights = weights.copy()

    def check_trade(
        self,
        ticker: str,
        new_weight: float,
        expected_return: float = 0.0,
        risk_amount: float = 0.0,
        current_date: Optional[str] = None,
    ) -> List[RiskCheck]:
        """
        Validate a proposed trade against all risk rules.

        Args:
            ticker: Asset ticker
            new_weight: Proposed weight after trade
            expected_return: Expected return for risk/reward calc
            risk_amount: Amount at risk for risk/reward calc
            current_date: Current date for daily limits

        Returns:
            List of RiskCheck results
        """
        checks = []
        date_key = current_date or pd.Timestamp.now().strftime("%Y-%m-%d")

        # 1. Position size limit
        if new_weight > self.limits.max_position_weight:
            checks.append(
                RiskCheck(
                    passed=False,
                    rule="MAX_POSITION_SIZE",
                    message=f"Position size {new_weight:.1%} exceeds limit of "
                    f"{self.limits.max_position_weight:.1%}",
                    severity="critical",
                    metric_value=new_weight,
                    limit_value=self.limits.max_position_weight,
                )
            )
        elif new_weight < self.limits.min_position_weight and new_weight > 0:
            checks.append(
                RiskCheck(
                    passed=False,
                    rule="MIN_POSITION_SIZE",
                    message=f"Position size {new_weight:.1%} below minimum of "
                    f"{self.limits.min_position_weight:.1%}",
                    severity="warning",
                    metric_value=new_weight,
                    limit_value=self.limits.min_position_weight,
                )
            )
        else:
            checks.append(
                RiskCheck(
                    passed=True,
                    rule="POSITION_SIZE",
                    message=f"Position size {new_weight:.1%} within limits",
                    severity="info",
                )
            )

        # 2. Daily trade limit
        daily_count = self.daily_trades.get(date_key, 0)
        if daily_count >= self.limits.max_daily_trades:
            checks.append(
                RiskCheck(
                    passed=False,
                    rule="MAX_DAILY_TRADES",
                    message=f"Daily trade limit ({self.limits.max_daily_trades}) reached",
                    severity="critical",
                    metric_value=daily_count,
                    limit_value=self.limits.max_daily_trades,
                )
            )
        else:
            checks.append(
                RiskCheck(
                    passed=True,
                    rule="DAILY_TRADE_COUNT",
                    message=f"Daily trades: {daily_count}/{self.limits.max_daily_trades}",
                    severity="info",
                )
            )

        # 3. Risk/Reward ratio
        if risk_amount > 0:
            rr_ratio = expected_return / risk_amount
            if rr_ratio < self.limits.min_risk_reward_ratio:
                checks.append(
                    RiskCheck(
                        passed=False,
                        rule="MIN_RISK_REWARD",
                        message=f"Risk/Reward {rr_ratio:.1f}:1 below minimum "
                        f"{self.limits.min_risk_reward_ratio:.1f}:1",
                        severity="warning",
                        metric_value=rr_ratio,
                        limit_value=self.limits.min_risk_reward_ratio,
                    )
                )
            else:
                checks.append(
                    RiskCheck(
                        passed=True,
                        rule="RISK_REWARD",
                        message=f"Risk/Reward {rr_ratio:.1f}:1 meets minimum",
                        severity="info",
                    )
                )

        # 4. Leverage check
        if self.current_weights is not None:
            total_weight = self.current_weights.sum() + new_weight
            if total_weight > self.limits.max_leverage:
                checks.append(
                    RiskCheck(
                        passed=False,
                        rule="MAX_LEVERAGE",
                        message=f"Total exposure {total_weight:.1%} exceeds leverage limit",
                        severity="critical",
                        metric_value=total_weight,
                        limit_value=self.limits.max_leverage,
                    )
                )

        return checks

    def check_portfolio_risk(self, returns: pd.Series) -> List[RiskCheck]:
        """
        Check overall portfolio risk metrics.

        Args:
            returns: Portfolio return series

        Returns:
            List of RiskCheck results
        """
        checks = []

        if self.risk_engine is None:
            return checks

        # 1. VaR limit
        var_95 = self.risk_engine.var_historical(0.95)
        if abs(var_95) > self.limits.max_portfolio_var_95:
            checks.append(
                RiskCheck(
                    passed=False,
                    rule="MAX_VAR_95",
                    message=f"Portfolio VaR (95%) {abs(var_95):.2%} exceeds limit "
                    f"{self.limits.max_portfolio_var_95:.2%}",
                    severity="critical",
                    metric_value=abs(var_95),
                    limit_value=self.limits.max_portfolio_var_95,
                )
            )

        # 2. Drawdown limit
        max_dd = abs(self.risk_engine.max_drawdown())
        if max_dd > self.limits.max_drawdown_limit:
            checks.append(
                RiskCheck(
                    passed=False,
                    rule="MAX_DRAWDOWN",
                    message=f"Drawdown {max_dd:.2%} exceeds limit "
                    f"{self.limits.max_drawdown_limit:.2%}",
                    severity="critical",
                    metric_value=max_dd,
                    limit_value=self.limits.max_drawdown_limit,
                )
            )

        # 3. Sharpe ratio minimum
        sharpe = self.risk_engine.sharpe_ratio()
        if sharpe < self.limits.min_sharpe_ratio:
            checks.append(
                RiskCheck(
                    passed=False,
                    rule="MIN_SHARPE",
                    message=f"Sharpe ratio {sharpe:.2f} below minimum "
                    f"{self.limits.min_sharpe_ratio:.2f}",
                    severity="warning",
                    metric_value=sharpe,
                    limit_value=self.limits.min_sharpe_ratio,
                )
            )

        # 4. Volatility regime check
        if self.limits.reduce_size_high_vol:
            vol = self.risk_engine.volatility()
            if vol > self.limits.high_vol_threshold:
                checks.append(
                    RiskCheck(
                        passed=False,
                        rule="HIGH_VOLATILITY_REGIME",
                        message=f"High volatility regime ({vol:.1%}). Consider reducing positions.",
                        severity="warning",
                        metric_value=vol,
                        limit_value=self.limits.high_vol_threshold,
                    )
                )

        return checks

    def can_execute_trade(
        self,
        ticker: str,
        new_weight: float,
        expected_return: float = 0.0,
        risk_amount: float = 0.0,
        current_date: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Check if trade can be executed (all critical checks pass).

        Returns:
            Tuple of (can_execute, list of blocking issues)
        """
        checks = self.check_trade(
            ticker=ticker,
            new_weight=new_weight,
            expected_return=expected_return,
            risk_amount=risk_amount,
            current_date=current_date,
        )

        blocking_issues = [c.message for c in checks if not c.passed and c.severity == "critical"]

        return len(blocking_issues) == 0, blocking_issues

    def record_trade(
        self,
        ticker: str,
        weight_change: float,
        current_date: Optional[str] = None,
    ):
        """Record a trade for daily limit tracking."""
        date_key = current_date or pd.Timestamp.now().strftime("%Y-%m-%d")

        # Update trade count
        self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1

        # Update turnover
        self.daily_turnover[date_key] = self.daily_turnover.get(date_key, 0.0) + abs(weight_change)

        # Update current weights
        if self.current_weights is not None:
            if ticker in self.current_weights.index:
                self.current_weights[ticker] += weight_change
            else:
                self.current_weights[ticker] = weight_change

    def get_risk_summary(self) -> Dict:
        """Get summary of current risk status."""
        if self.risk_engine is None:
            return {"error": "Risk engine not initialized"}

        portfolio_checks = self.check_portfolio_risk(pd.Series())

        return {
            "risk_engine_initialized": True,
            "num_critical_violations": sum(
                1 for c in portfolio_checks if not c.passed and c.severity == "critical"
            ),
            "num_warnings": sum(
                1 for c in portfolio_checks if not c.passed and c.severity == "warning"
            ),
            "portfolio_metrics": self.risk_engine.summary(),
            "current_limits": {
                "max_position": self.limits.max_position_weight,
                "max_daily_trades": self.limits.max_daily_trades,
                "max_var": self.limits.max_portfolio_var_95,
            },
        }


class PositionSizer:
    """
    Position sizing based on risk management rules.

    Implements Kelly criterion and risk-based sizing.
    """

    def __init__(
        self,
        max_position_weight: float = 0.25,
        risk_per_trade: float = 0.02,  # 2% risk per trade
    ):
        self.max_position_weight = max_position_weight
        self.risk_per_trade = risk_per_trade

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly criterion position size.

        f* = (p × b - q) / b
        where p = win rate, q = loss rate, b = avg_win/avg_loss

        Args:
            win_rate: Probability of win (0-1)
            avg_win: Average win amount (%)
            avg_loss: Average loss amount (%)

        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss == 0:
            return 0.0

        loss_rate = 1 - win_rate
        b = avg_win / avg_loss  # Win/loss ratio

        kelly = (win_rate * b - loss_rate) / b

        # Half-Kelly for safety
        return max(0.0, min(kelly * 0.5, self.max_position_weight))

    def risk_based_size(
        self,
        stop_loss_pct: float,
        portfolio_value: float = 1.0,
    ) -> float:
        """
        Calculate position size based on risk amount.

        position_size = (portfolio_value × risk_per_trade) / stop_loss

        Args:
            stop_loss_pct: Stop loss percentage (e.g., 0.05 for 5%)
            portfolio_value: Portfolio value (default: 1.0 for % sizing)

        Returns:
            Position size as fraction of portfolio
        """
        if stop_loss_pct == 0:
            return 0.0

        size = (portfolio_value * self.risk_per_trade) / stop_loss_pct
        return min(size, self.max_position_weight)

    def volatility_adjusted_size(
        self,
        base_size: float,
        current_vol: float,
        target_vol: float = 0.15,
    ) -> float:
        """
        Adjust position size based on current volatility.

        size = base_size × (target_vol / current_vol)

        Args:
            base_size: Base position size
            current_vol: Current realized volatility
            target_vol: Target volatility (default: 15%)

        Returns:
            Adjusted position size
        """
        if current_vol == 0:
            return base_size

        adjustment = target_vol / current_vol
        size = base_size * adjustment

        return min(size, self.max_position_weight)
