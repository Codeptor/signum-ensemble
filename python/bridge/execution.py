"""Execution bridge for live trading simulation.

Connects portfolio decisions to order execution with:
- Risk validation before execution
- Position tracking
- P&L monitoring
- Integration with Rust matching engine
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from python.portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Trade order."""

    ticker: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str = "MARKET"  # MARKET, LIMIT, etc.
    limit_price: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Fill:
    """Order fill/execution."""

    order: Order
    fill_price: float
    fill_quantity: float
    commission: float
    timestamp: datetime
    realized_pnl: Optional[float] = None


@dataclass
class Position:
    """Current position in an asset."""

    ticker: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def market_value(self, current_price: float) -> float:
        """Calculate current market value."""
        return self.quantity * current_price

    def update(self, fill: Fill) -> None:
        """Update position with new fill."""
        if fill.order.side == "BUY":
            # Calculate new average cost
            total_cost = (self.quantity * self.avg_cost) + (fill.fill_quantity * fill.fill_price)
            self.quantity += fill.fill_quantity
            if self.quantity > 0:
                self.avg_cost = total_cost / self.quantity
        else:  # SELL
            # Calculate realized P&L
            if self.quantity > 0:
                self.realized_pnl += fill.fill_quantity * (fill.fill_price - self.avg_cost)
            self.quantity -= fill.fill_quantity

        if abs(self.quantity) < 1e-10:
            self.quantity = 0.0
            self.avg_cost = 0.0


class ExecutionBridge:
    """
    Bridge between portfolio decisions and order execution.

    Validates orders through RiskManager, tracks positions,
    and monitors P&L in real-time.
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,  # 10 bps
    ):
        """
        Args:
            risk_manager: Risk manager for validation
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
        """
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        # State tracking
        self.positions: Dict[str, Position] = {}
        self.cash: float = initial_capital
        self.equity: float = initial_capital
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        self.daily_pnl: Dict[str, float] = {}

        # Performance tracking
        self.equity_curve: List[Dict] = [{"timestamp": datetime.now(), "equity": initial_capital}]

    def get_position(self, ticker: str) -> Position:
        """Get or create position for ticker."""
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
        return self.positions[ticker]

    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        current_price: float,
        current_date: Optional[str] = None,
    ) -> Optional[Fill]:
        """
        Submit order with risk validation.

        Args:
            ticker: Asset ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares/contracts
            current_price: Current market price
            current_date: Current date for risk checks

        Returns:
            Fill object if executed, None if rejected
        """
        # Validate through risk manager
        if self.risk_manager is not None:
            # Calculate weight
            portfolio_value = self.equity
            weight = (quantity * current_price) / portfolio_value if portfolio_value > 0 else 0

            can_execute, issues = self.risk_manager.can_execute_trade(
                ticker=ticker,
                new_weight=weight,
                current_date=current_date,
            )

            if not can_execute:
                logger.warning(f"Order rejected for {ticker}: {issues}")
                return None

        # Create order
        order = Order(
            ticker=ticker,
            side=side,
            quantity=quantity,
        )
        self.orders.append(order)

        # Simulate execution (immediate fill at current price)
        fill = self._execute_order(order, current_price)

        if fill:
            self.fills.append(fill)
            self._update_state(fill, current_price)

            # Record trade in risk manager
            if self.risk_manager is not None:
                weight_change = (
                    (fill.fill_quantity * fill.fill_price) / self.equity if self.equity > 0 else 0
                )
                self.risk_manager.record_trade(ticker, weight_change, current_date)

        return fill

    def _execute_order(self, order: Order, current_price: float) -> Optional[Fill]:
        """Execute order and return fill."""
        # Calculate commission
        trade_value = order.quantity * current_price
        commission = trade_value * self.commission_rate

        # Check if we have enough cash for buys
        if order.side == "BUY" and trade_value + commission > self.cash:
            logger.warning(f"Insufficient cash for {order.ticker} buy order")
            return None

        fill = Fill(
            order=order,
            fill_price=current_price,
            fill_quantity=order.quantity,
            commission=commission,
            timestamp=datetime.now(),
        )

        return fill

    def _update_state(self, fill: Fill, current_price: float) -> None:
        """Update portfolio state after fill."""
        position = self.get_position(fill.order.ticker)

        # Update position
        position.update(fill)

        # Update cash
        trade_value = fill.fill_quantity * fill.fill_price
        if fill.order.side == "BUY":
            self.cash -= trade_value + fill.commission
        else:
            self.cash += trade_value - fill.commission

        # Update equity
        self._update_equity(current_price)

    def _update_equity(self, current_price: float) -> None:
        """Update total equity."""
        positions_value = sum(pos.market_value(current_price) for pos in self.positions.values())
        self.equity = self.cash + positions_value

        # Record equity curve point
        self.equity_curve.append({"timestamp": datetime.now(), "equity": self.equity})

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update positions with current prices."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                position = self.positions[ticker]
                # Calculate unrealized P&L
                if position.quantity != 0:
                    position.unrealized_pnl = position.quantity * (price - position.avg_cost)

        # Update total equity
        if prices:
            # Use first price for equity calculation (simplified)
            self._update_equity(list(prices.values())[0])

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        return {
            "cash": self.cash,
            "equity": self.equity,
            "total_return": (self.equity / self.initial_capital) - 1,
            "total_return_pct": ((self.equity / self.initial_capital) - 1) * 100,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_realized_pnl + total_unrealized_pnl,
            "num_positions": len([p for p in self.positions.values() if p.quantity != 0]),
            "num_trades": len(self.fills),
            "total_commission": sum(f.commission for f in self.fills),
        }

    def get_position_summary(self) -> pd.DataFrame:
        """Get DataFrame of all positions."""
        data = []
        for ticker, pos in self.positions.items():
            if pos.quantity != 0:
                data.append(
                    {
                        "ticker": ticker,
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "realized_pnl": pos.realized_pnl,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                )
        return pd.DataFrame(data)

    def reconcile_target_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        current_date: Optional[str] = None,
    ) -> List[Fill]:
        """
        Reconcile current positions with target weights.

        Args:
            target_weights: Desired portfolio weights
            prices: Current prices for each asset
            current_date: Current date for risk checks

        Returns:
            List of fills from executed trades
        """
        fills = []

        for ticker, target_weight in target_weights.items():
            if ticker not in prices:
                continue

            current_price = prices[ticker]
            target_value = target_weight * self.equity
            target_quantity = target_value / current_price

            # Get current position
            current_position = self.get_position(ticker)
            current_quantity = current_position.quantity

            # Calculate trade needed
            quantity_diff = target_quantity - current_quantity

            if abs(quantity_diff) > 1e-6:  # Minimum trade size
                side = "BUY" if quantity_diff > 0 else "SELL"
                quantity = abs(quantity_diff)

                fill = self.submit_order(
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    current_price=current_price,
                    current_date=current_date,
                )

                if fill:
                    fills.append(fill)

        return fills


class PaperTradingEngine:
    """
    Paper trading engine for simulation.

    Simulates live trading without real capital at risk.
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 1_000_000.0,
    ):
        """
        Args:
            risk_manager: Risk manager for validation
            initial_capital: Starting capital
        """
        self.execution_bridge = ExecutionBridge(
            risk_manager=risk_manager,
            initial_capital=initial_capital,
        )
        self.trade_history: List[Dict] = []

    def run_strategy(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict:
        """
        Run strategy with signals.

        Args:
            signals: DataFrame with columns ['ticker', 'signal', 'weight']
            prices: DataFrame with price data

        Returns:
            Performance summary
        """
        for date in signals.index.get_level_values(0).unique():
            day_signals = signals.loc[date]
            day_prices = prices.loc[date] if date in prices.index else {}

            # Convert signals to target weights
            target_weights = {}
            for ticker in day_signals.index.get_level_values(0).unique():
                if "weight" in day_signals.columns:
                    target_weights[ticker] = day_signals.loc[ticker, "weight"]
                elif "signal" in day_signals.columns:
                    # Convert signal to weight
                    target_weights[ticker] = day_signals.loc[ticker, "signal"]

            # Reconcile positions
            fills = self.execution_bridge.reconcile_target_weights(
                target_weights=target_weights,
                prices=day_prices.to_dict() if hasattr(day_prices, "to_dict") else {},
                current_date=date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            )

            # Record trades
            for fill in fills:
                self.trade_history.append(
                    {
                        "date": date,
                        "ticker": fill.order.ticker,
                        "side": fill.order.side,
                        "quantity": fill.fill_quantity,
                        "price": fill.fill_price,
                        "commission": fill.commission,
                    }
                )

        return self.execution_bridge.get_portfolio_summary()
