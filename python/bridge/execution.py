"""Execution bridge for live trading simulation.

Connects portfolio decisions to order execution with:
- Risk validation before execution
- Position tracking
- P&L monitoring
- Integration with Rust matching engine
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
            self.timestamp = datetime.now(tz=timezone.utc)


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
        """Update position with new fill.

        Handles both long and short positions correctly:
        - Long entry: avg_cost weighted average
        - Short entry: avg_cost tracks entry price (negative quantity)
        - Long cover: P&L realized, quantity reduced
        - Short cover: P&L realized, quantity increased toward zero
        """
        if fill.order.side == "BUY":
            if self.quantity >= 0:
                # Adding to long position or flat -> long
                total_cost = (self.quantity * self.avg_cost) + (
                    fill.fill_quantity * fill.fill_price
                )
                self.quantity += fill.fill_quantity
                self.avg_cost = total_cost / self.quantity if self.quantity != 0 else 0.0
            else:
                # Covering short position
                cover_qty = min(fill.fill_quantity, abs(self.quantity))
                self.realized_pnl += cover_qty * (self.avg_cost - fill.fill_price)
                self.quantity += fill.fill_quantity
                if self.quantity > 0:
                    # Switch to long - new avg_cost is fill price for remaining
                    self.avg_cost = fill.fill_price
        else:  # SELL
            if self.quantity > 0:
                # Reducing long position
                sell_qty = min(fill.fill_quantity, self.quantity)
                self.realized_pnl += sell_qty * (fill.fill_price - self.avg_cost)
                self.quantity -= fill.fill_quantity
                if self.quantity < 0:
                    # Switch to short - new avg_cost is fill price for remaining
                    self.avg_cost = fill.fill_price
            else:
                # Adding to short position
                total_cost = (abs(self.quantity) * self.avg_cost) + (
                    fill.fill_quantity * fill.fill_price
                )
                self.quantity -= fill.fill_quantity
                self.avg_cost = total_cost / abs(self.quantity) if self.quantity != 0 else 0.0

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
        self._last_prices: Dict[str, float] = {}

        # Performance tracking
        self.equity_curve: List[Dict] = [
            {"timestamp": datetime.now(tz=timezone.utc), "equity": initial_capital}
        ]

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
            # C6 fix: calculate TARGET POSITION weight, not trade size weight.
            portfolio_value = self.equity
            if portfolio_value > 0:
                existing_pos = self.get_position(ticker)
                existing_value = existing_pos.quantity * current_price
                if side == "BUY":
                    target_value = existing_value + (quantity * current_price)
                else:
                    target_value = existing_value - (quantity * current_price)
                weight = target_value / portfolio_value
            else:
                weight = 0

            # H-STALE fix: bypass risk check for position closes (target weight ~0)
            # Closing stale positions should never be blocked by risk limits.
            is_closing = abs(weight) < 1e-6

            if not is_closing:
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
                # Negate weight for sells so risk manager tracks correctly
                if fill.order.side == "SELL":
                    weight_change = -weight_change
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

        # M7 fix: check if sell would open a short (sell more than we hold)
        if order.side == "SELL":
            current_pos = self.get_position(order.ticker) if hasattr(self, "positions") else None
            current_qty = current_pos.quantity if current_pos else 0
            if order.quantity > current_qty:
                # H-SHORT fix: short sells need margin — compare against equity
                # (not cash) since margin is based on total portfolio value
                short_value = (order.quantity - current_qty) * current_price
                if short_value + commission > self.equity:
                    logger.warning(f"Insufficient equity for {order.ticker} short sell")
                    return None

        fill = Fill(
            order=order,
            fill_price=current_price,
            fill_quantity=order.quantity,
            commission=commission,
            timestamp=datetime.now(tz=timezone.utc),
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
        self._update_equity({fill.order.ticker: current_price})

    def _update_equity(self, prices: Dict[str, float]) -> None:
        """Update total equity using per-position prices.

        Args:
            prices: Dictionary mapping ticker -> current price.
                    Positions without a price entry are valued at their last-known price.
        """
        # Update last-known prices
        self._last_prices.update(prices)

        positions_value = 0.0
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, self._last_prices.get(ticker))
            if price is None:
                # No price available - use avg_cost as last resort (P&L will show as zero)
                price = pos.avg_cost
            positions_value += pos.market_value(price)
        self.equity = self.cash + positions_value

        # Record equity curve point (M6 fix: cap at 10k entries to prevent unbounded growth)
        self.equity_curve.append(
            {"timestamp": datetime.now(tz=timezone.utc), "equity": self.equity}
        )
        if len(self.equity_curve) > 10_000:
            # H-EQCURVE fix: non-overlapping compaction — subsample old, keep recent
            historical = self.equity_curve[:-1000][::10]
            recent = self.equity_curve[-1000:]
            self.equity_curve = historical + recent

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update positions with current prices."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                position = self.positions[ticker]
                # Calculate unrealized P&L
                if position.quantity != 0:
                    position.unrealized_pnl = position.quantity * (price - position.avg_cost)

        # Update total equity using all position prices
        if prices:
            self._update_equity(prices)

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

        # Close positions not in target weights (stale positions)
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            if abs(pos.quantity) > 1e-6 and ticker not in target_weights:
                if ticker not in prices:
                    logger.warning(f"Cannot close stale position {ticker}: no price available")
                    continue
                # For short positions (quantity < 0), we need to BUY to cover
                close_side = "BUY" if pos.quantity < 0 else "SELL"
                fill = self.submit_order(
                    ticker=ticker,
                    side=close_side,
                    quantity=abs(pos.quantity),
                    current_price=prices[ticker],
                    current_date=current_date,
                )
                if fill:
                    fills.append(fill)

        # Snapshot equity before the loop so fills don't mutate targets mid-rebalance
        snapshot_equity = self.equity

        for ticker, target_weight in target_weights.items():
            if ticker not in prices:
                continue

            # M-NANWT fix: skip NaN weights to prevent positions being locked
            if pd.isna(target_weight):
                logger.warning(f"NaN weight for {ticker}, skipping (position unchanged)")
                continue

            current_price = prices[ticker]
            if current_price <= 0:
                logger.warning(f"Invalid price {current_price} for {ticker}, skipping")
                continue
            target_value = target_weight * snapshot_equity
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
