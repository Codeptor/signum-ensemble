"""Base broker interface for live trading.

Defines the contract that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class BrokerOrder:
    """Standardized order representation."""

    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = "market"  # market, limit, stop, etc.
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, ioc, fok
    client_order_id: Optional[str] = None
    # Bracket order fields (Alpaca OTO/OCO)
    order_class: Optional[str] = None  # 'bracket', 'oco', 'oto', or None for simple
    take_profit_limit_price: Optional[float] = None
    stop_loss_stop_price: Optional[float] = None
    stop_loss_limit_price: Optional[float] = None
    # Identification (populated when reading orders back from broker)
    order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    status: Optional[str] = None
    filled_avg_price: Optional[float] = None  # Actual fill price from broker
    filled_qty: Optional[float] = None  # Actual filled quantity (important for partial fills)
    created_at: Optional[str] = None  # ISO timestamp when order was created


@dataclass
class BrokerFill:
    """Standardized fill/execution representation."""

    order_id: str
    symbol: str
    side: str
    filled_qty: float
    filled_avg_price: float
    commission: float
    timestamp: datetime


@dataclass
class BrokerPosition:
    """Standardized position representation."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class BrokerAccount:
    """Standardized account information."""

    account_id: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    status: str  # ACTIVE, INACTIVE, etc.


class BaseBroker(ABC):
    """Abstract base class for broker implementations."""

    def __init__(self, paper_trading: bool = True):
        """
        Args:
            paper_trading: If True, use paper/sandbox environment
        """
        self.paper_trading = paper_trading

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API and verify credentials."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        """Get account information."""
        pass

    @abstractmethod
    def submit_order(self, order: BrokerOrder) -> str:
        """
        Submit an order to the broker.

        Returns:
            Order ID from broker
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order status by ID."""
        pass

    @abstractmethod
    def list_orders(self, status: str = "open") -> List[BrokerOrder]:
        """List orders with given status."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a symbol."""
        pass

    @abstractmethod
    def list_positions(self) -> List[BrokerPosition]:
        """List all positions."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        pass

    @abstractmethod
    def get_clock(self) -> Dict:
        """Get market clock (open/close times)."""
        pass

    def sync_positions(self, local_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Sync local positions with broker positions.

        Args:
            local_positions: Dictionary of {symbol: qty} from local tracking

        Returns:
            Dictionary of discrepancies {symbol: diff_qty}
        """
        broker_positions = {p.symbol: p.qty for p in self.list_positions()}
        discrepancies = {}

        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())

        for symbol in all_symbols:
            local_qty = local_positions.get(symbol, 0)
            broker_qty = broker_positions.get(symbol, 0)

            if abs(local_qty - broker_qty) > 1e-6:
                discrepancies[symbol] = broker_qty - local_qty

        return discrepancies

    def reconcile_portfolio(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
    ) -> List[str]:
        """
        Reconcile current portfolio with target weights.

        Args:
            target_weights: Target portfolio weights {symbol: weight}
            portfolio_value: Total portfolio value

        Returns:
            List of submitted order IDs
        """
        order_ids = []
        current_positions = {p.symbol: p.qty for p in self.list_positions()}

        # Close positions not in target weights (stale positions)
        for symbol, qty in current_positions.items():
            if qty > 1e-6 and symbol not in target_weights:
                order = BrokerOrder(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    order_type="market",
                )
                try:
                    order_id = self.submit_order(order)
                    order_ids.append(order_id)
                except Exception as e:
                    print(f"Failed to close stale position {symbol}: {e}")

        for symbol, target_weight in target_weights.items():
            if target_weight < 0:
                continue

            price = self.get_latest_price(symbol)
            target_qty = (target_weight * portfolio_value) / price
            current_qty = current_positions.get(symbol, 0)
            diff = target_qty - current_qty

            if abs(diff) > 1e-6:  # Minimum order size
                side = "buy" if diff > 0 else "sell"
                qty = abs(diff)

                order = BrokerOrder(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="market",
                )

                try:
                    order_id = self.submit_order(order)
                    order_ids.append(order_id)
                except Exception as e:
                    print(f"Failed to submit order for {symbol}: {e}")

        return order_ids
