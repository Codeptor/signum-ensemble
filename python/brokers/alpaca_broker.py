"""Alpaca broker implementation for paper and live trading."""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from python.brokers.base import (
    BaseBroker,
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
)

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker integration.

    Supports both paper trading (free) and live trading.
    Get API keys from: https://app.alpaca.markets/paper/dashboard/overview
    """

    def __init__(
        self,
        paper_trading: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Args:
            paper_trading: If True, use paper trading environment
            api_key: Alpaca API key (defaults to ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (defaults to ALPACA_API_SECRET env var)
        """
        super().__init__(paper_trading=paper_trading)

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables or pass directly."
            )

        # Import alpaca_trade_api here to avoid hard dependency
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api required. Install with: pip install alpaca-trade-api"
            )

        base_url = (
            "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
        )

        self.api = tradeapi.REST(self.api_key, self.api_secret, base_url, api_version="v2")

        self._connected = False

    def connect(self) -> bool:
        """Connect to Alpaca API and verify credentials."""
        try:
            # Test connection by getting account info
            account = self.api.get_account()
            self._connected = True
            logger.info(
                f"Connected to Alpaca {'paper' if self.paper_trading else 'live'} "
                f"trading. Account: {account.id}, Status: {account.status}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._connected

    def get_account(self) -> BrokerAccount:
        """Get Alpaca account information."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        account = self.api.get_account()

        return BrokerAccount(
            account_id=account.id,
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            status=account.status,
        )

    def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order to Alpaca.

        Supports simple orders (market, limit, stop) and bracket orders
        with attached stop-loss and/or take-profit legs.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            kwargs = {
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side,
                "type": order.order_type,
                "time_in_force": order.time_in_force,
            }

            if order.limit_price is not None:
                kwargs["limit_price"] = order.limit_price
            if order.stop_price is not None:
                kwargs["stop_price"] = order.stop_price
            if order.client_order_id is not None:
                kwargs["client_order_id"] = order.client_order_id

            # Bracket order support
            if order.order_class:
                kwargs["order_class"] = order.order_class

                if order.take_profit_limit_price is not None:
                    kwargs["take_profit"] = {"limit_price": str(order.take_profit_limit_price)}

                if order.stop_loss_stop_price is not None:
                    stop_loss = {"stop_price": str(order.stop_loss_stop_price)}
                    if order.stop_loss_limit_price is not None:
                        stop_loss["limit_price"] = str(order.stop_loss_limit_price)
                    kwargs["stop_loss"] = stop_loss

            alpaca_order = self.api.submit_order(**kwargs)

            order_desc = f"{order.side} {order.qty} {order.symbol} @{order.order_type}"
            if order.order_class:
                order_desc += f" [{order.order_class}]"
            logger.info(f"Order submitted: {order_desc} - ID: {alpaca_order.id}")

            return alpaca_order.id

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            cancelled = self.api.cancel_all_orders()
            count = len(cancelled) if cancelled else 0
            logger.info(f"Cancelled {count} open orders")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            raise

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order status by ID."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            order = self.api.get_order(order_id)
            return BrokerOrder(
                symbol=order.symbol,
                side=order.side,
                qty=float(order.qty),
                order_type=order.type,
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                time_in_force=order.time_in_force,
                client_order_id=order.client_order_id,
                order_id=order.id,
                parent_order_id=getattr(order, "parent_order_id", None),
                status=order.status,
                order_class=getattr(order, "order_class", None),
            )
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def list_orders(self, status: str = "open") -> List[BrokerOrder]:
        """List orders with given status."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            orders = self.api.list_orders(status=status)
            return [
                BrokerOrder(
                    symbol=o.symbol,
                    side=o.side,
                    qty=float(o.qty),
                    order_type=o.type,
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    stop_price=float(o.stop_price) if o.stop_price else None,
                    time_in_force=o.time_in_force,
                    client_order_id=o.client_order_id,
                    order_id=o.id,
                    parent_order_id=getattr(o, "parent_order_id", None),
                    status=o.status,
                    order_class=getattr(o, "order_class", None),
                )
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to list orders: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a symbol."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            position = self.api.get_position(symbol)
            return BrokerPosition(
                symbol=position.symbol,
                qty=float(position.qty),
                avg_entry_price=float(position.avg_entry_price),
                market_value=float(position.market_value),
                unrealized_pl=float(position.unrealized_pl),
                unrealized_plpc=float(position.unrealized_plpc),
            )
        except Exception:
            # Position might not exist
            return None

    def list_positions(self) -> List[BrokerPosition]:
        """List all positions."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            positions = self.api.list_positions()
            return [
                BrokerPosition(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                )
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            return []

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol.

        Tries Alpaca market data first (requires paid plan), then falls back
        to yfinance (free, ~15min delayed). The delay is acceptable because
        prices are only used for share sizing — orders execute at market price.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        # Try Alpaca first
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.debug(f"Alpaca price unavailable for {symbol}: {e}")

        # Fallback to yfinance
        return self._yfinance_price(symbol)

    def get_latest_prices(self, symbols: list[str]) -> Dict[str, float]:
        """Batch fetch latest prices for multiple symbols.

        Tries Alpaca first, falls back to yfinance batch download.
        More efficient than calling get_latest_price() in a loop.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        prices: Dict[str, float] = {}
        missing: list[str] = []

        # Try Alpaca for each symbol
        for sym in symbols:
            try:
                trade = self.api.get_latest_trade(sym)
                prices[sym] = float(trade.price)
            except Exception:
                missing.append(sym)

        # Batch fallback to yfinance for any Alpaca misses
        if missing:
            logger.info(f"Fetching {len(missing)} prices from yfinance: {missing}")
            try:
                import yfinance as yf

                data = yf.download(missing, period="5d", interval="1d", progress=False)
                if "Close" in data.columns or hasattr(data, "Close"):
                    close = data["Close"]
                    if isinstance(close, pd.Series):
                        # Single ticker returns a Series
                        last_price = float(close.dropna().iloc[-1])
                        prices[missing[0]] = last_price
                    else:
                        for sym in missing:
                            if sym in close.columns:
                                val = close[sym].dropna()
                                if len(val) > 0:
                                    prices[sym] = float(val.iloc[-1])
            except Exception as e:
                logger.warning(f"yfinance batch fallback failed: {e}")
                # Fall back to individual calls
                for sym in missing:
                    if sym not in prices:
                        try:
                            prices[sym] = self._yfinance_price(sym)
                        except Exception:
                            pass

        return prices

    @staticmethod
    def _yfinance_price(symbol: str) -> float:
        """Get a single price from yfinance."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get("lastPrice")
        if price and price > 0:
            logger.info(f"yfinance price for {symbol}: ${price:.2f}")
            return float(price)
        raise ValueError(f"Could not get price for {symbol} from yfinance")

    def get_clock(self) -> Dict:
        """Get market clock."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            clock = self.api.get_clock()
            return {
                "timestamp": clock.timestamp,
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
        except Exception as e:
            logger.error(f"Failed to get market clock: {e}")
            raise

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Asset symbol
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1H, 1D, etc.)
            limit: Number of bars to fetch

        Returns:
            List of bar dictionaries
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit)
            return [
                {
                    "timestamp": bar.t,
                    "open": bar.o,
                    "high": bar.h,
                    "low": bar.l,
                    "close": bar.c,
                    "volume": bar.v,
                }
                for bar in bars
            ]
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            raise


class AlpacaPaperTrading:
    """
    Convenience class for Alpaca paper trading.

    Wraps AlpacaBroker with paper trading enabled by default.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
        """
        self.broker = AlpacaBroker(
            paper_trading=True,
            api_key=api_key,
            api_secret=api_secret,
        )

    def connect(self) -> bool:
        """Connect to paper trading environment."""
        return self.broker.connect()

    def disconnect(self) -> None:
        """Disconnect from paper trading."""
        self.broker.disconnect()

    def __getattr__(self, name):
        """Delegate all other methods to broker."""
        return getattr(self.broker, name)
