"""Simulated exchange using the Rust matching engine for realistic backtesting.

Seeds a limit order book from OHLCV data and routes backtest orders through
the Rust price-time-priority matching engine, producing realistic:
  - Partial fills from limited liquidity
  - Price impact from sweeping multiple levels
  - Spread costs from bid-ask crossing
  - IOC/FOK behavior

Usage::

    from python.bridge.simulated_exchange import SimulatedExchange

    exchange = SimulatedExchange()
    exchange.seed_book("AAPL", mid_price=180.0, spread=0.02, depth=10)
    fills = exchange.submit_market_order("AAPL", side="BUY", quantity=100)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the Rust matching engine; fall back gracefully
try:
    from matching_engine_py import PyOrderBook, PyOrderType, PySide

    RUST_ENGINE_AVAILABLE = True
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    logger.warning(
        "matching_engine_py not installed. "
        "Install with: cd rust/matching-engine-py && maturin develop --release"
    )


@dataclass
class SimulatedFill:
    """A fill from the simulated exchange."""

    ticker: str
    side: str
    price: float
    quantity: int
    order_id: int
    is_partial: bool = False

    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class BookStats:
    """Summary statistics for a seeded order book."""

    ticker: str
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    mid_price: float | None
    bid_depth: int
    ask_depth: int
    total_bid_qty: int
    total_ask_qty: int


class SimulatedExchange:
    """Wraps the Rust matching engine for backtest order execution.

    Maintains per-ticker order books that can be seeded from OHLCV data
    or manually configured. Routes orders through the Rust engine for
    price-time-priority matching.

    Parameters
    ----------
    default_spread_bps : float
        Default bid-ask spread in basis points when seeding from OHLCV.
    default_depth_levels : int
        Number of price levels on each side when seeding.
    liquidity_per_level : int
        Base shares per price level (scaled by ADV if available).
    """

    def __init__(
        self,
        default_spread_bps: float = 5.0,
        default_depth_levels: int = 10,
        liquidity_per_level: int = 1000,
    ):
        if not RUST_ENGINE_AVAILABLE:
            raise ImportError(
                "Rust matching engine not available. "
                "Build it with: cd rust/matching-engine-py && maturin develop --release"
            )

        self.default_spread_bps = default_spread_bps
        self.default_depth_levels = default_depth_levels
        self.liquidity_per_level = liquidity_per_level

        self._books: dict[str, PyOrderBook] = {}
        self._next_order_id: int = 1
        self._fills: list[SimulatedFill] = []

    def _get_or_create_book(self, ticker: str) -> PyOrderBook:
        if ticker not in self._books:
            self._books[ticker] = PyOrderBook()
        return self._books[ticker]

    def _next_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_book(
        self,
        ticker: str,
        mid_price: float,
        spread_bps: float | None = None,
        depth_levels: int | None = None,
        qty_per_level: int | None = None,
        tick_size: float = 0.01,
    ) -> BookStats:
        """Seed a symmetric order book around a mid price.

        Places `depth_levels` limit orders on each side, with quantities
        that taper as you move away from the mid.

        Returns book statistics after seeding.
        """
        spread = spread_bps or self.default_spread_bps
        levels = depth_levels or self.default_depth_levels
        base_qty = qty_per_level or self.liquidity_per_level

        half_spread = mid_price * spread / 20_000  # spread_bps / 2 / 10000
        best_bid = round(mid_price - half_spread, 2)
        best_ask = round(mid_price + half_spread, 2)

        book = self._get_or_create_book(ticker)

        for i in range(levels):
            # Quantity tapers: more liquidity near the mid, less at extremes
            taper = max(1, int(base_qty * (1 - 0.5 * i / levels)))

            bid_price = round(best_bid - i * tick_size, 2)
            ask_price = round(best_ask + i * tick_size, 2)

            if bid_price > 0:
                book.submit(self._next_id(), PySide.Buy, bid_price, taper, PyOrderType.Limit)
            book.submit(self._next_id(), PySide.Sell, ask_price, taper, PyOrderType.Limit)

        return self.book_stats(ticker)

    def seed_from_ohlcv(
        self,
        ticker: str,
        close: float,
        high: float,
        low: float,
        volume: int,
        avg_price: float | None = None,
    ) -> BookStats:
        """Seed a book from a single OHLCV bar.

        Estimates spread from high-low range and scales liquidity by volume.
        """
        # Estimate spread from high-low range (Parkinson-inspired)
        if high > low and low > 0:
            hl_spread_bps = (high - low) / ((high + low) / 2) * 10_000
            # Use 10% of the H-L range as the tight spread
            spread_bps = max(1.0, min(hl_spread_bps * 0.1, 50.0))
        else:
            spread_bps = self.default_spread_bps

        # Scale liquidity by volume
        qty_per_level = max(100, volume // (self.default_depth_levels * 20))

        mid = avg_price or close

        return self.seed_book(
            ticker=ticker,
            mid_price=mid,
            spread_bps=spread_bps,
            qty_per_level=qty_per_level,
        )

    def reset_book(self, ticker: str) -> None:
        """Clear and recreate the order book for a ticker."""
        self._books[ticker] = PyOrderBook()

    def reset_all(self) -> None:
        """Clear all order books."""
        self._books.clear()

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
    ) -> list[SimulatedFill]:
        """Submit a market order. Returns list of fills (may be partial)."""
        book = self._get_or_create_book(ticker)
        py_side = PySide.Buy if side.upper() == "BUY" else PySide.Sell
        oid = self._next_id()

        execs = book.submit(oid, py_side, 0.0, quantity, PyOrderType.Market)

        fills = []
        total_filled = 0
        for e in execs:
            qty = e.quantity
            total_filled += qty
            f = SimulatedFill(
                ticker=ticker,
                side=side.upper(),
                price=e.price,
                quantity=qty,
                order_id=oid,
                is_partial=(total_filled < quantity),
            )
            fills.append(f)
            self._fills.append(f)

        if total_filled < quantity:
            logger.warning(
                f"{ticker}: market order filled {total_filled}/{quantity} "
                f"— insufficient liquidity"
            )

        return fills

    def submit_limit_order(
        self,
        ticker: str,
        side: str,
        price: float,
        quantity: int,
    ) -> list[SimulatedFill]:
        """Submit a limit order. Returns immediate fills (if any)."""
        book = self._get_or_create_book(ticker)
        py_side = PySide.Buy if side.upper() == "BUY" else PySide.Sell
        oid = self._next_id()

        execs = book.submit(oid, py_side, price, quantity, PyOrderType.Limit)

        fills = []
        for e in execs:
            f = SimulatedFill(
                ticker=ticker,
                side=side.upper(),
                price=e.price,
                quantity=e.quantity,
                order_id=oid,
            )
            fills.append(f)
            self._fills.append(f)

        return fills

    def submit_ioc_order(
        self,
        ticker: str,
        side: str,
        price: float,
        quantity: int,
    ) -> list[SimulatedFill]:
        """Submit an Immediate-Or-Cancel order."""
        book = self._get_or_create_book(ticker)
        py_side = PySide.Buy if side.upper() == "BUY" else PySide.Sell
        oid = self._next_id()

        execs = book.submit(oid, py_side, price, quantity, PyOrderType.ImmediateOrCancel)

        fills = []
        for e in execs:
            f = SimulatedFill(
                ticker=ticker,
                side=side.upper(),
                price=e.price,
                quantity=e.quantity,
                order_id=oid,
            )
            fills.append(f)
            self._fills.append(f)

        return fills

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def book_stats(self, ticker: str) -> BookStats:
        """Get summary statistics for a ticker's order book."""
        book = self._get_or_create_book(ticker)
        snap = book.snapshot(100)

        total_bid_qty = sum(q for _, q in snap.bids)
        total_ask_qty = sum(q for _, q in snap.asks)

        return BookStats(
            ticker=ticker,
            best_bid=snap.best_bid,
            best_ask=snap.best_ask,
            spread=snap.spread,
            mid_price=snap.mid_price,
            bid_depth=len(snap.bids),
            ask_depth=len(snap.asks),
            total_bid_qty=total_bid_qty,
            total_ask_qty=total_ask_qty,
        )

    def vwap(self, fills: list[SimulatedFill]) -> float:
        """Compute VWAP from a list of fills."""
        if not fills:
            return 0.0
        total_value = sum(f.price * f.quantity for f in fills)
        total_qty = sum(f.quantity for f in fills)
        return total_value / total_qty if total_qty > 0 else 0.0

    def total_cost_bps(
        self,
        fills: list[SimulatedFill],
        mid_price: float,
    ) -> float:
        """Compute total execution cost in basis points vs mid price."""
        if not fills or mid_price <= 0:
            return 0.0
        fill_vwap = self.vwap(fills)
        side = fills[0].side
        if side == "BUY":
            return (fill_vwap - mid_price) / mid_price * 10_000
        else:
            return (mid_price - fill_vwap) / mid_price * 10_000

    @property
    def all_fills(self) -> list[SimulatedFill]:
        """All fills across the exchange's lifetime."""
        return list(self._fills)
