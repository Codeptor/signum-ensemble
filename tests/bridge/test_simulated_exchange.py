"""Tests for the simulated exchange (Rust matching engine integration)."""

import pytest

from python.bridge.simulated_exchange import (
    RUST_ENGINE_AVAILABLE,
    BookStats,
    SimulatedExchange,
    SimulatedFill,
)

pytestmark = pytest.mark.skipif(
    not RUST_ENGINE_AVAILABLE,
    reason="Rust matching engine not installed",
)


@pytest.fixture
def exchange():
    return SimulatedExchange()


@pytest.fixture
def seeded_exchange(exchange):
    """Exchange with AAPL seeded at $180."""
    exchange.seed_book("AAPL", mid_price=180.0, spread_bps=5.0, qty_per_level=500)
    return exchange


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_seed_creates_book(self, exchange):
        stats = exchange.seed_book("AAPL", mid_price=180.0)
        assert isinstance(stats, BookStats)
        assert stats.ticker == "AAPL"
        assert stats.best_bid is not None
        assert stats.best_ask is not None

    def test_spread_matches(self, exchange):
        stats = exchange.seed_book("AAPL", mid_price=100.0, spread_bps=10.0)
        # 10 bps = 0.10 on $100, half-spread = 0.05
        assert stats.spread is not None
        assert stats.spread < 0.20  # should be around 0.10

    def test_depth_levels(self, exchange):
        stats = exchange.seed_book("AAPL", mid_price=100.0, depth_levels=5)
        assert stats.bid_depth == 5
        assert stats.ask_depth == 5

    def test_has_liquidity(self, exchange):
        stats = exchange.seed_book("AAPL", mid_price=100.0, qty_per_level=1000)
        assert stats.total_bid_qty > 0
        assert stats.total_ask_qty > 0

    def test_seed_from_ohlcv(self, exchange):
        stats = exchange.seed_from_ohlcv(
            "MSFT", close=400.0, high=405.0, low=395.0, volume=30_000_000
        )
        assert stats.best_bid is not None
        assert stats.best_ask is not None
        assert stats.total_bid_qty > 0

    def test_reset_book(self, seeded_exchange):
        seeded_exchange.reset_book("AAPL")
        stats = seeded_exchange.book_stats("AAPL")
        assert stats.best_bid is None
        assert stats.best_ask is None

    def test_reset_all(self, seeded_exchange):
        seeded_exchange.seed_book("MSFT", mid_price=400.0)
        seeded_exchange.reset_all()
        # Both books should be gone
        assert "AAPL" not in seeded_exchange._books
        assert "MSFT" not in seeded_exchange._books


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------


class TestMarketOrders:
    def test_buy_fills(self, seeded_exchange):
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 100)
        assert len(fills) >= 1
        total_qty = sum(f.quantity for f in fills)
        assert total_qty == 100

    def test_sell_fills(self, seeded_exchange):
        fills = seeded_exchange.submit_market_order("AAPL", "SELL", 100)
        assert len(fills) >= 1
        total_qty = sum(f.quantity for f in fills)
        assert total_qty == 100

    def test_buy_price_above_mid(self, seeded_exchange):
        stats = seeded_exchange.book_stats("AAPL")
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 50)
        vwap = seeded_exchange.vwap(fills)
        # Should buy at ask or higher
        assert vwap >= stats.best_bid

    def test_large_order_sweeps_levels(self, seeded_exchange):
        """Large order should fill across multiple price levels."""
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 2000)
        if len(fills) > 1:
            prices = [f.price for f in fills]
            # Prices should be non-decreasing for buys (sweeping asks)
            assert prices == sorted(prices)

    def test_fill_fields(self, seeded_exchange):
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 10)
        f = fills[0]
        assert isinstance(f, SimulatedFill)
        assert f.ticker == "AAPL"
        assert f.side == "BUY"
        assert f.quantity > 0
        assert f.price > 0
        assert f.order_id > 0


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


class TestLimitOrders:
    def test_aggressive_limit_fills(self, seeded_exchange):
        """Limit buy above best ask should fill immediately."""
        stats = seeded_exchange.book_stats("AAPL")
        fills = seeded_exchange.submit_limit_order("AAPL", "BUY", stats.best_ask + 1.0, 50)
        assert len(fills) >= 1

    def test_passive_limit_rests(self, seeded_exchange):
        """Limit buy below best bid should not fill."""
        stats = seeded_exchange.book_stats("AAPL")
        fills = seeded_exchange.submit_limit_order("AAPL", "BUY", stats.best_bid - 1.0, 50)
        assert len(fills) == 0


# ---------------------------------------------------------------------------
# IOC orders
# ---------------------------------------------------------------------------


class TestIOCOrders:
    def test_ioc_fills_available(self, seeded_exchange):
        stats = seeded_exchange.book_stats("AAPL")
        fills = seeded_exchange.submit_ioc_order("AAPL", "BUY", stats.best_ask, 50)
        assert len(fills) >= 1

    def test_ioc_no_resting(self, seeded_exchange):
        """IOC unfilled portion should not rest on book."""
        stats_before = seeded_exchange.book_stats("AAPL")
        bid_qty_before = stats_before.total_bid_qty
        # IOC buy below bid — nothing to fill, nothing should rest
        seeded_exchange.submit_ioc_order("AAPL", "BUY", stats_before.best_bid - 1.0, 100)
        stats_after = seeded_exchange.book_stats("AAPL")
        assert stats_after.total_bid_qty == bid_qty_before


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_vwap(self, seeded_exchange):
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 100)
        vwap = seeded_exchange.vwap(fills)
        assert vwap > 0

    def test_vwap_empty(self, exchange):
        assert exchange.vwap([]) == 0.0

    def test_total_cost_bps(self, seeded_exchange):
        stats = seeded_exchange.book_stats("AAPL")
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 100)
        cost = seeded_exchange.total_cost_bps(fills, stats.mid_price)
        # Cost should be positive for buys (paying more than mid)
        assert cost >= 0

    def test_all_fills_accumulate(self, seeded_exchange):
        seeded_exchange.submit_market_order("AAPL", "BUY", 10)
        seeded_exchange.submit_market_order("AAPL", "SELL", 10)
        assert len(seeded_exchange.all_fills) >= 2

    def test_book_stats(self, seeded_exchange):
        stats = seeded_exchange.book_stats("AAPL")
        assert stats.mid_price is not None
        assert stats.mid_price > 0
        assert stats.spread > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_book_market_order(self, exchange):
        """Market order on empty book should produce no fills."""
        fills = exchange.submit_market_order("AAPL", "BUY", 100)
        assert len(fills) == 0

    def test_multiple_tickers(self, exchange):
        exchange.seed_book("AAPL", mid_price=180.0)
        exchange.seed_book("MSFT", mid_price=400.0)
        fills_a = exchange.submit_market_order("AAPL", "BUY", 10)
        fills_m = exchange.submit_market_order("MSFT", "BUY", 10)
        assert fills_a[0].ticker == "AAPL"
        assert fills_m[0].ticker == "MSFT"

    def test_fill_value(self, seeded_exchange):
        fills = seeded_exchange.submit_market_order("AAPL", "BUY", 10)
        assert fills[0].value > 0
        assert fills[0].value == fills[0].price * fills[0].quantity
