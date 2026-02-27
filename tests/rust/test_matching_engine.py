"""Tests for the Rust matching engine Python bridge (PyO3)."""

import pytest

from matching_engine_py import PyExecution, PyL2Snapshot, PyOrderBook, PyOrderType, PySide


# ---------------------------------------------------------------------------
# Order book basics
# ---------------------------------------------------------------------------


class TestOrderBookBasics:
    def test_empty_book(self):
        book = PyOrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None

    def test_single_bid(self):
        book = PyOrderBook()
        execs = book.submit(1, PySide.Buy, 100.0, 10, PyOrderType.Limit)
        assert len(execs) == 0
        assert book.best_bid == 100.0
        assert book.best_ask is None

    def test_single_ask(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 105.0, 10, PyOrderType.Limit)
        assert book.best_ask == 105.0
        assert book.best_bid is None

    def test_spread(self):
        book = PyOrderBook()
        book.submit(1, PySide.Buy, 99.0, 10, PyOrderType.Limit)
        book.submit(2, PySide.Sell, 101.0, 10, PyOrderType.Limit)
        assert book.spread == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_exact_match(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 10, PyOrderType.Limit)
        assert len(execs) == 1
        assert execs[0].quantity == 10
        assert execs[0].price == pytest.approx(100.0)

    def test_partial_fill(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 5, PyOrderType.Limit)
        assert len(execs) == 1
        assert execs[0].quantity == 5
        # Remaining ask should still be there
        assert book.best_ask == 100.0

    def test_price_improvement(self):
        """Buy at 102 should match sell at 100 (price improvement)."""
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 102.0, 10, PyOrderType.Limit)
        assert len(execs) == 1
        assert execs[0].price == pytest.approx(100.0)  # Executes at resting price

    def test_price_time_priority(self):
        """Earlier order at same price gets filled first."""
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        book.submit(2, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(3, PySide.Buy, 100.0, 5, PyOrderType.Limit)
        assert execs[0].sell_order_id == 1

    def test_multi_level_fill(self):
        """Aggressive buy sweeps through multiple price levels."""
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 5, PyOrderType.Limit)
        book.submit(2, PySide.Sell, 101.0, 5, PyOrderType.Limit)
        execs = book.submit(3, PySide.Buy, 101.0, 8, PyOrderType.Limit)
        assert len(execs) == 2
        assert execs[0].price == pytest.approx(100.0)
        assert execs[0].quantity == 5
        assert execs[1].price == pytest.approx(101.0)
        assert execs[1].quantity == 3

    def test_execution_fields(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 10, PyOrderType.Limit)
        e = execs[0]
        assert isinstance(e, PyExecution)
        assert e.buy_order_id == 2
        assert e.sell_order_id == 1
        assert e.timestamp > 0


# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------


class TestOrderTypes:
    def test_market_order(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 5, PyOrderType.Limit)
        book.submit(2, PySide.Sell, 101.0, 5, PyOrderType.Limit)
        execs = book.submit(3, PySide.Buy, 0.0, 8, PyOrderType.Market)
        assert len(execs) == 2
        total = sum(e.quantity for e in execs)
        assert total == 8

    def test_ioc_partial(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 5, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 10, PyOrderType.ImmediateOrCancel)
        assert len(execs) == 1
        assert execs[0].quantity == 5
        # Unfilled remainder should NOT rest on the book
        assert book.best_bid is None

    def test_fok_rejected(self):
        """FOK should be rejected if not enough liquidity."""
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 5, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 10, PyOrderType.FillOrKill)
        assert len(execs) == 0
        # Original sell should remain
        assert book.best_ask == 100.0

    def test_fok_filled(self):
        """FOK should fill completely if liquidity is sufficient."""
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 10, PyOrderType.FillOrKill)
        assert len(execs) == 1
        assert execs[0].quantity == 10


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_existing(self):
        book = PyOrderBook()
        book.submit(1, PySide.Buy, 99.0, 10, PyOrderType.Limit)
        assert book.cancel(1) is True
        assert book.best_bid is None

    def test_cancel_nonexistent(self):
        book = PyOrderBook()
        assert book.cancel(999) is False


# ---------------------------------------------------------------------------
# L2 Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self):
        book = PyOrderBook()
        snap = book.snapshot(5)
        assert isinstance(snap, PyL2Snapshot)
        assert len(snap.bids) == 0
        assert len(snap.asks) == 0

    def test_snapshot_levels(self):
        book = PyOrderBook()
        book.submit(1, PySide.Buy, 99.0, 10, PyOrderType.Limit)
        book.submit(2, PySide.Buy, 98.0, 20, PyOrderType.Limit)
        book.submit(3, PySide.Sell, 101.0, 15, PyOrderType.Limit)
        snap = book.snapshot(5)
        assert len(snap.bids) == 2
        assert len(snap.asks) == 1
        assert snap.bids[0] == (99.0, 10)  # Best bid first
        assert snap.asks[0] == (101.0, 15)

    def test_snapshot_aggregates_quantity(self):
        """Multiple orders at same price should aggregate."""
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        book.submit(2, PySide.Sell, 100.0, 20, PyOrderType.Limit)
        snap = book.snapshot(5)
        assert snap.asks[0] == (100.0, 30)

    def test_snapshot_best_bid_ask(self):
        book = PyOrderBook()
        book.submit(1, PySide.Buy, 99.0, 10, PyOrderType.Limit)
        book.submit(2, PySide.Sell, 101.0, 10, PyOrderType.Limit)
        snap = book.snapshot(5)
        assert snap.best_bid == 99.0
        assert snap.best_ask == 101.0
        assert snap.spread == pytest.approx(2.0)
        assert snap.mid_price == pytest.approx(100.0)

    def test_snapshot_depth_limit(self):
        book = PyOrderBook()
        for i in range(10):
            book.submit(i + 1, PySide.Buy, 90.0 + i, 10, PyOrderType.Limit)
        snap = book.snapshot(3)
        assert len(snap.bids) == 3


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_execution_repr(self):
        book = PyOrderBook()
        book.submit(1, PySide.Sell, 100.0, 10, PyOrderType.Limit)
        execs = book.submit(2, PySide.Buy, 100.0, 10, PyOrderType.Limit)
        assert "Execution" in repr(execs[0])

    def test_book_repr(self):
        book = PyOrderBook()
        assert "OrderBook" in repr(book)

    def test_snapshot_repr(self):
        book = PyOrderBook()
        snap = book.snapshot(5)
        assert "L2Snapshot" in repr(snap)
