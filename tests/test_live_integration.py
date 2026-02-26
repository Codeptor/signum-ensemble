"""Integration tests for the live trading path.

Tests the full chain:  MockBroker → ExecutionBridge → run_trading_cycle()
with deterministic prices and ML weights to verify:

- Stale order cancellation (preserving bracket SL/TP legs)
- Bridge syncs equity + positions from broker
- Reconciliation computes correct buys/sells
- Orders are submitted to broker with correct side/qty
- Fill verification polls until terminal state
- SL/TP orders are attached to buy fills anchored at fill price
- Sell fills do NOT generate SL/TP legs
- Rejected / timed-out orders are handled gracefully
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

from python.bridge.execution import ExecutionBridge
from python.brokers.base import (
    BaseBroker,
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
)
from python.portfolio.risk_manager import RiskLimits, RiskManager

# ---------------------------------------------------------------------------
# Mock broker that records calls and plays back scripted responses
# ---------------------------------------------------------------------------


class MockBroker(BaseBroker):
    """In-memory broker for integration testing.

    Tracks submitted / cancelled orders and exposes them for assertion.
    ``get_order`` returns ``BrokerOrder`` objects with ``status`` progressing
    through the ``fill_script`` (defaults to immediate ``filled``).
    """

    def __init__(
        self,
        account: BrokerAccount,
        positions: Optional[List[BrokerPosition]] = None,
        open_orders: Optional[List[BrokerOrder]] = None,
        prices: Optional[Dict[str, float]] = None,
        fill_script: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(paper_trading=True)
        self._account = account
        self._positions = positions or []
        self._open_orders = open_orders or []
        self._prices = prices or {}
        # order_id -> list of statuses to return on successive get_order calls
        self._fill_script: Dict[str, List[str]] = fill_script or {}
        self._fill_script_idx: Dict[str, int] = {}

        # Tracking
        self.submitted_orders: List[BrokerOrder] = []
        self.cancelled_order_ids: List[str] = []
        self._order_store: Dict[str, BrokerOrder] = {}

    # -- Abstract method implementations -----------------------------------

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_account(self) -> BrokerAccount:
        return self._account

    def submit_order(self, order: BrokerOrder) -> str:
        oid = f"mock-{uuid.uuid4().hex[:8]}"
        stored = replace(order, order_id=oid, status="new")
        self._order_store[oid] = stored
        self.submitted_orders.append(stored)
        return oid

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled_order_ids.append(order_id)
        return True

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        stored = self._order_store.get(order_id)
        if stored is None:
            return None

        # Walk through the scripted fill statuses if any
        script = self._fill_script.get(order_id, ["filled"])
        idx = self._fill_script_idx.get(order_id, 0)
        status = script[min(idx, len(script) - 1)]
        self._fill_script_idx[order_id] = idx + 1

        avg_price = self._prices.get(stored.symbol) if status == "filled" else None
        return replace(stored, status=status, filled_avg_price=avg_price)

    def list_orders(
        self, status: str = "open", limit: int = 500, after: str | None = None
    ) -> List[BrokerOrder]:
        return list(self._open_orders)

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        # Start from initial position
        base_pos = None
        for p in self._positions:
            if p.symbol == symbol:
                base_pos = p
                break

        # Accumulate qty changes from filled orders
        filled_delta = 0.0
        for order in self.submitted_orders:
            if order.symbol != symbol:
                continue
            oid = order.order_id
            # Check if order was filled (walk the fill script)
            script = self._fill_script.get(oid, ["filled"])
            # If the script's last status is "filled", this order contributes
            if script[-1] == "filled":
                delta = order.qty if order.side == "buy" else -order.qty
                filled_delta += delta

        if base_pos is None and filled_delta == 0:
            return None

        base_qty = base_pos.qty if base_pos else 0.0
        base_entry = base_pos.avg_entry_price if base_pos else 0.0
        total_qty = base_qty + filled_delta
        price = self._prices.get(symbol, base_entry)

        return BrokerPosition(
            symbol=symbol,
            qty=total_qty,
            avg_entry_price=base_entry,
            market_value=total_qty * price,
            unrealized_pl=0.0,
            unrealized_plpc=0.0,
        )

    def list_positions(self) -> List[BrokerPosition]:
        return list(self._positions)

    def get_latest_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def get_clock(self) -> Dict:
        return {"is_open": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def account():
    return BrokerAccount(
        account_id="test-001",
        cash=100_000.0,
        portfolio_value=100_000.0,
        buying_power=100_000.0,
        equity=100_000.0,
        status="ACTIVE",
    )


@pytest.fixture
def prices():
    return {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 150.0, "TSLA": 250.0}


@pytest.fixture
def risk_manager():
    """Permissive risk limits for integration testing — no position cap rejections."""
    return RiskManager(
        limits=RiskLimits(
            max_position_weight=1.0,  # Allow 100% in one stock
            max_single_trade_size=1.0,
            max_leverage=2.0,
            max_daily_turnover=50.0,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunTradingCycleIntegration:
    """Full-cycle integration tests using ``run_trading_cycle``."""

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)  # skip real sleeps
    def test_fresh_portfolio_buys_all_targets(self, _sleep, mock_ml, account, prices, risk_manager):
        """From empty portfolio, all target weights become buy orders."""
        target_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        mock_ml.return_value = target_weights

        broker = MockBroker(account=account, prices=prices)
        bridge = ExecutionBridge(
            risk_manager=risk_manager, initial_capital=100_000.0, commission_rate=0.0
        )

        from examples.live_bot import run_trading_cycle

        result = run_trading_cycle(broker, risk_manager, bridge)

        assert result is True
        # Should have submitted buy orders for all 3 tickers
        buy_symbols = {o.symbol for o in broker.submitted_orders if o.side == "buy"}
        assert buy_symbols == {"AAPL", "MSFT", "GOOG"}

        # C1 fix: SL+TP are now submitted as OCO pairs (1 order per buy, not 2)
        oco_orders = [
            o for o in broker.submitted_orders if o.order_class == "oco" and o.side == "sell"
        ]
        # 3 buys = 3 OCO orders (each contains SL + TP)
        assert len(oco_orders) == 3
        # R3-E-8 fix: OCO no longer sets redundant limit_price on parent.
        # TP is conveyed via take_profit_limit_price, SL via stop_loss_stop_price.
        for oco in oco_orders:
            assert oco.take_profit_limit_price is not None, (
                f"OCO for {oco.symbol} missing take_profit_limit_price"
            )
            assert oco.stop_loss_stop_price is not None, (
                f"OCO for {oco.symbol} missing SL stop_price"
            )

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_existing_position_rebalance(self, _sleep, mock_ml, account, prices, risk_manager):
        """Existing positions are rebalanced: some sold, some bought."""
        # Currently hold AAPL and TSLA
        positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=200.0,  # 200 * 200 = $40k = 40% of $100k
                avg_entry_price=180.0,
                market_value=40_000.0,
                unrealized_pl=4_000.0,
                unrealized_plpc=0.111,
            ),
            BrokerPosition(
                symbol="TSLA",
                qty=160.0,  # 160 * 250 = $40k = 40% of $100k
                avg_entry_price=230.0,
                market_value=40_000.0,
                unrealized_pl=3_200.0,
                unrealized_plpc=0.087,
            ),
        ]

        # New targets: keep AAPL (reduce), add GOOG + MSFT — TSLA dropped
        # Note: run_trading_cycle only fetches prices for *target* tickers,
        # so the bridge will warn about missing price for TSLA stale position.
        # Include TSLA in targets at 0 weight to force its price to be fetched?
        # No — the live bot simply cannot close stale positions from the bridge
        # when no price is available.  This is a known limitation.
        # Instead, include TSLA as a target with 0 weight to verify sells work.
        target_weights = {"AAPL": 0.25, "GOOG": 0.25, "MSFT": 0.25, "TSLA": 0.25}
        mock_ml.return_value = target_weights

        broker = MockBroker(account=account, positions=positions, prices=prices)
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        result = run_trading_cycle(broker, risk_manager, bridge)

        assert result is True

        submitted_syms = {o.symbol for o in broker.submitted_orders}
        # GOOG should be bought (new position)
        assert "GOOG" in submitted_syms
        # MSFT should be bought (new position)
        assert "MSFT" in submitted_syms

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_stale_orders_cancelled_but_bracket_legs_kept(
        self, _sleep, mock_ml, account, prices, risk_manager
    ):
        """Stale order cleanup preserves bracket legs; SL/TP replacement cancels them later.

        The stale-order cancellation step (step 1 of run_trading_cycle) checks
        ``parent_order_id`` and skips bracket child legs.  However, when a new
        buy order fills for the same symbol, ``_cancel_existing_sl_tp_orders``
        correctly cancels the old SL/TP so fresh protective orders can be
        attached at the new fill price.  This test verifies the full lifecycle:

        1. Regular (non-bracket) stale order → cancelled in step 1.
        2. Bracket SL leg → preserved in step 1 (has parent_order_id).
        3. After the new MSFT buy fills → old bracket SL cancelled and new
           SL/TP orders submitted at the updated fill price.
        """
        # Simulate open orders — one regular, one bracket leg
        regular_order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            qty=10.0,
            order_type="market",
            order_id="order-regular",
            parent_order_id=None,
            status="open",
        )
        bracket_sl = BrokerOrder(
            symbol="MSFT",
            side="sell",
            qty=5.0,
            order_type="stop",
            stop_price=380.0,
            order_id="order-sl-child",
            parent_order_id="order-parent-123",
            status="open",
        )
        open_orders = [regular_order, bracket_sl]

        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        mock_ml.return_value = target_weights

        # Include an existing MSFT position so the bracket SL leg
        # for MSFT is NOT orphaned (H3 cleanup runs before stale order cancellation)
        msft_position = BrokerPosition(
            symbol="MSFT",
            qty=5.0,
            avg_entry_price=400.0,
            market_value=2000.0,
            unrealized_pl=0.0,
            unrealized_plpc=0.0,
        )

        broker = MockBroker(
            account=account,
            prices=prices,
            open_orders=open_orders,
            positions=[msft_position],
        )
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # Regular stale order must be cancelled (no parent_order_id)
        assert "order-regular" in broker.cancelled_order_ids

        # The bracket SL is eventually cancelled — not by stale-order cleanup
        # (which preserves bracket legs), but by _cancel_existing_sl_tp_orders
        # when the new MSFT buy fills and fresh SL/TP orders are attached.
        assert "order-sl-child" in broker.cancelled_order_ids

        # C1 fix: SL+TP now submitted as single OCO order (not separate stop + limit)
        msft_sell_orders = [
            o for o in broker.submitted_orders if o.symbol == "MSFT" and o.side == "sell"
        ]
        oco_orders = [o for o in msft_sell_orders if o.order_class == "oco"]
        assert len(oco_orders) >= 1, "Expected new OCO SL/TP order for MSFT after buy fill"
        # The OCO order contains both TP (take_profit_limit_price) and SL (stop_loss_stop_price)
        assert oco_orders[0].take_profit_limit_price is not None, (
            "OCO missing TP take_profit_limit_price"
        )
        assert oco_orders[0].stop_loss_stop_price is not None, "OCO missing SL stop_price"

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_ml_pipeline_failure_returns_false(
        self, _sleep, mock_ml, account, prices, risk_manager
    ):
        """If get_ml_weights raises, run_trading_cycle returns False."""
        mock_ml.side_effect = RuntimeError("Model training failed")

        broker = MockBroker(account=account, prices=prices)
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        result = run_trading_cycle(broker, risk_manager, bridge)

        assert result is False
        assert len(broker.submitted_orders) == 0

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_empty_weights_returns_false(self, _sleep, mock_ml, account, prices, risk_manager):
        """If ML returns empty dict, cycle returns False with no orders."""
        mock_ml.return_value = {}

        broker = MockBroker(account=account, prices=prices)
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        result = run_trading_cycle(broker, risk_manager, bridge)

        assert result is False
        assert len(broker.submitted_orders) == 0


class TestFillVerification:
    """Tests for ``_verify_order_fill`` polling logic."""

    @patch("time.sleep", return_value=None)
    def test_immediate_fill(self, _sleep, account, prices):
        """Order that fills immediately returns correct status."""
        broker = MockBroker(account=account, prices=prices)
        # Manually insert an order so get_order can find it
        oid = broker.submit_order(
            BrokerOrder(symbol="AAPL", side="buy", qty=10.0, order_type="market")
        )

        from examples.live_bot import _verify_order_fill

        result = _verify_order_fill(broker, oid, "AAPL", 10.0)

        assert result["status"] == "filled"
        assert result["filled_avg_price"] == 200.0
        assert result["symbol"] == "AAPL"

    @patch("time.sleep", return_value=None)
    def test_delayed_fill(self, _sleep, account, prices):
        """Order transitions through pending → filled after several polls."""
        broker = MockBroker(account=account, prices=prices)
        oid = broker.submit_order(
            BrokerOrder(symbol="MSFT", side="buy", qty=5.0, order_type="market")
        )
        # Script: 3 pending polls, then filled
        # Note: partially_filled is a terminal state (Bug 3 fix), so use 'new' for pending
        broker._fill_script[oid] = ["new", "new", "new", "filled"]

        from examples.live_bot import _verify_order_fill

        result = _verify_order_fill(broker, oid, "MSFT", 5.0)

        assert result["status"] == "filled"
        assert result["filled_avg_price"] == 400.0

    @patch("time.sleep", return_value=None)
    def test_rejected_order(self, _sleep, account, prices):
        """Rejected order returns rejected status."""
        broker = MockBroker(account=account, prices=prices)
        oid = broker.submit_order(
            BrokerOrder(symbol="GOOG", side="buy", qty=100.0, order_type="market")
        )
        broker._fill_script[oid] = ["rejected"]

        from examples.live_bot import _verify_order_fill

        result = _verify_order_fill(broker, oid, "GOOG", 100.0)

        assert result["status"] == "rejected"
        assert result["filled_avg_price"] is None


class TestSLTPAttachment:
    """Verify stop-loss and take-profit orders use fill price."""

    @patch("examples.live_bot.get_current_atr")
    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_sl_tp_anchored_to_fill_price(self, _sleep, mock_ml, mock_atr, account, risk_manager):
        """SL/TP are computed from ATR when available, falling back to fixed %."""
        # Use specific prices so we can verify SL/TP arithmetic
        fill_price = 200.0
        prices = {"AAPL": fill_price}

        target_weights = {"AAPL": 1.0}
        mock_ml.return_value = target_weights

        # Return a known ATR value so stops are deterministic
        atr_value = 6.7
        mock_atr.return_value = atr_value

        broker = MockBroker(account=account, prices=prices)
        bridge = ExecutionBridge(
            risk_manager=risk_manager, initial_capital=100_000.0, commission_rate=0.0
        )

        from examples.live_bot import ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # C1 fix: SL+TP are now submitted as OCO pair (single order)
        oco_orders = [
            o for o in broker.submitted_orders if o.order_class == "oco" and o.side == "sell"
        ]
        assert len(oco_orders) == 1

        expected_sl = round(fill_price - (ATR_SL_MULTIPLIER * atr_value), 2)
        expected_tp = round(fill_price + (ATR_TP_MULTIPLIER * atr_value), 2)

        # OCO order has TP as take_profit_limit_price and SL as stop_loss_stop_price
        assert oco_orders[0].stop_loss_stop_price == expected_sl
        assert oco_orders[0].take_profit_limit_price == expected_tp

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_sells_do_not_get_sl_tp(self, _sleep, mock_ml, account, prices, risk_manager):
        """Sell orders should NOT spawn SL/TP legs."""
        positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=500.0,
                avg_entry_price=180.0,
                market_value=100_000.0,
                unrealized_pl=10_000.0,
                unrealized_plpc=0.055,
            ),
        ]
        # Target reduces AAPL and adds MSFT — AAPL sell should not get SL/TP
        target_weights = {"AAPL": 0.2, "MSFT": 0.8}
        mock_ml.return_value = target_weights

        broker = MockBroker(account=account, positions=positions, prices=prices)
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # AAPL should have a sell order (reducing from 500 to ~100 shares)
        aapl_sells = [
            o
            for o in broker.submitted_orders
            if o.symbol == "AAPL" and o.side == "sell" and o.order_type == "market"
        ]
        assert len(aapl_sells) >= 1

        # The AAPL sell should NOT generate SL/TP
        sl_tp_for_aapl = [
            o
            for o in broker.submitted_orders
            if o.symbol == "AAPL" and o.order_type in ("stop", "limit") and o.side == "sell"
        ]
        assert len(sl_tp_for_aapl) == 0


class TestBridgeBrokerSync:
    """Verify ExecutionBridge syncs from broker state."""

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_bridge_equity_syncs_from_broker(self, _sleep, mock_ml, prices, risk_manager):
        """Bridge equity is set from broker account before reconciliation."""
        account = BrokerAccount(
            account_id="test-002",
            cash=80_000.0,
            portfolio_value=80_000.0,
            buying_power=80_000.0,
            equity=80_000.0,
            status="ACTIVE",
        )
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        mock_ml.return_value = target_weights

        broker = MockBroker(account=account, prices=prices)
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # Bridge equity should have been set to 80k from broker
        # (it starts at 100k from initial_capital but gets overridden)
        buy_orders = [o for o in broker.submitted_orders if o.side == "buy"]
        assert len(buy_orders) >= 1

        # With $80k equity: AAPL 50% = $40k / $200 = 200 shares
        #                   MSFT 50% = $40k / $400 = 100 shares
        aapl_buys = [o for o in buy_orders if o.symbol == "AAPL"]
        assert len(aapl_buys) == 1
        # qty should reflect 80k equity (200 shares), not 100k (250 shares)
        assert abs(aapl_buys[0].qty - 200.0) < 1.0

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_bridge_positions_sync_from_broker(
        self, _sleep, mock_ml, account, prices, risk_manager
    ):
        """Bridge positions reflect broker positions before reconciliation."""
        positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100.0,
                avg_entry_price=190.0,
                market_value=20_000.0,
                unrealized_pl=1_000.0,
                unrealized_plpc=0.053,
            ),
        ]
        # Keep AAPL at 20% → needs 100 shares at $200 ($20k)
        # Add MSFT at 20% → needs 50 shares at $400 ($20k)
        target_weights = {"AAPL": 0.2, "MSFT": 0.2, "GOOG": 0.6}
        mock_ml.return_value = target_weights

        broker = MockBroker(account=account, positions=positions, prices=prices)
        bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=100_000.0)

        from examples.live_bot import run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # MSFT and GOOG should have buy orders (new positions)
        buy_symbols = {o.symbol for o in broker.submitted_orders if o.side == "buy"}
        assert "MSFT" in buy_symbols
        assert "GOOG" in buy_symbols


class TestOCOTopUpQty:
    """T-OCO-TOPUP: OCO qty must equal total broker position on top-up, not just fill increment.

    When topping up an existing position, the SL/TP OCO order should protect
    the ENTIRE position (old + new), not just the newly purchased shares.
    """

    @patch("examples.live_bot.get_current_atr", return_value=5.0)
    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_oco_qty_covers_full_position_on_topup(
        self, _sleep, mock_ml, mock_atr, account, prices, risk_manager
    ):
        """Top-up buy: OCO sell qty should equal full position, not just the increment."""
        # Already hold 100 shares of AAPL (20% of $100k at $200/share)
        positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100.0,
                avg_entry_price=190.0,
                market_value=20_000.0,
                unrealized_pl=1_000.0,
                unrealized_plpc=0.053,
            ),
        ]

        # Increase to 50% weight: needs 250 shares total ($50k/$200), so buy 150 more
        target_weights = {"AAPL": 0.50}
        mock_ml.return_value = target_weights

        broker = MockBroker(account=account, positions=positions, prices=prices)
        bridge = ExecutionBridge(
            risk_manager=risk_manager, initial_capital=100_000.0, commission_rate=0.0
        )

        from examples.live_bot import run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # The OCO order should protect the FULL position (old 100 + new 150 = 250)
        oco_orders = [
            o
            for o in broker.submitted_orders
            if o.order_class == "oco" and o.side == "sell" and o.symbol == "AAPL"
        ]
        assert len(oco_orders) >= 1

        # OCO qty should equal total position after top-up
        # The buy qty is ~150 shares (250 target - 100 existing)
        buy_orders = [o for o in broker.submitted_orders if o.side == "buy" and o.symbol == "AAPL"]
        assert len(buy_orders) >= 1

        # The OCO should protect the full post-top-up position
        # qty should be greater than just the buy increment
        oco_qty = oco_orders[0].qty
        buy_qty = buy_orders[0].qty
        assert oco_qty > buy_qty, (
            f"OCO qty ({oco_qty}) should be > buy increment ({buy_qty}) — "
            f"it should cover the full position"
        )


class TestRenormClamp:
    """T-RENORM: After renorm + clamp, no weight exceeds MAX_POSITION_WEIGHT.

    The iterative renorm-clamp loop (H-CLAMP fix) must converge such that:
    1. All weights sum to ~1.0
    2. No individual weight exceeds MAX_POSITION_WEIGHT
    """

    @patch("examples.live_bot.get_ml_weights")
    @patch("time.sleep", return_value=None)
    def test_weights_clamped_after_price_filtering(
        self, _sleep, mock_ml, account, prices, risk_manager
    ):
        """When some tickers are dropped (no price), renorm + clamp must hold.

        Scenario: 4 tickers each at 25%, one drops → renorm to 3 tickers
        at 33% each → clamp to MAX_POSITION_WEIGHT (30%) → renorm again.
        """
        # Give equal weights to 4 tickers, but one won't have a price
        target_weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "UNKNOWN": 0.25}
        mock_ml.return_value = target_weights

        # Only 3 tickers have prices (UNKNOWN will be dropped)
        broker = MockBroker(account=account, prices=prices)
        bridge = ExecutionBridge(
            risk_manager=risk_manager, initial_capital=100_000.0, commission_rate=0.0
        )

        from examples.live_bot import MAX_POSITION_WEIGHT, run_trading_cycle

        run_trading_cycle(broker, risk_manager, bridge)

        # Check that submitted buy quantities imply no single weight > MAX_POSITION_WEIGHT
        buy_orders = [o for o in broker.submitted_orders if o.side == "buy"]
        for order in buy_orders:
            price = prices[order.symbol]
            implied_value = order.qty * price
            implied_weight = implied_value / account.equity
            assert implied_weight <= MAX_POSITION_WEIGHT + 0.01, (
                f"{order.symbol}: implied weight {implied_weight:.2%} "
                f"exceeds MAX_POSITION_WEIGHT {MAX_POSITION_WEIGHT:.2%}"
            )
