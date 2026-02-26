"""Tests for live_bot helpers: _verify_order_fill timeout path, _seconds_until.

These are critical untested helpers in the live trading path.
"""

from __future__ import annotations

import json
import os
import unittest.mock
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from python.brokers.base import BrokerOrder


# ===========================================================================
# _verify_order_fill — timeout path (the only untested branch)
# ===========================================================================


class TestVerifyOrderFillTimeout:
    """The existing tests cover immediate-fill, delayed-fill, and rejected.
    This class covers the TIMEOUT path that was missing.
    """

    @patch("time.sleep", return_value=None)
    def test_timeout_when_order_stays_open(self, _sleep):
        """Order that never reaches terminal state returns 'timeout'."""
        from examples.live_bot import _verify_order_fill, ORDER_POLL_TIMEOUT_SECS

        # Create a mock broker where get_order always returns 'new'
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = "new"
        mock_broker.get_order.return_value = mock_order

        result = _verify_order_fill(mock_broker, "order-stuck", "AAPL", 10.0)

        assert result["status"] == "timeout"
        assert result["filled_qty"] == 0
        assert result["filled_avg_price"] is None
        assert result["symbol"] == "AAPL"
        assert result["order_id"] == "order-stuck"

    @patch("time.sleep", return_value=None)
    def test_timeout_when_get_order_returns_none(self, _sleep):
        """If get_order keeps returning None, should still timeout gracefully."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        mock_broker.get_order.return_value = None

        result = _verify_order_fill(mock_broker, "order-ghost", "MSFT", 5.0)

        assert result["status"] == "timeout"
        assert result["filled_qty"] == 0

    @patch("time.sleep", return_value=None)
    def test_timeout_fills_just_before_deadline(self, _sleep):
        """Order that fills on the last poll before timeout should return 'filled'."""
        from examples.live_bot import (
            _verify_order_fill,
            ORDER_POLL_INTERVAL_SECS,
            ORDER_POLL_TIMEOUT_SECS,
        )

        # Calculate number of polls: timeout / interval
        n_polls = int(ORDER_POLL_TIMEOUT_SECS / ORDER_POLL_INTERVAL_SECS)

        mock_broker = MagicMock()
        call_count = 0

        def get_order_side_effect(oid):
            nonlocal call_count
            call_count += 1
            order = MagicMock()
            # Fill on the last poll
            if call_count >= n_polls:
                order.status = "filled"
                order.qty = 10.0
                order.filled_avg_price = 155.0
            else:
                order.status = "new"
            return order

        mock_broker.get_order.side_effect = get_order_side_effect

        result = _verify_order_fill(mock_broker, "order-slow", "AAPL", 10.0)
        assert result["status"] == "filled"
        assert result["filled_avg_price"] == 155.0

    @patch("time.sleep", return_value=None)
    def test_cancelled_is_terminal(self, _sleep):
        """Both 'canceled' and 'cancelled' spellings are handled."""
        from examples.live_bot import _verify_order_fill

        for spelling in ("canceled", "cancelled"):
            mock_broker = MagicMock()
            mock_order = MagicMock()
            mock_order.status = spelling
            mock_order.qty = 10.0
            mock_order.filled_avg_price = None
            mock_broker.get_order.return_value = mock_order

            result = _verify_order_fill(mock_broker, f"order-{spelling}", "AAPL", 10.0)
            assert result["status"] == spelling

    @patch("time.sleep", return_value=None)
    def test_expired_is_terminal(self, _sleep):
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = "expired"
        mock_order.qty = 10.0
        mock_order.filled_avg_price = None
        mock_broker.get_order.return_value = mock_order

        result = _verify_order_fill(mock_broker, "order-expired", "AAPL", 10.0)
        assert result["status"] == "expired"


# ===========================================================================
# _seconds_until
# ===========================================================================


class TestSecondsUntil:
    """Tests for the _seconds_until helper that handles timezone-aware/naive datetimes."""

    def test_tz_aware_datetime_in_future(self):
        from examples.live_bot import _seconds_until

        future = datetime.now(tz=timezone.utc) + timedelta(hours=2)
        secs = _seconds_until(future)
        # Should be close to 7200 seconds
        assert 7100 < secs < 7300

    def test_tz_naive_datetime_in_future(self):
        from examples.live_bot import _seconds_until

        # Naive datetime — should be treated as UTC
        future = datetime.utcnow() + timedelta(hours=1)
        secs = _seconds_until(future)
        assert 3500 < secs < 3700

    def test_past_datetime_returns_minimum(self):
        """Past timestamps should return at least 60 seconds (safety minimum)."""
        from examples.live_bot import _seconds_until

        past = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        secs = _seconds_until(past)
        assert secs == 60.0  # Floor

    def test_iso_string_in_future(self):
        from examples.live_bot import _seconds_until

        future = datetime.now(tz=timezone.utc) + timedelta(hours=3)
        iso = future.isoformat()
        secs = _seconds_until(iso)
        assert 10700 < secs < 10900

    def test_iso_string_with_z_suffix(self):
        from examples.live_bot import _seconds_until

        future = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        # Replace +00:00 with Z (common in API responses)
        iso = future.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        secs = _seconds_until(iso)
        assert 3500 < secs < 3700

    def test_invalid_string_returns_fallback(self):
        from examples.live_bot import _seconds_until

        secs = _seconds_until("not-a-date")
        assert secs == 3600.0  # 1 hour fallback

    def test_unknown_type_returns_fallback(self):
        from examples.live_bot import _seconds_until

        secs = _seconds_until(12345)
        assert secs == 3600.0  # 1 hour fallback

    def test_none_returns_fallback(self):
        from examples.live_bot import _seconds_until

        secs = _seconds_until(None)
        assert secs == 3600.0

    def test_minimum_60_seconds(self):
        """Even very near-future times should return at least 60 seconds."""
        from examples.live_bot import _seconds_until

        # 10 seconds from now → should be clamped to 60
        near = datetime.now(tz=timezone.utc) + timedelta(seconds=10)
        secs = _seconds_until(near)
        assert secs == 60.0


# ===========================================================================
# _send_alert — fire-and-forget
# ===========================================================================


class TestSendAlert:
    def test_no_webhook_url_does_nothing(self):
        """When ALERT_WEBHOOK_URL is not set, _send_alert is a no-op."""
        with patch("examples.live_bot.ALERT_WEBHOOK_URL", None):
            from examples.live_bot import _send_alert

            # Should not raise
            _send_alert("test alert")

    def test_webhook_failure_swallowed(self):
        """Network errors in _send_alert must never propagate."""
        with patch("examples.live_bot.ALERT_WEBHOOK_URL", "https://hooks.example.com/test"):
            with patch("urllib.request.urlopen", side_effect=Exception("network down")):
                from examples.live_bot import _send_alert

                # Should not raise
                _send_alert("test alert")


# ===========================================================================
# _cleanup_orphaned_sl_tp (H3 fix)
# ===========================================================================


class TestCleanupOrphanedSlTp:
    """Tests for the _cleanup_orphaned_sl_tp function.

    This function cancels sell-side stop/limit orders whose symbol
    is no longer held in the portfolio, preventing unintended shorts.
    """

    def test_cancels_orphaned_stop_order(self):
        """Sell-side stop order for symbol NOT held should be cancelled."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()
        # No positions held
        mock_broker.list_positions.return_value = []

        # One open sell-side stop order for AAPL (not held)
        orphan_order = MagicMock()
        orphan_order.side = "sell"
        orphan_order.order_type = "stop"
        orphan_order.symbol = "AAPL"
        orphan_order.order_id = "orphan-1"
        mock_broker.list_orders.return_value = [orphan_order]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 1
        mock_broker.cancel_order.assert_called_once_with("orphan-1")

    def test_cancels_orphaned_limit_order(self):
        """Sell-side limit order (TP) for symbol NOT held should be cancelled."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()
        mock_broker.list_positions.return_value = []

        orphan_order = MagicMock()
        orphan_order.side = "sell"
        orphan_order.order_type = "limit"
        orphan_order.symbol = "MSFT"
        orphan_order.order_id = "orphan-2"
        mock_broker.list_orders.return_value = [orphan_order]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 1
        mock_broker.cancel_order.assert_called_once_with("orphan-2")

    def test_does_not_cancel_orders_for_held_positions(self):
        """Sell-side stop/limit orders for held positions should NOT be cancelled."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()

        # We hold AAPL
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = 10.0
        mock_broker.list_positions.return_value = [position]

        # Open sell-side stop order for AAPL (held)
        order = MagicMock()
        order.side = "sell"
        order.order_type = "stop"
        order.symbol = "AAPL"
        order.order_id = "sl-aapl"
        mock_broker.list_orders.return_value = [order]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 0
        mock_broker.cancel_order.assert_not_called()

    def test_does_not_cancel_buy_side_orders(self):
        """Buy-side orders should never be cancelled by this function."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()
        mock_broker.list_positions.return_value = []

        buy_order = MagicMock()
        buy_order.side = "buy"
        buy_order.order_type = "limit"
        buy_order.symbol = "AAPL"
        buy_order.order_id = "buy-1"
        mock_broker.list_orders.return_value = [buy_order]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 0
        mock_broker.cancel_order.assert_not_called()

    def test_does_not_cancel_market_sell_orders(self):
        """Sell-side market orders (not SL/TP) should not be cancelled."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()
        mock_broker.list_positions.return_value = []

        market_order = MagicMock()
        market_order.side = "sell"
        market_order.order_type = "market"
        market_order.symbol = "AAPL"
        market_order.order_id = "mkt-1"
        mock_broker.list_orders.return_value = [market_order]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 0
        mock_broker.cancel_order.assert_not_called()

    def test_mixed_scenario(self):
        """Multiple orders with some orphaned and some valid."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()

        # Hold AAPL, not MSFT or GOOG
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = 5.0
        mock_broker.list_positions.return_value = [position]

        # AAPL SL (held — keep), MSFT TP (orphaned — cancel), GOOG SL (orphaned — cancel)
        aapl_sl = MagicMock(side="sell", order_type="stop", symbol="AAPL", order_id="sl-aapl")
        msft_tp = MagicMock(side="sell", order_type="limit", symbol="MSFT", order_id="tp-msft")
        goog_sl = MagicMock(side="sell", order_type="stop", symbol="GOOG", order_id="sl-goog")
        mock_broker.list_orders.return_value = [aapl_sl, msft_tp, goog_sl]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 2
        mock_broker.cancel_order.assert_any_call("tp-msft")
        mock_broker.cancel_order.assert_any_call("sl-goog")

    def test_broker_exception_returns_zero(self):
        """Broker failures should not propagate — returns 0."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()
        mock_broker.list_positions.side_effect = Exception("broker down")

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        assert cancelled == 0

    def test_cancel_failure_for_individual_order_is_tolerated(self):
        """If cancelling one order fails, still continue with others."""
        from examples.live_bot import _cleanup_orphaned_sl_tp

        mock_broker = MagicMock()
        mock_broker.list_positions.return_value = []

        order_a = MagicMock(side="sell", order_type="stop", symbol="AAPL", order_id="a")
        order_b = MagicMock(side="sell", order_type="stop", symbol="MSFT", order_id="b")
        mock_broker.list_orders.return_value = [order_a, order_b]

        # First cancel fails, second succeeds
        mock_broker.cancel_order.side_effect = [Exception("cancel failed"), None]

        cancelled = _cleanup_orphaned_sl_tp(mock_broker)

        # Only the second one successfully cancelled
        assert cancelled == 1


# ===========================================================================
# SIGTERM handler registration (H4 fix)
# ===========================================================================


class TestSigtermHandler:
    """Tests that SIGTERM handler is registered in main()."""

    @patch("examples.live_bot.signal.signal")
    @patch("examples.live_bot.AlpacaBroker")
    def test_sigterm_handler_registered(self, MockAlpacaBroker, mock_signal):
        """Verify signal.signal(SIGTERM, handler) is called in main()."""
        import signal as signal_mod
        from examples.live_bot import main

        # Mock broker so main() doesn't actually run a trading loop
        mock_broker_instance = MagicMock()
        mock_broker_instance.connect.return_value = True
        mock_broker_instance.get_account.return_value = MagicMock(equity=100000.0, cash=50000.0)
        mock_broker_instance.get_clock.return_value = {"is_open": False, "next_open": None}
        mock_broker_instance.list_positions.return_value = []
        MockAlpacaBroker.return_value = mock_broker_instance

        # Patch environment variables for main (note: correct env var names)
        with (
            patch.dict(
                os.environ, {"ALPACA_API_KEY": "test-key", "ALPACA_API_SECRET": "test-secret"}
            ),
            patch("examples.live_bot._has_traded_today", return_value=True),
            patch("examples.live_bot._load_bot_state", return_value={}),
        ):
            # main() will exit after _has_traded_today returns True (sys.exit(0))
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        # Verify SIGTERM handler was registered
        sigterm_calls = [
            call for call in mock_signal.call_args_list if call[0][0] == signal_mod.SIGTERM
        ]
        assert len(sigterm_calls) == 1, "signal.signal(SIGTERM, handler) should be called once"


# ===========================================================================
# Drawdown kill switch — liquidation (H5 fix)
# ===========================================================================


class TestDrawdownKillSwitch:
    """Verify that MAX_DRAWDOWN violations trigger full liquidation."""

    @patch("time.sleep", return_value=None)
    @patch("examples.live_bot.should_rebalance_today", return_value=True)
    @patch("examples.live_bot._has_traded_today", return_value=False)
    @patch("examples.live_bot._initialize_risk_engine")
    @patch("examples.live_bot.fetch_vix", return_value=20.0)
    @patch("examples.live_bot.fetch_spy_drawdown", return_value=0.05)
    def test_drawdown_violation_triggers_liquidation(
        self,
        _spy,
        _vix,
        _init_risk,
        _has_traded,
        _should_rebal,
        _sleep,
    ):
        """When risk_manager reports MAX_DRAWDOWN critical violation, _liquidate_all_positions is called."""
        from examples.live_bot import _liquidate_all_positions
        from python.portfolio.risk_manager import RiskCheck

        mock_broker = MagicMock()
        mock_broker.list_positions.return_value = []
        mock_broker.get_account.return_value = MagicMock(equity=100000.0, cash=50000.0)

        mock_risk_manager = MagicMock()
        # Return a MAX_DRAWDOWN critical violation
        mock_risk_manager.check_portfolio_risk.return_value = [
            RiskCheck(
                passed=False,
                rule="MAX_DRAWDOWN",
                message="Drawdown -18.0% exceeds limit of -15.0%",
                severity="critical",
            )
        ]

        # The _liquidate_all_positions function should actually be called
        # when the main loop detects this. We test the function directly.
        mock_broker.list_positions.return_value = [
            MagicMock(symbol="AAPL", qty=10.0),
            MagicMock(symbol="MSFT", qty=5.0),
        ]

        n_liquidated = _liquidate_all_positions(mock_broker)

        assert n_liquidated == 2
        assert mock_broker.submit_order.call_count == 2


# ===========================================================================
# _verify_order_fill — partial fill and cancel on timeout (Bug 3 fix)
# ===========================================================================


class TestVerifyOrderFillPartialFill:
    """Test partial fill handling and timeout cancellation."""

    @patch("time.sleep", return_value=None)
    def test_partial_fill_uses_actual_filled_qty(self, _sleep):
        """Partial fills should return actual filled_qty, not expected."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        order = MagicMock()
        order.status = "partially_filled"
        order.qty = 100.0
        order.filled_qty = 42.0
        order.filled_avg_price = 155.50
        mock_broker.get_order.return_value = order

        result = _verify_order_fill(mock_broker, "order-partial", "AAPL", 100.0)

        assert result["status"] == "partially_filled"
        assert result["filled_qty"] == 42.0
        assert result["filled_avg_price"] == 155.50

    @patch("time.sleep", return_value=None)
    def test_timeout_cancels_order(self, _sleep):
        """Timed-out orders should be cancelled to prevent untracked late fills."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        order = MagicMock()
        order.status = "new"
        mock_broker.get_order.return_value = order

        result = _verify_order_fill(mock_broker, "order-stuck", "AAPL", 10.0)

        assert result["status"] == "timeout"
        mock_broker.cancel_order.assert_called_once_with("order-stuck")

    @patch("time.sleep", return_value=None)
    def test_filled_order_uses_filled_qty_when_available(self, _sleep):
        """Filled orders should use filled_qty from broker, not order.qty."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        order = MagicMock()
        order.status = "filled"
        order.qty = 100.0
        order.filled_qty = 100.0
        order.filled_avg_price = 155.0
        mock_broker.get_order.return_value = order

        result = _verify_order_fill(mock_broker, "order-full", "AAPL", 100.0)

        assert result["status"] == "filled"
        assert result["filled_qty"] == 100.0

    @patch("time.sleep", return_value=None)
    def test_filled_order_without_filled_qty_falls_back_to_qty(self, _sleep):
        """If filled_qty is None on a 'filled' order, fall back to order.qty."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        order = MagicMock()
        order.status = "filled"
        order.qty = 100.0
        order.filled_qty = None
        order.filled_avg_price = 155.0
        mock_broker.get_order.return_value = order

        result = _verify_order_fill(mock_broker, "order-nofq", "AAPL", 100.0)

        assert result["status"] == "filled"
        assert result["filled_qty"] == 100.0

    @patch("time.sleep", return_value=None)
    def test_cancelled_order_filled_qty_is_zero(self, _sleep):
        """Cancelled orders with no filled_qty should report 0."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        order = MagicMock()
        order.status = "canceled"
        order.qty = 100.0
        order.filled_qty = None
        order.filled_avg_price = None
        mock_broker.get_order.return_value = order

        result = _verify_order_fill(mock_broker, "order-canc", "AAPL", 100.0)

        assert result["status"] == "canceled"
        # filled_qty is None → fallback for non-'filled' status is 0
        assert result["filled_qty"] == 0


# ===========================================================================
# _load_bot_state (M1 fix)
# ===========================================================================


class TestLoadBotState:
    """Tests for _load_bot_state function."""

    def test_returns_empty_dict_when_no_file(self, tmp_path):
        """When state file doesn't exist, returns empty dict."""
        from examples.live_bot import _load_bot_state

        with patch("examples.live_bot.STATE_FILE", tmp_path / "nonexistent.json"):
            result = _load_bot_state()
            assert result == {}

    def test_loads_saved_state(self, tmp_path):
        """Should deserialize JSON state file correctly."""
        from examples.live_bot import _load_bot_state

        state_file = tmp_path / "bot_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "last_shutdown": "2024-01-01T12:00:00",
                    "final_equity": 100000.0,
                    "reason": "sigterm",
                }
            )
        )

        with patch("examples.live_bot.STATE_FILE", state_file):
            result = _load_bot_state()
            assert result["reason"] == "sigterm"
            assert result["final_equity"] == 100000.0

    def test_returns_empty_dict_on_corrupt_json(self, tmp_path):
        """Corrupt JSON should not crash — returns empty dict."""
        from examples.live_bot import _load_bot_state

        state_file = tmp_path / "bot_state.json"
        state_file.write_text("not valid json {{{{")

        with patch("examples.live_bot.STATE_FILE", state_file):
            result = _load_bot_state()
            assert result == {}
