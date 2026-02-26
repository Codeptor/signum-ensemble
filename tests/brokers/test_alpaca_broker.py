"""Tests for AlpacaBroker with fully mocked Alpaca REST API.

Validates:
- Constructor (credentials, paper/live URL, missing creds)
- connect / disconnect / is_connected state machine
- _make_client_order_id determinism and idempotency
- submit_order: simple market, limit, bracket with SL/TP
- submit_order: auto-generates client_order_id when missing
- cancel_order / cancel_all_orders
- get_order / list_orders mapping to BrokerOrder
- get_position / list_positions mapping to BrokerPosition
- get_account mapping to BrokerAccount
- get_latest_price: Alpaca primary, yfinance fallback
- get_latest_prices: batch with partial Alpaca miss → yfinance
- get_clock / get_bars
- Error handling: not-connected guard on every method
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from python.brokers.base import BrokerAccount, BrokerOrder, BrokerPosition

# ---------------------------------------------------------------------------
# Helpers to build a fake alpaca_trade_api module
# ---------------------------------------------------------------------------


def _make_alpaca_account(**overrides):
    defaults = {
        "id": "acc-001",
        "cash": "50000.00",
        "portfolio_value": "100000.00",
        "buying_power": "100000.00",
        "equity": "100000.00",
        "status": "ACTIVE",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_alpaca_order(**overrides):
    defaults = {
        "id": "order-abc",
        "symbol": "AAPL",
        "side": "buy",
        "qty": "10",
        "type": "market",
        "limit_price": None,
        "stop_price": None,
        "time_in_force": "day",
        "client_order_id": "client-123",
        "parent_order_id": None,
        "status": "filled",
        "order_class": None,
        "filled_avg_price": "150.50",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_alpaca_position(**overrides):
    defaults = {
        "symbol": "AAPL",
        "qty": "100",
        "avg_entry_price": "145.00",
        "market_value": "15050.00",
        "unrealized_pl": "550.00",
        "unrealized_plpc": "0.0379",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_alpaca_trade(price: float = 150.0):
    return SimpleNamespace(price=price)


def _make_alpaca_clock(is_open=True):
    return SimpleNamespace(
        timestamp="2026-02-26T10:00:00-05:00",
        is_open=is_open,
        next_open="2026-02-27T09:30:00-05:00",
        next_close="2026-02-26T16:00:00-05:00",
    )


def _make_alpaca_bar(t="2026-02-25", o=145, h=152, lo=144, c=150, v=1_000_000):
    return SimpleNamespace(t=t, o=o, h=h, l=lo, c=c, v=v)


# ---------------------------------------------------------------------------
# Fixture: Create AlpacaBroker with mocked REST client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tradeapi():
    """Patch the alpaca_trade_api import inside alpaca_broker.py."""
    mock_module = MagicMock()
    mock_rest = MagicMock()
    mock_module.REST.return_value = mock_rest
    return mock_module, mock_rest


@pytest.fixture
def broker(mock_tradeapi):
    """Create an AlpacaBroker with mocked REST client, already connected."""
    mock_module, mock_rest = mock_tradeapi
    mock_rest.get_account.return_value = _make_alpaca_account()

    with patch.dict("sys.modules", {"alpaca_trade_api": mock_module}):
        from python.brokers.alpaca_broker import AlpacaBroker

        b = AlpacaBroker(
            paper_trading=True,
            api_key="test-key",
            api_secret="test-secret",
        )
        b.connect()  # sets _connected = True
    return b, mock_rest


@pytest.fixture
def disconnected_broker(mock_tradeapi):
    """Create an AlpacaBroker that is NOT connected."""
    mock_module, mock_rest = mock_tradeapi

    with patch.dict("sys.modules", {"alpaca_trade_api": mock_module}):
        from python.brokers.alpaca_broker import AlpacaBroker

        b = AlpacaBroker(
            paper_trading=True,
            api_key="test-key",
            api_secret="test-secret",
        )
    return b, mock_rest


# ===========================================================================
# Constructor
# ===========================================================================


class TestAlpacaBrokerConstructor:
    def test_paper_trading_url(self, mock_tradeapi):
        mock_module, mock_rest = mock_tradeapi
        with patch.dict("sys.modules", {"alpaca_trade_api": mock_module}):
            from python.brokers.alpaca_broker import AlpacaBroker

            AlpacaBroker(paper_trading=True, api_key="k", api_secret="s")
            mock_module.REST.assert_called_once_with(
                "k", "s", "https://paper-api.alpaca.markets", api_version="v2"
            )

    def test_live_trading_url(self, mock_tradeapi):
        mock_module, mock_rest = mock_tradeapi
        with patch.dict("sys.modules", {"alpaca_trade_api": mock_module}):
            from python.brokers.alpaca_broker import AlpacaBroker

            AlpacaBroker(paper_trading=False, api_key="k", api_secret="s")
            mock_module.REST.assert_called_once_with(
                "k", "s", "https://api.alpaca.markets", api_version="v2"
            )

    def test_missing_credentials_raises(self, mock_tradeapi):
        mock_module, _ = mock_tradeapi
        with patch.dict("sys.modules", {"alpaca_trade_api": mock_module}):
            with patch.dict("os.environ", {}, clear=True):
                from python.brokers.alpaca_broker import AlpacaBroker

                with pytest.raises(ValueError, match="API credentials required"):
                    AlpacaBroker(paper_trading=True, api_key=None, api_secret=None)

    def test_env_var_fallback(self, mock_tradeapi):
        mock_module, _ = mock_tradeapi
        with patch.dict("sys.modules", {"alpaca_trade_api": mock_module}):
            import os

            with patch.dict(
                os.environ,
                {"ALPACA_API_KEY": "env-key", "ALPACA_API_SECRET": "env-secret"},
            ):
                from python.brokers.alpaca_broker import AlpacaBroker

                b = AlpacaBroker(paper_trading=True)
                assert b.api_key == "env-key"
                assert b.api_secret == "env-secret"


# ===========================================================================
# Connection lifecycle
# ===========================================================================


class TestConnectionLifecycle:
    def test_connect_success(self, broker):
        b, mock_rest = broker
        assert b.is_connected() is True

    def test_connect_failure(self, disconnected_broker):
        """M-CONNECT: connect now raises (retried by @_retry_read)."""
        import pytest

        b, mock_rest = disconnected_broker
        mock_rest.get_account.side_effect = Exception("auth failed")
        with pytest.raises(Exception, match="auth failed"):
            b.connect()
        assert b.is_connected() is False

    def test_disconnect(self, broker):
        b, _ = broker
        assert b.is_connected() is True
        b.disconnect()
        assert b.is_connected() is False


# ===========================================================================
# Not-connected guard
# ===========================================================================


class TestNotConnectedGuard:
    """Every method that requires a connection should raise ConnectionError."""

    def test_get_account_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_account()

    def test_submit_order_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        order = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        with pytest.raises(ConnectionError):
            b.submit_order(order)

    def test_cancel_order_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.cancel_order("order-123")

    def test_cancel_all_orders_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.cancel_all_orders()

    def test_get_order_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_order("order-123")

    def test_list_orders_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.list_orders()

    def test_get_position_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_position("AAPL")

    def test_list_positions_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.list_positions()

    def test_get_latest_price_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_latest_price("AAPL")

    def test_get_latest_prices_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_latest_prices(["AAPL"])

    def test_get_clock_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_clock()

    def test_get_bars_requires_connection(self, disconnected_broker):
        b, _ = disconnected_broker
        with pytest.raises(ConnectionError):
            b.get_bars("AAPL")


# ===========================================================================
# Idempotency: _make_client_order_id
# ===========================================================================


class TestClientOrderIdIdempotency:
    def test_same_order_same_minute_same_id(self):
        """Identical orders within the same minute produce the same client_order_id."""
        from python.brokers.alpaca_broker import AlpacaBroker

        order = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        id1 = AlpacaBroker._make_client_order_id(order)
        id2 = AlpacaBroker._make_client_order_id(order)
        assert id1 == id2
        assert len(id1) == 24  # SHA-256 truncated to 24 hex chars

    def test_different_symbol_different_id(self):
        from python.brokers.alpaca_broker import AlpacaBroker

        o1 = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        o2 = BrokerOrder(symbol="MSFT", side="buy", qty=10, order_type="market")
        assert AlpacaBroker._make_client_order_id(o1) != AlpacaBroker._make_client_order_id(o2)

    def test_different_side_different_id(self):
        from python.brokers.alpaca_broker import AlpacaBroker

        o1 = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        o2 = BrokerOrder(symbol="AAPL", side="sell", qty=10, order_type="market")
        assert AlpacaBroker._make_client_order_id(o1) != AlpacaBroker._make_client_order_id(o2)

    def test_different_qty_different_id(self):
        from python.brokers.alpaca_broker import AlpacaBroker

        o1 = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        o2 = BrokerOrder(symbol="AAPL", side="buy", qty=20, order_type="market")
        assert AlpacaBroker._make_client_order_id(o1) != AlpacaBroker._make_client_order_id(o2)

    def test_same_day_same_id(self):
        """Same order on same day produces same id (C5 fix: day-level idempotency)."""
        from python.brokers.alpaca_broker import AlpacaBroker

        order = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")

        id1 = AlpacaBroker._make_client_order_id(order)
        id2 = AlpacaBroker._make_client_order_id(order)
        assert id1 == id2

    def test_changes_across_days(self):
        """When the date changes, the id changes."""
        from datetime import datetime as real_datetime
        from datetime import timezone

        from python.brokers.alpaca_broker import AlpacaBroker

        order = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")

        fake_dt_1 = MagicMock(wraps=real_datetime)
        fake_dt_1.now.return_value = real_datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_dt_2 = MagicMock(wraps=real_datetime)
        fake_dt_2.now.return_value = real_datetime(2026, 1, 2, tzinfo=timezone.utc)

        with patch("datetime.datetime", fake_dt_1):
            id1 = AlpacaBroker._make_client_order_id(order)
        with patch("datetime.datetime", fake_dt_2):
            id2 = AlpacaBroker._make_client_order_id(order)
        assert id1 != id2


# ===========================================================================
# get_account
# ===========================================================================


class TestGetAccount:
    def test_maps_all_fields(self, broker):
        b, mock_rest = broker
        mock_rest.get_account.return_value = _make_alpaca_account(
            id="acc-002",
            cash="25000.50",
            portfolio_value="75000.00",
            buying_power="50000.00",
            equity="75000.00",
            status="ACTIVE",
        )
        acct = b.get_account()
        assert isinstance(acct, BrokerAccount)
        assert acct.account_id == "acc-002"
        assert acct.cash == 25000.50
        assert acct.portfolio_value == 75000.00
        assert acct.buying_power == 50000.00
        assert acct.equity == 75000.00
        assert acct.status == "ACTIVE"


# ===========================================================================
# submit_order
# ===========================================================================


class TestSubmitOrder:
    def test_simple_market_order(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-001")

        order = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        oid = b.submit_order(order)

        assert oid == "order-001"
        call_kwargs = mock_rest.submit_order.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["side"] == "buy"
        assert call_kwargs["qty"] == 10
        assert call_kwargs["type"] == "market"
        assert call_kwargs["time_in_force"] == "day"
        # Auto-generated client_order_id
        assert "client_order_id" in call_kwargs
        assert len(call_kwargs["client_order_id"]) == 24

    def test_limit_order_passes_limit_price(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-002")

        order = BrokerOrder(symbol="AAPL", side="buy", qty=5, order_type="limit", limit_price=150.0)
        b.submit_order(order)

        call_kwargs = mock_rest.submit_order.call_args[1]
        assert call_kwargs["limit_price"] == 150.0

    def test_stop_order_passes_stop_price(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-003")

        order = BrokerOrder(symbol="AAPL", side="sell", qty=5, order_type="stop", stop_price=140.0)
        b.submit_order(order)

        call_kwargs = mock_rest.submit_order.call_args[1]
        assert call_kwargs["stop_price"] == 140.0

    def test_bracket_order(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-004")

        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            order_class="bracket",
            take_profit_limit_price=160.0,
            stop_loss_stop_price=140.0,
            stop_loss_limit_price=139.0,
        )
        b.submit_order(order)

        call_kwargs = mock_rest.submit_order.call_args[1]
        assert call_kwargs["order_class"] == "bracket"
        assert call_kwargs["take_profit"] == {"limit_price": "160.0"}
        assert call_kwargs["stop_loss"] == {"stop_price": "140.0", "limit_price": "139.0"}

    def test_bracket_order_without_limit_on_sl(self, broker):
        """Bracket with stop_loss_stop_price but no stop_loss_limit_price."""
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-005")

        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            order_class="bracket",
            take_profit_limit_price=160.0,
            stop_loss_stop_price=140.0,
        )
        b.submit_order(order)

        call_kwargs = mock_rest.submit_order.call_args[1]
        assert call_kwargs["stop_loss"] == {"stop_price": "140.0"}

    def test_auto_generates_client_order_id_when_none(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-006")

        order = BrokerOrder(
            symbol="AAPL", side="buy", qty=10, order_type="market", client_order_id=None
        )
        b.submit_order(order)

        call_kwargs = mock_rest.submit_order.call_args[1]
        assert "client_order_id" in call_kwargs
        assert len(call_kwargs["client_order_id"]) == 24

    def test_preserves_explicit_client_order_id(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.return_value = SimpleNamespace(id="order-007")

        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            client_order_id="my-custom-id",
        )
        b.submit_order(order)

        call_kwargs = mock_rest.submit_order.call_args[1]
        assert call_kwargs["client_order_id"] == "my-custom-id"

    def test_submit_order_propagates_exception(self, broker):
        b, mock_rest = broker
        mock_rest.submit_order.side_effect = Exception("Insufficient buying power")

        order = BrokerOrder(symbol="AAPL", side="buy", qty=10, order_type="market")
        with pytest.raises(Exception, match="Insufficient buying power"):
            b.submit_order(order)


# ===========================================================================
# cancel_order / cancel_all_orders
# ===========================================================================


class TestCancelOrders:
    def test_cancel_order_success(self, broker):
        b, mock_rest = broker
        assert b.cancel_order("order-123") is True
        mock_rest.cancel_order.assert_called_once_with("order-123")

    def test_cancel_order_failure_returns_false(self, broker):
        b, mock_rest = broker
        mock_rest.cancel_order.side_effect = Exception("not found")
        assert b.cancel_order("order-bad") is False

    def test_cancel_all_orders(self, broker):
        b, mock_rest = broker
        mock_rest.cancel_all_orders.return_value = ["o1", "o2", "o3"]
        count = b.cancel_all_orders()
        assert count == 3

    def test_cancel_all_orders_none_response(self, broker):
        b, mock_rest = broker
        mock_rest.cancel_all_orders.return_value = None
        count = b.cancel_all_orders()
        assert count == 0


# ===========================================================================
# get_order / list_orders
# ===========================================================================


class TestGetOrder:
    def test_maps_to_broker_order(self, broker):
        b, mock_rest = broker
        mock_rest.get_order.return_value = _make_alpaca_order(
            id="order-x",
            symbol="MSFT",
            side="sell",
            qty="50",
            type="limit",
            limit_price="400.00",
            stop_price=None,
            time_in_force="gtc",
            client_order_id="c-id",
            status="filled",
            filled_avg_price="399.50",
        )

        order = b.get_order("order-x")
        assert isinstance(order, BrokerOrder)
        assert order.symbol == "MSFT"
        assert order.side == "sell"
        assert order.qty == 50.0
        assert order.order_type == "limit"
        assert order.limit_price == 400.0
        assert order.stop_price is None
        assert order.time_in_force == "gtc"
        assert order.client_order_id == "c-id"
        assert order.order_id == "order-x"
        assert order.status == "filled"
        assert order.filled_avg_price == 399.50

    def test_get_order_failure_returns_none(self, broker):
        b, mock_rest = broker
        mock_rest.get_order.side_effect = Exception("not found")
        assert b.get_order("bad-id") is None

    def test_list_orders(self, broker):
        b, mock_rest = broker
        mock_rest.list_orders.return_value = [
            _make_alpaca_order(id="o1", symbol="AAPL"),
            _make_alpaca_order(id="o2", symbol="MSFT"),
        ]
        orders = b.list_orders(status="open")
        assert len(orders) == 2
        assert all(isinstance(o, BrokerOrder) for o in orders)
        mock_rest.list_orders.assert_called_once_with(status="open", limit=500)

    def test_list_orders_failure_returns_empty(self, broker):
        b, mock_rest = broker
        mock_rest.list_orders.side_effect = Exception("timeout")
        assert b.list_orders() == []


# ===========================================================================
# get_position / list_positions
# ===========================================================================


class TestPositions:
    def test_get_position_maps_fields(self, broker):
        b, mock_rest = broker
        mock_rest.get_position.return_value = _make_alpaca_position(
            symbol="GOOG",
            qty="25",
            avg_entry_price="2800.00",
            market_value="72000.00",
            unrealized_pl="2000.00",
            unrealized_plpc="0.0286",
        )
        pos = b.get_position("GOOG")
        assert isinstance(pos, BrokerPosition)
        assert pos.symbol == "GOOG"
        assert pos.qty == 25.0
        assert pos.avg_entry_price == 2800.0
        assert pos.market_value == 72000.0
        assert pos.unrealized_pl == 2000.0

    def test_get_position_not_found_returns_none(self, broker):
        b, mock_rest = broker
        mock_rest.get_position.side_effect = Exception("position not found")
        assert b.get_position("NOPE") is None

    def test_list_positions(self, broker):
        b, mock_rest = broker
        mock_rest.list_positions.return_value = [
            _make_alpaca_position(symbol="AAPL"),
            _make_alpaca_position(symbol="MSFT"),
        ]
        positions = b.list_positions()
        assert len(positions) == 2
        assert all(isinstance(p, BrokerPosition) for p in positions)

    def test_list_positions_failure_returns_empty(self, broker):
        b, mock_rest = broker
        mock_rest.list_positions.side_effect = Exception("timeout")
        assert b.list_positions() == []


# ===========================================================================
# get_latest_price (Alpaca primary → yfinance fallback)
# ===========================================================================


class TestGetLatestPrice:
    def test_uses_alpaca_when_available(self, broker):
        b, mock_rest = broker
        mock_rest.get_latest_trade.return_value = _make_alpaca_trade(price=155.0)
        assert b.get_latest_price("AAPL") == 155.0

    def test_falls_back_to_yfinance(self, broker):
        b, mock_rest = broker
        mock_rest.get_latest_trade.side_effect = Exception("no subscription")

        with patch.object(b, "_yfinance_price", return_value=152.0) as yf_mock:
            price = b.get_latest_price("AAPL")
            assert price == 152.0
            yf_mock.assert_called_once_with("AAPL")


# ===========================================================================
# get_latest_prices (batch with partial fallback)
# ===========================================================================


class TestGetLatestPrices:
    def test_all_from_alpaca(self, broker):
        b, mock_rest = broker
        # H-PRICES fix: now uses batch get_latest_trades API
        mock_rest.get_latest_trades.return_value = {
            "AAPL": _make_alpaca_trade(price=200.0),
            "MSFT": _make_alpaca_trade(price=200.0),
        }

        prices = b.get_latest_prices(["AAPL", "MSFT"])
        assert len(prices) == 2
        assert prices["AAPL"] == 200.0
        assert prices["MSFT"] == 200.0

    def test_partial_alpaca_miss_yfinance_fallback(self, broker):
        b, mock_rest = broker

        # H-PRICES fix: batch API returns only AAPL
        mock_rest.get_latest_trades.return_value = {
            "AAPL": _make_alpaca_trade(price=200.0),
        }

        # yfinance is imported locally inside get_latest_prices as
        # ``import yfinance as yf``, so we must patch at the yfinance
        # module level — not on alpaca_broker.yf (which doesn't exist).
        import pandas as pd

        # Single-ticker download returns a DataFrame whose "Close" column
        # is a Series (not a sub-DataFrame).
        mock_close_series = pd.Series(
            [400.0], index=pd.date_range("2026-02-25", periods=1), name="Close"
        )
        mock_download = pd.DataFrame({"Close": mock_close_series})

        with patch("yfinance.download", return_value=mock_download):
            prices = b.get_latest_prices(["AAPL", "MSFT"])
            assert prices["AAPL"] == 200.0
            assert prices["MSFT"] == 400.0


# ===========================================================================
# get_clock / get_bars
# ===========================================================================


class TestClockAndBars:
    def test_get_clock(self, broker):
        b, mock_rest = broker
        mock_rest.get_clock.return_value = _make_alpaca_clock(is_open=True)
        clock = b.get_clock()
        assert clock["is_open"] is True
        assert "next_open" in clock
        assert "next_close" in clock

    def test_get_bars(self, broker):
        b, mock_rest = broker
        mock_rest.get_bars.return_value = [
            _make_alpaca_bar(c=150),
            _make_alpaca_bar(c=152),
        ]
        bars = b.get_bars("AAPL", timeframe="1D", limit=2)
        assert len(bars) == 2
        assert bars[0]["close"] == 150
        assert bars[1]["close"] == 152
        mock_rest.get_bars.assert_called_once_with("AAPL", "1D", limit=2)

    def test_get_bars_propagates_error(self, broker):
        b, mock_rest = broker
        mock_rest.get_bars.side_effect = Exception("rate limit")
        with pytest.raises(Exception, match="rate limit"):
            b.get_bars("AAPL")
