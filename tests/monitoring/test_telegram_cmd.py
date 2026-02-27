"""Tests for python.monitoring.telegram_cmd — Telegram command handler.

Covers:
  - Command registration and dispatch
  - Individual command output formatting
  - Security: unauthorized chat rejected
  - Polling loop update processing
  - start_telegram_command_handler lifecycle
  - Dashboard API integration (mocked)
"""

from unittest.mock import patch  # noqa: I001

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level state between tests."""
    import python.monitoring.telegram_cmd as mod

    mod._last_update_id = 0
    yield


@pytest.fixture()
def _configured(monkeypatch):
    """Configure Telegram bot for tests."""
    monkeypatch.setattr("python.monitoring.telegram_cmd.TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setattr("python.monitoring.telegram_cmd.TELEGRAM_CHAT_ID", "999")


@pytest.fixture()
def _unconfigured(monkeypatch):
    """Ensure Telegram is not configured."""
    monkeypatch.setattr("python.monitoring.telegram_cmd.TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr("python.monitoring.telegram_cmd.TELEGRAM_CHAT_ID", "")


# ===========================================================================
# Configuration
# ===========================================================================


class TestConfiguration:
    def test_is_configured_when_set(self, _configured):
        from python.monitoring.telegram_cmd import _is_configured

        assert _is_configured() is True

    def test_is_not_configured_when_empty(self, _unconfigured):
        from python.monitoring.telegram_cmd import _is_configured

        assert _is_configured() is False


# ===========================================================================
# Command registration
# ===========================================================================


class TestCommandRegistry:
    def test_all_commands_registered(self):
        """All expected commands are in the registry."""
        from python.monitoring.telegram_cmd import _COMMANDS

        expected = [
            "/help",
            "/start",
            "/status",
            "/positions",
            "/equity",
            "/regime",
            "/health",
            "/trades",
            "/logs",
        ]
        for cmd in expected:
            assert cmd in _COMMANDS, f"Command {cmd} not registered"

    def test_help_lists_all_commands(self):
        """The /help output mentions all commands."""
        from python.monitoring.telegram_cmd import _COMMANDS

        help_text = _COMMANDS["/help"]()
        for cmd in ["/status", "/positions", "/equity", "/regime", "/health", "/trades", "/logs"]:
            assert cmd in help_text

    def test_start_returns_welcome(self):
        """The /start command returns a welcome message."""
        from python.monitoring.telegram_cmd import _COMMANDS

        text = _COMMANDS["/start"]()
        assert "Signum" in text
        assert "/help" in text


# ===========================================================================
# Individual commands (with mocked dashboard API)
# ===========================================================================


class TestStatusCommand:
    def test_status_formats_correctly(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_data = {
            "regime": {"regime": "normal", "vix": 18.5, "exposure_multiplier": 1.0},
            "account": {"equity": "100000.00"},
            "bot_state": {"last_trade_date": "2026-02-26"},
            "positions_count": 5,
        }
        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=mock_data):
            text = _COMMANDS["/status"]()
            assert "normal" in text
            assert "18.5" in text
            assert "$100,000.00" in text
            assert "2026-02-26" in text
            assert "5" in text

    def test_status_handles_dashboard_down(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=None):
            text = _COMMANDS["/status"]()
            assert "Could not reach" in text


class TestPositionsCommand:
    def test_positions_formats_correctly(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_data = {
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "market_value": "1500.00",
                    "unrealized_pl": "50.00",
                    "unrealized_plpc": "0.034",
                    "weight": "0.15",
                },
            ],
            "total_value": 10000.0,
            "total_unrealized_pl": 50.0,
        }
        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=mock_data):
            text = _COMMANDS["/positions"]()
            assert "AAPL" in text
            assert "10 shares" in text

    def test_positions_empty(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value={"positions": []}):
            text = _COMMANDS["/positions"]()
            assert "No open positions" in text


class TestEquityCommand:
    def test_equity_formats_correctly(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_account = {
            "equity": "100000.00",
            "cash": "5000.00",
            "buying_power": "10000.00",
            "status": "ACTIVE",
        }
        with patch("python.monitoring.telegram_cmd._dashboard_get") as mock_get:
            mock_get.side_effect = [
                mock_account,
                {"history": [{"equity": 95000}, {"equity": 100000}]},
            ]
            text = _COMMANDS["/equity"]()
            assert "$100,000.00" in text
            assert "ACTIVE" in text


class TestRegimeCommand:
    def test_regime_formats_correctly(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_data = {
            "regime": "caution",
            "vix": 28.5,
            "spy_drawdown": -0.08,
            "exposure_multiplier": 0.5,
            "message": "Elevated volatility",
        }
        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=mock_data):
            text = _COMMANDS["/regime"]()
            assert "caution" in text
            assert "28.5" in text
            assert "0.5x" in text


class TestHealthCommand:
    def test_health_formats_correctly(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_data = {
            "status": "healthy",
            "checks": {"dashboard": "ok", "bot": "ok"},
        }
        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=mock_data):
            text = _COMMANDS["/health"]()
            assert "HEALTHY" in text
            assert "dashboard: ok" in text


class TestTradesCommand:
    def test_trades_formats_correctly(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_data = {
            "bot_state": {
                "last_trade_date": "2026-02-26",
                "last_equity": 99500.0,
                "equity_peak": 100000.0,
                "positions_count": 8,
                "positions": {"AAPL": {}, "MSFT": {}},
            }
        }
        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=mock_data):
            text = _COMMANDS["/trades"]()
            assert "2026-02-26" in text
            assert "AAPL" in text


class TestLogsCommand:
    def test_logs_returns_lines(self):
        from python.monitoring.telegram_cmd import _COMMANDS

        mock_data = {"log": ["line1", "line2", "line3"]}
        with patch("python.monitoring.telegram_cmd._dashboard_get", return_value=mock_data):
            text = _COMMANDS["/logs"]()
            assert "line1" in text
            assert "line3" in text


# ===========================================================================
# Security: unauthorized chat
# ===========================================================================


class TestSecurity:
    def test_unauthorized_chat_ignored(self, _configured):
        """Messages from unauthorized chat IDs are ignored."""
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 666},  # wrong chat
                        "text": "/status",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                _process_updates()
                mock_send.assert_not_called()

    def test_authorized_chat_responds(self, _configured):
        """Messages from the authorized chat get a response."""
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 999},  # matches _configured fixture
                        "text": "/help",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                _process_updates()
                mock_send.assert_called_once()
                assert "/status" in mock_send.call_args[0][0]


# ===========================================================================
# Polling and dispatch
# ===========================================================================


class TestPolling:
    def test_unknown_command_responds(self, _configured):
        """Unknown commands get an error message."""
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 999},
                        "text": "/nonexistent",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                _process_updates()
                mock_send.assert_called_once()
                assert "Unknown command" in mock_send.call_args[0][0]

    def test_update_id_advances(self, _configured):
        """_last_update_id advances past processed updates."""
        import python.monitoring.telegram_cmd as mod
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 42,
                    "message": {
                        "chat": {"id": 999},
                        "text": "/help",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch("python.monitoring.telegram_cmd._send_message"):
                _process_updates()
                assert mod._last_update_id == 42

    def test_non_command_messages_ignored(self, _configured):
        """Regular text messages (not starting with /) are ignored."""
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 999},
                        "text": "hello there",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                _process_updates()
                mock_send.assert_not_called()

    def test_command_handler_error_swallowed(self, _configured):
        """If a command handler throws, error message is sent, no crash."""
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 999},
                        "text": "/status",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch(
                "python.monitoring.telegram_cmd._dashboard_get", side_effect=Exception("boom")
            ):
                with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                    _process_updates()  # should not raise
                    mock_send.assert_called_once()
                    assert "Error" in mock_send.call_args[0][0]

    def test_empty_updates_is_noop(self, _configured):
        """Empty update list doesn't crash."""
        from python.monitoring.telegram_cmd import _process_updates

        with patch(
            "python.monitoring.telegram_cmd._tg_api", return_value={"ok": True, "result": []}
        ):
            with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                _process_updates()
                mock_send.assert_not_called()


# ===========================================================================
# start_telegram_command_handler
# ===========================================================================


class TestStartHandler:
    def test_returns_none_when_unconfigured(self, _unconfigured):
        from python.monitoring.telegram_cmd import start_telegram_command_handler

        result = start_telegram_command_handler()
        assert result is None

    def test_returns_thread_when_configured(self, _configured):
        from python.monitoring.telegram_cmd import start_telegram_command_handler

        # Patch the polling loop to not actually run
        with patch("python.monitoring.telegram_cmd._polling_loop"):
            thread = start_telegram_command_handler()
            assert thread is not None
            assert thread.daemon is True
            assert thread.name == "telegram-cmd-handler"
            thread.join(timeout=2)

    def test_bot_command_with_at_suffix(self, _configured):
        """Commands like /status@sigum_paperbot are handled correctly."""
        from python.monitoring.telegram_cmd import _process_updates

        fake_updates = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 999},
                        "text": "/help@sigum_paperbot",
                    },
                }
            ],
        }
        with patch("python.monitoring.telegram_cmd._tg_api", return_value=fake_updates):
            with patch("python.monitoring.telegram_cmd._send_message") as mock_send:
                _process_updates()
                mock_send.assert_called_once()
                assert "/status" in mock_send.call_args[0][0]
