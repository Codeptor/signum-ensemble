"""
Telegram command handler — interactive bot control via Telegram messages.

Runs as a background daemon thread inside the live bot process, polling for
commands every few seconds. Queries the dashboard JSON API (localhost:8050)
to fetch live state without coupling to live_bot internals.

Commands:
  /status    — Bot state, regime, account overview
  /positions — Current holdings with weights and P&L
  /equity    — Portfolio value, cash, buying power
  /regime    — Market regime (VIX, SPY drawdown, exposure)
  /health    — System health check
  /trades    — Recent trade history from bot state
  /logs      — Last 20 log lines
  /help      — List available commands

Security:
  - Only responds to messages from the authorized TELEGRAM_CHAT_ID
  - Read-only: no commands can modify positions or trigger trades

Usage:
    from python.monitoring.telegram_cmd import start_telegram_command_handler
    start_telegram_command_handler()  # launches background thread
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger("signum.telegram_cmd")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DASHBOARD_BASE_URL = os.getenv("DASHBOARD_URL", "http://localhost:8050")

# Polling interval — how often to check for new messages
_POLL_INTERVAL_SECS = 3
# Maximum offset for Telegram getUpdates (tracks last processed message)
_last_update_id = 0


def _is_configured() -> bool:
    """Check if Telegram command handler can run."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


# ---------------------------------------------------------------------------
# Telegram API helpers
# ---------------------------------------------------------------------------


def _tg_api(method: str, payload: dict | None = None, timeout: int = 10) -> dict | None:
    """Call Telegram Bot API method. Returns parsed JSON or None on error."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    try:
        if payload:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        else:
            req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.debug(f"Telegram API call failed: {method}", exc_info=True)
        return None


def _send_message(text: str, parse_mode: str | None = None) -> None:
    """Send a text message to the configured chat."""
    payload: dict = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4096],  # Telegram message limit
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    _tg_api("sendMessage", payload)


def _dashboard_get(path: str) -> dict | None:
    """GET a dashboard API endpoint. Returns parsed JSON or None."""
    url = f"{DASHBOARD_BASE_URL}{path}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Read error body for context
        try:
            body = json.loads(e.read().decode("utf-8"))
            return body
        except Exception:
            return {"error": f"HTTP {e.code}"}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Command handlers — each returns a string response
# ---------------------------------------------------------------------------

_COMMANDS = {}


def _cmd(name: str):
    """Decorator to register a command handler."""

    def decorator(fn):
        _COMMANDS[name] = fn
        return fn

    return decorator


@_cmd("/help")
def _cmd_help() -> str:
    """List available commands."""
    lines = [
        "Signum Trading Bot — Commands\n",
        "/status     — Bot state, regime, account",
        "/positions  — Current holdings with P&L",
        "/equity     — Portfolio value & returns",
        "/regime     — Market regime (VIX, drawdown)",
        "/health     — System health check",
        "/trades     — Recent trade info",
        "/logs       — Last 20 log lines",
        "/help       — This message",
    ]
    return "\n".join(lines)


@_cmd("/start")
def _cmd_start() -> str:
    """Welcome message for new chat."""
    return (
        "Signum Trading Bot connected.\n"
        "Send /help to see available commands.\n"
        "All commands are read-only — no trades will be triggered."
    )


@_cmd("/status")
def _cmd_status() -> str:
    """Bot state, regime, and account overview."""
    data = _dashboard_get("/api/status")
    if data is None:
        return "Could not reach dashboard API. Is signum-dashboard running?"

    lines = ["STATUS\n"]

    # Regime
    regime = data.get("regime")
    if regime:
        r = regime.get("regime", "unknown")
        vix = regime.get("vix", "?")
        exposure = regime.get("exposure_multiplier", "?")
        lines.append(f"Regime: {r}")
        lines.append(f"VIX: {vix}")
        lines.append(f"Exposure: {exposure}x")
    else:
        lines.append("Regime: unavailable")

    # Account
    acct = data.get("account")
    if acct:
        equity = acct.get("equity", "?")
        lines.append(f"Equity: ${float(equity):,.2f}" if equity != "?" else "Equity: ?")

    # Bot state
    bot = data.get("bot_state")
    if bot:
        last_trade = bot.get("last_trade_date", "never")
        lines.append(f"Last trade: {last_trade}")

    lines.append(f"Positions: {data.get('positions_count', 0)}")
    return "\n".join(lines)


@_cmd("/positions")
def _cmd_positions() -> str:
    """Current holdings with weights and P&L."""
    data = _dashboard_get("/api/positions")
    if data is None:
        return "Could not reach dashboard API."
    if "error" in data:
        return f"Error: {data['error']}"

    positions = data.get("positions", [])
    if not positions:
        return "No open positions."

    total_value = data.get("total_value", 0)
    total_pl = data.get("total_unrealized_pl", 0)

    lines = [f"POSITIONS ({len(positions)})\n"]
    for p in positions:
        sym = p.get("symbol", "?")
        qty = p.get("qty", 0)
        mv = float(p.get("market_value", 0))
        pl = float(p.get("unrealized_pl", 0))
        pct = float(p.get("unrealized_plpc", 0)) * 100
        weight = float(p.get("weight", 0)) * 100
        sign = "+" if pl >= 0 else ""
        line = (
            f"  {sym}: {qty} shares, ${mv:,.0f} ({weight:.1f}%), "
            f"{sign}${pl:,.0f} ({sign}{pct:.1f}%)"
        )
        lines.append(line)

    lines.append(f"\nTotal: ${float(total_value):,.0f}")
    sign = "+" if total_pl >= 0 else ""
    lines.append(f"Unrealized P&L: {sign}${float(total_pl):,.0f}")
    return "\n".join(lines)


@_cmd("/equity")
def _cmd_equity() -> str:
    """Portfolio value, cash, buying power."""
    data = _dashboard_get("/api/account")
    if data is None:
        return "Could not reach dashboard API."
    if "error" in data:
        return f"Error: {data['error']}"

    equity = float(data.get("equity", 0))
    cash = float(data.get("cash", 0))
    bp = float(data.get("buying_power", 0))
    status = data.get("status", "unknown")

    lines = [
        "ACCOUNT\n",
        f"Equity:       ${equity:,.2f}",
        f"Cash:         ${cash:,.2f}",
        f"Buying power: ${bp:,.2f}",
        f"Status:       {status}",
    ]

    # Try to get return info from equity history
    eq_data = _dashboard_get("/api/equity")
    if eq_data and "history" in eq_data and len(eq_data["history"]) >= 2:
        history = eq_data["history"]
        first = history[0]["equity"]
        last = history[-1]["equity"]
        if first > 0:
            total_ret = (last - first) / first * 100
            sign = "+" if total_ret >= 0 else ""
            lines.append(f"Total return: {sign}{total_ret:.2f}%")

    return "\n".join(lines)


@_cmd("/regime")
def _cmd_regime() -> str:
    """Market regime details."""
    data = _dashboard_get("/api/regime")
    if data is None:
        return "Could not reach dashboard API."
    if "error" in data:
        return f"Error: {data['error']}"

    regime = data.get("regime", "unknown")
    vix = data.get("vix", "?")
    spy_dd = data.get("spy_drawdown", "?")
    exposure = data.get("exposure_multiplier", "?")
    msg = data.get("message", "")

    lines = [
        "MARKET REGIME\n",
        f"Regime:    {regime}",
        f"VIX:       {vix}",
        f"SPY DD:    {float(spy_dd) * 100:.1f}%"
        if isinstance(spy_dd, (int, float))
        else f"SPY DD:    {spy_dd}",
        f"Exposure:  {exposure}x",
    ]
    if msg:
        lines.append(f"\n{msg}")
    return "\n".join(lines)


@_cmd("/health")
def _cmd_health() -> str:
    """System health check."""
    data = _dashboard_get("/healthz")
    if data is None:
        return "Could not reach dashboard API. Dashboard may be down."

    status = data.get("status", "unknown")
    checks = data.get("checks", {})

    lines = [f"HEALTH: {status.upper()}\n"]
    for key, val in checks.items():
        if isinstance(val, dict):
            # Nested (like alerting status)
            lines.append(f"  {key}:")
            for k2, v2 in val.items():
                lines.append(f"    {k2}: {v2}")
        else:
            lines.append(f"  {key}: {val}")

    return "\n".join(lines)


@_cmd("/trades")
def _cmd_trades() -> str:
    """Recent trade info from bot state."""
    data = _dashboard_get("/api/bot")
    if data is None:
        return "Could not reach dashboard API."

    bot = data.get("bot_state")
    if not bot:
        return "No bot state available. Bot hasn't traded yet."

    lines = ["TRADE INFO\n"]
    lines.append(f"Last trade date: {bot.get('last_trade_date', 'never')}")
    lines.append(f"Last equity: ${float(bot.get('last_equity', 0)):,.2f}")
    lines.append(f"Equity peak: ${float(bot.get('equity_peak', 0)):,.2f}")
    lines.append(f"Positions: {bot.get('positions_count', 0)}")

    if bot.get("reason"):
        lines.append(f"Last shutdown: {bot['reason']}")

    positions = bot.get("positions", {})
    if positions:
        lines.append(f"\nHeld tickers: {', '.join(sorted(positions.keys()))}")

    return "\n".join(lines)


@_cmd("/logs")
def _cmd_logs() -> str:
    """Last 20 log lines."""
    data = _dashboard_get("/api/logs?lines=20")
    if data is None:
        return "Could not reach dashboard API."

    log_lines = data.get("log", [])
    if not log_lines:
        return "No log lines available."

    # Trim each line for Telegram readability
    trimmed = [line[:200] for line in log_lines[-20:]]
    return "\n".join(trimmed)


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


def _process_updates() -> None:
    """Poll for new Telegram messages and dispatch commands."""
    global _last_update_id

    params = {"offset": _last_update_id + 1, "timeout": 0, "limit": 10}
    result = _tg_api("getUpdates", params)
    if not result or not result.get("ok"):
        return

    for update in result.get("result", []):
        update_id = update.get("update_id", 0)
        if update_id > _last_update_id:
            _last_update_id = update_id

        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()

        # Security: only respond to authorized chat
        if chat_id != TELEGRAM_CHAT_ID:
            logger.debug(f"Ignoring message from unauthorized chat: {chat_id}")
            continue

        if not text.startswith("/"):
            continue

        # Extract command (ignore @botname suffix)
        cmd = text.split()[0].split("@")[0].lower()

        handler = _COMMANDS.get(cmd)
        if handler:
            try:
                response = handler()
                _send_message(response)
            except Exception:
                logger.debug(f"Command handler failed: {cmd}", exc_info=True)
                _send_message(f"Error processing {cmd}. Check bot logs.")
        else:
            _send_message(f"Unknown command: {cmd}\nSend /help for available commands.")


def _polling_loop() -> None:
    """Background polling loop — runs until process exits."""
    logger.info("Telegram command handler started")
    while True:
        try:
            _process_updates()
        except Exception:
            logger.debug("Telegram polling error", exc_info=True)
        time.sleep(_POLL_INTERVAL_SECS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_telegram_command_handler() -> threading.Thread | None:
    """Start the Telegram command polling loop in a background daemon thread.

    Returns the thread object, or None if Telegram is not configured.
    Safe to call even if not configured — returns None silently.
    """
    if not _is_configured():
        logger.info(
            "Telegram command handler not configured"
            " (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)"
        )
        return None

    thread = threading.Thread(
        target=_polling_loop,
        name="telegram-cmd-handler",
        daemon=True,
    )
    thread.start()
    logger.info(f"Telegram command handler started (polling every {_POLL_INTERVAL_SECS}s)")
    return thread
