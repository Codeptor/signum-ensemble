"""
Live Automated Trading Bot — ML-Driven

Runs the full pipeline daily:
  1. Train LightGBM on S&P 500 history
  2. Rank all S&P 500 stocks by predicted 5-day return
  3. Select top 10 stocks
  4. Run HRP portfolio optimization on those 10
  5. Submit orders to Alpaca to reach target weights

Schedule: run daily ~15 min before market close, or keep running in a loop.
"""

import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from python.alpha.features import get_current_atr
from python.alpha.predict import get_ml_weights
from python.bridge.execution import ExecutionBridge
from python.brokers.alpaca_broker import AlpacaBroker
from python.brokers.base import BrokerOrder
from python.data.config import RISK_ENGINE_CACHE_PATH, STALE_DATA_EXPOSURE_MULT
from python.data.sectors import DEFAULT_MAX_SECTOR_WEIGHT, SECTOR_MAP
from python.monitoring.alerting import AlertSeverity, send_alert, send_heartbeat, send_trade_summary
from python.monitoring.hmm_regime import HMMRegimeDetector, HMMRegimeState
from python.monitoring.regime import RegimeDetector, RegimeState, fetch_spy_drawdown, fetch_vix
from python.monitoring.telegram_cmd import start_telegram_command_handler
from python.portfolio.risk_manager import RiskLimits, RiskManager
from python.portfolio.tca import TradeRecord, TransactionCostAnalyzer

# --- Logging with rotation (Fix #36) ---
# JSON logging: set LOG_FORMAT=json for structured output (machine-parseable).
# Default: human-readable format for console/file debugging.


class _JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        import json as _json

        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return _json.dumps(entry, default=str)


_log_format = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
if _log_format == "json":
    _log_formatter = _JsonFormatter()
else:
    _log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_formatter)

# Rotate at 10 MB, keep 5 backups (≈50 MB max disk)
_file_handler = logging.handlers.RotatingFileHandler(
    "live_bot.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_console_handler, _file_handler],
)
logger = logging.getLogger("LiveBot")

# --- Configuration (all overridable via environment variables) ---


def _env_float(key: str, default: float) -> float:
    """Read a float from env, falling back to default."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(f"Invalid float for {key}={val!r}, using default {default}")
        return default


def _env_int(key: str, default: int) -> int:
    """Read an int from env, falling back to default."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(f"Invalid int for {key}={val!r}, using default {default}")
        return default


TOP_N_STOCKS = _env_int("TOP_N_STOCKS", 10)
OPTIMIZER_METHOD = os.getenv("OPTIMIZER_METHOD", "hrp")
MAX_POSITION_WEIGHT = _env_float("MAX_POSITION_WEIGHT", 0.30)
MAX_PORTFOLIO_VAR_95 = _env_float("MAX_PORTFOLIO_VAR_95", 0.06)
MAX_DRAWDOWN_LIMIT = _env_float("MAX_DRAWDOWN_LIMIT", 0.15)
# Bracket defaults are sized for a weekly holding period.  Tighter brackets
# (e.g., 2x/3x ATR) trigger on normal 1-2 day volatility spikes, prematurely
# stopping out positions before the 5-day prediction horizon plays out.
# 3x ATR SL / 5x ATR TP gives ~3:5 risk/reward and fewer whipsaws.
# Fixed fallback percentages apply when ATR is unavailable.
STOP_LOSS_PCT = _env_float("STOP_LOSS_PCT", 0.05)
TAKE_PROFIT_PCT = _env_float("TAKE_PROFIT_PCT", 0.15)
ATR_SL_MULTIPLIER = _env_float("ATR_SL_MULTIPLIER", 2.0)
ATR_TP_MULTIPLIER = _env_float("ATR_TP_MULTIPLIER", 3.0)
SLEEP_AFTER_TRADE_HOURS = _env_int("SLEEP_AFTER_TRADE_HOURS", 12)
SLEEP_MARKET_CLOSED_HOURS = _env_int("SLEEP_MARKET_CLOSED_HOURS", 1)
ORDER_POLL_INTERVAL_SECS = _env_int("ORDER_POLL_INTERVAL_SECS", 2)
ORDER_POLL_TIMEOUT_SECS = _env_int("ORDER_POLL_TIMEOUT_SECS", 60)
TERMINAL_ORDER_STATES = {
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
}

# --- Rebalancing schedule (Phase 1: Cost Reduction) ---
# Weekly rebalancing reduces transaction costs by ~74% vs daily.
# Wednesday chosen: avoids Monday/Friday effects, mid-week liquidity is higher.
REBALANCE_DAY = _env_int("REBALANCE_DAY", 2)  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
REBALANCE_FREQUENCY = os.getenv("REBALANCE_FREQUENCY", "weekly")  # "daily" or "weekly"

# --- Alerting (Fix #37) ---
# Alerting now handled by python.monitoring.alerting module (email + webhook).
# ALERT_WEBHOOK_URL is read directly by the alerting module from env.


def should_rebalance_today() -> bool:
    """Check if today is a rebalance day based on REBALANCE_FREQUENCY.

    Weekly: only trade on REBALANCE_DAY (default Wednesday).
    Daily: always trade (original behavior).
    H-TZ fix: use Eastern Time (market timezone) instead of naive local time.
    """
    if REBALANCE_FREQUENCY == "daily":
        return True
    return datetime.now(tz=ZoneInfo("America/New_York")).weekday() == REBALANCE_DAY


def _get_next_rebalance_date() -> datetime:
    """Return the next rebalance date (for sleep calculation)."""
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    if REBALANCE_FREQUENCY == "daily":
        # Next business day
        delta = 1 if now.weekday() < 4 else (7 - now.weekday())
        return now.replace(hour=9, minute=30, second=0) + pd.Timedelta(days=delta)

    # Weekly: find next occurrence of REBALANCE_DAY
    days_ahead = REBALANCE_DAY - now.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return now.replace(hour=9, minute=30, second=0) + pd.Timedelta(days=days_ahead)


def _liquidate_all_positions(broker: AlpacaBroker) -> int:
    """Close all open positions. Returns the number of confirmed liquidation fills.

    Used during regime halt and drawdown kill-switch events.
    Cancels all open orders first to prevent interference.

    H2 fix: verifies each liquidation fill instead of fire-and-forget.
    """
    # H-LIQRACE fix: retry cancel to prevent GTC stops racing with market sells
    for attempt in range(3):
        try:
            broker.cancel_all_orders()
            break
        except Exception as e:
            logger.warning(
                f"Could not cancel open orders during liquidation (attempt {attempt + 1}/3): {e}"
            )
            if attempt < 2:
                time.sleep(1)

    positions = broker.list_positions()
    if not positions:
        logger.info("No positions to liquidate.")
        return 0

    submitted = []
    for pos in positions:
        try:
            side = "sell" if pos.qty > 0 else "buy"
            qty = abs(pos.qty)
            order = BrokerOrder(
                symbol=pos.symbol,
                side=side,
                qty=qty,
                order_type="market",
                time_in_force="day",
            )
            order_id = broker.submit_order(order)
            submitted.append({"order_id": order_id, "symbol": pos.symbol, "qty": qty})
            logger.info(f"  Liquidating: {side.upper()} {qty:.4f} {pos.symbol}")
        except Exception as e:
            logger.error(f"  Failed to liquidate {pos.symbol}: {e}")

    # H2 fix: verify each liquidation fill
    confirmed = 0
    for entry in submitted:
        result = _verify_order_fill(broker, entry["order_id"], entry["symbol"], entry["qty"])
        if result["status"] in ("filled", "partially_filled"):
            confirmed += 1
            logger.info(f"  Liquidation confirmed: {entry['symbol']} ({result['status']})")
        else:
            logger.error(
                f"  Liquidation NOT confirmed: {entry['symbol']} — status={result['status']}. "
                f"MANUAL INTERVENTION MAY BE REQUIRED."
            )
            _send_alert(
                f"Liquidation of {entry['symbol']} not confirmed "
                f"(status={result['status']}). Check manually!",
                AlertSeverity.CRITICAL,
            )

    logger.info(
        f"Liquidation complete: {confirmed}/{len(submitted)} fills confirmed "
        f"out of {len(positions)} positions."
    )
    _send_alert(
        f"Liquidated {confirmed} positions due to emergency condition.",
        AlertSeverity.CRITICAL,
    )
    return confirmed


def _send_alert(message: str, severity: AlertSeverity = AlertSeverity.WARNING) -> None:
    """Fire-and-forget alert via email + webhook.

    Delegates to python.monitoring.alerting which handles:
    - SMTP email delivery (background thread)
    - Webhook POST (Slack/Discord/generic)
    - Rate limiting to prevent alert storms
    - Never raises — alerting failures are swallowed.
    """
    send_alert(message, severity=severity)


def _verify_order_fill(
    broker: AlpacaBroker,
    order_id: str,
    symbol: str,
    expected_qty: float,
) -> dict:
    """Poll broker until order reaches a terminal state.

    Returns a dict with keys: status, filled_qty, filled_avg_price, symbol, order_id.
    For partial fills, filled_qty reflects the *actual* filled amount (not expected).
    Timed-out orders are cancelled to prevent untracked late fills.
    """
    elapsed = 0.0
    while elapsed < ORDER_POLL_TIMEOUT_SECS:
        time.sleep(ORDER_POLL_INTERVAL_SECS)
        elapsed += ORDER_POLL_INTERVAL_SECS

        order = broker.get_order(order_id)
        if order is None:
            logger.warning(f"  Could not fetch order {order_id} for {symbol}")
            continue

        status = (order.status or "").lower()
        if status in TERMINAL_ORDER_STATES:
            # Use actual filled qty from broker (critical for partial fills)
            filled_avg_price = getattr(order, "filled_avg_price", None)
            # Alpaca's qty field on a terminal order reflects the original qty,
            # not the filled qty. For partial fills, use filled_qty if available.
            actual_filled_qty = getattr(order, "filled_qty", None)
            if actual_filled_qty is None:
                # Fallback: for 'filled' status, assume full fill
                actual_filled_qty = order.qty if status == "filled" else 0
            return {
                "status": status,
                "filled_qty": float(actual_filled_qty),
                "filled_avg_price": filled_avg_price,
                "symbol": symbol,
                "order_id": order_id,
            }

        logger.debug(f"  Order {order_id} ({symbol}): status={status}, waiting...")

    # Timeout — cancel the order to prevent untracked late fills
    logger.warning(
        f"  Order {order_id} ({symbol}) did not reach terminal state "
        f"within {ORDER_POLL_TIMEOUT_SECS}s — cancelling to prevent untracked fills"
    )
    try:
        broker.cancel_order(order_id)
        logger.info(f"  Cancelled timed-out order {order_id} ({symbol})")
    except Exception as e:
        logger.error(f"  Failed to cancel timed-out order {order_id}: {e}")

    # H-TIMEOUT fix: after cancel, re-query to capture any partial fill qty
    final_qty = 0.0
    final_price = None
    try:
        final_order = broker.get_order(order_id)
        if final_order and final_order.filled_qty is not None:
            final_qty = float(final_order.filled_qty)
            final_price = final_order.filled_avg_price
            if final_qty > 0:
                logger.info(
                    f"  Timeout order had partial fill: {final_qty:.4f} {symbol} @ {final_price}"
                )
    except Exception as e:
        logger.warning(
            f"  Post-cancel order query failed for {symbol}: {e} — recording fill as {final_qty}"
        )

    return {
        "status": "timeout",
        "filled_qty": final_qty,
        "filled_avg_price": final_price,
        "symbol": symbol,
        "order_id": order_id,
    }


def _cancel_existing_sl_tp_orders(broker: AlpacaBroker, symbol: str) -> int:
    """Cancel any existing stop-loss or take-profit orders for a symbol.

    Prevents SL/TP order accumulation when a position is topped up across
    multiple trading cycles.

    Returns the number of orders cancelled.
    """
    try:
        open_orders = broker.list_orders(status="open")
        cancelled = 0
        for order in open_orders:
            if order.symbol == symbol and order.order_type in ("stop", "limit"):
                # SL orders are stop orders, TP orders are limit orders
                # Both are sell-side (closing the position)
                # M-CANCEL fix: also check order_class to avoid cancelling
                # non-SL/TP sell-side limit orders (e.g. manual sells)
                is_sl_tp = getattr(order, "order_class", None) in ("oco", "oto", "bracket", None)
                if order.side.lower() == "sell" and order.order_id and is_sl_tp:
                    try:
                        broker.cancel_order(order.order_id)
                        cancelled += 1
                        logger.info(f"  Cancelled existing {order.order_type} order for {symbol}")
                    except Exception as e:
                        logger.warning(f"  Failed to cancel {order.order_type} for {symbol}: {e}")
        return cancelled
    except Exception as e:
        logger.warning(f"  Could not list orders to cancel SL/TP for {symbol}: {e}")
        return 0


def _cleanup_orphaned_sl_tp(broker: AlpacaBroker) -> int:
    """Cancel orphaned SL or TP orders for positions that no longer exist.

    H3 fix: When SL and TP are submitted as separate orders (not OCO), one
    can trigger while the other remains open. For example, if SL fills and
    closes the position, the TP sell order stays open and can later fill,
    creating an unintended short position.

    This function checks all open sell-side stop/limit orders and cancels
    any whose symbol is no longer held in the portfolio.

    Should be called at the start of each trading cycle.

    Returns the number of orphaned orders cancelled.
    """
    try:
        # Get current positions from broker
        positions = broker.list_positions()
        held_symbols = {p.symbol for p in positions if p.qty > 0}

        # Get all open orders
        open_orders = broker.list_orders(status="open")
        cancelled = 0

        for order in open_orders:
            # Only check sell-side stop/limit orders (SL/TP protective orders)
            if (
                order.side.lower() == "sell"
                and order.order_type in ("stop", "limit")
                and order.symbol not in held_symbols
                and order.order_id
            ):
                try:
                    broker.cancel_order(order.order_id)
                    cancelled += 1
                    logger.info(
                        f"  Cancelled orphaned {order.order_type} order for {order.symbol} "
                        f"(position no longer held)"
                    )
                except Exception as e:
                    logger.warning(f"  Failed to cancel orphaned order for {order.symbol}: {e}")

        if cancelled > 0:
            logger.info(f"Cleaned up {cancelled} orphaned SL/TP orders.")
        return cancelled
    except Exception as e:
        logger.warning(f"Could not clean up orphaned SL/TP orders: {e}")
        return 0


# --- State persistence (P1-6) ---
STATE_FILE = Path("data/bot_state.json")


def _load_bot_state() -> dict:
    """Load persisted bot state from disk."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load bot state: {e}")
        return {}


def _save_bot_state(state: dict) -> None:
    """Persist bot state to disk.

    H1 fix: atomic write via temp file + rename. A crash during write
    can't corrupt the state file because rename is atomic on POSIX.
    """
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(STATE_FILE)  # Atomic rename
    except Exception as e:
        logger.warning(f"Could not save bot state: {e}")


EQUITY_HISTORY_FILE = Path("data/equity_history.json")


def _append_equity_history(equity: float) -> None:
    """Append an equity snapshot to the equity history file (R3-E-12 fix).

    The dashboard reads this file to display the equity curve.
    Each entry is {date, equity}.
    """
    try:
        EQUITY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if EQUITY_HISTORY_FILE.exists():
            with open(EQUITY_HISTORY_FILE, "r") as f:
                records = json.load(f)
        records.append(
            {
                "date": datetime.now(tz=ZoneInfo("America/New_York")).isoformat(),
                "equity": round(equity, 2),
            }
        )
        # Keep last 2000 entries to avoid unbounded growth
        records = records[-2000:]
        tmp = EQUITY_HISTORY_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(records, f)
        tmp.replace(EQUITY_HISTORY_FILE)
    except Exception as e:
        logger.warning(f"Could not append equity history: {e}")


def _has_traded_today(broker: AlpacaBroker) -> bool:
    """Check if the bot has already submitted *entry* orders today.

    Prevents duplicate execution on restart by checking for recent buy-side
    orders from the current trading session.  GTC sell-side orders (SL/TP
    brackets from previous cycles) are excluded — otherwise a bracket fill
    on a rebalance morning would cause the bot to skip the weekly rebalance.

    C2 fix: created_at may be a datetime or string — normalize to string.
    M1 fix: use UTC date to match Alpaca's UTC timestamps.
    R3-E-13 fix: use ``after`` param to scope query to today's orders only,
    avoiding the 500-order pagination limit.
    """
    try:
        from datetime import timezone

        # R3-E-13 fix: scope query to today's orders using ``after`` parameter
        today_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        after_ts = f"{today_utc}T00:00:00Z"
        orders = broker.list_orders(status="all", after=after_ts)

        # Only count buy-side orders as "traded today".  Sell-side orders
        # include GTC SL/TP bracket fills from previous weeks that should
        # not prevent the current cycle's rebalance.
        entry_orders = [
            o
            for o in orders
            if o.status not in ("canceled", "cancelled", "expired")
            and o.order_id
            and getattr(o, "side", "") == "buy"
        ]

        if entry_orders:
            logger.info(f"Found {len(entry_orders)} buy-side orders executed today.")
            return True

        return False
    except Exception as e:
        logger.error(f"Could not check if traded today: {e}. HALTING to prevent duplicate trades.")
        # Fail closed: assume we traded to prevent duplicate execution
        return True


def _initialize_risk_engine(broker: AlpacaBroker, risk_manager: RiskManager) -> None:
    """Initialize the risk engine with historical returns for current positions.

    Without this, all risk checks (VaR, drawdown, volatility) are no-ops
    because risk_engine remains None.

    Circuit breaker: on yfinance failure, falls back to a disk-cached
    returns DataFrame from a previous successful initialization.
    """
    positions = broker.list_positions()
    if not positions:
        logger.info("No positions — skipping risk engine initialization.")
        return

    symbols = [p.symbol for p in positions]
    total_value = sum(p.market_value for p in positions)

    if total_value <= 0:
        logger.warning("Total position value <= 0 — skipping risk engine init.")
        return

    weights = pd.Series({p.symbol: p.market_value / total_value for p in positions})

    # Fetch 1-year of daily bars for portfolio risk calc
    returns_df = None
    try:
        import yfinance as yf

        data = yf.download(symbols, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data is not None and len(data) > 0:
            close_raw = data["Close"]
            close: pd.DataFrame = (
                close_raw.to_frame(name=symbols[0])
                if isinstance(close_raw, pd.Series)
                else pd.DataFrame(close_raw)
            )
            returns = close.pct_change().dropna()
            available = [s for s in symbols if s in returns.columns]
            if available:
                returns_df = pd.DataFrame(returns[available])
                # Persist for offline fallback
                try:
                    RISK_ENGINE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    tmp = RISK_ENGINE_CACHE_PATH.with_suffix(".tmp")
                    returns_df.to_parquet(tmp)
                    tmp.replace(RISK_ENGINE_CACHE_PATH)
                    logger.info("Persisted risk engine returns cache")
                except Exception as e2:
                    logger.warning(f"Could not persist risk engine cache: {e2}")
    except Exception as e:
        logger.warning(f"Failed to fetch risk engine data: {e} — trying disk cache")

    # Circuit breaker: fall back to disk cache
    if returns_df is None and RISK_ENGINE_CACHE_PATH.exists():
        try:
            returns_df = pd.read_parquet(RISK_ENGINE_CACHE_PATH)
            # Restrict to current symbols
            available = [s for s in symbols if s in returns_df.columns]
            if available:
                returns_df = returns_df[available]
                logger.info(
                    f"Loaded risk engine cache ({len(returns_df)} days, {len(available)} symbols)"
                )
            else:
                logger.warning("Risk engine cache has no matching symbols")
                returns_df = None
        except Exception as e:
            logger.warning(f"Risk engine disk cache failed: {e}")
            returns_df = None

    if returns_df is not None and len(returns_df) > 0:
        available = [s for s in symbols if s in returns_df.columns]
        risk_manager.initialize_portfolio_risk(
            returns=returns_df,
            weights=weights.reindex(available).fillna(0),
        )
        logger.info(
            f"Risk engine initialized with {len(available)} positions, "
            f"{len(returns_df)} days of history."
        )
    else:
        logger.warning("Risk engine could not be initialized — risk checks will be no-ops.")


def run_trading_cycle(
    broker: AlpacaBroker,
    risk_manager: RiskManager,
    bridge: ExecutionBridge,
    regime_detector: RegimeDetector | None = None,
    regime_state: RegimeState | None = None,
    tca_analyzer: TransactionCostAnalyzer | None = None,
) -> bool:
    """The core daily trading logic. Returns True if trades were executed."""
    logger.info("=" * 60)
    logger.info("Starting daily trading cycle...")
    logger.info("=" * 60)

    # Check for data staleness (P1-8)
    try:
        clock = broker.get_clock()
        if not clock.get("is_open", False):
            logger.info("Market is closed — skipping cycle")
            return False
    except Exception as e:
        logger.error(f"Could not check market status: {e}")
        _send_alert(f"Failed to check market status: {e}", AlertSeverity.CRITICAL)
        return False

    # 0. Clean up orphaned SL/TP orders from previous cycles (H3 fix)
    #    When SL triggers, TP remains open (and vice versa). If the position
    #    is closed, these orphaned sell orders can create naked short positions.
    _cleanup_orphaned_sl_tp(broker)

    # 1. Cancel stale open orders — but preserve bracket SL/TP legs
    #    Bracket legs (stop-loss, take-profit) have a parent_order_id and
    #    protect open positions. Cancelling them leaves the portfolio unhedged
    #    during the ~2.5 minute ML pipeline run.
    logger.info("Cancelling stale open orders (preserving bracket SL/TP legs)...")
    try:
        open_orders = broker.list_orders(status="open")
        cancelled = 0
        for order in open_orders:
            # Skip bracket child legs (SL/TP) — they protect existing positions
            if order.parent_order_id:
                logger.info(
                    f"  Keeping bracket leg: {order.order_type} {order.symbol} "
                    f"(parent: {order.parent_order_id})"
                )
                continue
            if order.order_id:
                broker.cancel_order(order.order_id)
                cancelled += 1
        if cancelled > 0:
            logger.info(f"  Cancelled {cancelled} stale orders.")
        else:
            logger.info("  No stale orders to cancel.")
    except Exception as e:
        logger.warning(f"  Could not cancel stale orders: {e}")

    # 2. Run the ML pipeline to get target weights
    logger.info("Running ML pipeline (train -> rank -> optimize)...")
    try:
        # Get current position weights for turnover-aware optimization
        current_positions = broker.list_positions()
        account_info = broker.get_account()
        current_weights = {}
        if current_positions and account_info.equity > 0:
            current_weights = {
                p.symbol: p.market_value / account_info.equity for p in current_positions
            }

        target_weights, stale_data = get_ml_weights(
            top_n=TOP_N_STOCKS,
            method=OPTIMIZER_METHOD,
            current_weights=current_weights if current_weights else None,
            max_weight=MAX_POSITION_WEIGHT,
        )
    except Exception as e:
        logger.error(f"ML pipeline failed: {e}", exc_info=True)
        _send_alert(f"ML pipeline failed: {e}", AlertSeverity.CRITICAL)
        return False

    if not target_weights:
        logger.error("ML pipeline returned no weights — skipping this cycle")
        return False

    # Circuit breaker: reduce exposure when trading on stale cached data
    if stale_data:
        mult = STALE_DATA_EXPOSURE_MULT
        logger.warning(
            f"STALE DATA detected — reducing all weights by {mult:.0%} (STALE_DATA_EXPOSURE_MULT)"
        )
        target_weights = {t: w * mult for t, w in target_weights.items()}
        _send_alert(
            f"Trading on stale data — exposure reduced to {mult:.0%} of target weights.",
        )

    logger.info(f"Target weights from ML pipeline ({len(target_weights)} positions):")
    for ticker, w in sorted(target_weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {ticker}: {w:.2%}")

    # Feature drift detection — compare inference features to training reference.
    # Informational only during paper trading: log and alert but don't block.
    import python.alpha.predict as _predict_mod

    drift_report = _predict_mod._last_drift_report
    if drift_report is not None:
        drifted_features = [k for k, v in drift_report.items() if v["drifted"]]
        total_features = len(drift_report)
        if drifted_features:
            drift_pct = len(drifted_features) / total_features * 100
            logger.warning(
                f"DRIFT ALERT: {len(drifted_features)}/{total_features} features "
                f"({drift_pct:.0f}%) show distribution drift: {drifted_features}"
            )
            # High-severity PSI (>0.2 indicates major shift)
            severe = [f for f in drifted_features if drift_report[f].get("psi", 0) > 0.2]
            if severe:
                logger.warning(f"  High PSI (>0.2) features: {severe}")
            _send_alert(
                f"Feature drift detected: {len(drifted_features)}/{total_features} "
                f"features drifted. High-PSI: {severe or 'none'}. "
                f"Model predictions may be degraded.",
            )
        else:
            logger.info(f"Drift check passed: 0/{total_features} features drifted")

    # Apply regime-based exposure adjustment (Phase 2: Risk Management)
    # H-CAUTION fix: track whether caution mode is active to prevent renorm undoing it
    in_caution_mode = False
    if regime_detector is not None and regime_state is not None:
        if regime_state.regime == "caution":
            target_weights = regime_detector.adjust_weights(target_weights, regime_state.regime)
            in_caution_mode = True
            logger.info(
                f"Regime-adjusted weights ({regime_state.regime}, "
                f"{regime_state.exposure_multiplier:.0%} exposure):"
            )
            for ticker, w in sorted(target_weights.items(), key=lambda x: -x[1]):
                logger.info(f"  {ticker}: {w:.2%}")

    # 2b. Apply graduated drawdown deleveraging
    # Smoothly reduces exposure between dd_deleverage_start and dd_hard_limit
    # instead of binary kill switch. Already updated in check_portfolio_risk.
    dd_state = risk_manager.last_drawdown_state
    if dd_state is not None and dd_state.is_deleveraging:
        factor = dd_state.exposure_factor
        target_weights = {t: w * factor for t, w in target_weights.items()}
        in_caution_mode = True  # Prevent renorm from undoing the reduction
        logger.warning(
            f"Drawdown control: {abs(dd_state.current_drawdown):.2%} drawdown — "
            f"scaling exposure to {factor:.0%}"
        )
        _send_alert(
            f"Drawdown deleveraging active: {abs(dd_state.current_drawdown):.2%} "
            f"drawdown, exposure scaled to {factor:.0%}",
        )

    # 3. Get current prices for order sizing (batch fetch, yfinance fallback)
    logger.info("Fetching current prices for order sizing...")
    prices = broker.get_latest_prices(list(target_weights.keys()))

    # Drop any tickers we couldn't price
    failed_tickers = set(target_weights) - set(prices)
    if failed_tickers:
        logger.warning(f"Removing tickers with no price data: {failed_tickers}")
        target_weights = {t: w for t, w in target_weights.items() if t in prices}

        # H-CAUTION fix: skip renormalization when in caution mode
        # (renorming to 1.0 would undo the 50% exposure reduction)
        if not in_caution_mode:
            # H-CLAMP fix: iterative renorm-clamp loop to maintain sum-to-1
            # while respecting MAX_POSITION_WEIGHT cap
            total = sum(target_weights.values())
            if total > 0:
                for _ in range(10):  # Max 10 iterations, converges in 2-3
                    target_weights = {t: w / total for t, w in target_weights.items()}
                    clamped = {t: min(w, MAX_POSITION_WEIGHT) for t, w in target_weights.items()}
                    total = sum(clamped.values())
                    if abs(total - 1.0) < 1e-6:
                        target_weights = clamped
                        break
                    target_weights = clamped
                else:
                    logger.info("Renorm-clamp loop: converged after max iterations")
                    target_weights = clamped

    if not target_weights:
        logger.error("No tradeable tickers remaining — skipping cycle")
        _send_alert("No tradeable tickers after price filtering", AlertSeverity.CRITICAL)
        return False

    # 4. Sync execution bridge with current account/positions from broker
    account = broker.get_account()
    logger.info(
        f"Account equity: ${account.equity:,.2f} | "
        f"Cash: ${account.cash:,.2f} | "
        f"Buying power: ${account.buying_power:,.2f}"
    )

    # Update bridge equity AND cash from broker (authoritative source)
    # Bug fix: bridge.cash was never synced, causing all buy orders to be
    # silently rejected after the first cycle (cash ≈ 0 from simulated fills).
    bridge.equity = account.equity
    bridge.cash = account.cash

    # R3-E-1 fix: reset ALL bridge positions before syncing from broker.
    # Without this, positions closed via SL/TP between cycles remain as
    # ghost entries in the bridge, causing spurious sell orders.
    for ticker in list(bridge.positions.keys()):
        bridge.positions[ticker].quantity = 0.0
        bridge.positions[ticker].avg_cost = 0.0

    # Sync current positions from broker into the bridge
    current_positions = broker.list_positions()
    for pos in current_positions:
        bridge_pos = bridge.get_position(pos.symbol)
        bridge_pos.quantity = pos.qty
        bridge_pos.avg_cost = pos.avg_entry_price

    logger.info(f"Current positions: {len(current_positions)}")
    for pos in current_positions:
        logger.info(
            f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
            f"(P&L: ${pos.unrealized_pl:.2f})"
        )

    # 5. Reconcile portfolio to target weights
    logger.info("Reconciling portfolio to target weights...")
    fills = bridge.reconcile_target_weights(
        target_weights=target_weights,
        prices=prices,
        current_date=datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d"),
    )

    if not fills:
        logger.info("Portfolio is already at target weights. No trades needed.")
        return False

    logger.info(f"Generated {len(fills)} fills:")
    for fill in fills:
        logger.info(f"  {fill}")

    # 6. Submit orders to Alpaca
    #    - Buys: simple market order first, then attach SL/TP anchored to fill price
    #    - Sells: simple market order
    logger.info("Submitting orders to Alpaca...")
    submitted_orders: list[dict] = []  # Track for fill verification
    for fill in fills:
        try:
            symbol = fill.order.ticker
            side = fill.order.side.lower()  # BUY/SELL -> buy/sell
            qty = round(abs(fill.fill_quantity), 4)  # Fractional shares OK (Fix #32)
            if qty < 0.0001:
                continue

            # Submit simple market order (SL/TP added after fill for buys)
            broker_order = BrokerOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="market",
                time_in_force="day",
            )

            # Capture decision price at submission time (before market impact)
            # for accurate TCA implementation shortfall calculation.
            decision_price = prices.get(symbol)

            order_id = broker.submit_order(broker_order)
            submitted_orders.append(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "needs_sl_tp": side == "buy",
                    "decision_price": decision_price,
                }
            )
            logger.info(f"  Submitted: {side.upper()} {qty:.4f} {symbol} -> Order ID: {order_id}")
        except Exception as e:
            logger.error(f"  Failed to submit order for {fill.order.ticker}: {e}")

    if not submitted_orders:
        logger.info("No orders were submitted.")
        return False

    # 7. Verify fills — poll each order until terminal state, then attach SL/TP
    logger.info(f"Verifying {len(submitted_orders)} order fills...")
    filled_count = 0
    partial_count = 0
    failed_count = 0

    for entry in submitted_orders:
        result = _verify_order_fill(broker, entry["order_id"], entry["symbol"], entry["qty"])
        status = result["status"]

        if status == "filled" or status == "partially_filled":
            actual_qty = float(result["filled_qty"])

            # H-PARTIAL fix: re-query broker for authoritative filled_qty
            # to reconcile partial fills that may have completed after our poll
            try:
                final_order = broker.get_order(entry["order_id"])
                if final_order and final_order.filled_qty is not None:
                    broker_qty = float(final_order.filled_qty)
                    if abs(broker_qty - actual_qty) > 1e-6:
                        logger.info(
                            f"  Reconciled fill qty for {entry['symbol']}: "
                            f"{actual_qty:.4f} -> {broker_qty:.4f}"
                        )
                        actual_qty = broker_qty
            except Exception as e:
                logger.warning(f"  Could not reconcile fill for {entry['symbol']}: {e}")

            if status == "filled":
                filled_count += 1
                logger.info(f"  FILLED: {entry['side'].upper()} {actual_qty:.4f} {entry['symbol']}")
            else:
                partial_count += 1
                logger.warning(
                    f"  PARTIAL: {entry['side'].upper()} {entry['symbol']} — "
                    f"filled {actual_qty}/{entry['qty']}"
                )

            # Attach SL/TP orders anchored to actual fill price (Fix #33)
            # C1 fix: submit as OCO pair so one filling cancels the other
            # C4 fix: size SL/TP for TOTAL position, not just this fill increment
            if (
                entry.get("needs_sl_tp")
                and result["filled_avg_price"] is not None
                and actual_qty > 0
            ):
                # C-OCO-2 fix: compute sl_tp_qty and sl_price unconditionally
                # before OCO try block so fallback always has valid values.
                fill_price = float(result["filled_avg_price"])

                # C4 fix: get total position qty from broker for SL/TP sizing
                try:
                    total_pos = broker.get_position(entry["symbol"])
                    sl_tp_qty = round(abs(total_pos.qty), 4) if total_pos else actual_qty
                except Exception as e:
                    logger.warning(
                        f"  get_position failed for {entry['symbol']}: {e}"
                        " — using fill qty for SL/TP"
                    )
                    sl_tp_qty = actual_qty

                # Try ATR-based stops first, fall back to fixed %
                # Use pre-fetched OHLCV from ML pipeline to avoid 10 extra
                # yfinance calls per rebalance cycle.
                _ohlcv_cache = _predict_mod._last_raw_ohlcv
                _sym_ohlcv = None
                if _ohlcv_cache is not None:
                    try:
                        if isinstance(_ohlcv_cache.columns, pd.MultiIndex):
                            _sym_ohlcv = _ohlcv_cache.xs(entry["symbol"], axis=1, level=0)
                        elif entry["symbol"] in _ohlcv_cache.columns:
                            _sym_ohlcv = _ohlcv_cache[[entry["symbol"]]]
                    except (KeyError, TypeError):
                        pass
                atr = get_current_atr(entry["symbol"], ohlcv_data=_sym_ohlcv)
                if atr is not None and atr > 0:
                    sl_price = round(fill_price - (ATR_SL_MULTIPLIER * atr), 2)
                    tp_price = round(fill_price + (ATR_TP_MULTIPLIER * atr), 2)
                    sl_label = f"{ATR_SL_MULTIPLIER}x ATR ({atr:.2f})"
                else:
                    sl_price = round(fill_price * (1 - STOP_LOSS_PCT), 2)
                    tp_price = round(fill_price * (1 + TAKE_PROFIT_PCT), 2)
                    sl_label = f"{STOP_LOSS_PCT:.0%} fixed (ATR unavailable)"

                # Sanity: SL must be below fill, TP must be above
                if sl_price >= fill_price:
                    sl_price = round(fill_price * (1 - STOP_LOSS_PCT), 2)
                    sl_label = f"{STOP_LOSS_PCT:.0%} fixed (ATR sanity fallback)"
                if tp_price <= fill_price:
                    tp_price = round(fill_price * (1 + TAKE_PROFIT_PCT), 2)

                try:
                    # Cancel any existing SL/TP orders for this symbol to prevent
                    # order accumulation across trading cycles (P0-1 fix)
                    _cancel_existing_sl_tp_orders(broker, entry["symbol"])

                    # C1/C-OCO-1 fix: submit SL+TP as OCO pair (one-cancels-other).
                    # When SL fills, TP is automatically cancelled (and vice versa),
                    # preventing orphaned orders from creating naked short positions.
                    # R3-E-8 fix: remove redundant limit_price on parent —
                    # OCO legs define their own prices via take_profit/stop_loss.
                    oco_order = BrokerOrder(
                        symbol=entry["symbol"],
                        side="sell",
                        qty=sl_tp_qty,
                        order_type="limit",
                        time_in_force="gtc",
                        order_class="oco",
                        take_profit_limit_price=tp_price,
                        stop_loss_stop_price=sl_price,
                    )
                    oco_id = broker.submit_order(oco_order)
                    logger.info(
                        f"    OCO SL/TP attached: SELL {sl_tp_qty:.4f} {entry['symbol']} "
                        f"SL@${sl_price} ({sl_label}) TP@${tp_price} "
                        f"(from fill ${fill_price:.2f}) -> {oco_id}"
                    )
                except Exception as e:
                    logger.error(f"    Failed to attach SL/TP for {entry['symbol']}: {e}")
                    # Fallback: submit at least a stop-loss (better than nothing)
                    try:
                        sl_order = BrokerOrder(
                            symbol=entry["symbol"],
                            side="sell",
                            qty=sl_tp_qty,
                            order_type="stop",
                            stop_price=sl_price,
                            time_in_force="gtc",
                        )
                        broker.submit_order(sl_order)
                        logger.info(f"    SL-only fallback attached for {entry['symbol']}")
                    except Exception as e2:
                        logger.error(f"    SL fallback also failed: {e2}")
            elif entry.get("needs_sl_tp") and actual_qty > 0:
                logger.warning(
                    f"    No fill price available for {entry['symbol']} — "
                    f"SL/TP not attached (using quoted price as fallback)"
                )
                # Fallback: use the quoted price from the prices dict
                if entry["symbol"] in prices:
                    try:
                        quoted = prices[entry["symbol"]]
                        sl_price = round(quoted * (1 - STOP_LOSS_PCT), 2)
                        sl_order = BrokerOrder(
                            symbol=entry["symbol"],
                            side="sell",
                            qty=actual_qty,
                            order_type="stop",
                            stop_price=sl_price,
                            time_in_force="gtc",
                        )
                        broker.submit_order(sl_order)
                        logger.info(f"    SL fallback @ ${sl_price} (quoted price)")
                    except Exception as e:
                        logger.error(f"    SL fallback also failed for {entry['symbol']}: {e}")

        elif status in ("canceled", "cancelled", "expired"):
            failed_count += 1
            logger.warning(
                f"  CANCELLED/EXPIRED: {entry['side'].upper()} {entry['qty']:.4f} {entry['symbol']}"
            )
        elif status == "rejected":
            failed_count += 1
            logger.error(
                f"  REJECTED: {entry['side'].upper()} {entry['qty']:.4f} {entry['symbol']}"
            )
            _send_alert(
                f"Order REJECTED by broker: "
                f"{entry['side'].upper()} {entry['qty']:.4f} {entry['symbol']}",
                AlertSeverity.CRITICAL,
            )
        else:
            # timeout or unknown
            logger.warning(
                f"  TIMEOUT: {entry['side'].upper()} {entry['qty']:.4f} {entry['symbol']} "
                f"— order still open after {ORDER_POLL_TIMEOUT_SECS}s"
            )

    logger.info(
        f"Fill verification: {filled_count} filled, "
        f"{partial_count} partial, {failed_count} failed "
        f"out of {len(submitted_orders)} submitted."
    )

    # Record trades for TCA analysis
    if tca_analyzer is not None:
        for entry in submitted_orders:
            try:
                order_info = broker.get_order(entry["order_id"])
                if order_info and order_info.filled_avg_price:
                    # Use pre-submission decision price (captured at order time)
                    # for accurate implementation shortfall measurement.
                    dp = entry.get("decision_price") or order_info.filled_avg_price
                    tca_analyzer.add_trade(
                        TradeRecord(
                            symbol=entry["symbol"],
                            side=entry["side"].upper(),
                            order_qty=entry["qty"],
                            fill_qty=float(order_info.filled_qty or 0),
                            fill_price=float(order_info.filled_avg_price),
                            decision_price=float(dp),
                            timestamp=datetime.now(tz=ZoneInfo("America/New_York")),
                        )
                    )
            except Exception as e:
                logger.debug(f"TCA recording failed for {entry['symbol']}: {e}")

        if tca_analyzer.n_trades > 0:
            tca_summary = tca_analyzer.summary()
            logger.info(
                f"TCA: {tca_summary.get('n_trades', 0)} trades, "
                f"mean IS={tca_summary.get('mean_is_bps', 0):.1f}bps, "
                f"mean fill rate={tca_summary.get('mean_fill_rate', 0):.1%}"
            )
            # Persist TCA trades for recovery across restarts
            try:
                tca_analyzer.save(Path("data/cache/tca_trades.json"))
            except Exception as e:
                logger.debug(f"TCA save failed: {e}")
            # Persist TCA report for /tca Telegram command
            try:
                tca_report_path = Path("data/cache/tca_report.json")
                tca_report_path.parent.mkdir(parents=True, exist_ok=True)
                import json as _json

                tca_report_path.write_text(
                    _json.dumps(tca_analyzer.to_json(), indent=2, default=str)
                )
            except Exception as e:
                logger.debug(f"TCA report save failed: {e}")

    # Send trade summary alert (email + webhook)
    account = broker.get_account()
    send_trade_summary(
        filled=filled_count,
        partial=partial_count,
        failed=failed_count,
        total=len(submitted_orders),
        positions=target_weights,
        equity=account.equity if account else None,
    )

    return filled_count > 0 or partial_count > 0


def _seconds_until(target_ts) -> float:
    """Calculate seconds from now until a target timestamp.

    Handles both timezone-aware and naive datetimes from Alpaca's clock API.
    Returns at least 60 seconds to avoid tight loops.
    """
    from datetime import timezone

    now = datetime.now(tz=timezone.utc)
    if hasattr(target_ts, "timestamp"):
        # datetime-like object
        target = target_ts if target_ts.tzinfo else target_ts.replace(tzinfo=timezone.utc)
        delta = (target - now).total_seconds()
    elif isinstance(target_ts, str):
        # ISO-format string
        try:
            target = datetime.fromisoformat(target_ts.replace("Z", "+00:00"))
            delta = (target - now).total_seconds()
        except ValueError:
            delta = 3600.0  # fallback 1 hour
    else:
        delta = 3600.0  # fallback 1 hour

    return max(delta, 60.0)


def _prune_mlflow_runs(max_age_days: int = 30) -> None:
    """Delete MLflow runs older than *max_age_days* to prevent disk bloat."""
    try:
        import shutil
        from pathlib import Path as _P

        mlruns_dir = _P("mlruns")
        if not mlruns_dir.exists():
            return

        cutoff = time.time() - max_age_days * 86400
        pruned = 0
        for experiment_dir in mlruns_dir.iterdir():
            if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
                continue
            for run_dir in experiment_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                meta = run_dir / "meta.yaml"
                if meta.exists() and meta.stat().st_mtime < cutoff:
                    shutil.rmtree(run_dir)
                    pruned += 1
        if pruned:
            logger.info(f"Pruned {pruned} MLflow runs older than {max_age_days} days")
    except Exception as e:
        logger.debug(f"MLflow pruning skipped: {e}")


def main():
    logger.info("=" * 60)
    logger.info("  LIVE TRADING BOT — ML-Driven")
    logger.info(f"  Universe: Top {TOP_N_STOCKS} S&P 500 by ML rank")
    logger.info(f"  Optimizer: {OPTIMIZER_METHOD.upper()}")
    logger.info(f"  Rebalance: {REBALANCE_FREQUENCY} (day={REBALANCE_DAY})")
    logger.info(f"  Max position: {MAX_POSITION_WEIGHT:.0%}")
    logger.info(
        f"  Stop-loss: {ATR_SL_MULTIPLIER}x ATR (fallback {STOP_LOSS_PCT:.0%}) | "
        f"Take-profit: {ATR_TP_MULTIPLIER}x ATR (fallback {TAKE_PROFIT_PCT:.0%})"
    )
    logger.info(f"  Max drawdown: {MAX_DRAWDOWN_LIMIT:.0%}")
    logger.info("=" * 60)

    # Prune old MLflow runs to prevent disk bloat on VPS
    _prune_mlflow_runs(max_age_days=30)

    # M3 fix: paper_trading from env var (default True for safety)
    paper_trading = os.getenv("LIVE_TRADING", "").lower() != "true"

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        logger.error("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        sys.exit(1)

    broker = AlpacaBroker(
        paper_trading=paper_trading,
        api_key=api_key,
        api_secret=api_secret,
    )

    if not broker.connect():
        logger.error("Failed to connect to Alpaca. Exiting.")
        _send_alert("Failed to connect to Alpaca on startup. Exiting.", AlertSeverity.CRITICAL)
        sys.exit(1)

    # Startup alert — confirms the bot is alive after deploy/restart
    _send_alert(
        f"Bot started. Mode: {'PAPER' if paper_trading else 'LIVE'}. "
        f"Rebalance: {REBALANCE_FREQUENCY} (day={REBALANCE_DAY}).",
        AlertSeverity.INFO,
    )

    # Start Telegram command handler (background polling thread)
    # Allows querying bot state via /status, /positions, /equity, etc.
    start_telegram_command_handler()

    # --- SIGTERM handler (H4): persist state on container/systemd kill ---
    # C3 fix: Save comprehensive state including positions and trade tracking,
    # not just equity. Use nonlocal-safe pattern for bridge reference.
    _bridge_ref = [None]  # Mutable container accessible from closure

    def _handle_sigterm(signum, frame):
        logger.info(f"Received signal {signum} (SIGTERM) — initiating graceful shutdown...")
        bridge_obj = _bridge_ref[0]
        state = {
            "last_shutdown": datetime.now(tz=ZoneInfo("America/New_York")).isoformat(),
            "reason": "sigterm",
        }
        if bridge_obj is not None:
            state["final_equity"] = bridge_obj.equity
            state["positions"] = {
                t: {"qty": p.quantity, "avg_cost": p.avg_cost}
                for t, p in bridge_obj.positions.items()
                if p.quantity != 0
            }
            state["cash"] = bridge_obj.cash
        _save_bot_state(state)
        _send_alert("Received SIGTERM — shutting down gracefully.", AlertSeverity.INFO)
        broker.disconnect()
        logger.info("SIGTERM shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # --- Load persisted state (M1): restore state from previous run ---
    saved_state = _load_bot_state()
    if saved_state:
        logger.info(f"Loaded bot state from previous run: {saved_state}")
    else:
        logger.info("No previous bot state found — starting fresh.")

    # Check if we've already traded today (duplicate execution guard - P0-5)
    # C-STARTUP fix: only exit on non-rebalance days if we've done entry trades.
    # GTC SL/TP fills on non-rebalance days should not cause early exit.
    if should_rebalance_today() and _has_traded_today(broker):
        logger.info("Already traded today. Exiting to prevent duplicate execution.")
        broker.disconnect()
        sys.exit(0)

    risk_limits = RiskLimits(
        max_position_weight=MAX_POSITION_WEIGHT,
        max_portfolio_var_95=MAX_PORTFOLIO_VAR_95,
        max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
        max_sector_weight=DEFAULT_MAX_SECTOR_WEIGHT,
    )
    risk_manager = RiskManager(limits=risk_limits, sector_map=SECTOR_MAP)

    # Initialize regime detectors — HMM (primary) with threshold fallback
    regime_detector = RegimeDetector()
    hmm_detector: HMMRegimeDetector | None = None
    try:
        import yfinance as yf

        spy = yf.download("SPY", period="2y", interval="1d", progress=False)
        if spy is not None and len(spy) > 60:
            close = spy["Close"]
            # yfinance 0.2.x with auto_adjust=True returns MultiIndex columns
            if hasattr(close, "columns"):
                close = close.squeeze()
            spy_returns = close.pct_change().dropna()
            hmm_detector = HMMRegimeDetector()
            hmm_detector.fit(spy_returns)
            logger.info(f"HMM regime detector fitted on {len(spy_returns)} days of SPY returns")
        else:
            logger.warning("Insufficient SPY data for HMM — using threshold fallback")
    except Exception as e:
        logger.warning(f"HMM regime detector init failed: {e} — using threshold fallback")

    # Initialize TCA analyzer — load previous trades from disk if available
    TCA_TRADES_FILE = Path("data/cache/tca_trades.json")
    tca_analyzer = TransactionCostAnalyzer.load(TCA_TRADES_FILE)

    # Initialize risk engine with current positions' historical data
    _initialize_risk_engine(broker, risk_manager)

    # Create execution bridge once — persists equity curve, P&L, and weight
    # history across cycles. Each cycle syncs positions/equity from broker.
    # Alpaca is commission-free for equities, but we track estimated spread
    # costs at 2 bps per trade as a proxy for real-world execution costs.
    account = broker.get_account()
    estimated_spread_bps = _env_float("ESTIMATED_SPREAD_BPS", 0.0002)
    bridge = ExecutionBridge(
        risk_manager=risk_manager,
        initial_capital=account.equity,
        commission_rate=estimated_spread_bps,
    )
    _bridge_ref[0] = bridge  # C3 fix: make bridge accessible to SIGTERM handler

    try:
        while True:
            clock = broker.get_clock()
            is_open = clock["is_open"]

            if is_open:
                # Weekly rebalancing: skip non-rebalance days (Phase 1 cost reduction)
                if not should_rebalance_today():
                    next_rebal = _get_next_rebalance_date()
                    logger.info(
                        f"Not a rebalance day ({REBALANCE_FREQUENCY}, "
                        f"day={REBALANCE_DAY}). Next rebalance: {next_rebal.date()}. "
                        f"Sleeping until next market open..."
                    )
                    next_open = clock.get("next_open")
                    if next_open:
                        sleep_secs = _seconds_until(next_open)
                        time.sleep(sleep_secs)
                    else:
                        time.sleep(60 * 60 * SLEEP_MARKET_CLOSED_HOURS)
                    continue

                # Check if already traded today (duplicate execution guard inside loop)
                if _has_traded_today(broker):
                    logger.info("Already traded today. Skipping cycle.")
                    time.sleep(60 * 60 * 4)  # Sleep 4 hours
                    continue

                # Re-initialize risk engine each cycle to capture new positions
                _initialize_risk_engine(broker, risk_manager)

                # R3-E-3 fix: sync risk_manager.current_weights from broker
                # so trade-size, leverage, and sector checks use actual positions
                positions = broker.list_positions()
                account = broker.get_account()
                if positions and account.equity > 0:
                    risk_manager.current_weights = pd.Series(
                        {p.symbol: p.market_value / account.equity for p in positions}
                    )

                # Check portfolio-level risk before trading
                # Get current position weights from broker
                if positions and account.equity > 0:
                    weights = pd.Series(
                        {p.symbol: p.market_value / account.equity for p in positions}
                    )
                else:
                    weights = pd.Series()
                # C-DD fix: pass live equity curve for accurate drawdown calc
                live_eq = [pt["equity"] for pt in bridge.equity_curve]
                risk_checks = risk_manager.check_portfolio_risk(weights, live_equity_curve=live_eq)
                critical_violations = [
                    c for c in risk_checks if not c.passed and c.severity == "critical"
                ]
                if critical_violations:
                    for v in critical_violations:
                        logger.warning(f"RISK VIOLATION: {v.rule} — {v.message}")
                    # H5 fix: Drawdown violations should liquidate, not just skip
                    drawdown_violations = [
                        v for v in critical_violations if v.rule == "MAX_DRAWDOWN"
                    ]
                    if drawdown_violations:
                        logger.warning(
                            "MAX_DRAWDOWN exceeded — liquidating all positions (kill switch)."
                        )
                        _send_alert(
                            f"DRAWDOWN KILL SWITCH: "
                            f"{drawdown_violations[0].message}. Liquidating all positions.",
                            AlertSeverity.CRITICAL,
                        )
                        _liquidate_all_positions(broker)
                        time.sleep(60 * 30)
                        continue

                    # Non-drawdown violations (VaR, leverage): reduce exposure to
                    # compliance instead of skipping the cycle entirely.  Scale
                    # all current positions proportionally so the breached metric
                    # falls within limits.
                    var_violations = [v for v in critical_violations if v.rule == "MAX_VAR_95"]
                    if var_violations and weights is not None and len(weights) > 0:
                        v = var_violations[0]
                        # scale_factor = limit / actual, clamped to (0.1, 0.95)
                        scale = max(0.1, min(0.95, v.limit_value / max(v.metric_value, 1e-9)))
                        scaled = weights * scale
                        logger.warning(
                            f"VaR breach — scaling positions by {scale:.2f} "
                            f"(VaR {v.metric_value:.2%} → target {v.limit_value:.2%})"
                        )
                        _send_alert(
                            f"VaR breach: scaling positions by {scale:.2f}. "
                            f"VaR={v.metric_value:.2%}, limit={v.limit_value:.2%}",
                            AlertSeverity.CRITICAL,
                        )
                        var_prices = {
                            p.symbol: p.market_value / p.qty
                            for p in positions
                            if p.qty > 0 and p.market_value > 0
                        }
                        bridge.reconcile_target_weights(
                            target_weights=scaled.to_dict(),
                            prices=var_prices,
                            current_date=datetime.now(tz=ZoneInfo("America/New_York")).strftime(
                                "%Y-%m-%d"
                            ),
                        )
                    else:
                        logger.warning(
                            "Skipping trade cycle due to critical risk violations "
                            "(no reduce-to-compliance path available)."
                        )
                        violations_str = "; ".join(
                            f"{v.rule}: {v.message}" for v in critical_violations
                        )
                        _send_alert(
                            f"Skipping trade cycle — critical risk violations: {violations_str}",
                            AlertSeverity.CRITICAL,
                        )
                    time.sleep(60 * 30)
                    continue

                # Check market regime — HMM (primary) with threshold fallback
                vix = fetch_vix()
                spy_dd = fetch_spy_drawdown()

                # Try HMM regime detection first (probabilistic, adaptive)
                hmm_state: HMMRegimeState | None = None
                if hmm_detector is not None:
                    try:
                        import yfinance as yf

                        spy_recent = yf.download("SPY", period="3mo", interval="1d", progress=False)
                        if spy_recent is not None and len(spy_recent) > 25:
                            spy_close = spy_recent["Close"]
                            if hasattr(spy_close, "columns"):
                                spy_close = spy_close.squeeze()
                            spy_rets = spy_close.pct_change().dropna()
                            hmm_state = hmm_detector.predict_regime(spy_rets)
                            logger.info(f"HMM regime: {hmm_state.message}")
                    except Exception as e:
                        logger.warning(f"HMM prediction failed: {e}")

                if vix is not None and spy_dd is not None:
                    # Threshold-based regime (always computed for comparison/fallback)
                    threshold_state = regime_detector.get_regime_state(vix, spy_dd)

                    # Use HMM regime if available, otherwise threshold
                    if hmm_state is not None:
                        # Map HMM states to RegimeState regimes.
                        # HMM exposure multipliers: low_vol=1.0, normal=0.7, high_vol=0.3
                        # Only escalate to "halt" when BOTH HMM says high_vol AND
                        # threshold detector also says halt (consensus required for
                        # the most drastic action — full liquidation).
                        hmm_regime_map = {
                            "low_vol": "normal",
                            "normal": "normal",
                            "high_vol": "caution",
                        }
                        mapped_regime = hmm_regime_map.get(hmm_state.regime, "caution")

                        # Escalate to halt only when both HMM and threshold agree
                        if hmm_state.regime == "high_vol" and threshold_state.regime == "halt":
                            mapped_regime = "halt"

                        regime_state = RegimeState(
                            regime=mapped_regime,
                            vix=vix,
                            spy_drawdown=spy_dd,
                            exposure_multiplier=hmm_state.exposure_multiplier,
                            message=f"HMM: {hmm_state.message} (threshold: {threshold_state.regime})",
                        )
                        logger.info(
                            f"Using HMM regime: {hmm_state.regime} "
                            f"(exposure={hmm_state.exposure_multiplier:.0%}), "
                            f"threshold would say: {threshold_state.regime}"
                        )
                    else:
                        regime_state = threshold_state
                        logger.info(f"Market regime (threshold): {regime_state.message}")

                    if regime_state.regime == "halt":
                        logger.warning("Market regime HALT — liquidating all positions.")
                        _send_alert(
                            f"Market regime HALT: VIX={vix:.1f}, "
                            f"SPY drawdown={spy_dd:.1%}. Liquidating all positions.",
                            AlertSeverity.CRITICAL,
                        )
                        _liquidate_all_positions(broker)
                        time.sleep(60 * 60)  # Re-check in 1 hour
                        continue
                else:
                    # Fail-closed — assume caution regime when data unavailable
                    logger.warning(
                        "Could not fetch regime data (VIX/SPY). "
                        "Fail-closed: assuming CAUTION regime (50% exposure)."
                    )
                    _send_alert("Regime data unavailable — defaulting to caution mode.")
                    regime_state = RegimeState(
                        regime="caution",
                        vix=0.0,
                        spy_drawdown=0.0,
                        exposure_multiplier=0.5,
                        message="CAUTION (default): regime data unavailable, fail-closed.",
                    )

                traded = run_trading_cycle(
                    broker,
                    risk_manager,
                    bridge,
                    regime_detector=regime_detector,
                    regime_state=regime_state,
                    tca_analyzer=tca_analyzer,
                )

                # Persist state after trading cycle (P1-6)
                # R3-E-7 fix: persist equity_peak for drawdown tracking across restarts
                current_equity = bridge.equity
                prev_state = _load_bot_state()
                equity_peak = max(
                    current_equity,
                    prev_state.get("equity_peak", current_equity),
                )
                _save_bot_state(
                    {
                        "last_trade_date": datetime.now(
                            tz=ZoneInfo("America/New_York")
                        ).isoformat(),
                        "last_equity": current_equity,
                        "equity_peak": equity_peak,
                        "positions_count": len(bridge.positions),
                    }
                )

                # R3-E-12 fix: append to equity_history.json for dashboard visibility
                _append_equity_history(current_equity)

                # Heartbeat: periodic "all is well" signal (1/hour max)
                send_heartbeat(
                    f"Cycle complete. Equity: ${current_equity:,.2f}, "
                    f"positions: {len(bridge.positions)}, "
                    f"traded: {'yes' if traded else 'no'}"
                )

                if traded:
                    # R3-E-2 fix: re-fetch clock after trading to get fresh next_open
                    fresh_clock = broker.get_clock()
                    next_open = fresh_clock.get("next_open")
                    if next_open:
                        sleep_secs = _seconds_until(next_open)
                        logger.info(
                            f"Trades executed. Sleeping {sleep_secs / 3600:.1f}h "
                            f"until next open ({next_open})..."
                        )
                        time.sleep(sleep_secs)
                    else:
                        logger.info(f"Sleeping {SLEEP_AFTER_TRADE_HOURS}h (fallback)...")
                        time.sleep(60 * 60 * SLEEP_AFTER_TRADE_HOURS)
                else:
                    # No trades — check again in 30 min
                    logger.info("No trades executed. Rechecking in 30 minutes...")
                    time.sleep(60 * 30)
            else:
                next_open = clock.get("next_open", "unknown")
                if next_open and next_open != "unknown":
                    sleep_secs = _seconds_until(next_open)
                    logger.info(
                        f"Market closed. Next open: {next_open}. "
                        f"Sleeping {sleep_secs / 3600:.1f}h..."
                    )
                    time.sleep(sleep_secs)
                else:
                    logger.info(
                        f"Market closed (next open unknown). "
                        f"Sleeping {SLEEP_MARKET_CLOSED_HOURS}h..."
                    )
                    time.sleep(60 * 60 * SLEEP_MARKET_CLOSED_HOURS)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")
        _send_alert("Bot stopped by user (Ctrl+C).", AlertSeverity.INFO)
    except Exception as e:
        logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))
        _send_alert(
            f"FATAL: Bot crashed at {now_et.isoformat()}: {e}",
            AlertSeverity.CRITICAL,
        )
    finally:
        # M-SIGTERM fix: don't overwrite richer SIGTERM state
        saved = _load_bot_state()
        if not saved or saved.get("reason") != "sigterm":
            _save_bot_state(
                {
                    "last_shutdown": datetime.now(tz=ZoneInfo("America/New_York")).isoformat(),
                    "final_equity": bridge.equity if "bridge" in locals() else None,
                    "reason": "shutdown",
                }
            )
        broker.disconnect()
        logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    main()
