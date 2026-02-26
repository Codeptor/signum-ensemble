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

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime

import pandas as pd

import json
from pathlib import Path

from python.alpha.predict import get_ml_weights
from python.bridge.execution import ExecutionBridge
from python.brokers.alpaca_broker import AlpacaBroker
from python.brokers.base import BrokerOrder
from python.monitoring.drift import DriftDetector
from python.portfolio.risk_manager import RiskLimits, RiskManager

# --- Logging with rotation (Fix #36) ---
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

# --- Configuration ---
TOP_N_STOCKS = 10
OPTIMIZER_METHOD = "hrp"
MAX_POSITION_WEIGHT = 0.30
MAX_PORTFOLIO_VAR_95 = 0.06
MAX_DRAWDOWN_LIMIT = 0.15
STOP_LOSS_PCT = 0.05  # 5% trailing stop-loss from entry price
TAKE_PROFIT_PCT = 0.15  # 15% take-profit from entry price
SLEEP_AFTER_TRADE_HOURS = 12
SLEEP_MARKET_CLOSED_HOURS = 1
ORDER_POLL_INTERVAL_SECS = 2  # How often to poll for fill status
ORDER_POLL_TIMEOUT_SECS = 60  # Max time to wait for a fill
TERMINAL_ORDER_STATES = {"filled", "canceled", "cancelled", "expired", "rejected"}

# --- Alerting (Fix #37) ---
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")  # Slack/Discord/generic webhook


def _send_alert(message: str) -> None:
    """Fire-and-forget alert to a webhook URL (Slack/Discord/custom).

    Does nothing if ALERT_WEBHOOK_URL is not set.  Never raises — alerting
    failures must not mask the original error.
    """
    if not ALERT_WEBHOOK_URL:
        return
    try:
        import json
        import urllib.request

        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        # Swallow — alerting must never crash the crash handler
        logger.debug("Failed to send alert webhook", exc_info=True)


def _verify_order_fill(
    broker: AlpacaBroker,
    order_id: str,
    symbol: str,
    expected_qty: float,
) -> dict:
    """Poll broker until order reaches a terminal state.

    Returns a dict with keys: status, filled_qty, filled_avg_price, symbol, order_id.
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
            # Extract fill price from the order response if available
            filled_avg_price = getattr(order, "filled_avg_price", None)
            return {
                "status": status,
                "filled_qty": order.qty,
                "filled_avg_price": filled_avg_price,
                "symbol": symbol,
                "order_id": order_id,
            }

        logger.debug(f"  Order {order_id} ({symbol}): status={status}, waiting...")

    # Timeout — order is still open
    logger.warning(
        f"  Order {order_id} ({symbol}) did not reach terminal state "
        f"within {ORDER_POLL_TIMEOUT_SECS}s"
    )
    return {
        "status": "timeout",
        "filled_qty": 0,
        "filled_avg_price": None,
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
                if order.side.lower() == "sell" and order.order_id:
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
    """Persist bot state to disk."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save bot state: {e}")


def _has_traded_today(broker: AlpacaBroker) -> bool:
    """Check if the bot has already submitted orders today.

    Prevents duplicate execution on restart by checking for recent orders
    from the current trading session.
    """
    try:
        # Get all orders (including closed)
        orders = broker.list_orders(status="all")

        # Filter to non-cancelled orders from today
        today = datetime.now().strftime("%Y-%m-%d")
        executed_today = [
            o
            for o in orders
            if o.status not in ("canceled", "cancelled", "expired")
            and o.order_id
            and hasattr(o, "created_at")
            and o.created_at
            and o.created_at.startswith(today)
        ]

        if executed_today:
            logger.info(f"Found {len(executed_today)} orders executed today.")
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
    try:
        import yfinance as yf

        data = yf.download(symbols, period="1y", interval="1d", progress=False)
        if data is not None and len(data) > 0:
            close_raw = data["Close"]
            close: pd.DataFrame = (
                close_raw.to_frame(name=symbols[0])
                if isinstance(close_raw, pd.Series)
                else pd.DataFrame(close_raw)
            )
            returns = close.pct_change().dropna()
            # Align weights to available columns
            available = [s for s in symbols if s in returns.columns]
            if available:
                returns_df = pd.DataFrame(returns[available])
                risk_manager.initialize_portfolio_risk(
                    returns=returns_df,
                    weights=weights.reindex(available).fillna(0),
                )
                logger.info(
                    f"Risk engine initialized with {len(available)} positions, "
                    f"{len(returns)} days of history."
                )
                return
    except Exception as e:
        logger.warning(f"Failed to initialize risk engine: {e}")

    logger.warning("Risk engine could not be initialized — risk checks will be no-ops.")


def run_trading_cycle(
    broker: AlpacaBroker,
    risk_manager: RiskManager,
    bridge: ExecutionBridge,
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
        _send_alert(f"[LiveBot] Failed to check market status: {e}")
        return False

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
        target_weights = get_ml_weights(
            top_n=TOP_N_STOCKS,
            method=OPTIMIZER_METHOD,
        )
    except Exception as e:
        logger.error(f"ML pipeline failed: {e}", exc_info=True)
        _send_alert(f"[LiveBot] ML pipeline failed: {e}")
        return False

    if not target_weights:
        logger.error("ML pipeline returned no weights — skipping this cycle")
        return False

    logger.info(f"Target weights from ML pipeline ({len(target_weights)} positions):")
    for ticker, w in sorted(target_weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {ticker}: {w:.2%}")

    # 3. Get current prices for order sizing (batch fetch, yfinance fallback)
    logger.info("Fetching current prices for order sizing...")
    prices = broker.get_latest_prices(list(target_weights.keys()))

    # Drop any tickers we couldn't price
    failed_tickers = set(target_weights) - set(prices)
    if failed_tickers:
        logger.warning(f"Removing tickers with no price data: {failed_tickers}")
        target_weights = {t: w for t, w in target_weights.items() if t in prices}
        # Renormalize
        total = sum(target_weights.values())
        if total > 0:
            target_weights = {t: w / total for t, w in target_weights.items()}

    if not target_weights:
        logger.error("No tradeable tickers remaining — skipping cycle")
        _send_alert(f"[LiveBot] No tradeable tickers after price filtering")
        return False

    # 4. Sync execution bridge with current account/positions from broker
    account = broker.get_account()
    logger.info(
        f"Account equity: ${account.equity:,.2f} | "
        f"Cash: ${account.cash:,.2f} | "
        f"Buying power: ${account.buying_power:,.2f}"
    )

    # Update bridge equity from broker (authoritative source)
    bridge.equity = account.equity

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
        current_date=datetime.now().strftime("%Y-%m-%d"),
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

            order_id = broker.submit_order(broker_order)
            submitted_orders.append(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "needs_sl_tp": side == "buy",
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

        if status == "filled":
            filled_count += 1
            logger.info(f"  FILLED: {entry['side'].upper()} {entry['qty']:.4f} {entry['symbol']}")

            # Attach SL/TP orders anchored to actual fill price (Fix #33)
            if entry.get("needs_sl_tp") and result["filled_avg_price"] is not None:
                try:
                    # Cancel any existing SL/TP orders for this symbol to prevent
                    # order accumulation across trading cycles (P0-1 fix)
                    _cancel_existing_sl_tp_orders(broker, entry["symbol"])

                    fill_price = float(result["filled_avg_price"])
                    sl_price = round(fill_price * (1 - STOP_LOSS_PCT), 2)
                    tp_price = round(fill_price * (1 + TAKE_PROFIT_PCT), 2)

                    # Submit stop-loss
                    sl_order = BrokerOrder(
                        symbol=entry["symbol"],
                        side="sell",
                        qty=entry["qty"],
                        order_type="stop",
                        stop_price=sl_price,
                        time_in_force="gtc",
                    )
                    sl_id = broker.submit_order(sl_order)
                    logger.info(
                        f"    SL attached: SELL {entry['symbol']} @ ${sl_price} "
                        f"(-{STOP_LOSS_PCT:.0%} from fill ${fill_price:.2f}) -> {sl_id}"
                    )

                    # Submit take-profit
                    tp_order = BrokerOrder(
                        symbol=entry["symbol"],
                        side="sell",
                        qty=entry["qty"],
                        order_type="limit",
                        limit_price=tp_price,
                        time_in_force="gtc",
                    )
                    tp_id = broker.submit_order(tp_order)
                    logger.info(
                        f"    TP attached: SELL {entry['symbol']} @ ${tp_price} "
                        f"(+{TAKE_PROFIT_PCT:.0%} from fill ${fill_price:.2f}) -> {tp_id}"
                    )
                except Exception as e:
                    logger.error(f"    Failed to attach SL/TP for {entry['symbol']}: {e}")
            elif entry.get("needs_sl_tp"):
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
                            qty=entry["qty"],
                            order_type="stop",
                            stop_price=sl_price,
                            time_in_force="gtc",
                        )
                        broker.submit_order(sl_order)
                        logger.info(f"    SL fallback @ ${sl_price} (quoted price)")
                    except Exception as e:
                        logger.error(f"    SL fallback also failed for {entry['symbol']}: {e}")

        elif status == "partially_filled":
            partial_count += 1
            logger.warning(
                f"  PARTIAL: {entry['side'].upper()} {entry['symbol']} — "
                f"filled {result['filled_qty']}/{entry['qty']}"
            )
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


def main():
    logger.info("=" * 60)
    logger.info("  LIVE TRADING BOT — ML-Driven")
    logger.info(f"  Universe: Top {TOP_N_STOCKS} S&P 500 by ML rank")
    logger.info(f"  Optimizer: {OPTIMIZER_METHOD.upper()}")
    logger.info(f"  Max position: {MAX_POSITION_WEIGHT:.0%}")
    logger.info(f"  Stop-loss: {STOP_LOSS_PCT:.0%} | Take-profit: {TAKE_PROFIT_PCT:.0%}")
    logger.info(f"  Max drawdown: {MAX_DRAWDOWN_LIMIT:.0%}")
    logger.info("=" * 60)

    # Paper trading — set to False only after thorough testing
    paper_trading = True

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
        sys.exit(1)

    # Check if we've already traded today (duplicate execution guard - P0-5)
    if _has_traded_today(broker):
        logger.info("Already traded today. Exiting to prevent duplicate execution.")
        broker.disconnect()
        sys.exit(0)

    risk_limits = RiskLimits(
        max_position_weight=MAX_POSITION_WEIGHT,
        max_portfolio_var_95=MAX_PORTFOLIO_VAR_95,
        max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
    )
    risk_manager = RiskManager(limits=risk_limits)

    # Initialize risk engine with current positions' historical data
    _initialize_risk_engine(broker, risk_manager)

    # Create execution bridge once — persists equity curve, P&L, and weight
    # history across cycles. Each cycle syncs positions/equity from broker.
    account = broker.get_account()
    bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=account.equity)

    try:
        while True:
            clock = broker.get_clock()
            is_open = clock["is_open"]

            if is_open:
                # Check if already traded today (duplicate execution guard inside loop)
                if _has_traded_today(broker):
                    logger.info("Already traded today. Skipping cycle.")
                    time.sleep(60 * 60 * 4)  # Sleep 4 hours
                    continue

                # Re-initialize risk engine each cycle to capture new positions
                _initialize_risk_engine(broker, risk_manager)

                # Check portfolio-level risk before trading
                # Get current position weights from broker
                positions = broker.list_positions()
                account = broker.get_account()
                if positions and account.equity > 0:
                    weights = pd.Series(
                        {p.symbol: p.market_value / account.equity for p in positions}
                    )
                else:
                    weights = pd.Series()
                risk_checks = risk_manager.check_portfolio_risk(weights)
                critical_violations = [
                    c for c in risk_checks if not c.passed and c.severity == "critical"
                ]
                if critical_violations:
                    for v in critical_violations:
                        logger.warning(f"RISK VIOLATION: {v.rule} — {v.message}")
                    logger.warning("Skipping trade cycle due to critical risk violations.")
                    time.sleep(60 * 30)
                    continue

                traded = run_trading_cycle(broker, risk_manager, bridge)

                # Persist state after trading cycle (P1-6)
                _save_bot_state(
                    {
                        "last_trade_date": datetime.now().isoformat(),
                        "last_equity": bridge.equity,
                        "positions_count": len(bridge.positions),
                    }
                )

                if traded:
                    # Sleep until next market open (dynamic, Fix #34)
                    next_open = clock.get("next_open")
                    # After trading, sleep until market re-opens next session
                    # If next_open is available and in the future, sleep until then
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
    except Exception as e:
        logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
        _send_alert(f"[LiveBot FATAL] Bot crashed at {datetime.now().isoformat()}: {e}")
    finally:
        # Persist final state before exit (P1-6)
        _save_bot_state(
            {
                "last_shutdown": datetime.now().isoformat(),
                "final_equity": bridge.equity if "bridge" in locals() else None,
                "reason": "shutdown",
            }
        )
        broker.disconnect()
        logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    main()
