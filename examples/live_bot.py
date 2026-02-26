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
import os
import sys
import time
from datetime import datetime

import pandas as pd

from python.alpha.predict import get_ml_weights
from python.bridge.execution import ExecutionBridge
from python.brokers.alpaca_broker import AlpacaBroker
from python.brokers.base import BrokerOrder
from python.portfolio.risk_manager import RiskLimits, RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_bot.log"),
    ],
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


def run_trading_cycle(broker: AlpacaBroker, risk_manager: RiskManager) -> bool:
    """The core daily trading logic. Returns True if trades were executed."""
    logger.info("=" * 60)
    logger.info("Starting daily trading cycle...")
    logger.info("=" * 60)

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
        return False

    # 4. Set up execution bridge with current account state
    account = broker.get_account()
    logger.info(
        f"Account equity: ${account.equity:,.2f} | "
        f"Cash: ${account.cash:,.2f} | "
        f"Buying power: ${account.buying_power:,.2f}"
    )

    bridge = ExecutionBridge(risk_manager=risk_manager, initial_capital=account.equity)

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

    # 6. Submit orders to Alpaca (bracket orders for buys, simple for sells)
    logger.info("Submitting orders to Alpaca...")
    orders_submitted = 0
    for fill in fills:
        try:
            symbol = fill.order.ticker
            side = fill.order.side.lower()  # BUY/SELL -> buy/sell
            qty = int(abs(fill.fill_quantity))  # Alpaca needs whole shares
            if qty == 0:
                continue

            if side == "buy" and symbol in prices:
                # Bracket order: market buy with stop-loss and take-profit
                entry_price = prices[symbol]
                stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
                take_profit_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

                broker_order = BrokerOrder(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="market",
                    time_in_force="gtc",
                    order_class="bracket",
                    take_profit_limit_price=take_profit_price,
                    stop_loss_stop_price=stop_price,
                )
                logger.info(
                    f"  Bracket: BUY {qty} {symbol} | "
                    f"SL ${stop_price} (-{STOP_LOSS_PCT:.0%}) | "
                    f"TP ${take_profit_price} (+{TAKE_PROFIT_PCT:.0%})"
                )
            else:
                # Simple market order for sells
                broker_order = BrokerOrder(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="market",
                    time_in_force="day",
                )

            order_id = broker.submit_order(broker_order)
            orders_submitted += 1
            logger.info(f"  Submitted: {side.upper()} {qty} {symbol} -> Order ID: {order_id}")
        except Exception as e:
            logger.error(f"  Failed to submit order for {fill.order.ticker}: {e}")

    logger.info(f"Trading cycle complete. {orders_submitted} orders submitted.")
    return orders_submitted > 0


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

    risk_limits = RiskLimits(
        max_position_weight=MAX_POSITION_WEIGHT,
        max_portfolio_var_95=MAX_PORTFOLIO_VAR_95,
        max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
    )
    risk_manager = RiskManager(limits=risk_limits)

    # Initialize risk engine with current positions' historical data
    _initialize_risk_engine(broker, risk_manager)

    try:
        while True:
            clock = broker.get_clock()
            is_open = clock["is_open"]

            if is_open:
                # Re-initialize risk engine each cycle to capture new positions
                _initialize_risk_engine(broker, risk_manager)

                # Check portfolio-level risk before trading
                risk_checks = risk_manager.check_portfolio_risk(pd.Series())
                critical_violations = [
                    c for c in risk_checks if not c.passed and c.severity == "critical"
                ]
                if critical_violations:
                    for v in critical_violations:
                        logger.warning(f"RISK VIOLATION: {v.rule} — {v.message}")
                    logger.warning("Skipping trade cycle due to critical risk violations.")
                    time.sleep(60 * 30)
                    continue

                traded = run_trading_cycle(broker, risk_manager)

                if traded:
                    logger.info(f"Sleeping {SLEEP_AFTER_TRADE_HOURS}h until next cycle...")
                    time.sleep(60 * 60 * SLEEP_AFTER_TRADE_HOURS)
                else:
                    # No trades — check again in 30 min
                    logger.info("No trades executed. Rechecking in 30 minutes...")
                    time.sleep(60 * 30)
            else:
                next_open = clock.get("next_open", "unknown")
                logger.info(
                    f"Market closed. Next open: {next_open}. "
                    f"Sleeping {SLEEP_MARKET_CLOSED_HOURS}h..."
                )
                time.sleep(60 * 60 * SLEEP_MARKET_CLOSED_HOURS)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
    finally:
        broker.disconnect()
        logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    main()
