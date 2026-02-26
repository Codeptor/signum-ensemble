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
SLEEP_AFTER_TRADE_HOURS = 12
SLEEP_MARKET_CLOSED_HOURS = 1


def run_trading_cycle(broker: AlpacaBroker, risk_manager: RiskManager) -> bool:
    """The core daily trading logic. Returns True if trades were executed."""
    logger.info("=" * 60)
    logger.info("Starting daily trading cycle...")
    logger.info("=" * 60)

    # 1. Run the ML pipeline to get target weights
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

    # 2. Get current prices for order sizing
    prices = {}
    for sym in target_weights:
        try:
            prices[sym] = broker.get_latest_price(sym)
        except Exception as e:
            logger.warning(f"Could not fetch price for {sym}: {e}")

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

    # 3. Set up execution bridge with current account state
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

    # 4. Reconcile portfolio to target weights
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

    # 5. Submit orders to Alpaca
    logger.info("Submitting orders to Alpaca...")
    orders_submitted = 0
    for fill in fills:
        try:
            symbol = fill.order.ticker
            side = fill.order.side.lower()  # BUY/SELL -> buy/sell
            qty = int(abs(fill.fill_quantity))  # Alpaca needs whole shares
            if qty == 0:
                continue
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

    try:
        while True:
            clock = broker.get_clock()
            is_open = clock["is_open"]

            if is_open:
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
