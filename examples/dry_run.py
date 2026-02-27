"""
Dry-run the ML pipeline without submitting orders.

Works any time — market open or closed. Shows:
  1. Model training stats
  2. Top 10 stock picks with ML scores
  3. HRP-optimized target weights
  4. What orders would be generated given current Alpaca positions
"""

import logging
import os
import sys

from python.alpha.predict import get_ml_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DryRun")


def main():
    logger.info("=" * 60)
    logger.info("  DRY RUN — ML Pipeline (no orders submitted)")
    logger.info("=" * 60)

    weights, stale_data = get_ml_weights(top_n=10, method="hrp")

    if stale_data:
        logger.warning("WARNING: Pipeline used stale cached data — results may be degraded")

    if not weights:
        logger.error("Pipeline returned no weights")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("  FINAL TARGET PORTFOLIO")
    print("=" * 60)
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:6s}  {w:6.2%}")
    print(f"\n  Total positions: {len(weights)}")
    print(f"  Sum of weights:  {sum(weights.values()):.4f}")

    # Show what $100k portfolio would look like
    equity = 100_000
    print(f"\n  Hypothetical ${equity:,.0f} allocation:")
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        dollars = w * equity
        print(f"  {ticker:6s}  ${dollars:>10,.2f}")

    # Optionally show current Alpaca positions for comparison
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if api_key and api_secret:
        print("\n" + "=" * 60)
        print("  CURRENT ALPACA POSITIONS")
        print("=" * 60)
        try:
            from python.brokers.alpaca_broker import AlpacaBroker

            broker = AlpacaBroker(paper_trading=True, api_key=api_key, api_secret=api_secret)
            if broker.connect():
                account = broker.get_account()
                print(f"  Equity:       ${account.equity:>12,.2f}")
                print(f"  Cash:         ${account.cash:>12,.2f}")
                print(f"  Buying Power: ${account.buying_power:>12,.2f}")

                positions = broker.list_positions()
                if positions:
                    print(f"\n  Open positions ({len(positions)}):")
                    for pos in positions:
                        current_weight = pos.market_value / account.equity
                        target = weights.get(pos.symbol, 0.0)
                        diff = target - current_weight
                        arrow = "+" if diff > 0 else ""
                        print(
                            f"  {pos.symbol:6s}  "
                            f"{pos.qty:>6.0f} shares  "
                            f"now={current_weight:5.1%}  "
                            f"target={target:5.1%}  "
                            f"delta={arrow}{diff:5.1%}"
                        )

                    # Show new positions to open
                    current_symbols = {pos.symbol for pos in positions}
                    new_buys = {t for t in weights if t not in current_symbols}
                    if new_buys:
                        print(f"\n  New positions to open: {', '.join(sorted(new_buys))}")

                    # Show positions to close
                    closes = {pos.symbol for pos in positions if pos.symbol not in weights}
                    if closes:
                        print(f"  Positions to close:   {', '.join(sorted(closes))}")
                else:
                    print("  No open positions")

                broker.disconnect()
        except Exception as e:
            logger.warning(f"Could not fetch Alpaca data: {e}")

    print("\n" + "=" * 60)
    print("  Dry run complete. No orders were submitted.")
    print("=" * 60)


if __name__ == "__main__":
    main()
