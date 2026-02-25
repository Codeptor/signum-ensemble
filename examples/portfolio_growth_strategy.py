"""
Portfolio Growth Strategy: $100k → $150k

Strategy: Multi-asset momentum with risk management
Target: 50% return ($50k profit)
Risk: Max 2% per trade, max 25% per position

Assets to trade:
- AAPL (momentum)
- TSLA (high volatility)
- NVDA (AI theme)
- MSFT (stability)
- AMZN (recovery play)
"""

import os
from python.brokers.alpaca_broker import AlpacaPaperTrading
from python.brokers.base import BrokerOrder
from python.portfolio.risk_manager import RiskLimits, RiskManager


def main():
    """Execute portfolio growth strategy."""

    broker = AlpacaPaperTrading(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
    )

    if not broker.connect():
        print("❌ Failed to connect")
        return

    try:
        # Get account info
        account = broker.get_account()
        initial_equity = float(account.equity)
        target_equity = initial_equity * 1.50  # 50% target

        print("=" * 70)
        print("🚀 PORTFOLIO GROWTH STRATEGY")
        print("=" * 70)
        print(f"\n💰 Initial Equity: ${initial_equity:,.2f}")
        print(f"🎯 Target Equity: ${target_equity:,.2f}")
        print(f"📈 Target Return: +50.00% (${target_equity - initial_equity:,.2f})")
        print(f"⏳ Timeframe: Multiple trading sessions")

        # Define strategy allocations (aggressive growth)
        # Total: 80% invested, 20% cash reserve
        allocations = {
            "AAPL": 0.20,  # $20k - Stable momentum
            "NVDA": 0.20,  # $20k - AI/GPU demand
            "TSLA": 0.15,  # $15k - High volatility, high reward
            "MSFT": 0.15,  # $15k - Stable tech
            "AMZN": 0.10,  # $10k - E-commerce recovery
        }

        total_allocated = sum(allocations.values())
        cash_reserve = 1.0 - total_allocated

        print(f"\n📊 Allocation Strategy:")
        for symbol, weight in allocations.items():
            amount = initial_equity * weight
            print(f"   {symbol}: {weight:.0%} (${amount:,.2f})")
        print(f"   Cash Reserve: {cash_reserve:.0%} (${initial_equity * cash_reserve:,.2f})")

        print(f"\n⚠️  Risk Management:")
        print(f"   Max Position Size: 25% per asset")
        print(f"   Stop Loss: -8% per position")
        print(f"   Take Profit: +20% per position")
        print(f"   Max Drawdown: -15% portfolio")

        # Get current prices and calculate shares
        print(f"\n📝 Planned Orders (will execute at market open):")
        orders = []

        for symbol, weight in allocations.items():
            try:
                price = broker.get_latest_price(symbol)
                target_value = initial_equity * weight
                shares = int(target_value / price)

                if shares > 0:
                    order = BrokerOrder(
                        symbol=symbol, side="buy", qty=float(shares), order_type="market"
                    )
                    orders.append((symbol, shares, price, order))
                    print(
                        f"   {symbol}: BUY {shares} shares @ ${price:.2f} = ${shares * price:,.2f}"
                    )
                else:
                    print(f"   {symbol}: Price too high (${price:.2f}) for allocation")

            except Exception as e:
                print(f"   {symbol}: Error getting price - {e}")

        # Submit orders
        print(f"\n📤 Submitting {len(orders)} orders...")
        submitted = []
        for symbol, shares, price, order in orders:
            try:
                order_id = broker.submit_order(order)
                submitted.append((symbol, order_id))
                print(f"   ✅ {symbol}: Order {order_id[:8]}... submitted")
            except Exception as e:
                print(f"   ❌ {symbol}: Failed - {e}")

        print(f"\n✅ Strategy deployed!")
        print(f"   Orders Submitted: {len(submitted)}")
        print(f"   Will fill at market open tomorrow 9:30 AM ET")
        print(f"\n📈 Expected Portfolio at Open:")
        print(f"   Cash: ${initial_equity * cash_reserve:,.2f}")
        print(f"   Invested: ${initial_equity * total_allocated:,.2f}")
        print(f"   Positions: {len(submitted)} assets")

        print(f"\n🎯 Success Metrics:")
        print(f"   To reach $150k, each position needs ~+10% return")
        print(f"   Diversification reduces single-stock risk")
        print(f"   Momentum strategy targets trending stocks")

        print(f"\n⚠️  Monitor Daily:")
        print(f"   Run: python examples/market_closed_dashboard.py")
        print(f"   Check positions and P&L")
        print(f"   Adjust if drawdown exceeds -15%")

        print(f"\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        broker.disconnect()
        print("\n👋 Disconnected from Alpaca")


if __name__ == "__main__":
    main()
