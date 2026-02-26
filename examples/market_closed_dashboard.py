"""
Market Closed Dashboard - What you can do while market is closed.

Shows account status, pending orders, and prepares for next session.
"""

import os
from datetime import datetime, timedelta

from python.brokers.alpaca_broker import AlpacaPaperTrading
from python.portfolio.risk_manager import RiskLimits, RiskManager


def format_time_until(target_time):
    """Format time until market opens."""
    now = datetime.now(target_time.tzinfo)
    diff = target_time - now

    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)

    return f"{hours}h {minutes}m"


def main():
    """Run market closed dashboard."""
    # Connect to broker
    broker = AlpacaPaperTrading(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
    )

    if not broker.connect():
        print("❌ Failed to connect")
        return

    try:
        print("=" * 60)
        print("📊 MARKET CLOSED DASHBOARD")
        print("=" * 60)

        # Market clock
        clock = broker.get_clock()
        next_open = clock["next_open"]
        next_close = clock["next_close"]

        print(f"\n🕐 Market Status:")
        print(f"   Current Time (ET): {clock['timestamp']}")
        print(f"   Status: 🔴 CLOSED")
        print(f"   Opens In: {format_time_until(next_open)}")
        print(f"   Next Open: {next_open}")
        print(f"   Next Close: {next_close}")

        # Account info
        account = broker.get_account()
        print(f"\n💰 Account Summary:")
        print(f"   Account ID: {account.account_id}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Status: {account.status}")

        # Positions
        positions = broker.list_positions()
        print(f"\n📈 Current Positions ({len(positions)}):")
        if positions:
            total_value = 0
            total_pl = 0
            for pos in positions:
                print(f"   {pos.symbol}:")
                print(f"      Qty: {pos.qty}")
                print(f"      Avg Entry: ${pos.avg_entry_price:.2f}")
                print(f"      Market Value: ${pos.market_value:,.2f}")
                print(
                    f"      Unrealized P&L: ${pos.unrealized_pl:,.2f} ({pos.unrealized_plpc:+.2%})"
                )
                total_value += pos.market_value
                total_pl += pos.unrealized_pl
            print(f"\n   Total Position Value: ${total_value:,.2f}")
            print(f"   Total Unrealized P&L: ${total_pl:,.2f}")
        else:
            print("   No open positions")

        # Orders
        open_orders = broker.list_orders("open")
        all_orders = broker.list_orders("all")

        print(f"\n📋 Orders:")
        print(f"   Open Orders: {len(open_orders)}")
        print(f"   Total Orders Today: {len(all_orders)}")

        if open_orders:
            print(f"\n   Pending Orders:")
            for order in open_orders:
                print(
                    f"      {order.symbol} {order.side.upper()} {order.qty} @ {order.order_type.upper()}"
                )

        if all_orders:
            print(f"\n   Recent Order History:")
            for order in all_orders[-5:]:  # Last 5
                print(
                    f"      {order.symbol} {order.side.upper()} {order.qty} @ {order.order_type.upper()}"
                )

        # Historical data example
        print(f"\n📊 Historical Data (AAPL - Last 5 Days):")
        try:
            bars = broker.get_bars("AAPL", timeframe="1D", limit=5)
            for bar in bars:
                print(
                    f"   {bar['timestamp'][:10]}: ${bar['close']:.2f} (Vol: {bar['volume']:,.0f})"
                )
        except Exception as e:
            print(f"   Could not fetch: {e}")

        # Risk check
        print(f"\n⚠️  Risk Check:")
        if positions:
            position_value = sum(p.market_value for p in positions)
            concentration = position_value / account.equity if account.equity > 0 else 0
            print(f"   Portfolio Concentration: {concentration:.1%}")
            print(
                f"   Cash Reserve: {(account.cash / account.equity):.1%}"
                if account.equity > 0
                else "   N/A"
            )
        else:
            print(f"   No positions to analyze")

        print(f"\n" + "=" * 60)
        print("✅ Ready for market open!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
