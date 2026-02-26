#!/usr/bin/env python3
"""Paper trading portfolio tracker CLI.

Displays a snapshot of paper trading portfolio health by querying the
Alpaca API and local bot state.  Designed to be run ad-hoc:

    uv run python examples/paper_trading_tracker.py
    uv run python examples/paper_trading_tracker.py --json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Try loading .env (dotenv is NOT a hard dependency)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOT_STATE_FILE = Path("data/bot_state.json")
MAX_POSITION_WEIGHT_WARN = 0.30  # warn if any position > 30%
MAX_DRAWDOWN_WARN = 0.15  # warn if drawdown > 15%

logger = logging.getLogger(__name__)


# ===================================================================
# Data collection helpers
# ===================================================================


def _connect_broker():
    """Create and connect an AlpacaBroker (paper mode).

    Returns the broker instance or None on failure.
    """
    try:
        from python.brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper_trading=True)
        broker.connect()
        return broker
    except Exception as exc:
        logger.warning("Could not connect to Alpaca: %s", exc)
        return None


def _fetch_account(broker) -> Optional[Dict[str, Any]]:
    try:
        acct = broker.get_account()
        return {
            "equity": acct.equity,
            "cash": acct.cash,
            "buying_power": acct.buying_power,
            "portfolio_value": acct.portfolio_value,
            "status": acct.status,
        }
    except Exception as exc:
        logger.warning("Could not fetch account: %s", exc)
        return None


def _fetch_positions(broker) -> List[Dict[str, Any]]:
    try:
        positions = broker.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry": p.avg_entry_price,
                "current_price": round(p.market_value / p.qty, 2) if p.qty else 0.0,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pl,
                "pnl_pct": round(p.unrealized_plpc * 100, 2),
            }
            for p in positions
        ]
    except Exception as exc:
        logger.warning("Could not fetch positions: %s", exc)
        return []


def _fetch_recent_orders(broker, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        orders = broker.list_orders(status="all")
        # Most recent first (Alpaca returns most-recent first by default)
        orders = orders[:limit]
        return [
            {
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "status": o.status,
                "created_at": o.created_at or "",
            }
            for o in orders
        ]
    except Exception as exc:
        logger.warning("Could not fetch orders: %s", exc)
        return []


def _load_bot_state() -> Dict[str, Any]:
    if not BOT_STATE_FILE.exists():
        return {}
    try:
        with open(BOT_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


# ===================================================================
# Computed metrics
# ===================================================================


def _compute_portfolio_metrics(
    positions: List[Dict[str, Any]], account: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    equity = account["equity"] if account else 0.0
    total_market_value = sum(p["market_value"] for p in positions)
    cash_alloc = equity - total_market_value if equity else 0.0

    weights = []
    for p in positions:
        w = abs(p["market_value"]) / equity * 100 if equity else 0.0
        weights.append(w)

    largest_weight = max(weights) if weights else 0.0
    # HHI = sum of squared weight fractions (0–1 scale)
    hhi = sum((w / 100) ** 2 for w in weights) if weights else 0.0

    return {
        "total_market_value": round(total_market_value, 2),
        "cash_allocation": round(cash_alloc, 2),
        "cash_pct": round(cash_alloc / equity * 100, 2) if equity else 0.0,
        "num_positions": len(positions),
        "largest_position_weight": round(largest_weight, 2),
        "hhi_concentration": round(hhi, 4),
    }


def _compute_risk_warnings(
    positions: List[Dict[str, Any]],
    account: Optional[Dict[str, Any]],
    bot_state: Dict[str, Any],
) -> List[str]:
    warnings: List[str] = []
    equity = account["equity"] if account else 0.0

    # Position concentration warnings
    for p in positions:
        if equity > 0:
            weight = abs(p["market_value"]) / equity
            if weight > MAX_POSITION_WEIGHT_WARN:
                warnings.append(
                    f"POSITION CONCENTRATION: {p['symbol']} is "
                    f"{weight * 100:.1f}% of portfolio (limit: "
                    f"{MAX_POSITION_WEIGHT_WARN * 100:.0f}%)"
                )

    # Drawdown from peak equity
    peak_equity = bot_state.get("last_equity") or bot_state.get("final_equity")
    if peak_equity and equity:
        peak_equity = float(peak_equity)
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown > MAX_DRAWDOWN_WARN:
                warnings.append(
                    f"DRAWDOWN: {drawdown * 100:.1f}% from peak equity "
                    f"${peak_equity:,.2f} (limit: {MAX_DRAWDOWN_WARN * 100:.0f}%)"
                )

    return warnings


# ===================================================================
# Formatters
# ===================================================================

_SEPARATOR = "-" * 72


def _fmt_currency(val: float) -> str:
    if val < 0:
        return f"-${abs(val):,.2f}"
    return f"${val:,.2f}"


def _fmt_pct(val: float) -> str:
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def _print_header(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def _display_text(
    account: Optional[Dict[str, Any]],
    positions: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    bot_state: Dict[str, Any],
    orders: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n  Paper Trading Tracker  |  {now}")

    # --- Account summary ---
    _print_header("Account Summary")
    if account:
        day_pnl = account["equity"] - account.get("portfolio_value", account["equity"])
        # Alpaca equity and portfolio_value are usually equal;
        # real day P&L comes from change vs previous close which we
        # approximate from bot_state if available.
        prev_equity = bot_state.get("last_equity") or bot_state.get("final_equity")
        if prev_equity:
            prev_equity = float(prev_equity)
            day_pnl = account["equity"] - prev_equity
            day_pnl_pct = (day_pnl / prev_equity * 100) if prev_equity else 0.0
        else:
            day_pnl = 0.0
            day_pnl_pct = 0.0

        rows = [
            ("Equity", _fmt_currency(account["equity"])),
            ("Cash", _fmt_currency(account["cash"])),
            ("Buying Power", _fmt_currency(account["buying_power"])),
            ("Day P&L", f"{_fmt_currency(day_pnl)}  ({_fmt_pct(day_pnl_pct)})"),
            ("Status", account["status"]),
        ]
        for label, val in rows:
            print(f"  {label:<18s} {val}")
    else:
        print("  (Alpaca unreachable)")

    # --- Position table ---
    _print_header("Positions")
    if positions:
        equity = account["equity"] if account else 0.0
        header = f"  {'Symbol':<8s} {'Qty':>8s} {'AvgEntry':>10s} {'Price':>10s} {'MktVal':>12s} {'PnL':>12s} {'PnL%':>8s} {'Weight':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in sorted(positions, key=lambda x: abs(x["market_value"]), reverse=True):
            weight = abs(p["market_value"]) / equity * 100 if equity else 0.0
            pnl_str = _fmt_currency(p["unrealized_pnl"])
            print(
                f"  {p['symbol']:<8s} {p['qty']:>8.1f} {p['avg_entry']:>10.2f} "
                f"{p['current_price']:>10.2f} {_fmt_currency(p['market_value']):>12s} "
                f"{pnl_str:>12s} {_fmt_pct(p['pnl_pct']):>8s} {weight:>7.1f}%"
            )
    else:
        print("  No open positions.")

    # --- Portfolio metrics ---
    _print_header("Portfolio Metrics")
    rows = [
        ("Total Mkt Value", _fmt_currency(metrics["total_market_value"])),
        (
            "Cash Allocation",
            f"{_fmt_currency(metrics['cash_allocation'])}  ({metrics['cash_pct']:.1f}%)",
        ),
        ("# Positions", str(metrics["num_positions"])),
        ("Largest Weight", f"{metrics['largest_position_weight']:.1f}%"),
        ("HHI Concentration", f"{metrics['hhi_concentration']:.4f}"),
    ]
    for label, val in rows:
        print(f"  {label:<18s} {val}")

    # --- Bot state ---
    _print_header("Bot State")
    if bot_state:
        for key in (
            "last_trade_date",
            "last_shutdown",
            "last_equity",
            "final_equity",
            "positions_count",
            "reason",
            "cash",
        ):
            if key in bot_state:
                val = bot_state[key]
                if isinstance(val, float):
                    val = _fmt_currency(val)
                print(f"  {key:<20s} {val}")
    else:
        print(f"  No bot state found ({BOT_STATE_FILE})")

    # --- Recent orders ---
    _print_header("Recent Orders (last 10)")
    if orders:
        header = f"  {'Symbol':<8s} {'Side':<6s} {'Qty':>8s} {'Status':<14s} {'Created'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for o in orders:
            created = o["created_at"][:19] if len(o["created_at"]) > 19 else o["created_at"]
            print(
                f"  {o['symbol']:<8s} {o['side']:<6s} {o['qty']:>8.1f} {o['status']:<14s} {created}"
            )
    else:
        print("  No recent orders.")

    # --- Risk warnings ---
    if warnings:
        _print_header("!! Risk Warnings !!")
        for w in warnings:
            print(f"  ** {w}")
    else:
        _print_header("Risk Checks")
        print("  All clear — no warnings.")

    print(f"\n{_SEPARATOR}\n")


def _build_json_output(
    account: Optional[Dict[str, Any]],
    positions: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    bot_state: Dict[str, Any],
    orders: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "positions": positions,
        "portfolio_metrics": metrics,
        "bot_state": bot_state,
        "recent_orders": orders,
        "risk_warnings": warnings,
    }


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper trading portfolio tracker — snapshot of portfolio health."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=BOT_STATE_FILE,
        help=f"Path to bot_state.json (default: {BOT_STATE_FILE})",
    )
    args = parser.parse_args()

    # Suppress noisy library logs in CLI mode
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    global BOT_STATE_FILE  # noqa: PLW0603
    BOT_STATE_FILE = args.state_file

    # 1. Connect to Alpaca
    broker = _connect_broker()

    # 2. Gather data (graceful on failure)
    account = _fetch_account(broker) if broker else None
    positions = _fetch_positions(broker) if broker else []
    orders = _fetch_recent_orders(broker) if broker else []
    bot_state = _load_bot_state()

    # 3. Compute derived metrics
    metrics = _compute_portfolio_metrics(positions, account)
    warnings = _compute_risk_warnings(positions, account, bot_state)

    # 4. Enrich positions with weight
    if account and account["equity"] > 0:
        for p in positions:
            p["weight"] = round(abs(p["market_value"]) / account["equity"] * 100, 2)
    else:
        for p in positions:
            p["weight"] = 0.0

    # 5. Output
    if args.json_output:
        output = _build_json_output(account, positions, metrics, bot_state, orders, warnings)
        print(json.dumps(output, indent=2, default=str))
    else:
        _display_text(account, positions, metrics, bot_state, orders, warnings)

    # 6. Disconnect
    if broker:
        try:
            broker.disconnect()
        except Exception:
            pass

    # Non-zero exit if there are risk warnings
    if warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
