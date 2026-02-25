"""Tests for execution bridge module."""

import pandas as pd
import pytest

from python.bridge.execution import (
    ExecutionBridge,
    Fill,
    Order,
    PaperTradingEngine,
    Position,
)
from python.portfolio.risk_manager import RiskLimits, RiskManager


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test order creation."""
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)

        assert order.ticker == "AAPL"
        assert order.side == "BUY"
        assert order.quantity == 100.0
        assert order.order_type == "MARKET"
        assert order.timestamp is not None


class TestPosition:
    """Test Position class."""

    def test_position_creation(self):
        """Test position creation."""
        pos = Position(ticker="AAPL", quantity=100.0, avg_cost=150.0)

        assert pos.ticker == "AAPL"
        assert pos.quantity == 100.0
        assert pos.avg_cost == 150.0
        assert pos.realized_pnl == 0.0

    def test_market_value(self):
        """Test market value calculation."""
        pos = Position(ticker="AAPL", quantity=100.0, avg_cost=150.0)
        value = pos.market_value(160.0)

        assert value == 16000.0

    def test_position_update_buy(self):
        """Test position update with buy."""
        pos = Position(ticker="AAPL")
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)
        fill = Fill(
            order=order,
            fill_price=150.0,
            fill_quantity=100.0,
            commission=10.0,
            timestamp=pd.Timestamp.now(),
        )

        pos.update(fill)

        assert pos.quantity == 100.0
        assert pos.avg_cost == 150.0

    def test_position_update_sell(self):
        """Test position update with sell."""
        pos = Position(ticker="AAPL", quantity=100.0, avg_cost=150.0)
        order = Order(ticker="AAPL", side="SELL", quantity=50.0)
        fill = Fill(
            order=order,
            fill_price=160.0,
            fill_quantity=50.0,
            commission=10.0,
            timestamp=pd.Timestamp.now(),
        )

        pos.update(fill)

        assert pos.quantity == 50.0
        # Realized P&L: 50 * (160 - 150) = 500
        assert pos.realized_pnl == 500.0


class TestExecutionBridge:
    """Test ExecutionBridge class."""

    def test_initialization(self):
        """Test bridge initialization."""
        bridge = ExecutionBridge(initial_capital=1000000.0)

        assert bridge.initial_capital == 1000000.0
        assert bridge.cash == 1000000.0
        assert bridge.equity == 1000000.0
        assert len(bridge.positions) == 0

    def test_get_position(self):
        """Test getting/creating position."""
        bridge = ExecutionBridge()
        pos = bridge.get_position("AAPL")

        assert pos.ticker == "AAPL"
        assert pos.quantity == 0.0
        assert "AAPL" in bridge.positions

    def test_submit_buy_order(self):
        """Test submitting buy order."""
        bridge = ExecutionBridge(initial_capital=100000.0)
        fill = bridge.submit_order("AAPL", "BUY", 10.0, 150.0)

        assert fill is not None
        assert fill.order.ticker == "AAPL"
        assert fill.order.side == "BUY"
        assert fill.fill_quantity == 10.0
        assert fill.fill_price == 150.0

        # Check position updated
        pos = bridge.get_position("AAPL")
        assert pos.quantity == 10.0

        # Check cash reduced
        expected_cash = 100000.0 - (10.0 * 150.0) - fill.commission
        assert bridge.cash == pytest.approx(expected_cash, abs=0.01)

    def test_submit_sell_order(self):
        """Test submitting sell order."""
        bridge = ExecutionBridge(initial_capital=100000.0)

        # First buy some shares
        bridge.submit_order("AAPL", "BUY", 10.0, 150.0)

        # Then sell
        fill = bridge.submit_order("AAPL", "SELL", 5.0, 160.0)

        assert fill is not None

        # Check position
        pos = bridge.get_position("AAPL")
        assert pos.quantity == 5.0

    def test_insufficient_cash(self):
        """Test order rejected due to insufficient cash."""
        bridge = ExecutionBridge(initial_capital=1000.0)
        fill = bridge.submit_order("AAPL", "BUY", 100.0, 150.0)

        assert fill is None

    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        bridge = ExecutionBridge(initial_capital=100000.0)
        bridge.submit_order("AAPL", "BUY", 10.0, 150.0)

        summary = bridge.get_portfolio_summary()

        assert "cash" in summary
        assert "equity" in summary
        assert "num_positions" in summary
        assert summary["num_positions"] == 1

    def test_reconcile_target_weights(self):
        """Test reconciling positions with target weights."""
        bridge = ExecutionBridge(initial_capital=100000.0)

        # Start with some cash
        target_weights = {"AAPL": 0.5, "MSFT": 0.3}
        prices = {"AAPL": 150.0, "MSFT": 200.0}

        fills = bridge.reconcile_target_weights(target_weights, prices)

        # Should have fills for both
        assert len(fills) > 0

        # Check positions
        aapl_pos = bridge.get_position("AAPL")
        assert aapl_pos.quantity > 0

    def test_with_risk_manager(self):
        """Test execution with risk manager validation."""
        limits = RiskLimits(max_position_weight=0.10)  # 10% max
        risk_mgr = RiskManager(limits=limits)
        bridge = ExecutionBridge(
            risk_manager=risk_mgr,
            initial_capital=100000.0,
        )

        # Try to buy 20% position (should be rejected)
        fill = bridge.submit_order("AAPL", "BUY", 200.0, 100.0)

        assert fill is None  # Should be rejected


class TestPaperTradingEngine:
    """Test PaperTradingEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = PaperTradingEngine(initial_capital=1000000.0)

        assert engine.execution_bridge.initial_capital == 1000000.0
        assert len(engine.trade_history) == 0

    def test_run_strategy(self):
        """Test running strategy."""
        engine = PaperTradingEngine(initial_capital=100000.0)

        # Create sample signals
        dates = pd.date_range("2024-01-01", periods=3)
        signals_data = []
        for date in dates:
            signals_data.append({"date": date, "ticker": "AAPL", "weight": 0.5})
            signals_data.append({"date": date, "ticker": "MSFT", "weight": 0.3})

        signals = pd.DataFrame(signals_data)
        signals = signals.set_index(["date", "ticker"])

        # Create sample prices
        prices_data = []
        for date in dates:
            prices_data.append({"date": date, "AAPL": 150.0, "MSFT": 200.0})

        prices = pd.DataFrame(prices_data).set_index("date")

        # Run strategy
        summary = engine.run_strategy(signals, prices)

        assert "equity" in summary
        assert "total_return" in summary
        assert len(engine.trade_history) > 0
