"""Tests for RiskManager module."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.risk_manager import (
    PositionSizer,
    RiskCheck,
    RiskLimits,
    RiskManager,
)


@pytest.fixture
def risk_limits():
    """Create test risk limits."""
    return RiskLimits(
        max_position_weight=0.25,
        max_sector_weight=0.50,
        min_position_weight=0.01,
        max_portfolio_var_95=0.05,
        max_drawdown_limit=0.20,
        min_sharpe_ratio=0.0,
        max_daily_trades=10,  # Set to 10 for testing
        max_daily_turnover=1.0,
        max_single_trade_size=0.15,
        min_risk_reward_ratio=2.0,
        max_leverage=1.0,
    )


@pytest.fixture
def risk_manager(risk_limits):
    """Create test risk manager."""
    return RiskManager(limits=risk_limits)


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.0005, 0.02, 100),
            "MSFT": np.random.normal(0.0003, 0.018, 100),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    return pd.Series({"AAPL": 0.5, "MSFT": 0.5})


class TestRiskLimits:
    """Test RiskLimits dataclass."""

    def test_default_limits(self):
        """Test default limit values."""
        limits = RiskLimits()

        assert limits.max_position_weight == 0.25
        assert limits.max_daily_trades == 20
        assert limits.min_risk_reward_ratio == 2.0

    def test_custom_limits(self):
        """Test custom limit configuration."""
        limits = RiskLimits(
            max_position_weight=0.30,
            max_daily_trades=20,
        )

        assert limits.max_position_weight == 0.30
        assert limits.max_daily_trades == 20


class TestRiskCheck:
    """Test RiskCheck dataclass."""

    def test_risk_check_creation(self):
        """Test RiskCheck creation."""
        check = RiskCheck(
            passed=True,
            rule="TEST_RULE",
            message="Test message",
            severity="info",
        )

        assert check.passed is True
        assert check.rule == "TEST_RULE"
        assert check.message == "Test message"
        assert check.severity == "info"


class TestRiskManagerInitialization:
    """Test RiskManager initialization."""

    def test_init_with_limits(self, risk_limits):
        """Test initialization with custom limits."""
        manager = RiskManager(limits=risk_limits)

        assert manager.limits == risk_limits
        assert isinstance(manager.daily_trades, dict)
        assert len(manager.daily_trades) == 0
        assert manager.risk_engine is None

    def test_init_without_limits(self):
        """Test initialization with default limits."""
        manager = RiskManager()

        assert isinstance(manager.limits, RiskLimits)
        assert manager.limits.max_position_weight == 0.25


class TestTradeValidation:
    """Test trade validation."""

    def test_valid_trade(self, risk_manager):
        """Test validation of valid trade."""
        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,  # Within 25% limit
            current_date="2024-01-01",
        )

        passed_checks = [c for c in checks if c.passed]
        assert len(passed_checks) > 0

    def test_position_size_limit_violation(self, risk_manager):
        """Test detection of position size limit violation."""
        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.30,  # Exceeds 25% limit
            current_date="2024-01-01",
        )

        failed_checks = [c for c in checks if not c.passed]
        assert any(c.rule == "MAX_POSITION_SIZE" for c in failed_checks)

    def test_min_position_size_violation(self, risk_manager):
        """Test detection of minimum position size violation."""
        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.005,  # Below 1% minimum
            current_date="2024-01-01",
        )

        failed_checks = [c for c in checks if not c.passed]
        assert any(c.rule == "MIN_POSITION_SIZE" for c in failed_checks)

    def test_daily_trade_limit(self, risk_manager):
        """Test daily trade limit enforcement."""
        # Simulate 10 trades already made
        risk_manager.daily_trades["2024-01-01"] = 10

        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,
            current_date="2024-01-01",
        )

        failed_checks = [c for c in checks if not c.passed]
        assert any(c.rule == "MAX_DAILY_TRADES" for c in failed_checks)

    def test_risk_reward_validation(self, risk_manager):
        """Test risk/reward ratio validation."""
        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,
            expected_return=0.02,  # 2% expected
            risk_amount=0.02,  # 2% risk (1:1 ratio, below 2:1 minimum)
            current_date="2024-01-01",
        )

        failed_checks = [c for c in checks if not c.passed]
        assert any(c.rule == "MIN_RISK_REWARD" for c in failed_checks)

    def test_good_risk_reward_passes(self, risk_manager):
        """Test that good risk/reward passes."""
        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,
            expected_return=0.04,  # 4% expected
            risk_amount=0.01,  # 1% risk (4:1 ratio, above 2:1 minimum)
            current_date="2024-01-01",
        )

        rr_checks = [c for c in checks if c.rule == "RISK_REWARD"]
        assert len(rr_checks) > 0
        assert rr_checks[0].passed is True


class TestMaxSingleTradeSize:
    """Test MAX_SINGLE_TRADE_SIZE check (M4 fix).

    This check warns when a single trade's weight change exceeds
    max_single_trade_size. Severity is 'warning', not 'critical',
    so it won't block trades via can_execute_trade().
    """

    def test_large_trade_from_zero_triggers_warning(self, risk_manager):
        """Opening a new 20% position exceeds 15% single trade limit."""
        risk_manager.current_weights = pd.Series({"MSFT": 0.10})

        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,
            current_date="2024-01-01",
        )

        trade_size_checks = [c for c in checks if c.rule == "MAX_SINGLE_TRADE_SIZE"]
        assert len(trade_size_checks) == 1
        assert trade_size_checks[0].passed is False
        assert trade_size_checks[0].severity == "warning"

    def test_small_trade_passes(self, risk_manager):
        """A 10% weight change is within 15% limit."""
        risk_manager.current_weights = pd.Series({"AAPL": 0.10})

        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,
            current_date="2024-01-01",
        )

        trade_size_checks = [c for c in checks if c.rule == "SINGLE_TRADE_SIZE"]
        assert len(trade_size_checks) == 1
        assert trade_size_checks[0].passed is True

    def test_trade_from_existing_position(self, risk_manager):
        """Reducing from 25% to 5% is a 20% change — exceeds 15% limit."""
        risk_manager.current_weights = pd.Series({"AAPL": 0.25})

        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.05,
            current_date="2024-01-01",
        )

        trade_size_checks = [c for c in checks if c.rule == "MAX_SINGLE_TRADE_SIZE"]
        assert len(trade_size_checks) == 1
        assert trade_size_checks[0].passed is False

    def test_no_current_weights_uses_absolute(self, risk_manager):
        """When current_weights is None, uses absolute new_weight as trade size."""
        risk_manager.current_weights = None

        checks = risk_manager.check_trade(
            ticker="AAPL",
            new_weight=0.20,
            current_date="2024-01-01",
        )

        trade_size_checks = [c for c in checks if c.rule == "MAX_SINGLE_TRADE_SIZE"]
        assert len(trade_size_checks) == 1
        assert trade_size_checks[0].passed is False

    def test_warning_does_not_block_trade(self, risk_manager):
        """MAX_SINGLE_TRADE_SIZE is severity=warning, so can_execute_trade still True."""
        risk_manager.current_weights = pd.Series({"MSFT": 0.10})

        can_execute, issues = risk_manager.can_execute_trade(
            ticker="AAPL",
            new_weight=0.20,  # 20% trade from 0%, exceeds 15% limit
            current_date="2024-01-01",
        )

        # Should still be allowed (warning, not critical)
        assert can_execute is True


class TestCanExecuteTrade:
    """Test can_execute_trade method."""

    def test_can_execute_valid_trade(self, risk_manager):
        """Test that valid trades can execute."""
        can_execute, issues = risk_manager.can_execute_trade(
            ticker="AAPL",
            new_weight=0.20,
            current_date="2024-01-01",
        )

        assert can_execute is True
        assert len(issues) == 0

    def test_cannot_execute_oversized_trade(self, risk_manager):
        """Test that oversized trades are blocked."""
        can_execute, issues = risk_manager.can_execute_trade(
            ticker="AAPL",
            new_weight=0.50,  # Too large
            current_date="2024-01-01",
        )

        assert can_execute is False
        assert len(issues) > 0


class TestTradeRecording:
    """Test trade recording."""

    def test_record_trade_updates_count(self, risk_manager):
        """Test that recording trade updates daily count."""
        risk_manager.record_trade("AAPL", 0.20, "2024-01-01")

        assert risk_manager.daily_trades["2024-01-01"] == 1

        risk_manager.record_trade("MSFT", 0.15, "2024-01-01")

        assert risk_manager.daily_trades["2024-01-01"] == 2

    def test_record_trade_updates_turnover(self, risk_manager):
        """Test that recording trade updates turnover."""
        risk_manager.record_trade("AAPL", 0.20, "2024-01-01")

        assert risk_manager.daily_turnover["2024-01-01"] == 0.20

    def test_record_trade_updates_weights(self, risk_manager):
        """Test that recording trade updates current weights."""
        risk_manager.current_weights = pd.Series({"AAPL": 0.0, "MSFT": 0.0})

        risk_manager.record_trade("AAPL", 0.20, "2024-01-01")

        assert risk_manager.current_weights["AAPL"] == 0.20

    def test_record_trade_ignores_small_changes(self, risk_manager):
        """Test that small weight changes are not recorded."""
        risk_manager.current_weights = pd.Series({"AAPL": 0.20})

        # Change of 0.0005 (0.05%) is below 0.001 threshold
        risk_manager.record_trade("AAPL", 0.0005, "2024-01-01")

        assert risk_manager.daily_trades.get("2024-01-01", 0) == 0


class TestPortfolioRiskChecks:
    """Test portfolio-level risk checks."""

    def test_check_portfolio_risk_without_engine(self, risk_manager):
        """Test that checks work without risk engine."""
        checks = risk_manager.check_portfolio_risk(pd.Series())

        assert checks == []

    def test_var_limit_check(self, risk_manager, sample_returns, sample_weights):
        """Test VaR limit check."""
        risk_manager.initialize_portfolio_risk(sample_returns, sample_weights)
        checks = risk_manager.check_portfolio_risk(sample_returns.iloc[:, 0])

        # Should have some checks (VaR, drawdown, etc.)
        assert len(checks) > 0


class TestRiskSummary:
    """Test risk summary generation."""

    def test_risk_summary_without_engine(self, risk_manager):
        """Test summary without initialized engine."""
        summary = risk_manager.get_risk_summary()

        assert "error" in summary

    def test_risk_summary_with_engine(self, risk_manager, sample_returns, sample_weights):
        """Test summary with initialized engine."""
        risk_manager.initialize_portfolio_risk(sample_returns, sample_weights)
        summary = risk_manager.get_risk_summary()

        assert "risk_engine_initialized" in summary
        assert summary["risk_engine_initialized"] is True
        assert "portfolio_metrics" in summary


class TestPositionSizerKelly:
    """Test PositionSizer Kelly criterion calculations."""

    def test_kelly_size_calculation(self):
        """Test Kelly criterion calculation."""
        sizer = PositionSizer()

        # 60% win rate, 2:1 win/loss ratio
        kelly = sizer.kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.05)

        assert kelly > 0
        assert kelly <= 0.25  # Half-Kelly, max position

    def test_kelly_with_no_loss(self):
        """Test Kelly with zero loss (should be infinite but capped)."""
        sizer = PositionSizer()

        kelly = sizer.kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.0)

        assert kelly == 0.0  # Returns 0 when avg_loss is 0 to avoid division by zero

    def test_kelly_safety_factor(self):
        """Test that Kelly uses half-Kelly for safety."""
        sizer = PositionSizer()

        # Pure Kelly would be 0.25, half-Kelly is 0.125
        kelly = sizer.kelly_size(win_rate=0.60, avg_win=0.10, avg_loss=0.05)

        # Should be using half-Kelly
        assert kelly < 0.20


class TestPositionSizerRiskBased:
    """Test PositionSizer risk-based calculations."""

    def test_risk_based_size(self):
        """Test risk-based position sizing."""
        sizer = PositionSizer(risk_per_trade=0.02)  # 2% risk

        # 5% stop loss
        size = sizer.risk_based_size(stop_loss_pct=0.05)

        # size = (1.0 * 0.02) / 0.05 = 0.40, but capped at 0.25
        assert size == pytest.approx(0.25, abs=0.01)

    def test_risk_based_size_respects_max(self):
        """Test that risk-based sizing respects max position limit."""
        sizer = PositionSizer(max_position_weight=0.25, risk_per_trade=0.02)

        # Very tight stop would suggest large position
        size = sizer.risk_based_size(stop_loss_pct=0.01)

        assert size <= 0.25

    def test_risk_based_size_zero_stop(self):
        """Test risk-based sizing with zero stop loss."""
        sizer = PositionSizer()

        size = sizer.risk_based_size(stop_loss_pct=0.0)

        assert size == 0.0


class TestPositionSizerVolatility:
    """Test PositionSizer volatility-based adjustments."""

    def test_volatility_adjustment_high_vol(self):
        """Test position reduction in high volatility."""
        sizer = PositionSizer()

        # High vol (30%) should reduce position
        size = sizer.volatility_adjusted_size(
            base_size=0.20,
            current_vol=0.30,
            target_vol=0.15,
        )

        # size = 0.20 * (0.15 / 0.30) = 0.10
        assert size == pytest.approx(0.10, abs=0.01)

    def test_volatility_adjustment_low_vol(self):
        """Test position increase in low volatility."""
        sizer = PositionSizer()

        # Low vol (10%) should increase position
        size = sizer.volatility_adjusted_size(
            base_size=0.20,
            current_vol=0.10,
            target_vol=0.15,
        )

        # size = 0.20 * (0.15 / 0.10) = 0.30, but capped at 0.25
        assert size > 0.20

    def test_volatility_adjustment_zero_vol(self):
        """Test volatility adjustment with zero volatility."""
        sizer = PositionSizer()

        size = sizer.volatility_adjusted_size(
            base_size=0.20,
            current_vol=0.0,
            target_vol=0.15,
        )

        assert size == 0.20  # Returns base_size when vol is 0
