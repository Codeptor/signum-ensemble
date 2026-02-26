"""Shared pytest fixtures for the quant test suite.

Only fixtures that are truly duplicated across 2+ test files and are
general-purpose belong here.  Local fixtures in individual test files
take precedence over conftest when both define the same name, so
existing tests are unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.brokers.base import BrokerAccount, BrokerOrder, BrokerPosition
from python.portfolio.risk_manager import RiskLimits, RiskManager


# ---------------------------------------------------------------------------
# Returns DataFrames — used across portfolio, monitoring, and backtest tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_returns_2asset():
    """100-day daily returns for 2 assets (AAPL, MSFT).

    Seed 42.  Used by test_risk_manager and anywhere a small
    two-asset return DataFrame is needed.
    """
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
def sample_returns_3asset():
    """252-day daily returns for 3 assets (AAPL, MSFT, GOOGL).

    Seed 42.  Used by test_risk_enhanced, test_risk, and other portfolio tests.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.0005, 0.02, 252),
            "MSFT": np.random.normal(0.0003, 0.018, 252),
            "GOOGL": np.random.normal(0.0004, 0.022, 252),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_returns_4asset():
    """252-day daily returns for 4 assets (AAPL, MSFT, GOOGL, AMZN).

    Seed 42.  Used by test_risk_attribution and risk parity tests.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.0005, 0.02, 252),
            "MSFT": np.random.normal(0.0003, 0.018, 252),
            "GOOGL": np.random.normal(0.0004, 0.022, 252),
            "AMZN": np.random.normal(0.0002, 0.025, 252),
        },
        index=dates,
    )
    return returns


# ---------------------------------------------------------------------------
# Weights — used across portfolio tests
# ---------------------------------------------------------------------------


@pytest.fixture
def equal_weights_3():
    """Equal weights for 3 assets (AAPL, MSFT, GOOGL), summing to ~1.0.

    Used by test_risk_enhanced and other portfolio tests.
    """
    return pd.Series({"AAPL": 0.333, "MSFT": 0.333, "GOOGL": 0.334})


@pytest.fixture
def equal_weights_4():
    """Equal weights for 4 assets (AAPL, MSFT, GOOGL, AMZN).

    Used by test_risk_attribution and risk parity tests.
    """
    return pd.Series({"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25})


@pytest.fixture
def sample_weights():
    """Portfolio weights for 2 assets (AAPL, MSFT).

    Used by test_risk_manager alongside sample_returns_2asset.
    """
    return pd.Series({"AAPL": 0.5, "MSFT": 0.5})


# ---------------------------------------------------------------------------
# Risk management fixtures — used in risk_manager and live_integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def risk_limits():
    """Standard test risk limits with conservative defaults.

    max_position_weight=0.25, max_daily_trades=10, etc.
    Used by test_risk_manager.
    """
    return RiskLimits(
        max_position_weight=0.25,
        max_sector_weight=0.50,
        min_position_weight=0.01,
        max_portfolio_var_95=0.05,
        max_drawdown_limit=0.20,
        min_sharpe_ratio=0.0,
        max_daily_trades=10,
        max_daily_turnover=1.0,
        max_single_trade_size=0.15,
        min_risk_reward_ratio=2.0,
        max_leverage=1.0,
    )


@pytest.fixture
def permissive_risk_limits():
    """Permissive risk limits for integration testing — no rejections.

    max_position_weight=1.0, max_leverage=2.0, etc.
    Used by test_live_integration.
    """
    return RiskLimits(
        max_position_weight=1.0,
        max_single_trade_size=1.0,
        max_leverage=2.0,
        max_daily_turnover=50.0,
    )


# ---------------------------------------------------------------------------
# Broker fixtures — used in live_integration, broker, and bridge tests
# ---------------------------------------------------------------------------


@pytest.fixture
def broker_account():
    """Standard test broker account with $100k equity.

    Used by test_live_integration and broker tests.
    """
    return BrokerAccount(
        account_id="test-001",
        cash=100_000.0,
        portfolio_value=100_000.0,
        buying_power=100_000.0,
        equity=100_000.0,
        status="ACTIVE",
    )


@pytest.fixture
def stock_prices():
    """Dict of stock prices for common test tickers.

    AAPL=200, MSFT=400, GOOG=150, TSLA=250.
    Used by test_live_integration and broker tests.
    """
    return {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 150.0, "TSLA": 250.0}


# ---------------------------------------------------------------------------
# Price data fixtures — used in optimizer and integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_price_data_5asset():
    """Synthetic price data for 5 assets over 252 business days.

    Seed 42.  Prices are generated as exp(cumulative normal returns) * 100.
    Used by test_optimizer and integration tests.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.randn(252, 5) * 0.01, axis=0)) * 100,
        index=dates,
        columns=tickers,
    )
    return prices


# ---------------------------------------------------------------------------
# OHLCV fixtures — used in alpha/features and alpha/atr tests
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_ticker_ohlcv():
    """Multi-ticker OHLCV long-format data for AAPL and MSFT (60 business days).

    Seed 42.  Used by test_features and test_atr.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    frames = []
    for ticker in ["AAPL", "MSFT"]:
        base = 150 + np.cumsum(np.random.randn(60) * 2)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": base + np.random.randn(60),
                    "high": base + abs(np.random.randn(60)) * 2,
                    "low": base - abs(np.random.randn(60)) * 2,
                    "close": base,
                    "volume": np.random.randint(500000, 2000000, 60).astype(float),
                },
                index=dates,
            )
        )
    return pd.concat(frames)
