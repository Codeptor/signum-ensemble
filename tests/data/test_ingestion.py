import pandas as pd
import pytest

from python.data.ingestion import fetch_ohlcv, fetch_sp500_tickers, YFINANCE_TIMEOUT


@pytest.mark.network
def test_fetch_sp500_tickers_returns_list():
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400
    assert "AAPL" in tickers


@pytest.mark.network
def test_fetch_ohlcv_returns_dataframe():
    df = fetch_ohlcv(["AAPL", "MSFT"], period="5d")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


class TestYfinanceTimeout:
    """Test that YFINANCE_TIMEOUT is defined and reasonable (M6 fix)."""

    def test_timeout_is_defined(self):
        """YFINANCE_TIMEOUT should be a positive integer."""
        assert isinstance(YFINANCE_TIMEOUT, (int, float))
        assert YFINANCE_TIMEOUT > 0

    def test_timeout_is_30_seconds(self):
        """Default timeout should be 30 seconds."""
        assert YFINANCE_TIMEOUT == 30
