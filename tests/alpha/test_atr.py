"""Tests for ATR computation and helper functions (Phase 2.3)."""

import numpy as np
import pandas as pd
import pytest

from python.alpha.features import compute_atr, _compute_atr_single


@pytest.fixture
def single_ticker_ohlc():
    """Single ticker OHLC data with known values for ATR verification."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    base = 100 + np.cumsum(np.random.randn(60) * 1.5)

    return pd.DataFrame(
        {
            "open": base + np.random.randn(60) * 0.5,
            "high": base + abs(np.random.randn(60)) * 2,
            "low": base - abs(np.random.randn(60)) * 2,
            "close": base,
            "volume": np.random.randint(500_000, 2_000_000, 60).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def multi_ticker_ohlc():
    """Multi-ticker OHLC data."""
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
                    "high": base + abs(np.random.randn(60)) * 3,
                    "low": base - abs(np.random.randn(60)) * 3,
                    "close": base,
                    "volume": np.random.randint(500_000, 2_000_000, 60).astype(float),
                },
                index=dates,
            )
        )
    return pd.concat(frames)


class TestComputeATRSingle:
    def test_atr_column_added(self, single_ticker_ohlc):
        result = _compute_atr_single(single_ticker_ohlc, window=14)
        assert "atr_14" in result.columns

    def test_atr_custom_window(self, single_ticker_ohlc):
        result = _compute_atr_single(single_ticker_ohlc, window=7)
        assert "atr_7" in result.columns

    def test_atr_positive_values(self, single_ticker_ohlc):
        result = _compute_atr_single(single_ticker_ohlc, window=14)
        valid = result["atr_14"].dropna()
        assert (valid > 0).all(), "ATR should always be positive"

    def test_atr_nan_warmup(self, single_ticker_ohlc):
        """ATR should be NaN for the first window-1 rows + 1 for the shift."""
        result = _compute_atr_single(single_ticker_ohlc, window=14)
        # First 14 rows should have NaN (13 from window, 1 from shift)
        assert result["atr_14"].iloc[:14].isna().sum() >= 13

    def test_atr_non_nan_after_warmup(self, single_ticker_ohlc):
        result = _compute_atr_single(single_ticker_ohlc, window=14)
        # After warmup, should have values
        valid = result["atr_14"].iloc[15:]
        assert valid.notna().all()

    def test_atr_preserves_other_columns(self, single_ticker_ohlc):
        result = _compute_atr_single(single_ticker_ohlc, window=14)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_atr_preserves_index(self, single_ticker_ohlc):
        result = _compute_atr_single(single_ticker_ohlc, window=14)
        pd.testing.assert_index_equal(result.index, single_ticker_ohlc.index)


class TestComputeATRMultiTicker:
    def test_multi_ticker_dispatch(self, multi_ticker_ohlc):
        result = compute_atr(multi_ticker_ohlc, window=14)
        assert "atr_14" in result.columns
        # Both tickers present
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_multi_ticker_row_count(self, multi_ticker_ohlc):
        result = compute_atr(multi_ticker_ohlc, window=14)
        assert len(result) == len(multi_ticker_ohlc)

    def test_single_ticker_dispatch(self, single_ticker_ohlc):
        """When no 'ticker' column, compute_atr should work on single df."""
        result = compute_atr(single_ticker_ohlc, window=14)
        assert "atr_14" in result.columns


class TestATRValues:
    def test_atr_responds_to_volatility(self):
        """Higher volatility stock should have higher ATR."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="B")

        # Low volatility stock
        base_low = 100 + np.cumsum(np.random.randn(60) * 0.5)
        low_vol = pd.DataFrame(
            {
                "open": base_low,
                "high": base_low + 0.5,
                "low": base_low - 0.5,
                "close": base_low,
            },
            index=dates,
        )

        # High volatility stock
        base_high = 100 + np.cumsum(np.random.randn(60) * 3)
        high_vol = pd.DataFrame(
            {
                "open": base_high,
                "high": base_high + 5.0,
                "low": base_high - 5.0,
                "close": base_high,
            },
            index=dates,
        )

        atr_low = _compute_atr_single(low_vol, window=14)["atr_14"].dropna().mean()
        atr_high = _compute_atr_single(high_vol, window=14)["atr_14"].dropna().mean()

        assert atr_high > atr_low, "High-vol stock should have higher ATR"

    def test_atr_true_range_definition(self):
        """ATR should use the maximum of the three true range components."""
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        # Construct data where gap-based TR is larger than high-low
        df = pd.DataFrame(
            {
                "open": [100] * 20,
                "high": [101] * 20,  # H-L = 2
                "low": [99] * 20,
                "close": [100] * 3 + [110] + [100] * 16,  # Gap up on day 3
            },
            index=dates,
        )
        result = _compute_atr_single(df, window=5)
        # After the gap, ATR should spike
        atr_after_gap = result["atr_5"].iloc[8]  # After warmup + gap
        assert atr_after_gap > 2.0  # Must be > basic H-L
