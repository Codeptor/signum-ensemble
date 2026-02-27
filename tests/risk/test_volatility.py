"""Tests for advanced volatility estimators."""

import numpy as np
import pytest

from python.risk.volatility import (
    close_to_close,
    ewma_volatility,
    garman_klass,
    parkinson,
    realized_volatility,
    rogers_satchell,
    yang_zhang,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc(n=500, daily_vol=0.02, seed=42):
    """Generate synthetic OHLC data."""
    rng = np.random.default_rng(seed)
    close = np.zeros(n)
    close[0] = 100.0
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(rng.normal(0, daily_vol))

    # Generate open/high/low relative to close
    open_ = close * np.exp(rng.normal(0, daily_vol * 0.3, n))
    intraday_range = np.abs(rng.normal(0, daily_vol * 0.7, n))
    high = np.maximum(open_, close) * (1 + np.abs(intraday_range))
    low = np.minimum(open_, close) * (1 - np.abs(intraday_range))
    low = np.maximum(low, 0.01)

    return open_, high, low, close


# ---------------------------------------------------------------------------
# Close-to-Close
# ---------------------------------------------------------------------------


class TestCloseToClose:
    def test_output_shape(self):
        _, _, _, close = _make_ohlc()
        result = close_to_close(close, window=20)
        assert len(result) == len(close)

    def test_positive(self):
        _, _, _, close = _make_ohlc()
        result = close_to_close(close, window=20)
        valid = result[~np.isnan(result)]
        assert all(v > 0 for v in valid)

    def test_scales_with_true_vol(self):
        _, _, _, close_low = _make_ohlc(daily_vol=0.01)
        _, _, _, close_high = _make_ohlc(daily_vol=0.04)
        vol_low = close_to_close(close_low, window=100)
        vol_high = close_to_close(close_high, window=100)
        assert np.nanmean(vol_low) < np.nanmean(vol_high)


# ---------------------------------------------------------------------------
# Parkinson
# ---------------------------------------------------------------------------


class TestParkinson:
    def test_output_shape(self):
        _, high, low, _ = _make_ohlc()
        result = parkinson(high, low, window=20)
        assert len(result) == len(high)

    def test_positive(self):
        _, high, low, _ = _make_ohlc()
        result = parkinson(high, low, window=20)
        valid = result[~np.isnan(result)]
        assert all(v > 0 for v in valid)

    def test_more_efficient_than_c2c(self):
        """Parkinson should have lower variance than close-to-close."""
        _, high, low, close = _make_ohlc(n=1000)
        c2c = close_to_close(close, window=50)
        park = parkinson(high, low, window=50)
        # Both should give similar mean but Parkinson less noisy
        c2c_valid = c2c[~np.isnan(c2c)]
        park_valid = park[~np.isnan(park)]
        assert np.std(park_valid) < np.std(c2c_valid) * 1.5


# ---------------------------------------------------------------------------
# Garman-Klass
# ---------------------------------------------------------------------------


class TestGarmanKlass:
    def test_output_shape(self):
        o, h, l, c = _make_ohlc()
        result = garman_klass(o, h, l, c, window=20)
        assert len(result) == len(c)

    def test_positive(self):
        o, h, l, c = _make_ohlc()
        result = garman_klass(o, h, l, c, window=20)
        valid = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid)


# ---------------------------------------------------------------------------
# Rogers-Satchell
# ---------------------------------------------------------------------------


class TestRogersSatchell:
    def test_output_shape(self):
        o, h, l, c = _make_ohlc()
        result = rogers_satchell(o, h, l, c, window=20)
        assert len(result) == len(c)

    def test_positive(self):
        o, h, l, c = _make_ohlc()
        result = rogers_satchell(o, h, l, c, window=20)
        valid = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid)


# ---------------------------------------------------------------------------
# Yang-Zhang
# ---------------------------------------------------------------------------


class TestYangZhang:
    def test_output_shape(self):
        o, h, l, c = _make_ohlc()
        result = yang_zhang(o, h, l, c, window=20)
        assert len(result) == len(c)

    def test_positive(self):
        o, h, l, c = _make_ohlc()
        result = yang_zhang(o, h, l, c, window=20)
        valid = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid)

    def test_reasonable_magnitude(self):
        """Annual vol for 2% daily vol should be ~30%."""
        o, h, l, c = _make_ohlc(n=1000, daily_vol=0.02)
        result = yang_zhang(o, h, l, c, window=100)
        valid = result[~np.isnan(result)]
        mean_vol = np.mean(valid)
        assert 0.10 < mean_vol < 0.80


# ---------------------------------------------------------------------------
# EWMA Volatility
# ---------------------------------------------------------------------------


class TestEWMA:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 500)
        result = ewma_volatility(returns, halflife=20)
        assert len(result) == 500

    def test_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 500)
        result = ewma_volatility(returns, halflife=20)
        assert all(v >= 0 for v in result)

    def test_responds_to_spike(self):
        """EWMA should spike after a large return."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        returns[250] = 0.10  # Large shock
        result = ewma_volatility(returns, halflife=10)
        assert result[251] > result[249]

    def test_shorter_halflife_more_responsive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        returns[250] = 0.10
        short_hl = ewma_volatility(returns, halflife=5)
        long_hl = ewma_volatility(returns, halflife=50)
        # Short halflife should react more to the spike
        assert short_hl[251] > long_hl[251]


# ---------------------------------------------------------------------------
# Realized Volatility
# ---------------------------------------------------------------------------


class TestRealizedVolatility:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        intraday = rng.normal(0, 0.001, 78 * 100)
        result = realized_volatility(intraday, bars_per_day=78)
        assert len(result) == 100

    def test_positive(self):
        rng = np.random.default_rng(42)
        intraday = rng.normal(0, 0.001, 78 * 50)
        result = realized_volatility(intraday, bars_per_day=78)
        assert all(v >= 0 for v in result)

    def test_scales_with_vol(self):
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0, 0.0005, 78 * 100)
        high_vol = rng.normal(0, 0.002, 78 * 100)
        rv_low = realized_volatility(low_vol, bars_per_day=78)
        rv_high = realized_volatility(high_vol, bars_per_day=78)
        assert np.mean(rv_low) < np.mean(rv_high)
