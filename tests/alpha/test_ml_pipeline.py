"""Tests for the predict.py ML pipeline sub-functions.

Tests each component of the get_ml_weights pipeline with mocked I/O:
  - fetch_universe: ticker filtering, min-history dropping
  - compute_features: runs feature pipeline
  - rank_stocks: selects top-N tickers from model predictions
  - optimize_weights: portfolio optimization with weight normalization
  - get_ml_weights: end-to-end orchestration

All network calls (yfinance, S&P 500 scraping) are patched out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from python.alpha.predict import (
    compute_features,
    fetch_universe,
    optimize_weights,
    rank_stocks,
)

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_long_ohlcv(tickers: list[str], n_days: int = 100) -> pd.DataFrame:
    """Build synthetic long-format OHLCV data with a DatetimeIndex."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    frames = []
    for ticker in tickers:
        close = 100 + rng.standard_normal(n_days).cumsum()
        df = pd.DataFrame(
            {
                "open": close + rng.uniform(-1, 1, n_days),
                "high": close + abs(rng.standard_normal(n_days)),
                "low": close - abs(rng.standard_normal(n_days)),
                "close": close,
                "volume": rng.integers(100_000, 5_000_000, n_days).astype(float),
                "ticker": ticker,
            },
            index=dates,
        )
        frames.append(df)
    return pd.concat(frames).sort_index()


def _make_featured_df(tickers: list[str], n_days: int = 100) -> pd.DataFrame:
    """Build synthetic featured DataFrame with a DatetimeIndex and ``ticker`` as a column.

    The real pipeline produces a DataFrame whose index is a DatetimeIndex
    (or a MultiIndex ``(date, row_idx)``) with ``ticker`` remaining as a
    regular **column**.  ``rank_stocks()`` relies on this — it slices the
    latest date via ``.loc[latest_date]`` and then accesses
    ``latest["ticker"]``.

    To allow ``.index.get_level_values(0).max()`` (used in ``rank_stocks``
    to find the latest date), we build a simple DatetimeIndex where each
    date is repeated once per ticker.
    """
    from python.alpha.train import FEATURE_COLS

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n_days)

    frames = []
    for ticker in tickers:
        data = {col: rng.standard_normal(n_days) for col in FEATURE_COLS}
        data["ticker"] = ticker
        data["close"] = 100 + rng.standard_normal(n_days).cumsum()
        df = pd.DataFrame(data, index=dates)
        frames.append(df)

    result = pd.concat(frames).sort_index()
    # Keep ticker as a column — do NOT push it into the index.
    # The DatetimeIndex already has duplicate dates (one per ticker),
    # which is exactly the shape rank_stocks expects.
    return result


# ===========================================================================
# fetch_universe
# ===========================================================================


class TestFetchUniverse:
    @patch("python.alpha.predict.reshape_ohlcv_wide_to_long")
    @patch("python.alpha.predict.fetch_ohlcv")
    def test_drops_tickers_with_insufficient_history(self, mock_fetch, mock_reshape):
        """Tickers with < 80 rows of data should be dropped."""
        # AAPL has 100 days, NEWIPO has only 30
        dates_full = pd.bdate_range("2025-01-01", periods=100)
        dates_short = pd.bdate_range("2025-08-01", periods=30)
        rng = np.random.default_rng(42)

        aapl = pd.DataFrame(
            {
                "open": rng.uniform(100, 200, 100),
                "close": rng.uniform(100, 200, 100),
                "high": rng.uniform(100, 200, 100),
                "low": rng.uniform(100, 200, 100),
                "volume": rng.integers(1e5, 1e6, 100).astype(float),
                "ticker": "AAPL",
            },
            index=dates_full,
        )
        newipo = pd.DataFrame(
            {
                "open": rng.uniform(50, 60, 30),
                "close": rng.uniform(50, 60, 30),
                "high": rng.uniform(50, 60, 30),
                "low": rng.uniform(50, 60, 30),
                "volume": rng.integers(1e5, 1e6, 30).astype(float),
                "ticker": "NEWIPO",
            },
            index=dates_short,
        )
        long = pd.concat([aapl, newipo]).sort_index()

        mock_fetch.return_value = MagicMock()  # raw df (unused)
        mock_reshape.return_value = long

        result = fetch_universe(["AAPL", "NEWIPO"])
        assert "AAPL" in result["ticker"].values
        assert "NEWIPO" not in result["ticker"].values

    @patch("python.alpha.predict.reshape_ohlcv_wide_to_long")
    @patch("python.alpha.predict.fetch_ohlcv")
    def test_returns_long_format(self, mock_fetch, mock_reshape):
        long = _make_long_ohlcv(["AAPL", "MSFT"], n_days=100)
        mock_fetch.return_value = MagicMock()
        mock_reshape.return_value = long

        result = fetch_universe(["AAPL", "MSFT"])
        assert "ticker" in result.columns
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}


# ===========================================================================
# compute_features
# ===========================================================================


class TestComputeFeatures:
    def test_adds_feature_columns(self):
        """Feature pipeline adds expected columns to long-format data."""
        long = _make_long_ohlcv(["AAPL", "MSFT"], n_days=100)
        featured = compute_features(long)

        # Should have more columns than the input
        assert len(featured.columns) > len(long.columns)
        # Should contain key technical features
        for col in ["ret_5d", "rsi_14", "macd", "volume_ratio"]:
            assert col in featured.columns, f"Missing expected feature: {col}"


# ===========================================================================
# rank_stocks
# ===========================================================================


class TestRankStocks:
    def test_returns_top_n_tickers(self):
        """rank_stocks should return exactly top_n tickers."""
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        featured = _make_featured_df(tickers, n_days=50)

        # Create a mock model — feature_cols should exclude non-feature columns
        mock_model = MagicMock()
        mock_model.feature_cols = [c for c in featured.columns if c not in ("close", "ticker")]

        # Return deterministic scores — highest to last ticker
        def predict_side_effect(df):
            n = len(df)
            return np.arange(n, dtype=np.float64)

        mock_model.predict.side_effect = predict_side_effect

        top = rank_stocks(mock_model, featured, top_n=3)
        assert len(top) == 3
        assert all(isinstance(t, str) for t in top)

    def test_returns_empty_when_no_valid_data(self):
        """When all feature data is NaN, should return empty list."""
        from python.alpha.train import FEATURE_COLS

        dates = pd.bdate_range("2025-01-01", periods=5)
        # Need at least 2 tickers per date so .loc[date] returns a DataFrame
        # (a single row returns a Series, which has no .columns attribute).
        frames = []
        for ticker in ["AAPL", "MSFT"]:
            data = {col: [np.nan] * 5 for col in FEATURE_COLS}
            data["ticker"] = ticker
            df = pd.DataFrame(data, index=dates)
            frames.append(df)
        all_nan_df = pd.concat(frames).sort_index()

        mock_model = MagicMock()
        mock_model.feature_cols = FEATURE_COLS

        result = rank_stocks(mock_model, all_nan_df, top_n=5)
        assert result == []


# ===========================================================================
# optimize_weights
# ===========================================================================


class TestOptimizeWeights:
    @patch("python.alpha.predict.extract_close_prices")
    @patch("python.alpha.predict.fetch_ohlcv")
    def test_returns_normalized_weights(self, mock_fetch, mock_extract):
        """Weights should sum to ~1.0 after normalization."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-01", periods=100)
        prices = pd.DataFrame(
            {t: 100 + rng.standard_normal(100).cumsum() for t in tickers},
            index=dates,
        )
        mock_fetch.return_value = MagicMock()
        mock_extract.return_value = prices

        weights = optimize_weights(tickers, method="hrp")
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01
        for w in weights.values():
            assert w > 0

    @patch("python.alpha.predict.extract_close_prices")
    @patch("python.alpha.predict.fetch_ohlcv")
    def test_equal_weight_fallback(self, mock_fetch, mock_extract):
        """When not enough price data, falls back to equal weight."""
        # Return nearly empty price DataFrame
        mock_fetch.return_value = MagicMock()
        mock_extract.return_value = pd.DataFrame()

        tickers = ["AAPL", "MSFT", "GOOG"]
        weights = optimize_weights(tickers, method="hrp")
        # Should fall back to equal weight
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    @patch("python.alpha.predict.extract_close_prices")
    @patch("python.alpha.predict.fetch_ohlcv")
    def test_unknown_method_fallback_to_hrp(self, mock_fetch, mock_extract):
        """Unknown optimization method falls back to HRP."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-01", periods=100)
        prices = pd.DataFrame(
            {t: 100 + rng.standard_normal(100).cumsum() for t in tickers},
            index=dates,
        )
        mock_fetch.return_value = MagicMock()
        mock_extract.return_value = prices

        weights = optimize_weights(tickers, method="unknown_method")
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    @patch("python.alpha.predict.extract_close_prices")
    @patch("python.alpha.predict.fetch_ohlcv")
    def test_filters_near_zero_weights(self, mock_fetch, mock_extract):
        """Weights below 0.001 should be filtered out."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-01", periods=100)
        prices = pd.DataFrame(
            {t: 100 + rng.standard_normal(100).cumsum() for t in tickers},
            index=dates,
        )
        mock_fetch.return_value = MagicMock()
        mock_extract.return_value = prices

        weights = optimize_weights(tickers, method="hrp")
        for w in weights.values():
            assert w > 0.001


# ===========================================================================
# get_ml_weights (end-to-end with all sub-steps mocked)
# ===========================================================================


class TestGetMlWeightsOrchestration:
    """H11 update: get_ml_weights now fetches raw OHLCV once and passes it as
    ``price_data`` to optimize_weights (no double fetch).  It no longer calls
    fetch_universe; instead it calls fetch_ohlcv + reshape_ohlcv_wide_to_long
    directly, so mocks must match the new flow."""

    @patch("python.alpha.predict.optimize_weights")
    @patch("python.alpha.predict.rank_stocks")
    @patch("python.alpha.predict.compute_features")
    @patch("python.alpha.predict.reshape_ohlcv_wide_to_long")
    @patch("python.alpha.predict.fetch_ohlcv")
    @patch("python.alpha.predict.train_model")
    @patch("python.data.ingestion.fetch_sp500_tickers")
    def test_full_pipeline_happy_path(
        self,
        mock_sp500,
        mock_train,
        mock_fetch_ohlcv,
        mock_reshape,
        mock_features,
        mock_rank,
        mock_optimize,
    ):
        from python.alpha.predict import get_ml_weights

        mock_sp500.return_value = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        mock_train.return_value = MagicMock()  # trained model

        # Simulate raw OHLCV and long-format with enough rows per ticker
        raw_ohlcv = MagicMock()
        mock_fetch_ohlcv.return_value = raw_ohlcv

        dates = pd.bdate_range("2024-01-01", periods=100)
        rows = []
        for d in dates:
            for t in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]:
                rows.append({"ticker": t, "close": 100.0})
        long_df = pd.DataFrame(rows, index=np.tile(dates, 5))
        long_df.index.name = "date"
        long_df["ticker"] = [t for d in dates for t in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]]
        mock_reshape.return_value = long_df

        mock_features.return_value = MagicMock()  # featured data
        mock_rank.return_value = ["AAPL", "MSFT", "GOOG"]
        mock_optimize.return_value = {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}

        weights, stale_data = get_ml_weights(top_n=3, method="hrp")

        assert weights == {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}
        assert stale_data is False
        mock_train.assert_called_once()
        mock_rank.assert_called_once()
        # H11: optimize_weights now receives price_data (the raw OHLCV)
        mock_optimize.assert_called_once_with(
            ["AAPL", "MSFT", "GOOG"],
            method="hrp",
            current_weights=None,
            turnover_threshold=0.2,
            max_weight=None,
            price_data=raw_ohlcv,
        )

    @patch("python.alpha.predict.rank_stocks")
    @patch("python.alpha.predict.compute_features")
    @patch("python.alpha.predict.reshape_ohlcv_wide_to_long")
    @patch("python.alpha.predict.fetch_ohlcv")
    @patch("python.alpha.predict.train_model")
    @patch("python.data.ingestion.fetch_sp500_tickers")
    def test_returns_empty_when_no_picks(
        self,
        mock_sp500,
        mock_train,
        mock_fetch_ohlcv,
        mock_reshape,
        mock_features,
        mock_rank,
    ):
        """When rank_stocks returns empty, get_ml_weights returns empty dict."""
        from python.alpha.predict import get_ml_weights

        mock_sp500.return_value = ["AAPL", "MSFT"]
        mock_train.return_value = MagicMock()
        mock_fetch_ohlcv.return_value = MagicMock()

        dates = pd.bdate_range("2024-01-01", periods=100)
        rows = []
        for d in dates:
            for t in ["AAPL", "MSFT"]:
                rows.append({"ticker": t, "close": 100.0})
        long_df = pd.DataFrame(rows, index=np.tile(dates, 2))
        long_df.index.name = "date"
        long_df["ticker"] = [t for d in dates for t in ["AAPL", "MSFT"]]
        mock_reshape.return_value = long_df

        mock_features.return_value = MagicMock()
        mock_rank.return_value = []  # no picks

        weights, stale_data = get_ml_weights(top_n=10)
        assert weights == {}
        assert stale_data is False

    @patch("python.alpha.predict._load_cached_model")
    @patch("python.alpha.predict.train_model")
    def test_train_failure_propagates(self, mock_train, mock_load_cached):
        """If training fails and no cached model exists, exception propagates."""
        from python.alpha.predict import get_ml_weights

        mock_train.side_effect = RuntimeError("training failed")
        mock_load_cached.return_value = (None, False)  # no cached model

        with pytest.raises(RuntimeError, match="training failed"):
            get_ml_weights()
