"""Tests for survivorship-bias-free data pipeline.

Covers:
- fetch_tiingo_ohlcv(): Tiingo API fallback for delisted/acquired tickers
- fetch_ohlcv_with_delisted(): Combined yfinance + Tiingo data fetcher
- SurvivalUniverseProvider integration in train_model()
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import python.data.ingestion as ingestion_mod
from python.data.config import DELISTED_CACHE_DIR, TIINGO_API_TOKEN
from python.data.ingestion import fetch_ohlcv_with_delisted, fetch_tiingo_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiingo_api_response(ticker: str, n_days: int = 10) -> list[dict]:
    """Build a list of dicts mimicking Tiingo daily prices JSON response."""
    rng = np.random.default_rng(hash(ticker) % 2**31)
    base_price = rng.uniform(50, 300)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        price = base_price + rng.normal(0, 2)
        rows.append(
            {
                "date": d.strftime("%Y-%m-%dT00:00:00+00:00"),
                "close": price,
                "high": price + rng.uniform(0.5, 3),
                "low": price - rng.uniform(0.5, 3),
                "open": price + rng.normal(0, 1),
                "volume": int(rng.integers(500_000, 5_000_000)),
                "adjClose": price * 0.99,
                "adjHigh": (price + rng.uniform(0.5, 3)) * 0.99,
                "adjLow": (price - rng.uniform(0.5, 3)) * 0.99,
                "adjOpen": (price + rng.normal(0, 1)) * 0.99,
                "adjVolume": int(rng.integers(500_000, 5_000_000)),
            }
        )
    return rows


def _make_multiindex_ohlcv(
    tickers: list[str],
    n_days: int = 10,
    start: str = "2023-01-02",
) -> pd.DataFrame:
    """Build a MultiIndex DataFrame matching yfinance output format."""
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(42)
    panels = {}
    for ticker in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if field == "Volume":
                panels[(ticker, field)] = rng.integers(100_000, 1_000_000, size=n_days).astype(
                    float
                )
            else:
                panels[(ticker, field)] = rng.uniform(100, 200, size=n_days)

    df = pd.DataFrame(panels, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ===========================================================================
# fetch_tiingo_ohlcv
# ===========================================================================


class TestFetchTiingoOhlcv:
    """Tests for Tiingo API-based OHLCV fetcher."""

    def test_returns_empty_when_no_token(self, monkeypatch):
        """Should return empty DataFrame when TIINGO_API_TOKEN is not set."""
        # TIINGO_API_TOKEN is imported inside the function from python.data.config
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "")
        result = fetch_tiingo_ohlcv(["TWTR"], "2023-01-01", "2023-12-31")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_for_empty_tickers(self):
        """Should return empty DataFrame when ticker list is empty."""
        result = fetch_tiingo_ohlcv([], "2023-01-01", "2023-12-31")
        assert result.empty

    def test_successful_fetch_returns_multiindex(self, monkeypatch, tmp_path):
        """Successful API response should produce MultiIndex DataFrame."""
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        api_data = _make_tiingo_api_response("TWTR", n_days=5)
        response_bytes = json.dumps(api_data).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_tiingo_ohlcv(["TWTR"], "2023-01-01", "2023-12-31")

        assert not result.empty
        assert isinstance(result.columns, pd.MultiIndex)
        tickers = result.columns.get_level_values(0).unique()
        assert "TWTR" in tickers

        # Should have OHLCV columns
        fields = set(result.columns.get_level_values(1).unique())
        assert fields == {"Open", "High", "Low", "Close", "Volume"}

    def test_caches_fetched_data(self, monkeypatch, tmp_path):
        """Fetched data should be cached locally as parquet."""
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        api_data = _make_tiingo_api_response("ATVI", n_days=3)
        response_bytes = json.dumps(api_data).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            fetch_tiingo_ohlcv(["ATVI"], "2023-01-01", "2023-12-31")

        cache_file = tmp_path / "ATVI.parquet"
        assert cache_file.exists()

    def test_uses_cache_on_second_call(self, monkeypatch, tmp_path):
        """Second call should use cache, not hit API again."""
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        # Create cache file
        dates = pd.bdate_range("2023-01-02", periods=5)
        cached_df = pd.DataFrame(
            {
                "Open": np.arange(5, dtype=float) + 100,
                "High": np.arange(5, dtype=float) + 105,
                "Low": np.arange(5, dtype=float) + 95,
                "Close": np.arange(5, dtype=float) + 100,
                "Volume": np.arange(5, dtype=float) + 1_000_000,
            },
            index=dates,
        )
        (tmp_path / "CERN.parquet").parent.mkdir(parents=True, exist_ok=True)
        cached_df.to_parquet(tmp_path / "CERN.parquet")

        # urlopen should NOT be called
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = fetch_tiingo_ohlcv(["CERN"], "2023-01-02", "2023-01-06")
            mock_urlopen.assert_not_called()

        assert not result.empty
        assert "CERN" in result.columns.get_level_values(0).unique()

    def test_handles_404_gracefully(self, monkeypatch, tmp_path):
        """404 errors (unknown ticker) should be silently skipped."""
        import urllib.error

        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        def mock_urlopen(req, **kwargs):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = fetch_tiingo_ohlcv(["NOSUCH"], "2023-01-01", "2023-12-31")

        assert result.empty

    def test_handles_empty_api_response(self, monkeypatch, tmp_path):
        """Empty JSON array from API should be skipped."""
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"[]"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_tiingo_ohlcv(["FRC"], "2023-01-01", "2023-12-31")

        assert result.empty

    def test_multiple_tickers(self, monkeypatch, tmp_path):
        """Multiple tickers should produce combined MultiIndex DataFrame."""
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        call_count = 0

        def mock_urlopen(req, **kwargs):
            nonlocal call_count
            call_count += 1
            # Determine ticker from URL
            url = req.full_url
            if "TWTR" in url:
                data = _make_tiingo_api_response("TWTR", 3)
            elif "ATVI" in url:
                data = _make_tiingo_api_response("ATVI", 3)
            else:
                data = []

            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(data).encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = fetch_tiingo_ohlcv(["TWTR", "ATVI"], "2023-01-01", "2023-12-31")

        assert not result.empty
        tickers = set(result.columns.get_level_values(0).unique())
        assert tickers == {"TWTR", "ATVI"}
        assert call_count == 2

    def test_uses_adjusted_prices(self, monkeypatch, tmp_path):
        """Tiingo's adjusted prices (adjOpen etc.) should be used, not raw."""
        monkeypatch.setattr("python.data.config.TIINGO_API_TOKEN", "test_token")
        monkeypatch.setattr("python.data.config.DELISTED_CACHE_DIR", tmp_path)

        api_data = [
            {
                "date": "2023-01-02T00:00:00+00:00",
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 1_000_000,
                "adjOpen": 50.0,  # Different from raw
                "adjHigh": 55.0,
                "adjLow": 45.0,
                "adjClose": 52.5,
                "adjVolume": 2_000_000,
            }
        ]

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_data).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_tiingo_ohlcv(["TEST"], "2023-01-01", "2023-12-31")

        # Should use adjusted prices
        assert result[("TEST", "Close")].iloc[0] == 52.5
        assert result[("TEST", "Open")].iloc[0] == 50.0
        assert result[("TEST", "Volume")].iloc[0] == 2_000_000


# ===========================================================================
# fetch_ohlcv_with_delisted
# ===========================================================================


class TestFetchOhlcvWithDelisted:
    """Tests for the combined yfinance + Tiingo data fetcher."""

    def test_no_historical_tickers_returns_yfinance_only(self, monkeypatch):
        """When no historical tickers are missing, should return yfinance data only."""
        yf_df = _make_multiindex_ohlcv(["AAPL", "MSFT"], n_days=5)
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        result = fetch_ohlcv_with_delisted(
            current_tickers=["AAPL", "MSFT"],
            historical_tickers=[],
            period="2y",
        )
        assert set(result.columns.get_level_values(0).unique()) == {"AAPL", "MSFT"}

    def test_yfinance_resolves_historical_skips_tiingo(self, monkeypatch):
        """If yfinance resolves all historical tickers, Tiingo should not be called."""
        # yfinance returns data for both current and historical tickers
        yf_df = _make_multiindex_ohlcv(["AAPL", "MSFT", "XOM"], n_days=5)
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        with patch.object(ingestion_mod, "fetch_tiingo_ohlcv") as mock_tiingo:
            result = fetch_ohlcv_with_delisted(
                current_tickers=["AAPL", "MSFT"],
                historical_tickers=["XOM"],  # Still trading, yfinance has it
                period="2y",
            )
            mock_tiingo.assert_not_called()

        tickers = set(result.columns.get_level_values(0).unique())
        assert "XOM" in tickers

    def test_tiingo_fallback_for_missing_tickers(self, monkeypatch):
        """Tickers missing from yfinance should be fetched via Tiingo."""
        # yfinance returns only current tickers
        yf_df = _make_multiindex_ohlcv(["AAPL", "MSFT"], n_days=5)
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        # Tiingo returns delisted ticker
        tiingo_df = _make_multiindex_ohlcv(["TWTR"], n_days=3, start="2023-01-02")

        with patch.object(
            ingestion_mod, "fetch_tiingo_ohlcv", return_value=tiingo_df
        ) as mock_tiingo:
            result = fetch_ohlcv_with_delisted(
                current_tickers=["AAPL", "MSFT"],
                historical_tickers=["TWTR"],
                period="2y",
            )
            mock_tiingo.assert_called_once()

        tickers = set(result.columns.get_level_values(0).unique())
        assert tickers == {"AAPL", "MSFT", "TWTR"}

    def test_empty_tiingo_result_returns_yfinance_only(self, monkeypatch):
        """If Tiingo returns nothing, should gracefully return yfinance data."""
        yf_df = _make_multiindex_ohlcv(["AAPL"], n_days=5)
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        with patch.object(ingestion_mod, "fetch_tiingo_ohlcv", return_value=pd.DataFrame()):
            result = fetch_ohlcv_with_delisted(
                current_tickers=["AAPL"],
                historical_tickers=["FRC"],  # FDIC seizure — no data anywhere
                period="2y",
            )

        tickers = set(result.columns.get_level_values(0).unique())
        assert tickers == {"AAPL"}

    def test_infers_date_range_from_yfinance(self, monkeypatch):
        """Start/end dates for Tiingo should be inferred from yfinance data."""
        yf_df = _make_multiindex_ohlcv(["AAPL"], n_days=10, start="2023-06-01")
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        captured_args = {}

        def capture_tiingo(tickers, start_date, end_date):
            captured_args["start"] = start_date
            captured_args["end"] = end_date
            return pd.DataFrame()

        with patch.object(ingestion_mod, "fetch_tiingo_ohlcv", side_effect=capture_tiingo):
            fetch_ohlcv_with_delisted(
                current_tickers=["AAPL"],
                historical_tickers=["TWTR"],
                period="2y",
            )

        # Should match yfinance date range
        assert captured_args["start"] == yf_df.index.min().strftime("%Y-%m-%d")
        assert captured_args["end"] == yf_df.index.max().strftime("%Y-%m-%d")

    def test_merged_dataframe_has_aligned_dates(self, monkeypatch):
        """Merged DataFrame should have both tickers' data on overlapping dates."""
        yf_df = _make_multiindex_ohlcv(["AAPL"], n_days=10, start="2023-01-02")
        # Delisted ticker only has 5 days (stopped trading earlier)
        tiingo_df = _make_multiindex_ohlcv(["TWTR"], n_days=5, start="2023-01-02")
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        with patch.object(ingestion_mod, "fetch_tiingo_ohlcv", return_value=tiingo_df):
            result = fetch_ohlcv_with_delisted(
                current_tickers=["AAPL"],
                historical_tickers=["TWTR"],
                period="2y",
            )

        # AAPL should have 10 days
        aapl_close = result[("AAPL", "Close")]
        assert aapl_close.notna().sum() == 10

        # TWTR should have 5 days, NaN for the rest
        twtr_close = result[("TWTR", "Close")]
        assert twtr_close.notna().sum() == 5

    def test_logs_missing_count(self, monkeypatch, caplog):
        """Should log how many tickers were resolved by each source."""
        yf_df = _make_multiindex_ohlcv(["AAPL"], n_days=5)
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        tiingo_df = _make_multiindex_ohlcv(["TWTR"], n_days=3)

        with patch.object(ingestion_mod, "fetch_tiingo_ohlcv", return_value=tiingo_df):
            with caplog.at_level(logging.INFO, logger="python.data.ingestion"):
                fetch_ohlcv_with_delisted(
                    current_tickers=["AAPL"],
                    historical_tickers=["TWTR"],
                    period="2y",
                )

        # Should log the combined ticker count
        assert any("Combined OHLCV" in rec.message for rec in caplog.records)


# ===========================================================================
# Config: TIINGO_API_TOKEN and DELISTED_CACHE_DIR
# ===========================================================================


class TestTiingoConfig:
    """Verify Tiingo config is properly defined."""

    def test_tiingo_token_defined(self):
        """TIINGO_API_TOKEN should be defined in config (may be empty)."""
        from python.data.config import TIINGO_API_TOKEN

        assert isinstance(TIINGO_API_TOKEN, str)

    def test_delisted_cache_dir_defined(self):
        """DELISTED_CACHE_DIR should be a Path object."""
        from python.data.config import DELISTED_CACHE_DIR

        assert isinstance(DELISTED_CACHE_DIR, Path)
        assert "delisted" in str(DELISTED_CACHE_DIR)


# ===========================================================================
# Integration: SurvivalUniverseProvider plumbing in train_model
# ===========================================================================


class TestSurvivalUniversePlumbing:
    """Test the survivorship-bias logic from train_model() in isolation.

    We replicate the exact code path from predict.py lines 525-559 here
    rather than calling train_model() directly, which drags in the full
    ML pipeline and hangs on feature computation.
    """

    def _run_survivorship_logic(
        self,
        current_tickers: list[str],
        provider_cls,
        fetch_ohlcv_fn,
        fetch_with_delisted_fn,
    ) -> tuple[object, list[str]]:
        """Replicate the survivorship-bias plumbing from train_model().

        Returns (raw_data, historical_removals) so callers can assert.
        """
        from datetime import date as _date
        from datetime import timedelta as _td

        tickers = current_tickers

        historical_removals: list[str] = []
        try:
            provider = provider_cls()
            lookback_start = _date.today() - _td(days=365 * 2)
            historical_universe = provider.get_universe(lookback_start)
            historical_removals = sorted(set(historical_universe) - set(tickers))
        except Exception:
            pass

        if historical_removals:
            raw = fetch_with_delisted_fn(
                current_tickers=tickers,
                historical_tickers=historical_removals,
                period="2y",
            )
        else:
            raw = fetch_ohlcv_fn(tickers, period="2y")

        return raw, historical_removals

    def test_detects_removed_tickers(self):
        """Provider returning extra tickers should produce historical_removals."""

        class MockProvider:
            def get_universe(self, as_of):
                return ["AAPL", "MSFT", "GOOG", "TWTR", "ATVI"]

        calls = []

        def mock_with_delisted(**kwargs):
            calls.append(kwargs)
            return _make_multiindex_ohlcv(
                kwargs["current_tickers"] + kwargs["historical_tickers"],
                n_days=10,
            )

        _, removals = self._run_survivorship_logic(
            current_tickers=["AAPL", "MSFT", "GOOG"],
            provider_cls=MockProvider,
            fetch_ohlcv_fn=lambda *a, **kw: None,
            fetch_with_delisted_fn=mock_with_delisted,
        )
        assert set(removals) == {"ATVI", "TWTR"}
        assert len(calls) == 1
        assert set(calls[0]["historical_tickers"]) == {"ATVI", "TWTR"}

    def test_no_removals_uses_regular_fetch(self):
        """When provider returns same tickers, should use regular fetch_ohlcv."""

        class MockProvider:
            def get_universe(self, as_of):
                return ["AAPL", "MSFT"]

        regular_calls = []

        def mock_fetch(tickers, **kw):
            regular_calls.append(tickers)
            return _make_multiindex_ohlcv(tickers, n_days=5)

        _, removals = self._run_survivorship_logic(
            current_tickers=["AAPL", "MSFT"],
            provider_cls=MockProvider,
            fetch_ohlcv_fn=mock_fetch,
            fetch_with_delisted_fn=lambda **kw: None,
        )
        assert removals == []
        assert len(regular_calls) == 1

    def test_provider_failure_falls_back_to_regular(self):
        """If provider raises, should fall back to regular fetch_ohlcv."""

        class FailingProvider:
            def get_universe(self, as_of):
                raise RuntimeError("Network error")

        regular_calls = []

        def mock_fetch(tickers, **kw):
            regular_calls.append(tickers)
            return _make_multiindex_ohlcv(tickers, n_days=5)

        _, removals = self._run_survivorship_logic(
            current_tickers=["AAPL", "MSFT"],
            provider_cls=FailingProvider,
            fetch_ohlcv_fn=mock_fetch,
            fetch_with_delisted_fn=lambda **kw: None,
        )
        assert removals == []
        assert len(regular_calls) == 1

    def test_code_path_exists_in_predict_module(self):
        """Verify the survivorship-bias code exists in train_model source."""
        import inspect

        import python.alpha.predict as predict_mod

        source = inspect.getsource(predict_mod.train_model)
        assert "SurvivalUniverseProvider" in source
        assert "fetch_ohlcv_with_delisted" in source
        assert "historical_removals" in source


# ===========================================================================
# Suggested test: Tiingo fallback for truly delisted ticker
# ===========================================================================


class TestTiingoFallbackForDelisted:
    """Test that yfinance failure → Tiingo fallback → merged DataFrame works
    end-to-end for a known delisted ticker (e.g., SIVB)."""

    def test_yfinance_fails_tiingo_succeeds(self, monkeypatch):
        """When yfinance returns empty for a delisted ticker, Tiingo fills the gap."""
        # yfinance only returns current tickers — SIVB is missing
        yf_df = _make_multiindex_ohlcv(["AAPL", "MSFT"], n_days=10)
        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", lambda *a, **kw: yf_df)

        # Tiingo returns SIVB data (shorter — delisted partway through)
        tiingo_df = _make_multiindex_ohlcv(["SIVB"], n_days=6, start="2023-01-02")

        tiingo_called_with = {}

        def mock_tiingo(tickers, start_date, end_date):
            tiingo_called_with["tickers"] = tickers
            return tiingo_df

        with patch.object(ingestion_mod, "fetch_tiingo_ohlcv", side_effect=mock_tiingo):
            result = fetch_ohlcv_with_delisted(
                current_tickers=["AAPL", "MSFT"],
                historical_tickers=["SIVB"],
                period="2y",
            )

        # Tiingo should have been called with SIVB
        assert tiingo_called_with["tickers"] == ["SIVB"]

        # All three tickers should be in the result
        tickers = set(result.columns.get_level_values(0).unique())
        assert tickers == {"AAPL", "MSFT", "SIVB"}

        # SIVB should have fewer non-NaN rows (it was delisted)
        sivb_close = result[("SIVB", "Close")]
        aapl_close = result[("AAPL", "Close")]
        assert sivb_close.notna().sum() < aapl_close.notna().sum()


# ===========================================================================
# Suggested test: NaN handling in cross-sectional features when ticker drops
# ===========================================================================


class TestCrossSectionalNaNDropout:
    """Verify that compute_cross_sectional_features handles tickers that
    abruptly stop having data halfway through the time series.

    This validates the 'natural drop-out' theory: when a delisted ticker's
    data ends, later dates should rank the remaining tickers out of the
    correct (smaller) denominator, and the dead ticker should get NaN ranks
    rather than throwing an error.
    """

    def _make_dropout_dataframe(self) -> pd.DataFrame:
        """Create a long-format DataFrame where one ticker dies midway.

        5 tickers × 10 dates, but 'DEAD' has NaN after day 5.
        """
        dates = pd.bdate_range("2023-01-02", periods=10)
        rng = np.random.default_rng(42)
        rows = []
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "DEAD"]
        for d in dates:
            for t in tickers:
                day_idx = list(dates).index(d)
                if t == "DEAD" and day_idx >= 5:
                    # Dead ticker: still has rows, but NaN values
                    rows.append(
                        {
                            "date": d,
                            "ticker": t,
                            "close": np.nan,
                            "open": np.nan,
                            "high": np.nan,
                            "low": np.nan,
                            "volume": np.nan,
                            "ret_5d": np.nan,
                            "ret_20d": np.nan,
                            "vol_20d": np.nan,
                            "dollar_volume_20d": np.nan,
                        }
                    )
                else:
                    rows.append(
                        {
                            "date": d,
                            "ticker": t,
                            "close": rng.uniform(100, 200),
                            "open": rng.uniform(100, 200),
                            "high": rng.uniform(100, 200),
                            "low": rng.uniform(100, 200),
                            "volume": float(rng.integers(100_000, 1_000_000)),
                            "ret_5d": rng.normal(0, 0.05),
                            "ret_20d": rng.normal(0, 0.10),
                            "vol_20d": rng.uniform(0.01, 0.05),
                            "dollar_volume_20d": rng.uniform(1e7, 1e9),
                        }
                    )
        df = pd.DataFrame(rows)
        df = df.set_index("date")
        return df

    def test_no_errors_on_dropout(self):
        """compute_cross_sectional_features should not raise when a ticker drops out."""
        from python.alpha.features import compute_cross_sectional_features

        df = self._make_dropout_dataframe()
        # Should complete without error
        result = compute_cross_sectional_features(df)
        assert "cs_ret_rank_5d" in result.columns
        assert "cs_ret_rank_20d" in result.columns

    def test_dead_ticker_gets_nan_ranks_after_dropout(self):
        """After the dead ticker stops having data, its rank columns should be NaN."""
        from python.alpha.features import compute_cross_sectional_features

        df = self._make_dropout_dataframe()
        result = compute_cross_sectional_features(df)

        # Get dead ticker's ranks for dates after dropout (day 5+)
        dead_rows = result[result["ticker"] == "DEAD"]
        dates = sorted(dead_rows.index.unique())
        late_dates = dates[5:]  # dates where DEAD has NaN data

        for d in late_dates:
            dead_on_d = dead_rows.loc[d]
            if isinstance(dead_on_d, pd.DataFrame):
                dead_on_d = dead_on_d.iloc[0]
            # All rank columns should be NaN (na_option='keep')
            assert pd.isna(dead_on_d["cs_ret_rank_5d"]), (
                f"DEAD ticker should have NaN rank on {d}, got {dead_on_d['cs_ret_rank_5d']}"
            )

    def test_surviving_tickers_rank_denominator_adjusts(self):
        """After dropout, surviving tickers should be ranked out of 4, not 5.

        With na_option='keep', NaN values are excluded from the ranking.
        So with 4 live tickers + 1 dead (NaN), pct=True should produce
        ranks from {0.25, 0.5, 0.75, 1.0} — i.e., denominator = 4.
        """
        from python.alpha.features import compute_cross_sectional_features

        df = self._make_dropout_dataframe()
        result = compute_cross_sectional_features(df)

        dates = sorted(result.index.unique())

        # Before dropout (5 tickers alive): ranks should use denominator 5
        early_date = dates[2]
        early_ranks = result.loc[early_date, "cs_ret_rank_5d"].dropna()
        assert len(early_ranks) == 5, f"Expected 5 ranked tickers on {early_date}"
        # With 5 tickers, possible ranks are {0.2, 0.4, 0.6, 0.8, 1.0}
        expected_5 = {0.2, 0.4, 0.6, 0.8, 1.0}
        for r in early_ranks:
            assert any(abs(r - e) < 0.01 for e in expected_5), (
                f"Unexpected rank {r} with 5 tickers (expected one of {expected_5})"
            )

        # After dropout (4 tickers alive): ranks should use denominator 4
        late_date = dates[7]
        late_alive = result.loc[late_date]
        late_alive = late_alive[late_alive["ticker"] != "DEAD"]
        late_ranks = late_alive["cs_ret_rank_5d"].dropna()
        assert len(late_ranks) == 4, f"Expected 4 ranked tickers on {late_date}"
        # With 4 tickers, possible ranks are {0.25, 0.5, 0.75, 1.0}
        expected_4 = {0.25, 0.5, 0.75, 1.0}
        for r in late_ranks:
            assert any(abs(r - e) < 0.01 for e in expected_4), (
                f"Unexpected rank {r} with 4 tickers (expected one of {expected_4})"
            )

    def test_early_dates_all_five_tickers_ranked(self):
        """Before dropout, all 5 tickers should have valid (non-NaN) ranks."""
        from python.alpha.features import compute_cross_sectional_features

        df = self._make_dropout_dataframe()
        result = compute_cross_sectional_features(df)

        dates = sorted(result.index.unique())
        early_date = dates[0]
        ranks = result.loc[early_date, "cs_ret_rank_5d"]
        assert ranks.notna().sum() == 5


# ===========================================================================
# Suggested test: Universe union deduplication
# ===========================================================================


class TestUniverseUnion:
    """Verify set(current_tickers) | set(historical_tickers) deduplication."""

    def test_deduplicates_overlap(self):
        """Tickers in both current and historical should appear only once."""
        current = ["AAPL", "MSFT", "GOOG", "XOM"]
        historical = ["XOM", "TWTR", "ATVI"]

        union = sorted(set(current) | set(historical))
        assert len(union) == 6  # AAPL, ATVI, GOOG, MSFT, TWTR, XOM
        assert union == ["AAPL", "ATVI", "GOOG", "MSFT", "TWTR", "XOM"]

    def test_no_overlap(self):
        """When lists are disjoint, union should be the concatenation."""
        current = ["AAPL", "MSFT"]
        historical = ["TWTR", "ATVI"]

        union = sorted(set(current) | set(historical))
        assert len(union) == 4

    def test_empty_historical(self):
        """Empty historical list should return only current."""
        current = ["AAPL", "MSFT"]
        historical: list[str] = []

        union = sorted(set(current) | set(historical))
        assert union == ["AAPL", "MSFT"]

    def test_does_not_drop_valid_names(self):
        """All valid names from both lists must appear in the union."""
        current = ["AAPL", "MSFT", "GOOG"]
        historical = ["TWTR", "ATVI", "SIVB", "FRC", "CERN"]

        union = set(current) | set(historical)
        for t in current + historical:
            assert t in union, f"{t} was dropped from the union"

    def test_fetch_ohlcv_with_delisted_passes_union_to_yfinance(self, monkeypatch):
        """fetch_ohlcv_with_delisted should pass current ∪ historical to yfinance."""
        captured_tickers = []

        def mock_fetch_ohlcv(tickers, **kw):
            captured_tickers.extend(tickers)
            return _make_multiindex_ohlcv(tickers, n_days=5)

        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", mock_fetch_ohlcv)

        fetch_ohlcv_with_delisted(
            current_tickers=["AAPL", "MSFT"],
            historical_tickers=["XOM", "TWTR"],
            period="2y",
        )

        # Should have passed the sorted union to yfinance
        assert set(captured_tickers) == {"AAPL", "MSFT", "XOM", "TWTR"}

    def test_overlapping_tickers_not_double_fetched(self, monkeypatch):
        """If a ticker is in both current and historical, it should not be fetched twice."""
        yf_call_count = 0

        def mock_fetch_ohlcv(tickers, **kw):
            nonlocal yf_call_count
            yf_call_count += 1
            # XOM is in both lists but should only appear once
            assert tickers.count("XOM") <= 1, "XOM appeared multiple times in fetch"
            return _make_multiindex_ohlcv(tickers, n_days=5)

        monkeypatch.setattr(ingestion_mod, "fetch_ohlcv", mock_fetch_ohlcv)

        fetch_ohlcv_with_delisted(
            current_tickers=["AAPL", "MSFT", "XOM"],
            historical_tickers=["XOM", "TWTR"],
            period="2y",
        )
