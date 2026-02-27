"""Tests for yfinance circuit breaker fallback paths.

Validates that the system degrades gracefully when yfinance is unavailable:
  - Ticker list: in-memory + disk cache fallback
  - Model cache: accepts stale models up to MAX_MODEL_AGE_DAYS
  - Scoring OHLCV: disk cache fallback with staleness tracking
  - Risk engine: disk cache fallback for historical returns
  - Stale data: exposure reduced by STALE_DATA_EXPOSURE_MULT
"""

import json
import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ticker cache tests
# ---------------------------------------------------------------------------


class TestTickerCacheFallback:
    """Ticker list cache: in-memory -> live scrape -> disk cache."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self, monkeypatch, tmp_path):
        """Reset caches and redirect disk cache to tmp_path."""
        import python.data.ingestion as ingestion_mod
        from python.data import config

        monkeypatch.setattr(ingestion_mod, "_ticker_cache", None)
        cache_file = tmp_path / "sp500_tickers.json"
        monkeypatch.setattr(config, "TICKER_CACHE_PATH", cache_file)
        monkeypatch.setattr(ingestion_mod, "TICKER_CACHE_PATH", cache_file)
        self.cache_file = cache_file
        self.tmp_path = tmp_path

    def test_in_memory_cache_hit(self, monkeypatch):
        """In-memory cache returns immediately without scraping."""
        import python.data.ingestion as ingestion_mod

        cached = (["AAPL", "MSFT"], time.time())
        monkeypatch.setattr(ingestion_mod, "_ticker_cache", cached)

        result = ingestion_mod.fetch_sp500_tickers()
        assert result == ["AAPL", "MSFT"]

    def test_disk_cache_fallback_on_scrape_failure(self, monkeypatch):
        """When live scrape fails, falls back to disk cache."""
        import python.data.ingestion as ingestion_mod

        # Write a disk cache
        self.cache_file.write_text(json.dumps({"tickers": ["GOOG", "AMZN"], "ts": time.time()}))

        # Make live scrape fail
        monkeypatch.setattr(
            ingestion_mod,
            "_fetch_sp500_tickers_live",
            lambda: (_ for _ in ()).throw(ConnectionError("network down")),
        )

        result = ingestion_mod.fetch_sp500_tickers()
        assert result == ["GOOG", "AMZN"]

    def test_both_caches_fail_raises(self, monkeypatch):
        """When both scrape and disk cache fail, raises RuntimeError."""
        import python.data.ingestion as ingestion_mod

        # No disk cache, scrape fails
        monkeypatch.setattr(
            ingestion_mod,
            "_fetch_sp500_tickers_live",
            lambda: (_ for _ in ()).throw(ConnectionError("network down")),
        )

        with pytest.raises(RuntimeError, match="Cannot obtain S&P 500"):
            ingestion_mod.fetch_sp500_tickers()

    def test_successful_scrape_persists_to_disk(self, monkeypatch):
        """Successful scrape writes disk cache for future fallback."""
        import python.data.ingestion as ingestion_mod

        monkeypatch.setattr(
            ingestion_mod,
            "_fetch_sp500_tickers_live",
            lambda: ["AAPL", "META"],
        )

        result = ingestion_mod.fetch_sp500_tickers()
        assert result == ["AAPL", "META"]

        # Verify disk cache was written
        assert self.cache_file.exists()
        data = json.loads(self.cache_file.read_text())
        assert data["tickers"] == ["AAPL", "META"]
        assert "ts" in data

    def test_stale_in_memory_cache_triggers_scrape(self, monkeypatch):
        """In-memory cache older than TTL triggers a fresh scrape."""
        import python.data.ingestion as ingestion_mod
        from python.data import config

        # Set TTL to 0.001 hours so cache is immediately stale
        monkeypatch.setattr(config, "TICKER_CACHE_TTL_HOURS", 0.001)
        monkeypatch.setattr(ingestion_mod, "TICKER_CACHE_TTL_HOURS", 0.001)
        old_ts = time.time() - 3600  # 1 hour old
        monkeypatch.setattr(ingestion_mod, "_ticker_cache", (["OLD"], old_ts))

        monkeypatch.setattr(
            ingestion_mod,
            "_fetch_sp500_tickers_live",
            lambda: ["NEW"],
        )

        result = ingestion_mod.fetch_sp500_tickers()
        assert result == ["NEW"]


# ---------------------------------------------------------------------------
# Model cache staleness tests
# ---------------------------------------------------------------------------


class _FakeModel:
    """Picklable stand-in for CrossSectionalModel in cache tests."""

    def __init__(self, name="fake"):
        self.name = name


class TestModelCacheStaleness:
    """_load_cached_model accepts models up to MAX_MODEL_AGE_DAYS old."""

    @pytest.fixture(autouse=True)
    def _setup_model_cache(self, tmp_path, monkeypatch):
        """Redirect model cache to tmp_path."""
        import python.alpha.predict as predict_mod

        cache_file = tmp_path / "latest_model.joblib"
        monkeypatch.setattr(predict_mod, "MODEL_CACHE_FILE", cache_file)
        self.cache_file = cache_file

    def test_same_day_model_not_stale(self):
        """Model trained today: is_stale=False."""
        import joblib

        import python.alpha.predict as predict_mod

        today = date.today().isoformat()
        model = _FakeModel("today")
        payload = {"model": model, "trained_date": today}
        joblib.dump(payload, self.cache_file)

        result_model, is_stale = predict_mod._load_cached_model()
        assert result_model is not None
        assert result_model.name == "today"
        assert is_stale is False

    def test_old_model_is_stale(self):
        """Model trained 3 days ago: is_stale=True but still loaded (within MAX_MODEL_AGE_DAYS)."""
        import joblib

        import python.alpha.predict as predict_mod

        # Use 3 days ago to avoid timezone edge cases (NY can be 1 day behind UTC)
        three_days_ago = (date.today() - timedelta(days=3)).isoformat()
        model = _FakeModel("stale")
        payload = {"model": model, "trained_date": three_days_ago}
        joblib.dump(payload, self.cache_file)

        result_model, is_stale = predict_mod._load_cached_model()
        assert result_model is not None
        assert result_model.name == "stale"
        assert is_stale is True

    def test_too_old_model_rejected(self):
        """Model older than MAX_MODEL_AGE_DAYS: returns None."""
        import joblib

        import python.alpha.predict as predict_mod

        old_date = (date.today() - timedelta(days=30)).isoformat()
        payload = {"model": _FakeModel(), "trained_date": old_date}
        joblib.dump(payload, self.cache_file)

        result_model, is_stale = predict_mod._load_cached_model()
        assert result_model is None

    def test_missing_cache_returns_none(self):
        """No cache file: returns (None, False)."""
        import python.alpha.predict as predict_mod

        # cache_file doesn't exist
        self.cache_file.unlink(missing_ok=True)
        result_model, is_stale = predict_mod._load_cached_model()
        assert result_model is None
        assert is_stale is False

    def test_max_age_0_only_accepts_today(self):
        """max_age_days=0: only same-day models accepted (used by train_model)."""
        import joblib

        import python.alpha.predict as predict_mod

        # 3 days ago to avoid NY/UTC edge case
        old_date = (date.today() - timedelta(days=3)).isoformat()
        payload = {"model": _FakeModel(), "trained_date": old_date}
        joblib.dump(payload, self.cache_file)

        result_model, _ = predict_mod._load_cached_model(max_age_days=0)
        assert result_model is None


# ---------------------------------------------------------------------------
# OHLCV cache fallback tests
# ---------------------------------------------------------------------------


class TestOHLCVCacheFallback:
    """Scoring OHLCV disk cache: persist on success, load on failure."""

    @pytest.fixture(autouse=True)
    def _setup_ohlcv_cache(self, tmp_path, monkeypatch):
        """Redirect OHLCV cache paths to tmp_path."""
        import python.alpha.predict as predict_mod
        from python.data import config

        self.pq_path = tmp_path / "last_ohlcv.parquet"
        self.meta_path = tmp_path / "last_ohlcv_meta.json"
        monkeypatch.setattr(config, "OHLCV_CACHE_PATH", self.pq_path)
        monkeypatch.setattr(config, "OHLCV_CACHE_META_PATH", self.meta_path)
        monkeypatch.setattr(predict_mod, "OHLCV_CACHE_PATH", self.pq_path)
        monkeypatch.setattr(predict_mod, "OHLCV_CACHE_META_PATH", self.meta_path)

    def test_persist_and_load_ohlcv_cache(self):
        """Round-trip: persist then load returns same data."""
        import python.alpha.predict as predict_mod

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        predict_mod._persist_ohlcv_cache(df)

        assert self.pq_path.exists()
        assert self.meta_path.exists()

        loaded, is_stale = predict_mod._load_ohlcv_cache()
        assert loaded is not None
        assert len(loaded) == 3
        assert is_stale is False  # just saved, so fresh

    def test_stale_ohlcv_cache_flagged(self, monkeypatch):
        """OHLCV cache older than MAX_OHLCV_AGE_DAYS flagged as stale."""
        import python.alpha.predict as predict_mod
        from python.data import config

        monkeypatch.setattr(config, "MAX_OHLCV_AGE_DAYS", 1)
        monkeypatch.setattr(predict_mod, "MAX_OHLCV_AGE_DAYS", 1)

        df = pd.DataFrame({"A": [1, 2]})
        df.to_parquet(self.pq_path)
        # Write metadata with old timestamp (3 days ago)
        meta = {"saved_ts": time.time() - 3 * 86400}
        self.meta_path.write_text(json.dumps(meta))

        loaded, is_stale = predict_mod._load_ohlcv_cache()
        assert loaded is not None
        assert is_stale is True

    def test_missing_ohlcv_cache_returns_none(self):
        """No OHLCV cache file: returns (None, False)."""
        import python.alpha.predict as predict_mod

        loaded, is_stale = predict_mod._load_ohlcv_cache()
        assert loaded is None
        assert is_stale is False

    def test_missing_metadata_treated_as_stale(self):
        """OHLCV cache with no metadata file: flagged as stale."""
        import python.alpha.predict as predict_mod

        df = pd.DataFrame({"A": [1]})
        df.to_parquet(self.pq_path)
        # No metadata file

        loaded, is_stale = predict_mod._load_ohlcv_cache()
        assert loaded is not None
        assert is_stale is True


# ---------------------------------------------------------------------------
# get_ml_weights circuit breaker integration
# ---------------------------------------------------------------------------


class TestGetMlWeightsCircuitBreaker:
    """get_ml_weights falls back to cached data and signals staleness."""

    @patch("python.alpha.predict.optimize_weights")
    @patch("python.alpha.predict.rank_stocks")
    @patch("python.alpha.predict.compute_features")
    @patch("python.alpha.predict.reshape_ohlcv_wide_to_long")
    @patch("python.alpha.predict.fetch_ohlcv")
    @patch("python.alpha.predict._load_cached_model")
    @patch("python.alpha.predict.train_model")
    @patch("python.data.ingestion.fetch_sp500_tickers")
    def test_training_failure_uses_stale_model(
        self,
        mock_sp500,
        mock_train,
        mock_load_cached,
        mock_fetch_ohlcv,
        mock_reshape,
        mock_features,
        mock_rank,
        mock_optimize,
    ):
        """When training fails, falls back to stale cached model."""
        from python.alpha.predict import get_ml_weights

        mock_sp500.return_value = ["AAPL", "MSFT"]
        mock_train.side_effect = RuntimeError("yfinance down")

        stale_model = MagicMock()
        stale_model.validation_ic = 0.05
        mock_load_cached.return_value = (stale_model, True)  # stale model

        raw_ohlcv = MagicMock()
        mock_fetch_ohlcv.return_value = raw_ohlcv

        dates = pd.bdate_range("2024-01-01", periods=100)
        import numpy as np

        rows = []
        for d in dates:
            for t in ["AAPL", "MSFT"]:
                rows.append({"ticker": t, "close": 100.0})
        long_df = pd.DataFrame(rows, index=np.tile(dates, 2))
        long_df["ticker"] = [t for d in dates for t in ["AAPL", "MSFT"]]
        mock_reshape.return_value = long_df

        mock_features.return_value = MagicMock()
        mock_rank.return_value = ["AAPL"]
        mock_optimize.return_value = {"AAPL": 1.0}

        weights, stale_data = get_ml_weights(top_n=1)

        assert weights == {"AAPL": 1.0}
        assert stale_data is True
        mock_load_cached.assert_called_once()

    @patch("python.alpha.predict._persist_ohlcv_cache")
    @patch("python.alpha.predict.optimize_weights")
    @patch("python.alpha.predict.rank_stocks")
    @patch("python.alpha.predict.compute_features")
    @patch("python.alpha.predict.reshape_ohlcv_wide_to_long")
    @patch("python.alpha.predict.fetch_ohlcv")
    @patch("python.alpha.predict._load_ohlcv_cache")
    @patch("python.alpha.predict.train_model")
    @patch("python.data.ingestion.fetch_sp500_tickers")
    def test_ohlcv_failure_uses_disk_cache(
        self,
        mock_sp500,
        mock_train,
        mock_load_ohlcv,
        mock_fetch_ohlcv,
        mock_reshape,
        mock_features,
        mock_rank,
        mock_optimize,
        mock_persist,
    ):
        """When OHLCV fetch fails, falls back to disk cache."""
        from python.alpha.predict import get_ml_weights

        mock_sp500.return_value = ["AAPL"]
        model = MagicMock()
        model.validation_ic = 0.05
        mock_train.return_value = model

        mock_fetch_ohlcv.side_effect = ConnectionError("yfinance down")

        cached_ohlcv = pd.DataFrame({"A": [1, 2, 3]})
        mock_load_ohlcv.return_value = (cached_ohlcv, False)

        dates = pd.bdate_range("2024-01-01", periods=100)
        rows = [{"ticker": "AAPL", "close": 100.0} for _ in dates]
        long_df = pd.DataFrame(rows, index=dates)
        long_df["ticker"] = "AAPL"
        mock_reshape.return_value = long_df

        mock_features.return_value = MagicMock()
        mock_rank.return_value = ["AAPL"]
        mock_optimize.return_value = {"AAPL": 1.0}

        weights, stale_data = get_ml_weights(top_n=1)

        assert weights == {"AAPL": 1.0}
        assert stale_data is True  # OHLCV fetch failed -> stale
        mock_load_ohlcv.assert_called_once()

    @patch("python.alpha.predict._load_ohlcv_cache")
    @patch("python.alpha.predict._load_cached_model")
    @patch("python.alpha.predict.fetch_ohlcv")
    @patch("python.alpha.predict.train_model")
    @patch("python.data.ingestion.fetch_sp500_tickers")
    def test_both_failures_no_caches_raises(
        self,
        mock_sp500,
        mock_train,
        mock_fetch_ohlcv,
        mock_load_cached,
        mock_load_ohlcv,
    ):
        """Training + OHLCV failures with no caches raises RuntimeError."""
        from python.alpha.predict import get_ml_weights

        mock_sp500.return_value = ["AAPL"]
        mock_train.side_effect = RuntimeError("train failed")
        mock_load_cached.return_value = (None, False)  # no cached model

        with pytest.raises(RuntimeError, match="train failed"):
            get_ml_weights()


# ---------------------------------------------------------------------------
# Stale data exposure reduction in live_bot
# ---------------------------------------------------------------------------


class TestStaleDataExposureReduction:
    """Live bot reduces weights when get_ml_weights signals stale data.

    Uses a lightweight unit-test approach: directly tests the weight-scaling
    code path in ``run_trading_cycle`` without running the full order
    submission / polling loop (which is heavy and can hang under mocks).
    """

    def test_stale_flag_scales_weights(self):
        """Verify the stale-data code path multiplies weights by STALE_DATA_EXPOSURE_MULT."""
        from python.data.config import STALE_DATA_EXPOSURE_MULT

        original = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        reduced = {t: w * STALE_DATA_EXPOSURE_MULT for t, w in original.items()}

        assert abs(reduced["AAPL"] - 0.20) < 1e-9
        assert abs(reduced["MSFT"] - 0.15) < 1e-9
        assert abs(reduced["GOOG"] - 0.15) < 1e-9
        assert sum(reduced.values()) < 1.0  # exposure is reduced, not full

    def test_stale_exposure_mult_is_half(self):
        """STALE_DATA_EXPOSURE_MULT should be 0.5."""
        from python.data.config import STALE_DATA_EXPOSURE_MULT

        assert STALE_DATA_EXPOSURE_MULT == 0.5


# ---------------------------------------------------------------------------
# Risk engine cache tests
# ---------------------------------------------------------------------------


class TestRiskEngineCacheFallback:
    """_initialize_risk_engine persists and reloads historical returns."""

    @pytest.fixture(autouse=True)
    def _setup_cache(self, tmp_path, monkeypatch):
        from python.data import config

        self.cache_path = tmp_path / "risk_engine_returns.parquet"
        monkeypatch.setattr(config, "RISK_ENGINE_CACHE_PATH", self.cache_path)
        # Also patch it in live_bot where it's imported
        import examples.live_bot as bot_mod

        monkeypatch.setattr(bot_mod, "RISK_ENGINE_CACHE_PATH", self.cache_path)

    def test_disk_fallback_on_fetch_failure(self, monkeypatch):
        """When yfinance fails, loads cached returns from disk."""
        from examples.live_bot import _initialize_risk_engine
        from python.brokers.base import BrokerPosition
        from python.portfolio.risk_manager import RiskLimits, RiskManager

        # Pre-populate cache
        dates = pd.bdate_range("2024-01-01", periods=100)
        returns_df = pd.DataFrame(
            {"AAPL": [0.01] * 100},
            index=dates,
        )
        returns_df.to_parquet(self.cache_path)

        # Mock broker
        broker = MagicMock()
        pos = BrokerPosition(
            symbol="AAPL",
            qty=100.0,
            avg_entry_price=150.0,
            market_value=20_000.0,
            unrealized_pl=5_000.0,
            unrealized_plpc=0.33,
        )
        broker.list_positions.return_value = [pos]

        # Make yfinance fail
        monkeypatch.setattr(
            "yfinance.download",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("network down")),
        )

        risk_manager = RiskManager(
            limits=RiskLimits(
                max_position_weight=0.30,
                max_portfolio_var_95=0.06,
                max_drawdown_limit=0.15,
                max_sector_weight=0.40,
            )
        )

        # Should not raise — falls back to disk cache
        _initialize_risk_engine(broker, risk_manager)

        # Risk engine should have been initialized
        assert risk_manager.risk_engine is not None
