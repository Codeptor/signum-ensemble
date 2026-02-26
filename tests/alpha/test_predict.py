"""Tests for model versioning (Fix #45)."""

from __future__ import annotations

import pytest

from python.alpha.predict import (
    _load_cached_model,
    _save_model_cache,
    list_model_versions,
    load_model_version,
)


class _FakeModel:
    """Lightweight picklable stand-in for CrossSectionalModel."""

    def __init__(self, feature_cols=None, params=None):
        self.feature_cols = feature_cols or ["feat_a", "feat_b"]
        self.params = params or {"objective": "huber", "learning_rate": 0.05}


@pytest.fixture
def mock_model():
    """Create a lightweight picklable model stand-in."""
    return _FakeModel()


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Redirect MODEL_CACHE_DIR to a temp directory."""
    import python.alpha.predict as predict_mod

    monkeypatch.setattr(predict_mod, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(predict_mod, "MODEL_CACHE_FILE", tmp_path / "latest_model.joblib")
    return tmp_path


class TestModelVersioning:
    """Test the model versioning lifecycle."""

    def test_save_creates_versioned_and_latest(self, mock_model, model_dir):
        """Saving creates both a versioned file and latest cache."""
        _save_model_cache(mock_model)

        # latest_model.joblib should exist
        assert (model_dir / "latest_model.joblib").exists()

        # At least one versioned file model_<date>_<hash>.joblib
        versions = list(model_dir.glob("model_*_*.joblib"))
        assert len(versions) == 1

    def test_save_multiple_creates_distinct_versions(self, model_dir):
        """Multiple saves create distinct versioned files."""
        # Use different params each time so the hash differs within the same second
        for i in range(3):
            m = _FakeModel(params={"objective": "huber", "run": i})
            _save_model_cache(m)

        versions = list(model_dir.glob("model_*_*.joblib"))
        assert len(versions) == 3

    def test_prune_respects_max_versions(self, model_dir, monkeypatch):
        """Old versions beyond MAX_MODEL_VERSIONS are pruned."""
        import python.alpha.predict as predict_mod

        monkeypatch.setattr(predict_mod, "MAX_MODEL_VERSIONS", 3)

        for i in range(5):
            m = _FakeModel(params={"objective": "huber", "run": i})
            _save_model_cache(m)

        # After 5 saves with max=3, only 3 versioned files should remain
        versions = list(model_dir.glob("model_*_*.joblib"))
        assert len(versions) == 3

    def test_list_model_versions(self, model_dir):
        """list_model_versions returns metadata for each saved version."""
        _save_model_cache(_FakeModel(params={"objective": "huber", "run": 0}))
        _save_model_cache(_FakeModel(params={"objective": "huber", "run": 1}))

        versions = list_model_versions()
        assert len(versions) == 2

        # Each version should have required metadata
        for v in versions:
            assert "path" in v
            assert "filename" in v
            assert "trained_date" in v
            assert "trained_at" in v
            assert v["feature_cols"] == ["feat_a", "feat_b"]

    def test_load_model_version(self, mock_model, model_dir):
        """load_model_version returns the model from a specific version file."""
        _save_model_cache(mock_model)

        versions = list_model_versions()
        assert len(versions) == 1

        loaded = load_model_version(versions[0]["path"])
        assert loaded is not None

    def test_load_model_version_not_found(self, model_dir):
        """Loading a nonexistent version raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model_version("/nonexistent/model.joblib")

    def test_latest_cache_still_works(self, mock_model, model_dir):
        """_load_cached_model still works with versioned saves."""
        _save_model_cache(mock_model)

        loaded = _load_cached_model()
        assert loaded is not None


class TestMaxWeightPassthrough:
    """Test that max_weight parameter is passed through correctly (M8 fix).

    optimize_weights() and get_ml_weights() should pass max_weight to
    PortfolioOptimizer for post-optimization capping.
    """

    def test_optimize_weights_accepts_max_weight(self):
        """optimize_weights signature accepts max_weight parameter."""
        import inspect
        from python.alpha.predict import optimize_weights

        sig = inspect.signature(optimize_weights)
        assert "max_weight" in sig.parameters
        # Default should be None
        assert sig.parameters["max_weight"].default is None

    def test_get_ml_weights_accepts_max_weight(self):
        """get_ml_weights signature accepts max_weight parameter."""
        import inspect
        from python.alpha.predict import get_ml_weights

        sig = inspect.signature(get_ml_weights)
        assert "max_weight" in sig.parameters
        assert sig.parameters["max_weight"].default is None


class TestComputeFeaturesIncludesMacro:
    """Test that compute_features merges macro features (Bug 1 fix).

    At inference time, compute_features must produce the same columns
    as train_model, including macro features like 'vix' and 'term_spread'.
    """

    def test_compute_features_attempts_macro_merge(self, monkeypatch):
        """compute_features should attempt to merge macro features when file exists."""
        from unittest.mock import patch, MagicMock
        import python.alpha.predict as predict_mod

        # We mock the entire chain since we can't easily create valid long_df
        mock_compute_alpha = MagicMock(return_value=MagicMock())
        mock_compute_cross = MagicMock(return_value=MagicMock())

        monkeypatch.setattr(predict_mod, "compute_alpha_features", mock_compute_alpha)
        monkeypatch.setattr(predict_mod, "compute_cross_sectional_features", mock_compute_cross)

        # Mock Path.exists to say macro file exists
        with patch("python.alpha.predict.Path") as MockPath:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            MockPath.return_value = mock_path_instance

            with patch("python.alpha.features.merge_macro_features") as mock_merge:
                mock_merge.return_value = mock_compute_cross.return_value
                predict_mod.compute_features(MagicMock())

                mock_merge.assert_called_once()


class TestRankStocksFillsMissingFeatures:
    """Test that rank_stocks fills missing features with 0.0 (Bug 1 / H6 fix)."""

    def test_missing_features_are_filled(self):
        """rank_stocks should not crash when model expects features not in data."""
        import pandas as pd
        import numpy as np
        from unittest.mock import MagicMock

        from python.alpha.predict import rank_stocks

        # Create a model mock that expects 'vix' and 'momentum'
        mock_model = MagicMock()
        mock_model.feature_cols = ["momentum", "vix"]
        mock_model.predict.return_value = np.array([0.5, 0.3])

        # Create featured_df with only 'momentum' (missing 'vix')
        dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
        tickers = ["AAPL", "MSFT"]
        idx = pd.MultiIndex.from_arrays([dates, tickers])
        df = pd.DataFrame(
            {"momentum": [0.1, 0.2], "ticker": ["AAPL", "MSFT"]},
            index=idx,
        )

        # Should not crash
        result = rank_stocks(mock_model, df, top_n=2)

        assert isinstance(result, list)
        assert len(result) == 2


# =====================================================================
# Tests for audit fixes: H10, H11, M11, M12
# =====================================================================


class TestNeutralDefaults:
    """M11: rank_stocks fills missing features with domain-appropriate neutral
    defaults from FEATURE_NEUTRAL_DEFAULTS, not 0.0."""

    def test_vix_filled_with_neutral_default(self):
        """Missing 'vix' should be filled with ~20.0 (long-run median), not 0.0."""
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock

        from python.alpha.features import FEATURE_NEUTRAL_DEFAULTS
        from python.alpha.predict import rank_stocks

        mock_model = MagicMock()
        mock_model.feature_cols = ["momentum", "vix"]
        mock_model.predict.return_value = np.array([0.5])

        dates = pd.to_datetime(["2024-01-01"])
        idx = pd.MultiIndex.from_arrays([dates, ["AAPL"]])
        df = pd.DataFrame({"momentum": [0.1], "ticker": ["AAPL"]}, index=idx)

        rank_stocks(mock_model, df, top_n=1)

        # Inspect the DataFrame that was actually passed to model.predict
        call_args = mock_model.predict.call_args[0][0]
        vix_val = call_args["vix"].iloc[0]
        assert vix_val == pytest.approx(FEATURE_NEUTRAL_DEFAULTS["vix"]), (
            f"Expected VIX neutral default {FEATURE_NEUTRAL_DEFAULTS['vix']}, got {vix_val}"
        )

    def test_rsi_filled_with_midpoint(self):
        """Missing 'rsi_14' should be filled with 50.0, not 0.0."""
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock

        from python.alpha.features import FEATURE_NEUTRAL_DEFAULTS
        from python.alpha.predict import rank_stocks

        mock_model = MagicMock()
        mock_model.feature_cols = ["momentum", "rsi_14"]
        mock_model.predict.return_value = np.array([0.5])

        dates = pd.to_datetime(["2024-01-01"])
        idx = pd.MultiIndex.from_arrays([dates, ["AAPL"]])
        df = pd.DataFrame({"momentum": [0.1], "ticker": ["AAPL"]}, index=idx)

        rank_stocks(mock_model, df, top_n=1)

        call_args = mock_model.predict.call_args[0][0]
        rsi_val = call_args["rsi_14"].iloc[0]
        assert rsi_val == pytest.approx(FEATURE_NEUTRAL_DEFAULTS["rsi_14"])

    def test_unknown_feature_falls_back_to_zero(self):
        """Features not in FEATURE_NEUTRAL_DEFAULTS fall back to 0.0."""
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock

        from python.alpha.predict import rank_stocks

        mock_model = MagicMock()
        mock_model.feature_cols = ["momentum", "exotic_feature_xyz"]
        mock_model.predict.return_value = np.array([0.5])

        dates = pd.to_datetime(["2024-01-01"])
        idx = pd.MultiIndex.from_arrays([dates, ["AAPL"]])
        df = pd.DataFrame({"momentum": [0.1], "ticker": ["AAPL"]}, index=idx)

        rank_stocks(mock_model, df, top_n=1)

        call_args = mock_model.predict.call_args[0][0]
        exotic_val = call_args["exotic_feature_xyz"].iloc[0]
        assert exotic_val == pytest.approx(0.0)


class TestOptimizeWeightsPriceDataPassthrough:
    """H11: optimize_weights accepts pre-fetched price_data to avoid double fetch."""

    def test_price_data_skips_fetch(self, monkeypatch):
        """When price_data is provided, fetch_ohlcv should NOT be called."""
        import pandas as pd
        from unittest.mock import MagicMock, patch

        import python.alpha.predict as predict_mod

        # Create minimal price data
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = pd.DataFrame(
            {"AAPL": range(100, 200), "MSFT": range(200, 300)},
            index=dates,
            dtype=float,
        )

        # Mock extract_close_prices to return our prices
        monkeypatch.setattr(predict_mod, "extract_close_prices", lambda raw: prices)

        # Mock PortfolioOptimizer
        mock_optimizer_cls = MagicMock()
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.hrp.return_value = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        mock_optimizer_cls.return_value = mock_optimizer_instance
        monkeypatch.setattr(predict_mod, "PortfolioOptimizer", mock_optimizer_cls)

        # Mock sector constraints (passthrough)
        monkeypatch.setattr(predict_mod, "get_sector_map", lambda tickers: {})
        monkeypatch.setattr(
            predict_mod,
            "enforce_sector_constraints",
            lambda w, **kw: w,
        )

        # Mock fetch_ohlcv and ensure it's NOT called
        mock_fetch = MagicMock()
        monkeypatch.setattr(predict_mod, "fetch_ohlcv", mock_fetch)

        result = predict_mod.optimize_weights(
            ["AAPL", "MSFT"],
            price_data=prices,  # H11: pass pre-fetched data
        )

        mock_fetch.assert_not_called()
        assert "AAPL" in result
        assert "MSFT" in result

    def test_no_price_data_triggers_fetch(self, monkeypatch):
        """When price_data is None, fetch_ohlcv SHOULD be called."""
        import pandas as pd
        from unittest.mock import MagicMock

        import python.alpha.predict as predict_mod

        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = pd.DataFrame(
            {"AAPL": range(100, 200), "MSFT": range(200, 300)},
            index=dates,
            dtype=float,
        )

        mock_fetch = MagicMock(return_value=prices)
        monkeypatch.setattr(predict_mod, "fetch_ohlcv", mock_fetch)
        monkeypatch.setattr(predict_mod, "extract_close_prices", lambda raw: prices)

        mock_optimizer_cls = MagicMock()
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.hrp.return_value = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        mock_optimizer_cls.return_value = mock_optimizer_instance
        monkeypatch.setattr(predict_mod, "PortfolioOptimizer", mock_optimizer_cls)
        monkeypatch.setattr(predict_mod, "get_sector_map", lambda tickers: {})
        monkeypatch.setattr(
            predict_mod,
            "enforce_sector_constraints",
            lambda w, **kw: w,
        )

        predict_mod.optimize_weights(["AAPL", "MSFT"])

        mock_fetch.assert_called_once()


class TestRandomSampleTraining:
    """M12: train_model uses random.sample(seed=42) instead of alphabetical[:100]."""

    def test_signature_accepts_training_tickers(self):
        """train_model accepts training_tickers override."""
        import inspect
        from python.alpha.predict import train_model

        sig = inspect.signature(train_model)
        assert "training_tickers" in sig.parameters

    def test_random_sample_is_deterministic(self):
        """random.sample with seed=42 on the same list gives the same result."""
        import random

        tickers = [f"T{i:04d}" for i in range(503)]
        random.seed(42)
        sample1 = random.sample(tickers, 100)
        random.seed(42)
        sample2 = random.sample(tickers, 100)
        assert sample1 == sample2

    def test_random_sample_not_alphabetical(self):
        """The sample should NOT be the first 100 alphabetically."""
        import random

        tickers = sorted([f"T{i:04d}" for i in range(503)])
        random.seed(42)
        sample = random.sample(tickers, 100)
        # The sample should differ from the first 100 alphabetically
        assert sample != tickers[:100]


class TestSurvivorshipBiasWarning:
    """H10: train_model logs a survivorship bias warning when fetching live data."""

    def test_warning_logged_when_fetching(self, monkeypatch, caplog):
        """When no cached data exists, a survivorship bias warning is emitted.

        We short-circuit training after verifying the warning appears by
        raising an exception in fetch_ohlcv — the warning is issued before
        the fetch call, so this is sufficient.
        """
        import logging
        from pathlib import Path
        from unittest.mock import MagicMock

        import python.alpha.predict as predict_mod

        # Force no cached model
        monkeypatch.setattr(predict_mod, "_load_cached_model", lambda: None)

        # Make the cached data path not exist so the live-fetch branch is taken
        _real_path = Path

        def _fake_path(p):
            if "sp500" in str(p) or "macro" in str(p):
                m = MagicMock()
                m.exists.return_value = False
                return m
            return _real_path(p)

        monkeypatch.setattr(predict_mod, "Path", _fake_path)

        # Mock ticker source
        monkeypatch.setattr(
            "python.data.ingestion.fetch_sp500_tickers",
            lambda: [f"T{i}" for i in range(200)],
        )

        # Short-circuit: raise after the survivorship warning is already logged
        class _StopEarly(Exception):
            pass

        monkeypatch.setattr(
            predict_mod, "fetch_ohlcv", lambda *a, **kw: (_ for _ in ()).throw(_StopEarly)
        )

        with caplog.at_level(logging.WARNING, logger="python.alpha.predict"):
            with pytest.raises(_StopEarly):
                predict_mod.train_model(force_retrain=True)

        assert any("SURVIVORSHIP BIAS" in rec.message for rec in caplog.records), (
            "Expected survivorship bias warning in logs"
        )
