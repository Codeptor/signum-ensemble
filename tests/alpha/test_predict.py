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
