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
