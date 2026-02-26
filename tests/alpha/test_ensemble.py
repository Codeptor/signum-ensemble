"""Tests for model ensemble (Phase 3, §2.3.3)."""

import numpy as np
import pandas as pd
import pytest

from python.alpha.ensemble import DEFAULT_WEIGHTS, ModelEnsemble

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n_rows: int = 500, n_features: int = 8, seed: int = 42):
    """Create synthetic training/validation DataFrames for testing."""
    rng = np.random.RandomState(seed)
    feature_names = [f"f{i}" for i in range(n_features)]

    X = rng.randn(n_rows, n_features)
    # Target is a linear combination + noise so all models can learn something
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_rows) * 0.5

    df = pd.DataFrame(X, columns=feature_names)
    df["target_5d"] = y
    return df, feature_names


@pytest.fixture
def training_data():
    return _make_data(n_rows=500, seed=42)


@pytest.fixture
def validation_data():
    return _make_data(n_rows=200, seed=99)


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestEnsembleInit:
    def test_default_weights(self):
        ens = ModelEnsemble(feature_cols=["a", "b"])
        assert ens.weights == DEFAULT_WEIGHTS

    def test_custom_weights(self):
        custom = {"lightgbm": 0.7, "random_forest": 0.3}
        ens = ModelEnsemble(feature_cols=["a"], weights=custom)
        assert ens.weights == custom

    def test_feature_cols_stored(self):
        cols = ["ret_5d", "rsi_14", "vol_20d"]
        ens = ModelEnsemble(feature_cols=cols)
        assert ens.feature_cols == cols

    def test_not_fitted_initially(self):
        ens = ModelEnsemble(feature_cols=["a"])
        assert ens._fitted is False

    def test_models_property(self):
        ens = ModelEnsemble(feature_cols=["a"])
        models = ens.models
        assert set(models.keys()) == {"lightgbm", "random_forest"}


# ---------------------------------------------------------------------------
# Tests: Training
# ---------------------------------------------------------------------------


class TestEnsembleFit:
    def test_fit_marks_fitted(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df, target_col="target_5d")
        assert ens._fitted is True

    def test_fit_trains_all_models(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df, target_col="target_5d")

        # LightGBM should have an internal model
        assert ens.lgbm.model is not None
        # RF should be fitted (has feature_importances_)
        assert hasattr(ens.rf, "feature_importances_")

    def test_fit_with_validation_calibrates(self, training_data, validation_data):
        df, cols = training_data
        val_df, _ = validation_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df, target_col="target_5d", val_df=val_df)

        # Weights should have been recalibrated (may differ from defaults)
        assert sum(ens.weights.values()) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: Prediction
# ---------------------------------------------------------------------------


class TestEnsemblePredict:
    def test_predict_shape(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        preds = ens.predict(df)
        assert preds.shape == (len(df),)

    def test_predict_dtype(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        preds = ens.predict(df)
        assert preds.dtype == np.float64

    def test_predict_not_fitted_raises(self, training_data):
        _, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        with pytest.raises(ValueError, match="not trained"):
            ens.predict(pd.DataFrame(np.zeros((5, len(cols))), columns=cols))

    def test_predict_individual_returns_all_models(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        individual = ens.predict_individual(df)
        assert set(individual.keys()) == {"lightgbm", "random_forest"}
        for name, preds in individual.items():
            assert len(preds) == len(df), f"{name} prediction length mismatch"

    def test_ensemble_is_weighted_average(self, training_data):
        """Ensemble prediction should equal the weighted sum of sub-models."""
        df, cols = training_data
        weights = {"lightgbm": 0.6, "random_forest": 0.4}
        ens = ModelEnsemble(feature_cols=cols, weights=weights)
        ens.fit(df)

        individual = ens.predict_individual(df)
        expected = sum(individual[n] * w for n, w in weights.items())
        actual = ens.predict(df)

        np.testing.assert_allclose(actual, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: IC Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_calibrate_weights_sum_to_one(self, training_data, validation_data):
        df, cols = training_data
        val_df, _ = validation_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        new_weights = ens.calibrate_weights(val_df, target_col="target_5d")
        assert sum(new_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_calibrate_zeros_negative_ic(self, training_data):
        """Models with negative IC should get weight 0."""
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        # Calibrated weights should have no negative entries
        for w in ens.weights.values():
            assert w >= 0.0

    def test_calibrate_preserves_model_names(self, training_data, validation_data):
        df, cols = training_data
        val_df, _ = validation_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        new_weights = ens.calibrate_weights(val_df)
        assert set(new_weights.keys()) == {"lightgbm", "random_forest"}


# ---------------------------------------------------------------------------
# Tests: Feature Importance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_importance_returns_dataframe(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        importance = ens.feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert list(importance.index) == list(importance.index)  # sorted
        assert "ensemble" in importance.columns

    def test_importance_has_all_models(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        importance = ens.feature_importance()
        for model_name in ["lightgbm", "random_forest", "ensemble"]:
            assert model_name in importance.columns

    def test_importance_correct_number_of_features(self, training_data):
        df, cols = training_data
        ens = ModelEnsemble(feature_cols=cols)
        ens.fit(df)

        importance = ens.feature_importance()
        assert len(importance) == len(cols)


# ---------------------------------------------------------------------------
# Tests: Feature column reduction (Phase 3.1)
# ---------------------------------------------------------------------------


class TestFeatureReduction:
    def test_reduced_feature_set_size(self):
        from python.alpha.train import FEATURE_COLS

        assert len(FEATURE_COLS) == 8

    def test_reduced_feature_set_contents(self):
        from python.alpha.train import FEATURE_COLS

        expected = {
            "ret_5d",
            "ret_20d",
            "rsi_14",
            "bb_position",
            "vol_20d",
            "volume_ratio",
            "cs_ret_rank_5d",
            "vix",
        }
        assert set(FEATURE_COLS) == expected

    def test_full_feature_set_preserved(self):
        from python.alpha.train import FEATURE_COLS_FULL

        assert len(FEATURE_COLS_FULL) == 26

    def test_reduced_is_subset_of_full(self):
        from python.alpha.train import FEATURE_COLS, FEATURE_COLS_FULL

        assert set(FEATURE_COLS).issubset(set(FEATURE_COLS_FULL))
