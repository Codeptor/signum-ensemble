"""Tests for SHAP-based explainability and alpha decay analysis."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from python.alpha.explainability import (
    alpha_decay_curve,
    compute_shap_importance,
    shap_stability_across_folds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shap_df(features, seed=42):
    """Create a fake SHAP importance DataFrame."""
    rng = np.random.RandomState(seed)
    importance = rng.rand(len(features))
    df = pd.DataFrame({"feature": features, "mean_abs_shap": importance})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Tests: compute_shap_importance
# ---------------------------------------------------------------------------


class TestComputeShapImportance:
    def test_returns_dataframe_with_expected_columns(self):
        """SHAP importance should return feature, mean_abs_shap, rank columns."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 4), columns=["a", "b", "c", "d"])
        y = X["a"] * 2 + rng.randn(100) * 0.1
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"feature", "mean_abs_shap", "rank"}
        assert len(result) == 4

    def test_features_are_ranked(self):
        """Features should be sorted by mean_abs_shap descending."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 3), columns=["x", "y", "z"])
        target = X["x"] * 5 + rng.randn(100) * 0.01
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, target)

        result = compute_shap_importance(model, X)
        assert list(result["rank"]) == [1, 2, 3]
        # Values should be descending
        vals = result["mean_abs_shap"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_subsampling_caps_rows(self):
        """When X exceeds max_samples, it should subsample."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(200, 2), columns=["a", "b"])
        y = rng.randn(200)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Should not raise with max_samples < len(X)
        result = compute_shap_importance(model, X, max_samples=50)
        assert len(result) == 2

    def test_numpy_array_input(self):
        """Should accept numpy arrays with explicit feature names."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = X[:, 0] * 2 + rng.randn(100) * 0.1
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(
            model, X, feature_names=["feat_a", "feat_b", "feat_c"]
        )
        assert list(result["feature"]) != []
        assert "feat_a" in result["feature"].values

    def test_numpy_array_auto_names(self):
        """Without feature_names, should auto-generate f0, f1, ..."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = rng.randn(50)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X)
        assert set(result["feature"]) == {"f0", "f1"}


# ---------------------------------------------------------------------------
# Tests: shap_stability_across_folds
# ---------------------------------------------------------------------------


class TestShapStability:
    def test_empty_input(self):
        """Empty list should return zero overlap."""
        result = shap_stability_across_folds([])
        assert result["top_k_overlap"] == 0.0
        assert result["always_top_k"] == []
        assert result["never_top_k"] == []

    def test_single_fold(self):
        """Single fold should have perfect overlap with itself."""
        features = ["a", "b", "c", "d", "e"]
        df = _make_shap_df(features, seed=1)
        result = shap_stability_across_folds([df], top_k=3)
        # Single fold: no pairs to compare, so overlap is 0.0
        assert result["top_k_overlap"] == 0.0
        assert result["n_folds"] == 1

    def test_identical_folds_perfect_overlap(self):
        """Identical folds should have Jaccard=1.0."""
        features = ["a", "b", "c", "d", "e"]
        df = _make_shap_df(features, seed=42)
        result = shap_stability_across_folds([df, df, df], top_k=3)
        assert result["top_k_overlap"] == pytest.approx(1.0)
        assert result["rank_correlation"] == pytest.approx(1.0)
        assert len(result["always_top_k"]) == 3

    def test_disjoint_folds_zero_overlap(self):
        """Completely different top-k sets should have Jaccard=0."""
        features = ["a", "b", "c", "d", "e", "f"]
        # Fold 1: top-3 = a, b, c
        df1 = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": [10, 9, 8, 1, 0.5, 0.1],
        })
        # Fold 2: top-3 = d, e, f
        df2 = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": [0.1, 0.5, 1, 10, 9, 8],
        })
        result = shap_stability_across_folds([df1, df2], top_k=3)
        assert result["top_k_overlap"] == pytest.approx(0.0)
        assert result["always_top_k"] == []

    def test_never_top_k_computed(self):
        """Features never in top-k of any fold should be identified."""
        features = ["a", "b", "c", "d", "e"]
        df1 = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": [10, 9, 1, 0.5, 0.1],
        })
        df2 = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": [10, 8, 2, 0.3, 0.1],
        })
        result = shap_stability_across_folds([df1, df2], top_k=2)
        # top-2 in both folds: {a, b} — so c, d, e are never top-2
        assert set(result["never_top_k"]) == {"c", "d", "e"}

    def test_n_folds_tracked(self):
        """Should report the correct number of folds."""
        features = ["x", "y", "z"]
        dfs = [_make_shap_df(features, seed=i) for i in range(5)]
        result = shap_stability_across_folds(dfs, top_k=2)
        assert result["n_folds"] == 5


# ---------------------------------------------------------------------------
# Tests: alpha_decay_curve
# ---------------------------------------------------------------------------


class TestAlphaDecayCurve:
    @pytest.fixture
    def synthetic_signal(self):
        """Create a synthetic signal + returns for alpha decay testing."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2023-01-01", periods=100)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

        # Multi-index predictions
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        predictions = pd.Series(rng.randn(len(idx)), index=idx)

        # Returns DataFrame (columns = tickers)
        returns_df = pd.DataFrame(
            rng.randn(100, 5) * 0.02,
            index=dates,
            columns=tickers,
        )
        # Make returns cumulative prices-like for pct_change
        returns_df = (1 + returns_df).cumprod() * 100

        return predictions, returns_df

    def test_returns_dataframe(self, synthetic_signal):
        """Alpha decay should return a DataFrame."""
        predictions, returns_df = synthetic_signal
        result = alpha_decay_curve(predictions, returns_df, horizons=[1, 3, 5])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, synthetic_signal):
        """Should have horizon, ic, ic_std, ir, n_dates columns."""
        predictions, returns_df = synthetic_signal
        result = alpha_decay_curve(predictions, returns_df, horizons=[1, 5])
        expected_cols = {"horizon", "ic", "ic_std", "ir", "n_dates"}
        assert set(result.columns) == expected_cols

    def test_horizons_preserved(self, synthetic_signal):
        """Output should contain rows for requested horizons."""
        predictions, returns_df = synthetic_signal
        horizons = [1, 3, 5, 10]
        result = alpha_decay_curve(predictions, returns_df, horizons=horizons)
        # May have fewer if some horizons have insufficient data
        assert all(h in horizons for h in result["horizon"].values)

    def test_default_horizons(self, synthetic_signal):
        """Without explicit horizons, should use default [1,2,3,5,10,15,20]."""
        predictions, returns_df = synthetic_signal
        result = alpha_decay_curve(predictions, returns_df)
        # With 100 dates, short horizons should work
        assert len(result) > 0
        assert 1 in result["horizon"].values

    def test_ic_values_bounded(self, synthetic_signal):
        """IC (Spearman) should be in [-1, 1]."""
        predictions, returns_df = synthetic_signal
        result = alpha_decay_curve(predictions, returns_df, horizons=[1, 3])
        for _, row in result.iterrows():
            assert -1.0 <= row["ic"] <= 1.0

    def test_empty_on_insufficient_data(self):
        """With too few dates, should return empty DataFrame."""
        dates = pd.bdate_range("2023-01-01", periods=3)
        tickers = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, tickers])
        predictions = pd.Series(np.random.randn(6), index=idx)
        returns_df = pd.DataFrame(
            np.random.randn(3, 2), index=dates, columns=tickers
        )
        result = alpha_decay_curve(predictions, returns_df, horizons=[20])
        assert len(result) == 0
