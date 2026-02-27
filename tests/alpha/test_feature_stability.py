"""Tests for feature importance stability analysis."""

import numpy as np
import pandas as pd
import pytest

from python.alpha.feature_stability import (
    FeatureStabilityAnalyzer,
    StabilityReport,
)


def _make_importances(n_features=10, n_folds=5, seed=42, noise=0.1):
    """Create synthetic feature importances with controllable noise."""
    rng = np.random.default_rng(seed)
    # True importances (decreasing)
    true_imp = np.linspace(1.0, 0.1, n_features)
    features = [f"feat_{i}" for i in range(n_features)]
    folds = {}
    for f in range(n_folds):
        noisy = true_imp + rng.normal(0, noise, n_features)
        folds[f"fold_{f}"] = pd.Series(np.maximum(noisy, 0), index=features)
    return folds


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasicAnalysis:
    def test_returns_report(self):
        imps = _make_importances()
        analyzer = FeatureStabilityAnalyzer()
        report = analyzer.analyze(imps)
        assert isinstance(report, StabilityReport)

    def test_feature_stats_shape(self):
        imps = _make_importances(n_features=10, n_folds=5)
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert len(report.feature_stats) == 10
        assert "stability_score" in report.feature_stats.columns

    def test_stability_matrix_shape(self):
        imps = _make_importances(n_folds=5)
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.stability_matrix.shape == (5, 5)

    def test_stability_matrix_symmetric(self):
        imps = _make_importances()
        report = FeatureStabilityAnalyzer().analyze(imps)
        mat = report.stability_matrix.values
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_stability_matrix_diagonal_ones(self):
        imps = _make_importances()
        report = FeatureStabilityAnalyzer().analyze(imps)
        np.testing.assert_array_almost_equal(
            np.diag(report.stability_matrix.values), 1.0
        )

    def test_accepts_list(self):
        """Should accept list of Series instead of dict."""
        features = [f"feat_{i}" for i in range(5)]
        imps = [
            pd.Series(np.random.rand(5), index=features)
            for _ in range(3)
        ]
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.n_folds == 3


# ---------------------------------------------------------------------------
# Stability detection
# ---------------------------------------------------------------------------


class TestStabilityDetection:
    def test_stable_features_detected(self):
        """Low noise → features should be stable."""
        imps = _make_importances(noise=0.01)
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.n_stable > 0

    def test_unstable_features_with_noise(self):
        """High noise → some features should be unstable."""
        imps = _make_importances(noise=0.5)
        report = FeatureStabilityAnalyzer(stability_threshold=0.8).analyze(imps)
        assert report.n_unstable > 0

    def test_completely_random_importances(self):
        """Random importances should be mostly unstable."""
        rng = np.random.default_rng(42)
        features = [f"feat_{i}" for i in range(10)]
        imps = {
            f"fold_{f}": pd.Series(rng.random(10), index=features)
            for f in range(5)
        }
        report = FeatureStabilityAnalyzer(stability_threshold=0.7).analyze(imps)
        assert report.mean_stability < 0.5

    def test_identical_importances_all_stable(self):
        """Identical importances across folds → all stable."""
        features = [f"feat_{i}" for i in range(5)]
        base = pd.Series([0.5, 0.3, 0.1, 0.05, 0.05], index=features)
        imps = {f"fold_{f}": base.copy() for f in range(4)}
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.n_stable == 5
        assert report.mean_stability == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Properties and summaries
# ---------------------------------------------------------------------------


class TestProperties:
    def test_stable_features_list(self):
        imps = _make_importances(noise=0.01)
        report = FeatureStabilityAnalyzer().analyze(imps)
        stable = report.stable_features
        assert isinstance(stable, list)
        assert len(stable) == report.n_stable

    def test_unstable_features_list(self):
        imps = _make_importances(noise=0.5)
        report = FeatureStabilityAnalyzer().analyze(imps)
        unstable = report.unstable_features
        assert isinstance(unstable, list)
        assert len(unstable) == report.n_unstable

    def test_n_features_consistent(self):
        imps = _make_importances(n_features=8)
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.n_features == 8
        assert report.n_stable + report.n_unstable == 8

    def test_summary_string(self):
        imps = _make_importances()
        report = FeatureStabilityAnalyzer().analyze(imps)
        s = report.summary()
        assert "stable" in s.lower()
        assert "folds" in s.lower()

    def test_mean_stability_bounded(self):
        imps = _make_importances()
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert -1.0 <= report.mean_stability <= 1.0


# ---------------------------------------------------------------------------
# Select stable features
# ---------------------------------------------------------------------------


class TestSelectStable:
    def test_returns_list(self):
        imps = _make_importances(noise=0.01)
        analyzer = FeatureStabilityAnalyzer()
        selected = analyzer.select_stable_features(imps)
        assert isinstance(selected, list)
        assert len(selected) > 0

    def test_with_rank_filter(self):
        imps = _make_importances(noise=0.01)
        analyzer = FeatureStabilityAnalyzer()
        selected = analyzer.select_stable_features(imps, min_importance_rank=5)
        # Should only include top-5 ranked features
        assert len(selected) <= 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_fold_raises(self):
        features = [f"feat_{i}" for i in range(5)]
        imps = {"fold_0": pd.Series(np.random.rand(5), index=features)}
        with pytest.raises(ValueError, match="at least 2 folds"):
            FeatureStabilityAnalyzer().analyze(imps)

    def test_two_folds_works(self):
        features = [f"feat_{i}" for i in range(5)]
        imps = {
            "fold_0": pd.Series([0.5, 0.3, 0.1, 0.05, 0.05], index=features),
            "fold_1": pd.Series([0.45, 0.35, 0.1, 0.05, 0.05], index=features),
        }
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.n_folds == 2

    def test_missing_features_filled(self):
        """Features present in some folds but not others should be handled."""
        imps = {
            "fold_0": pd.Series({"a": 0.5, "b": 0.3, "c": 0.2}),
            "fold_1": pd.Series({"a": 0.4, "b": 0.4, "d": 0.2}),
        }
        report = FeatureStabilityAnalyzer().analyze(imps)
        assert report.n_features == 4  # a, b, c, d
