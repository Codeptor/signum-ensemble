"""Tests for conformal prediction position sizing."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from python.alpha.conformal import ConformalPositionSizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n=300, n_features=5, seed=42):
    """Synthetic regression data with clear signal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n) * 0.3
    return X, y


@pytest.fixture
def fitted_sizer():
    """Return a fitted + conformalized ConformalPositionSizer."""
    X, y = _make_data(300)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    sizer = ConformalPositionSizer(
        base_estimator=model, confidence_levels=[0.90, 0.95]
    )
    # Train on first 200, calibrate on last 100
    sizer.fit(X[:200], y[:200])
    sizer.conformalize(X[200:], y[200:])
    return sizer


# ---------------------------------------------------------------------------
# Tests: Fitting
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_succeeds(self):
        X, y = _make_data(200)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        sizer = ConformalPositionSizer(base_estimator=model)
        sizer.fit(X[:150], y[:150])
        assert sizer._fitted is True

    def test_fit_returns_self(self):
        X, y = _make_data(100)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        sizer = ConformalPositionSizer(base_estimator=model)
        result = sizer.fit(X, y)
        assert result is sizer

    def test_conformalize_succeeds(self):
        X, y = _make_data(200)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        sizer = ConformalPositionSizer(base_estimator=model)
        sizer.fit(X[:150], y[:150])
        sizer.conformalize(X[150:], y[150:])
        assert sizer._conformalized is True

    def test_conformalize_without_fit_raises(self):
        X, y = _make_data(100)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        sizer = ConformalPositionSizer(base_estimator=model)
        with pytest.raises(ValueError, match="not fitted"):
            sizer.conformalize(X, y)


# ---------------------------------------------------------------------------
# Tests: Prediction
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_shapes(self, fitted_sizer):
        X, _ = _make_data(50, seed=99)
        y_pred, y_pis = fitted_sizer.predict(X)
        assert y_pred.shape == (50,)
        # y_pis shape: (n_samples, 2, n_levels)
        assert y_pis.shape == (50, 2, 2)

    def test_predict_intervals_ordered(self, fitted_sizer):
        """Lower bound should be <= upper bound."""
        X, _ = _make_data(50, seed=99)
        _, y_pis = fitted_sizer.predict(X)
        for level_idx in range(2):
            lower = y_pis[:, 0, level_idx]
            upper = y_pis[:, 1, level_idx]
            assert np.all(lower <= upper)

    def test_wider_interval_at_higher_confidence(self, fitted_sizer):
        """95% CI should be wider than 90% CI."""
        X, _ = _make_data(50, seed=99)
        _, y_pis = fitted_sizer.predict(X)
        width_90 = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
        width_95 = (y_pis[:, 1, 1] - y_pis[:, 0, 1]).mean()
        assert width_95 >= width_90

    def test_predict_not_conformalized_raises(self):
        X, y = _make_data(100)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        sizer = ConformalPositionSizer(base_estimator=model)
        sizer.fit(X[:80], y[:80])
        with pytest.raises(ValueError, match="Not conformalized"):
            sizer.predict(X[-20:])


# ---------------------------------------------------------------------------
# Tests: Interval widths
# ---------------------------------------------------------------------------


class TestIntervalWidths:
    def test_widths_positive(self, fitted_sizer):
        X, _ = _make_data(50, seed=99)
        widths = fitted_sizer.interval_widths(X)
        assert np.all(widths >= 0)

    def test_widths_shape(self, fitted_sizer):
        X, _ = _make_data(50, seed=99)
        widths = fitted_sizer.interval_widths(X)
        assert widths.shape == (50,)


# ---------------------------------------------------------------------------
# Tests: Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_position_sizes_shape(self, fitted_sizer):
        X, _ = _make_data(50, seed=99)
        sizes = fitted_sizer.position_sizes(X, base_size=0.05)
        assert sizes.shape == (50,)

    def test_position_sizes_positive(self, fitted_sizer):
        X, _ = _make_data(50, seed=99)
        sizes = fitted_sizer.position_sizes(X, base_size=0.05, min_size=0.001)
        assert np.all(sizes > 0)

    def test_position_sizes_capped(self, fitted_sizer):
        X, _ = _make_data(50, seed=99)
        sizes = fitted_sizer.position_sizes(X, base_size=0.05, max_size=0.10)
        assert np.all(sizes <= 0.10)

    def test_narrow_intervals_get_larger_sizes(self, fitted_sizer):
        """High-confidence predictions should get larger position sizes."""
        X, _ = _make_data(200, seed=99)
        sizes = fitted_sizer.position_sizes(X, base_size=1.0)
        widths = fitted_sizer.interval_widths(X)

        # Rank correlation: narrower intervals -> larger sizes
        from scipy.stats import spearmanr

        corr, _ = spearmanr(widths, sizes)
        assert corr < 0  # Negative correlation: narrow -> large


# ---------------------------------------------------------------------------
# Tests: Coverage validation
# ---------------------------------------------------------------------------


class TestCoverageValidation:
    def test_coverage_structure(self, fitted_sizer):
        X, y = _make_data(100, seed=99)
        result = fitted_sizer.validate_coverage(X, y)
        assert "level_0.9" in result
        assert "level_0.95" in result
        assert "target_coverage" in result["level_0.9"]
        assert "empirical_coverage" in result["level_0.9"]
        assert "coverage_gap" in result["level_0.9"]
        assert "is_valid" in result["level_0.9"]

    def test_coverage_targets(self, fitted_sizer):
        X, y = _make_data(100, seed=99)
        result = fitted_sizer.validate_coverage(X, y)
        assert result["level_0.9"]["target_coverage"] == pytest.approx(0.90)
        assert result["level_0.95"]["target_coverage"] == pytest.approx(0.95)

    def test_coverage_reasonable(self, fitted_sizer):
        """Empirical coverage should be in a reasonable range."""
        X, y = _make_data(200, seed=99)
        result = fitted_sizer.validate_coverage(X, y)
        cov = result["level_0.9"]["empirical_coverage"]
        # Should be at least 50% and at most 100%
        assert 0.5 <= cov <= 1.0


# ---------------------------------------------------------------------------
# Tests: JSON export
# ---------------------------------------------------------------------------


class TestToJson:
    def test_json_not_fitted(self):
        model = RandomForestRegressor(n_estimators=5)
        sizer = ConformalPositionSizer(base_estimator=model)
        result = sizer.to_json()
        assert result["fitted"] is False
        assert result["conformalized"] is False

    def test_json_fitted(self, fitted_sizer):
        result = fitted_sizer.to_json()
        assert result["fitted"] is True
        assert result["conformalized"] is True
        assert result["confidence_levels"] == [0.90, 0.95]
