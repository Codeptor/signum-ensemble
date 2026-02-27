"""Tests for CPCV and Probability of Backtest Overfitting."""

import numpy as np
import pandas as pd
import pytest
from itertools import combinations
from math import comb

from python.backtest.cpcv import (
    cpcv_split,
    cpcv_evaluate,
    probability_of_backtest_overfitting,
    _compute_metric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """600-row dataset for basic CPCV tests (row-level splitting)."""
    np.random.seed(42)
    n = 600
    X = np.random.randn(n, 5)
    y = X @ np.random.randn(5) + np.random.randn(n) * 0.3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target_5d"] = y
    return df, [f"f{i}" for i in range(5)]


@pytest.fixture
def dated_data():
    """Panel data with DatetimeIndex (2 tickers, 300 dates = 600 rows)."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=300)
    tickers = ["A", "B"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t})
    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"])
    n = len(df)
    X = np.random.randn(n, 5)
    for i in range(5):
        df[f"f{i}"] = X[:, i]
    df["target_5d"] = X @ np.random.randn(5) + np.random.randn(n) * 0.3
    return df, [f"f{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Tests: cpcv_split
# ---------------------------------------------------------------------------


class TestCpcvSplit:
    def test_correct_number_of_paths_row_level(self):
        """C(6,2) = 15 paths for 6 blocks, 2 test blocks."""
        paths = list(cpcv_split(600, n_blocks=6, n_test_blocks=2))
        assert len(paths) == comb(6, 2)

    def test_correct_number_of_paths_different_params(self):
        """C(8,3) = 56 paths."""
        paths = list(cpcv_split(800, n_blocks=8, n_test_blocks=3))
        assert len(paths) == comb(8, 3)

    def test_no_overlap_between_train_test(self):
        """Train and test indices must not overlap."""
        for train_idx, test_idx, _ in cpcv_split(600, n_blocks=6, n_test_blocks=2):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_test_blocks_returned(self):
        """Each path should report which blocks are held out."""
        for _, _, test_blocks in cpcv_split(600, n_blocks=6, n_test_blocks=2):
            assert len(test_blocks) == 2
            assert all(0 <= b < 6 for b in test_blocks)

    def test_all_block_combinations_covered(self):
        """Every C(n,k) combination should appear exactly once."""
        seen = set()
        for _, _, test_blocks in cpcv_split(600, n_blocks=6, n_test_blocks=2):
            seen.add(test_blocks)
        expected = set(combinations(range(6), 2))
        assert seen == expected

    def test_purge_creates_gap(self):
        """Purge should remove indices just before test blocks."""
        paths = list(cpcv_split(600, n_blocks=6, n_test_blocks=1, purge_pct=0.10))
        for train_idx, test_idx, _ in paths:
            if test_idx:
                test_min = min(test_idx)
                # Some rows before test_min should be excluded from train
                if test_min > 0:
                    near_before = set(range(max(0, test_min - 5), test_min))
                    assert len(near_before & set(train_idx)) < len(near_before)

    def test_date_aware_split(self, dated_data):
        """Date-aware CPCV should produce valid paths."""
        df, _ = dated_data
        dates_arr = df.index.get_level_values(0).values
        paths = list(
            cpcv_split(len(df), n_blocks=6, n_test_blocks=2, dates=dates_arr)
        )
        assert len(paths) == comb(6, 2)
        for train_idx, test_idx, _ in paths:
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_raises_on_too_few_blocks(self):
        with pytest.raises(ValueError, match="n_blocks must be >= 3"):
            list(cpcv_split(100, n_blocks=2, n_test_blocks=1))

    def test_raises_on_invalid_test_blocks(self):
        with pytest.raises(ValueError, match="n_test_blocks"):
            list(cpcv_split(100, n_blocks=6, n_test_blocks=6))

    def test_train_plus_test_covers_most_data(self):
        """Union of train + test should cover most rows (minus purge/embargo)."""
        for train_idx, test_idx, _ in cpcv_split(
            600, n_blocks=6, n_test_blocks=2, purge_pct=0.01, embargo_pct=0.01
        ):
            covered = len(set(train_idx) | set(test_idx))
            assert covered > 500  # At least 83% of 600


# ---------------------------------------------------------------------------
# Tests: _compute_metric
# ---------------------------------------------------------------------------


class TestComputeMetric:
    def test_ic_perfect_correlation(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _compute_metric(preds, targets, "ic") == pytest.approx(1.0)

    def test_ic_negative_correlation(self):
        preds = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _compute_metric(preds, targets, "ic") == pytest.approx(-1.0)

    def test_ic_with_nans(self):
        preds = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_metric(preds, targets, "ic")
        assert -1.0 <= result <= 1.0

    def test_sharpe_metric(self):
        preds = np.zeros(100)
        targets = np.random.randn(100) * 0.01 + 0.001
        result = _compute_metric(preds, targets, "sharpe")
        assert isinstance(result, float)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _compute_metric(np.arange(10.0), np.arange(10.0), "bad")

    def test_too_few_valid_returns_zero(self):
        preds = np.array([np.nan, np.nan, np.nan, 1.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert _compute_metric(preds, targets, "ic") == 0.0


# ---------------------------------------------------------------------------
# Tests: probability_of_backtest_overfitting
# ---------------------------------------------------------------------------


class TestPBO:
    def test_no_overfitting_returns_low_pbo(self):
        """When IS and OOS are correlated, PBO should be low."""
        np.random.seed(42)
        n = 50
        base = np.random.randn(n)
        is_metrics = list(base + np.random.randn(n) * 0.1)
        oos_metrics = list(base + np.random.randn(n) * 0.1)
        result = probability_of_backtest_overfitting(is_metrics, oos_metrics)
        assert result["pbo"] < 0.5

    def test_overfitting_returns_high_pbo(self):
        """When IS and OOS are anti-correlated, PBO should be high."""
        np.random.seed(42)
        n = 50
        is_metrics = list(np.linspace(0, 1, n))
        oos_metrics = list(np.linspace(1, 0, n))
        result = probability_of_backtest_overfitting(is_metrics, oos_metrics)
        assert result["pbo"] > 0.3

    def test_pbo_bounded(self):
        """PBO should be between 0 and 1."""
        is_m = [0.1, 0.2, 0.3, 0.4, 0.5]
        oos_m = [0.05, 0.15, 0.25, 0.35, 0.45]
        result = probability_of_backtest_overfitting(is_m, oos_m)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_result_keys(self):
        result = probability_of_backtest_overfitting([0.1, 0.2], [0.05, 0.15])
        expected_keys = {
            "pbo", "n_paths", "is_mean", "oos_mean",
            "is_std", "oos_std", "degradation", "logit_lambda",
        }
        assert set(result.keys()) == expected_keys

    def test_degradation_positive_when_overfit(self):
        """Degradation (IS - OOS mean) should be positive when IS > OOS."""
        is_m = [0.5, 0.6, 0.7]
        oos_m = [0.1, 0.2, 0.3]
        result = probability_of_backtest_overfitting(is_m, oos_m)
        assert result["degradation"] > 0

    def test_few_paths_returns_safe_defaults(self):
        result = probability_of_backtest_overfitting([0.1], [0.05])
        assert result["pbo"] == 0.0
        assert result["n_paths"] == 1

    def test_logit_lambda_finite(self):
        """Logit should be finite even at extreme PBO values."""
        is_m = list(np.linspace(0, 1, 20))
        oos_m = list(np.linspace(1, 0, 20))
        result = probability_of_backtest_overfitting(is_m, oos_m)
        assert np.isfinite(result["logit_lambda"])

    def test_n_paths_matches_input(self):
        is_m = [0.1, 0.2, 0.3, 0.4, 0.5]
        oos_m = [0.05, 0.1, 0.15, 0.2, 0.25]
        result = probability_of_backtest_overfitting(is_m, oos_m)
        assert result["n_paths"] == 5


# ---------------------------------------------------------------------------
# Tests: cpcv_evaluate (integration)
# ---------------------------------------------------------------------------


class TestCpcvEvaluate:
    def test_returns_correct_structure(self, simple_data):
        df, cols = simple_data
        is_m, oos_m, n_paths = cpcv_evaluate(
            df, cols, n_blocks=4, n_test_blocks=1
        )
        assert len(is_m) == n_paths
        assert len(oos_m) == n_paths
        assert n_paths == comb(4, 1)

    def test_all_metrics_are_floats(self, simple_data):
        df, cols = simple_data
        is_m, oos_m, _ = cpcv_evaluate(
            df, cols, n_blocks=4, n_test_blocks=1
        )
        for v in is_m + oos_m:
            assert isinstance(v, float)

    def test_is_generally_higher_than_oos(self, simple_data):
        """In-sample IC is typically higher than OOS (some overfitting expected)."""
        df, cols = simple_data
        is_m, oos_m, _ = cpcv_evaluate(
            df, cols, n_blocks=4, n_test_blocks=1
        )
        assert np.mean(is_m) >= np.mean(oos_m) - 0.1  # Allow some noise

    def test_pbo_from_cpcv(self, simple_data):
        """End-to-end: CPCV evaluate -> PBO computation."""
        df, cols = simple_data
        is_m, oos_m, n_paths = cpcv_evaluate(
            df, cols, n_blocks=4, n_test_blocks=1
        )
        result = probability_of_backtest_overfitting(is_m, oos_m)
        assert 0.0 <= result["pbo"] <= 1.0
        assert result["n_paths"] == n_paths
