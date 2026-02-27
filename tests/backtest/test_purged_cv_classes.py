"""Tests for purged and embargo cross-validation classes."""

import numpy as np
import pytest

from python.backtest.purged_cv import (
    CVResult,
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedWalkForward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(T=500, N=5, seed=42):
    """Generate synthetic feature matrix and label boundaries."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    indices = np.arange(T)
    # Labels span 3 bars forward
    labels_start = indices.copy()
    labels_end = np.minimum(indices + 3, T - 1)
    return X, labels_start, labels_end


# ---------------------------------------------------------------------------
# PurgedKFold
# ---------------------------------------------------------------------------


class TestPurgedKFold:
    def test_yields_correct_number_of_folds(self):
        X, ls, le = _make_data()
        cv = PurgedKFold(n_splits=5, purge_window=3, embargo_window=2)
        folds = list(cv.split(X, ls, le))
        assert len(folds) == 5

    def test_get_n_splits(self):
        cv = PurgedKFold(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_no_overlap_train_test(self):
        """Train and test indices must be disjoint."""
        X, ls, le = _make_data()
        cv = PurgedKFold(n_splits=5, purge_window=5, embargo_window=2)
        for train_idx, test_idx in cv.split(X, ls, le):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_test_indices_cover_all_samples(self):
        """Union of test sets should cover most samples."""
        X, ls, le = _make_data(T=100)
        cv = PurgedKFold(n_splits=5, purge_window=1, embargo_window=0)
        all_test = set()
        for _, test_idx in cv.split(X, ls, le):
            all_test.update(test_idx)
        assert len(all_test) == 100

    def test_purge_removes_nearby_training(self):
        """With large purge window, training set should be smaller."""
        X, ls, le = _make_data(T=200)
        cv_small = PurgedKFold(n_splits=5, purge_window=1, embargo_window=0)
        cv_large = PurgedKFold(n_splits=5, purge_window=20, embargo_window=0)
        trains_small = [len(t) for t, _ in cv_small.split(X, ls, le)]
        trains_large = [len(t) for t, _ in cv_large.split(X, ls, le)]
        assert sum(trains_large) < sum(trains_small)

    def test_embargo_removes_after_test(self):
        """Embargo removes samples immediately after test set."""
        X, ls, le = _make_data(T=200)
        cv_no = PurgedKFold(n_splits=5, purge_window=0, embargo_window=0)
        cv_emb = PurgedKFold(n_splits=5, purge_window=0, embargo_window=10)
        trains_no = [len(t) for t, _ in cv_no.split(X, ls, le)]
        trains_emb = [len(t) for t, _ in cv_emb.split(X, ls, le)]
        assert sum(trains_emb) < sum(trains_no)

    def test_default_labels(self):
        """Should work without explicit label boundaries."""
        X = np.random.default_rng(42).standard_normal((100, 3))
        cv = PurgedKFold(n_splits=5)
        folds = list(cv.split(X))
        assert len(folds) == 5
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_train_indices_sorted(self):
        X, ls, le = _make_data()
        cv = PurgedKFold(n_splits=5)
        for train_idx, _ in cv.split(X, ls, le):
            assert np.all(np.diff(train_idx) > 0)

    def test_small_dataset(self):
        """Should handle small datasets without crashing."""
        X = np.random.default_rng(42).standard_normal((20, 2))
        cv = PurgedKFold(n_splits=4, purge_window=1, embargo_window=1)
        folds = list(cv.split(X))
        assert len(folds) >= 1


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV
# ---------------------------------------------------------------------------


class TestCombinatorialPurgedCV:
    def test_n_combinations(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        # C(6,2) = 15
        assert cv.n_combinations == 15

    def test_n_backtest_paths(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        # C(5,1) = 5
        assert cv.n_backtest_paths == 5

    def test_get_n_splits(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        assert cv.get_n_splits() == 15

    def test_yields_correct_number_of_splits(self):
        X, ls, le = _make_data(T=300)
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, purge_window=2, embargo_window=1
        )
        folds = list(cv.split(X, ls, le))
        assert len(folds) == 15

    def test_no_overlap_train_test(self):
        X, ls, le = _make_data(T=300)
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, purge_window=3, embargo_window=2
        )
        for train_idx, test_idx in cv.split(X, ls, le):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_purge_removes_training(self):
        """Larger purge window should remove more training samples."""
        X, ls, le = _make_data(T=300)
        cv_small = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, purge_window=1, embargo_window=0
        )
        cv_large = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, purge_window=15, embargo_window=0
        )
        trains_s = [len(t) for t, _ in cv_small.split(X, ls, le)]
        trains_l = [len(t) for t, _ in cv_large.split(X, ls, le)]
        assert sum(trains_l) < sum(trains_s)

    def test_default_labels(self):
        X = np.random.default_rng(42).standard_normal((200, 3))
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        folds = list(cv.split(X))
        assert len(folds) > 0

    def test_three_test_splits(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=3)
        # C(6,3) = 20
        assert cv.n_combinations == 20
        X = np.random.default_rng(42).standard_normal((300, 3))
        folds = list(cv.split(X))
        assert len(folds) == 20


# ---------------------------------------------------------------------------
# PurgedWalkForward
# ---------------------------------------------------------------------------


class TestPurgedWalkForward:
    def test_yields_folds(self):
        X = np.random.default_rng(42).standard_normal((500, 3))
        cv = PurgedWalkForward(
            n_splits=5, min_train_size=100, test_size=50,
            purge_window=3, embargo_window=2,
        )
        folds = list(cv.split(X))
        assert len(folds) >= 1

    def test_get_n_splits(self):
        cv = PurgedWalkForward(n_splits=8)
        assert cv.get_n_splits() == 8

    def test_no_overlap(self):
        X = np.random.default_rng(42).standard_normal((500, 3))
        cv = PurgedWalkForward(
            n_splits=5, min_train_size=100, test_size=50,
            purge_window=5, embargo_window=3,
        )
        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_train_before_test(self):
        """All training samples should come before test samples."""
        X = np.random.default_rng(42).standard_normal((500, 3))
        cv = PurgedWalkForward(
            n_splits=5, min_train_size=100, test_size=50,
            purge_window=3, embargo_window=2,
        )
        for train_idx, test_idx in cv.split(X):
            assert train_idx[-1] < test_idx[0]

    def test_expanding_window(self):
        """Expanding mode: training sets grow over folds."""
        X = np.random.default_rng(42).standard_normal((1000, 3))
        cv = PurgedWalkForward(
            n_splits=5, min_train_size=100, test_size=50,
            purge_window=3, embargo_window=2, expanding=True,
        )
        sizes = [len(t) for t, _ in cv.split(X)]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_sliding_window(self):
        """Sliding mode: training sets stay roughly same size."""
        X = np.random.default_rng(42).standard_normal((1000, 3))
        cv = PurgedWalkForward(
            n_splits=5, min_train_size=200, test_size=50,
            purge_window=3, embargo_window=2, expanding=False,
        )
        sizes = [len(t) for t, _ in cv.split(X)]
        if len(sizes) >= 2:
            assert sizes[-1] - sizes[0] < 100

    def test_gap_between_train_and_test(self):
        """Purge + embargo should create a gap."""
        X = np.random.default_rng(42).standard_normal((500, 3))
        purge = 5
        embargo = 3
        cv = PurgedWalkForward(
            n_splits=3, min_train_size=100, test_size=50,
            purge_window=purge, embargo_window=embargo,
        )
        for train_idx, test_idx in cv.split(X):
            gap = test_idx[0] - train_idx[-1]
            assert gap >= purge + embargo


# ---------------------------------------------------------------------------
# CVResult
# ---------------------------------------------------------------------------


class TestCVResult:
    def test_mean_score(self):
        r = CVResult(scores=[0.8, 0.9, 0.7], train_sizes=[100] * 3, test_sizes=[20] * 3)
        assert r.mean_score == pytest.approx(0.8, abs=0.01)

    def test_std_score(self):
        r = CVResult(scores=[0.8, 0.9, 0.7], train_sizes=[100] * 3, test_sizes=[20] * 3)
        assert r.std_score > 0

    def test_n_folds(self):
        r = CVResult(scores=[0.5, 0.6], train_sizes=[100, 100], test_sizes=[20, 20])
        assert r.n_folds == 2

    def test_sharpe_of_scores(self):
        r = CVResult(scores=[0.8, 0.9, 0.7], train_sizes=[100] * 3, test_sizes=[20] * 3)
        expected = r.mean_score / r.std_score
        assert r.sharpe_of_scores == pytest.approx(expected, abs=0.01)

    def test_sharpe_zero_std(self):
        r = CVResult(scores=[0.5, 0.5, 0.5], train_sizes=[100] * 3, test_sizes=[20] * 3)
        assert r.sharpe_of_scores == 0.0

    def test_empty_scores(self):
        r = CVResult(scores=[], train_sizes=[], test_sizes=[])
        assert r.mean_score == 0.0
        assert r.n_folds == 0

    def test_single_score(self):
        r = CVResult(scores=[0.85], train_sizes=[100], test_sizes=[20])
        assert r.mean_score == 0.85
        assert r.std_score == 0.0

    def test_summary_string(self):
        r = CVResult(scores=[0.8, 0.9, 0.7], train_sizes=[100] * 3, test_sizes=[20] * 3)
        s = r.summary()
        assert "3 folds" in s
        assert "mean=" in s
