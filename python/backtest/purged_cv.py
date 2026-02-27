"""Purged and embargo cross-validation for financial ML.

Standard k-fold CV leaks information through overlapping labels and
autocorrelated features. This module implements:
  1. Purged k-fold: removes training samples that overlap with test labels.
  2. Embargo: adds a buffer after purging to account for serial correlation.
  3. Combinatorial purged CV (CPCV): exhaustive path generation for
     unbiased backtesting (de Prado 2018).

Usage::

    cv = PurgedKFold(n_splits=5, purge_window=5, embargo_window=2)
    for train_idx, test_idx in cv.split(X, labels_start, labels_end):
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])

References:
  - de Prado (2018), "Advances in Financial Machine Learning", Ch. 7
  - Bailey et al. (2014), "The Deflated Sharpe Ratio"
"""

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------


@dataclass
class PurgedKFold:
    """K-fold CV with purging and embargo for financial time series.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    purge_window : int
        Number of samples to remove from training set around each
        test boundary to prevent label leakage.
    embargo_window : int
        Additional buffer after purge window for serial correlation.
    """

    n_splits: int = 5
    purge_window: int = 3
    embargo_window: int = 2

    def split(
        self,
        X: np.ndarray,
        labels_start: np.ndarray | None = None,
        labels_end: np.ndarray | None = None,
    ):
        """Generate train/test indices with purging and embargo.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (T, N).
        labels_start : np.ndarray, optional
            Start index of each label's information set.
            If None, uses sample index.
        labels_end : np.ndarray, optional
            End index of each label's information set.
            If None, uses sample index + 1.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        T = len(X)
        indices = np.arange(T)

        if labels_start is None:
            labels_start = indices.copy()
        if labels_end is None:
            labels_end = indices + 1

        fold_size = T // self.n_splits

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else T
            test_idx = indices[test_start:test_end]

            # Find training indices with purging
            train_mask = np.ones(T, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purge: remove training samples whose labels overlap test period
            test_label_start = labels_start[test_start]
            test_label_end = labels_end[test_end - 1]

            for i in range(T):
                if not train_mask[i]:
                    continue
                # If this training sample's label overlaps with test period
                if (
                    labels_end[i] > test_label_start - self.purge_window
                    and labels_start[i] < test_label_end + self.purge_window
                ):
                    train_mask[i] = False

            # Embargo: remove samples right after test set
            embargo_end = min(T, test_end + self.embargo_window)
            train_mask[test_end:embargo_end] = False

            # Also embargo before test set
            embargo_before = max(0, test_start - self.embargo_window)
            train_mask[embargo_before:test_start] = False

            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                logger.warning(f"Fold {fold}: empty training set after purging")
                continue

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Combinatorial Purged CV (CPCV)
# ---------------------------------------------------------------------------


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (de Prado 2018).

    Generates all C(n_splits, n_test_splits) combinations of test
    groups, producing more backtest paths than standard k-fold.

    Parameters
    ----------
    n_splits : int
        Number of groups to divide the data into.
    n_test_splits : int
        Number of groups to use as test set per combination.
    purge_window : int
        Purging buffer.
    embargo_window : int
        Embargo buffer.
    """

    n_splits: int = 6
    n_test_splits: int = 2
    purge_window: int = 3
    embargo_window: int = 2

    @property
    def n_combinations(self) -> int:
        """Total number of train/test combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    @property
    def n_backtest_paths(self) -> int:
        """Number of unique backtest paths (each sample tested once)."""
        from math import comb
        return comb(self.n_splits - 1, self.n_test_splits - 1)

    def split(
        self,
        X: np.ndarray,
        labels_start: np.ndarray | None = None,
        labels_end: np.ndarray | None = None,
    ):
        """Generate all combinatorial train/test splits.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        T = len(X)
        indices = np.arange(T)

        if labels_start is None:
            labels_start = indices.copy()
        if labels_end is None:
            labels_end = indices + 1

        # Divide into groups
        group_boundaries = np.array_split(indices, self.n_splits)

        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            # Test indices
            test_idx = np.concatenate([group_boundaries[g] for g in test_groups])

            # Training: everything else with purging and embargo
            train_mask = np.ones(T, dtype=bool)
            train_mask[test_idx] = False

            # Apply purging and embargo around each test group
            for g in test_groups:
                g_start = group_boundaries[g][0]
                g_end = group_boundaries[g][-1] + 1

                # Purge window
                purge_start = max(0, g_start - self.purge_window)
                purge_end = min(T, g_end + self.purge_window)
                train_mask[purge_start:purge_end] = False

                # Embargo
                embargo_end = min(T, g_end + self.purge_window + self.embargo_window)
                train_mask[g_end:embargo_end] = False

            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_combinations


# ---------------------------------------------------------------------------
# Walk-forward with purging
# ---------------------------------------------------------------------------


@dataclass
class PurgedWalkForward:
    """Walk-forward validation with purging.

    Expanding or sliding window with purge/embargo buffers.

    Parameters
    ----------
    n_splits : int
        Number of forward steps.
    min_train_size : int
        Minimum training window size.
    test_size : int
        Size of each test window.
    purge_window : int
        Purge buffer.
    embargo_window : int
        Embargo buffer.
    expanding : bool
        If True, training window expands. If False, it slides.
    """

    n_splits: int = 5
    min_train_size: int = 100
    test_size: int = 50
    purge_window: int = 3
    embargo_window: int = 2
    expanding: bool = True

    def split(self, X: np.ndarray):
        """Generate walk-forward train/test splits.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        T = len(X)
        indices = np.arange(T)

        # Calculate step size to fit n_splits
        total_test = self.test_size * self.n_splits
        available = T - self.min_train_size
        if available < total_test:
            step = max(1, available // self.n_splits)
        else:
            step = self.test_size

        for fold in range(self.n_splits):
            test_start = self.min_train_size + fold * step
            test_end = min(test_start + self.test_size, T)

            if test_start >= T:
                break

            # Training window
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.min_train_size)

            # Apply purge and embargo
            train_end = max(0, test_start - self.purge_window - self.embargo_window)

            if train_end <= train_start:
                continue

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# CV score aggregation
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Aggregated cross-validation results."""

    scores: list[float]
    train_sizes: list[int]
    test_sizes: list[int]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0

    @property
    def n_folds(self) -> int:
        return len(self.scores)

    @property
    def sharpe_of_scores(self) -> float:
        """Sharpe ratio of OOS scores (higher = more consistent)."""
        if self.std_score < 1e-12:
            return 0.0
        return self.mean_score / self.std_score

    def summary(self) -> str:
        return (
            f"CV: {self.n_folds} folds, "
            f"mean={self.mean_score:.4f} ± {self.std_score:.4f}, "
            f"Sharpe(scores)={self.sharpe_of_scores:.2f}"
        )
