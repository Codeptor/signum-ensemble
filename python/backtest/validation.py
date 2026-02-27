"""Walk-forward, purged k-fold, and CPCV validation for financial models.

Phase 4 additions:
  - ``purged_kfold_split`` — generator yielding (train, test) index pairs with
    purge and embargo gaps to eliminate look-ahead bias from overlapping
    return windows.
  - ``purged_kfold_cv`` — convenience function that trains + evaluates a model
    across purged k-fold splits, returning mean and std IC.
"""

import logging
from typing import Generator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.6,
    embargo_days: int = 22,
):
    """Walk-forward cross-validation with embargo period.

    Yields (train_indices, test_indices) tuples.

    R3-V-1 fix: embargo is now computed in *date-space* (unique dates)
    rather than row-space, so it correctly represents trading days even
    with panel data (multiple rows per date).
    """
    # Determine if we have date information in the index
    if hasattr(df.index, "get_level_values"):
        try:
            dates = df.index.get_level_values(0)
        except Exception:
            dates = df.index
    else:
        dates = df.index

    unique_dates = dates.unique().sort_values()
    n_dates = len(unique_dates)

    if n_dates < n_splits + embargo_days:
        return

    min_train_dates = max(1, int(n_dates * train_pct / (n_splits + train_pct * (1 - n_splits))))
    total_test_dates = n_dates - min_train_dates - embargo_days
    test_date_size = total_test_dates // n_splits

    if test_date_size <= 0:
        return

    for i in range(n_splits):
        test_date_start = min_train_dates + embargo_days + i * test_date_size
        test_date_end = test_date_start + test_date_size if i < n_splits - 1 else n_dates
        train_date_end = test_date_start - embargo_days

        if train_date_end <= 0 or test_date_start >= n_dates:
            continue

        train_dates = set(unique_dates[:train_date_end])
        test_dates = set(unique_dates[test_date_start : min(test_date_end, n_dates)])

        train_idx = [j for j, d in enumerate(dates) if d in train_dates]
        test_idx = [j for j, d in enumerate(dates) if d in test_dates]

        if train_idx and test_idx:
            yield train_idx, test_idx


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    Adjusts for multiple testing by comparing against the expected maximum
    Sharpe under the null hypothesis.
    """
    from scipy import stats

    # R3-B-3 fix: guard n_trials <= 1 (no multiple-testing penalty)
    if n_trials <= 1 or n_observations <= 0:
        return 1.0

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    log_n = np.log(n_trials)
    if log_n <= 0:
        return 1.0
    e_max = stats.norm.ppf(1 - 1 / n_trials) * (1 - 0.5772 / log_n)

    # Standard error of Sharpe
    se = np.sqrt((1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2) / n_observations)
    if se <= 0:
        return 1.0

    # Probability that the observed Sharpe exceeds the expected max under null
    dsr = stats.norm.cdf((sharpe - e_max) / se)
    return float(dsr)


# ---------------------------------------------------------------------------
# Phase 4: Purged k-fold cross-validation
# ---------------------------------------------------------------------------


def purged_kfold_split(
    n_samples: int,
    n_splits: int = 5,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    dates: Optional[np.ndarray] = None,
    horizon: int = 5,
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Purged k-fold split with embargo (Lopez de Prado, 2018).

    Unlike standard k-fold, this accounts for temporal dependence in
    financial data by:

    1. **Purge**: Removing days *before* each test fold from the training
       set to eliminate label overlap from the forward-return horizon.
    2. **Embargo**: Removing days *after* each test fold from the training
       set to prevent information leakage from auto-correlated features.

    C-PURGE fix: the split now operates on **unique dates**, not raw row
    indices.  With a panel of 500 tickers per date, the old row-based
    approach created a purge gap of ~0 calendar days (``int(fold_size *
    0.02)`` ≈ 4 rows ≈ 0 dates), providing zero leakage protection.
    The new version ensures the purge gap is *at least* ``horizon``
    trading days in calendar space.

    Args:
        n_samples: Total number of observations (assumed time-ordered).
        n_splits: Number of folds (default 5).
        purge_pct: Fraction of date-level fold size to purge before test.
        embargo_pct: Fraction of date-level fold size to embargo after test.
        dates: Optional 1-D array of date labels (one per row).  When
            provided, the split operates in date-space.  When None, falls
            back to row-level splitting (backward-compatible).
        horizon: Forward-return horizon in trading days.  Purge gap is
            ``max(horizon, int(fold_dates * purge_pct))`` so it is never
            smaller than the label window.

    Yields:
        (train_indices, test_indices) tuples for each fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_samples < n_splits:
        raise ValueError(f"n_samples ({n_samples}) must be >= n_splits ({n_splits})")

    # ----- Date-aware path (C-PURGE fix) ----- #
    if dates is not None:
        unique_dates = np.unique(dates)
        unique_dates.sort()
        n_dates = len(unique_dates)

        if n_dates < n_splits:
            raise ValueError(f"Only {n_dates} unique dates but {n_splits} splits requested")

        fold_dates = n_dates // n_splits
        purge_days = max(horizon, int(fold_dates * purge_pct))
        embargo_days = max(1, int(fold_dates * embargo_pct))

        # Build a row-index lookup: date -> list of row positions
        date_to_rows: dict = {}
        for idx, d in enumerate(dates):
            date_to_rows.setdefault(d, []).append(idx)

        for i in range(n_splits):
            test_date_start = i * fold_dates
            test_date_end = (i + 1) * fold_dates if i < n_splits - 1 else n_dates

            # Purge and embargo in date-space
            purge_date_start = max(0, test_date_start - purge_days)
            embargo_date_end = min(n_dates, test_date_end + embargo_days)

            # Collect row indices for train / test
            train_dates_before = unique_dates[:purge_date_start]
            train_dates_after = unique_dates[embargo_date_end:]
            test_dates = unique_dates[test_date_start:test_date_end]

            train_idx = []
            for d in train_dates_before:
                train_idx.extend(date_to_rows[d])
            for d in train_dates_after:
                train_idx.extend(date_to_rows[d])

            test_idx = []
            for d in test_dates:
                test_idx.extend(date_to_rows[d])

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield sorted(train_idx), sorted(test_idx)
        return

    # ----- Legacy row-level path (backward-compatible) ----- #
    fold_size = n_samples // n_splits
    purge_size = max(1, int(fold_size * purge_pct))
    embargo_size = max(1, int(fold_size * embargo_pct))

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples

        purge_start = max(0, test_start - purge_size)
        embargo_end = min(n_samples, test_end + embargo_size)

        train_before = list(range(0, purge_start))
        train_after = list(range(embargo_end, n_samples))
        train_idx = train_before + train_after
        test_idx = list(range(test_start, test_end))

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx


def purged_kfold_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_5d",
    n_splits: int = 5,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    model_factory: Optional[object] = None,
    horizon: int = 5,
) -> tuple[float, float, list[float]]:
    """Run purged k-fold cross-validation and return IC statistics.

    Trains a fresh model per fold and evaluates Information Coefficient
    (Spearman rank correlation between predictions and true targets).

    C-PURGE fix: when ``df`` has a DatetimeIndex (or a MultiIndex whose
    level-0 is datetime), the split is performed in date-space with a
    purge gap of at least ``horizon`` trading days.

    Args:
        df: DataFrame with feature columns and target, assumed time-ordered.
        feature_cols: List of feature column names.
        target_col: Target column name.
        n_splits: Number of CV folds.
        purge_pct: Purge fraction (see ``purged_kfold_split``).
        embargo_pct: Embargo fraction.
        model_factory: Callable that returns a fresh model instance with
            ``.fit(df, target_col)`` and ``.predict(df)`` methods.
            Defaults to ``CrossSectionalModel(feature_cols=feature_cols)``.
        horizon: Forward-return horizon in trading days (default 5).

    Returns:
        (mean_ic, std_ic, fold_ics) — mean IC, standard deviation, and
        per-fold IC values.
    """
    from python.alpha.model import CrossSectionalModel

    n_samples = len(df)

    # C-PURGE fix: extract date array for date-aware splitting
    dates_array: Optional[np.ndarray] = None
    if isinstance(df.index, pd.MultiIndex):
        dates_array = df.index.get_level_values(0).values
    elif isinstance(df.index, pd.DatetimeIndex):
        dates_array = df.index.values

    fold_ics: list[float] = []

    for fold_num, (train_idx, test_idx) in enumerate(
        purged_kfold_split(
            n_samples,
            n_splits,
            purge_pct,
            embargo_pct,
            dates=dates_array,
            horizon=horizon,
        )
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Create a fresh model for each fold
        if model_factory is not None:
            model = model_factory()  # type: ignore[operator]
        else:
            model = CrossSectionalModel(feature_cols=feature_cols)

        # Train
        model.fit(train_df, target_col=target_col)

        # Predict and compute IC
        preds = model.predict(test_df)
        y_true = test_df[target_col].values

        # Handle NaN in predictions or targets
        mask = ~np.isnan(preds) & ~np.isnan(y_true)
        if mask.sum() < 5:
            logger.warning(f"Fold {fold_num + 1}: too few valid predictions, skipping")
            continue

        # H-PEARSON fix: use Spearman rank IC instead of Pearson.
        from scipy.stats import spearmanr

        ic_val, _ = spearmanr(preds[mask], y_true[mask])
        ic = float(ic_val) if not np.isnan(ic_val) else 0.0
        fold_ics.append(ic)
        logger.info(
            f"Fold {fold_num + 1}/{n_splits}: IC = {ic:.4f} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

    if not fold_ics:
        logger.error("No valid folds — returning zero IC")
        return 0.0, 0.0, []

    mean_ic = float(np.mean(fold_ics))
    std_ic = float(np.std(fold_ics))
    logger.info(f"Purged {n_splits}-fold CV: IC = {mean_ic:.4f} ± {std_ic:.4f}")

    return mean_ic, std_ic, fold_ics
