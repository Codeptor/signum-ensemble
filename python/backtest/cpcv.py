"""Combinatorial Purged Cross-Validation (CPCV) and Probability of Backtest Overfitting.

Implements:
  - ``cpcv_split`` — generates all combinatorial train/test splits from
    N purged time blocks, testing ``n_test_blocks`` at a time.
  - ``cpcv_evaluate`` — trains a model across CPCV paths and returns
    per-path performance (Sharpe or IC).
  - ``probability_of_backtest_overfitting`` — PBO per Lopez de Prado (2018):
    fraction of combinatorial paths where the best in-sample strategy
    underperforms the median out-of-sample.

References:
  - Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 12.
  - Bailey, D., Borwein, J., Lopez de Prado, M., Zhu, Q. (2017).
    "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
"""

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def cpcv_split(
    n_samples: int,
    n_blocks: int = 6,
    n_test_blocks: int = 2,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    dates: Optional[np.ndarray] = None,
    horizon: int = 5,
):
    """Combinatorial Purged Cross-Validation split generator.

    Divides data into ``n_blocks`` contiguous time blocks, then generates
    all C(n_blocks, n_test_blocks) train/test combinations.  Purge and
    embargo gaps are applied at block boundaries to prevent look-ahead bias.

    Args:
        n_samples: Total number of observations.
        n_blocks: Number of contiguous time blocks (default 6).
        n_test_blocks: Number of blocks held out for testing (default 2).
        purge_pct: Fraction of block size to purge before test blocks.
        embargo_pct: Fraction of block size to embargo after test blocks.
        dates: Optional date array for date-aware splitting.
        horizon: Forward-return horizon in trading days.

    Yields:
        (train_indices, test_indices, test_block_ids) for each combination.
    """
    if n_blocks < 3:
        raise ValueError("n_blocks must be >= 3")
    if n_test_blocks < 1 or n_test_blocks >= n_blocks:
        raise ValueError("n_test_blocks must be >= 1 and < n_blocks")

    # Build blocks in date-space or row-space
    if dates is not None:
        unique_dates = np.unique(dates)
        unique_dates.sort()
        n_dates = len(unique_dates)

        if n_dates < n_blocks:
            raise ValueError(f"Only {n_dates} dates but {n_blocks} blocks requested")

        block_size_dates = n_dates // n_blocks
        purge_days = max(horizon, int(block_size_dates * purge_pct))
        embargo_days = max(1, int(block_size_dates * embargo_pct))

        # Map dates to row indices
        date_to_rows: dict = {}
        for idx, d in enumerate(dates):
            date_to_rows.setdefault(d, []).append(idx)

        # Define block boundaries in date-space
        block_date_ranges = []
        for b in range(n_blocks):
            start = b * block_size_dates
            end = (b + 1) * block_size_dates if b < n_blocks - 1 else n_dates
            block_date_ranges.append((start, end))

        for test_blocks in combinations(range(n_blocks), n_test_blocks):
            test_blocks_set = set(test_blocks)

            # Collect test dates
            test_date_indices = set()
            for b in test_blocks:
                s, e = block_date_ranges[b]
                test_date_indices.update(range(s, e))

            # Determine purge/embargo zones around each test block
            excluded_date_indices = set(test_date_indices)
            for b in test_blocks:
                s, e = block_date_ranges[b]
                # Purge before test block
                for d_idx in range(max(0, s - purge_days), s):
                    excluded_date_indices.add(d_idx)
                # Embargo after test block
                for d_idx in range(e, min(n_dates, e + embargo_days)):
                    excluded_date_indices.add(d_idx)

            # Build row indices
            train_idx = []
            test_idx = []
            for d_idx in range(n_dates):
                rows = date_to_rows.get(unique_dates[d_idx], [])
                if d_idx in test_date_indices:
                    test_idx.extend(rows)
                elif d_idx not in excluded_date_indices:
                    train_idx.extend(rows)

            if train_idx and test_idx:
                yield sorted(train_idx), sorted(test_idx), test_blocks
    else:
        # Row-level splitting
        block_size = n_samples // n_blocks
        purge_size = max(1, int(block_size * purge_pct))
        embargo_size = max(1, int(block_size * embargo_pct))

        block_ranges = []
        for b in range(n_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < n_blocks - 1 else n_samples
            block_ranges.append((start, end))

        for test_blocks in combinations(range(n_blocks), n_test_blocks):
            test_blocks_set = set(test_blocks)

            test_rows = set()
            for b in test_blocks:
                s, e = block_ranges[b]
                test_rows.update(range(s, e))

            excluded = set(test_rows)
            for b in test_blocks:
                s, e = block_ranges[b]
                for r in range(max(0, s - purge_size), s):
                    excluded.add(r)
                for r in range(e, min(n_samples, e + embargo_size)):
                    excluded.add(r)

            train_idx = [r for r in range(n_samples) if r not in excluded]
            test_idx = sorted(test_rows)

            if train_idx and test_idx:
                yield train_idx, test_idx, test_blocks


def cpcv_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_5d",
    n_blocks: int = 6,
    n_test_blocks: int = 2,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    model_factory=None,
    horizon: int = 5,
    metric: str = "ic",
) -> tuple[list[float], list[float], int]:
    """Evaluate a model using CPCV, returning in-sample and OOS metrics per path.

    Args:
        df: DataFrame with features and target.
        feature_cols: Feature column names.
        target_col: Target column.
        n_blocks: Number of time blocks.
        n_test_blocks: Blocks held out per path.
        purge_pct: Purge fraction.
        embargo_pct: Embargo fraction.
        model_factory: Callable returning a fresh model with fit/predict.
        horizon: Forward return horizon.
        metric: "ic" (Spearman rank IC) or "sharpe" (annualized Sharpe).

    Returns:
        (is_metrics, oos_metrics, n_paths) — lists of in-sample and OOS
        metrics for each combinatorial path, plus total number of paths.
    """
    from python.alpha.model import CrossSectionalModel

    dates_array: Optional[np.ndarray] = None
    if isinstance(df.index, pd.MultiIndex):
        dates_array = df.index.get_level_values(0).values
    elif isinstance(df.index, pd.DatetimeIndex):
        dates_array = df.index.values

    is_metrics = []
    oos_metrics = []
    n_paths = 0

    for train_idx, test_idx, test_blocks in cpcv_split(
        n_samples=len(df),
        n_blocks=n_blocks,
        n_test_blocks=n_test_blocks,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        dates=dates_array,
        horizon=horizon,
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if model_factory is not None:
            model = model_factory()
        else:
            model = CrossSectionalModel(feature_cols=feature_cols)

        model.fit(train_df, target_col=target_col)

        # In-sample metric
        is_preds = model.predict(train_df)
        is_metric = _compute_metric(
            is_preds, train_df[target_col].values, metric
        )

        # Out-of-sample metric
        oos_preds = model.predict(test_df)
        oos_metric = _compute_metric(
            oos_preds, test_df[target_col].values, metric
        )

        is_metrics.append(is_metric)
        oos_metrics.append(oos_metric)
        n_paths += 1

        logger.info(
            f"CPCV path {n_paths} (test blocks {test_blocks}): "
            f"IS={is_metric:.4f}, OOS={oos_metric:.4f}"
        )

    return is_metrics, oos_metrics, n_paths


def _compute_metric(preds: np.ndarray, targets: np.ndarray, metric: str) -> float:
    """Compute a single performance metric from predictions and targets."""
    mask = ~np.isnan(preds) & ~np.isnan(targets)
    if mask.sum() < 5:
        return 0.0

    if metric == "ic":
        ic, _ = spearmanr(preds[mask], targets[mask])
        return float(ic) if not np.isnan(ic) else 0.0
    elif metric == "sharpe":
        # Treat predictions as signal-aligned returns proxy
        returns = targets[mask]
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def probability_of_backtest_overfitting(
    is_metrics: list[float],
    oos_metrics: list[float],
) -> dict:
    """Compute Probability of Backtest Overfitting (PBO).

    For each combinatorial path, ranks in-sample performance.  PBO is
    the fraction of paths where the strategy with the *best* in-sample
    rank has below-median out-of-sample performance.

    In the single-strategy case (our typical use), this simplifies to:
    PBO = fraction of paths where IS > median(IS) but OOS < median(OOS).

    Args:
        is_metrics: In-sample metric per CPCV path.
        oos_metrics: Out-of-sample metric per CPCV path.

    Returns:
        Dictionary with:
          - pbo: Probability of backtest overfitting [0, 1].
          - n_paths: Total number of combinatorial paths.
          - is_mean: Mean in-sample metric.
          - oos_mean: Mean OOS metric.
          - is_std: Std of in-sample metric.
          - oos_std: Std of OOS metric.
          - degradation: Mean (IS - OOS), measures overfitting magnitude.
          - logit_lambda: Logit of PBO (log(PBO / (1-PBO))), useful for
            comparing across studies.
    """
    if len(is_metrics) < 2 or len(oos_metrics) < 2:
        return {
            "pbo": 0.0,
            "n_paths": len(is_metrics),
            "is_mean": float(np.mean(is_metrics)) if is_metrics else 0.0,
            "oos_mean": float(np.mean(oos_metrics)) if oos_metrics else 0.0,
            "is_std": 0.0,
            "oos_std": 0.0,
            "degradation": 0.0,
            "logit_lambda": 0.0,
        }

    is_arr = np.array(is_metrics)
    oos_arr = np.array(oos_metrics)

    is_median = np.median(is_arr)
    oos_median = np.median(oos_arr)

    # PBO: fraction of paths where IS is above-median but OOS is below-median
    above_is_median = is_arr >= is_median
    below_oos_median = oos_arr < oos_median
    overfit_paths = above_is_median & below_oos_median
    pbo = float(overfit_paths.sum()) / len(is_arr)

    # Degradation: average drop from IS to OOS performance
    degradation = float(np.mean(is_arr - oos_arr))

    # Logit-lambda (bounded to avoid inf)
    pbo_clipped = np.clip(pbo, 0.01, 0.99)
    logit_lambda = float(np.log(pbo_clipped / (1 - pbo_clipped)))

    return {
        "pbo": pbo,
        "n_paths": len(is_arr),
        "is_mean": float(np.mean(is_arr)),
        "oos_mean": float(np.mean(oos_arr)),
        "is_std": float(np.std(is_arr)),
        "oos_std": float(np.std(oos_arr)),
        "degradation": degradation,
        "logit_lambda": logit_lambda,
    }
