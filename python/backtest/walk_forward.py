"""Walk-forward optimization engine for strategy parameter selection.

Implements anchored and rolling walk-forward analysis with proper
train/test separation, parameter optimization on in-sample windows,
and out-of-sample evaluation to detect overfitting and parameter decay.

Walk-forward is the gold standard for strategy validation — unlike a
single train/test split, it produces multiple OOS periods and reveals
whether optimized parameters remain stable over time.

Usage::

    wfo = WalkForwardOptimizer(
        returns=returns_df,
        param_grid={"lookback": [20, 60, 120], "threshold": [0.5, 1.0, 1.5]},
    )
    result = wfo.run(strategy_fn, metric="sharpe")
    # result.oos_equity → concatenated out-of-sample equity curve
    # result.is_degradation → True if OOS performance decays over windows

References:
  - Pardo (2008), "The Evaluation and Optimization of Trading Strategies"
  - Bailey et al. (2014), "The Deflated Sharpe Ratio"
"""

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_id: int
    train_start: Any  # date or int
    train_end: Any
    test_start: Any
    test_end: Any
    best_params: dict
    is_metric: float  # in-sample metric value
    oos_metric: float  # out-of-sample metric value
    oos_returns: pd.Series  # out-of-sample returns
    all_params_metrics: list[dict] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization results."""

    windows: list[WindowResult]
    oos_equity: pd.Series  # concatenated OOS equity curve
    oos_returns: pd.Series  # concatenated OOS returns

    # Aggregate metrics
    mean_oos_metric: float
    median_oos_metric: float
    std_oos_metric: float
    mean_is_metric: float

    # Degradation analysis
    is_degradation: bool  # True if OOS performance trends down
    degradation_slope: float  # regression slope of OOS metric over windows
    efficiency_ratio: float  # mean(OOS metric) / mean(IS metric)

    # Parameter stability
    param_stability: dict[str, float]  # CV of each param across windows

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def total_oos_days(self) -> int:
        return len(self.oos_returns)

    @property
    def oos_sharpe(self) -> float:
        """Annualized Sharpe of concatenated OOS returns."""
        if len(self.oos_returns) < 2:
            return 0.0
        ann = np.sqrt(252)
        return float(self.oos_returns.mean() / max(self.oos_returns.std(), 1e-10) * ann)

    @property
    def oos_total_return(self) -> float:
        return float((1 + self.oos_returns).prod() - 1)

    @property
    def oos_max_drawdown(self) -> float:
        equity = (1 + self.oos_returns).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak.clip(lower=1e-10)
        return float(dd.min())

    def summary(self) -> str:
        return (
            f"Walk-Forward: {self.n_windows} windows, "
            f"{self.total_oos_days} OOS days\n"
            f"  OOS Sharpe={self.oos_sharpe:.3f}, "
            f"Return={self.oos_total_return:.2%}, "
            f"MaxDD={self.oos_max_drawdown:.2%}\n"
            f"  Efficiency={self.efficiency_ratio:.2%}, "
            f"Degradation={self.is_degradation} "
            f"(slope={self.degradation_slope:.4f})\n"
            f"  Param stability: {self.param_stability}"
        )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() < 1e-10:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _sortino(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() < 1e-10:
        return float(returns.mean() * np.sqrt(252) / 1e-10) if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(252))


def _calmar(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak.clip(lower=1e-10)).min()
    if abs(max_dd) < 1e-10:
        return 0.0
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    return float(ann_ret / abs(max_dd))


def _total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


METRIC_FUNCTIONS: dict[str, Callable[[pd.Series], float]] = {
    "sharpe": _sharpe,
    "sortino": _sortino,
    "calmar": _calmar,
    "total_return": _total_return,
}


# ---------------------------------------------------------------------------
# Walk-Forward Optimizer
# ---------------------------------------------------------------------------


class WalkForwardOptimizer:
    """Walk-forward optimization with anchored or rolling windows.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (columns=tickers, index=dates).
    param_grid : dict[str, list]
        Parameter grid to search. Keys are parameter names, values are lists
        of values to try. All combinations are evaluated.
    n_windows : int
        Number of walk-forward windows. More windows = more robust but each
        window is shorter.
    train_ratio : float
        Fraction of each window used for training (default 0.7 = 70% train).
    anchored : bool
        If True, training window starts from the beginning of data (expanding).
        If False, rolling window of fixed size.
    purge_days : int
        Days to purge between train and test to prevent leakage.
    min_train_days : int
        Minimum training period in days. Windows shorter than this are skipped.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        param_grid: dict[str, list],
        n_windows: int = 5,
        train_ratio: float = 0.7,
        anchored: bool = False,
        purge_days: int = 5,
        min_train_days: int = 60,
    ):
        self.returns = returns.sort_index()
        self.param_grid = param_grid
        self.n_windows = n_windows
        self.train_ratio = train_ratio
        self.anchored = anchored
        self.purge_days = purge_days
        self.min_train_days = min_train_days

        # Pre-compute all parameter combinations
        keys = sorted(param_grid.keys())
        self._param_keys = keys
        if len(keys) == 0:
            raise ValueError("param_grid produced no parameter combinations")

        self._param_combos = [
            dict(zip(keys, vals))
            for vals in product(*(param_grid[k] for k in keys))
        ]

        if len(self._param_combos) == 0:
            raise ValueError("param_grid produced no parameter combinations")

    def _generate_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        dates = self.returns.index
        n = len(dates)

        if self.n_windows < 1:
            raise ValueError("n_windows must be >= 1")

        windows = []

        if self.anchored:
            # Anchored: train always starts at beginning, test windows slide
            # Divide the latter portion into n_windows test segments
            min_train = max(self.min_train_days, int(n * self.train_ratio * 0.5))
            remaining = n - min_train
            test_size = remaining // self.n_windows

            if test_size < 5:
                raise ValueError(
                    f"Not enough data for {self.n_windows} anchored windows "
                    f"({n} days, min_train={min_train})"
                )

            for w in range(self.n_windows):
                train_start = dates[0]
                test_start_idx = min_train + w * test_size + self.purge_days
                test_end_idx = min(min_train + (w + 1) * test_size, n) - 1

                if test_start_idx >= n or test_end_idx <= test_start_idx:
                    continue

                train_end = dates[test_start_idx - self.purge_days - 1]
                test_start = dates[test_start_idx]
                test_end = dates[test_end_idx]

                windows.append((train_start, train_end, test_start, test_end))
        else:
            # Rolling: fixed-size train window slides forward
            window_size = n // self.n_windows
            train_size = int(window_size * self.train_ratio)

            if train_size < self.min_train_days:
                raise ValueError(
                    f"Train size {train_size} < min_train_days {self.min_train_days}. "
                    f"Reduce n_windows or min_train_days."
                )

            for w in range(self.n_windows):
                base = w * window_size
                train_start_idx = base
                train_end_idx = base + train_size - 1
                test_start_idx = train_end_idx + self.purge_days + 1
                test_end_idx = min(base + window_size - 1, n - 1)

                if test_start_idx >= n or test_end_idx <= test_start_idx:
                    continue

                windows.append((
                    dates[train_start_idx],
                    dates[train_end_idx],
                    dates[test_start_idx],
                    dates[test_end_idx],
                ))

        return windows

    def run(
        self,
        strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
        metric: str = "sharpe",
        custom_metric_fn: Optional[Callable[[pd.Series], float]] = None,
    ) -> WalkForwardResult:
        """Run walk-forward optimization.

        Parameters
        ----------
        strategy_fn : callable
            Function(returns_df, params_dict) -> pd.Series of portfolio returns.
            Called with the training data and each parameter combination to
            evaluate in-sample, then with test data and the best params for OOS.
        metric : str
            Optimization metric: "sharpe", "sortino", "calmar", "total_return".
        custom_metric_fn : callable, optional
            Custom metric function(returns) -> float. Overrides `metric` if provided.

        Returns
        -------
        WalkForwardResult
        """
        if custom_metric_fn is not None:
            metric_fn = custom_metric_fn
        elif metric in METRIC_FUNCTIONS:
            metric_fn = METRIC_FUNCTIONS[metric]
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from {list(METRIC_FUNCTIONS)}"
            )

        windows = self._generate_windows()
        if len(windows) == 0:
            raise ValueError("No valid windows generated. Check data length and parameters.")

        logger.info(
            f"Walk-forward: {len(windows)} windows, "
            f"{len(self._param_combos)} param combos, "
            f"{'anchored' if self.anchored else 'rolling'}"
        )

        results: list[WindowResult] = []

        for w_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_data = self.returns.loc[train_start:train_end]
            test_data = self.returns.loc[test_start:test_end]

            if len(train_data) < self.min_train_days or len(test_data) < 2:
                logger.warning(f"Window {w_id}: skipped (train={len(train_data)}, test={len(test_data)})")
                continue

            # Optimize on in-sample
            best_params = None
            best_is_metric = -np.inf
            all_params_metrics = []

            for params in self._param_combos:
                try:
                    is_returns = strategy_fn(train_data, params)
                    is_val = metric_fn(is_returns)
                except Exception as e:
                    logger.debug(f"Window {w_id}, params {params}: {e}")
                    is_val = -np.inf

                all_params_metrics.append({"params": params, "metric": is_val})

                if is_val > best_is_metric:
                    best_is_metric = is_val
                    best_params = params

            if best_params is None:
                logger.warning(f"Window {w_id}: all parameter combos failed")
                continue

            # Evaluate on OOS with best params
            try:
                oos_returns = strategy_fn(test_data, best_params)
                oos_val = metric_fn(oos_returns)
            except Exception as e:
                logger.warning(f"Window {w_id} OOS eval failed: {e}")
                continue

            results.append(WindowResult(
                window_id=w_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                is_metric=best_is_metric,
                oos_metric=oos_val,
                oos_returns=oos_returns,
                all_params_metrics=all_params_metrics,
            ))

            logger.info(
                f"Window {w_id}: IS {metric}={best_is_metric:.4f}, "
                f"OOS {metric}={oos_val:.4f}, params={best_params}"
            )

        if len(results) == 0:
            raise RuntimeError("All walk-forward windows failed. Check strategy_fn and data.")

        return self._aggregate(results)

    def _aggregate(self, results: list[WindowResult]) -> WalkForwardResult:
        """Aggregate window results into a WalkForwardResult."""
        # Concatenate OOS returns
        oos_parts = [w.oos_returns for w in results]
        oos_returns = pd.concat(oos_parts)
        # Remove any duplicate indices (shouldn't happen but be safe)
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
        oos_returns = oos_returns.sort_index()

        # Build equity curve
        oos_equity = (1 + oos_returns).cumprod()

        # Aggregate metrics
        oos_metrics = [w.oos_metric for w in results]
        is_metrics = [w.is_metric for w in results]

        mean_oos = float(np.mean(oos_metrics))
        median_oos = float(np.median(oos_metrics))
        std_oos = float(np.std(oos_metrics))
        mean_is = float(np.mean(is_metrics))

        # Efficiency ratio: OOS / IS
        efficiency = mean_oos / max(abs(mean_is), 1e-10)

        # Degradation: linear regression of OOS metric over window index
        x = np.arange(len(oos_metrics), dtype=float)
        if len(x) >= 2:
            slope = float(np.polyfit(x, oos_metrics, 1)[0])
        else:
            slope = 0.0

        # Degradation is significant if slope is negative and meaningful
        is_degradation = slope < -abs(mean_oos) * 0.1 if abs(mean_oos) > 1e-10 else slope < -0.01

        # Parameter stability: CV of each parameter across windows
        param_stability = {}
        for key in self._param_keys:
            values = []
            for w in results:
                v = w.best_params.get(key)
                if isinstance(v, (int, float)):
                    values.append(float(v))
            if len(values) >= 2:
                mean_v = np.mean(values)
                std_v = np.std(values)
                param_stability[key] = float(std_v / max(abs(mean_v), 1e-10))
            else:
                param_stability[key] = 0.0

        result = WalkForwardResult(
            windows=results,
            oos_equity=oos_equity,
            oos_returns=oos_returns,
            mean_oos_metric=mean_oos,
            median_oos_metric=median_oos,
            std_oos_metric=std_oos,
            mean_is_metric=mean_is,
            is_degradation=is_degradation,
            degradation_slope=slope,
            efficiency_ratio=efficiency,
            param_stability=param_stability,
        )

        logger.info(result.summary())
        return result
