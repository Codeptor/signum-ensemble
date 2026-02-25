import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/processed")


def compute_metrics(returns: pd.Series, periods_per_year: float = 252 / 5) -> dict:
    """Compute basic performance metrics for a return series."""
    if len(returns) == 0:
        return {"ann_return": 0.0, "ann_volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()

    return {
        "ann_return": float(ann_return),
        "ann_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def monte_carlo_resampling(
    returns: pd.Series, n_simulations: int = 1000, periods_per_year: float = 252 / 5
) -> dict:
    """Perform bootstrap resampling of the return series to estimate metric distributions."""
    n_obs = len(returns)
    simulated_sharpes = []
    simulated_returns = []
    simulated_dds = []

    returns_arr = returns.values
    np.random.seed(42)

    for _ in range(n_simulations):
        # Sample with replacement
        idx = np.random.randint(0, n_obs, n_obs)
        sim_rets = returns_arr[idx]

        sim_series = pd.Series(sim_rets)
        metrics = compute_metrics(sim_series, periods_per_year)

        simulated_sharpes.append(metrics["sharpe_ratio"])
        simulated_returns.append(metrics["ann_return"])
        simulated_dds.append(metrics["max_drawdown"])

    return {
        "sharpe_ratio": {
            "mean": float(np.mean(simulated_sharpes)),
            "5th_percentile": float(np.percentile(simulated_sharpes, 5)),
            "95th_percentile": float(np.percentile(simulated_sharpes, 95)),
        },
        "ann_return": {
            "mean": float(np.mean(simulated_returns)),
            "5th_percentile": float(np.percentile(simulated_returns, 5)),
            "95th_percentile": float(np.percentile(simulated_returns, 95)),
        },
        "max_drawdown": {
            "mean": float(np.mean(simulated_dds)),
            "5th_percentile": float(np.percentile(simulated_dds, 5)),
            "95th_percentile": float(np.percentile(simulated_dds, 95)),
        },
    }


def regime_stress_tests(returns: pd.Series, periods_per_year: float = 252 / 5) -> dict:
    """Calculate performance metrics across predefined historical regimes."""
    # Ensure datetime index
    returns.index = pd.to_datetime(returns.index)

    regimes = {
        "2022_Tightening_Cycle": ("2022-01-01", "2022-12-31"),
        "2023_Recovery": ("2023-01-01", "2023-12-31"),
        "2024_Bull_Market": ("2024-01-01", "2024-12-31"),
        "2025_Present": ("2025-01-01", "2026-12-31"),
    }

    results = {}
    for regime_name, (start_date, end_date) in regimes.items():
        mask = (returns.index >= start_date) & (returns.index <= end_date)
        period_returns = returns[mask]

        if len(period_returns) > 5:  # Need at least a few points to calculate meaningful metrics
            results[regime_name] = compute_metrics(period_returns, periods_per_year)
        else:
            results[regime_name] = None

    return results


def run_robustness_analysis(returns_file: Path = RESULTS_DIR / "backtest_returns.parquet") -> dict:
    """Run full robustness analysis on a backtest return series."""
    if not returns_file.exists():
        logger.error(f"Returns file not found: {returns_file}")
        return {}

    df = pd.read_parquet(returns_file)
    if "return" not in df.columns:
        logger.error("Returns DataFrame must have a 'return' column")
        return {}

    returns = df["return"]

    # 5-day rebalancing implies ~50 periods per year
    periods_per_year = 252 / 5

    logger.info("Running Monte Carlo bootstrap resampling...")
    mc_results = monte_carlo_resampling(
        returns, n_simulations=1000, periods_per_year=periods_per_year
    )

    logger.info("Running Historical Regime Stress Tests...")
    regime_results = regime_stress_tests(returns, periods_per_year=periods_per_year)

    full_metrics = compute_metrics(returns, periods_per_year=periods_per_year)

    report = {
        "full_period": full_metrics,
        "monte_carlo_bootstrap": mc_results,
        "regime_stress_tests": regime_results,
    }

    # Save to disk
    output_path = RESULTS_DIR / "robustness_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved robustness report to {output_path}")

    # Print summary
    logger.info("\n=== Robustness Report ===")
    logger.info(
        f"Full Period Sharpe: {full_metrics['sharpe_ratio']:.3f} | Max DD: {full_metrics['max_drawdown']:.2%}"
    )
    logger.info("\nMonte Carlo Resampling (90% Confidence Interval):")
    logger.info(
        f"  Sharpe Ratio: [{mc_results['sharpe_ratio']['5th_percentile']:.3f}, {mc_results['sharpe_ratio']['95th_percentile']:.3f}]"
    )
    logger.info(
        f"  Ann Return:   [{mc_results['ann_return']['5th_percentile']:.2%}, {mc_results['ann_return']['95th_percentile']:.2%}]"
    )
    logger.info(
        f"  Max Drawdown: [{mc_results['max_drawdown']['5th_percentile']:.2%}, {mc_results['max_drawdown']['95th_percentile']:.2%}]"
    )

    logger.info("\nRegime Stress Tests (Sharpe / Return / MaxDD):")
    for regime, metrics in regime_results.items():
        if metrics:
            logger.info(
                f"  {regime:22}: {metrics['sharpe_ratio']:>5.2f} / {metrics['ann_return']:>6.2%} / {metrics['max_drawdown']:>6.2%}"
            )
        else:
            logger.info(f"  {regime:22}: Insufficient data")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_robustness_analysis()
