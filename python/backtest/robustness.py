"""Robustness analysis: Monte Carlo, stress testing, and scenario analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/processed")

# Historical crisis scenarios for stress testing
HISTORICAL_SCENARIOS = {
    "2008_Financial_Crisis": ("2008-09-01", "2009-03-31"),
    "2020_COVID_Crash": ("2020-02-19", "2020-03-23"),
    "2022_Rate_Hikes": ("2022-01-01", "2022-10-31"),
    "Dot_Com_Bust": ("2000-03-01", "2002-10-01"),
    "Flash_Crash_2010": ("2010-05-06", "2010-05-06"),
    "2022_Tightening": ("2022-01-01", "2022-12-31"),
    "2023_Recovery": ("2023-01-01", "2023-12-31"),
    "2024_Bull_Market": ("2024-01-01", "2024-12-31"),
}


def compute_sharpe(
    returns: pd.Series | np.ndarray,
    periods_per_year: float = 252 / 5,
    risk_free_rate: float = 0.05,
) -> float:
    """Centralized Sharpe ratio calculation (geometric annualization, rf-adjusted).

    M-SHARPE3 fix: all modules should use this single implementation to avoid
    the three divergent formulas previously scattered across run.py,
    robustness.py, and regime_analysis.py.

    Args:
        returns: Period returns (e.g. 5-day rebalance returns).
        periods_per_year: Number of return periods per year (default 252/5 for 5-day).
        risk_free_rate: Annual risk-free rate (default 5%).

    Returns:
        Annualized Sharpe ratio (geometric return minus rf, divided by annualized vol).
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    return float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0


def compute_metrics(
    returns: pd.Series,
    periods_per_year: float = 252 / 5,
    risk_free_rate: float = 0.05,
) -> dict:
    """Compute basic performance metrics for a return series.

    M-SHARPE3 fix: delegates Sharpe calculation to ``compute_sharpe()`` which
    uses geometric annualization and subtracts the risk-free rate, consistent
    with run.py and risk.py.
    """
    if len(returns) == 0:
        return {
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    ann_return = float((1 + returns.mean()) ** periods_per_year - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = compute_sharpe(returns, periods_per_year, risk_free_rate)

    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()

    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(max_dd),
    }


class StressTester:
    """
    Comprehensive stress testing for portfolios.

    Includes historical scenarios, hypothetical shocks, and Monte Carlo stress tests.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[pd.Series] = None,
    ) -> None:
        """
        Args:
            returns: DataFrame with asset returns (columns = assets)
            weights: Portfolio weights (default: equal weight)
        """
        self.returns = returns
        self.tickers = list(returns.columns)

        if weights is not None:
            self.weights = weights.reindex(self.tickers).fillna(0.0)
        else:
            self.weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)

        self.portfolio_returns = (returns * self.weights).sum(axis=1)

    def historical_stress_test(
        self,
        scenario_name: str,
        scenario_returns: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Test portfolio against historical crisis period.

        Args:
            scenario_name: Name of scenario from HISTORICAL_SCENARIOS
            scenario_returns: Optional DataFrame with historical returns for the period

        Returns:
            Dictionary with stress test results
        """
        if scenario_name not in HISTORICAL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        start, end = HISTORICAL_SCENARIOS[scenario_name]

        if scenario_returns is not None:
            # Use provided returns
            period_returns = scenario_returns.loc[start:end]
            port_returns = (period_returns * self.weights).sum(axis=1)
        else:
            # Use portfolio returns
            port_returns = self.portfolio_returns.loc[start:end]

        if len(port_returns) == 0:
            return {
                "scenario": scenario_name,
                "period": f"{start} to {end}",
                "error": "No data available for period",
            }

        total_return = (1 + port_returns).prod() - 1
        max_dd = self._calculate_max_dd(port_returns)
        worst_day = port_returns.min()
        volatility = port_returns.std() * np.sqrt(252)

        return {
            "scenario": scenario_name,
            "period": f"{start} to {end}",
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "max_drawdown": float(max_dd),
            "max_drawdown_pct": float(max_dd * 100),
            "worst_day": float(worst_day),
            "worst_day_pct": float(worst_day * 100),
            "volatility": float(volatility),
            "volatility_pct": float(volatility * 100),
            "num_days": len(port_returns),
        }

    def hypothetical_shock_test(
        self,
        shocks: Dict[str, float],
        shock_type: str = "absolute",
    ) -> Dict:
        """
        Test portfolio against hypothetical return shocks.

        Args:
            shocks: Dict of {asset: shock_return} (e.g., {"AAPL": -0.20} for -20%)
            shock_type: "absolute" for total returns, "excess" for returns vs expected

        Returns:
            Dictionary with shock impact analysis
        """
        total_impact = 0.0
        asset_impacts = {}

        for asset, shock in shocks.items():
            if asset in self.weights.index:
                impact = self.weights[asset] * shock
                total_impact += impact
                asset_impacts[asset] = {
                    "weight": float(self.weights[asset]),
                    "shock": float(shock),
                    "shock_pct": float(shock * 100),
                    "impact": float(impact),
                    "impact_pct": float(impact * 100),
                }

        return {
            "shock_type": shock_type,
            "total_portfolio_impact": float(total_impact),
            "total_portfolio_impact_pct": float(total_impact * 100),
            "asset_impacts": asset_impacts,
            "num_assets_affected": len(asset_impacts),
        }

    def monte_carlo_stress(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 21,
        vol_multiplier: float = 2.0,
        mean_adjustment: float = 0.0,
        random_seed: int = 42,
    ) -> Dict:
        """
        Monte Carlo stress test with elevated volatility.

        Args:
            n_simulations: Number of Monte Carlo simulations
            horizon_days: Number of days per simulation
            vol_multiplier: Multiplier for volatility (e.g., 2.0 = 2x normal vol)
            mean_adjustment: Adjustment to mean return (e.g., -0.001 for -10bp/day)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with MC stress test results
        """
        rng = np.random.default_rng(random_seed)

        # Adjust parameters for stress
        mean = self.portfolio_returns.mean() + mean_adjustment
        vol = self.portfolio_returns.std() * vol_multiplier

        # Run simulations
        simulations = rng.normal(mean, vol, (n_simulations, horizon_days))
        total_returns = (1 + simulations).prod(axis=1) - 1

        return {
            "n_simulations": n_simulations,
            "horizon_days": horizon_days,
            "vol_multiplier": vol_multiplier,
            "mean_adjustment": mean_adjustment,
            "expected_return": float(total_returns.mean()),
            "expected_return_pct": float(total_returns.mean() * 100),
            "std_dev": float(total_returns.std()),
            "std_dev_pct": float(total_returns.std() * 100),
            "var_95": float(-np.percentile(total_returns, 5)),
            "var_95_pct": float(-np.percentile(total_returns, 5) * 100),
            "var_99": float(-np.percentile(total_returns, 1)),
            "var_99_pct": float(-np.percentile(total_returns, 1) * 100),
            "worst_case": float(-total_returns.min()),
            "worst_case_pct": float(-total_returns.min() * 100),
            "prob_loss": float((total_returns < 0).mean()),
            "prob_loss_pct": float((total_returns < 0).mean() * 100),
            "prob_10pct_loss": float((total_returns < -0.10).mean()),
            "prob_10pct_loss_pct": float((total_returns < -0.10).mean() * 100),
            "prob_20pct_loss": float((total_returns < -0.20).mean()),
            "prob_20pct_loss_pct": float((total_returns < -0.20).mean() * 100),
        }

    def correlation_breakdown(
        self,
        threshold_percentile: float = 10,
    ) -> Dict:
        """
        Analyze correlation breakdown during stress periods.

        Args:
            threshold_percentile: Percentile for stress threshold

        Returns:
            Dictionary with correlation analysis
        """
        threshold = np.percentile(self.portfolio_returns, threshold_percentile)
        stress_mask = self.portfolio_returns <= threshold

        normal_corr = self.returns.corr()
        stress_corr = self.returns[stress_mask].corr()

        # Calculate correlation increase
        corr_diff = stress_corr - normal_corr

        # Average correlation in each regime
        avg_normal_corr = normal_corr.values[np.triu_indices_from(normal_corr.values, k=1)].mean()
        avg_stress_corr = stress_corr.values[np.triu_indices_from(stress_corr.values, k=1)].mean()

        # Find pairs with largest correlation increase
        corr_diff_flat = corr_diff.where(np.triu(np.ones(corr_diff.shape), k=1).astype(bool))
        max_increase = corr_diff_flat.stack().nlargest(5)

        return {
            "threshold_return": float(threshold),
            "threshold_return_pct": float(threshold * 100),
            "stress_periods": int(stress_mask.sum()),
            "avg_correlation_normal": float(avg_normal_corr),
            "avg_correlation_stress": float(avg_stress_corr),
            "correlation_increase": float(avg_stress_corr - avg_normal_corr),
            "max_correlation_increases": {
                f"{idx[0]}-{idx[1]}": float(val) for idx, val in max_increase.items()
            },
        }

    def run_all_stress_tests(
        self,
        hypothetical_shocks: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict:
        """
        Run comprehensive stress testing suite.

        Args:
            hypothetical_shocks: Optional dict of named shock scenarios

        Returns:
            Dictionary with all stress test results
        """
        results = {
            "historical_scenarios": {},
            "hypothetical_shocks": {},
            "monte_carlo_stress": {},
            "correlation_breakdown": self.correlation_breakdown(),
        }

        # Historical scenarios
        for scenario in ["2008_Financial_Crisis", "2020_COVID_Crash", "2022_Rate_Hikes"]:
            try:
                results["historical_scenarios"][scenario] = self.historical_stress_test(scenario)
            except Exception as e:
                logger.warning(f"Failed to run {scenario}: {e}")
                results["historical_scenarios"][scenario] = {"error": str(e)}

        # Hypothetical shocks
        default_shocks = {
            "Tech_Crash": {"AAPL": -0.30, "MSFT": -0.25, "GOOGL": -0.35},
            "Financial_Crisis": {"JPM": -0.40, "BAC": -0.45, "GS": -0.35},
            "Market_Crash": {"SPY": -0.20, "QQQ": -0.25},
        }

        shocks_to_test = hypothetical_shocks or default_shocks
        for name, shocks in shocks_to_test.items():
            results["hypothetical_shocks"][name] = self.hypothetical_shock_test(shocks)

        # Monte Carlo stress tests
        results["monte_carlo_stress"]["moderate"] = self.monte_carlo_stress(
            vol_multiplier=1.5, horizon_days=21
        )
        results["monte_carlo_stress"]["severe"] = self.monte_carlo_stress(
            vol_multiplier=2.5, horizon_days=21
        )
        results["monte_carlo_stress"]["extreme"] = self.monte_carlo_stress(
            vol_multiplier=3.0, horizon_days=63
        )

        return results

    @staticmethod
    def _calculate_max_dd(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        return float(drawdowns.min())


def monte_carlo_resampling(
    returns: pd.Series, n_simulations: int = 1000, periods_per_year: float = 252 / 5
) -> dict:
    """Perform bootstrap resampling of the return series to estimate metric distributions."""
    n_obs = len(returns)
    simulated_sharpes = []
    simulated_returns = []
    simulated_dds = []

    returns_arr = returns.values
    rng = np.random.default_rng(42)

    # M-BOOTSTRAP fix: use stationary block bootstrap to preserve
    # autocorrelation in the return series.  Average block length is set
    # to the cube root of n_obs (standard rule-of-thumb for stationary
    # bootstrap), with geometric random block lengths.
    avg_block_len = max(int(round(n_obs ** (1 / 3))), 2)

    for _ in range(n_simulations):
        # Stationary block bootstrap: randomly sample blocks of geometric
        # random length, wrapping around the end of the series.
        sim_rets = np.empty(n_obs)
        i = 0
        while i < n_obs:
            start = rng.integers(0, n_obs)
            # Geometric random block length (mean = avg_block_len)
            block_len = rng.geometric(1.0 / avg_block_len)
            block_len = min(block_len, n_obs - i)  # don't exceed remaining
            for j in range(block_len):
                sim_rets[i] = returns_arr[(start + j) % n_obs]
                i += 1
                if i >= n_obs:
                    break

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

        if len(period_returns) > 5:  # Need at least a few points
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
        f"Full Period Sharpe: {full_metrics['sharpe_ratio']:.3f} | "
        f"Max DD: {full_metrics['max_drawdown']:.2%}"
    )
    logger.info("\nMonte Carlo Resampling (90% Confidence Interval):")
    logger.info(
        f"  Sharpe Ratio: [{mc_results['sharpe_ratio']['5th_percentile']:.3f}, "
        f"{mc_results['sharpe_ratio']['95th_percentile']:.3f}]"
    )
    logger.info(
        f"  Ann Return:   [{mc_results['ann_return']['5th_percentile']:.2%}, "
        f"{mc_results['ann_return']['95th_percentile']:.2%}]"
    )
    logger.info(
        f"  Max Drawdown: [{mc_results['max_drawdown']['5th_percentile']:.2%}, "
        f"{mc_results['max_drawdown']['95th_percentile']:.2%}]"
    )

    logger.info("\nRegime Stress Tests (Sharpe / Return / MaxDD):")
    for regime, metrics in regime_results.items():
        if metrics:
            logger.info(
                f"  {regime:22}: {metrics['sharpe_ratio']:>5.2f} / "
                f"{metrics['ann_return']:>6.2%} / {metrics['max_drawdown']:>6.2%}"
            )
        else:
            logger.info(f"  {regime:22}: Insufficient data")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_robustness_analysis()
