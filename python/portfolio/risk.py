"""Risk metrics: VaR, CVaR, drawdown, concentration, and advanced risk measures."""

import numpy as np
import pandas as pd
from scipy import stats


class RiskEngine:
    """Portfolio risk calculation engine with comprehensive metrics."""

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        rf_rate: float = 0.02,
        benchmark_returns: pd.Series = None,
    ):
        """
        Args:
            returns: DataFrame with asset returns (columns = assets)
            weights: Portfolio weights
            rf_rate: Annual risk-free rate (default: 2%)
            benchmark_returns: Optional benchmark returns for Information Ratio
        """
        self.returns = returns
        self.weights = weights
        self.rf_rate = rf_rate
        self.benchmark_returns = benchmark_returns
        self.portfolio_returns = (returns * weights).sum(axis=1)
        self.ann_factor = 252  # Trading days per year

    # =========================================================================
    # VaR Methods
    # =========================================================================

    def var_parametric(self, confidence: float = 0.95) -> float:
        """Parametric VaR assuming normal distribution."""
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        return stats.norm.ppf(1 - confidence, mu, sigma)

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical simulation VaR."""
        return float(np.percentile(self.portfolio_returns, (1 - confidence) * 100))

    def var_cornish_fisher(self, confidence: float = 0.95) -> float:
        """
        VaR with Cornish-Fisher expansion for non-normality.
        Accounts for skewness and kurtosis (fat tails).
        """
        z = stats.norm.ppf(confidence)
        s = stats.skew(self.portfolio_returns)  # Skewness
        k = stats.kurtosis(self.portfolio_returns)  # Excess kurtosis

        # Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36

        return -(self.portfolio_returns.mean() + z_cf * self.portfolio_returns.std())

    def cvar_historical(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) via historical simulation."""
        var = self.var_historical(confidence)
        tail = self.portfolio_returns[self.portfolio_returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    # =========================================================================
    # Volatility Metrics
    # =========================================================================

    def volatility(self, annualized: bool = True) -> float:
        """Standard deviation of returns."""
        vol = self.portfolio_returns.std()
        if annualized:
            vol *= np.sqrt(self.ann_factor)
        return vol

    def downside_deviation(self, threshold: float = 0, annualized: bool = True) -> float:
        """Standard deviation of returns below threshold."""
        downside = self.portfolio_returns[self.portfolio_returns < threshold]
        if len(downside) == 0:
            return 0.0
        dd = downside.std()
        if annualized:
            dd *= np.sqrt(self.ann_factor)
        return dd

    def beta(self, market_returns: pd.Series = None) -> float:
        """Beta relative to market or benchmark."""
        if market_returns is None:
            market_returns = self.benchmark_returns

        if market_returns is None:
            return np.nan

        aligned = pd.concat([self.portfolio_returns, market_returns], axis=1).dropna()
        if len(aligned) < 2:
            return np.nan

        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

    # =========================================================================
    # Drawdown Analysis
    # =========================================================================

    def drawdowns(self) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max

    def max_drawdown(self) -> float:
        """Maximum drawdown of the portfolio."""
        return float(self.drawdowns().min())

    def avg_drawdown(self) -> float:
        """Average drawdown (excluding zeros)."""
        dd = self.drawdowns()
        return dd[dd < 0].mean() if (dd < 0).any() else 0.0

    def drawdown_duration(self) -> dict:
        """Drawdown duration statistics."""
        dd = self.drawdowns()
        in_drawdown = dd < 0

        durations = []
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return {
            "max_duration": max(durations) if durations else 0,
            "avg_duration": np.mean(durations) if durations else 0.0,
            "current_duration": current_duration,
            "num_drawdowns": len(durations),
        }

    # =========================================================================
    # Risk-Adjusted Returns
    # =========================================================================

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio."""
        excess_return = self.portfolio_returns.mean() * self.ann_factor - self.rf_rate
        vol = self.volatility(annualized=True)
        return excess_return / vol if vol > 0 else 0.0

    def sortino_ratio(self, threshold: float = 0) -> float:
        """
        Sortino ratio using downside deviation only.
        Better than Sharpe for asymmetric returns.
        """
        excess_return = self.portfolio_returns.mean() * self.ann_factor - self.rf_rate
        dd = self.downside_deviation(threshold=threshold, annualized=True)
        return excess_return / dd if dd > 0 else 0.0

    def calmar_ratio(self) -> float:
        """
        Calmar ratio (annual return / max drawdown).
        Focuses on capital preservation.
        """
        annual_return = self.portfolio_returns.mean() * self.ann_factor
        max_dd = abs(self.max_drawdown())
        return annual_return / max_dd if max_dd > 0 else 0.0

    def omega_ratio(self, threshold: float = 0) -> float:
        """
        Omega ratio (gains vs losses).
        Considers entire return distribution.
        """
        returns_above = self.portfolio_returns[self.portfolio_returns > threshold] - threshold
        returns_below = threshold - self.portfolio_returns[self.portfolio_returns <= threshold]

        if returns_below.sum() == 0:
            return np.inf

        return returns_above.sum() / returns_below.sum()

    def information_ratio(self, benchmark_returns: pd.Series = None) -> float:
        """Information ratio vs benchmark (alpha per unit tracking error)."""
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns

        if benchmark_returns is None:
            return np.nan

        active_returns = self.portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.ann_factor)
        active_return = active_returns.mean() * self.ann_factor

        return active_return / tracking_error if tracking_error > 0 else 0.0

    # =========================================================================
    # Rolling Metrics
    # =========================================================================

    def rolling_sharpe(self, window: int = 63, risk_free: float = None) -> pd.Series:
        """Rolling Sharpe ratio (default: 63 days = ~3 months)."""
        if risk_free is None:
            risk_free = self.rf_rate

        excess = self.portfolio_returns - risk_free / self.ann_factor
        rolling_mean = excess.rolling(window).mean() * self.ann_factor
        rolling_std = excess.rolling(window).std() * np.sqrt(self.ann_factor)
        return rolling_mean / rolling_std

    def rolling_var(self, window: int = 63, confidence: float = 0.95) -> pd.Series:
        """Rolling historical VaR."""
        return self.portfolio_returns.rolling(window).apply(
            lambda x: -np.percentile(x, (1 - confidence) * 100),
            raw=True,
        )

    def rolling_max_drawdown(self, window: int = 63) -> pd.Series:
        """Rolling maximum drawdown."""

        def max_dd(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdowns = (cumulative - running_max) / running_max
            return drawdowns.min()

        return self.portfolio_returns.rolling(window).apply(max_dd, raw=False)

    def rolling_beta(self, market_returns: pd.Series, window: int = 63) -> pd.Series:
        """Rolling beta vs market."""

        def calc_beta(window_data):
            if len(window_data) < 2:
                return np.nan
            port_ret = window_data.iloc[:, 0]
            mkt_ret = window_data.iloc[:, 1]
            cov = np.cov(port_ret, mkt_ret)
            return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

        combined = pd.concat([self.portfolio_returns, market_returns], axis=1)
        return (
            combined.rolling(window)
            .apply(
                lambda x: calc_beta(x.to_frame()),
                raw=False,
            )
            .iloc[:, 0]
        )

    def volatility_regime(
        self, window: int = 63, low_threshold: float = 0.10, high_threshold: float = 0.20
    ) -> pd.Series:
        """Classify volatility regime (low/normal/high)."""
        vol = self.portfolio_returns.rolling(window).std() * np.sqrt(self.ann_factor)

        def classify(v):
            if pd.isna(v):
                return np.nan
            if v < low_threshold:
                return "low"
            elif v > high_threshold:
                return "high"
            else:
                return "normal"

        return vol.apply(classify)

    # =========================================================================
    # Concentration
    # =========================================================================

    def concentration(self) -> pd.Series:
        """Herfindahl-Hirschman Index and effective number of bets."""
        hhi = (self.weights**2).sum()
        return pd.Series({"hhi": hhi, "effective_n": 1 / hhi if hhi > 0 else np.inf})

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> dict:
        """Full risk summary with all metrics."""
        dd_stats = self.drawdown_duration()
        total_return = (1 + self.portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (self.ann_factor / len(self.portfolio_returns)) - 1

        summary_dict = {
            # Returns
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_return_pct": annual_return * 100,
            # Volatility
            "annualized_volatility": self.volatility(),
            "annualized_volatility_pct": self.volatility() * 100,
            "downside_deviation": self.downside_deviation(),
            "downside_deviation_pct": self.downside_deviation() * 100,
            # VaR & CVaR
            "var_95_parametric": self.var_parametric(0.95),
            "var_95_parametric_pct": self.var_parametric(0.95) * 100,
            "var_95_historical": self.var_historical(0.95),
            "var_95_historical_pct": self.var_historical(0.95) * 100,
            "var_99_historical": self.var_historical(0.99),
            "var_99_historical_pct": self.var_historical(0.99) * 100,
            "var_95_cornish_fisher": self.var_cornish_fisher(0.95),
            "var_95_cornish_fisher_pct": self.var_cornish_fisher(0.95) * 100,
            "cvar_95": self.cvar_historical(0.95),
            "cvar_95_pct": self.cvar_historical(0.95) * 100,
            # Drawdowns
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_pct": self.max_drawdown() * 100,
            "avg_drawdown": self.avg_drawdown(),
            "avg_drawdown_pct": self.avg_drawdown() * 100,
            "max_drawdown_duration": dd_stats["max_duration"],
            "avg_drawdown_duration": dd_stats["avg_duration"],
            "num_drawdowns": dd_stats["num_drawdowns"],
            # Risk-Adjusted Returns
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "omega_ratio": self.omega_ratio(),
            # Concentration
            "hhi": self.concentration()["hhi"],
            "effective_n": self.concentration()["effective_n"],
            # Distribution Statistics
            "skewness": stats.skew(self.portfolio_returns),
            "kurtosis": stats.kurtosis(self.portfolio_returns),
        }

        # Add Information Ratio if benchmark available
        if self.benchmark_returns is not None:
            summary_dict["information_ratio"] = self.information_ratio()
            summary_dict["beta"] = self.beta()

        return summary_dict
