"""Ornstein-Uhlenbeck parameter estimation and mean-reversion signals.

Implements tools for statistical arbitrage based on the OU process:
  1. MLE and OLS estimation of OU parameters (kappa, mu, sigma).
  2. Kalman filter for dynamic hedge ratio estimation.
  3. Optimal entry/exit thresholds via expected P&L.
  4. Multi-asset spread construction via PCA eigenvectors.
  5. Half-life and mean-reversion speed metrics.

The Ornstein-Uhlenbeck process: dX = kappa * (mu - X) dt + sigma dW
  - kappa: speed of mean reversion (higher = faster reversion)
  - mu: long-run mean
  - sigma: volatility of the process

Usage::

    estimator = OUEstimator()
    params = estimator.fit(spread_series)
    # params = OUParams(kappa=12.5, mu=0.001, sigma=0.03, half_life=20.2)

    kf = KalmanHedgeRatio()
    betas = kf.filter(y=price_a, x=price_b)

References:
  - Ornstein & Uhlenbeck (1930), "On the Theory of Brownian Motion"
  - Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equity Market"
  - de Prado (2018), Ch. 2 — "Financial Data Structures"
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OU Parameter Estimation
# ---------------------------------------------------------------------------


@dataclass
class OUParams:
    """Estimated Ornstein-Uhlenbeck parameters."""

    kappa: float  # Mean-reversion speed
    mu: float  # Long-run mean
    sigma: float  # Process volatility
    half_life: float  # Days to revert halfway
    log_likelihood: float  # MLE log-likelihood (if available)

    @property
    def is_mean_reverting(self) -> bool:
        return self.kappa > 0 and np.isfinite(self.half_life)

    @property
    def mean_reversion_speed(self) -> str:
        if self.half_life < 5:
            return "very_fast"
        elif self.half_life < 15:
            return "fast"
        elif self.half_life < 30:
            return "moderate"
        elif self.half_life < 60:
            return "slow"
        else:
            return "very_slow"


class OUEstimator:
    """Estimate Ornstein-Uhlenbeck parameters from a time series.

    Supports two methods:
      - OLS: Simple regression of d(X) on X_{t-1}.
      - MLE: Maximum likelihood for discrete-time OU.

    Parameters
    ----------
    dt : float
        Time step (1.0 = daily).
    method : str
        'ols' or 'mle'.
    """

    def __init__(self, dt: float = 1.0, method: str = "ols"):
        self.dt = dt
        self.method = method

    def fit(self, series: np.ndarray) -> OUParams:
        """Estimate OU parameters from a time series.

        Parameters
        ----------
        series : np.ndarray
            Spread or price series.

        Returns
        -------
        OUParams
        """
        series = np.asarray(series, dtype=float).ravel()
        if len(series) < 10:
            return OUParams(kappa=0.0, mu=0.0, sigma=0.0, half_life=float("inf"), log_likelihood=0.0)

        if self.method == "mle":
            return self._fit_mle(series)
        return self._fit_ols(series)

    def _fit_ols(self, series: np.ndarray) -> OUParams:
        """OLS estimation: d(X) = a + b * X_{t-1}."""
        dx = np.diff(series)
        x_lag = series[:-1]

        X = np.column_stack([np.ones(len(x_lag)), x_lag])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, dx, rcond=None)
        except np.linalg.LinAlgError:
            return OUParams(0.0, 0.0, 0.0, float("inf"), 0.0)

        a, b = beta[0], beta[1]

        if b >= 0:
            return OUParams(0.0, float(np.mean(series)), float(np.std(dx)), float("inf"), 0.0)

        kappa = -b / self.dt
        mu = -a / b
        residuals = dx - X @ beta
        sigma_e = float(np.std(residuals, ddof=2))
        sigma = sigma_e / np.sqrt(self.dt)
        half_life = np.log(2) / kappa

        return OUParams(
            kappa=float(kappa),
            mu=float(mu),
            sigma=sigma,
            half_life=float(half_life),
            log_likelihood=0.0,
        )

    def _fit_mle(self, series: np.ndarray) -> OUParams:
        """Maximum likelihood estimation for discrete OU.

        The transition distribution is:
          X_{t+1} | X_t ~ N(mu + (X_t - mu) * exp(-kappa*dt), sigma^2/(2*kappa) * (1 - exp(-2*kappa*dt)))
        """
        n = len(series) - 1
        x = series[:-1]
        y = series[1:]

        # OLS of y on x to get initial estimates
        sx = np.sum(x)
        sy = np.sum(y)
        sxx = np.sum(x * x)
        sxy = np.sum(x * y)
        syy = np.sum(y * y)

        denom = n * sxx - sx * sx
        if abs(denom) < 1e-16:
            return OUParams(0.0, float(np.mean(series)), float(np.std(np.diff(series))), float("inf"), 0.0)

        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n

        if b <= 0 or b >= 1:
            # Not mean-reverting in MLE framework
            mu_est = float(np.mean(series))
            sig_est = float(np.std(np.diff(series)))
            return OUParams(0.0, mu_est, sig_est, float("inf"), 0.0)

        kappa = -np.log(b) / self.dt
        mu = a / (1 - b)

        # Residual variance
        predicted = a + b * x
        residuals = y - predicted
        sigma_e2 = float(np.mean(residuals**2))

        # Convert to continuous-time sigma
        sigma2 = sigma_e2 * 2 * kappa / (1 - np.exp(-2 * kappa * self.dt))
        sigma = np.sqrt(max(sigma2, 0))

        half_life = np.log(2) / kappa

        # Log-likelihood
        var_trans = sigma2 / (2 * kappa) * (1 - np.exp(-2 * kappa * self.dt))
        ll = -0.5 * n * np.log(2 * np.pi * max(var_trans, 1e-16)) - 0.5 * np.sum(residuals**2) / max(var_trans, 1e-16)

        return OUParams(
            kappa=float(kappa),
            mu=float(mu),
            sigma=float(sigma),
            half_life=float(half_life),
            log_likelihood=float(ll),
        )


# ---------------------------------------------------------------------------
# Kalman Filter for Dynamic Hedge Ratio
# ---------------------------------------------------------------------------


@dataclass
class KalmanState:
    """Kalman filter state."""

    beta: float  # Current hedge ratio estimate
    P: float  # Estimation variance
    R: float  # Measurement noise variance (estimated)


class KalmanHedgeRatio:
    """Dynamic hedge ratio estimation via Kalman filter.

    Models: y_t = beta_t * x_t + epsilon_t
    State: beta_t = beta_{t-1} + eta_t

    Parameters
    ----------
    delta : float
        State transition noise (controls how fast beta adapts).
        Higher = more responsive, lower = more stable.
    initial_beta : float
        Starting hedge ratio.
    initial_var : float
        Initial estimation variance.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        initial_beta: float = 1.0,
        initial_var: float = 1.0,
    ):
        self.delta = delta
        self.initial_beta = initial_beta
        self.initial_var = initial_var

    def filter(
        self, y: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter to estimate dynamic hedge ratio.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (price of asset A).
        x : np.ndarray
            Independent variable (price of asset B).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (betas, spreads) — time series of hedge ratios and spreads.
        """
        n = len(y)
        betas = np.zeros(n)
        spreads = np.zeros(n)
        P = self.initial_var
        beta = self.initial_beta
        R = 1.0  # Initial measurement noise estimate

        for t in range(n):
            # Prediction
            P_pred = P + self.delta

            # Observation
            x_t = x[t]
            y_pred = beta * x_t
            e = y[t] - y_pred  # Innovation

            # Innovation variance
            S = x_t**2 * P_pred + R
            if abs(S) < 1e-16:
                betas[t] = beta
                spreads[t] = e
                continue

            # Kalman gain
            K = P_pred * x_t / S

            # Update
            beta = beta + K * e
            P = (1 - K * x_t) * P_pred

            # Estimate R (measurement noise)
            R = max(0.9 * R + 0.1 * e**2, 1e-8)

            betas[t] = beta
            spreads[t] = e

        return betas, spreads


# ---------------------------------------------------------------------------
# Optimal Entry/Exit Thresholds
# ---------------------------------------------------------------------------


@dataclass
class OptimalThresholds:
    """Optimal trading thresholds for OU process."""

    entry_long: float  # Enter long spread below this z-score
    exit_long: float  # Exit long above this z-score
    entry_short: float  # Enter short above this z-score
    exit_short: float  # Exit short below this z-score
    expected_pnl_per_trade: float
    expected_trades_per_year: float


def optimal_ou_thresholds(
    params: OUParams,
    trading_cost: float = 0.001,
    holding_cost: float = 0.0,
) -> OptimalThresholds:
    """Compute optimal entry/exit thresholds for OU process.

    Uses the expected profit calculation for symmetric thresholds:
    E[PnL] = sigma/sqrt(2*kappa) * (entry_z - exit_z) * f(kappa, entry, exit) - 2*cost

    Parameters
    ----------
    params : OUParams
    trading_cost : float
        Round-trip trading cost (in spread units).
    holding_cost : float
        Per-period holding cost.

    Returns
    -------
    OptimalThresholds
    """
    if not params.is_mean_reverting or params.kappa < 0.001:
        return OptimalThresholds(
            entry_long=-2.0, exit_long=0.0,
            entry_short=2.0, exit_short=0.0,
            expected_pnl_per_trade=0.0,
            expected_trades_per_year=0.0,
        )

    # Equilibrium standard deviation of OU process
    sigma_eq = params.sigma / np.sqrt(2 * params.kappa)

    # Search for optimal thresholds
    best_pnl_rate = -float("inf")
    best_entry = 2.0
    best_exit = 0.5

    for entry_z in np.arange(1.0, 3.5, 0.25):
        for exit_z in np.arange(0.0, entry_z, 0.25):
            # Expected profit per trade (in sigma_eq units)
            profit = sigma_eq * (entry_z - exit_z)

            # Expected time per trade (approximate for OU)
            # Time to revert from entry_z to exit_z
            if params.kappa > 0:
                t_trade = (entry_z - exit_z) / (params.kappa * (entry_z + exit_z) / 2)
            else:
                t_trade = float("inf")

            net_pnl = profit - trading_cost - holding_cost * t_trade

            if t_trade > 0:
                pnl_rate = net_pnl / t_trade
            else:
                pnl_rate = -float("inf")

            if pnl_rate > best_pnl_rate:
                best_pnl_rate = pnl_rate
                best_entry = entry_z
                best_exit = exit_z

    # Expected trades per year
    # Mean first passage time from 0 to entry_z for OU ≈ entry_z / (kappa * sigma_eq)
    t_to_entry = best_entry * sigma_eq / max(params.sigma, 1e-12)
    t_round_trip = t_to_entry + (best_entry - best_exit) / max(params.kappa * best_entry * sigma_eq, 1e-12)
    trades_per_year = 252 / max(t_round_trip, 1) if t_round_trip > 0 else 0

    exp_pnl = sigma_eq * (best_entry - best_exit) - trading_cost

    return OptimalThresholds(
        entry_long=-best_entry,
        exit_long=-best_exit,
        entry_short=best_entry,
        exit_short=best_exit,
        expected_pnl_per_trade=float(exp_pnl),
        expected_trades_per_year=float(trades_per_year),
    )


# ---------------------------------------------------------------------------
# PCA Spread Construction
# ---------------------------------------------------------------------------


def pca_spreads(
    prices: np.ndarray,
    n_spreads: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct mean-reverting spreads from PCA eigenvectors.

    Uses the smallest eigenvalues (most mean-reverting components).

    Parameters
    ----------
    prices : np.ndarray (T, N)
        Price matrix for N assets.
    n_spreads : int
        Number of spreads to construct.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (spreads, weights) where:
        - spreads: (T, n_spreads) spread values
        - weights: (n_spreads, N) portfolio weights for each spread
    """
    T, N = prices.shape
    n_spreads = min(n_spreads, N)

    # Log prices
    log_prices = np.log(np.maximum(prices, 1e-10))

    # Demean
    X = log_prices - log_prices.mean(axis=0, keepdims=True)

    # Eigendecomposition
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Smallest eigenvalues → most mean-reverting
    idx = np.argsort(eigenvalues)[:n_spreads]
    weights = eigenvectors[:, idx].T  # (n_spreads, N)

    # Construct spreads
    spreads = log_prices @ weights.T  # (T, n_spreads)

    return spreads, weights


# ---------------------------------------------------------------------------
# Z-Score Signal Generation
# ---------------------------------------------------------------------------


def zscore_signal(
    spread: np.ndarray,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> np.ndarray:
    """Generate trading signals from z-score of spread.

    Parameters
    ----------
    spread : np.ndarray
    lookback : int
        Rolling window for mean/std.
    entry_z : float
        Entry threshold.
    exit_z : float
        Exit threshold.

    Returns
    -------
    np.ndarray
        Signal: +1 (long spread), -1 (short spread), 0 (flat).
    """
    n = len(spread)
    signals = np.zeros(n)
    position = 0

    for i in range(lookback, n):
        window = spread[i - lookback : i]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std < 1e-12:
            continue

        z = (spread[i] - mean) / std

        if position == 0:
            if z < -entry_z:
                position = 1  # Long spread (cheap)
            elif z > entry_z:
                position = -1  # Short spread (expensive)
        elif position == 1:
            if z > -exit_z:
                position = 0
        elif position == -1:
            if z < exit_z:
                position = 0

        signals[i] = position

    return signals
