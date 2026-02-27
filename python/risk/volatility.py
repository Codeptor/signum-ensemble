"""Advanced volatility estimators from OHLC data.

Implements a hierarchy of efficiency-increasing estimators:
  1. Close-to-close: Standard but least efficient.
  2. Parkinson (1980): Uses high-low range, 5x more efficient.
  3. Garman-Klass (1980): Uses OHLC, 8x more efficient.
  4. Rogers-Satchell (1991): Drift-independent, handles trending markets.
  5. Yang-Zhang (2000): Most efficient OHLC estimator, handles drift + jumps.
  6. EWMA: Exponentially weighted for adaptive volatility tracking.
  7. Realized volatility from intraday returns.

Usage::

    vol = yang_zhang(open_, high, low, close, window=20)
    ewma = ewma_volatility(returns, halflife=20)
    rv = realized_volatility(intraday_returns, bars_per_day=78)

References:
  - Parkinson (1980), "The Extreme Value Method for Estimating the Variance"
  - Garman & Klass (1980), "On the Estimation of Security Price Volatilities"
  - Rogers & Satchell (1991), "Estimating Variance From High, Low and Closing Prices"
  - Yang & Zhang (2000), "Drift Independent Volatility Estimation"
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Close-to-Close
# ---------------------------------------------------------------------------


def close_to_close(
    close: np.ndarray,
    window: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Standard close-to-close volatility.

    Parameters
    ----------
    close : np.ndarray
        Closing prices.
    window : int
        Rolling window.
    annualize : int
        Trading days per year for annualization.

    Returns
    -------
    np.ndarray
        Annualized rolling volatility.
    """
    log_ret = np.diff(np.log(close))
    n = len(log_ret)
    result = np.full(n + 1, np.nan)
    for i in range(window, n + 1):
        chunk = log_ret[i - window : i]
        result[i] = np.std(chunk, ddof=1) * np.sqrt(annualize)
    return result


# ---------------------------------------------------------------------------
# Parkinson
# ---------------------------------------------------------------------------


def parkinson(
    high: np.ndarray,
    low: np.ndarray,
    window: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Parkinson range-based volatility estimator.

    sigma^2 = 1/(4*n*ln(2)) * sum(ln(H/L)^2)

    ~5x more efficient than close-to-close.
    """
    log_hl = np.log(high / np.maximum(low, 1e-10))
    rs = log_hl**2

    n = len(rs)
    result = np.full(n, np.nan)
    factor = 1 / (4 * np.log(2))

    for i in range(window, n):
        var = factor * np.mean(rs[i - window : i])
        result[i] = np.sqrt(max(var, 0) * annualize)

    return result


# ---------------------------------------------------------------------------
# Garman-Klass
# ---------------------------------------------------------------------------


def garman_klass(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Garman-Klass OHLC volatility estimator.

    sigma^2 = 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2

    ~8x more efficient than close-to-close.
    """
    log_hl = np.log(high / np.maximum(low, 1e-10))
    log_co = np.log(close / np.maximum(open_, 1e-10))

    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    n = len(gk)
    result = np.full(n, np.nan)

    for i in range(window, n):
        var = np.mean(gk[i - window : i])
        result[i] = np.sqrt(max(var, 0) * annualize)

    return result


# ---------------------------------------------------------------------------
# Rogers-Satchell
# ---------------------------------------------------------------------------


def rogers_satchell(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Rogers-Satchell volatility (drift-independent).

    sigma^2 = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

    Handles trending markets correctly.
    """
    log_hc = np.log(high / np.maximum(close, 1e-10))
    log_ho = np.log(high / np.maximum(open_, 1e-10))
    log_lc = np.log(low / np.maximum(close, 1e-10))
    log_lo = np.log(low / np.maximum(open_, 1e-10))

    rs = log_hc * log_ho + log_lc * log_lo

    n = len(rs)
    result = np.full(n, np.nan)

    for i in range(window, n):
        var = np.mean(rs[i - window : i])
        result[i] = np.sqrt(max(var, 0) * annualize)

    return result


# ---------------------------------------------------------------------------
# Yang-Zhang
# ---------------------------------------------------------------------------


def yang_zhang(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Yang-Zhang volatility estimator (most efficient OHLC).

    Combines overnight, Rogers-Satchell, and open-to-close components
    with optimal weights for minimum variance estimation.

    sigma^2 = sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_rs^2
    where k = 0.34 / (1.34 + (n+1)/(n-1))
    """
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(window + 1, n):
        w_open = open_[i - window : i]
        w_high = high[i - window : i]
        w_low = low[i - window : i]
        w_close = close[i - window : i]
        w_prev_close = close[i - window - 1 : i - 1]

        # Overnight returns (log)
        log_oc = np.log(w_open / np.maximum(w_prev_close, 1e-10))
        sigma_o2 = np.var(log_oc, ddof=1)

        # Close-to-open returns
        log_co = np.log(w_close / np.maximum(w_open, 1e-10))
        sigma_c2 = np.var(log_co, ddof=1)

        # Rogers-Satchell component
        log_hc = np.log(w_high / np.maximum(w_close, 1e-10))
        log_ho = np.log(w_high / np.maximum(w_open, 1e-10))
        log_lc = np.log(w_low / np.maximum(w_close, 1e-10))
        log_lo = np.log(w_low / np.maximum(w_open, 1e-10))
        sigma_rs2 = np.mean(log_hc * log_ho + log_lc * log_lo)

        # Optimal weight
        wn = window
        k = 0.34 / (1.34 + (wn + 1) / (wn - 1))

        sigma2 = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2
        result[i] = np.sqrt(max(sigma2, 0) * annualize)

    return result


# ---------------------------------------------------------------------------
# EWMA Volatility
# ---------------------------------------------------------------------------


def ewma_volatility(
    returns: np.ndarray,
    halflife: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Exponentially weighted moving average volatility.

    Parameters
    ----------
    returns : np.ndarray
        Simple or log returns.
    halflife : int
        EWMA half-life in periods.
    annualize : int
        Trading days for annualization.

    Returns
    -------
    np.ndarray
        EWMA volatility series.
    """
    lam = 1 - np.log(2) / halflife
    n = len(returns)
    var = np.zeros(n)
    var[0] = returns[0] ** 2

    for i in range(1, n):
        var[i] = lam * var[i - 1] + (1 - lam) * returns[i] ** 2

    return np.sqrt(var * annualize)


# ---------------------------------------------------------------------------
# Realized Volatility
# ---------------------------------------------------------------------------


def realized_volatility(
    intraday_returns: np.ndarray,
    bars_per_day: int = 78,
    annualize: int = 252,
) -> np.ndarray:
    """Realized volatility from high-frequency returns.

    RV = sqrt(sum(r_i^2)) per day, annualized.

    Parameters
    ----------
    intraday_returns : np.ndarray
        Intraday return series.
    bars_per_day : int
        Number of intraday bars per trading day.
    annualize : int
        Trading days per year.

    Returns
    -------
    np.ndarray
        Daily realized volatility, annualized.
    """
    n = len(intraday_returns)
    n_days = n // bars_per_day
    rv = np.zeros(n_days)

    for d in range(n_days):
        start = d * bars_per_day
        end = start + bars_per_day
        rv[d] = np.sqrt(np.sum(intraday_returns[start:end] ** 2))

    return rv * np.sqrt(annualize)
