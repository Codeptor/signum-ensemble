"""Technical feature computation inspired by Qlib Alpha158."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from python.risk.volatility import parkinson, yang_zhang

logger = logging.getLogger(__name__)

# R3-P-4 fix: resolve paths relative to project root, not CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BOUNDS_DIR = _PROJECT_ROOT / "data" / "models"

# Columns that should be winsorized to limit outlier impact (Fix #20)
_WINSORIZE_COLS = [
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
    "volume_ratio",
    "amihud_illiq",
    "bid_ask_proxy",
    "vol_yz_20d",
    "vol_park_20d",
]

# C9 fix: Neutral default values for features when data is unavailable.
# Using 0.0 for VIX (which has a floor ~9 and mean ~20) would trick the
# model into predicting as if the market is extremely calm.  These values
# represent the approximate long-run median for each feature so the model
# produces roughly neutral predictions when a feature is missing.
FEATURE_NEUTRAL_DEFAULTS: dict[str, float] = {
    "vix": 20.0,  # long-run VIX median
    "vix_ma_ratio": 1.0,  # VIX at its own moving average
    "term_spread": 1.0,  # ~100bps normal spread
    "term_spread_change_20d": 0.0,  # no change
    "rsi_14": 50.0,  # midpoint (neither overbought nor oversold)
    "bb_position": 0.5,  # middle of Bollinger Band
    "vol_20d": 0.015,  # ~24% annualised vol (typical for S&P stocks)
    "volume_ratio": 1.0,  # average volume
    "ret_5d": 0.0,  # no return
    "ret_10d": 0.0,
    "ret_20d": 0.0,
    "cs_ret_rank_5d": 0.5,  # median rank
    "cs_ret_rank_20d": 0.5,
    "cs_vol_rank_20d": 0.5,
    "cs_volume_rank": 0.5,
    "vol_yz_20d": 0.015,  # ~24% annualised vol (same as vol_20d)
    "vol_park_20d": 0.015,
}


def winsorize(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    lower: float = 0.005,
    upper: float = 0.995,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Clip feature columns at the given percentiles to limit outlier impact.

    M15 fix: operates on a copy so callers' DataFrames are not mutated.
    H6 fix: default percentiles widened from 1st/99th to 0.5th/99.5th to
    better accommodate fat-tailed return distributions.

    C-WINS fix: if ``bounds`` is provided, those fixed (lo, hi) pairs are
    applied instead of recomputing from the current data.  This ensures
    training and inference use identical clipping thresholds.

    Args:
        df: Input DataFrame.
        cols: Columns to winsorize. If None, uses default feature list.
        lower: Lower percentile (e.g. 0.005 for 0.5th percentile).
        upper: Upper percentile (e.g. 0.995 for 99.5th percentile).
        bounds: Optional pre-computed {col: (lo, hi)} dict.  When provided,
            ``lower``/``upper`` percentile args are ignored and these fixed
            values are used instead.
    """
    df = df.copy()  # M15 fix: never mutate caller's DataFrame
    cols = cols or [c for c in _WINSORIZE_COLS if c in df.columns]
    for col in cols:
        if bounds is not None and col in bounds:
            lo, hi = bounds[col]
        else:
            lo = df[col].quantile(lower)
            hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def compute_winsorize_bounds(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    lower: float = 0.005,
    upper: float = 0.995,
) -> dict[str, tuple[float, float]]:
    """Compute per-column winsorization bounds from training data.

    C-WINS fix: these bounds should be persisted at training time and
    loaded at inference time so the model sees identical feature
    distributions in both regimes.

    Args:
        df: Training DataFrame.
        cols: Columns to compute bounds for.  Defaults to ``_WINSORIZE_COLS``.
        lower: Lower percentile.
        upper: Upper percentile.

    Returns:
        Dict mapping column name to (lo, hi) tuple.
    """
    cols = cols or [c for c in _WINSORIZE_COLS if c in df.columns]
    bounds: dict[str, tuple[float, float]] = {}
    for col in cols:
        lo = float(df[col].quantile(lower))
        hi = float(df[col].quantile(upper))
        bounds[col] = (lo, hi)
    return bounds


def save_winsorize_bounds(
    bounds: dict[str, tuple[float, float]],
    path: str | Path | None = None,
) -> Path:
    """Persist winsorization bounds to a JSON file.

    Args:
        bounds: Dict of {col: (lo, hi)}.
        path: Output path.  Defaults to ``data/models/winsorize_bounds.json``.

    Returns:
        The path the file was written to.
    """
    if path is None:
        path = _BOUNDS_DIR / "winsorize_bounds.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON can't serialize tuples — convert to lists
    serializable = {col: list(vals) for col, vals in bounds.items()}
    path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Saved winsorize bounds ({len(bounds)} cols) to {path}")
    return path


def load_winsorize_bounds(
    path: str | Path | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Load persisted winsorization bounds.

    Args:
        path: Path to the JSON file.  Defaults to
            ``data/models/winsorize_bounds.json``.

    Returns:
        Dict of {col: (lo, hi)}, or None if the file doesn't exist.
    """
    if path is None:
        path = _BOUNDS_DIR / "winsorize_bounds.json"
    path = Path(path)

    if not path.exists():
        logger.warning(f"No winsorize bounds file at {path} — will compute from data")
        return None

    raw = json.loads(path.read_text())
    bounds = {col: (vals[0], vals[1]) for col, vals in raw.items()}
    logger.info(f"Loaded winsorize bounds ({len(bounds)} cols) from {path}")
    return bounds


def _scrub_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN throughout a DataFrame.

    C11 fix: inf values from log(0), division by zero, or pct_change on
    zero-valued series propagate through the pipeline and cause undefined
    behaviour in LightGBM (which handles NaN but not inf).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = df[numeric_cols].isin([np.inf, -np.inf])
    if mask.any().any():
        n_inf = int(mask.sum().sum())
        logger.warning(f"Replaced {n_inf} inf values with NaN in feature pipeline")
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def compute_alpha_features(
    df: pd.DataFrame,
    winsorize_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Compute technical features per ticker.

    Input: DataFrame with columns [ticker, open, high, low, close, volume] and DatetimeIndex.
    Output: DataFrame with original columns plus computed features.

    H7 fix: winsorize is applied uniformly *after* all per-ticker features
    are computed, ensuring consistent clipping regardless of computation order.

    C-WINS fix: accepts optional pre-computed winsorize bounds so that
    inference uses the same clipping thresholds as training.

    Args:
        df: Long-format OHLCV DataFrame with a ``ticker`` column.
        winsorize_bounds: Optional dict of {col: (lo, hi)} from training.
            If None, bounds are computed from the current data (training
            behaviour).  Pass the output of ``load_winsorize_bounds()``
            at inference time for consistency.
    """
    results = []
    for ticker, group in df.groupby("ticker"):
        feats = _compute_single_ticker(group.copy())
        feats["ticker"] = ticker
        results.append(feats)
    out = pd.concat(results).sort_index()

    # H7 fix: winsorize all feature columns after computation, not piecemeal
    # C-WINS fix: use pre-computed bounds when available
    out = winsorize(out, bounds=winsorize_bounds)

    return out


def _compute_single_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single ticker's OHLCV data."""
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # Returns (R3-P-17 fix: skip ret_1d — computed but never used by model)
    for d in [5, 10, 20]:
        df[f"ret_{d}d"] = c.pct_change(d)

    # Moving averages (R3-P-13 fix: compute rolling mean once and reuse)
    for w in [5, 10, 20, 60]:
        rolling_mean = c.rolling(w).mean()
        df[f"ma_{w}"] = rolling_mean
        df[f"ma_ratio_{w}"] = np.where(rolling_mean != 0, c / rolling_mean, np.nan)

    # Volatility (using log returns for time-additivity — Fix #21)
    # C11 fix: guard log(0) and log(negative) which produce -inf / NaN
    ratio = c / c.shift(1)
    ratio = ratio.clip(lower=1e-10)  # prevent log(0) → -inf
    log_ret = np.log(ratio)
    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = log_ret.rolling(w).std()

    # Yang-Zhang volatility (8x more efficient than close-to-close, uses OHLC)
    # Returns annualized vol — divide by sqrt(252) to get daily scale matching vol_20d
    yz = yang_zhang(o.values, h.values, lo.values, c.values, window=20, annualize=1)
    df["vol_yz_20d"] = yz

    # Parkinson range-based volatility (5x more efficient, uses high-low)
    park = parkinson(h.values, lo.values, window=20, annualize=1)
    df["vol_park_20d"] = park

    # RSI
    for w in [14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / np.where(loss != 0, loss, np.nan)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Bollinger Bands
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = np.where(bb_range != 0, (c - df["bb_lower"]) / bb_range, 0.5)

    # Volume features (R3-P-18 fix: compute rolling mean once and reuse)
    vol_ma = v.rolling(10).mean()
    df["volume_ma_10"] = vol_ma
    df["volume_ratio"] = np.where(vol_ma != 0, v / vol_ma, 1.0)

    # Liquidity features
    dollar_vol = c * v
    df["dollar_volume_20d"] = dollar_vol.rolling(20).mean()
    df["amihud_illiq"] = (
        (c.pct_change().abs() / np.where(dollar_vol != 0, dollar_vol, np.nan)).rolling(20).mean()
    )
    df["bid_ask_proxy"] = np.where(c != 0, (h - lo) / c, 0.0)  # Corwin-Schultz spread proxy

    # Open-close range
    df["oc_range"] = np.where(c != 0, (c - o) / c, 0.0)

    # Momentum 12-1: 12-month return excluding last month (Jegadeesh & Titman)
    # The most documented equity anomaly — excludes the short-term reversal month
    ret_12m = c.pct_change(252)  # ~12 months of trading days
    ret_1m = c.pct_change(21)  # ~1 month
    df["mom_12_1"] = ret_12m - ret_1m

    # Mean-reversion z-score: (price - 60d MA) / 60d std
    # Identifies overbought/oversold conditions
    ma_60 = c.rolling(60).mean()
    std_60 = c.rolling(60).std()
    df["mr_zscore_60"] = np.where(std_60 > 0, (c - ma_60) / std_60, 0.0)

    # C11 fix: scrub any inf/-inf that slipped through division or log
    df = _scrub_infinities(df)

    return df


def compute_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Compute forward returns as the prediction target.

    IMPORTANT: This uses future data and must only be used for label creation,
    never as a feature.

    C-TARGET fix: target winsorization has been removed.  Previously the
    target was clipped *before* ``compute_residual_target()`` subtracted
    the cross-sectional mean, biasing the residual toward zero for extreme
    movers.  LightGBM's Huber loss already down-weights outlier targets,
    making explicit winsorization unnecessary and actively harmful.
    """
    results = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        g[f"target_{horizon}d"] = g["close"].pct_change(periods=horizon).shift(-horizon)
        results.append(g)
    out = pd.concat(results).sort_index()

    # C-TARGET fix: do NOT winsorize targets — Huber loss handles outliers,
    # and clipping before residualization biases the residual.
    return out


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional (relative) features by ranking within each date.

    Input: DataFrame with DatetimeIndex and columns including ticker, ret_5d, ret_20d,
           vol_20d, dollar_volume_20d (output of compute_alpha_features).
    Output: Same DataFrame with additional cs_* rank columns (0-1 percentile ranks).

    C10 fix: when all stocks in a cross-section have the same value,
    ``rank(pct=True)`` can produce degenerate results or NaN depending on
    the method. We use ``method='average'`` (pandas default) which assigns
    0.5 when all values are tied — a neutral rank.  The result is then
    scrubbed for any inf values.
    """
    df = df.copy()  # avoid mutating caller's DataFrame
    rank_specs = {
        "cs_ret_rank_5d": "ret_5d",
        "cs_ret_rank_20d": "ret_20d",
        "cs_vol_rank_20d": "vol_20d",
        "cs_volume_rank": "dollar_volume_20d",
    }
    for new_col, src_col in rank_specs.items():
        if src_col in df.columns:
            # C10 fix: rank with method='average' handles ties safely.
            # When all values are identical, average rank = 0.5 (neutral).
            df[new_col] = df.groupby(level=0)[src_col].rank(
                pct=True, method="average", na_option="keep"
            )

    # Sector-relative momentum: stock's return minus sector average return
    # Isolates stock-specific momentum from sector rotation effects
    if "ret_20d" in df.columns and "ticker" in df.columns:
        try:
            from python.data.sectors import get_sector

            df["_sector"] = df["ticker"].map(get_sector)
            sector_mean = df.groupby([df.index.get_level_values(0), "_sector"])[
                "ret_20d"
            ].transform("mean")
            df["sector_rel_mom"] = df["ret_20d"] - sector_mean
            df.drop(columns=["_sector"], inplace=True)
        except Exception as exc:
            logger.warning(f"Could not compute sector-relative momentum: {exc}")

    # C11 fix: scrub any inf that may leak from upstream computations
    df = _scrub_infinities(df)
    return df


def merge_macro_features(
    df: pd.DataFrame,
    macro_path: str | Path = "data/raw/macro_indicators.parquet",
) -> pd.DataFrame:
    """Merge macro regime features (VIX, yields) into the feature DataFrame.

    Loads macro data, computes derived signals, and broadcasts to all tickers
    via date alignment with forward-fill.

    M-MACROPATH fix: when ``macro_path`` is relative, it is resolved relative
    to the project root (2 levels up from this file: ``python/alpha/``) so
    that the function works regardless of the caller's working directory.
    """
    import logging

    logger = logging.getLogger(__name__)

    # M-MACROPATH fix: resolve relative paths against project root
    macro_path = Path(macro_path)
    if not macro_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        macro_path = project_root / macro_path

    if not macro_path.exists():
        logger.warning(f"Macro indicators file not found: {macro_path}. Skipping macro features.")
        return df

    try:
        macro = pd.read_parquet(macro_path)
    except Exception as e:
        logger.warning(f"Could not load macro indicators: {e}. Skipping macro features.")
        return df

    # Derived macro features
    # R3-P-6 fix: guard against division by zero when rolling mean is 0 or all-NaN
    vix_ma = macro["vix"].rolling(20).mean()
    macro["vix_ma_ratio"] = np.where(
        (vix_ma != 0) & (~np.isnan(vix_ma)), macro["vix"] / vix_ma, 1.0
    )
    macro["term_spread"] = macro["us10y"] - macro["us3m"]
    macro["term_spread_change_20d"] = macro["term_spread"].diff(20)

    # Align: macro is daily with Date index, df has DatetimeIndex
    macro.index = pd.to_datetime(macro.index)
    macro_cols = ["vix", "vix_ma_ratio", "term_spread", "term_spread_change_20d"]
    macro = macro[macro_cols]

    # Forward-fill and join on date (level-0 of df's index)
    macro = macro.reindex(df.index.get_level_values(0).unique()).ffill()

    # H-MACRO fix: for dates before the first macro observation (e.g. early
    # training rows), ffill produces NaN which then drops rows from the
    # training set.  Back-fill first, then fill any remaining NaN with
    # domain-appropriate neutral defaults.
    macro = macro.bfill()
    _macro_neutral = {
        "vix": 20.0,
        "vix_ma_ratio": 1.0,
        "term_spread": 1.0,
        "term_spread_change_20d": 0.0,
    }
    for col in macro_cols:
        if col in _macro_neutral:
            macro[col] = macro[col].fillna(_macro_neutral[col])

    for col in macro_cols:
        df[col] = df.index.get_level_values(0).map(macro[col])

    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Average True Range for dynamic stop-loss placement.

    ATR measures a stock's natural volatility, enabling adaptive stop-loss
    distances that account for each stock's typical price movement.

    Args:
        df: DataFrame with columns [high, low, close] and a DatetimeIndex.
            Can be single-ticker or multi-ticker (with 'ticker' column).
        window: ATR lookback period in days (default 14).

    Returns:
        DataFrame with added ``atr_{window}`` column.
    """
    if "ticker" in df.columns:
        # Multi-ticker: compute per ticker
        results = []
        for ticker, group in df.groupby("ticker"):
            g = group.copy()
            g = _compute_atr_single(g, window)
            results.append(g)
        return pd.concat(results).sort_index()
    else:
        return _compute_atr_single(df.copy(), window)


def _compute_atr_single(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute ATR for a single ticker's OHLC data."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{window}"] = true_range.rolling(window).mean()

    return df


def get_current_atr(
    symbol: str,
    window: int = 14,
    period: str = "3mo",
    default: float | None = None,
    ohlcv_data: pd.DataFrame | None = None,
) -> float | None:
    """Get current ATR for a single symbol.

    Convenience function for the live bot to get ATR for stop-loss placement.

    If ``ohlcv_data`` is provided (a DataFrame with High, Low, Close columns
    for the given symbol), it is used directly — avoiding a redundant
    ``yf.download()`` call.  Otherwise falls back to fetching from Yahoo
    Finance.

    M-ATR fix: added ``default`` parameter and NaN guard.  When the ATR
    computation fails or returns NaN, the function returns ``default``
    instead of ``None``.  Callers can pass a neutral default (e.g. a
    fraction of current price) to avoid downstream ``None`` type errors.

    Args:
        symbol: Ticker symbol.
        window: ATR lookback period.
        period: Yahoo Finance period string for data fetch (ignored if
            ``ohlcv_data`` is provided).
        default: Value to return if ATR cannot be computed.
        ohlcv_data: Optional pre-fetched OHLCV DataFrame for this symbol.
            Must contain High, Low, and Close columns.

    Returns:
        Current ATR value, or ``default`` if data unavailable or NaN.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        if ohlcv_data is not None and len(ohlcv_data) >= window + 1:
            data = ohlcv_data
        else:
            import yfinance as yf

            data = yf.download(symbol, period=period, interval="1d", progress=False)

        if data is None or len(data) < window + 1:
            logger.warning(f"Insufficient data for ATR computation: {symbol}")
            return default

        # Compute ATR
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()

        current_atr = float(atr.iloc[-1])
        # M-ATR fix: guard against NaN result (e.g. all-NaN rolling window)
        if pd.isna(current_atr):
            logger.warning(f"ATR({window}) for {symbol} is NaN — returning default")
            return default
        logger.debug(f"ATR({window}) for {symbol}: {current_atr:.2f}")
        return current_atr
    except Exception as e:
        logger.warning(f"Failed to compute ATR for {symbol}: {e}")
        return default


def compute_residual_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Subtract cross-sectional mean return to create a market-neutral target.

    Preserves the raw target as raw_target_{horizon}d and replaces
    target_{horizon}d with the residual (stock return minus market mean).
    """
    target_col = f"target_{horizon}d"
    if target_col not in df.columns:
        return df

    df[f"raw_target_{horizon}d"] = df[target_col]
    cs_mean = df.groupby(level=0)[target_col].transform("mean")
    df[target_col] = df[target_col] - cs_mean
    return df
