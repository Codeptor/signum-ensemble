"""Technical feature computation inspired by Qlib Alpha158."""

import numpy as np
import pandas as pd

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
]


def winsorize(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip feature columns at the given percentiles to limit outlier impact.

    Args:
        df: Input DataFrame.
        cols: Columns to winsorize. If None, uses default feature list.
        lower: Lower percentile (e.g. 0.01 for 1st percentile).
        upper: Upper percentile (e.g. 0.99 for 99th percentile).
    """
    cols = cols or [c for c in _WINSORIZE_COLS if c in df.columns]
    for col in cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def compute_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features per ticker.

    Input: DataFrame with columns [ticker, open, high, low, close, volume] and DatetimeIndex.
    Output: DataFrame with original columns plus computed features.
    """
    results = []
    for ticker, group in df.groupby("ticker"):
        feats = _compute_single_ticker(group.copy())
        feats["ticker"] = ticker
        results.append(feats)
    return pd.concat(results).sort_index()


def _compute_single_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single ticker's OHLCV data."""
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # Returns
    for d in [1, 5, 10, 20]:
        df[f"ret_{d}d"] = c.pct_change(d)

    # Moving averages
    for w in [5, 10, 20, 60]:
        df[f"ma_{w}"] = c.rolling(w).mean()
        # Guard against division by zero in rolling mean
        rolling_mean = c.rolling(w).mean()
        df[f"ma_ratio_{w}"] = np.where(rolling_mean != 0, c / rolling_mean, np.nan)

    # Volatility (using log returns for time-additivity — Fix #21)
    log_ret = np.log(c / c.shift(1))
    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = log_ret.rolling(w).std()

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

    # Volume features
    df["volume_ma_10"] = v.rolling(10).mean()
    vol_ma = v.rolling(10).mean()
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

    return df


def compute_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Compute forward returns as the prediction target.

    IMPORTANT: This uses future data and must only be used for label creation,
    never as a feature.
    """
    results = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        g[f"target_{horizon}d"] = g["close"].pct_change(periods=horizon).shift(-horizon)
        results.append(g)
    out = pd.concat(results).sort_index()

    # Winsorize targets at 1st/99th percentile to limit outlier impact
    target_col = f"target_{horizon}d"
    if target_col in out.columns:
        out = winsorize(out, cols=[target_col])
    return out


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional (relative) features by ranking within each date.

    Input: DataFrame with DatetimeIndex and columns including ticker, ret_5d, ret_20d,
           vol_20d, dollar_volume_20d (output of compute_alpha_features).
    Output: Same DataFrame with additional cs_* rank columns (0-1 percentile ranks).
    """
    rank_specs = {
        "cs_ret_rank_5d": "ret_5d",
        "cs_ret_rank_20d": "ret_20d",
        "cs_vol_rank_20d": "vol_20d",
        "cs_volume_rank": "dollar_volume_20d",
    }
    for new_col, src_col in rank_specs.items():
        if src_col in df.columns:
            df[new_col] = df.groupby(level=0)[src_col].rank(pct=True)
    return df


def merge_macro_features(
    df: pd.DataFrame,
    macro_path: str = "data/raw/macro_indicators.parquet",
) -> pd.DataFrame:
    """Merge macro regime features (VIX, yields) into the feature DataFrame.

    Loads macro data, computes derived signals, and broadcasts to all tickers
    via date alignment with forward-fill.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check if macro file exists
    import os

    if not os.path.exists(macro_path):
        logger.warning(f"Macro indicators file not found: {macro_path}. Skipping macro features.")
        return df

    try:
        macro = pd.read_parquet(macro_path)
    except Exception as e:
        logger.warning(f"Could not load macro indicators: {e}. Skipping macro features.")
        return df

    # Derived macro features
    macro["vix_ma_ratio"] = macro["vix"] / macro["vix"].rolling(20).mean()
    macro["term_spread"] = macro["us10y"] - macro["us3m"]
    macro["term_spread_change_20d"] = macro["term_spread"].diff(20)

    # Align: macro is daily with Date index, df has DatetimeIndex
    macro.index = pd.to_datetime(macro.index)
    macro_cols = ["vix", "vix_ma_ratio", "term_spread", "term_spread_change_20d"]
    macro = macro[macro_cols]

    # Forward-fill and join on date (level-0 of df's index)
    macro = macro.reindex(df.index.get_level_values(0).unique()).ffill()
    for col in macro_cols:
        df[col] = df.index.get_level_values(0).map(macro[col])

    return df


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
