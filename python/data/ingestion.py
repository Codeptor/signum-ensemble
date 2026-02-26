"""Data ingestion from yfinance and other free sources.

NOTE — Survivorship bias (Finding #29):
``fetch_sp500_tickers()`` scrapes the **current** S&P 500 constituents from
Wikipedia.  When used for historical backtests this introduces survivorship
bias: companies that were removed, delisted, or acquired before today are
excluded, which inflates backtest returns.  For paper-trading the current
universe is acceptable, but any backtest that uses this function should
clearly state that the universe is point-in-time *inaccurate*.  A proper
fix requires a point-in-time membership table (e.g. Sharadar, daily
snapshots of the index).
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from python.data.config import DEFAULT_INTERVAL, DEFAULT_PERIOD, RAW_DIR

logger = logging.getLogger(__name__)

# Default timeout in seconds for yfinance HTTP calls (M6 fix)
# Prevents indefinite hangs from network issues or API throttling.
YFINANCE_TIMEOUT = 30


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia.

    H12 fix: the scraping is made more robust against Wikipedia table
    structure changes by searching all tables for one containing a
    column with "Symbol" or "Ticker" in its name (case-insensitive),
    rather than hardcoding ``[0]`` and ``"Symbol"``.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # H-WIKI fix: add timeout to prevent indefinite hangs
    tables = pd.read_html(
        url,
        storage_options={"User-Agent": "quant-platform/0.1", "timeout": "15"},
    )

    # H12 fix: search tables for a column that looks like ticker symbols
    symbol_col = None
    target_table = None
    for tbl in tables:
        for col in tbl.columns:
            col_lower = str(col).lower()
            if col_lower in ("symbol", "ticker", "ticker symbol"):
                target_table = tbl
                symbol_col = col
                break
        if target_table is not None:
            break

    if target_table is None or symbol_col is None:
        # Fallback to original behaviour if heuristic fails
        logger.warning(
            "Could not find symbol column by name; falling back to first table, 'Symbol' column"
        )
        target_table = tables[0]
        symbol_col = "Symbol"

    tickers = target_table[symbol_col].str.replace(".", "-", regex=False).tolist()
    # Filter out any non-string or NaN entries
    tickers = [t for t in tickers if isinstance(t, str) and len(t) > 0]

    # H-WIKI fix: validate ticker count — S&P 500 should have ~500 members.
    # If the count is wildly off, the scrape likely hit the wrong table.
    n = len(tickers)
    if n < 490 or n > 520:
        logger.warning(
            f"H-WIKI: fetched {n} tickers from Wikipedia (expected 490-520). "
            "The table structure may have changed."
        )
    return sorted(tickers)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_ohlcv(
    tickers: list[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download OHLCV data for given tickers via yfinance."""
    logger.info(f"Fetching OHLCV for {len(tickers)} tickers, period={period}")
    # H-YFIN fix: always use auto_adjust=True so Close/Open/High/Low are
    # split-and-dividend adjusted.  Without this, features computed on raw
    # prices show spurious jumps at ex-dividend and split dates.
    df = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        threads=True,
        timeout=YFINANCE_TIMEOUT,
        auto_adjust=True,
    )

    # --- NaN validation (Finding #28) ---
    if df.empty:
        raise ValueError("yfinance returned an empty DataFrame — check tickers/period")
    nan_frac = df.isna().mean()
    # Per-column NaN fraction; warn if any column exceeds 5%
    high_nan = nan_frac[nan_frac > 0.05]
    if not high_nan.empty:
        logger.warning(f"Columns with >5%% NaN after download:\n{high_nan}")

    # M13 fix: detect tickers whose data is entirely NaN (download failed silently).
    # yfinance with threads=True swallows per-ticker errors and returns NaN columns.
    if isinstance(df.columns, pd.MultiIndex):
        ticker_level = df.columns.get_level_values(0).unique()
        all_nan_tickers = []
        for t in ticker_level:
            if df[t].isna().all().all():
                all_nan_tickers.append(t)
        if all_nan_tickers:
            logger.warning(
                f"M13: {len(all_nan_tickers)} tickers returned entirely NaN data "
                f"(likely failed silently): {all_nan_tickers[:20]}..."
                if len(all_nan_tickers) > 20
                else f"M13: {len(all_nan_tickers)} tickers returned entirely NaN data: "
                f"{all_nan_tickers}"
            )
            # Drop entirely-NaN tickers to prevent corrupted downstream features
            df = df.drop(columns=all_nan_tickers, level=0)

    # Forward-fill small gaps (weekends/holidays already absent), then
    # aggressively clean remaining NaNs to prevent corrupted rolling features.
    df = df.ffill(limit=3)

    # C-NAN fix: ``dropna(how="all")`` only dropped rows where EVERY column
    # was NaN, leaving partial-NaN rows that corrupt rolling features for
    # the affected tickers.  Instead, drop *ticker columns* that still have
    # >5% NaN after ffill (these tickers had extended data gaps), then drop
    # any remaining fully-NaN rows.
    if isinstance(df.columns, pd.MultiIndex):
        # Per-ticker NaN check on the MultiIndex structure
        ticker_level = df.columns.get_level_values(0).unique()
        high_nan_tickers = []
        for t in ticker_level:
            nan_frac_t = df[t].isna().mean().mean()
            if nan_frac_t > 0.05:
                high_nan_tickers.append(t)
        if high_nan_tickers:
            logger.warning(
                f"C-NAN: dropping {len(high_nan_tickers)} tickers with >5% NaN "
                f"after ffill: {high_nan_tickers[:20]}"
            )
            df = df.drop(columns=high_nan_tickers, level=0)

    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        # M-LOGMSG fix: message now accurately describes what happens — we
        # drop fully-NaN rows and fill scattered gaps, not "dropping N NaN values".
        remaining_nan_rows = df.isna().any(axis=1).sum()
        logger.warning(
            f"Residual NaN after ffill + ticker pruning: {remaining_nans} values "
            f"across {remaining_nan_rows} rows. Dropping all-NaN rows and "
            f"filling scattered single-cell gaps (ffill/bfill limit=1)."
        )
        # Drop rows that are entirely NaN, then forward-fill any scattered
        # single-cell gaps that survived (limited to 1 for safety).
        df = df.dropna(axis=0, how="all")
        df = df.ffill(limit=1).bfill(limit=1)
    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_fred_macro() -> pd.DataFrame:
    """Fetch key macro indicators via Yahoo Finance index proxies.

    H13 note: despite the function name, this does NOT use the FRED API.
    It downloads Yahoo Finance index tickers (``^VIX``, ``^TNX``, ``^IRX``)
    which have no SLA and may be renamed, removed, or return stale data.
    A proper fix would use the ``fredapi`` package with a FRED API key,
    but that requires registration.  For paper trading this is acceptable.

    Forward-fill is limited to 5 days to prevent stale data from persisting
    indefinitely during extended outages.
    """
    macro_tickers = {
        "^VIX": "vix",
        "^TNX": "us10y",
        "^IRX": "us3m",
    }
    frames = {}
    for ticker, name in macro_tickers.items():
        try:
            data = yf.download(
                ticker,
                period=DEFAULT_PERIOD,
                interval=DEFAULT_INTERVAL,
                timeout=YFINANCE_TIMEOUT,
                auto_adjust=True,
            )
            if data is None or data.empty:
                logger.warning(f"Macro ticker {ticker} ({name}): no data returned, skipping")
                continue
            close = data["Close"]
            # yfinance may return DataFrame instead of Series; squeeze to 1-D
            if isinstance(close, pd.DataFrame):
                close = close.squeeze(axis=1)
            nan_count = int(close.isna().sum())
            if nan_count > 0:
                logger.warning(
                    f"Macro ticker {ticker}: {nan_count} NaN values — forward-filling (limit=5)"
                )
                # H13 fix: limit ffill to 5 days so stale data doesn't persist indefinitely
                close = close.ffill(limit=5)
            frames[name] = close
        except Exception as e:
            logger.warning(f"Failed to fetch macro ticker {ticker} ({name}): {e}")
            continue

    if not frames:
        raise ValueError("All macro ticker downloads failed — cannot build macro DataFrame")
    return pd.DataFrame(frames)


def reshape_ohlcv_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance MultiIndex-column OHLCV to long format.

    yfinance with group_by='ticker' returns columns like (AAPL, Close), (AAPL, Open), ...
    This converts to rows with columns: [ticker, open, high, low, close, volume]
    and a DatetimeIndex.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df  # already flat

    frames = []
    tickers = df.columns.get_level_values(0).unique()
    for ticker in tickers:
        ticker_df = df[ticker].copy()
        ticker_df.columns = ticker_df.columns.str.lower()
        ticker_df["ticker"] = ticker
        frames.append(ticker_df)
    return pd.concat(frames).sort_index()


def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a simple (date x ticker) close-price matrix from yfinance output.

    Returns DataFrame with DatetimeIndex and one column per ticker.

    H-MULTIIDX fix: uses ``df.xs("Close", axis=1, level=...)`` to safely
    handle both ``(ticker, OHLCV)`` and ``(OHLCV, ticker)`` MultiIndex
    layouts from different yfinance versions, instead of hard-coding
    ``df[(t, "Close")]`` which assumes a specific level order.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Determine which level contains the OHLCV labels
        level_0_vals = set(df.columns.get_level_values(0).unique())
        ohlcv_names = {
            "Close",
            "Open",
            "High",
            "Low",
            "Volume",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "Adj Close",
        }
        if level_0_vals & ohlcv_names:
            # Level 0 is OHLCV (e.g. newer yfinance: ("Close", "AAPL"))
            close_df = df.xs("Close", axis=1, level=0)
        else:
            # Level 0 is ticker (e.g. group_by="ticker": ("AAPL", "Close"))
            close_df = df.xs("Close", axis=1, level=1)
        return close_df
    return df


def save_raw_data(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame as Parquet in the raw data directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} rows to {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickers = fetch_sp500_tickers()
    logger.info(f"Got {len(tickers)} S&P 500 tickers")
    ohlcv = fetch_ohlcv(tickers)
    save_raw_data(ohlcv, "sp500_ohlcv")
    macro = fetch_fred_macro()
    save_raw_data(macro, "macro_indicators")
