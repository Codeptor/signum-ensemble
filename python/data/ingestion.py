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

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from python.data.config import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    RAW_DIR,
    TICKER_CACHE_PATH,
    TICKER_CACHE_TTL_HOURS,
)

logger = logging.getLogger(__name__)

# Default timeout in seconds for yfinance HTTP calls (M6 fix)
# Prevents indefinite hangs from network issues or API throttling.
YFINANCE_TIMEOUT = 30

# In-memory ticker cache: (tickers, fetch_timestamp)
_ticker_cache: tuple[list[str], float] | None = None


def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers with in-memory + disk cache.

    Returns cached list if it's younger than ``TICKER_CACHE_TTL_HOURS``.
    Falls back to the disk cache if Wikipedia is unreachable.
    """
    global _ticker_cache

    # 1. In-memory cache (fastest)
    if _ticker_cache is not None:
        tickers, ts = _ticker_cache
        age_hours = (time.time() - ts) / 3600
        if age_hours < TICKER_CACHE_TTL_HOURS:
            return tickers

    # 2. Try live scrape
    try:
        tickers = _fetch_sp500_tickers_live()
        _ticker_cache = (tickers, time.time())
        # Persist to disk for offline fallback
        TICKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TICKER_CACHE_PATH.write_text(json.dumps({"tickers": tickers, "ts": time.time()}))
        return tickers
    except Exception as e:
        logger.warning(f"Wikipedia scrape failed: {e} — trying disk cache")

    # 3. Disk cache fallback
    if TICKER_CACHE_PATH.exists():
        try:
            data = json.loads(TICKER_CACHE_PATH.read_text())
            tickers = data["tickers"]
            age_hours = (time.time() - data["ts"]) / 3600
            logger.info(f"Using cached ticker list ({len(tickers)} tickers, {age_hours:.0f}h old)")
            _ticker_cache = (tickers, data["ts"])
            return tickers
        except Exception as e2:
            logger.error(f"Disk ticker cache also failed: {e2}")

    raise RuntimeError("Cannot obtain S&P 500 ticker list: Wikipedia and disk cache both failed")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_sp500_tickers_live() -> list[str]:
    """Scrape S&P 500 tickers from Wikipedia (with retry).

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
            # R3-P-12 fix: parenthesize conditional to prevent string concat bug
            if len(all_nan_tickers) > 20:
                msg = (
                    f"M13: {len(all_nan_tickers)} tickers returned entirely NaN data "
                    f"(likely failed silently): {all_nan_tickers[:20]}..."
                )
            else:
                msg = (
                    f"M13: {len(all_nan_tickers)} tickers returned entirely NaN data: "
                    f"{all_nan_tickers}"
                )
            logger.warning(msg)
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


def fetch_tiingo_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV from Tiingo for tickers yfinance can't resolve (delisted/acquired).

    Tiingo retains data for 65K+ tickers including delisted equities.
    Free tier: 500 API calls/day — sufficient for ~50-100 delisted tickers.

    Returns data in the same MultiIndex format as fetch_ohlcv() so downstream
    code (reshape_ohlcv_wide_to_long, etc.) works without changes.

    Args:
        tickers: List of ticker symbols to fetch.
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV field) matching
        the yfinance output format. Empty DataFrame if no data or no API token.
    """
    import urllib.error
    import urllib.request

    from python.data.config import DELISTED_CACHE_DIR, TIINGO_API_TOKEN

    if not TIINGO_API_TOKEN:
        logger.warning("TIINGO_API_TOKEN not set — cannot fetch delisted ticker data")
        return pd.DataFrame()

    if not tickers:
        return pd.DataFrame()

    DELISTED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}
    fetched = 0
    cached = 0

    for ticker in tickers:
        # Check local cache first (delisted data never changes)
        cache_path = DELISTED_CACHE_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            try:
                df_cached = pd.read_parquet(cache_path)
                # Filter to requested date range
                mask = (df_cached.index >= start_date) & (df_cached.index <= end_date)
                df_filtered = df_cached.loc[mask]
                if not df_filtered.empty:
                    frames[ticker] = df_filtered
                    cached += 1
                    continue
            except Exception:
                pass  # Re-fetch on cache corruption

        # Fetch from Tiingo API
        url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            f"?startDate={start_date}&endDate={end_date}"
            f"&token={TIINGO_API_TOKEN}"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "signum-quant/0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                import json as _json

                data = _json.loads(resp.read().decode("utf-8"))

            if not data:
                logger.debug(f"Tiingo: no data for {ticker}")
                continue

            df_ticker = pd.DataFrame(data)
            df_ticker["date"] = pd.to_datetime(df_ticker["date"]).dt.tz_localize(None)
            df_ticker = df_ticker.set_index("date")

            # Map Tiingo columns to yfinance column names
            col_map = {
                "adjOpen": "Open",
                "adjHigh": "High",
                "adjLow": "Low",
                "adjClose": "Close",
                "adjVolume": "Volume",
            }
            df_mapped = df_ticker.rename(columns=col_map)[list(col_map.values())]
            df_mapped = df_mapped.dropna(subset=["Close"])

            if df_mapped.empty:
                continue

            # Cache locally (full date range — trim later)
            try:
                df_mapped.to_parquet(cache_path)
            except Exception:
                pass

            frames[ticker] = df_mapped
            fetched += 1

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"Tiingo: {ticker} not found (404)")
            else:
                logger.warning(f"Tiingo: HTTP {e.code} for {ticker}")
        except Exception as e:
            logger.debug(f"Tiingo: failed for {ticker}: {e}")

    if not frames:
        logger.info("Tiingo: no additional data fetched for delisted tickers")
        return pd.DataFrame()

    logger.info(
        f"Tiingo: fetched {fetched} tickers (API), {cached} from cache, "
        f"{len(tickers) - fetched - cached} unavailable"
    )

    # Build MultiIndex DataFrame matching yfinance format
    panels = {}
    for ticker, df_t in frames.items():
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_t.columns:
                panels[(ticker, col)] = df_t[col]

    if not panels:
        return pd.DataFrame()

    result = pd.DataFrame(panels)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


def fetch_ohlcv_with_delisted(
    current_tickers: list[str],
    historical_tickers: list[str],
    period: str = DEFAULT_PERIOD,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV for current + delisted tickers using yfinance + Tiingo fallback.

    This is the survivorship-bias-free version of fetch_ohlcv().

    1. Fetch all current_tickers via yfinance (fast, batch download)
    2. Identify which historical_tickers are missing from the yfinance result
    3. Fetch missing tickers from Tiingo (delisted/acquired stocks)
    4. Merge into a single MultiIndex DataFrame

    Args:
        current_tickers: Tickers currently in the S&P 500.
        historical_tickers: Additional tickers that were in the S&P 500
            during the training window but have since been removed.
        period: yfinance period string (e.g., "2y") for current tickers.
        start_date: Explicit start date for Tiingo (YYYY-MM-DD).
        end_date: Explicit end date for Tiingo (YYYY-MM-DD).

    Returns:
        Combined MultiIndex DataFrame with OHLCV for all available tickers.
    """
    # Step 1: fetch current tickers via yfinance (the fast path)
    all_tickers = sorted(set(current_tickers) | set(historical_tickers))
    yf_df = fetch_ohlcv(all_tickers, period=period)

    # Step 2: identify tickers that yfinance couldn't resolve
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_tickers = set(yf_df.columns.get_level_values(0).unique())
    else:
        yf_tickers = set(yf_df.columns)

    missing_tickers = sorted(set(historical_tickers) - yf_tickers)

    if not missing_tickers:
        logger.info("All historical tickers resolved by yfinance — no Tiingo fallback needed")
        return yf_df

    logger.info(
        f"yfinance missing {len(missing_tickers)} historical tickers: "
        f"{missing_tickers[:20]}{'...' if len(missing_tickers) > 20 else ''}"
    )

    # Step 3: compute date range for Tiingo
    if start_date is None:
        # Infer from yfinance data
        start_date = yf_df.index.min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = yf_df.index.max().strftime("%Y-%m-%d")

    tiingo_df = fetch_tiingo_ohlcv(missing_tickers, start_date, end_date)

    if tiingo_df.empty:
        logger.info("Tiingo returned no additional data — proceeding with yfinance only")
        return yf_df

    # Step 4: merge yfinance + Tiingo DataFrames
    # Align on date index, fill missing dates naturally
    combined = pd.concat([yf_df, tiingo_df], axis=1)

    tiingo_tickers = set(tiingo_df.columns.get_level_values(0).unique())
    still_missing = set(missing_tickers) - tiingo_tickers
    if still_missing:
        logger.warning(
            f"Still missing {len(still_missing)} tickers after Tiingo fallback "
            f"(truly unavailable): {sorted(still_missing)[:20]}"
        )

    total_tickers = len(combined.columns.get_level_values(0).unique())
    logger.info(
        f"Combined OHLCV: {total_tickers} tickers "
        f"({len(yf_tickers)} yfinance + {len(tiingo_tickers)} Tiingo)"
    )

    return combined


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
