import os
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Tiingo API (fallback for delisted/acquired ticker data) ---
# Free tier: 500 API calls/day, covers delisted equities.
# Register at https://api.tiingo.com to get a token.
TIINGO_API_TOKEN = os.getenv("TIINGO_API_TOKEN", "")
DELISTED_CACHE_DIR = DATA_DIR / "cache" / "delisted"

SP500_UNIVERSE = "sp500"
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"

# Single source of truth for risk-free rate across the entire codebase.
# Used in Sharpe/Sortino calculations, risk engine, and backtests.
# As of 2025, the 10-year Treasury yield is ~4.3%, so 5% is conservative.
RISK_FREE_RATE = 0.05

# --- yfinance circuit breaker: staleness policies ---
# Maximum age (in days) of a cached model before refusing to trade.
# LightGBM trains on 2y of data — a model from a few days ago is fine.
MAX_MODEL_AGE_DAYS = 7

# Maximum age (in days) of cached scoring OHLCV before refusing to trade.
# Scoring data determines cross-sectional ranks; >1 day stale means
# today's price action is missing.  2 days covers weekend gaps.
MAX_OHLCV_AGE_DAYS = 2

# Maximum age for Wikipedia S&P 500 ticker list cache (in hours).
# Constituents change quarterly; hourly granularity is more than enough.
TICKER_CACHE_TTL_HOURS = 24

# Exposure reduction multiplier when trading on stale data.
# Applied as an additional scaling factor on top of regime exposure.
STALE_DATA_EXPOSURE_MULT = 0.5

# Paths for persisted caches
OHLCV_CACHE_PATH = DATA_DIR / "cache" / "last_ohlcv.parquet"
OHLCV_CACHE_META_PATH = DATA_DIR / "cache" / "last_ohlcv_meta.json"
TICKER_CACHE_PATH = DATA_DIR / "cache" / "sp500_tickers.json"
RISK_ENGINE_CACHE_PATH = DATA_DIR / "cache" / "risk_engine_returns.parquet"
