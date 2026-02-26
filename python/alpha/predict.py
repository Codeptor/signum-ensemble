"""Live inference pipeline: fetch data -> features -> train -> rank -> optimize weights.

This module is the glue between the ML model and the live trading bot.
It handles the full pipeline from raw OHLCV data to target portfolio weights.
"""

import hashlib
import json
import logging
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

from python.alpha.features import (
    compute_alpha_features,
    compute_cross_sectional_features,
)
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.data.ingestion import (
    extract_close_prices,
    fetch_ohlcv,
    reshape_ohlcv_wide_to_long,
)
from python.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

# Minimum history needed for feature computation (60-day MA + 20-day rolling)
MIN_HISTORY_DAYS = "6mo"

# Training lookback — how much S&P 500 history to train the model on
TRAINING_LOOKBACK = "5y"

# Model cache directory
MODEL_CACHE_DIR = Path("data/models")
MODEL_CACHE_FILE = MODEL_CACHE_DIR / "latest_model.joblib"

# Model versioning — keep last N versions (Fix #45)
MAX_MODEL_VERSIONS = 10


def _load_cached_model() -> CrossSectionalModel | None:
    """Load a cached model if it was trained today. Returns None if stale or missing."""
    if not MODEL_CACHE_FILE.exists():
        return None

    try:
        cached = joblib.load(MODEL_CACHE_FILE)
        trained_date = cached.get("trained_date")
        model = cached.get("model")

        if trained_date == date.today().isoformat() and model is not None:
            logger.info(f"Loaded cached model from {MODEL_CACHE_FILE} (trained {trained_date})")
            return model

        logger.info(f"Cached model is stale (trained {trained_date}), will retrain")
        return None
    except Exception as e:
        logger.warning(f"Failed to load cached model: {e}")
        return None


def _save_model_cache(model: CrossSectionalModel) -> None:
    """Save a trained model with versioning (Fix #45).

    Creates a timestamped version file alongside the ``latest_model.joblib``
    cache.  Old versions beyond ``MAX_MODEL_VERSIONS`` are pruned.

    Each version is saved as ``model_<date>_<hash>.joblib`` where *hash* is a
    short SHA-256 of the model parameters for uniqueness when multiple models
    are trained on the same day.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    now_iso = datetime.utcnow().isoformat(timespec="seconds")

    # Deterministic short hash from model params + timestamp for uniqueness
    hash_input = json.dumps(model.params, sort_keys=True, default=str) + now_iso
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    payload = {
        "model": model,
        "trained_date": today,
        "trained_at": now_iso,
        "feature_cols": model.feature_cols,
        "params": model.params,
    }

    # 1. Save versioned copy
    version_name = f"model_{today}_{short_hash}.joblib"
    version_path = MODEL_CACHE_DIR / version_name
    joblib.dump(payload, version_path)
    logger.info(f"Saved model version: {version_path}")

    # 2. Overwrite latest cache (for fast daily reload)
    joblib.dump(payload, MODEL_CACHE_FILE)
    logger.info(f"Updated latest model cache: {MODEL_CACHE_FILE}")

    # 3. Prune old versions
    _prune_old_versions()


def _prune_old_versions() -> None:
    """Keep only the most recent ``MAX_MODEL_VERSIONS`` versioned model files."""
    versions = sorted(
        MODEL_CACHE_DIR.glob("model_*_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in versions[MAX_MODEL_VERSIONS:]:
        old.unlink()
        logger.info(f"Pruned old model version: {old.name}")


def list_model_versions() -> list[dict]:
    """List all saved model versions with metadata.

    Returns a list of dicts with keys: path, trained_date, trained_at, params.
    Sorted newest-first.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    versions = sorted(
        MODEL_CACHE_DIR.glob("model_*_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result = []
    for v in versions:
        try:
            meta = joblib.load(v)
            result.append(
                {
                    "path": str(v),
                    "filename": v.name,
                    "trained_date": meta.get("trained_date", "unknown"),
                    "trained_at": meta.get("trained_at", "unknown"),
                    "feature_cols": meta.get("feature_cols"),
                }
            )
        except Exception as e:
            logger.warning(f"Could not read version {v.name}: {e}")
    return result


def load_model_version(version_path: str) -> CrossSectionalModel:
    """Load a specific model version by path.

    Args:
        version_path: Path to the versioned joblib file.

    Returns:
        The deserialized CrossSectionalModel.

    Raises:
        FileNotFoundError: If the version file does not exist.
        ValueError: If the file does not contain a valid model.
    """
    path = Path(version_path)
    if not path.exists():
        raise FileNotFoundError(f"Model version not found: {version_path}")

    cached = joblib.load(path)
    model = cached.get("model")
    if model is None:
        raise ValueError(f"No model found in {version_path}")
    logger.info(
        f"Loaded model version {path.name} (trained {cached.get('trained_date', 'unknown')})"
    )
    return model


def fetch_universe(tickers: list[str], period: str = MIN_HISTORY_DAYS) -> pd.DataFrame:
    """Fetch OHLCV data for the given tickers.

    Returns long-format DataFrame with columns: [ticker, open, high, low, close, volume].
    """
    raw = fetch_ohlcv(tickers, period=period)
    long = reshape_ohlcv_wide_to_long(raw)
    # Drop tickers with too little data (IPOs, delistings)
    counts = long.groupby("ticker").size()
    min_rows = 80  # Need ~80 trading days for 60-day MA + buffer
    valid_tickers = counts[counts >= min_rows].index
    dropped = set(long["ticker"].unique()) - set(valid_tickers)
    if dropped:
        logger.warning(f"Dropped {len(dropped)} tickers with insufficient history: {dropped}")
    return long[long["ticker"].isin(valid_tickers)]


def compute_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline on long-format OHLCV data."""
    featured = compute_alpha_features(long_df)
    featured = compute_cross_sectional_features(featured)
    return featured


def train_model(
    training_tickers: list[str] | None = None,
    data_path: str | None = None,
    force_retrain: bool = False,
) -> CrossSectionalModel:
    """Train a fresh LightGBM model on historical S&P 500 data.

    Uses either a cached parquet file or fetches fresh data.
    Returns a fitted CrossSectionalModel.

    If a model was already trained today, returns the cached version
    unless force_retrain=True.
    """
    # Check cache first
    if not force_retrain:
        cached_model = _load_cached_model()
        if cached_model is not None:
            return cached_model

    from python.alpha.features import compute_forward_returns, compute_residual_target

    cached = Path(data_path) if data_path else Path("data/raw/sp500_ohlcv.parquet")

    if cached.exists():
        logger.info(f"Loading training data from {cached}")
        raw = pd.read_parquet(cached)
        long = reshape_ohlcv_wide_to_long(raw)
    else:
        logger.info("No cached data found, fetching S&P 500 history for training...")
        from python.data.ingestion import fetch_sp500_tickers

        tickers = training_tickers or fetch_sp500_tickers()[:100]  # Top 100 for speed
        raw = fetch_ohlcv(tickers, period=TRAINING_LOOKBACK)
        long = reshape_ohlcv_wide_to_long(raw)

    featured = compute_alpha_features(long)
    featured = compute_cross_sectional_features(featured)

    # Try to merge macro features if available
    macro_path = Path("data/raw/macro_indicators.parquet")
    if macro_path.exists():
        from python.alpha.features import merge_macro_features

        try:
            featured = merge_macro_features(featured, str(macro_path))
        except Exception as e:
            logger.warning(f"Could not merge macro features: {e}")

    labeled = compute_forward_returns(featured, horizon=5)
    labeled = compute_residual_target(labeled, horizon=5)

    # Use only the feature columns that exist in the data
    available_cols = [c for c in FEATURE_COLS if c in labeled.columns]
    missing = set(FEATURE_COLS) - set(available_cols)
    if missing:
        logger.warning(f"Missing feature columns (will use subset): {missing}")

    labeled = labeled.dropna(subset=available_cols + ["target_5d"])

    # Date-based train/val split: reserve last 20% of dates for validation
    # with 5-day embargo to prevent target leakage (same approach as train.py)
    dates = labeled.index.get_level_values(0).unique().sort_values()
    split_date = dates[int(len(dates) * 0.8)]
    embargo_offset = pd.tseries.offsets.BDay(5)
    embargo_date = split_date + embargo_offset

    train_data = labeled.loc[labeled.index.get_level_values(0) <= split_date]
    val_data = labeled.loc[labeled.index.get_level_values(0) >= embargo_date]

    logger.info(
        f"Training on {len(train_data)} samples (up to {split_date.date()}), "
        f"validating on {len(val_data)} samples (from {embargo_date.date()})"
    )

    model = CrossSectionalModel(model_type="lightgbm", feature_cols=available_cols)

    if len(val_data) > 0:
        model.fit(train_data, target_col="target_5d", val_df=val_data)
        # Log validation IC
        val_preds = model.predict(val_data)
        ic = pd.Series(val_preds, index=val_data.index).corr(val_data["target_5d"])
        logger.info(f"Live model validation IC: {ic:.4f}")
    else:
        logger.warning("No validation data available — training without early stopping")
        model.fit(train_data, target_col="target_5d")

    # Cache the trained model
    _save_model_cache(model)

    return model


def rank_stocks(
    model: CrossSectionalModel,
    featured_df: pd.DataFrame,
    top_n: int = 10,
) -> list[str]:
    """Use the trained model to rank stocks and return top N tickers.

    Only uses the most recent date's cross-section for ranking.
    """
    # Get the latest date's data
    latest_date = featured_df.index.get_level_values(0).max()
    latest = featured_df.loc[latest_date].copy()

    # Ensure all feature columns exist and drop NaN rows
    available_cols = [c for c in model.feature_cols if c in latest.columns]
    latest = latest.dropna(subset=available_cols)

    if len(latest) == 0:
        logger.error("No valid data for prediction on latest date")
        return []

    # Predict raw scores (not ranks — we just need to sort)
    scores = model.predict(latest)
    score_series = pd.Series(scores, index=latest["ticker"].values)

    # Take top N by predicted return
    top = score_series.nlargest(top_n)
    logger.info(f"Top {top_n} stocks by ML score:\n{top}")
    return list(top.index)


def optimize_weights(
    tickers: list[str],
    period: str = MIN_HISTORY_DAYS,
    method: str = "hrp",
) -> dict[str, float]:
    """Run portfolio optimization on selected tickers.

    Returns a dict of {ticker: weight}.
    """
    raw = fetch_ohlcv(tickers, period=period)
    prices = extract_close_prices(raw)

    # Drop columns (tickers) with too many NaNs
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    prices = prices.dropna()

    if prices.empty or len(prices.columns) < 2:
        logger.warning("Not enough price data for optimization, using equal weight")
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}

    optimizer = PortfolioOptimizer(prices)

    if method == "hrp":
        weights = optimizer.hrp()
    elif method == "min_cvar":
        weights = optimizer.min_cvar()
    elif method == "risk_parity":
        weights = optimizer.risk_parity()
    else:
        logger.warning(f"Unknown method '{method}', falling back to HRP")
        weights = optimizer.hrp()

    # Convert to dict and filter out near-zero weights
    weight_dict = {t: float(w) for t, w in weights.items() if w > 0.001}

    # Renormalize after filtering
    total = sum(weight_dict.values())
    if total > 0:
        weight_dict = {t: w / total for t, w in weight_dict.items()}

    logger.info(f"Optimized weights ({method}):\n{weight_dict}")
    return weight_dict


def get_ml_weights(
    top_n: int = 10,
    method: str = "hrp",
    training_data_path: str | None = None,
) -> dict[str, float]:
    """End-to-end pipeline: train model -> rank S&P 500 -> optimize top N.

    This is the single entry point the live bot calls.

    Returns dict of {ticker: target_weight}.
    """
    logger.info(f"=== ML Pipeline: top {top_n} stocks, {method} optimization ===")

    # Step 1: Train model on historical data
    logger.info("Step 1/4: Training model...")
    model = train_model(data_path=training_data_path)

    # Step 2: Fetch recent data for the full universe to rank
    logger.info("Step 2/4: Fetching recent data for universe ranking...")
    from python.data.ingestion import fetch_sp500_tickers

    universe = fetch_sp500_tickers()
    # Fetch enough history for feature computation
    universe_data = fetch_universe(universe, period=MIN_HISTORY_DAYS)

    # Step 3: Compute features and rank
    logger.info("Step 3/4: Computing features and ranking stocks...")
    featured = compute_features(universe_data)
    top_tickers = rank_stocks(model, featured, top_n=top_n)

    if not top_tickers:
        logger.error("ML pipeline produced no stock picks — aborting")
        return {}

    # Step 4: Optimize weights for the selected stocks
    logger.info(f"Step 4/4: Optimizing weights for {top_tickers}...")
    weights = optimize_weights(top_tickers, method=method)

    logger.info(f"=== ML Pipeline complete: {len(weights)} positions ===")
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {ticker}: {w:.1%}")

    return weights
