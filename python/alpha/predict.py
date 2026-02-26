"""Live inference pipeline: fetch data -> features -> train -> rank -> optimize weights.

This module is the glue between the ML model and the live trading bot.
It handles the full pipeline from raw OHLCV data to target portfolio weights.
"""

import hashlib
import json
import logging
import random
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from python.alpha.features import (
    FEATURE_NEUTRAL_DEFAULTS,
    compute_alpha_features,
    compute_cross_sectional_features,
    compute_winsorize_bounds,
    load_winsorize_bounds,
    save_winsorize_bounds,
)
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.data.ingestion import (
    extract_close_prices,
    fetch_ohlcv,
    reshape_ohlcv_wide_to_long,
)
from python.data.sectors import enforce_sector_constraints, get_sector_map
from python.monitoring.drift import DriftDetector
from python.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

# Minimum history needed for feature computation (60-day MA + 20-day rolling).
# H-HRP fix: increased from 6mo to 1y.  HRP on 6 months of 10-stock data
# produces noisy correlation estimates; 1 year provides ~252 observations
# which is sufficient for stable hierarchical clustering.
MIN_HISTORY_DAYS = "1y"

# Training lookback — how much S&P 500 history to train the model on.
# H-SURV fix: reduced from 5y to 2y.  With 5y lookback using *current*
# S&P 500 constituents, survivorship bias inflates apparent performance:
# companies removed due to bankruptcy/delisting/M&A during the window are
# excluded.  2y reduces the bias magnitude from ~1-2% annual return
# inflation (5y) to ~0.3-0.5% (2y) while still providing sufficient
# training data (~500 dates × 500 tickers = 250k samples).
TRAINING_LOOKBACK = "2y"

# R3-P-9/P-10 fix: resolve paths relative to project root, not CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE_DIR = _PROJECT_ROOT / "data" / "models"
MODEL_CACHE_FILE = MODEL_CACHE_DIR / "latest_model.joblib"
FEATURE_REFERENCE_FILE = MODEL_CACHE_DIR / "feature_reference.parquet"

# Model versioning — keep last N versions (Fix #45)
MAX_MODEL_VERSIONS = 10

# R3-P-8 fix: module-level storage for last drift report (replaces function attribute)
_last_drift_report: dict | None = None


def _load_cached_model() -> CrossSectionalModel | None:
    """Load a cached model if it was trained today. Returns None if stale or missing."""
    if not MODEL_CACHE_FILE.exists():
        return None

    try:
        cached = joblib.load(MODEL_CACHE_FILE)
        trained_date = cached.get("trained_date")
        model = cached.get("model")

        # R3-P-14 fix: use NY timezone for staleness check (market-day aligned)
        from zoneinfo import ZoneInfo

        today_ny = date.today()  # fallback
        try:
            today_ny = datetime.now(tz=ZoneInfo("America/New_York")).date()
        except Exception:
            pass
        if trained_date == today_ny.isoformat() and model is not None:
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
    # R3-P-14 fix: use NY timezone for staleness key
    try:
        from zoneinfo import ZoneInfo as _ZI

        today = datetime.now(tz=_ZI("America/New_York")).date().isoformat()
    except Exception:
        today = date.today().isoformat()
    now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

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


def _save_feature_reference(featured_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Save a random sample of training features as the drift detection reference.

    Stores a parquet file with the feature columns from the training data.
    A sample of up to 5000 rows is saved to keep file size reasonable while
    preserving distributional properties.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    available = [c for c in feature_cols if c in featured_df.columns]
    if not available:
        logger.warning("No feature columns found for drift reference — skipping save")
        return

    ref = featured_df[available].dropna()
    if len(ref) > 5000:
        ref = ref.sample(n=5000, random_state=42)
    ref.to_parquet(FEATURE_REFERENCE_FILE)
    logger.info(f"Saved feature reference ({len(ref)} rows, {len(available)} features)")


def _load_feature_reference() -> pd.DataFrame | None:
    """Load the saved feature reference for drift detection."""
    if not FEATURE_REFERENCE_FILE.exists():
        return None
    try:
        return pd.read_parquet(FEATURE_REFERENCE_FILE)
    except Exception as e:
        logger.warning(f"Failed to load feature reference: {e}")
        return None


def check_feature_drift(
    current_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    threshold: float = 0.01,
) -> dict | None:
    """Compare current inference features against training reference.

    Args:
        current_features: DataFrame of features computed at inference time.
        feature_cols: Subset of columns to check. If None, checks all
            columns present in both reference and current data.
        threshold: KS-test p-value threshold for declaring drift (default 0.01).

    Returns:
        Drift report dict from ``DriftDetector.detect()``, or None if no
        reference is available.  The report maps feature names to dicts with
        keys: ks_statistic, p_value, psi, drifted.
    """
    reference = _load_feature_reference()
    if reference is None:
        logger.info("No feature reference available — skipping drift check")
        return None

    # Restrict to requested feature columns
    if feature_cols:
        cols = [c for c in feature_cols if c in reference.columns and c in current_features.columns]
    else:
        cols = [c for c in reference.columns if c in current_features.columns]

    if not cols:
        logger.warning("No overlapping feature columns for drift check")
        return None

    detector = DriftDetector(reference[cols], threshold=threshold)

    # R3-P-3 fix: compare only the latest date's cross-section against training.
    # The featured DataFrame has a DatetimeIndex with duplicate dates (one per ticker).
    # Using all rows would compare training data against itself (1yr overlap with 2yr training).
    latest_date = current_features.index.max()
    current_slice = current_features.loc[latest_date][cols].dropna()
    # If loc returns a Series (single row), reshape to DataFrame
    if isinstance(current_slice, pd.Series):
        current_slice = current_slice.to_frame().T

    if len(current_slice) < 10:
        logger.warning(f"Too few rows ({len(current_slice)}) for meaningful drift detection")
        return None

    report = detector.detect(current_slice)
    drifted = [k for k, v in report.items() if v["drifted"]]
    if drifted:
        logger.warning(f"Feature drift detected in {len(drifted)}/{len(cols)} features: {drifted}")
        for feat in drifted:
            r = report[feat]
            logger.warning(
                f"  {feat}: KS={r['ks_statistic']:.3f}, p={r['p_value']:.4f}, PSI={r['psi']:.3f}"
            )
    else:
        logger.info(f"No feature drift detected ({len(cols)} features checked)")

    return report


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
    """Run the full feature pipeline on long-format OHLCV data.

    Includes macro features (VIX, term spread) when the data file exists,
    so that inference-time features match what the model was trained on.

    R3-P-2 fix: loads saved winsorize bounds from training so inference
    uses identical clipping thresholds (C-WINS fix wiring).
    """
    # Load training-time winsorize bounds for consistent feature distributions
    bounds = load_winsorize_bounds()
    if bounds:
        logger.info(f"Using saved winsorize bounds ({len(bounds)} cols) for inference")

    featured = compute_alpha_features(long_df, winsorize_bounds=bounds)
    featured = compute_cross_sectional_features(featured)

    # Merge macro features at inference time (same as train_model does)
    # This prevents KeyError when the model expects 'vix' but it's missing.
    macro_path = Path("data/raw/macro_indicators.parquet")
    if macro_path.exists():
        from python.alpha.features import merge_macro_features

        try:
            featured = merge_macro_features(featured, str(macro_path))
        except Exception as e:
            logger.warning(f"Could not merge macro features at inference: {e}")

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

        # H10 warning: survivorship bias — current S&P 500 list is used to
        # fetch historical data.  Companies removed from the index (bankruptcy,
        # delisting, M&A) before today are excluded, which inflates apparent
        # model performance.  For paper trading this is acceptable; any backtest
        # should use a point-in-time membership table instead.
        logger.warning(
            "SURVIVORSHIP BIAS: training on *current* S&P 500 constituents. "
            "Delisted/removed companies are excluded from history. "
            "Use a point-in-time membership table for unbiased backtests."
        )

        # M12 fix: use a random sample instead of the first 100 alphabetically.
        # Alphabetical slicing (A-D) introduces systematic sector/name bias.
        # Seed is fixed for reproducibility across runs on the same day.
        all_tickers = fetch_sp500_tickers()
        random.seed(42)
        tickers = training_tickers or random.sample(all_tickers, min(100, len(all_tickers)))
        raw = fetch_ohlcv(tickers, period=TRAINING_LOOKBACK)
        long = reshape_ohlcv_wide_to_long(raw)

    # R3-P-2 fix: compute winsorize bounds from training data and save them.
    # First pass: compute features without bounds to get the raw distributions,
    # then compute bounds from training split, save, and re-apply.
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

    # Compute and save winsorize bounds from the full training data
    # so inference uses identical clipping thresholds (C-WINS fix)
    bounds = compute_winsorize_bounds(featured)
    save_winsorize_bounds(bounds)
    logger.info(f"Saved winsorize bounds ({len(bounds)} cols) for inference consistency")

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
    ic = None

    if len(val_data) > 0:
        model.fit(train_data, target_col="target_5d", val_df=val_data)
        # Log validation IC
        val_preds = model.predict(val_data)
        ic = pd.Series(val_preds, index=val_data.index).corr(val_data["target_5d"])
        logger.info(f"Live model validation IC: {ic:.4f}")
    else:
        logger.warning("No validation data available — training without early stopping")
        model.fit(train_data, target_col="target_5d")

    # --- MLflow tracking (best-effort: never crash training) ---
    try:
        import mlflow

        with mlflow.start_run(run_name="live_lgbm_alpha"):
            mlflow.log_params(model.params)
            mlflow.log_metric("train_size", len(train_data))
            mlflow.log_metric("val_size", len(val_data))
            mlflow.log_metric("feature_count", len(available_cols))
            if ic is not None:
                mlflow.log_metric("validation_ic", float(ic))
            mlflow.log_param("trained_date", date.today().isoformat())
            mlflow.log_param("feature_columns", json.dumps(available_cols))

            # Log feature importance as artifact
            try:
                importance = model.feature_importance()
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False, prefix="feat_importance_"
                ) as f:
                    importance.to_csv(f)
                    f.flush()
                    mlflow.log_artifact(f.name, artifact_path="feature_importance")
            except Exception as e:
                logger.debug(f"Could not log feature importance artifact: {e}")

        logger.info("MLflow run logged successfully for live_lgbm_alpha")
    except Exception as e:
        logger.warning(f"MLflow tracking failed (training unaffected): {e}")

    # Cache the trained model and save feature reference for drift detection
    _save_model_cache(model)
    _save_feature_reference(train_data, available_cols)

    return model


def rank_stocks(
    model: CrossSectionalModel,
    featured_df: pd.DataFrame,
    top_n: int = 10,
) -> list[str]:
    """Use the trained model to rank stocks and return top N tickers.

    Only uses the most recent date's cross-section for ranking.

    Handles missing features gracefully: if the model was trained with
    features that aren't available at inference time (e.g. 'vix' when
    macro data is unavailable), those features are filled with 0.0 and
    a warning is logged. This prevents KeyError crashes in production.
    """
    # Get the latest date's data
    latest_date = featured_df.index.get_level_values(0).max()
    latest = featured_df.loc[latest_date].copy()

    # Check which model features are available vs missing
    missing_cols = [c for c in model.feature_cols if c not in latest.columns]

    if missing_cols:
        # M11 fix: use domain-appropriate neutral defaults instead of 0.0.
        # Filling VIX with 0.0 tricks the model into "extreme calm" predictions;
        # FEATURE_NEUTRAL_DEFAULTS provides long-run median values per feature.
        fill_values = {col: FEATURE_NEUTRAL_DEFAULTS.get(col, 0.0) for col in missing_cols}
        logger.warning(
            f"Missing {len(missing_cols)} features at inference time: {missing_cols}. "
            f"Filling with neutral defaults: {fill_values} — predictions may be degraded."
        )
        for col in missing_cols:
            latest[col] = fill_values[col]

    latest = latest.dropna(subset=model.feature_cols)

    if len(latest) == 0:
        logger.error("No valid data for prediction on latest date")
        return []

    # Predict raw scores (not ranks — we just need to sort)
    scores = model.predict(latest)

    # H-TICKER fix: resolve ambiguity between MultiIndex level and column.
    # After .loc[latest_date], ticker may be:
    #   (a) a column named "ticker" (from compute_alpha_features), OR
    #   (b) the DataFrame index (from MultiIndex level-1).
    # Use whichever is available, preferring the column if both exist.
    if "ticker" in latest.columns:
        ticker_labels = latest["ticker"].values
    elif hasattr(latest.index, "name") and latest.index.name == "ticker":
        ticker_labels = latest.index.values
    else:
        # Last resort: try level-1 values if it's still a MultiIndex
        ticker_labels = (
            latest.index.get_level_values(-1).values
            if isinstance(latest.index, pd.MultiIndex)
            else latest.index.values
        )
    score_series = pd.Series(scores, index=ticker_labels)

    # H-SIGNAL fix: the model predicts market-neutral residual returns, but
    # the portfolio is long-only.  When the median prediction is negative
    # (i.e. the model expects most stocks to underperform), blindly buying
    # the "least negative" stocks still results in losses.  Gate net
    # exposure: if median < 0, reduce top_n proportionally.
    median_score = float(score_series.median())
    if median_score < 0:
        # Scale down: e.g. if median = -0.01 and 75th pctl = 0.005,
        # only keep stocks with clearly positive predicted residual.
        n_positive = int((score_series > 0).sum())
        effective_n = max(1, min(top_n, n_positive))
        if effective_n < top_n:
            logger.warning(
                f"H-SIGNAL: median prediction={median_score:.4f} < 0, "
                f"reducing top_n from {top_n} to {effective_n} "
                f"({n_positive} stocks have positive predicted return)"
            )
            top_n = effective_n

    # Take top N by predicted return
    top = score_series.nlargest(top_n)
    logger.info(f"Top {top_n} stocks by ML score:\n{top}")
    return list(top.index)


def optimize_weights(
    tickers: list[str],
    period: str = MIN_HISTORY_DAYS,
    method: str = "hrp",
    current_weights: dict[str, float] | None = None,
    turnover_threshold: float = 0.20,
    max_weight: float | None = None,
    price_data: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Run portfolio optimization on selected tickers.

    Args:
        tickers: List of tickers to optimize.
        period: Price history lookback period.
        method: Optimization method ('hrp', 'min_cvar', 'risk_parity').
        current_weights: Current portfolio weights for turnover-aware optimization.
            If provided, rebalancing is skipped when turnover < turnover_threshold.
        turnover_threshold: Minimum turnover to justify rebalancing (default 20%).
        max_weight: Optional cap on individual asset weight (e.g. 0.30).
            Passed through to PortfolioOptimizer for post-optimization capping.
        price_data: H11 fix — pre-fetched raw OHLCV DataFrame from yfinance.
            If provided, skips the second fetch call and uses this data directly.
            This prevents data inconsistency from two independent HTTP calls.

    Returns a dict of {ticker: weight}.
    """
    if price_data is not None:
        raw = price_data
    else:
        raw = fetch_ohlcv(tickers, period=period)
    prices = extract_close_prices(raw)

    # R3-P-1 fix: filter to only the selected tickers.
    # When price_data contains the full 500-ticker universe (H11 reuse),
    # we must restrict to just the top-N tickers passed in `tickers`.
    available_tickers = [t for t in tickers if t in prices.columns]
    if available_tickers:
        prices = prices[available_tickers]
    else:
        logger.warning("None of the selected tickers found in price data")

    # Drop columns (tickers) with too many NaNs
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    prices = prices.dropna()

    if prices.empty or len(prices.columns) < 2:
        logger.warning("Not enough price data for optimization, using equal weight")
        # Use surviving tickers (those with valid price data), not the original list
        surviving_tickers = list(prices.columns) if not prices.empty else tickers
        if not surviving_tickers:
            surviving_tickers = tickers
        n = len(surviving_tickers)
        return {t: 1.0 / n for t in surviving_tickers}

    # Convert current_weights dict to Series for optimizer
    cw_series = None
    if current_weights:
        cw_series = pd.Series(current_weights)

    optimizer = PortfolioOptimizer(
        prices,
        max_weight=max_weight,
        current_weights=cw_series,
        turnover_threshold=turnover_threshold,
    )

    # Use turnover-aware optimization if current weights provided
    if cw_series is not None:
        weights = optimizer.optimize_with_turnover_penalty(method=method)
    elif method == "hrp":
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

    # Apply sector constraints (Phase 2: Risk Management)
    sector_map = get_sector_map(list(weight_dict.keys()))
    weight_dict = enforce_sector_constraints(weight_dict, sector_map=sector_map)

    logger.info(f"Optimized weights ({method}, sector-constrained):\n{weight_dict}")
    return weight_dict


def get_ml_weights(
    top_n: int = 10,
    method: str = "hrp",
    training_data_path: str | None = None,
    current_weights: dict[str, float] | None = None,
    turnover_threshold: float = 0.20,
    max_weight: float | None = None,
) -> dict[str, float]:
    """End-to-end pipeline: train model -> rank S&P 500 -> optimize top N.

    This is the single entry point the live bot calls.

    Args:
        top_n: Number of top stocks to select.
        method: Optimization method.
        training_data_path: Optional path to cached training data.
        current_weights: Current portfolio weights for turnover-aware optimization.
        turnover_threshold: Minimum turnover to justify rebalancing.
        max_weight: Optional cap on individual asset weight (e.g. 0.30).

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

    # H11 fix: fetch raw OHLCV once and reuse for both ranking and optimization.
    # Previously, fetch_universe fetched data for ranking, then optimize_weights
    # fetched again independently — the two HTTP calls could return different data.
    raw_ohlcv = fetch_ohlcv(universe, period=MIN_HISTORY_DAYS)
    universe_data = reshape_ohlcv_wide_to_long(raw_ohlcv)
    # Apply minimum-history filter
    counts = universe_data.groupby("ticker").size()
    min_rows = 80
    valid_tickers = counts[counts >= min_rows].index
    dropped = set(universe_data["ticker"].unique()) - set(valid_tickers)
    if dropped:
        logger.warning(f"Dropped {len(dropped)} tickers with insufficient history: {dropped}")
    universe_data = universe_data[universe_data["ticker"].isin(valid_tickers)]

    # Step 3: Compute features, check drift, and rank
    logger.info("Step 3/4: Computing features and ranking stocks...")
    featured = compute_features(universe_data)

    # Drift detection: compare current features against training reference
    drift_report = check_feature_drift(featured, feature_cols=model.feature_cols)
    # R3-P-8 fix: store in module-level variable (not function attribute)
    global _last_drift_report
    _last_drift_report = drift_report

    # Log drift metrics to MLflow (best-effort)
    if drift_report:
        try:
            import mlflow

            with mlflow.start_run(run_name="live_drift_check"):
                drifted_features = [k for k, v in drift_report.items() if v["drifted"]]
                psi_values = [v["psi"] for v in drift_report.values()]
                mlflow.log_metric("n_drifted_features", len(drifted_features))
                mlflow.log_metric("n_features_checked", len(drift_report))
                mlflow.log_metric("max_psi", max(psi_values) if psi_values else 0.0)
                if drifted_features:
                    mlflow.log_param("drifted_features", json.dumps(drifted_features))
        except Exception as e:
            logger.warning(f"MLflow drift logging failed (non-fatal): {e}")

    top_tickers = rank_stocks(model, featured, top_n=top_n)

    if not top_tickers:
        logger.error("ML pipeline produced no stock picks — aborting")
        return {}

    # Step 4: Optimize weights using the SAME data (H11 fix — no double fetch)
    logger.info(f"Step 4/4: Optimizing weights for {top_tickers}...")
    weights = optimize_weights(
        top_tickers,
        method=method,
        current_weights=current_weights,
        turnover_threshold=turnover_threshold,
        max_weight=max_weight,
        price_data=raw_ohlcv,  # H11: reuse pre-fetched data
    )

    logger.info(f"=== ML Pipeline complete: {len(weights)} positions ===")
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {ticker}: {w:.1%}")

    return weights
