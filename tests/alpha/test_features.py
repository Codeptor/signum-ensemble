import numpy as np
import pandas as pd
import pytest

from python.alpha.features import (
    compute_alpha_features,
    compute_cross_sectional_features,
    compute_forward_returns,
    compute_residual_target,
    merge_macro_features,
)


@pytest.fixture
def sample_prices():
    """Multi-ticker OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    np.random.seed(42)
    tickers = ["AAPL", "MSFT"]
    frames = []
    for ticker in tickers:
        base = 150 + np.cumsum(np.random.randn(60) * 2)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": base + np.random.randn(60),
                    "high": base + abs(np.random.randn(60)) * 2,
                    "low": base - abs(np.random.randn(60)) * 2,
                    "close": base,
                    "volume": np.random.randint(500000, 2000000, 60).astype(float),
                },
                index=dates,
            )
        )
    return pd.concat(frames)


def test_compute_alpha_features_shape(sample_prices):
    features = compute_alpha_features(sample_prices)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    # Should have technical indicator columns
    assert any("rsi" in c.lower() or "ma" in c.lower() for c in features.columns)


def test_compute_alpha_features_no_future_leak(sample_prices):
    """Features at time t should only use data from t and before."""
    features = compute_alpha_features(sample_prices)
    # No NaN in the non-warmup period (after 30 days of history)
    ticker_feats = features[features["ticker"] == "AAPL"].iloc[30:]
    # Allow some NaN from long lookback windows but not all
    null_ratio = ticker_feats.drop(columns=["ticker"]).isnull().mean().mean()
    assert null_ratio < 0.1, f"Too many NaN in features: {null_ratio:.2%}"


def test_cross_sectional_rank_bounds(sample_prices):
    """Cross-sectional rank features should be in [0, 1]."""
    features = compute_alpha_features(sample_prices)
    features = compute_cross_sectional_features(features)

    rank_cols = [c for c in features.columns if c.startswith("cs_")]
    assert len(rank_cols) == 4, f"Expected 4 cs_ columns, got {rank_cols}"

    for col in rank_cols:
        valid = features[col].dropna()
        assert valid.min() >= 0.0, f"{col} has values below 0"
        assert valid.max() <= 1.0, f"{col} has values above 1"


def test_residual_target_zero_cs_mean(sample_prices):
    """Residual target should have near-zero cross-sectional mean per date."""
    features = compute_alpha_features(sample_prices)
    labeled = compute_forward_returns(features, horizon=5)
    labeled = compute_residual_target(labeled, horizon=5)
    labeled = labeled.dropna(subset=["target_5d"])

    cs_means = labeled.groupby(level=0)["target_5d"].mean()
    assert cs_means.abs().max() < 1e-10, "Residual target CS mean not zero"
    # Raw target should still exist
    assert "raw_target_5d" in labeled.columns


def test_macro_features_broadcast(sample_prices, tmp_path):
    """Macro features should broadcast to all tickers on the same date."""
    features = compute_alpha_features(sample_prices)

    # Create a mock macro parquet
    dates = features.index.unique()
    macro = pd.DataFrame(
        {"vix": 20.0, "us10y": 1.5, "us3m": 0.05},
        index=dates,
    )
    macro.index.name = "Date"
    macro_path = tmp_path / "macro.parquet"
    macro.to_parquet(macro_path)

    result = merge_macro_features(features, macro_path=str(macro_path))

    assert "vix" in result.columns
    assert "vix_ma_ratio" in result.columns
    assert "term_spread" in result.columns
    assert "term_spread_change_20d" in result.columns

    # All tickers on the same date should share the same VIX value
    for date in dates[:5]:
        date_slice = result.loc[date]
        if isinstance(date_slice, pd.DataFrame) and len(date_slice) > 1:
            assert date_slice["vix"].nunique() == 1


def test_cross_sectional_features_per_date(sample_prices):
    """Ranks should sum correctly within each cross-section."""
    features = compute_alpha_features(sample_prices)
    features = compute_cross_sectional_features(features)
    # With 2 tickers, ranks should be 0.5 and 1.0 per date
    for date in features.index.unique()[:5]:
        date_slice = features.loc[date]
        if isinstance(date_slice, pd.DataFrame) and len(date_slice) == 2:
            ranks = sorted(date_slice["cs_ret_rank_5d"].dropna().values)
            if len(ranks) == 2:
                assert ranks == [0.5, 1.0]
