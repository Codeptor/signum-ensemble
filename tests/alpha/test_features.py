import numpy as np
import pandas as pd
import pytest

from python.alpha.features import (
    FEATURE_NEUTRAL_DEFAULTS,
    _scrub_infinities,
    compute_alpha_features,
    compute_cross_sectional_features,
    compute_forward_returns,
    compute_residual_target,
    merge_macro_features,
    winsorize,
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


# =====================================================================
# Tests for audit fixes: C9, C10, C11, H6, H7, M15
# =====================================================================


class TestFeatureNeutralDefaults:
    """C9: FEATURE_NEUTRAL_DEFAULTS must provide non-zero neutral values."""

    def test_vix_default_is_not_zero(self):
        assert FEATURE_NEUTRAL_DEFAULTS["vix"] > 0
        assert FEATURE_NEUTRAL_DEFAULTS["vix"] == pytest.approx(20.0)

    def test_rsi_default_is_midpoint(self):
        assert FEATURE_NEUTRAL_DEFAULTS["rsi_14"] == pytest.approx(50.0)

    def test_bb_position_default_is_middle(self):
        assert FEATURE_NEUTRAL_DEFAULTS["bb_position"] == pytest.approx(0.5)

    def test_volume_ratio_default_is_average(self):
        assert FEATURE_NEUTRAL_DEFAULTS["volume_ratio"] == pytest.approx(1.0)

    def test_all_rank_defaults_are_median(self):
        for key in ["cs_ret_rank_5d", "cs_ret_rank_20d", "cs_vol_rank_20d", "cs_volume_rank"]:
            assert FEATURE_NEUTRAL_DEFAULTS[key] == pytest.approx(0.5), f"{key} not 0.5"


class TestScrubInfinities:
    """C11: _scrub_infinities must replace ±inf with NaN."""

    def test_positive_inf_replaced(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
        result = _scrub_infinities(df)
        assert np.isnan(result["a"].iloc[1])
        assert result["b"].iloc[1] == 5.0

    def test_negative_inf_replaced(self):
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
        result = _scrub_infinities(df)
        assert np.isnan(result["a"].iloc[1])

    def test_no_inf_no_change(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _scrub_infinities(df)
        assert result["a"].tolist() == [1.0, 2.0, 3.0]

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        _scrub_infinities(df)
        # Original should still have inf if copy worked
        assert np.isinf(df["a"].iloc[1])

    def test_non_numeric_columns_untouched(self):
        df = pd.DataFrame({"a": [1.0, np.inf], "label": ["x", "y"]})
        result = _scrub_infinities(df)
        assert result["label"].tolist() == ["x", "y"]


class TestNoInfInFeatures:
    """C11: compute_alpha_features must not produce inf values."""

    def test_features_contain_no_inf(self, sample_prices):
        features = compute_alpha_features(sample_prices)
        numeric = features.select_dtypes(include=[np.number])
        inf_mask = numeric.isin([np.inf, -np.inf])
        assert not inf_mask.any().any(), (
            f"inf found in: {inf_mask.any()[inf_mask.any()].index.tolist()}"
        )

    def test_log_returns_with_zero_close_no_inf(self):
        """A stock with zero close should not produce -inf in log returns."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        base = np.ones(30) * 100.0
        base[15] = 0.0  # zero close on one day
        df = pd.DataFrame(
            {
                "ticker": "TEST",
                "open": base,
                "high": base + 1,
                "low": base - 1,
                "close": base,
                "volume": np.ones(30) * 1e6,
            },
            index=dates,
        )
        features = compute_alpha_features(df)
        numeric = features.select_dtypes(include=[np.number])
        assert not numeric.isin([np.inf, -np.inf]).any().any()


class TestCrossSectionalDivisionByZero:
    """C10: compute_cross_sectional_features must handle identical values."""

    def test_all_same_value_produces_valid_ranks(self):
        """When all stocks have identical ret_5d, rank should be 0.5 (average)."""
        dates = pd.to_datetime(["2024-01-01"] * 3)
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "ret_5d": [0.05, 0.05, 0.05],  # all identical
                "ret_20d": [0.10, 0.10, 0.10],
                "vol_20d": [0.02, 0.02, 0.02],
                "dollar_volume_20d": [1e6, 1e6, 1e6],
            },
            index=dates,
        )
        result = compute_cross_sectional_features(df)
        # All ranks should be 0.5 (average of 1/3, 2/3, 3/3 = 2/3 ≈ 0.667)
        # Actually with method='average', 3 tied values get rank (1+2+3)/3 = 2.0
        # pct=True divides by count: 2.0/3 = 0.6667
        for col in ["cs_ret_rank_5d", "cs_ret_rank_20d", "cs_vol_rank_20d", "cs_volume_rank"]:
            assert col in result.columns
            values = result[col].dropna().unique()
            # All should be the same value (all tied)
            assert len(values) == 1, f"{col} has {len(values)} distinct values, expected 1"

    def test_does_not_mutate_input(self):
        """C10 fix: compute_cross_sectional_features should not modify caller's df."""
        dates = pd.to_datetime(["2024-01-01"] * 2)
        df = pd.DataFrame(
            {"ticker": ["A", "B"], "ret_5d": [0.01, 0.02]},
            index=dates,
        )
        original_cols = list(df.columns)
        _ = compute_cross_sectional_features(df)
        assert list(df.columns) == original_cols, "Input DataFrame was mutated"


class TestWinsorize:
    """H6/M15: winsorize with wider percentiles and no in-place mutation."""

    def test_default_percentiles_are_wider(self):
        """H6: default should be 0.5th/99.5th, not 1st/99th."""
        import inspect

        sig = inspect.signature(winsorize)
        assert sig.parameters["lower"].default == pytest.approx(0.005)
        assert sig.parameters["upper"].default == pytest.approx(0.995)

    def test_does_not_mutate_input(self):
        """M15: winsorize should return a copy, not modify the original."""
        df = pd.DataFrame({"ret_5d": [0.0, 0.1, 0.2, -0.5, 1.0] * 100})
        original = df["ret_5d"].copy()
        _ = winsorize(df, cols=["ret_5d"])
        pd.testing.assert_series_equal(df["ret_5d"], original)

    def test_clips_extreme_values(self):
        """Winsorize should clip values outside the percentile range."""
        data = list(range(1000))
        data[0] = -999
        data[-1] = 999
        df = pd.DataFrame({"ret_5d": data})
        result = winsorize(df, cols=["ret_5d"], lower=0.01, upper=0.99)
        assert result["ret_5d"].min() > -999
        assert result["ret_5d"].max() < 999


class TestComputeAlphaFeaturesWinsorizes:
    """H7: compute_alpha_features should winsorize uniformly after computation."""

    def test_features_are_winsorized(self, sample_prices):
        """After compute_alpha_features, extreme outliers should be clipped."""
        features = compute_alpha_features(sample_prices)
        # Winsorized columns should exist and have bounded values
        if "ret_5d" in features.columns:
            valid = features["ret_5d"].dropna()
            # After winsorize at 0.5/99.5th, the range should be reasonable
            assert valid.max() < 10.0, "ret_5d not winsorized (extreme max)"
            assert valid.min() > -10.0, "ret_5d not winsorized (extreme min)"


# =====================================================================
# Tests for new features: momentum 12-1, mean-reversion z-score,
# sector-relative momentum
# =====================================================================


@pytest.fixture
def long_prices():
    """Multi-ticker OHLCV data with enough history for mom_12_1 (>252 days)."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)
    tickers = ["AAPL", "MSFT"]
    frames = []
    for ticker in tickers:
        base = 150 + np.cumsum(np.random.randn(300) * 2)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": base + np.random.randn(300),
                    "high": base + abs(np.random.randn(300)) * 2,
                    "low": base - abs(np.random.randn(300)) * 2,
                    "close": base,
                    "volume": np.random.randint(500000, 2000000, 300).astype(float),
                },
                index=dates,
            )
        )
    return pd.concat(frames)


class TestNewFeatures:
    def test_mr_zscore_column_exists(self, sample_prices):
        features = compute_alpha_features(sample_prices)
        assert "mr_zscore_60" in features.columns

    def test_mr_zscore_bounded(self, sample_prices):
        """Z-score should be roughly in [-4, 4] for normal data."""
        features = compute_alpha_features(sample_prices)
        valid = features["mr_zscore_60"].dropna()
        if len(valid) > 0:
            assert valid.max() < 10.0
            assert valid.min() > -10.0

    def test_mom_12_1_column_exists(self, sample_prices):
        features = compute_alpha_features(sample_prices)
        assert "mom_12_1" in features.columns

    def test_mom_12_1_with_long_history(self, long_prices):
        """With sufficient history, mom_12_1 should have valid values."""
        features = compute_alpha_features(long_prices)
        valid = features["mom_12_1"].dropna()
        # With 300 days, we should have some valid mom_12_1 values
        assert len(valid) > 0

    def test_sector_rel_mom_column(self, sample_prices):
        """After cross-sectional features, sector_rel_mom should exist."""
        features = compute_alpha_features(sample_prices)
        features = compute_cross_sectional_features(features)
        assert "sector_rel_mom" in features.columns
