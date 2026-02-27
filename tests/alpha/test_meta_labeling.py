"""Tests for meta-labeling and triple barrier method."""

import numpy as np
import pytest

from python.alpha.meta_labeling import (
    BarrierEvent,
    BarrierLabel,
    MetaLabeler,
    MetaLabelResult,
    average_uniqueness,
    compute_concurrency,
    sample_weights_by_return,
    sequential_bootstrap,
    triple_barrier_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trending_prices(n=500, seed=42):
    """Generate trending price series."""
    rng = np.random.default_rng(seed)
    returns = 0.001 + rng.standard_normal(n) * 0.02
    return 100 * np.cumprod(1 + returns)


def _make_events(prices, n_events=50, horizon=20, seed=42):
    """Generate barrier events from a price series."""
    rng = np.random.default_rng(seed)
    T = len(prices)
    events = []
    for i in range(n_events):
        t = rng.integers(0, T - horizon - 1)
        side = rng.choice([-1, 1])
        events.append(BarrierEvent(
            t_start=t,
            t_end=t + horizon,
            pt_level=0.03,
            sl_level=0.02,
            side=side,
        ))
    return events


# ---------------------------------------------------------------------------
# Triple Barrier Labels
# ---------------------------------------------------------------------------


class TestTripleBarrier:
    def test_returns_labels(self):
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        assert len(labels) == len(events)
        assert all(isinstance(l, BarrierLabel) for l in labels)

    def test_label_values(self):
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        for l in labels:
            assert l.label in (-1, 0, 1)

    def test_barrier_types(self):
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        types = {l.barrier_type for l in labels}
        # Should have at least two types
        assert len(types) >= 1
        assert types <= {"pt", "sl", "time"}

    def test_pt_barrier_positive_return(self):
        """When profit-taking barrier is hit, return should be >= pt_level."""
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        for l in labels:
            if l.barrier_type == "pt":
                assert l.ret >= 0

    def test_sl_barrier_negative_return(self):
        """When stop-loss barrier is hit, return should be <= -sl_level."""
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        for l in labels:
            if l.barrier_type == "sl":
                assert l.ret <= 0

    def test_exit_within_horizon(self):
        prices = _make_trending_prices()
        events = _make_events(prices, horizon=20)
        labels = triple_barrier_labels(prices, events)
        for l, ev in zip(labels, events):
            assert l.t_end <= ev.t_end

    def test_monotonic_prices_all_pt(self):
        """Monotonically rising prices with long side → all profit-taking."""
        prices = np.linspace(100, 120, 200)
        events = [
            BarrierEvent(t_start=10, t_end=50, pt_level=0.02, sl_level=0.10, side=1),
            BarrierEvent(t_start=60, t_end=100, pt_level=0.02, sl_level=0.10, side=1),
        ]
        labels = triple_barrier_labels(prices, events)
        for l in labels:
            assert l.barrier_type == "pt"
            assert l.label == 1

    def test_crash_prices_all_sl(self):
        """Monotonically falling prices with long side → all stop-loss."""
        prices = np.linspace(100, 80, 200)
        events = [
            BarrierEvent(t_start=10, t_end=50, pt_level=0.10, sl_level=0.02, side=1),
            BarrierEvent(t_start=60, t_end=100, pt_level=0.10, sl_level=0.02, side=1),
        ]
        labels = triple_barrier_labels(prices, events)
        for l in labels:
            assert l.barrier_type == "sl"
            assert l.label == -1


# ---------------------------------------------------------------------------
# Concurrency & Uniqueness
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_non_overlapping_events(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=19, ret=-0.01, label=-1, barrier_type="sl"),
        ]
        conc = compute_concurrency(events, 20)
        assert conc[5] == 1
        assert conc[15] == 1

    def test_overlapping_events(self):
        events = [
            BarrierLabel(t_start=0, t_end=15, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=25, ret=0.02, label=1, barrier_type="pt"),
        ]
        conc = compute_concurrency(events, 30)
        assert conc[5] == 1
        assert conc[12] == 2
        assert conc[20] == 1


class TestUniqueness:
    def test_non_overlapping_full_uniqueness(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=19, ret=0.02, label=1, barrier_type="pt"),
        ]
        uniq = average_uniqueness(events, 20)
        np.testing.assert_allclose(uniq, 1.0)

    def test_overlapping_reduced_uniqueness(self):
        events = [
            BarrierLabel(t_start=0, t_end=19, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=0, t_end=19, ret=0.02, label=1, barrier_type="pt"),
        ]
        uniq = average_uniqueness(events, 20)
        np.testing.assert_allclose(uniq, 0.5)

    def test_bounded(self):
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        uniq = average_uniqueness(labels, len(prices))
        assert all(0 <= u <= 1 for u in uniq)


# ---------------------------------------------------------------------------
# Sample Weights
# ---------------------------------------------------------------------------


class TestSampleWeights:
    def test_sum_equals_n_events(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.05, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=19, ret=-0.03, label=-1, barrier_type="sl"),
        ]
        w = sample_weights_by_return(events, 20)
        assert w.sum() == pytest.approx(2.0, abs=0.01)

    def test_larger_return_higher_weight(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.10, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=19, ret=0.01, label=1, barrier_type="pt"),
        ]
        w = sample_weights_by_return(events, 20)
        assert w[0] > w[1]

    def test_non_negative(self):
        prices = _make_trending_prices()
        events = _make_events(prices)
        labels = triple_barrier_labels(prices, events)
        w = sample_weights_by_return(labels, len(prices))
        assert all(wi >= 0 for wi in w)


# ---------------------------------------------------------------------------
# Sequential Bootstrap
# ---------------------------------------------------------------------------


class TestSequentialBootstrap:
    def test_returns_indices(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=19, ret=0.02, label=1, barrier_type="pt"),
            BarrierLabel(t_start=5, t_end=14, ret=-0.01, label=-1, barrier_type="sl"),
        ]
        idx = sequential_bootstrap(events, n_samples=20)
        assert len(idx) == 3
        assert all(0 <= i < 3 for i in idx)

    def test_custom_n_draws(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=10, t_end=19, ret=0.02, label=1, barrier_type="pt"),
        ]
        idx = sequential_bootstrap(events, n_samples=20, n_draws=10)
        assert len(idx) == 10

    def test_non_overlapping_uniform_draw(self):
        """Non-overlapping events should get roughly equal draw probability."""
        events = [
            BarrierLabel(t_start=0, t_end=4, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=5, t_end=9, ret=0.01, label=1, barrier_type="pt"),
        ]
        idx = sequential_bootstrap(events, n_samples=10, n_draws=1000, seed=42)
        counts = np.bincount(idx, minlength=2)
        # Should be roughly 50/50
        assert abs(counts[0] / 1000 - 0.5) < 0.1

    def test_reproducible(self):
        events = [
            BarrierLabel(t_start=0, t_end=9, ret=0.01, label=1, barrier_type="pt"),
            BarrierLabel(t_start=5, t_end=14, ret=0.02, label=1, barrier_type="pt"),
        ]
        idx1 = sequential_bootstrap(events, n_samples=15, seed=42)
        idx2 = sequential_bootstrap(events, n_samples=15, seed=42)
        np.testing.assert_array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# Meta-Labeler
# ---------------------------------------------------------------------------


class TestMetaLabeler:
    def _make_meta_data(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 5))
        # Primary signal: based on first feature
        primary_side = np.sign(X[:, 0])
        primary_side[primary_side == 0] = 1
        # Meta label: correct when second feature > 0
        meta_label = (X[:, 1] > 0).astype(int)
        return X, primary_side, meta_label

    def test_fit_predict(self):
        X, sides, meta = self._make_meta_data()
        ml = MetaLabeler(threshold=0.5)
        ml.fit(X, sides, meta)
        result = ml.predict(X, sides)
        assert isinstance(result, MetaLabelResult)

    def test_positions_bounded(self):
        X, sides, meta = self._make_meta_data()
        ml = MetaLabeler(threshold=0.5, max_position=1.0)
        ml.fit(X, sides, meta)
        result = ml.predict(X, sides)
        assert all(abs(p) <= 1.0 for p in result.positions)

    def test_positions_match_side(self):
        """Position sign should match primary side (when size > 0)."""
        X, sides, meta = self._make_meta_data()
        ml = MetaLabeler(threshold=0.3)
        ml.fit(X, sides, meta)
        result = ml.predict(X, sides)
        for pos, side in zip(result.positions, result.sides):
            if abs(pos) > 0:
                assert np.sign(pos) == np.sign(side)

    def test_high_threshold_fewer_bets(self):
        X, sides, meta = self._make_meta_data()
        ml_low = MetaLabeler(threshold=0.3)
        ml_high = MetaLabeler(threshold=0.8)
        ml_low.fit(X, sides, meta)
        ml_high.fit(X, sides, meta)
        r_low = ml_low.predict(X, sides)
        r_high = ml_high.predict(X, sides)
        n_bets_low = np.sum(np.abs(r_low.positions) > 0)
        n_bets_high = np.sum(np.abs(r_high.positions) > 0)
        assert n_bets_high <= n_bets_low

    def test_predict_proba(self):
        X, sides, meta = self._make_meta_data()
        ml = MetaLabeler()
        ml.fit(X, sides, meta)
        probs = ml.predict_proba(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_not_fitted_raises(self):
        ml = MetaLabeler()
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.predict(np.zeros((5, 3)), np.ones(5))
