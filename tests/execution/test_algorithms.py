"""Tests for execution algorithms."""

import numpy as np
import pytest
from datetime import datetime, timedelta

from python.execution.algorithms import (
    ChildSlice,
    ExecutionReport,
    ISAlgo,
    ParentOrder,
    ParticipationAlgo,
    TWAPAlgo,
    VWAPAlgo,
    VolumeProfile,
    create_algo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    qty=10000,
    minutes=60,
    side="BUY",
    urgency=0.5,
):
    start = datetime(2024, 6, 15, 10, 0, 0)
    end = start + timedelta(minutes=minutes)
    return ParentOrder("AAPL", side, qty, start, end, urgency=urgency)


def _flat_prices(n, mid=150.0):
    return np.full(n, mid)


# ---------------------------------------------------------------------------
# ParentOrder
# ---------------------------------------------------------------------------


class TestParentOrder:
    def test_duration_seconds(self):
        order = _make_order(minutes=60)
        assert order.duration_seconds == 3600

    def test_n_minutes(self):
        order = _make_order(minutes=30)
        assert order.n_minutes == 30


# ---------------------------------------------------------------------------
# VolumeProfile
# ---------------------------------------------------------------------------


class TestVolumeProfile:
    def test_uniform_sums_to_one(self):
        vp = VolumeProfile.uniform()
        assert sum(vp.fractions) == pytest.approx(1.0)

    def test_u_shaped_sums_to_one(self):
        vp = VolumeProfile.u_shaped()
        assert sum(vp.fractions) == pytest.approx(1.0)

    def test_u_shaped_edges_higher(self):
        vp = VolumeProfile.u_shaped(bucket_minutes=5)
        # First and last bucket should have more volume than middle
        mid = len(vp.fractions) // 2
        assert vp.fractions[0] > vp.fractions[mid]
        assert vp.fractions[-1] > vp.fractions[mid]


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------


class TestTWAP:
    def test_schedule_sums_to_total(self):
        order = _make_order(qty=10000)
        algo = TWAPAlgo(order, n_slices=20)
        schedule = algo.generate_schedule()
        total = sum(s.target_qty for s in schedule)
        assert total == 10000

    def test_n_slices_correct(self):
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, n_slices=10)
        schedule = algo.generate_schedule()
        assert len(schedule) == 10

    def test_slices_roughly_equal(self):
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, n_slices=10)
        schedule = algo.generate_schedule()
        qtys = [s.target_qty for s in schedule]
        assert max(qtys) - min(qtys) <= 1  # At most 1 share difference

    def test_timing_evenly_spaced(self):
        order = _make_order(qty=1000, minutes=60)
        algo = TWAPAlgo(order, n_slices=6)
        schedule = algo.generate_schedule()
        gaps = [
            (schedule[i + 1].target_time - schedule[i].target_time).total_seconds()
            for i in range(len(schedule) - 1)
        ]
        # All gaps should be equal (10 minutes = 600 seconds)
        assert all(g == pytest.approx(600, abs=1) for g in gaps)

    def test_last_slice_is_final(self):
        order = _make_order()
        algo = TWAPAlgo(order, n_slices=5)
        schedule = algo.generate_schedule()
        assert schedule[-1].is_final

    def test_randomize(self):
        order = _make_order(qty=10000)
        algo = TWAPAlgo(order, n_slices=20, randomize_pct=0.2, seed=42)
        schedule = algo.generate_schedule()
        total = sum(s.target_qty for s in schedule)
        assert total == 10000
        qtys = [s.target_qty for s in schedule]
        # With randomization, not all slices should be exactly equal
        assert len(set(qtys)) > 1

    def test_name(self):
        algo = TWAPAlgo(_make_order(), n_slices=5)
        assert algo.name == "TWAP"


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_schedule_sums_to_total(self):
        order = _make_order(qty=10000, minutes=60)
        algo = VWAPAlgo(order)
        schedule = algo.generate_schedule()
        total = sum(s.target_qty for s in schedule)
        assert total == 10000

    def test_more_at_edges_u_shape(self):
        order = _make_order(qty=10000, minutes=60)
        vp = VolumeProfile.u_shaped(trading_minutes=60, bucket_minutes=5)
        algo = VWAPAlgo(order, volume_profile=vp)
        schedule = algo.generate_schedule()
        qtys = [s.target_qty for s in schedule]
        mid = len(qtys) // 2
        # U-shape: first slice should be larger than middle
        assert qtys[0] >= qtys[mid]

    def test_uniform_profile(self):
        order = _make_order(qty=1000, minutes=30)
        vp = VolumeProfile.uniform(trading_minutes=30, bucket_minutes=5)
        algo = VWAPAlgo(order, volume_profile=vp)
        schedule = algo.generate_schedule()
        qtys = [s.target_qty for s in schedule]
        # Uniform → slices should be roughly equal
        assert max(qtys) - min(qtys) <= 5

    def test_name(self):
        algo = VWAPAlgo(_make_order())
        assert algo.name == "VWAP"


# ---------------------------------------------------------------------------
# Implementation Shortfall
# ---------------------------------------------------------------------------


class TestIS:
    def test_schedule_sums_to_total(self):
        order = _make_order(qty=10000, urgency=0.5)
        algo = ISAlgo(order, n_steps=20)
        schedule = algo.generate_schedule()
        total = sum(s.target_qty for s in schedule)
        assert total == 10000

    def test_high_urgency_front_loaded(self):
        order = _make_order(qty=10000, urgency=0.9)
        algo = ISAlgo(order, n_steps=20)
        schedule = algo.generate_schedule()
        # First half should have more shares than second half
        half = len(schedule) // 2
        first_half = sum(s.target_qty for s in schedule[:half])
        second_half = sum(s.target_qty for s in schedule[half:])
        assert first_half >= second_half

    def test_low_urgency_nearly_uniform(self):
        order = _make_order(qty=10000, urgency=0.0)
        algo = ISAlgo(order, n_steps=10)
        schedule = algo.generate_schedule()
        qtys = [s.target_qty for s in schedule]
        # Low urgency → nearly TWAP-like
        cv = np.std(qtys) / max(np.mean(qtys), 1e-8)
        assert cv < 0.5  # Coefficient of variation should be low

    def test_trajectory_shape(self):
        order = _make_order(qty=10000, urgency=0.7)
        algo = ISAlgo(order, n_steps=20)
        trajectory = algo._optimal_trajectory()
        assert len(trajectory) == 20
        assert trajectory.sum() == pytest.approx(10000, abs=1)

    def test_name(self):
        algo = ISAlgo(_make_order())
        assert algo.name == "IS_AlmgrenChriss"


# ---------------------------------------------------------------------------
# Participation (POV)
# ---------------------------------------------------------------------------


class TestParticipation:
    def test_schedule_sums_to_total(self):
        order = _make_order(qty=5000, minutes=60)
        algo = ParticipationAlgo(order, target_rate=0.10, expected_adv=1_000_000)
        schedule = algo.generate_schedule()
        total = sum(s.target_qty for s in schedule)
        assert total == 5000

    def test_respects_target_rate(self):
        order = _make_order(qty=100000, minutes=60)
        algo = ParticipationAlgo(
            order, target_rate=0.10, expected_adv=1_000_000, check_interval_minutes=5
        )
        schedule = algo.generate_schedule()
        # Each slice should be roughly target_rate * vol_per_interval
        vol_per_interval = 1_000_000 * 5 / 390
        expected_per_slice = int(vol_per_interval * 0.10)
        # First few slices should be close to expected
        for s in schedule[:3]:
            assert abs(s.target_qty - expected_per_slice) < expected_per_slice * 0.5

    def test_name(self):
        algo = ParticipationAlgo(_make_order())
        assert algo.name == "POV"


# ---------------------------------------------------------------------------
# Simulation / ExecutionReport
# ---------------------------------------------------------------------------


class TestSimulation:
    def test_simulation_returns_report(self):
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, n_slices=10)
        prices = _flat_prices(10)
        report = algo.simulate(prices)
        assert isinstance(report, ExecutionReport)

    def test_total_filled(self):
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, n_slices=10)
        report = algo.simulate(_flat_prices(10))
        assert report.total_filled == 1000

    def test_fill_rate_100pct(self):
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, n_slices=10)
        report = algo.simulate(_flat_prices(10))
        assert report.fill_rate == pytest.approx(1.0)

    def test_buy_slippage_positive(self):
        """Buy orders should have positive slippage (pay more than mid)."""
        order = _make_order(qty=1000, side="BUY")
        algo = TWAPAlgo(order, n_slices=5)
        report = algo.simulate(_flat_prices(5, mid=100.0), spread_bps=10)
        assert report.is_bps > 0

    def test_sell_slippage_positive(self):
        """Sell orders should also show positive IS (receive less than mid)."""
        order = _make_order(qty=1000, side="SELL")
        algo = TWAPAlgo(order, n_slices=5)
        report = algo.simulate(_flat_prices(5, mid=100.0), spread_bps=10)
        assert report.is_bps > 0

    def test_summary_string(self):
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, n_slices=5)
        report = algo.simulate(_flat_prices(5))
        s = report.summary()
        assert "TWAP" in s
        assert "filled" in s


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_twap(self):
        algo = create_algo("TWAP", _make_order(), n_slices=5)
        assert isinstance(algo, TWAPAlgo)

    def test_vwap(self):
        algo = create_algo("VWAP", _make_order())
        assert isinstance(algo, VWAPAlgo)

    def test_is(self):
        algo = create_algo("IS", _make_order(), n_steps=10)
        assert isinstance(algo, ISAlgo)

    def test_pov(self):
        algo = create_algo("POV", _make_order())
        assert isinstance(algo, ParticipationAlgo)

    def test_case_insensitive(self):
        algo = create_algo("twap", _make_order(), n_slices=5)
        assert isinstance(algo, TWAPAlgo)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown algo"):
            create_algo("UNKNOWN", _make_order())
