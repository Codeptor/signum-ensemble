"""Tests for drawdown control and dynamic protection."""

import numpy as np
import pytest

from python.portfolio.drawdown_control import (
    CPPIOverlay,
    CPPIState,
    DrawdownController,
    DrawdownState,
    LossBudgetAllocator,
    drawdown_from_path,
    estimate_recovery_time,
)


# ---------------------------------------------------------------------------
# CPPI Overlay
# ---------------------------------------------------------------------------


class TestCPPIOverlay:
    def test_initialize(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
        state = cppi.initialize(1.0)
        assert isinstance(state, CPPIState)
        assert state.floor_value == pytest.approx(0.90)
        assert state.cushion == pytest.approx(0.10)

    def test_risky_allocation_at_start(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
        state = cppi.initialize(1.0)
        # m * cushion / V = 3 * 0.10 / 1.0 = 0.30
        assert state.risky_allocation == pytest.approx(0.30)

    def test_allocation_increases_above_floor(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
        cppi.initialize(1.0)
        state = cppi.update(1.05)
        # cushion = 1.05 - 0.90 = 0.15, risky = 3*0.15/1.05 ≈ 0.4286
        assert state.risky_allocation > 0.30

    def test_allocation_zero_below_floor(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
        cppi.initialize(1.0)
        state = cppi.update(0.89)
        assert state.risky_allocation == 0.0

    def test_max_risky_cap(self):
        cppi = CPPIOverlay(floor_pct=0.50, multiplier=10.0, max_risky=1.0)
        cppi.initialize(1.0)
        state = cppi.update(1.5)
        assert state.risky_allocation <= 1.0

    def test_ratchet_floor_increases(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0, ratchet=True, ratchet_pct=0.5)
        cppi.initialize(1.0)
        state = cppi.update(1.10)
        # Floor should have ratcheted up: 0.90 + 0.5 * 0.10 = 0.95
        assert state.floor_value == pytest.approx(0.95)

    def test_no_ratchet_on_decline(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0, ratchet=True, ratchet_pct=0.5)
        cppi.initialize(1.0)
        cppi.update(1.10)  # peak
        state = cppi.update(1.05)  # decline
        # Floor should stay at 0.95 (from the peak update)
        assert state.floor_value == pytest.approx(0.95)

    def test_adjust_weights(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
        cppi.initialize(1.0)
        weights = {"AAPL": 0.5, "GOOG": 0.5}
        adjusted = cppi.adjust_weights(weights, 1.0)
        # risky_allocation = 0.30
        assert adjusted["AAPL"] == pytest.approx(0.15)
        assert adjusted["GOOG"] == pytest.approx(0.15)

    def test_adjust_weights_auto_init(self):
        cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
        weights = {"A": 1.0}
        adjusted = cppi.adjust_weights(weights, 1.0)
        assert 0 <= adjusted["A"] <= 1.0


# ---------------------------------------------------------------------------
# Drawdown Controller
# ---------------------------------------------------------------------------


class TestDrawdownController:
    def test_no_deleveraging_above_threshold(self):
        ctrl = DrawdownController(max_dd=0.10)
        state = ctrl.update(1.0)
        assert state.exposure_factor == 1.0
        assert not state.is_deleveraging

    def test_deleveraging_starts_at_max_dd(self):
        ctrl = DrawdownController(max_dd=0.10, hard_limit=0.20)
        ctrl.update(1.0)  # set peak
        state = ctrl.update(0.89)  # -11% drawdown
        assert state.is_deleveraging
        assert state.exposure_factor < 1.0

    def test_zero_exposure_at_hard_limit(self):
        ctrl = DrawdownController(max_dd=0.10, hard_limit=0.20)
        ctrl.update(1.0)
        state = ctrl.update(0.80)  # -20% drawdown
        assert state.exposure_factor == pytest.approx(0.0, abs=1e-10)

    def test_hysteresis_recovery(self):
        ctrl = DrawdownController(max_dd=0.10, hard_limit=0.20, recovery_threshold=0.05)
        ctrl.update(1.0)
        ctrl.update(0.88)  # -12%, start deleveraging
        assert ctrl._is_deleveraging
        # Recover to -4% (below recovery threshold)
        state = ctrl.update(0.96)
        assert not state.is_deleveraging
        assert state.exposure_factor == 1.0

    def test_hysteresis_no_premature_recovery(self):
        ctrl = DrawdownController(max_dd=0.10, hard_limit=0.20, recovery_threshold=0.05)
        ctrl.update(1.0)
        ctrl.update(0.88)  # start deleveraging
        # Recover to -8% (still above recovery threshold)
        state = ctrl.update(0.92)
        assert state.is_deleveraging

    def test_linear_interpolation(self):
        ctrl = DrawdownController(max_dd=0.10, hard_limit=0.20, deleverage_speed=1.0)
        ctrl.update(1.0)
        # At midpoint between max_dd and hard_limit (-15%)
        state = ctrl.update(0.85)
        assert state.exposure_factor == pytest.approx(0.5, abs=0.01)

    def test_adjust_weights(self):
        ctrl = DrawdownController(max_dd=0.10, hard_limit=0.20)
        ctrl.update(1.0)
        weights = {"A": 0.6, "B": 0.4}
        adjusted = ctrl.adjust_weights(weights, 0.85)
        assert adjusted["A"] < 0.6
        assert adjusted["B"] < 0.4

    def test_drawdown_state_values(self):
        ctrl = DrawdownController(max_dd=0.10)
        ctrl.update(1.0)
        state = ctrl.update(0.95)
        assert state.current_drawdown == pytest.approx(-0.05)
        assert state.peak_value == 1.0
        assert state.current_value == 0.95


# ---------------------------------------------------------------------------
# Loss Budget Allocator
# ---------------------------------------------------------------------------


class TestLossBudgetAllocator:
    def test_equal_allocation(self):
        alloc = LossBudgetAllocator(total_budget=0.10)
        budgets = alloc.allocate(["alpha", "beta", "gamma"])
        for s, b in budgets.items():
            assert b == pytest.approx(0.10 / 3)

    def test_weighted_allocation(self):
        alloc = LossBudgetAllocator(
            total_budget=0.10,
            strategy_weights={"alpha": 2.0, "beta": 1.0},
        )
        budgets = alloc.allocate(["alpha", "beta"])
        assert budgets["alpha"] == pytest.approx(2 / 3 * 0.10)
        assert budgets["beta"] == pytest.approx(1 / 3 * 0.10)

    def test_update_pnl_no_breach(self):
        alloc = LossBudgetAllocator(total_budget=0.10)
        budget = alloc.update_pnl("alpha", 0.01)  # slight profit
        assert not budget.is_breached
        assert budget.remaining_budget > 0

    def test_update_pnl_breach(self):
        alloc = LossBudgetAllocator(total_budget=0.02)
        alloc.update_pnl("alpha", 0.05)  # profit peak
        budget = alloc.update_pnl("alpha", -0.01)  # lost 0.06 from peak
        assert budget.is_breached

    def test_get_all_budgets(self):
        alloc = LossBudgetAllocator(total_budget=0.10)
        alloc.update_pnl("alpha", 0.01)
        alloc.update_pnl("beta", -0.01)
        all_b = alloc.get_all_budgets()
        assert len(all_b) == 2


# ---------------------------------------------------------------------------
# Recovery Time Estimation
# ---------------------------------------------------------------------------


class TestRecoveryTime:
    def test_zero_drawdown(self):
        t = estimate_recovery_time(0.0, 0.10, 0.15)
        assert t == 0.0

    def test_positive_drift_finite(self):
        t = estimate_recovery_time(0.20, 0.10, 0.15)
        assert np.isfinite(t)
        assert t > 0

    def test_negative_drift_infinite(self):
        t = estimate_recovery_time(0.20, -0.05, 0.15)
        assert t == float("inf")

    def test_larger_dd_longer_recovery(self):
        t1 = estimate_recovery_time(0.10, 0.10, 0.15)
        t2 = estimate_recovery_time(0.30, 0.10, 0.15)
        assert t2 > t1

    def test_higher_return_faster_recovery(self):
        t1 = estimate_recovery_time(0.20, 0.05, 0.15)
        t2 = estimate_recovery_time(0.20, 0.15, 0.15)
        assert t2 < t1

    def test_confidence_level(self):
        t50 = estimate_recovery_time(0.20, 0.10, 0.15, confidence=0.5)
        t90 = estimate_recovery_time(0.20, 0.10, 0.15, confidence=0.9)
        assert t90 > t50


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestDrawdownFromPath:
    def test_monotonic_up_zero_dd(self):
        path = np.array([1.0, 1.1, 1.2, 1.3])
        dd = drawdown_from_path(path)
        np.testing.assert_allclose(dd, 0.0, atol=1e-10)

    def test_known_drawdown(self):
        path = np.array([1.0, 1.1, 1.0, 1.2])
        dd = drawdown_from_path(path)
        # At index 2: (1.0 - 1.1) / 1.1 ≈ -0.0909
        assert dd[2] == pytest.approx(-1 / 11, abs=0.001)

    def test_max_drawdown(self):
        path = np.array([1.0, 0.8, 0.9, 0.7, 1.0])
        dd = drawdown_from_path(path)
        assert dd.min() == pytest.approx(-0.30, abs=0.01)

    def test_shape_preserved(self):
        path = np.array([1.0, 1.1, 0.9, 1.2, 1.0])
        dd = drawdown_from_path(path)
        assert len(dd) == len(path)
