"""Portfolio drawdown control and dynamic protection.

Implements dynamic risk management overlays:
  1. CPPI (Constant Proportion Portfolio Insurance) — maintains a floor
     on portfolio value by dynamically adjusting risky allocation.
  2. Drawdown-triggered deleveraging — reduces exposure when drawdown
     exceeds thresholds, with hysteresis to prevent whipsawing.
  3. Max loss budget allocation — distributes loss budget across
     sub-strategies proportional to expected contribution.
  4. Recovery time estimation — expected time to recover from drawdowns
     based on drift/volatility assumptions.

Usage::

    cppi = CPPIOverlay(floor_pct=0.90, multiplier=3.0)
    new_weights = cppi.adjust(weights, portfolio_value=1.05, peak_value=1.10)

    ctrl = DrawdownController(max_dd=0.10, deleverage_speed=0.5)
    factor = ctrl.compute_exposure_factor(current_dd=-0.07)

References:
  - Black & Jones (1987), "Simplifying Portfolio Insurance"
  - Grossman & Zhou (1996), "Optimal Investment Strategies for CPPI"
  - Roncalli (2013), "Risk Parity and Beyond", Ch. 5 — drawdown management
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPPI Overlay
# ---------------------------------------------------------------------------


@dataclass
class CPPIState:
    """CPPI state at a point in time."""

    portfolio_value: float
    floor_value: float
    cushion: float  # portfolio_value - floor_value
    risky_allocation: float  # fraction in risky assets
    multiplier: float


class CPPIOverlay:
    """Constant Proportion Portfolio Insurance.

    Allocates to risky assets as: w_risky = m * (V - F) / V
    where m = multiplier, V = portfolio value, F = floor value.

    Parameters
    ----------
    floor_pct : float
        Floor as fraction of initial (or peak) portfolio value.
    multiplier : float
        CPPI multiplier (higher = more aggressive when above floor).
    ratchet : bool
        If True, floor ratchets up with new highs (lock in gains).
    ratchet_pct : float
        Fraction of new highs to add to floor (0 = no ratchet, 1 = full).
    max_risky : float
        Maximum allocation to risky assets.
    """

    def __init__(
        self,
        floor_pct: float = 0.90,
        multiplier: float = 3.0,
        ratchet: bool = False,
        ratchet_pct: float = 0.5,
        max_risky: float = 1.0,
    ):
        self.floor_pct = floor_pct
        self.multiplier = multiplier
        self.ratchet = ratchet
        self.ratchet_pct = ratchet_pct
        self.max_risky = max_risky
        self._floor = None
        self._peak = None

    def initialize(self, initial_value: float) -> CPPIState:
        """Initialize CPPI with starting portfolio value."""
        self._floor = initial_value * self.floor_pct
        self._peak = initial_value
        cushion = initial_value - self._floor
        risky = min(self.multiplier * cushion / initial_value, self.max_risky)
        return CPPIState(
            portfolio_value=initial_value,
            floor_value=self._floor,
            cushion=cushion,
            risky_allocation=max(risky, 0.0),
            multiplier=self.multiplier,
        )

    def update(self, portfolio_value: float) -> CPPIState:
        """Update CPPI allocation given current portfolio value.

        Parameters
        ----------
        portfolio_value : float

        Returns
        -------
        CPPIState
        """
        if self._floor is None:
            return self.initialize(portfolio_value)

        # Ratchet floor up
        if self.ratchet and portfolio_value > self._peak:
            gain = portfolio_value - self._peak
            self._floor += gain * self.ratchet_pct
            self._peak = portfolio_value
        elif portfolio_value > self._peak:
            self._peak = portfolio_value

        cushion = portfolio_value - self._floor
        if portfolio_value > 0 and cushion > 0:
            risky = min(self.multiplier * cushion / portfolio_value, self.max_risky)
        else:
            risky = 0.0

        return CPPIState(
            portfolio_value=portfolio_value,
            floor_value=self._floor,
            cushion=max(cushion, 0.0),
            risky_allocation=max(risky, 0.0),
            multiplier=self.multiplier,
        )

    def adjust_weights(
        self, weights: dict[str, float], portfolio_value: float
    ) -> dict[str, float]:
        """Scale portfolio weights by CPPI risky allocation.

        Parameters
        ----------
        weights : dict
            Target weights (should sum to ~1).
        portfolio_value : float

        Returns
        -------
        dict
            Adjusted weights (cash remainder implicit).
        """
        state = self.update(portfolio_value)
        factor = state.risky_allocation
        return {k: v * factor for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Drawdown-triggered Deleveraging
# ---------------------------------------------------------------------------


@dataclass
class DrawdownState:
    """Current drawdown controller state."""

    current_drawdown: float  # Negative number (e.g., -0.08)
    exposure_factor: float  # 0 to 1
    is_deleveraging: bool
    peak_value: float
    current_value: float


class DrawdownController:
    """Reduce exposure when drawdown exceeds thresholds.

    Uses hysteresis: starts deleveraging at max_dd, fully out at
    hard_limit, and doesn't re-lever until drawdown recovers to
    recovery_threshold.

    Parameters
    ----------
    max_dd : float
        Drawdown level where deleveraging begins (e.g., 0.10 = 10%).
    hard_limit : float
        Drawdown level where exposure goes to zero.
    recovery_threshold : float
        Drawdown must recover to this level before re-leveraging.
    deleverage_speed : float
        How fast to reduce exposure (1.0 = linear between max_dd and hard_limit).
    """

    def __init__(
        self,
        max_dd: float = 0.10,
        hard_limit: float = 0.20,
        recovery_threshold: float = 0.05,
        deleverage_speed: float = 1.0,
    ):
        self.max_dd = max_dd
        self.hard_limit = hard_limit
        self.recovery_threshold = recovery_threshold
        self.deleverage_speed = deleverage_speed
        self._peak = None
        self._is_deleveraging = False

    def update(self, portfolio_value: float) -> DrawdownState:
        """Update drawdown state and compute exposure factor.

        Parameters
        ----------
        portfolio_value : float

        Returns
        -------
        DrawdownState
        """
        if self._peak is None or portfolio_value > self._peak:
            self._peak = portfolio_value

        dd = (portfolio_value - self._peak) / self._peak if self._peak > 0 else 0.0
        dd_abs = abs(dd)

        # Check recovery (hysteresis)
        if self._is_deleveraging and dd_abs < self.recovery_threshold:
            self._is_deleveraging = False

        # Check if we should start deleveraging
        if dd_abs >= self.max_dd:
            self._is_deleveraging = True

        # Compute exposure factor
        if not self._is_deleveraging:
            factor = 1.0
        elif dd_abs >= self.hard_limit:
            factor = 0.0
        else:
            # Linear interpolation between max_dd and hard_limit
            range_dd = self.hard_limit - self.max_dd
            if range_dd > 0:
                progress = (dd_abs - self.max_dd) / range_dd
                factor = max(0.0, 1.0 - progress * self.deleverage_speed)
            else:
                factor = 0.0

        return DrawdownState(
            current_drawdown=dd,
            exposure_factor=factor,
            is_deleveraging=self._is_deleveraging,
            peak_value=self._peak,
            current_value=portfolio_value,
        )

    def adjust_weights(
        self, weights: dict[str, float], portfolio_value: float
    ) -> dict[str, float]:
        """Scale weights by exposure factor."""
        state = self.update(portfolio_value)
        return {k: v * state.exposure_factor for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Max Loss Budget Allocation
# ---------------------------------------------------------------------------


@dataclass
class LossBudget:
    """Loss budget allocation for a strategy."""

    strategy: str
    allocated_budget: float  # Max loss in portfolio units
    used_budget: float  # Loss consumed so far
    remaining_budget: float
    is_breached: bool


class LossBudgetAllocator:
    """Allocate and track maximum loss budgets across strategies.

    Parameters
    ----------
    total_budget : float
        Total portfolio loss budget (e.g., 0.05 = 5% of AUM).
    strategy_weights : dict[str, float]
        Relative weights for budget allocation (will be normalized).
    """

    def __init__(
        self,
        total_budget: float = 0.05,
        strategy_weights: dict[str, float] | None = None,
    ):
        self.total_budget = total_budget
        self._strategy_weights = strategy_weights or {}
        self._strategy_pnl: dict[str, float] = {}
        self._peak_values: dict[str, float] = {}

    def allocate(self, strategies: list[str]) -> dict[str, float]:
        """Allocate loss budget across strategies.

        Parameters
        ----------
        strategies : list[str]

        Returns
        -------
        dict mapping strategy name to allocated loss budget.
        """
        n = len(strategies)
        if not self._strategy_weights:
            weights = {s: 1.0 / n for s in strategies}
        else:
            total_w = sum(self._strategy_weights.get(s, 1.0) for s in strategies)
            weights = {
                s: self._strategy_weights.get(s, 1.0) / total_w
                for s in strategies
            }
        return {s: w * self.total_budget for s, w in weights.items()}

    def update_pnl(self, strategy: str, pnl: float) -> LossBudget:
        """Track cumulative PnL and check budget breach.

        Parameters
        ----------
        strategy : str
        pnl : float
            Cumulative PnL (negative = loss).

        Returns
        -------
        LossBudget
        """
        self._strategy_pnl[strategy] = pnl
        if strategy not in self._peak_values:
            self._peak_values[strategy] = 0.0
        self._peak_values[strategy] = max(self._peak_values[strategy], pnl)

        allocation = self.allocate(list(self._strategy_pnl.keys()))
        budget = allocation.get(strategy, self.total_budget / max(len(self._strategy_pnl), 1))

        # Loss = peak - current (always positive or zero)
        used = max(0.0, self._peak_values[strategy] - pnl)

        return LossBudget(
            strategy=strategy,
            allocated_budget=budget,
            used_budget=used,
            remaining_budget=max(0.0, budget - used),
            is_breached=used >= budget,
        )

    def get_all_budgets(self) -> list[LossBudget]:
        """Get budget status for all tracked strategies."""
        return [self.update_pnl(s, p) for s, p in self._strategy_pnl.items()]


# ---------------------------------------------------------------------------
# Recovery Time Estimation
# ---------------------------------------------------------------------------


def estimate_recovery_time(
    drawdown: float,
    annual_return: float,
    annual_vol: float,
    confidence: float = 0.5,
) -> float:
    """Estimate time to recover from a drawdown.

    Uses the approximation for geometric Brownian motion:
      E[T_recovery] ≈ drawdown / (mu - sigma^2/2)
    where mu is the continuous drift and sigma is the continuous vol.

    For confidence levels other than 0.5 (median), uses the
    inverse Gaussian distribution approximation.

    Parameters
    ----------
    drawdown : float
        Drawdown magnitude (positive, e.g., 0.20 for 20%).
    annual_return : float
        Expected annual return (e.g., 0.10).
    annual_vol : float
        Annual volatility (e.g., 0.15).
    confidence : float
        Probability of recovery by the returned time (0.5 = median).

    Returns
    -------
    float
        Estimated recovery time in years. Returns inf if drift <= 0.
    """
    dd = abs(drawdown)
    if dd < 1e-10:
        return 0.0

    # Continuous drift
    mu = annual_return - 0.5 * annual_vol**2
    if mu <= 0:
        return float("inf")

    # Log barrier to cross
    barrier = -np.log(1 - dd)  # ln(peak/trough) = ln(1/(1-dd))

    # Median recovery time (inverse Gaussian: median ≈ mean for small vol)
    mean_time = barrier / mu

    if abs(confidence - 0.5) < 0.01:
        return float(mean_time)

    # Approximate quantile using inverse Gaussian CDF
    # For the first passage time of BM with drift mu and diffusion sigma:
    # The CDF has a known form involving the normal distribution.
    # We use a simple scaling: T_q ≈ mean_time * correction
    from scipy.stats import norm

    z = norm.ppf(confidence)
    # Approximate: variance of recovery time ≈ barrier * sigma^2 / mu^3
    var_time = barrier * annual_vol**2 / mu**3
    std_time = np.sqrt(max(var_time, 0))
    return float(max(mean_time + z * std_time, 0.0))


def drawdown_from_path(values: np.ndarray) -> np.ndarray:
    """Compute running drawdown from a value series.

    Parameters
    ----------
    values : np.ndarray
        Portfolio value path.

    Returns
    -------
    np.ndarray
        Drawdown series (negative values).
    """
    peak = np.maximum.accumulate(values)
    return (values - peak) / np.maximum(peak, 1e-10)
