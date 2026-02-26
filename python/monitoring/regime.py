"""Market regime detection for risk management.

Monitors VIX and SPY drawdown to classify the current market regime as
normal, caution, or halt. Used by the live bot to scale exposure or stop
trading entirely during extreme conditions.

Regime thresholds (from improvement plan):
- Normal: VIX < 30, SPY drawdown < 10%
- Caution: VIX 30-40 or SPY drawdown 10-15% → reduce exposure 50%
- Halt: VIX > 40 or SPY drawdown > 15% → close all positions
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Snapshot of current market regime."""

    regime: str  # "normal", "caution", "halt"
    vix: float
    spy_drawdown: float
    exposure_multiplier: float  # 1.0, 0.5, or 0.0
    message: str


class RegimeDetector:
    """Detect market regime for risk management.

    Uses VIX level and SPY drawdown from recent highs to classify
    the market into three regimes with corresponding exposure adjustments.

    H9 fix: hysteresis bands prevent rapid regime flipping when VIX or
    drawdown oscillates around a threshold.  Escalation (normal→caution→halt)
    uses the primary thresholds; de-escalation requires crossing a lower
    threshold (primary minus the hysteresis band).

    Args:
        vix_caution: VIX threshold for caution regime (default 30).
        vix_halt: VIX threshold for halt regime (default 40).
        drawdown_caution: SPY drawdown threshold for caution (default 0.10).
        drawdown_halt: SPY drawdown threshold for halt (default 0.15).
        caution_multiplier: Exposure multiplier in caution regime (default 0.5).
        vix_hysteresis: VIX hysteresis band — de-escalation requires VIX
            to fall this far below the escalation threshold (default 2.0).
        drawdown_hysteresis: Drawdown hysteresis band (default 0.02 = 2pp).
    """

    def __init__(
        self,
        vix_caution: float = 30.0,
        vix_halt: float = 40.0,
        drawdown_caution: float = 0.10,
        drawdown_halt: float = 0.15,
        caution_multiplier: float = 0.5,
        vix_hysteresis: float = 2.0,
        drawdown_hysteresis: float = 0.02,
    ):
        self.vix_caution = vix_caution
        self.vix_halt = vix_halt
        self.drawdown_caution = drawdown_caution
        self.drawdown_halt = drawdown_halt
        self.caution_multiplier = caution_multiplier
        self.vix_hysteresis = vix_hysteresis
        self.drawdown_hysteresis = drawdown_hysteresis
        # H9: track the previous regime for hysteresis logic
        self._previous_regime: str = "normal"

    def get_regime(self, vix: float, spy_drawdown: float) -> str:
        """Classify current market regime with hysteresis.

        H9 fix: escalation uses the primary thresholds.  De-escalation
        requires crossing the threshold minus the hysteresis band, preventing
        rapid flipping when values oscillate around a boundary.

        Args:
            vix: Current VIX level.
            spy_drawdown: Current SPY drawdown from recent high (positive value,
                e.g. 0.12 means 12% below high).

        Returns:
            Regime string: "normal", "caution", or "halt".
        """
        spy_drawdown = abs(spy_drawdown)  # Ensure positive
        prev = self._previous_regime

        # --- Escalation: uses primary thresholds (immediate) ---
        if vix > self.vix_halt or spy_drawdown > self.drawdown_halt:
            regime = "halt"
        elif vix > self.vix_caution or spy_drawdown > self.drawdown_caution:
            regime = "caution"
        else:
            regime = "normal"

        # --- Hysteresis: sticky de-escalation ---
        # Only allow de-escalation if we've crossed below threshold - band.
        #
        # M-HYSTERESIS fix: the original AND logic for halt→caution required
        # BOTH VIX and drawdown to clear their bands before de-escalating.
        # During prolonged drawdowns (e.g. 2022 bear), VIX can normalize
        # while drawdown persists, locking the strategy in halt for months.
        # Changed to: if VIX normalizes (below caution-band), allow
        # de-escalation to caution even if drawdown persists.  Full
        # de-escalation to normal still requires both to clear.
        if prev == "halt" and regime != "halt":
            vix_clear = vix <= (self.vix_halt - self.vix_hysteresis)
            dd_clear = spy_drawdown <= (self.drawdown_halt - self.drawdown_hysteresis)
            if vix_clear or dd_clear:
                # At least one signal has cleared — allow partial de-escalation
                # to caution (not straight to normal)
                regime = "caution"
            else:
                regime = "halt"
        elif prev == "caution" and regime == "normal":
            # Stay in caution until VIX drops below caution-band AND drawdown below caution-band
            vix_clear = vix <= (self.vix_caution - self.vix_hysteresis)
            dd_clear = spy_drawdown <= (self.drawdown_caution - self.drawdown_hysteresis)
            if not (vix_clear and dd_clear):
                regime = "caution"

        self._previous_regime = regime
        return regime

    def get_regime_state(self, vix: float, spy_drawdown: float) -> RegimeState:
        """Get full regime state with exposure multiplier and message.

        Args:
            vix: Current VIX level.
            spy_drawdown: Current SPY drawdown from recent high.

        Returns:
            RegimeState with regime classification, metrics, and exposure multiplier.
        """
        regime = self.get_regime(vix, spy_drawdown)

        if regime == "halt":
            return RegimeState(
                regime="halt",
                vix=vix,
                spy_drawdown=abs(spy_drawdown),
                exposure_multiplier=0.0,
                message=(
                    f"HALT: VIX={vix:.1f} (limit {self.vix_halt}), "
                    f"SPY drawdown={abs(spy_drawdown):.1%} (limit {self.drawdown_halt:.0%}). "
                    f"All positions should be closed."
                ),
            )
        elif regime == "caution":
            return RegimeState(
                regime="caution",
                vix=vix,
                spy_drawdown=abs(spy_drawdown),
                exposure_multiplier=self.caution_multiplier,
                message=(
                    f"CAUTION: VIX={vix:.1f} (limit {self.vix_caution}), "
                    f"SPY drawdown={abs(spy_drawdown):.1%} (limit {self.drawdown_caution:.0%}). "
                    f"Reducing exposure to {self.caution_multiplier:.0%}."
                ),
            )
        else:
            return RegimeState(
                regime="normal",
                vix=vix,
                spy_drawdown=abs(spy_drawdown),
                exposure_multiplier=1.0,
                message=f"Normal: VIX={vix:.1f}, SPY drawdown={abs(spy_drawdown):.1%}.",
            )

    @staticmethod
    def is_halt(regime: str) -> bool:
        """Check if a regime requires full liquidation.

        M14 fix: callers should use this instead of checking ``weights == {}``
        to distinguish "halt — close everything" from "no positions to adjust".
        """
        return regime == "halt"

    def adjust_weights(
        self,
        weights: dict[str, float],
        regime: str,
    ) -> dict[str, float]:
        """Scale portfolio weights based on regime.

        Args:
            weights: Dict of ticker -> target weight.
            regime: Market regime ("normal", "caution", "halt").

        Returns:
            Adjusted weights dict.
            - halt:    ``{}`` — use ``is_halt(regime)`` to confirm liquidation intent.
            - caution: weights scaled by ``caution_multiplier``.
            - normal:  original weights unchanged.
        """
        if regime == "halt":
            logger.warning("Regime HALT: returning empty weights (close all positions)")
            return {}
        elif regime == "caution":
            adjusted = {t: w * self.caution_multiplier for t, w in weights.items()}
            logger.info(
                f"Regime CAUTION: scaled {len(weights)} positions by {self.caution_multiplier:.0%}"
            )
            return adjusted
        else:
            return weights


def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    """Safely extract a 1-D Close series from yfinance output.

    H8 fix: recent yfinance versions (>=0.2.31) return a DataFrame with
    MultiIndex columns even for a single ticker, e.g. ("Close", "^VIX").
    Indexing with ``["Close"]`` then returns a DataFrame, not a Series,
    and ``float()`` on a DataFrame raises TypeError.

    This helper handles both the old (flat columns) and new (MultiIndex)
    formats by squeezing any 2-D result to 1-D.
    """
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze(axis=1)
    return close


def fetch_vix() -> Optional[float]:
    """Fetch current VIX level from Yahoo Finance.

    M-VIX fix: warns if the latest VIX observation is >1 trading day old
    (e.g. stale data on Mondays or during market holidays), and returns a
    neutral default (20.0) if the data is >3 trading days stale.

    Returns None if the fetch fails entirely (caller should use a safe default).
    """
    try:
        import yfinance as yf

        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if vix_data is not None and len(vix_data) > 0:
            close = _extract_close_series(vix_data)

            # M-VIX fix: check staleness of the latest VIX observation
            last_date = close.index[-1]
            if hasattr(last_date, "date"):
                last_date = last_date.date()
            import datetime

            today = datetime.date.today()
            days_stale = (today - last_date).days
            # Weekends are expected (2 days stale on Monday), but >3 calendar
            # days suggests a data gap or outage.
            if days_stale > 3:
                logger.warning(
                    f"M-VIX: latest VIX observation is {days_stale} days old "
                    f"({last_date}). Returning neutral default (20.0)."
                )
                return 20.0
            elif days_stale > 1:
                logger.info(
                    f"M-VIX: latest VIX observation is {days_stale} days old "
                    f"({last_date}) — likely weekend/holiday, using as-is."
                )

            vix = float(close.iloc[-1])
            logger.info(f"Current VIX: {vix:.1f}")
            return vix
    except Exception as e:
        logger.warning(f"Failed to fetch VIX: {e}")
    return None


def fetch_spy_drawdown(lookback_days: int = 252) -> Optional[float]:
    """Calculate current SPY drawdown from its rolling high.

    Args:
        lookback_days: Number of trading days to look back for the high.

    Returns:
        Drawdown as a positive float (e.g. 0.08 means 8% below high).
        Returns None if the fetch fails.
    """
    try:
        import yfinance as yf

        spy_data = yf.download("SPY", period="1y", interval="1d", progress=False)
        if spy_data is not None and len(spy_data) > 0:
            close = _extract_close_series(spy_data)
            rolling_high = close.rolling(lookback_days, min_periods=1).max()
            current = float(close.iloc[-1])
            high = float(rolling_high.iloc[-1])
            if high > 0:
                drawdown = (high - current) / high
                logger.info(f"SPY drawdown from {lookback_days}d high: {drawdown:.1%}")
                return drawdown
    except Exception as e:
        logger.warning(f"Failed to fetch SPY drawdown: {e}")
    return None
