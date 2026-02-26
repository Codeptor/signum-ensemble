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

import numpy as np
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

    Args:
        vix_caution: VIX threshold for caution regime (default 30).
        vix_halt: VIX threshold for halt regime (default 40).
        drawdown_caution: SPY drawdown threshold for caution (default 0.10).
        drawdown_halt: SPY drawdown threshold for halt (default 0.15).
        caution_multiplier: Exposure multiplier in caution regime (default 0.5).
    """

    def __init__(
        self,
        vix_caution: float = 30.0,
        vix_halt: float = 40.0,
        drawdown_caution: float = 0.10,
        drawdown_halt: float = 0.15,
        caution_multiplier: float = 0.5,
    ):
        self.vix_caution = vix_caution
        self.vix_halt = vix_halt
        self.drawdown_caution = drawdown_caution
        self.drawdown_halt = drawdown_halt
        self.caution_multiplier = caution_multiplier

    def get_regime(self, vix: float, spy_drawdown: float) -> str:
        """Classify current market regime.

        Args:
            vix: Current VIX level.
            spy_drawdown: Current SPY drawdown from recent high (positive value,
                e.g. 0.12 means 12% below high).

        Returns:
            Regime string: "normal", "caution", or "halt".
        """
        spy_drawdown = abs(spy_drawdown)  # Ensure positive

        if vix > self.vix_halt or spy_drawdown > self.drawdown_halt:
            return "halt"
        elif vix > self.vix_caution or spy_drawdown > self.drawdown_caution:
            return "caution"
        else:
            return "normal"

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
            Adjusted weights. Empty dict for halt, scaled for caution,
            unchanged for normal.
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


def fetch_vix() -> Optional[float]:
    """Fetch current VIX level from Yahoo Finance.

    Returns None if the fetch fails (caller should use a safe default).
    """
    try:
        import yfinance as yf

        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if vix_data is not None and len(vix_data) > 0:
            vix = float(vix_data["Close"].iloc[-1])
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
            close = spy_data["Close"]
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
