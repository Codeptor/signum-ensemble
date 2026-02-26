"""Bridge between ML model predictions and Black-Litterman views.

Converts ML return predictions + confidence scores into the views format
expected by portfolio optimization (skfolio / Riskfolio-Lib).
"""

import numpy as np
import pandas as pd


def create_bl_views(
    predictions: pd.Series,
    confidences: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Convert ML predictions to Black-Litterman absolute views.

    Args:
        predictions: Expected returns per ticker from ML model.
        confidences: Confidence scores per ticker (0-1 scale, Idzorek method).

    Returns:
        views: Predicted returns (same as input predictions).
        view_confidences: Idzorek confidence per view (0-1 scale), passed
            directly to ``skfolio.prior.BlackLitterman(view_confidences=...)``.

    Note (Fix #42):
        skfolio's ``BlackLitterman`` expects Idzorek confidence in [0, 1] and
        internally converts to the uncertainty matrix ``Omega``.  Previous code
        applied its own uncertainty mapping which double-converted the values.
        We now pass the ML confidence scores through directly so skfolio's
        internal Idzorek→Omega conversion is applied correctly.
    """
    # Clamp to valid range — skfolio requires strictly (0, 1]
    clamped = confidences.clip(lower=0.01, upper=1.0)
    return predictions, clamped


def create_picking_matrix(
    tickers: list[str],
    views: pd.Series,
) -> np.ndarray:
    """Create the picking matrix P for Black-Litterman.

    For absolute views (each view is about one asset), P is an identity matrix
    over the assets with views.
    """
    n_views = len(views)
    n_assets = len(tickers)
    picking = np.zeros((n_views, n_assets))
    for i, ticker in enumerate(views.index):
        j = tickers.index(ticker)
        picking[i, j] = 1.0
    return picking
