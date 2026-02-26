import pandas as pd

from python.bridge.bl_views import create_bl_views


def test_create_bl_views():
    """ML predictions should convert to Black-Litterman absolute views."""
    predictions = pd.Series(
        {"AAPL": 0.02, "MSFT": 0.01, "GOOG": -0.005, "AMZN": 0.015, "META": -0.01},
        name="predicted_return",
    )
    confidences = pd.Series(
        {"AAPL": 0.8, "MSFT": 0.6, "GOOG": 0.3, "AMZN": 0.7, "META": 0.4},
        name="confidence",
    )

    views, view_confidences = create_bl_views(predictions, confidences)

    assert len(views) == 5
    assert len(view_confidences) == 5
    # Higher ML confidence → higher Idzorek confidence (Fix #42)
    assert view_confidences["AAPL"] > view_confidences["GOOG"]
    # All confidences in valid range (0, 1]
    assert all(0 < c <= 1.0 for c in view_confidences)
