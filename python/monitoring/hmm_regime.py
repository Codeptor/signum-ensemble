"""Hidden Markov Model regime detection for market state classification.

Data-driven alternative to threshold-based RegimeDetector. Uses a Gaussian
HMM on returns and volatility features to learn latent market states.

Key advantages over thresholds:
  - Adapts to changing market structure (no hard-coded VIX levels)
  - Captures multi-dimensional regime features simultaneously
  - Provides probabilistic regime assignments (not binary)

References:
  - Hamilton, 1989 — "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
  - QuantStart — "Market Regime Detection Using Hidden Markov Models"
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# Regime labels ordered by increasing risk (assigned post-fit by volatility)
REGIME_LABELS = ["low_vol", "normal", "high_vol"]


@dataclass
class HMMRegimeState:
    """Snapshot of HMM-detected market regime."""

    regime: str  # "low_vol", "normal", "high_vol"
    regime_id: int  # 0, 1, 2
    probabilities: dict[str, float]  # regime -> probability
    exposure_multiplier: float  # 1.0, 0.7, 0.3
    message: str


@dataclass
class HMMRegimeDetector:
    """Gaussian HMM-based market regime detection.

    Fits a 3-state Gaussian HMM on a feature matrix derived from market
    returns and volatility. States are automatically labelled by their
    mean volatility: the highest-volatility state is "high_vol" (crisis),
    the lowest is "low_vol" (calm), and the middle is "normal".

    Typical usage::

        detector = HMMRegimeDetector()
        detector.fit(spy_returns)
        state = detector.predict_regime(recent_returns)
        print(state.regime, state.exposure_multiplier)

    Args:
        n_states: Number of hidden states (default 3).
        n_iter: EM iterations (default 100).
        covariance_type: HMM covariance type (default "full").
        random_state: Random seed for reproducibility.
        exposure_map: Dict mapping regime label to exposure multiplier.
    """

    n_states: int = 3
    n_iter: int = 100
    covariance_type: str = "full"
    random_state: int = 42
    exposure_map: dict[str, float] = field(
        default_factory=lambda: {"low_vol": 1.0, "normal": 0.7, "high_vol": 0.3}
    )

    _model: Optional[GaussianHMM] = field(default=None, init=False, repr=False)
    _state_order: Optional[list[int]] = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False)

    def _build_features(self, returns: pd.Series) -> np.ndarray:
        """Build feature matrix from a returns series.

        Features:
          1. Daily returns
          2. 5-day rolling volatility (annualised)
          3. 20-day rolling volatility (annualised)

        Args:
            returns: Daily returns series (e.g. SPY).

        Returns:
            2-D array of shape (n_obs, 3).
        """
        vol_5 = returns.rolling(5, min_periods=2).std() * np.sqrt(252)
        vol_20 = returns.rolling(20, min_periods=5).std() * np.sqrt(252)

        features = pd.DataFrame({
            "ret": returns,
            "vol_5": vol_5,
            "vol_20": vol_20,
        }).dropna()

        return features.values

    def fit(self, returns: pd.Series) -> "HMMRegimeDetector":
        """Fit the HMM on historical returns.

        After fitting, states are reordered so that state 0 = lowest
        volatility ("low_vol") and state N-1 = highest volatility ("high_vol").

        Args:
            returns: Daily returns series (at least 60 observations).

        Returns:
            self (for chaining).
        """
        X = self._build_features(returns)

        if len(X) < 60:
            raise ValueError(
                f"Need at least 60 observations after feature construction, got {len(X)}"
            )

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(X)

        # Order states by mean volatility (vol_20 = feature index 2)
        mean_vol = self._model.means_[:, 2]
        self._state_order = list(np.argsort(mean_vol))  # low→high vol

        self._fitted = True
        logger.info(
            f"HMM fitted: {self.n_states} states, "
            f"mean vol = {[f'{mean_vol[s]:.2f}' for s in self._state_order]}"
        )
        return self

    def _map_state(self, raw_state: int) -> tuple[int, str]:
        """Map raw HMM state to ordered regime index and label."""
        ordered_idx = self._state_order.index(raw_state)
        if self.n_states == 3:
            label = REGIME_LABELS[ordered_idx]
        else:
            label = f"state_{ordered_idx}"
        return ordered_idx, label

    def predict_states(self, returns: pd.Series) -> pd.DataFrame:
        """Predict regime states for a returns series.

        Args:
            returns: Daily returns series.

        Returns:
            DataFrame with columns: date, regime_id, regime, probability.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._build_features(returns)
        raw_states = self._model.predict(X)
        probs = self._model.predict_proba(X)

        # Map to ordered labels
        records = []
        feature_dates = returns.dropna().index[-len(X):]
        for i, (date, raw_s) in enumerate(zip(feature_dates, raw_states)):
            ordered_idx, label = self._map_state(raw_s)
            prob_dict = {}
            for j in range(self.n_states):
                _, j_label = self._map_state(j)
                prob_dict[j_label] = float(probs[i, j])

            records.append({
                "date": date,
                "regime_id": ordered_idx,
                "regime": label,
                **{f"prob_{k}": v for k, v in prob_dict.items()},
            })

        return pd.DataFrame(records)

    def predict_regime(self, returns: pd.Series) -> HMMRegimeState:
        """Predict the current (latest) regime from recent returns.

        Args:
            returns: Recent daily returns (at least 25 observations).

        Returns:
            HMMRegimeState with regime label, probabilities, and exposure.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._build_features(returns)
        if len(X) == 0:
            return HMMRegimeState(
                regime="normal",
                regime_id=1,
                probabilities={"low_vol": 0.0, "normal": 1.0, "high_vol": 0.0},
                exposure_multiplier=self.exposure_map.get("normal", 0.7),
                message="Insufficient data for HMM — defaulting to normal",
            )

        # Get probabilities for the last observation
        probs = self._model.predict_proba(X)
        last_probs = probs[-1]
        raw_state = int(np.argmax(last_probs))
        ordered_idx, label = self._map_state(raw_state)

        prob_dict = {}
        for j in range(self.n_states):
            _, j_label = self._map_state(j)
            prob_dict[j_label] = float(last_probs[j])

        exposure = self.exposure_map.get(label, 0.7)

        return HMMRegimeState(
            regime=label,
            regime_id=ordered_idx,
            probabilities=prob_dict,
            exposure_multiplier=exposure,
            message=(
                f"HMM regime: {label} "
                f"(P={prob_dict.get(label, 0):.1%}, exposure={exposure:.0%})"
            ),
        )

    def compare_with_threshold(
        self,
        returns: pd.Series,
        vix_series: pd.Series,
        spy_drawdown_series: pd.Series,
    ) -> pd.DataFrame:
        """Compare HMM regimes against threshold-based detection.

        Aligns HMM predictions with threshold-based regimes for
        side-by-side comparison. Useful for validation and dashboard display.

        Args:
            returns: Daily returns used for HMM.
            vix_series: Daily VIX values (same index as returns).
            spy_drawdown_series: Daily SPY drawdown from high.

        Returns:
            DataFrame with date, hmm_regime, threshold_regime, agreement.
        """
        from python.monitoring.regime import RegimeDetector

        hmm_df = self.predict_states(returns)

        threshold_det = RegimeDetector()
        threshold_regimes = []
        for _, row in hmm_df.iterrows():
            date = row["date"]
            vix = vix_series.get(date, 20.0) if date in vix_series.index else 20.0
            dd = (
                spy_drawdown_series.get(date, 0.0)
                if date in spy_drawdown_series.index
                else 0.0
            )
            threshold_regimes.append(threshold_det.get_regime(vix, dd))

        hmm_df["threshold_regime"] = threshold_regimes

        # Map threshold regimes to comparable labels
        threshold_map = {"normal": "low_vol", "caution": "normal", "halt": "high_vol"}
        hmm_df["threshold_mapped"] = hmm_df["threshold_regime"].map(threshold_map)
        hmm_df["agreement"] = hmm_df["regime"] == hmm_df["threshold_mapped"]

        agreement_pct = hmm_df["agreement"].mean() * 100
        logger.info(f"HMM vs threshold agreement: {agreement_pct:.1f}%")

        return hmm_df

    def to_json(self) -> dict:
        """Export model state as JSON-serializable dict."""
        if not self._fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_states": self.n_states,
            "state_means": {
                REGIME_LABELS[i]: {
                    "return": float(self._model.means_[s, 0]),
                    "vol_5d": float(self._model.means_[s, 1]),
                    "vol_20d": float(self._model.means_[s, 2]),
                }
                for i, s in enumerate(self._state_order)
            },
            "exposure_map": self.exposure_map,
        }
