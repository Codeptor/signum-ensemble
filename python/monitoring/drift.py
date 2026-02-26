"""Feature and prediction drift detection."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects distribution drift between reference and current data."""

    def __init__(self, reference: pd.DataFrame, threshold: float = 0.05) -> None:
        self.reference = reference
        self.threshold = threshold

    def detect(self, current: pd.DataFrame) -> dict:
        """Run KS test for each feature and return drift report."""
        report = {}
        for col in self.reference.columns:
            if col not in current.columns:
                continue
            ref_vals = self.reference[col].dropna()
            cur_vals = current[col].dropna()

            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            psi = self._psi(ref_vals, cur_vals)

            report[col] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "psi": psi,
                "drifted": bool(p_value < self.threshold),
            }

        drifted = [k for k, v in report.items() if v["drifted"]]
        if drifted:
            logger.warning(f"Drift detected in features: {drifted}")

        return report

    @staticmethod
    def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Population Stability Index.

        Bin edges are derived from reference distribution quantiles (Fix #41)
        so that results are stable regardless of the current distribution's
        range.  Values outside the reference range fall into the first/last bin.
        """
        # Use reference quantiles for bin edges — ensures stable bins
        quantiles = np.linspace(0, 1, bins + 1)
        breakpoints = np.quantile(reference.dropna(), quantiles)
        # Ensure monotonically increasing (can have ties in low-cardinality data)
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            # Not enough distinct values to compute PSI meaningfully
            return 0.0
        # Extend edges to capture any out-of-range values in current
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

        # Avoid log(0)
        ref_counts = np.clip(ref_counts, 1e-6, None)
        cur_counts = np.clip(cur_counts, 1e-6, None)

        return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
