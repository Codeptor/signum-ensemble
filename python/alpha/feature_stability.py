"""Feature importance stability analysis across cross-validation folds.

Computes the stability of feature rankings across CPCV or walk-forward
folds using Spearman rank correlation. Unstable features that flip ranks
between folds are candidates for removal — they likely represent overfit
patterns that degrade out-of-sample performance.

Usage::

    analyzer = FeatureStabilityAnalyzer()
    report = analyzer.analyze(importance_per_fold)
    # report.unstable_features → features to consider dropping
    # report.stability_matrix → pairwise fold correlation

References:
  - Lopez de Prado (2018), Ch. 8 — "Feature Importance"
  - Nogueira, Sechidis & Brown (2018), "On the Stability of Feature Selection Algorithms"
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """Results of feature importance stability analysis."""

    # Per-feature metrics
    feature_stats: pd.DataFrame
    # Columns: mean_rank, std_rank, cv_rank, mean_importance,
    #          stability_score, is_stable

    # Pairwise fold correlation matrix
    stability_matrix: pd.DataFrame  # (n_folds, n_folds) Spearman correlations

    # Aggregate metrics
    mean_stability: float  # Average pairwise Spearman rho
    n_stable: int
    n_unstable: int
    n_features: int
    n_folds: int

    @property
    def stable_features(self) -> list[str]:
        return list(self.feature_stats[self.feature_stats["is_stable"]].index)

    @property
    def unstable_features(self) -> list[str]:
        return list(self.feature_stats[~self.feature_stats["is_stable"]].index)

    def summary(self) -> str:
        return (
            f"Feature Stability: {self.n_stable}/{self.n_features} stable "
            f"(mean rho={self.mean_stability:.3f}, {self.n_folds} folds)"
        )


class FeatureStabilityAnalyzer:
    """Analyze stability of feature importances across CV folds.

    Parameters
    ----------
    stability_threshold : float
        Minimum stability score (0-1) to be considered stable.
        Default 0.5 means a feature must maintain roughly consistent
        rank across at least half the folds.
    rank_cv_threshold : float
        Maximum coefficient of variation of rank to be stable.
        Default 0.5 = rank standard deviation can be at most 50% of mean rank.
    """

    def __init__(
        self,
        stability_threshold: float = 0.5,
        rank_cv_threshold: float = 0.5,
    ):
        self.stability_threshold = stability_threshold
        self.rank_cv_threshold = rank_cv_threshold

    def analyze(
        self,
        importance_per_fold: dict[str, pd.Series] | list[pd.Series],
    ) -> StabilityReport:
        """Analyze feature importance stability across folds.

        Parameters
        ----------
        importance_per_fold : dict or list
            Mapping of fold_name -> feature importances (pd.Series indexed by feature).
            If a list, fold names are generated as "fold_0", "fold_1", etc.

        Returns
        -------
        StabilityReport
        """
        if isinstance(importance_per_fold, list):
            importance_per_fold = {
                f"fold_{i}": s for i, s in enumerate(importance_per_fold)
            }

        # Build importance matrix: (n_features, n_folds)
        imp_df = pd.DataFrame(importance_per_fold)
        imp_df = imp_df.fillna(0.0)
        features = list(imp_df.index)
        folds = list(imp_df.columns)
        n_features = len(features)
        n_folds = len(folds)

        if n_folds < 2:
            raise ValueError("Need at least 2 folds for stability analysis")

        # Compute ranks within each fold (higher importance = lower rank number)
        rank_df = imp_df.rank(ascending=False)

        # Pairwise Spearman correlation between fold rankings
        stability_matrix = np.zeros((n_folds, n_folds))
        for i in range(n_folds):
            for j in range(n_folds):
                if i == j:
                    stability_matrix[i, j] = 1.0
                elif i < j:
                    rho, _ = spearmanr(
                        rank_df.iloc[:, i].values,
                        rank_df.iloc[:, j].values,
                    )
                    rho = rho if not np.isnan(rho) else 0.0
                    stability_matrix[i, j] = rho
                    stability_matrix[j, i] = rho

        stab_df = pd.DataFrame(stability_matrix, index=folds, columns=folds)

        # Mean pairwise stability (excluding diagonal)
        n_pairs = n_folds * (n_folds - 1) / 2
        upper_tri = stability_matrix[np.triu_indices(n_folds, k=1)]
        mean_stability = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0

        # Per-feature statistics
        feature_rows = []
        for feat in features:
            ranks = rank_df.loc[feat].values
            importances = imp_df.loc[feat].values

            mean_rank = float(np.mean(ranks))
            std_rank = float(np.std(ranks))
            cv_rank = std_rank / max(mean_rank, 1e-6)
            mean_imp = float(np.mean(importances))

            # Stability score: 1 - normalized rank CV
            # A feature that always has the same rank gets score 1.0
            # A feature that jumps around gets score near 0
            max_possible_cv = (n_features - 1) / (n_features / 2)  # worst case
            stability_score = max(0.0, 1.0 - cv_rank / max(max_possible_cv, 1e-6))

            is_stable = (
                stability_score >= self.stability_threshold
                and cv_rank <= self.rank_cv_threshold
            )

            feature_rows.append({
                "mean_rank": mean_rank,
                "std_rank": std_rank,
                "cv_rank": cv_rank,
                "mean_importance": mean_imp,
                "stability_score": stability_score,
                "is_stable": is_stable,
            })

        feature_stats = pd.DataFrame(feature_rows, index=features)
        feature_stats = feature_stats.sort_values("stability_score", ascending=False)

        n_stable = int(feature_stats["is_stable"].sum())
        n_unstable = n_features - n_stable

        report = StabilityReport(
            feature_stats=feature_stats,
            stability_matrix=stab_df,
            mean_stability=mean_stability,
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_features=n_features,
            n_folds=n_folds,
        )

        logger.info(report.summary())
        return report

    def select_stable_features(
        self,
        importance_per_fold: dict[str, pd.Series] | list[pd.Series],
        min_importance_rank: int | None = None,
    ) -> list[str]:
        """Convenience: return list of stable feature names.

        Parameters
        ----------
        importance_per_fold : dict or list
            Same as analyze().
        min_importance_rank : int, optional
            Also require features to have mean rank <= this value.

        Returns
        -------
        list[str]
            Names of stable features.
        """
        report = self.analyze(importance_per_fold)
        stable = report.feature_stats[report.feature_stats["is_stable"]]

        if min_importance_rank is not None:
            stable = stable[stable["mean_rank"] <= min_importance_rank]

        return list(stable.index)
