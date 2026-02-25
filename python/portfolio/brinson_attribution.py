"""Brinson performance attribution model.

Decomposes portfolio returns into:
- Allocation effect: from sector/asset class weighting decisions
- Selection effect: from security selection within sectors
- Interaction effect: combination of allocation and selection

Reference: Brinson, Hood & Beebower (1986), "Determinants of Portfolio Performance"
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class BrinsonAttribution:
    """
    Brinson-Fachler attribution model for performance decomposition.

    Attributes:
        portfolio_weights: Portfolio sector/asset weights
        benchmark_weights: Benchmark sector/asset weights
        portfolio_returns: Portfolio sector/asset returns
        benchmark_returns: Benchmark sector/asset returns
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ):
        """
        Args:
            portfolio_weights: Portfolio weights by sector/asset
            benchmark_weights: Benchmark weights by sector/asset
            portfolio_returns: Portfolio returns by sector/asset
            benchmark_returns: Benchmark returns by sector/asset
        """
        self.portfolio_weights = portfolio_weights
        self.benchmark_weights = benchmark_weights
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns

    def _calculate_total_return(self, weights: pd.Series, returns: pd.Series) -> float:
        """Calculate total weighted return."""
        common_idx = weights.index.intersection(returns.index)
        return (weights[common_idx] * returns[common_idx]).sum()

    def attribution(self) -> Dict:
        """
        Calculate Brinson attribution effects.

        Returns:
            Dictionary with:
            - allocation_effect: Weighting decisions
            - selection_effect: Security selection
            - interaction_effect: Combined effect
            - total_excess_return: Sum of all effects
            - portfolio_return: Total portfolio return
            - benchmark_return: Total benchmark return
        """
        # Align indices
        common_idx = (
            self.portfolio_weights.index.intersection(self.benchmark_weights.index)
            .intersection(self.portfolio_returns.index)
            .intersection(self.benchmark_returns.index)
        )

        wp = self.portfolio_weights[common_idx]
        wb = self.benchmark_weights[common_idx]
        rp = self.portfolio_returns[common_idx]
        rb = self.benchmark_returns[common_idx]

        # Total returns
        portfolio_return = self._calculate_total_return(wp, rp)
        benchmark_return = self._calculate_total_return(wb, rb)
        total_excess = portfolio_return - benchmark_return

        # Allocation effect: (Wp - Wb) × (Rb - Rb_total)
        allocation = (wp - wb) * (rb - benchmark_return)
        allocation_effect = allocation.sum()

        # Selection effect: Wb × (Rp - Rb)
        selection = wb * (rp - rb)
        selection_effect = selection.sum()

        # Interaction effect: (Wp - Wb) × (Rp - Rb)
        interaction = (wp - wb) * (rp - rb)
        interaction_effect = interaction.sum()

        return {
            "allocation_effect": allocation_effect,
            "allocation_effect_pct": allocation_effect * 100,
            "selection_effect": selection_effect,
            "selection_effect_pct": selection_effect * 100,
            "interaction_effect": interaction_effect,
            "interaction_effect_pct": interaction_effect * 100,
            "total_excess_return": total_excess,
            "total_excess_return_pct": total_excess * 100,
            "portfolio_return": portfolio_return,
            "portfolio_return_pct": portfolio_return * 100,
            "benchmark_return": benchmark_return,
            "benchmark_return_pct": benchmark_return * 100,
            "attribution_check": allocation_effect + selection_effect + interaction_effect,
            "sector_breakdown": {
                "allocation": allocation.to_dict(),
                "selection": selection.to_dict(),
                "interaction": interaction.to_dict(),
            },
        }

    def attribution_report(self) -> str:
        """Generate formatted attribution report."""
        result = self.attribution()

        report = []
        report.append("=" * 60)
        report.append("BRINSON PERFORMANCE ATTRIBUTION")
        report.append("=" * 60)
        report.append("")
        report.append(f"Portfolio Return:     {result['portfolio_return_pct']:.2f}%")
        report.append(f"Benchmark Return:     {result['benchmark_return_pct']:.2f}%")
        report.append(f"Excess Return:        {result['total_excess_return_pct']:.2f}%")
        report.append("")
        report.append("-" * 60)
        report.append("ATTRIBUTION EFFECTS")
        report.append("-" * 60)
        report.append(f"Allocation Effect:    {result['allocation_effect_pct']:.2f}%")
        report.append(f"Selection Effect:     {result['selection_effect_pct']:.2f}%")
        report.append(f"Interaction Effect:   {result['interaction_effect_pct']:.2f}%")
        report.append("")
        report.append(f"Attribution Check:    {result['attribution_check'] * 100:.2f}%")
        report.append("=" * 60)

        return "\n".join(report)


def calculate_brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict:
    """
    Convenience function for Brinson attribution.

    Args:
        portfolio_weights: Portfolio weights by sector
        benchmark_weights: Benchmark weights by sector
        portfolio_returns: Portfolio returns by sector
        benchmark_returns: Benchmark returns by sector

    Returns:
        Attribution results dictionary
    """
    model = BrinsonAttribution(
        portfolio_weights,
        benchmark_weights,
        portfolio_returns,
        benchmark_returns,
    )
    return model.attribution()
