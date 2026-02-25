"""Meta-allocator blend optimization."""

import itertools
import logging
from pathlib import Path

import pandas as pd

from python.backtest.run import run_backtest
from python.data.ingestion import extract_close_prices

logger = logging.getLogger(__name__)


def generate_blend_grid(methods: list[str], step: float = 0.2) -> list[dict]:
    """Generate all valid weight combinations that sum to 1.0."""
    grid = [i * step for i in range(int(1.0 / step) + 1)]
    valid_blends = []
    for combo in itertools.product(grid, repeat=len(methods)):
        if abs(sum(combo) - 1.0) < 1e-6:
            blend = {m: round(w, 2) for m, w in zip(methods, combo) if w > 0}
            valid_blends.append(blend)
    return valid_blends


def optimize_blend() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = pd.read_parquet("data/raw/sp500_ohlcv.parquet")
    prices = extract_close_prices(raw)

    methods = ["equal_weight", "hrp", "min_cvar"]
    blends = generate_blend_grid(methods, step=0.2)

    logger.info(f"Generated {len(blends)} valid blend combinations.")

    results = []

    for blend in blends:
        logger.info(f"Evaluating blend: {blend}")
        try:
            r = run_backtest(
                prices,
                optimizer_method=blend,
                transaction_cost_bps=10.0,
                max_weight=0.15,
                blend_alpha=0.5,
                initial_capital=10_000_000.0,
                impact_coeff=0.1,
                fixed_bps=5.0,
            )
            # Objective: Sharpe - 0.5 * Avg_Turnover
            score = r["sharpe_ratio"] - 0.5 * r["avg_turnover"]

            results.append(
                {
                    "blend": str(blend),
                    "sharpe_net": r["sharpe_ratio"],
                    "ann_return": r["annualized_return"],
                    "max_dd": r["max_drawdown"],
                    "avg_turnover": r["avg_turnover"],
                    "score": score,
                }
            )
        except Exception as e:
            logger.error(f"Failed to evaluate blend {blend}: {e}")

    if not results:
        logger.error("No blends were successfully evaluated.")
        return

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    logger.info("\n=== Top 5 Blend Configurations ===")
    logger.info(results_df.head(5).to_string(index=False, float_format="%.3f"))

    # Save the results
    output_path = Path("data/processed/blend_optimization.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved blend optimization results to {output_path}")


if __name__ == "__main__":
    optimize_blend()
