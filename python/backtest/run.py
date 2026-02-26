"""Full backtesting pipeline: train -> predict -> optimize -> evaluate."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from python.alpha.features import (
    compute_alpha_features,
    compute_cross_sectional_features,
    compute_forward_returns,
    compute_residual_target,
    merge_macro_features,
)
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.backtest.validation import deflated_sharpe_ratio, walk_forward_split
from python.bridge.bl_views import create_bl_views
from python.data.ingestion import extract_close_prices, reshape_ohlcv_wide_to_long
from python.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/processed")

# Minimum trailing days needed for covariance estimation
MIN_PRICE_HISTORY = 60

# C-SHARPE fix: risk-free rate for Sharpe ratio calculation.
# Approximate 1-year US Treasury bill rate as of 2025.
RISK_FREE_RATE = 0.05


def _allocate(
    top_tickers: list[str],
    raw_preds: pd.Series,
    prices: pd.DataFrame,
    date,
    method: str | dict,
    max_weight: float,
) -> pd.Series:
    """Compute portfolio weights for a single rebalance date, supporting blended strategies."""
    if isinstance(method, str):
        methods = {method: 1.0}
    elif isinstance(method, dict):
        total = sum(method.values())
        if total <= 0:
            raise ValueError("Blend weights must sum to > 0")
        methods = {k: v / total for k, v in method.items()}
    else:
        raise TypeError("method must be a string or a dictionary of method: weight")

    # Get trailing price history for selected tickers
    available = [t for t in top_tickers if t in prices.columns]
    prices_window = prices.loc[:date, available].dropna(axis=1).tail(252)

    # Drop tickers with insufficient history
    prices_window = prices_window.loc[:, prices_window.count() >= MIN_PRICE_HISTORY]
    valid_tickers = list(prices_window.columns)

    if len(valid_tickers) < 3:
        weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
        return _apply_max_weight(weights, max_weight)

    try:
        optimizer = PortfolioOptimizer(prices_window)
    except Exception as e:
        logger.warning(f"Failed to initialize optimizer ({e}), falling back to equal-weight")
        weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
        return _apply_max_weight(weights, max_weight)

    blended_weights = pd.Series(0.0, index=top_tickers)

    for m_name, m_weight in methods.items():
        if m_weight <= 0:
            continue

        if m_name == "equal_weight":
            w = pd.Series(1.0 / len(top_tickers), index=top_tickers)
        else:
            try:
                if m_name == "black_litterman":
                    preds_valid = raw_preds.reindex(valid_tickers).dropna()
                    if len(preds_valid) < 3:
                        w = optimizer.hrp()
                    else:
                        pred_range = preds_valid.max() - preds_valid.min()
                        if pred_range > 0:
                            conf = (preds_valid - preds_valid.min()) / pred_range
                            conf = conf.clip(0.1, 0.9)
                        else:
                            conf = pd.Series(0.5, index=preds_valid.index)
                        views, view_confs = create_bl_views(preds_valid, conf)
                        w = optimizer.black_litterman(views, view_confs)
                elif m_name == "hrp":
                    w = optimizer.hrp()
                elif m_name == "min_cvar":
                    w = optimizer.min_cvar()
                elif m_name == "risk_parity":
                    w = optimizer.risk_parity()
                else:
                    raise ValueError(f"Unknown optimizer method: {m_name}")
            except Exception as e:
                logger.warning(
                    f"Optimizer {m_name} failed ({e}), using equal-weight for this component"
                )
                w = pd.Series(1.0 / len(top_tickers), index=top_tickers)

        w = w.reindex(top_tickers, fill_value=0.0)
        blended_weights += w * m_weight

    if blended_weights.sum() > 0:
        blended_weights /= blended_weights.sum()
    else:
        blended_weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)

    return _apply_max_weight(blended_weights, max_weight)


def _apply_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    """Clip weights to max and iteratively renormalize (R3-B-5 fix).

    A single clip-then-renorm can push previously-uncapped weights above
    max_weight. This uses the same iterative approach as ``_cap_weights``
    in optimizer.py: clip, redistribute surplus to uncapped, repeat.
    """
    if max_weight >= 1.0:
        return weights
    w = weights.copy()
    for _ in range(20):
        excess_mask = w > max_weight
        if not excess_mask.any():
            break
        surplus = (w[excess_mask] - max_weight).sum()
        w[excess_mask] = max_weight
        uncapped = ~excess_mask
        unc_total = w[uncapped].sum()
        if unc_total > 0:
            w[uncapped] += surplus * (w[uncapped] / unc_total)
        else:
            break
    total = w.sum()
    if total > 0:
        w /= total
    return w


def run_backtest(
    prices: pd.DataFrame,
    n_splits: int = 5,
    top_n: int = 20,
    rebalance_days: int = 5,
    optimizer_method: str | dict = "black_litterman",
    transaction_cost_bps: float = 15.0,  # 15 bps (realistic, up from 10)
    max_weight: float = 0.15,
    blend_alpha: float = 0.5,
    macro_path: str = "data/raw/macro_indicators.parquet",
    liquidity_filter_pct: float = 0.2,
    vix_scaling: bool = True,
    ohlcv: pd.DataFrame | None = None,
    initial_capital: float = 10_000_000.0,
    impact_coeff: float = 0.1,
    fixed_bps: float = 7.5,  # 7.5 bps fixed commission (up from 5)
    slippage_bps: float = 5.0,  # 5 bps slippage estimate (NEW)
    enable_risk_manager: bool = True,
    risk_limits: dict | None = None,
) -> dict:
    """Walk-forward backtest with ML alpha, portfolio optimization, and transaction costs.

    Parameters
    ----------
    prices : DataFrame with DatetimeIndex, columns = tickers, values = close prices
    n_splits : number of walk-forward folds
    top_n : number of top-ranked tickers to hold each period
    rebalance_days : holding period (forward return horizon)
    optimizer_method : 'equal_weight', 'black_litterman', 'hrp', 'min_cvar', 'risk_parity'
    transaction_cost_bps : one-way transaction cost in basis points (default 15)
    max_weight : maximum single-position weight (e.g. 0.15 = 15%)
    blend_alpha : weight on new allocation vs previous (1.0 = fully new, 0.5 = 50/50 blend)
    macro_path : path to macro indicators parquet (None to skip macro features)
    liquidity_filter_pct : exclude bottom N% by dollar volume (0.0 to disable)
    vix_scaling : if True, scale position size by VIX regime
    ohlcv : pre-built long-format OHLCV DataFrame (if None, synthesized from prices)
    fixed_bps : fixed commission in basis points per trade (default 7.5)
    slippage_bps : estimated slippage in basis points (default 5.0)
    """
    if ohlcv is None:
        ohlcv_frames = []
        for ticker in prices.columns:
            ohlcv_frames.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "open": prices[ticker],
                        "high": prices[ticker] * 1.01,
                        "low": prices[ticker] * 0.99,
                        "close": prices[ticker],
                        "volume": 1e6,
                    },
                    index=prices.index,
                )
            )
        ohlcv = pd.concat(ohlcv_frames)
    featured = compute_alpha_features(ohlcv)
    featured = compute_cross_sectional_features(featured)
    if macro_path:
        featured = merge_macro_features(featured, macro_path=macro_path)
    labeled = compute_forward_returns(featured, horizon=rebalance_days)
    labeled = compute_residual_target(labeled, horizon=rebalance_days)
    labeled = labeled.dropna(subset=FEATURE_COLS + [f"target_{rebalance_days}d"])

    all_returns = []
    weight_history = []
    prev_weights = pd.Series(dtype=float)
    total_turnover = 0.0
    n_rebalances = 0
    current_capital = initial_capital

    # Initialize RiskManager if enabled
    risk_manager = None
    if enable_risk_manager:
        from python.portfolio.risk_manager import RiskLimits, RiskManager

        limits = RiskLimits(**risk_limits) if risk_limits else RiskLimits()
        risk_manager = RiskManager(limits=limits)
        logger.info(
            f"RiskManager enabled with limits: max_position={limits.max_position_weight:.1%}, "
            f"max_daily_trades={limits.max_daily_trades}"
        )

    for fold, (train_idx, test_idx) in enumerate(walk_forward_split(labeled, n_splits=n_splits)):
        logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        train = labeled.iloc[train_idx]
        test = labeled.iloc[test_idx]

        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col=f"target_{rebalance_days}d")

        preds = model.predict(test)
        raw_pred_series = pd.Series(preds, index=test.index)
        ranks = raw_pred_series.groupby(test.index).rank(pct=True)

        test_with_preds = test.copy()
        test_with_preds["pred_rank"] = ranks.values
        test_with_preds["raw_pred"] = preds

        # Sub-sample to rebalance frequency to avoid overlapping returns
        dates = test_with_preds.index.get_level_values(0).unique().sort_values()
        rebalance_dates = dates[::rebalance_days]
        test_with_preds = test_with_preds[
            test_with_preds.index.get_level_values(0).isin(rebalance_dates)
        ]

        for date, group in test_with_preds.groupby(level=0):
            # Liquidity filter: exclude bottom N% by dollar volume
            if liquidity_filter_pct > 0 and "dollar_volume_20d" in group.columns:
                vol_threshold = group["dollar_volume_20d"].quantile(liquidity_filter_pct)
                group = group[group["dollar_volume_20d"] >= vol_threshold]

            top = group.nlargest(top_n, "pred_rank")
            top_tickers = top["ticker"].tolist()
            raw_preds_top = top.set_index("ticker")["raw_pred"]

            # Portfolio optimization
            weights = _allocate(
                top_tickers,
                raw_preds_top,
                prices,
                date,
                method=optimizer_method,
                max_weight=max_weight,
            )

            # VIX-based position scaling: high VIX → reduce exposure (rest = cash)
            if vix_scaling and "vix" in group.columns:
                vix_level = group["vix"].iloc[0]
                if vix_level < 25:
                    scale = 1.0
                elif vix_level <= 35:
                    scale = 0.85
                else:
                    scale = 0.7
                weights = weights * scale

            # Blend with previous weights to dampen turnover
            if len(prev_weights) > 0 and blend_alpha < 1.0:
                common = sorted(set(weights.index) | set(prev_weights.index))
                w_target = weights.reindex(common, fill_value=0.0)
                w_prev = prev_weights.reindex(common, fill_value=0.0)
                weights = blend_alpha * w_target + (1 - blend_alpha) * w_prev
                weights = weights[weights > 1e-6]
                weights /= weights.sum()

            # Risk Manager: Validate position sizes before execution
            if risk_manager is not None:
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                risk_violations = []

                for ticker, weight in weights.items():
                    can_trade, issues = risk_manager.can_execute_trade(
                        ticker=ticker,
                        new_weight=weight,
                        current_date=date_str,
                    )
                    if not can_trade:
                        risk_violations.extend(issues)
                        # Reduce position to comply with limits
                        weights[ticker] = min(weight, risk_manager.limits.max_position_weight)

                if risk_violations:
                    logger.warning(f"Risk violations on {date_str}: {len(risk_violations)} issues")
                    # Re-normalize weights after adjustments
                    if weights.sum() > 0:
                        weights /= weights.sum()

                # Record trades for daily limit tracking
                for ticker in weights.index:
                    weight_change = abs(weights[ticker] - prev_weights.get(ticker, 0))
                    if weight_change > 0.001:  # Only record meaningful changes
                        risk_manager.record_trade(ticker, weight_change, date_str)

            # Extract current day's liquidity profiles
            top_data = group.set_index("ticker")

            # Transaction costs (two-way: sell old + buy new)
            # Align weights to common index for turnover calculation
            all_tickers = sorted(set(weights.index) | set(prev_weights.index))
            w_new = weights.reindex(all_tickers, fill_value=0.0)
            w_old = prev_weights.reindex(all_tickers, fill_value=0.0)
            turnover = (w_new - w_old).abs().sum()
            total_turnover += turnover
            n_rebalances += 1

            # 1. Trade Value
            trade_sizes = (w_new - w_old).abs()
            trade_value = trade_sizes * current_capital

            # 2. Fixed Commissions
            fixed_cost = trade_value * (fixed_bps / 10_000)

            # 3. Slippage Cost (constant estimate per trade)
            slippage_cost = trade_value * (slippage_bps / 10_000)

            # 4. Spread Drag
            # Let's parameterize spread multiplier or zero it for debugging
            spread = top_data["bid_ask_proxy"].reindex(all_tickers).fillna(0).clip(0, 0.02)
            spread_cost = trade_value * spread * 0.5  # Half-spread as transaction cost

            # 5. Market Impact
            adv = top_data["dollar_volume_20d"].reindex(all_tickers).fillna(1e6)
            volatility = top_data["vol_20d"].reindex(all_tickers).fillna(0.02)

            participation = trade_value / adv
            if (participation > 0.1).any():
                logger.warning(
                    f"Capacity limit breached on {date}. "
                    f"Max participation: {participation.max():.1%}"
                )

            impact_cost = trade_value * impact_coeff * volatility * np.sqrt(participation)

            # Total Costs (fixed + slippage + spread + impact)
            total_cost_usd = (fixed_cost + slippage_cost + spread_cost + impact_cost).sum()
            cost_pct = total_cost_usd / current_capital

            # Weighted return net of costs (use raw returns, not residual)
            # R3-B-4 fix: get returns for ALL held tickers (including blended
            # carryovers from prev_weights), not just this period's top-N.
            raw_col = f"raw_target_{rebalance_days}d"
            pnl_col = raw_col if raw_col in top.columns else f"target_{rebalance_days}d"
            # Start with top tickers' returns
            ticker_returns = top.set_index("ticker")[pnl_col]
            # Add any carryover tickers from blending that aren't in this period's top
            missing_tickers = [t for t in weights.index if t not in ticker_returns.index]
            if missing_tickers and "ticker" in group.columns:
                carryover = group[group["ticker"].isin(missing_tickers)].set_index("ticker")
                if pnl_col in carryover.columns:
                    ticker_returns = pd.concat([ticker_returns, carryover[pnl_col]])
            aligned_w = weights.reindex(ticker_returns.index, fill_value=0.0)
            # Renormalize aligned weights to account for any still-missing tickers
            aw_sum = aligned_w.sum()
            if aw_sum > 0 and abs(aw_sum - 1.0) > 0.01:
                aligned_w = aligned_w / aw_sum
            ret_gross = (ticker_returns * aligned_w).sum()
            ret_net = ret_gross - cost_pct

            # Update Capital (R3-B-8 fix: clamp to prevent negative capital)
            current_capital *= 1 + max(ret_net, -1.0)
            current_capital = max(current_capital, 0.0)

            all_returns.append(
                {
                    "date": date,
                    "return": ret_net,
                    "return_gross": ret_gross,
                    "turnover": turnover,
                    "cost": cost_pct,
                }
            )

            # Record weight snapshot
            weight_history.append({"date": date, **weights.to_dict()})
            prev_weights = weights

    returns_df = pd.DataFrame(all_returns).set_index("date")
    portfolio_returns = returns_df["return"]
    gross_returns = returns_df["return_gross"]

    # C-SHARPE fix: use geometric annualization and subtract risk-free rate.
    # Old formula used arithmetic mean × (252/rebal_days) without rf,
    # inconsistent with risk.py and standard practice.
    periods_per_year = 252 / rebalance_days
    ann_return = (1 + portfolio_returns.mean()) ** periods_per_year - 1
    ann_vol = portfolio_returns.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    ann_return_gross = (1 + gross_returns.mean()) ** periods_per_year - 1
    # R3-B-1 fix: gross Sharpe uses gross-return vol, not net-return vol
    ann_vol_gross = gross_returns.std() * np.sqrt(periods_per_year)
    sharpe_gross = (ann_return_gross - RISK_FREE_RATE) / ann_vol_gross if ann_vol_gross > 0 else 0

    avg_turnover = total_turnover / max(n_rebalances, 1)
    cumulative = (1 + portfolio_returns).cumprod()
    max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()

    # Latest weights
    latest_weights = prev_weights

    # Weight history DataFrame
    weight_history_df = pd.DataFrame(weight_history).set_index("date").fillna(0.0)

    # Build results dict
    results = {
        "portfolio_returns": portfolio_returns,
        "gross_returns": gross_returns,
        "weights": latest_weights,
        "weight_history": weight_history_df,
        "turnover_series": returns_df["turnover"],
        "annualized_return": ann_return,
        "annualized_return_gross": ann_return_gross,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sharpe_ratio_gross": sharpe_gross,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "total_cost_bps": transaction_cost_bps,
        "deflated_sharpe": deflated_sharpe_ratio(
            sharpe, n_trials=n_splits, n_observations=len(portfolio_returns)
        ),
        "optimizer_method": optimizer_method,
        "n_folds": n_splits,
        "risk_manager_enabled": enable_risk_manager,
    }

    # Add comprehensive risk metrics if we have enough history
    if len(portfolio_returns) > 30:
        from python.portfolio.risk import RiskEngine

        # Use actual portfolio weights from the simulation (Fix #38)
        risk_tickers = [t for t in latest_weights.index if t in prices.columns]
        if risk_tickers:
            risk_weights = latest_weights.reindex(risk_tickers).fillna(0.0)
            if risk_weights.sum() > 0:
                risk_weights = risk_weights / risk_weights.sum()
            risk_returns = prices[risk_tickers].pct_change().dropna().iloc[-min(len(prices), 252) :]
            risk_engine = RiskEngine(
                returns=risk_returns,
                weights=risk_weights,
            )

            # Add key risk metrics to results
            results["risk_metrics"] = {
                "sortino_ratio": risk_engine.sortino_ratio(),
                "calmar_ratio": risk_engine.calmar_ratio(),
                "omega_ratio": risk_engine.omega_ratio(),
                "var_95": risk_engine.var_historical(0.95),
                "cvar_95": risk_engine.cvar_historical(0.95),
                "downside_deviation": risk_engine.downside_deviation(),
            }

    # Add RiskManager summary if enabled
    if risk_manager is not None:
        results["risk_summary"] = risk_manager.get_risk_summary()

    return results


def save_results(results: dict, output_dir: Path = RESULTS_DIR) -> None:
    """Persist backtest results to disk for the dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results["portfolio_returns"].to_frame("return").to_parquet(
        output_dir / "backtest_returns.parquet"
    )
    results["weights"].to_json(output_dir / "backtest_weights.json")
    results["weight_history"].to_parquet(output_dir / "backtest_weight_history.parquet")
    results["turnover_series"].to_frame("turnover").to_parquet(
        output_dir / "backtest_turnover.parquet"
    )

    serializable_keys = {
        "annualized_return",
        "annualized_return_gross",
        "annualized_volatility",
        "sharpe_ratio",
        "sharpe_ratio_gross",
        "max_drawdown",
        "avg_turnover",
        "total_cost_bps",
        "deflated_sharpe",
        "optimizer_method",
        "n_folds",
    }
    metrics = {}
    for k, v in results.items():
        if k in serializable_keys:
            metrics[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
    with open(output_dir / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved backtest results to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = pd.read_parquet("data/raw/sp500_ohlcv.parquet")
    prices = extract_close_prices(raw)
    real_ohlcv = reshape_ohlcv_wide_to_long(raw)

    # Strategy comparison
    methods = ["equal_weight", "hrp", "risk_parity", "black_litterman", "min_cvar"]
    comparison = []
    for method in methods:
        r = run_backtest(
            prices,
            optimizer_method=method,
            transaction_cost_bps=10.0,
            max_weight=0.15,
            blend_alpha=0.3,
            macro_path="data/raw/macro_indicators.parquet",
            ohlcv=real_ohlcv,
        )
        comparison.append(
            {
                "method": method,
                "sharpe_net": r["sharpe_ratio"],
                "sharpe_gross": r["sharpe_ratio_gross"],
                "ann_return": r["annualized_return"],
                "max_dd": r["max_drawdown"],
                "avg_turnover": r["avg_turnover"],
            }
        )
        # Save the HRP run as the primary result (best risk-adjusted optimizer)
        if method == "hrp":
            save_results(r)

    comp_df = pd.DataFrame(comparison).set_index("method")
    logger.info(f"\nStrategy Comparison:\n{comp_df.to_string(float_format='%.3f')}")

    # Save comparison
    comp_df.to_csv(RESULTS_DIR / "strategy_comparison.csv")
    logger.info(f"Saved strategy comparison to {RESULTS_DIR / 'strategy_comparison.csv'}")
