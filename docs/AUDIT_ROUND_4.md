# Audit Round 4 — Comprehensive Bug Fixes

**Date:** 2026-03-01
**Branch:** `feature/comprehensive-improvements` (Bot B — ensemble pipeline)
**Commits:** 13 fix commits (416f3ef..5d7cec3)
**Files changed:** 21 files, +544/-201 lines
**Test result:** 1444 passed, 0 failed (was 1443 — gained 1 test)

---

## Summary

Full codebase audit identifying 6 P0, 17 P1, and 5 P2 issues. All 28 items resolved (1 P1 verified as non-issue). Fixes span the ML pipeline, risk management, execution, monitoring, and documentation.

---

## Before vs After

### P0: Critical (would cause money loss or silent failure in production)

| # | Issue | Before | After | Files |
|---|-------|--------|-------|-------|
| P0-1 | OCO orders rejected by Alpaca | Parent order missing `limit_price` — ALL OCO orders fell through to SL-only fallback. Take-profit never triggered. | `limit_price=tp_price` set on parent. OCO bracket accepted by Alpaca. | `live_bot.py` |
| P0-2 | `list_positions()` returns `[]` on API failure | Failure silently returned empty list. Bot thought it had zero positions and rebalanced from scratch, potentially doubling exposure. | Raises exception. All 5 callers handle gracefully (abort, retry, or continue with cached data). | `alpaca_broker.py`, `live_bot.py` |
| P0-3 | Dashboard API has no authentication | Anyone with the VPS IP could read positions, equity, and risk metrics via `/api/*` endpoints. | Bearer token auth on all `/api/*` routes (`DASHBOARD_API_TOKEN` env var). `/healthz` and UI remain public. | `dashboard.py` |
| P0-4 | Drawdown kill switch uses stale bridge equity | `bridge.equity_curve` only updates weekly (on rebalance). A 15% intra-week crash wouldn't trigger liquidation until next Wednesday. | Uses `broker.get_account().equity` (real-time from Alpaca) with persisted `equity_peak` in `bot_state.json`. | `live_bot.py` |
| P0-5 | `is_connected()` returns stale boolean | On a long-running VPS, sessions expire but `_connected` stays `True`. All downstream operations fail silently. | Actually probes `api.get_account()`. On failure, sets `_connected = False`. | `alpaca_broker.py` |
| P0-6 | Double-winsorization train/serve skew | Features winsorized twice during training (once in `compute_alpha_features`, again in CV loop), but only once during inference. Distribution skew between train and serve. | `compute_alpha_features(skip_winsorize=True)` in training. CV loop manages its own per-fold winsorization. Inference path unchanged. | `features.py`, `train.py` |

### P1: Important (correctness, robustness, or data quality)

| # | Issue | Before | After | Files |
|---|-------|--------|-------|-------|
| P1-7 | EWM without `min_periods` | MACD ema12/ema26/signal computed from first data point, producing unstable warm-up features. | `min_periods=12/26/9` set on all EWM calls. NaN during warm-up. | `features.py` |
| P1-8 | `sector_rel_mom` failure = KeyError | On sector lookup failure, `sector_rel_mom` column didn't exist. Model crashed on missing feature. | Exception fills `sector_rel_mom` with NaN. | `features.py` |
| P1-9 | Minimum portfolio size = 1 | Single-stock portfolio possible when most predictions are negative. Catastrophic idiosyncratic risk. | Minimum 3 stocks. | `predict.py` |
| P1-10 | Global mutable `_last_prediction_scores` | Hidden data flow via module-level global. Stale scores on error paths, not thread-safe. | `rank_stocks()` returns `(tickers, scores)` tuple. Explicit data flow. | `predict.py`, tests |
| P1-11 | Drawdown controller deadlock | After hard liquidation (factor=0), peak persists at old high. Portfolio can't recover (no positions = no growth). | Reset peak to current value after hard liquidation. System can re-enter. | `drawdown_control.py` |
| P1-12 | VaR limit = 6% daily | 6% daily VaR95 means $6K potential loss per day on $100K — far too loose for weekly rebalancing. | 2.5% daily VaR95 — consistent with institutional equity L/S. | `risk_manager.py` |
| P1-13 | `record_trade` doesn't enforce limits | Trade could be recorded even if it violated `max_single_trade_size`. | Returns `False` and blocks trades exceeding limit. Defense-in-depth. | `risk_manager.py` |
| P1-14 | HMM never refits | Model fitted once at startup. Months later, volatility clusters are stale. | `needs_refit()` method with 30-day interval. Caller can check and refit. | `hmm_regime.py` |
| P1-15 | VIX thresholds: docs say 25/35, code uses 30/40 | AGENTS.md had wrong VIX thresholds, misleading operators. | AGENTS.md updated to match actual code (30/40). | `AGENTS.md` |
| P1-16 | CatBoost uses RMSE loss | LightGBM uses Huber (robust to outliers), CatBoost uses RMSE (sensitive). Inconsistent signal quality across ensemble. | CatBoost switched to `Huber:delta=1.0`. All tree models now use robust loss. | `ensemble.py` |
| P1-17 | Negative IC clamped to 0 | A model with IC=-0.05 (inversely predictive) was treated the same as IC=0 (random). Lost useful information. | Negative IC preserved. `calibrate_weights()` already clamps at weighting stage. | `ensemble.py` |
| P1-18 | Confidence sizing maps min-score to 0 | [0,1] normalization gave the lowest-scored stock zero conviction weight, effectively removing it despite optimizer selection. | [0.5, 1.5] range keeps every stock at 50-150% of base weight. | `predict.py` |
| P1-19 | TCA penalizes favorable execution | `total_cost_bps = abs(is_bps) + commission` counted favorable fills (negative IS) as costs. | `total_cost_bps = is_bps + commission` (signed). Preserves cost/savings distinction. | `tca.py` |
| P1-20 | SHAP only explains LightGBM | In ensemble mode, SHAP only ran on LightGBM sub-model. Misleading feature importance for a 3-model ensemble. | SHAP computed for LightGBM + Random Forest, averaged. More representative. | `train.py` |
| P1-21 | VaR sign convention inconsistency | *Investigated: VaR is consistently negative in the live pipeline. `risk_manager` uses `abs()` for limit comparison correctly.* | Non-issue — no change needed. | — |
| P1-22 | Telegram alerts block trading loop | `_send_telegram()` synchronous — 100-2000ms latency during network issues could block rebalance. | Background threads for Telegram and webhook (same pattern as email). | `alerting.py` |
| P1-23 | `compute_residual_target` mutates input | In-place modification of the `labeled` DataFrame before CV splitting could corrupt upstream references. | `.copy()` before modification. | `features.py` |

### P2: Improvement (defense-in-depth, accuracy, observability)

| # | Issue | Before | After | Files |
|---|-------|--------|-------|-------|
| P2-24 | No pre-batch portfolio VaR check | Risk checks ran per-trade but not on the whole proposed portfolio before any orders. | Pre-rebalance check: if proposed portfolio breaches critical limits, rebalance blocked entirely. | `live_bot.py` |
| P2-25 | No liquidity gate | Could attempt large orders in low-liquidity names, causing market impact. | Orders exceeding 5% of 20-day ADV are skipped with a warning. | `live_bot.py` |
| P2-26 | Equity curve compaction loses drawdowns | `[::10]` subsampling could skip the exact equity trough, making drawdown calculations optimistic. | Min/max preserved in each 10-point window during compaction. | `execution.py` |
| P2-27 | Dashboard regime detector loses hysteresis | New `RegimeDetector()` created each call, resetting `_previous_regime`. Hysteresis logic defeated. | Module-level detector persists across calls. | `dashboard.py` |
| P2-28 | Drift detection false alarm rate ~43% | Testing 11 features at p<0.05 without correction: family-wise error rate ≈ 1-(1-0.05)^11 ≈ 43%. | Bonferroni correction: p < 0.05/n_features. Family-wise error rate = 5%. | `drift.py` |

---

## Risk Impact Assessment

### What these fixes prevent:

1. **P0-1 + P0-4**: Combined, these could cause >15% portfolio loss. OCO orders weren't working (no take-profit) AND the kill switch wasn't monitoring real-time equity. A flash crash scenario: positions have no upside exit, and the emergency brake doesn't engage until next weekly cycle.

2. **P0-2**: Position doubling. If Alpaca API hiccups during rebalance, the bot thought it had zero positions and would buy the full target allocation again.

3. **P0-6**: Train/serve skew degraded model performance silently. Features saw different distributions at training vs inference, reducing effective IC.

4. **P1-11**: After any hard drawdown event, the system was permanently locked out of trading. Required manual intervention to restart.

5. **P1-12**: The 6% VaR limit was cosmetic — virtually never triggered. The new 2.5% limit is a meaningful constraint.

### What these fixes improve:

1. **P1-16 + P1-17**: Ensemble signal quality. CatBoost now uses robust loss, and negative IC is preserved for proper weighting.

2. **P1-18**: Portfolio allocation quality. Lowest-conviction stocks no longer get zero weight.

3. **P2-24 + P2-25**: Pre-trade safety gates. VaR check prevents entering a high-risk portfolio. Liquidity gate prevents market impact.

4. **P2-28**: Drift monitoring reliability. False alarm rate dropped from ~43% to 5%.

---

## Test Results

```
Before: 1443 tests passing
After:  1444 tests passing (1 new test for P1-13 record_trade enforcement)
```

All 1444 tests pass in ~210 seconds. No regressions.
