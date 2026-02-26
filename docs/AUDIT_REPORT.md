# Signum Quant Platform — Codebase Audit Report

**Date:** 2026-02-26  
**Scope:** Full codebase review of all Python source, tests, configuration, and infrastructure  
**Platform:** ML-driven live trading bot (LightGBM ranking + HRP optimization) targeting Alpaca paper trading  
**Test baseline:** 122 tests passing

---

## Executive Summary

This audit reviewed every Python source file, test file, configuration file, and infrastructure script in the Signum platform. **45 findings** were identified across three severity levels:

| Severity | Count | Description |
|----------|-------|-------------|
| **P0 — Critical** | 14 | Will cause incorrect behavior, financial loss, or data corruption |
| **P1 — High** | 24 | Significant issues affecting robustness, correctness, or reliability |
| **P2 — Medium** | 7 | Improvements for code quality and maintainability |

The most dangerous cluster of bugs is in `execution.py` (the ExecutionBridge), where equity is calculated using a single price for all positions, positions are never closed when removed from targets, and equity mutates mid-reconciliation. Together these mean the bot's internal state diverges from reality on every cycle.

The second critical cluster is in `live_bot.py`, where the risk engine is never initialized (making all risk checks no-ops), bracket stop-loss orders are cancelled during the ML pipeline leaving the portfolio unhedged, and the bridge is recreated every cycle discarding all state.

The third critical cluster is in the ML pipeline (`train.py`, `predict.py`, `model.py`), where train/val splits use iloc (look-ahead bias), the live model trains on all data with no validation, and LightGBM has no early stopping.

---

## P0 — Critical Findings

These will cause incorrect behavior or financial loss. Fix before any live trading.

### 1. `_update_equity` uses a single price for ALL positions

**File:** `python/bridge/execution.py:225-226`

```python
total_position_value = sum(pos.market_value(current_price) for pos in self.positions.values())
```

If the portfolio holds AAPL and MSFT, both positions are valued using whichever stock's price was passed. This corrupts equity, all weight calculations, and all downstream risk checks.

**Fix:** Accept a `Dict[str, float]` of prices and look up each position's price individually.  
**Effort:** 30 minutes

---

### 2. `update_prices` passes only first price to equity calculation

**File:** `python/bridge/execution.py:242-243`

```python
self._update_equity(list(prices.values())[0])
```

After correctly updating per-position unrealized P&L with individual prices, it then recalculates total equity using only the first price in the dict.

**Fix:** Refactor `_update_equity` to accept the full price dict (see Finding #1).  
**Effort:** Included in Finding #1

---

### 3. `reconcile_target_weights` doesn't close stale positions

**File:** `python/bridge/execution.py:279-328`

The method only iterates over `target_weights.items()`. Positions held in the portfolio but absent from the new target weights are never sold, causing them to accumulate indefinitely.

**Fix:** Add a loop that identifies positions not in `target_weights` and generates sell orders to close them.  
**Effort:** 1 hour

---

### 4. Equity mutates mid-reconciliation loop

**File:** `python/bridge/execution.py:303`

Each simulated fill within the reconciliation loop changes `self.equity`, making subsequent `target_value = target_weight * self.equity` calculations wrong for later tickers. The last ticker in the loop sees a materially different equity than the first.

**Fix:** Snapshot equity before the loop and use the snapshot for all target calculations.  
**Effort:** 15 minutes

---

### 5. `weight_change` always positive for sells

**File:** `python/bridge/execution.py:178-181`

```python
weight_change = (fill.fill_quantity * fill.fill_price) / self.equity
```

`fill_quantity` and `fill_price` are both positive regardless of buy/sell side, so `weight_change` is always positive. This corrupts `current_weights` tracking in the risk manager — sells are recorded as weight increases.

**Fix:** Negate `weight_change` when `fill.side == "sell"`.  
**Effort:** 15 minutes

---

### 6. `reconcile_portfolio` skips target_weight=0

**File:** `python/brokers/base.py:186`

```python
if target_weight <= 0: continue
```

Positions that should be fully exited (target weight of 0) are skipped, so they are never closed.

**Fix:** Change condition to only skip negative weights; handle zero-weight targets by generating close orders.  
**Effort:** 30 minutes

---

### 7. Train/val split uses iloc — look-ahead bias

**File:** `python/alpha/train.py:44-46`

```python
split_idx = int(len(labeled) * 0.8)
train = labeled.iloc[:split_idx]
val = labeled.iloc[split_idx:]
```

For cross-sectional (panel) data, rows from the same date can appear in both train and val sets because iloc splits by row index, not by date. This is textbook look-ahead bias and will make backtest metrics unreliable.

**Fix:** Split by date with an embargo gap (e.g., 5 trading days). Sort by date, find the 80th percentile date, add embargo, assign sets.  
**Effort:** 1 hour

---

### 8. Live model trains on ALL data with no validation

**File:** `python/alpha/predict.py:96-163`

The `train_model()` function used during live inference trains on the entire dataset including the most recent data that overlaps with forward-looking prediction targets. There is no held-out evaluation.

**Fix:** Reserve the most recent N days for validation, add early stopping, log validation metrics.  
**Effort:** 2 hours

---

### 9. No early stopping in LightGBM

**File:** `python/alpha/model.py:57-58`

150 trees are trained unconditionally with no `eval_set` and no `callbacks`. The model will overfit, especially on small datasets.

**Fix:** Add `eval_set`, `callbacks=[lgb.early_stopping(10)]`, and a validation split.  
**Effort:** 30 minutes (dependent on Finding #7/#8 for proper split)

---

### 10. Risk engine never initialized in live bot

**File:** `examples/live_bot.py` (full file)

`risk_manager.initialize_portfolio_risk()` is never called, so `risk_engine` is always `None`. Every call to `check_portfolio_risk()` returns an empty list of violations. The `MAX_DRAWDOWN_LIMIT = 0.15` constant is defined but never checked against actual drawdown.

**Fix:** Call `initialize_portfolio_risk()` after fetching initial positions and prices. Add drawdown check in the main loop.  
**Effort:** 1 hour

---

### 11. Bracket orders cancelled during ML pipeline

**File:** `examples/live_bot.py:56-63`

`cancel_all_orders()` removes ALL pending orders — including stop-loss legs from bracket orders — before the ML pipeline runs. The pipeline takes ~2.5 minutes, during which the portfolio has zero downside protection.

**Fix:** Only cancel non-bracket orders, or cancel bracket orders only after new stop-losses are placed.  
**Effort:** 2 hours

---

### 12. ExecutionBridge recreated every cycle

**File:** `examples/live_bot.py:110`

```python
bridge = ExecutionBridge(...)
```

This is inside the main loop, so every cycle discards all internal state: positions, equity curve, P&L history, and weight tracking. The bridge starts fresh with no knowledge of what happened in prior cycles.

**Fix:** Move bridge creation outside the loop. Alternatively, hydrate the bridge from Alpaca's positions/account at the start of each cycle.  
**Effort:** 1 hour

---

### 13. No fill verification

**File:** `examples/live_bot.py` (full file)

Orders are submitted to Alpaca and the bot assumes they executed successfully. There is no polling of order status, no handling of partial fills, no retry logic. The bot then sleeps for 12 hours.

**Fix:** After submitting orders, poll `broker.get_order(order_id)` until terminal state. Handle partial fills and rejections.  
**Effort:** 3 hours

---

### 14. Spread cost hardcoded to zero (debug code)

**File:** `python/backtest/run.py:324`

```python
spread_cost = trade_value * spread * 0.5 * 0.0  # Setting to 0 for a moment
```

The `* 0.0` at the end zeroes out all spread costs in backtests. This makes backtest returns unrealistically high and masks the impact of transaction costs.

**Fix:** Remove `* 0.0`.  
**Effort:** 1 minute

---

## P1 — High Findings

Significant issues that should be fixed before production use.

### 15. Leverage check double-counts existing positions

**File:** `python/portfolio/risk_manager.py:204-205`

```python
total_weight = self.current_weights.sum() + new_weight
```

If this is a weight change for an existing position, the new weight is added to the total that already includes the old weight, double-counting.

**Fix:** Subtract the old weight for the symbol before adding the new weight.  
**Effort:** 30 minutes

---

### 16. Turnover limit defined but never enforced

**File:** `python/portfolio/risk_manager.py`

`max_daily_turnover` is tracked via `record_trade()` but never checked in `check_trade()` or `can_execute_trade()`.

**Fix:** Add turnover check in `check_trade()`.  
**Effort:** 30 minutes

---

### 17. Sector weight limit never enforced

**File:** `python/portfolio/risk_manager.py`

`max_sector_weight` is defined in `RiskLimits` but no sector data is tracked or checked anywhere.

**Fix:** Add sector mapping and enforcement, or remove the unused config to avoid false sense of security.  
**Effort:** 2 hours (if implementing), 5 minutes (if removing)

---

### 18. No max-weight constraint in HRP optimizer

**File:** `python/portfolio/optimizer.py`

HRP can produce weights up to ~15%+ for individual positions with no cap. There is also no error handling if skfolio raises an exception.

**Fix:** Add post-optimization weight capping with redistribution. Wrap skfolio calls in try/except.  
**Effort:** 1 hour

---

### 19. Duplicate features: `bid_ask_proxy` and `hl_range`

**File:** `python/alpha/features.py:73,76`

Both compute `(high - low) / close`. One is redundant and adds a perfectly correlated feature to the model.

**Fix:** Remove one of them.  
**Effort:** 10 minutes

---

### 20. No winsorization of features or targets

**File:** `python/alpha/features.py`

Extreme outliers flow directly into the model. LightGBM is somewhat robust to outliers, but extreme values in targets can still distort training.

**Fix:** Add percentile clipping (e.g., 1st/99th) to features and targets.  
**Effort:** 30 minutes

---

### 21. Volatility computed on arithmetic returns

**File:** `python/alpha/features.py`

```python
c.pct_change().rolling(w).std()
```

Log returns are more appropriate for volatility estimation as they are additive over time.

**Fix:** Use `np.log(c / c.shift(1))` instead of `pct_change()`.  
**Effort:** 15 minutes

---

### 22. MSE loss for ranking task

**File:** `python/alpha/model.py:29`

```python
"objective": "regression", "metric": "mse"
```

MSE is dominated by outlier targets. For cross-sectional stock ranking, a ranking objective (`lambdarank`) or robust loss (`huber`) would be more appropriate.

**Fix:** Switch to `"objective": "huber"` or evaluate lambdarank.  
**Effort:** 30 minutes + evaluation time

---

### 23. `predict()` crashes if model is None

**File:** `python/alpha/model.py:70`

No null check before `self.model.predict(features)`. If the model failed to load or train, this produces an unhandled AttributeError.

**Fix:** Add `if self.model is None: raise ValueError(...)`.  
**Effort:** 5 minutes

---

### 24. No random seed in LightGBM

**File:** `python/alpha/model.py`

Results are not reproducible across runs. Different seeds can produce materially different rankings.

**Fix:** Add `"seed": 42` and `"feature_fraction_seed": 42` to params.  
**Effort:** 5 minutes

---

### 25. Global random state in robustness tests

**File:** `python/backtest/robustness.py:193,339`

```python
np.random.seed(42)
```

Uses deprecated global random state. If any other code also sets global state, results become order-dependent.

**Fix:** Use `rng = np.random.default_rng(42)` and pass to all random calls.  
**Effort:** 30 minutes

---

### 26. `upsert_ohlcv` uses row-by-row SELECT queries

**File:** `python/data/store.py:30-61`

For each row in the DataFrame, a separate SELECT query checks if the row exists before inserting. This is O(n) queries per row, extremely slow for bulk data loads.

**Fix:** Use `INSERT ... ON CONFLICT DO UPDATE` (upsert) or bulk insert with a staging table.  
**Effort:** 1 hour

---

### 27. No retry/backoff on data ingestion

**File:** `python/data/ingestion.py`

yfinance and Wikipedia API calls have no retry logic. A single network hiccup causes immediate crash.

**Fix:** Add `tenacity.retry` with exponential backoff.  
**Effort:** 30 minutes

---

### 28. No NaN validation after yfinance download

**File:** `python/data/ingestion.py`

Missing data passes through silently. A stock with NaN prices will propagate through features, training, and predictions.

**Fix:** Add NaN checks after download; log warnings and drop or interpolate as appropriate.  
**Effort:** 30 minutes

---

### 29. Survivorship bias in S&P 500 universe

**File:** `python/data/ingestion.py`

`fetch_sp500_tickers()` scrapes current S&P 500 constituents from Wikipedia. Historical backtests using current constituents exclude companies that were delisted, acquired, or removed — classic survivorship bias that inflates backtest returns.

**Fix:** Maintain a point-in-time membership table (e.g., from Sharadar or manual snapshots). For paper trading only, this is acceptable but should be documented.  
**Effort:** 4+ hours (if implementing), 5 minutes (if documenting the limitation)

---

### 30. No retry/backoff on Alpaca API calls

**File:** `python/brokers/alpaca_broker.py`

Every API call has broad `except Exception` with no retry. Transient network failures cause the bot to skip trades entirely.

**Fix:** Add `tenacity.retry` with backoff to all broker API calls.  
**Effort:** 1 hour

---

### 31. No duplicate-order protection

**File:** `python/brokers/alpaca_broker.py`

No `client_order_id` is auto-generated. If a network timeout causes a retry at a higher level, the same order can be submitted twice.

**Fix:** Generate deterministic `client_order_id` based on symbol + timestamp + side + quantity.  
**Effort:** 30 minutes

---

### 32. Fractional shares truncated to int

**File:** `examples/live_bot.py:149`

```python
qty = int(abs(fill.fill_quantity))
```

Alpaca supports fractional shares, but the bot truncates to integers. For a $100k portfolio with 10 positions, this can leave $500+ in uninvested cash, and for expensive stocks (BRK.B, AMZN) the rounding error is significant.

**Fix:** Use fractional quantities; remove `int()` cast.  
**Effort:** 15 minutes

---

### 33. Stop-loss anchored to stale price

**File:** `examples/live_bot.py`

Stop-loss prices are calculated from `prices[symbol]` which comes from yfinance (potentially 15-minute delayed), not from the actual fill price. This can set stop-losses at inappropriate levels.

**Fix:** Use the fill price from the order response to calculate stop-loss levels.  
**Effort:** 30 minutes

---

### 34. 12-hour sleep too rigid

**File:** `examples/live_bot.py:44`

The bot sleeps for exactly 12 hours regardless of market hours, holidays, or early closes. It may attempt to trade when the market is closed, or miss trading windows.

**Fix:** Calculate next market open dynamically using `exchange_calendars` or Alpaca's calendar API.  
**Effort:** 1 hour

---

### 35. CI doesn't install broker extras

**File:** `.github/workflows/ci.yml`

CI only installs `[dev,portfolio]` extras, not `[broker]`. The entire broker integration layer (`alpaca_broker.py`) is untested in CI.

**Fix:** Add `[broker]` to the pip install command in CI.  
**Effort:** 5 minutes

---

### 36. No log rotation

**File:** `examples/live_bot.py`

`live_bot.log` grows unbounded. On a long-running bot, this can fill the disk.

**Fix:** Use `RotatingFileHandler` with max size and backup count.  
**Effort:** 15 minutes

---

### 37. No alerting on bot failure

**File:** All live trading files

If the bot crashes, there is no notification. It fails silently until manually discovered.

**Fix:** Add a try/except in the main loop that sends alerts (Slack webhook, email, or similar) on unhandled exceptions.  
**Effort:** 1-2 hours

---

### 38. Risk metrics use dummy equal weights in backtest

**File:** `python/backtest/run.py:415-424`

```python
dummy_weights = pd.Series({ticker: 1.0 / len(prices.columns) for ticker in prices.columns[:5]})
```

Risk attribution in the backtest summary uses equal weights across only the first 5 tickers, not the actual portfolio weights from the simulation.

**Fix:** Pass actual portfolio weights to the risk metrics calculation.  
**Effort:** 30 minutes

---

## P2 — Medium Findings

Improvements for code quality, correctness, and maintainability.

### 39. `train_start` calculated then immediately overridden

**File:** `python/backtest/validation.py:29-31`

```python
train_start = max(0, ...)
train_start = 0
```

The first calculation is dead code — it's immediately overwritten on the next line.

**Fix:** Remove the dead assignment.  
**Effort:** 1 minute

---

### 40. Sharpe ratio uses arithmetic annualization

**File:** `python/portfolio/risk.py:152-154`

```python
mean() * 252
```

Arithmetic annualization slightly overstates compounded returns. For a Sharpe of ~1.6 this makes a small but measurable difference.

**Fix:** Use geometric annualization: `((1 + mean)**252 - 1)`.  
**Effort:** 10 minutes

---

### 41. PSI bins use global min/max

**File:** `python/monitoring/drift.py`

Bin edges for Population Stability Index are computed from global min/max rather than reference distribution quantiles. This can produce misleading PSI values when the test distribution has different range.

**Fix:** Use reference distribution quantiles for bin edges.  
**Effort:** 30 minutes

---

### 42. BL views uncertainty may not match skfolio expectation

**File:** `python/bridge/bl_views.py`

`view_confidences` in skfolio's `BlackLitterman` expects Idzorek confidence (0-1), but the code converts to uncertainty values. The mapping may not match skfolio's internal implementation.

**Fix:** Verify against skfolio documentation; add a test with known inputs/outputs.  
**Effort:** 1 hour

---

### 43. Missing type hints across multiple files

Various files lack type annotations on function signatures, making the codebase harder to maintain and preventing static analysis tools from catching errors.

**Fix:** Add type hints incrementally, starting with public API surfaces.  
**Effort:** Ongoing

---

### 44. No integration tests for live trading path

Tests exist for individual components but not for the full `live_bot.py` → `predict.py` → `alpaca_broker.py` chain.

**Fix:** Add integration tests with a mock broker that simulate a full trading cycle.  
**Effort:** 4+ hours

---

### 45. No model versioning or artifact tracking

Trained models are saved to a single path and overwritten. There's no way to reproduce past predictions or roll back to a prior model version.

**Fix:** Add model versioning (timestamp or hash in filename) and optionally integrate with DVC or MLflow.  
**Effort:** 2-4 hours

---

## Recommended Fix Order

### Sprint 1 (Immediate — before any live trading)

| # | Finding | Effort | Impact |
|---|---------|--------|--------|
| 1-2 | Fix equity calculation in ExecutionBridge | 30 min | Eliminates portfolio valuation corruption |
| 3 | Close stale positions | 1 hr | Prevents position accumulation |
| 4 | Snapshot equity before reconciliation | 15 min | Correct target calculations |
| 5 | Fix weight_change sign for sells | 15 min | Correct risk manager tracking |
| 6 | Fix base broker zero-weight skip | 30 min | Positions actually close |
| 10 | Initialize risk engine in live bot | 1 hr | Risk checks actually work |
| 11 | Don't cancel bracket stop-losses | 2 hr | Maintain downside protection |
| 12 | Move bridge outside loop | 1 hr | Preserve state across cycles |
| 13 | Add fill verification | 3 hr | Know if orders executed |
| 14 | Remove `* 0.0` from spread cost | 1 min | Realistic backtest costs |

**Total Sprint 1:** ~10 hours

### Sprint 2 (ML Pipeline Fixes)

| # | Finding | Effort | Impact |
|---|---------|--------|--------|
| 7 | Date-based train/val split | 1 hr | Eliminate look-ahead bias |
| 8 | Validation set for live model | 2 hr | Detect overfitting |
| 9 | Early stopping in LightGBM | 30 min | Prevent overfitting |
| 22 | Switch to huber loss | 30 min | Robust to outlier targets |
| 24 | Add random seed | 5 min | Reproducible results |
| 19 | Remove duplicate feature | 10 min | Cleaner feature set |
| 20 | Add winsorization | 30 min | Handle outliers |

**Total Sprint 2:** ~5 hours

### Sprint 3 (Reliability & Operations)

| # | Finding | Effort | Impact |
|---|---------|--------|--------|
| 27-28 | Retry + NaN validation for data | 1 hr | Resilient data pipeline |
| 30-31 | Retry + idempotency for broker | 1.5 hr | Resilient order execution |
| 32-33 | Fractional shares + fill-price stop-loss | 45 min | Better execution quality |
| 34 | Dynamic sleep based on market hours | 1 hr | Correct scheduling |
| 35 | Add broker extras to CI | 5 min | Test coverage |
| 36-37 | Log rotation + alerting | 1.5 hr | Operational visibility |

**Total Sprint 3:** ~6 hours

### Sprint 4 (Risk Manager & Backtest Accuracy)

Remaining P1 and P2 items: leverage double-count, turnover enforcement, sector limits, HRP weight cap, backtest risk metrics, and code quality improvements.

**Total Sprint 4:** ~8 hours

---

## Total Estimated Effort

| Sprint | Effort | Focus |
|--------|--------|-------|
| Sprint 1 | ~10 hours | Execution correctness (stop the bleeding) |
| Sprint 2 | ~5 hours | ML pipeline integrity |
| Sprint 3 | ~6 hours | Reliability and operations |
| Sprint 4 | ~8 hours | Risk management and polish |
| **Total** | **~29 hours** | |

---

## Appendix: Files Audited

Every file below was read in full during this audit:

```
python/alpha/features.py
python/alpha/model.py
python/alpha/train.py
python/alpha/predict.py
python/alpha/tft_model.py
python/data/config.py
python/data/ingestion.py
python/data/models.py
python/data/store.py
python/portfolio/optimizer.py
python/portfolio/risk.py
python/portfolio/risk_manager.py
python/portfolio/risk_attribution.py
python/portfolio/blend_optimizer.py
python/portfolio/brinson_attribution.py
python/bridge/execution.py
python/bridge/bl_views.py
python/brokers/base.py
python/brokers/alpaca_broker.py
python/brokers/factory.py
python/monitoring/drift.py
python/monitoring/dashboard.py
python/backtest/run.py
python/backtest/validation.py
python/backtest/robustness.py
examples/live_bot.py
examples/dry_run.py
examples/live_trading_example.py
examples/market_closed_dashboard.py
examples/portfolio_growth_strategy.py
tests/ (18 test files, ~122 tests)
pyproject.toml
.github/workflows/ci.yml
Makefile
dvc.yaml
infra/docker-compose.yml
run_live_bot.sh
```
