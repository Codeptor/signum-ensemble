# Parallel Agent Coordination — Audit Round 2

**Created:** 2026-02-27
**Purpose:** Fix all findings from paper trading readiness audit round 2
**Source:** Compiled from 6 parallel audit agents reviewing the full codebase

---

## Worktree Layout

| Agent | Worktree Path | Branch | Scope |
|-------|---------------|--------|-------|
| **Claude Code** | `.worktrees/claude-code` | `fix/audit-round2-execution` | Live bot, broker, execution bridge, risk manager |
| **OpenCode** | `.worktrees/opencode` | `fix/audit-round2-pipeline` | Features, predict, ensemble, model, ingestion, backtest, portfolio/risk.py, optimizer, regime |

Both branch from `main` at commit `152b865`.

---

## File Ownership (STRICT - no cross-editing)

### Claude Code owns:
- `examples/live_bot.py`
- `python/brokers/alpaca_broker.py`
- `python/brokers/base.py`
- `python/bridge/execution.py`
- `python/portfolio/risk_manager.py`
- `tests/` files for the above modules

### OpenCode owns:
- `python/alpha/features.py`
- `python/alpha/predict.py`
- `python/alpha/ensemble.py`
- `python/alpha/model.py`
- `python/alpha/train.py`
- `python/monitoring/regime.py`
- `python/data/ingestion.py`
- `python/backtest/validation.py`
- `python/backtest/run.py`
- `python/backtest/robustness.py`
- `python/backtest/regime_analysis.py`
- `python/portfolio/risk.py`
- `python/portfolio/optimizer.py`
- `tests/` files for the above modules

### Shared (coordinate before editing):
- `pyproject.toml`
- `AGENTS.md` (this file)

---

## Task Assignments

### Claude Code — Branch: `fix/audit-round2-execution`

#### Phase 1 — CRITICAL (must fix for paper trading)

| ID | File | Fix |
|----|------|-----|
| **C-OCO-1** | `live_bot.py:732` | OCO order malformed — set `take_profit_limit_price=tp_price` on BrokerOrder instead of `limit_price=tp_price` on parent |
| **C-OCO-2** | `live_bot.py:755` | SL fallback uses unreliable `dir()` — restructure to compute `sl_price`/`sl_tp_qty` unconditionally before OCO try block |
| **C-STARTUP** | `live_bot.py:910` | `_has_traded_today` exits process on non-rebalance days if GTC SL/TP fills exist — make conditional on `should_rebalance_today()` OR filter to buy-side entry orders only |
| **C-WARN** | `risk_manager.py:409` | `MAX_SINGLE_TRADE_SIZE` and `MAX_SECTOR_WEIGHT` are `severity="warning"` so never block — change to `severity="critical"` |
| **C-INIT** | `risk_manager.py:83` | `current_weights=None` disables leverage/trade-size checks — initialize to `pd.Series(dtype=float)` |
| **C-SECTOR** | `risk_manager.py:291` | Sector weight check uses signed `new_weight` — use `abs()` for gross sector exposure |
| **C-DD** | `risk_manager.py:341` | Drawdown kill switch reads historical yfinance data — wire to live `ExecutionBridge.equity_curve` or pass actual portfolio equity |

#### Phase 2 — HIGH

| ID | File | Fix |
|----|------|-----|
| **H-PARTIAL** | `live_bot.py:677` | Partial fills not reconciled — after fill, re-query broker for actual `filled_qty` and update risk manager weights accordingly |
| **H-TIMEOUT** | `live_bot.py:244` | Timeout returns `filled_qty=0` — after cancel, do one more `get_order()` to capture actual partial fill qty |
| **H-LIQRACE** | `live_bot.py:133` | Liquidation: if `cancel_all_orders()` fails, GTC stops may race with market sells — retry cancel or verify cancellation |
| **H-CLAMP** | `live_bot.py:568` | Post-renorm clamping breaks sum-to-1 — add iterative renorm-clamp loop (renorm → clamp → renorm until stable) |
| **H-CAUTION** | `live_bot.py:549-576` | Caution mode 50% scaling undone by price-filter renorm — skip renormalization when in caution mode, or renorm to caution target (0.5) not 1.0 |
| **H-TZ** | `live_bot.py:88` | `should_rebalance_today()` uses naive local time — use `datetime.now(tz=ZoneInfo("America/New_York"))` |
| **H-IDTZ** | `alpaca_broker.py:121` | `client_order_id` uses local `date.today()` while `_has_traded_today` uses UTC — switch to UTC in both places |
| **H-EQCURVE** | `execution.py:299` | Equity curve compaction overlaps — fix: `historical = self.equity_curve[:-1000][::10]; recent = self.equity_curve[-1000:]; self.equity_curve = historical + recent` |
| **H-SHORT** | `execution.py:244` | Short sell cash check compares against `self.cash` — should compare against `self.equity` or margin-adjusted fraction |
| **H-STALE** | `execution.py:370` | Stale position closes rejected by risk manager — bypass risk check for closes (target weight is 0.0) |
| **H-PRICES** | `alpaca_broker.py:407` | Sequential per-symbol price fetch hits rate limits — use Alpaca's batch `get_latest_trades(symbols)` API |
| **H-PRINT** | `base.py:207,233` | `print()` in reconcile_portfolio — replace with `logger.error()` |

#### Phase 3 — MEDIUM

| ID | File | Fix |
|----|------|-----|
| **M-CONNECT** | `alpaca_broker.py:93` | `@_retry_read` on `connect()` ineffective — remove internal try/except, let decorator handle retry |
| **M-NANWT** | `execution.py:469` | NaN signal weights lock positions permanently — add `pd.isna(weight)` guard |
| **M-GETPOS** | `alpaca_broker.py:344` | Unexpected API errors return `None` — raise on non-404 errors |
| **M-SIGTERM** | `live_bot.py:879` | SIGTERM `finally` block overwrites richer handler state — check for existing SIGTERM state before overwriting |
| **M-CANCEL** | `live_bot.py:275` | SL/TP cancel filter matches all sell-side limits — add `order_class` check |
| **M-SECTORPASS** | `risk_manager.py:287` | Sector check emits no passing RiskCheck — add else branch with passing check |

#### Test Coverage Gaps to Fill

| ID | File | What to Test |
|----|------|-------------|
| **T-SHORT** | `test_risk_manager.py` | Short position (`new_weight < 0`) must trigger MAX_POSITION_SIZE and MAX_LEVERAGE |
| **T-OCO-TOPUP** | `test_live_integration.py` | OCO qty must equal total broker position on top-up, not just fill increment |
| **T-ATOMIC** | `test_live_bot_helpers.py` | Atomic write: `.tmp` residue from crash, old state survives mid-write |
| **T-LIQFAIL** | `test_live_bot_helpers.py` | Liquidation: timeout alert fires; submit failure for one symbol doesn't block others |
| **T-HASDAY** | `test_live_bot_helpers.py` | `_has_traded_today`: string vs datetime `created_at`, yesterday orders, broker exception |
| **T-FLIP** | `test_execution.py` | `Position.update()` short-to-long flip: avg_cost and realized PnL correctness |
| **T-NOPRICE** | `test_execution.py` | `reconcile_target_weights` silently skips stale position with no price |
| **T-RENORM** | `test_live_integration.py` | After renorm + clamp, no weight exceeds MAX_POSITION_WEIGHT |

---

### OpenCode — Branch: `fix/audit-round2-pipeline`

#### Phase 1 — CRITICAL (must fix for paper trading)

| ID | File | Fix |
|----|------|-----|
| **C-WINS** | `features.py:112` | Winsorization bounds differ train vs inference — save per-column (lo, hi) bounds at training time and apply fixed bounds at inference. Add `save_winsorize_bounds()` / `load_winsorize_bounds()` |
| **C-TARGET** | `features.py:204` | Target winsorized BEFORE residualization — either (a) remove target winsorize from `compute_forward_returns` and apply after `compute_residual_target`, or (b) skip target winsorize entirely (Huber loss handles outliers) |
| **C-NAN** | `ingestion.py:130` | `dropna(how="all")` keeps partial-NaN rows — change to per-ticker NaN handling: drop individual ticker columns where NaN persists after ffill, not rows |
| **C-CORR** | `ensemble.py:203` | `np.corrcoef` returns NaN on zero-variance predictions — guard with `np.isnan(ic)` check; replace `max(0.0, ic)` with `0.0 if np.isnan(ic) or ic < 0 else ic` |
| **C-PURGE** | `validation.py:114` | Purge gap in row-space, not date-space — split on unique dates, not raw row indices. Purge must be ≥ horizon (5 trading days) in calendar space |
| **C-SHARPE** | `run.py:377` | Backtest Sharpe uses arithmetic annualization without rf subtraction — switch to geometric: `ann_return = (1 + mean)**(252/rebal_days) - 1` and subtract `rf_rate` |

#### Phase 2 — HIGH

| ID | File | Fix |
|----|------|-----|
| **H-ICVAL** | `ensemble.py:126` | Same `val_df` used for LightGBM early stopping AND IC calibration — split val into early-stopping holdout and IC-calibration holdout |
| **H-PEARSON** | `validation.py:196` | IC computed as Pearson — switch to `scipy.stats.spearmanr` for rank IC. Also fix in `ensemble.py:203-208` |
| **H-YFIN** | `ingestion.py:87` | `auto_adjust` not set in `yf.download` — add `auto_adjust=True` to every call |
| **H-WIKI** | `ingestion.py:45` | Wikipedia scrape: no timeout, no ticker count validation — add requests timeout and assert `490 <= len(tickers) <= 520` |
| **H-SURV** | `predict.py:249` | Survivorship bias acknowledged but unmitigated — add a `TRAINING_LOOKBACK = "2y"` option to reduce bias magnitude; add doc comment estimating bias |
| **H-SIGNAL** | `predict.py:370` | Market-neutral residual signal drives directional long-only — add minimum score threshold: if median prediction < 0, reduce exposure proportionally |
| **H-SORTINO** | `risk.py:81` | `downside_deviation` uses `n_downside` denominator — fix: `dd = np.sqrt(np.minimum(excess, 0)**2).mean())` over ALL observations |
| **H-WINSGLOB** | `features.py:201` | Target winsorization uses full-dataset percentiles — compute bounds on train split only (may be fixed by C-WINS) |
| **H-MULTIIDX** | `ingestion.py:215` | `df[(t, "Close")]` assumes MultiIndex level order — use `df.xs("Close", axis=1, level=1)` or check level values |
| **H-MACRO** | `features.py:267` | VIX macro merge drops training rows via NaN — add bfill for dates before first macro observation, or fill VIX with neutral default (20.0) |
| **H-TICKER** | `predict.py:367` | `get_level_values(0)` implies MultiIndex but `latest["ticker"]` assumes column — pick one convention and enforce |
| **H-HRP** | `optimizer.py:68` | HRP on 6mo/10-stock is noisy — increase to `MIN_HISTORY_DAYS = "1y"` or add shrinkage to correlation estimate |

#### Phase 3 — MEDIUM

| ID | File | Fix |
|----|------|-----|
| **M-BOOTSTRAP** | `robustness.py:341` | IID bootstrap destroys autocorrelation — switch to stationary block bootstrap |
| **M-EARLYSTOP** | `model.py:86` | `early_stopping(stopping_rounds=10)` too aggressive — increase to 50 |
| **M-SHARPE3** | `robustness.py:38` | Third distinct Sharpe formula — centralize into shared utility |
| **M-REGIME-SHARPE** | `regime_analysis.py:143` | Sharpe denominator uses `returns.std()` not `excess.std()` — fix |
| **M-ATR** | `features.py:370` | `get_current_atr` returns `None` on failure — add type guard or return neutral default |
| **M-MACROPATH** | `features.py:242` | Relative `macro_path` — resolve relative to project root using `Path(__file__).parent` |
| **M-LOGMSG** | `ingestion.py:129` | Warning says "dropping N NaN values" but only drops fully-empty rows — fix message to match behavior |
| **M-VIX** | `regime.py:228` | VIX stale on Mondays — add check: if last VIX date is >1 trading day old, log warning and use VIX futures fallback or neutral default |
| **M-HYSTERESIS** | `regime.py:103` | AND de-escalation locks strategy in halt for months — change to: if VIX normalizes, allow caution even if drawdown persists |
| **M-TURNOVER** | `optimizer.py:207` | Turnover underestimates sell-side — ensure `current_weights` includes all held positions, not just target tickers |

---

## Status Tracking

### Claude Code Status
- [x] Phase 1: C-OCO-1, C-OCO-2, C-STARTUP, C-WARN, C-INIT, C-SECTOR, C-DD (+ H-TZ)
- [x] Phase 2: H-PARTIAL, H-TIMEOUT, H-LIQRACE, H-CLAMP, H-CAUTION, H-IDTZ, H-EQCURVE, H-SHORT, H-STALE, H-PRICES, H-PRINT
- [x] Phase 3: M-CONNECT, M-NANWT, M-GETPOS, M-SIGTERM, M-CANCEL, M-SECTORPASS
- [x] Tests: T-SHORT, T-OCO-TOPUP, T-ATOMIC, T-LIQFAIL, T-HASDAY, T-FLIP, T-NOPRICE, T-RENORM

### OpenCode Status
- [x] Phase 1: C-WINS, C-TARGET, C-NAN, C-CORR, C-PURGE, C-SHARPE (commit 7de00a7)
- [x] Phase 2: H-ICVAL, H-PEARSON, H-YFIN, H-WIKI, H-SURV, H-SIGNAL, H-SORTINO, H-WINSGLOB, H-MULTIIDX, H-MACRO, H-TICKER, H-HRP (commit d295a8d)
- [x] Phase 3: M-BOOTSTRAP, M-EARLYSTOP, M-SHARPE3, M-REGIME-SHARPE, M-ATR, M-MACROPATH, M-LOGMSG, M-VIX, M-HYSTERESIS, M-TURNOVER (commit 8d4023b)

---

## Communication Protocol

1. **Before editing a shared file:** Write which file and why in your status section
2. **When done with a batch:** Commit to your branch, update status here
3. **If you need something from the other agent's files:** Ask the user to relay
4. **Merge order:** Claude Code merges first (execution layer), OpenCode second (pipeline layer)

## Breaking Changes to Watch

If OpenCode changes the signature or behavior of any function called by `live_bot.py` or `execution.py`, note it here:
- `get_current_atr()` now accepts optional `default` parameter (backward-compatible, defaults to None)
- `RegimeDetector.get_regime()` halt→caution de-escalation changed from AND to OR logic (M-HYSTERESIS)
- `compute_metrics()` in robustness.py now uses geometric Sharpe with rf=5% (values will differ from old arithmetic formula)

If Claude Code changes `RiskManager` API, note here:
- (none yet)

## Running Tests

```bash
# From your worktree root:
uv run pytest tests/ -x -q

# Claude Code:
uv run pytest tests/test_live_bot_helpers.py tests/test_live_integration.py tests/bridge/ tests/brokers/ tests/portfolio/test_risk_manager.py -x -q

# OpenCode:
uv run pytest tests/alpha/ tests/backtest/ tests/data/ tests/portfolio/test_risk.py tests/monitoring/ -x -q
```
