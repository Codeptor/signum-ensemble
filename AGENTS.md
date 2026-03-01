# Signum — Agent Instructions (Ensemble Branch)

## What This Is

Signum is an automated quantitative equity trading bot. It trains a **LightGBM + CatBoost + RF ensemble** with a Ridge stacking meta-learner weekly on S&P 500 data, selects the top 10 stocks by predicted 5-day residual return, optimizes portfolio weights via HRP with confidence-weighted sizing, and executes through Alpaca with ATR-based stop-loss/take-profit brackets.

**This is the `feature/comprehensive-improvements` branch — Bot B (ensemble pipeline).**
**Bot A (simple LightGBM) runs on main branch at `209.38.122.78`.**

**Status:** Paper trading on a dedicated VPS. Running side-by-side with Bot A for 3+ months to compare strategies before evaluating for real capital.

## Critical Rules

- **Use `uv run` for everything** — `uv run python -m pytest tests/ -x -q --tb=short`, never raw `python` or `pip`
- **Use `pnpm` for the dashboard** — `pnpm dev`, `pnpm build`, never npm/yarn
- **Test command:** `uv run python -m pytest tests/ -x -q --tb=short` — must pass **1443+ tests**
- **Never commit secrets** — `.env`, API keys, SSH keys (`deploy/signum_ed25519`) are gitignored
- **The bot defaults to paper trading.** Only `LIVE_TRADING=true` env var activates real money. Do not set this.
- **LSP errors** about unresolved imports (numpy, pandas, pytest, etc.) are pre-existing venv-path issues — **ignore them**

## Architecture

### Trading Bot (Python)

```
examples/live_bot.py          Entry point — runs weekly on Wednesdays
    │
    ├── python/data/ingestion.py       Scrape S&P 500 tickers, fetch 2yr OHLCV
    ├── python/alpha/features.py       27 computed features (11 active in reduced set) + winsorization + volatility estimators
    ├── python/alpha/model.py          LightGBM (Huber loss) wrapper
    ├── python/alpha/ensemble.py       LightGBM + CatBoost + RF + Ridge stacking meta-learner
    ├── python/alpha/train.py          Purged walk-forward CV + SHAP + alpha decay + MLflow
    ├── python/alpha/predict.py        End-to-end: data → features → rank → optimize → confidence sizing
    ├── python/alpha/explainability.py SHAP feature importance per CV fold
    │
    ├── python/risk/volatility.py      Yang-Zhang, Parkinson, Garman-Klass, EWMA volatility
    │
    ├── python/portfolio/optimizer.py  HRP, Min-CVaR, Black-Litterman, Risk Parity
    ├── python/portfolio/risk.py       VaR, CVaR, Sharpe, Sortino, drawdowns
    ├── python/portfolio/risk_manager.py  Real-time trade gating + graduated drawdown control
    ├── python/portfolio/tca.py        Transaction cost analysis (IS bps, fill rates)
    ├── python/portfolio/drawdown_control.py  CPPI overlay, drawdown deleveraging
    │
    ├── python/bridge/execution.py     Order submission, position tracking, P&L
    ├── python/brokers/alpaca_broker.py  Alpaca Markets API implementation
    ├── python/brokers/base.py         Abstract broker interface + data classes
    │
    ├── python/monitoring/alerting.py     Multi-channel alerts (Telegram, Resend, SendGrid, SMTP, webhook)
    ├── python/monitoring/telegram_cmd.py Telegram command handler (/status, /positions, /tca, etc.)
    ├── python/monitoring/dashboard.py    Dash web UI + JSON API endpoints + /healthz + /api/tca
    ├── python/monitoring/regime.py       VIX/SPY-based threshold regime detector
    ├── python/monitoring/hmm_regime.py   Gaussian HMM regime detector (primary)
    └── python/monitoring/drift.py        KS test + PSI feature drift detection
```

### Web Dashboard (Next.js)

```
dashboard/                              Next.js 16 app (pnpm, Vercel-deployed)
    ├── app/page.tsx                    Single-page A/B comparison dashboard
    ├── app/layout.tsx                  Root layout (dark mode, JetBrains Mono)
    ├── app/api/bot/[bot]/[...path]/    Server-side API proxy → VPS backends
    │       route.ts                    Maps bot-a/bot-b → BOT_A_URL/BOT_B_URL env vars
    ├── lib/api.ts                      8 fetch functions (status, positions, risk, tca, drift, equity, logs, health)
    ├── lib/types.ts                    TypeScript interfaces for all API responses
    └── components/ui/                  shadcn/ui primitives (card, table, chart, badge, tabs, etc.)
```

## Dashboard

The Next.js dashboard provides **real-time A/B comparison** of Bot A vs Bot B during paper trading.

**URL:** Deployed on Vercel (`preferredRegion = "bom1"` Mumbai for low VPS latency)
**Env vars:** `BOT_A_URL`, `BOT_B_URL` (VPS Dash server URLs, set in Vercel project settings)

### Panels (top to bottom)

1. **Header** — "SIGNUM" branding, market session badge (Pre-market/Open/After-hours/Closed) with countdown, Bot A/B tab switcher, triple clock (NY/IST/UTC), auto-refresh countdown (30s)
2. **Comparison Strip** — Side-by-side Bot A vs Bot B: health dot, equity, P&L vs $100K start, positions, regime, VIX, SPY drawdown
3. **Hero Metrics** — Portfolio equity/cash/buying power, market regime + exposure multiplier, bot state (online/offline + last shutdown)
4. **Dual Equity Chart** — Recharts AreaChart overlaying Bot A (solid) and Bot B (dashed) equity curves with tooltip
5. **Risk Metrics** — Sharpe, Sortino, max drawdown, current drawdown, VaR/CVaR 95%, win rate, total trades
6. **Sector Exposure** — Horizontal bar chart showing GICS sector weights (when positions exist)
7. **Positions Table** — Symbol, qty, avg entry, current price, market value, P&L ($/%); footer totals row
8. **Logs / TCA / Drift** — Live bot logs (80 lines), IS bps + fill rate, drifted feature count + names

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Switch to Bot A |
| `2` | Switch to Bot B |
| `R` | Force immediate refresh |
| `Space` | Pause/resume auto-refresh |

### Backend API Endpoints (consumed by dashboard)

| Endpoint | Data |
|----------|------|
| `/api/status` | Tick, regime, equity, VIX, SPY drawdown, position count |
| `/api/positions` | Open positions with entry/current prices and P&L |
| `/api/risk` | Sharpe, Sortino, VaR, CVaR, drawdowns, win rate |
| `/api/tca` | Avg IS bps, fill rate, trade count |
| `/api/drift` | Feature drift KS test + PSI results |
| `/api/equity` | Equity history time series |
| `/api/logs?lines=80` | Recent bot log output |
| `/healthz` | Health check (online/offline + last shutdown) |

### Tech Stack

| Component | Detail |
|-----------|--------|
| Framework | Next.js 16.1.6 (App Router) |
| React | 19.2.3 |
| Styling | Tailwind CSS v4 + shadcn/ui (radix-lyra, oklch dark theme) |
| Charts | Recharts 2.15.4 |
| Font | JetBrains Mono |
| Icons | @hugeicons/react + @hugeicons/core-free-icons |
| Deployment | Vercel (Mumbai region) |
| Package manager | pnpm |

## Key Differences from Bot A (main branch)

| Feature | Bot A (main) | Bot B (this branch) |
|---------|-------------|-------------------|
| Model | Single LightGBM | LightGBM + CatBoost + RF + Ridge stacking |
| Features | 22 alpha features | 27 features (11 active in reduced set: +mom_12_1, mr_zscore_60, vol_yz_20d, sector_rel_mom) |
| Cross-validation | Date-based split | Purged walk-forward CV (5 folds, 22-day embargo) |
| Regime detection | VIX/SPY threshold only | HMM (primary) + threshold (fallback, consensus for halt) |
| Position sizing | HRP weights only | HRP + confidence-weighted blend (70/30) |
| Drawdown control | Binary kill switch at 15% | Graduated deleveraging + kill switch |
| Explainability | None | SHAP per fold + alpha decay analysis |
| Tracking | None | MLflow experiment tracking |
| TCA | None | Implementation shortfall in bps per trade |
| Dashboard | None | Next.js A/B comparison dashboard on Vercel |
| Tests | 594 | 1443+ |

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Timezone | NY (`ZoneInfo("America/New_York")`) for trading, UTC for internal timestamps | Market hours are Eastern |
| Risk-free rate | `RISK_FREE_RATE = 0.05` in `python/data/config.py` | Centralized, used by all Sharpe calcs |
| Embargo | 22 business days | Matches longest feature lookback window |
| OCO brackets | 2x ATR SL / 3x ATR TP | Adapts to per-stock volatility |
| Training universe | Full ~500 S&P 500 tickers | No sampling |
| IC quality gate | `MIN_VALIDATION_IC = 0.02` | Falls back to equal-weight if model is weak |
| Covariance | Ledoit-Wolf shrinkage (OAS fallback) | Stable with 10 stocks |
| Ensemble weights | LightGBM 45% + CatBoost 30% + RF 25% (base), Ridge stacking | Diversified signal |
| Confidence sizing | blend_alpha=0.3 (70% risk-based, 30% conviction) | Conservative blend |
| HMM regime | Consensus with threshold for halt (both must agree) | Prevents false liquidation |
| MLflow | Local file store (`./mlruns/`), auto-pruned at 30 days | No external server needed |
| `get_ml_weights()` | Returns `(dict, bool)` tuple — `(weights, stale_data)` | All callers must destructure |

## Trading Schedule

- **Rebalance:** Weekly on Wednesdays (configurable: `REBALANCE_DAY=2`)
- **Between rebalances:** Bot sleeps. GTC stop-loss and take-profit orders sit on Alpaca's servers.
- **Regime detection:** Continuous — HMM + VIX/SPY drawdown checked each cycle

| Regime | HMM State | Threshold Condition | Action |
|--------|-----------|-------------------|--------|
| Normal | Low-vol | VIX < 25, SPY DD < 8% | Full exposure |
| Caution | High-vol OR crisis | VIX 25-35 or SPY DD 8-15% | 50% exposure |
| Halt | Crisis (consensus) | VIX > 35 AND SPY DD > 15% AND HMM=crisis | Liquidate, wait |

## Risk Limits

| Check | Limit | Blocks trade? |
|-------|-------|--------------|
| Max position weight | 30% | Yes |
| Max sector weight | 25% | Yes |
| Max single trade size | 15% | Yes |
| Max leverage | 1.0x | Yes |
| Max drawdown | 15% | Kill switch — liquidates all |
| Graduated drawdown | 5-15% | Proportional deleveraging |
| Max daily trades | 50 | Warning only |
| Max daily turnover | 100% | Warning only |

## Running Tests

```bash
# Full suite (should pass 1443+ tests in ~140s)
uv run python -m pytest tests/ -x -q --tb=short

# Specific modules
uv run python -m pytest tests/monitoring/ -x -q            # Alerting + Telegram + dashboard + HMM
uv run python -m pytest tests/alpha/ -x -q                 # ML pipeline + ensemble + explainability
uv run python -m pytest tests/backtest/ -x -q              # Backtesting + CPCV + purged CV
uv run python -m pytest tests/portfolio/ -x -q             # Optimization + risk + TCA + drawdown
uv run python -m pytest tests/risk/ -x -q                  # Volatility estimators
uv run python -m pytest tests/bridge/ tests/brokers/ -x -q # Execution + brokers
uv run python -m pytest tests/test_live_bot_helpers.py tests/test_live_integration.py -x -q  # Live bot
```

## Alerting

### Telegram Bot

**Outbound alerts** (18 events) fire automatically:
- Bot startup, shutdown, trade cycle summary, heartbeat (hourly)
- Stale data, order timeout, partial fill, OCO failure, risk violation
- ML pipeline failure, Alpaca connect failure
- Caution mode, halt mode, drawdown kill switch, liquidation

**Interactive commands** (polling every 3s):
- `/status` `/positions` `/equity` `/regime` `/health` `/trades` `/logs` `/tca` `/help`

### Config (in VPS `.env`)
```
TELEGRAM_BOT_TOKEN=<bot token from @BotFather>
TELEGRAM_CHAT_ID=<your chat ID>
```

## Research Modules (not in live pipeline)

These 24+ modules exist for research and backtesting only. They do NOT import into or affect the live trading path:

- **Alpha:** conformal prediction, feature importance/stability, meta-labeling, online learning, pairs trading, signal combining
- **Portfolio:** analytics, blend optimizer, Brinson attribution, factor risk, Kelly sizing, market impact, Monte Carlo, regime optimizer, turnover
- **Execution:** TWAP/VWAP algorithms, microstructure analysis, advanced TCA (IS decomposition)
- **Risk:** factor risk model
- **Backtest:** CPCV, PBO, walk-forward optimization

## Known Limitations (documented, not fixing)

- **Survivorship bias:** Uses current S&P 500 list for historical data
- **`alpaca-trade-api` deprecated:** Works fine but Alpaca recommends migration to `alpaca-py`
- **`rolling_beta` crash:** Only called from dashboard, never from live bot
- **HMM re-downloads SPY 3mo each cycle:** Acceptable for weekly rebalancing frequency
- **MLflow local store:** Auto-pruned at 30 days; no remote tracking server

## Audit History

Three audit rounds (113+ findings resolved) + post-audit hardening:

| Round | Findings | Focus |
|-------|----------|-------|
| 1 | 45 | Full codebase review (14 P0, 24 P1, 7 P2) |
| 2 | 56 | Parallel audit by 6 agents (execution + ML pipeline) |
| 3 | 37 | Final pre-paper-trading hardening |
| Post | — | yfinance circuit breaker, alerting module, Telegram commands, /healthz, structured logging |
| Feature | 28 | Bug fixes cherry-picked, ensemble + HMM + TCA + confidence sizing integrated |

See `docs/AUDIT_REPORT.md` and `docs/PAPER_TRADING_READINESS.md` for details.
