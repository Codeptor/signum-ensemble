# Signum

[![CI](https://github.com/Codeptor/signum/actions/workflows/ci.yml/badge.svg)](https://github.com/Codeptor/signum/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-1443%2B%20passed-brightgreen)]()
[![Paper Trading](https://img.shields.io/badge/status-paper%20trading-blue)]()

Automated quantitative equity trading system. Trains a **LightGBM + CatBoost + RF ensemble** with a Ridge stacking meta-learner weekly on S&P 500 data, selects top 10 stocks by predicted 5-day residual return, optimizes portfolio weights via HRP with confidence-weighted sizing, and executes through Alpaca with ATR-based stop-loss/take-profit brackets.

Currently paper trading on a VPS. Bot A (LightGBM, `main` branch) and Bot B (ensemble, `feature/comprehensive-improvements` branch) run side-by-side for A/B comparison via a [Next.js dashboard](#dashboard).

## How It Works

Every Wednesday at market open, the bot runs a full cycle:

```
Scrape S&P 500 tickers (Wikipedia)
        │
        ▼
Fetch 2yr daily OHLCV (yfinance, full ~500 S&P 500 universe)
        │
        ▼
Compute 27 alpha features (11 active in reduced set: momentum, volatility,
    RSI, volume, cross-sectional ranks, sector-relative momentum, Yang-Zhang vol)
        │
        ▼
Train LightGBM + CatBoost + RF ensemble
    (Huber loss, residual return target, purged walk-forward CV,
     5 folds with 22-day embargo, SHAP importance per fold)
        │
        ▼
Score all ~500 stocks via Ridge stacking meta-learner
        │
        ▼
Select top 10 by predicted residual return
        │
        ▼
Optimize weights via HRP (Ledoit-Wolf covariance shrinkage)
    + confidence-weighted sizing (70% HRP, 30% conviction blend)
        │
        ▼
Risk checks (position size, sector exposure, leverage, VaR, drawdown)
        │
        ▼
Execute via Alpaca (sells first, then buys, poll for fills)
        │
        ▼
Attach OCO brackets (SL = 2x ATR below fill, TP = 3x ATR above fill)
        │
        ▼
Log TCA (implementation shortfall in bps), send trade summary via Telegram,
    persist state, sleep until next Wednesday
```

Between Wednesdays the bot sleeps. GTC stop-loss and take-profit orders sit on Alpaca's servers and fire automatically.

### Regime Detection

The bot uses a **Gaussian HMM** (primary) with **VIX/SPY threshold** (fallback). Both must agree to trigger halt — prevents false liquidation.

**HMM regime detector** (3 hidden states, trained on daily returns + rolling vol):

| HMM State | Exposure | Description |
|-----------|----------|-------------|
| **Low-vol** | 100% | Normal market conditions |
| **Normal** | 70% | Elevated volatility |
| **High-vol** | 30% | Crisis-level volatility |

**Threshold fallback** (used when HMM unavailable, consensus required for halt):

| Regime | Condition | Action |
|--------|-----------|--------|
| **Normal** | VIX < 25, SPY DD < 8% | Full exposure |
| **Caution** | VIX 25-35 or SPY DD 8-15% | 50% exposure (all weights halved) |
| **Halt** | VIX > 35 and SPY DD > 15% (+ HMM agrees) | Liquidate everything, wait 1 hour |

De-escalation uses OR logic (either VIX or drawdown clearing allows caution).

### Model Ensemble

Three-model ensemble with Ridge stacking meta-learner:

| Model | Base Weight | Role |
|-------|-------------|------|
| **LightGBM** | 45% | Gradient boosting, captures non-linear feature interactions |
| **CatBoost** | 30% | Gradient boosting with ordered target encoding |
| **Random Forest** | 25% | Bagging, robust to outliers |

Base weights are dynamically recalibrated each training cycle using Spearman rank IC on held-out validation data. A **Ridge regression** meta-learner is trained on out-of-sample base model predictions for final scoring. Falls back to IC-weighted averaging when meta-learner is unavailable.

### Alpha Features

27 features computed, 11 active in the reduced set used by the ensemble:

| Feature | Category |
|---------|----------|
| `ret_5d`, `ret_20d` | Momentum (short/medium-term) |
| `mom_12_1` | Jegadeesh-Titman 12-1 momentum |
| `rsi_14` | Mean reversion (RSI) |
| `bb_position` | Mean reversion (Bollinger position) |
| `mr_zscore_60` | Mean reversion (z-score) |
| `vol_20d` | Volatility (close-to-close) |
| `vol_yz_20d` | Volatility (Yang-Zhang OHLC) |
| `volume_ratio` | Volume confirmation |
| `cs_ret_rank_5d` | Cross-sectional relative strength |
| `sector_rel_mom` | Sector-relative momentum |

Full 27-feature set preserved for comparison. Features are winsorized at training time with bounds saved and reapplied at inference.

### Training Pipeline

- **Purged walk-forward CV** — 5 folds, 22 business day embargo (matches longest feature lookback), 5-day label purge
- **SHAP explainability** — per-fold feature importance via `TreeExplainer`, cross-fold stability (Jaccard similarity, Spearman rank correlation)
- **Alpha decay analysis** — IC measured at horizons [1, 5, 10, 20] days with signal half-life estimation
- **MLflow tracking** — all metrics, parameters, and fold results logged per training run (local file store, auto-pruned at 30 days)
- **IC quality gate** — `MIN_VALIDATION_IC = 0.02`; falls back to equal-weight if model is weak

## Deployment

The bot runs on a VPS as two systemd services:

| Service | What | Port |
|---------|------|------|
| `signum-bot` | Trading bot (sleeps between Wednesdays) | — |
| `signum-dashboard` | Dash web UI + JSON API | 8050 (localhost only) |

Both auto-restart on crash and start on boot.

A **Next.js dashboard** is deployed on Vercel for real-time A/B comparison of Bot A vs Bot B. See [Dashboard](#dashboard).

### Quick Deploy

```bash
# 1. Clone and install
git clone https://github.com/Codeptor/signum.git
cd signum
uv sync

# 2. Configure
cp .env.example .env
# Edit .env: set ALPACA_API_KEY and ALPACA_API_SECRET
# Get keys from https://app.alpaca.markets/paper/dashboard/overview

# 3. Dry run (full ML pipeline, no orders)
uv run python examples/dry_run.py

# 4. Run locally
uv run python examples/live_bot.py

# 5. Or deploy to VPS with systemd
sudo cp deploy/signum-bot.service /etc/systemd/system/
sudo systemctl enable --now signum-bot
```

### CLI Shortcuts (zsh)

After setup, these commands are available:

```
signum            SSH into the VPS
signum -h         Show all commands
signum-dash       Open dashboard (SSH tunnel + browser)
signum-status     Account, regime, bot state
signum-logs       Stream live bot logs
signum-positions  Current open positions with P&L
signum-regime     Market regime (VIX, SPY drawdown)
signum-restart    Restart the bot service
signum-stop       Stop the bot service
signum-deploy     Push local code to VPS and restart
```

## Monitoring

### Telegram Bot (Primary)

Control the bot from your phone via Telegram commands:

| Command | What it does |
|---------|-------------|
| `/status` | Bot state, regime, account overview |
| `/positions` | Current holdings with weights and P&L |
| `/equity` | Portfolio value, cash, buying power, total return |
| `/regime` | Market regime (VIX, SPY drawdown, exposure) |
| `/health` | System health check |
| `/trades` | Recent trade info |
| `/logs` | Last 20 log lines |
| `/tca` | Transaction cost analysis (IS bps, fill rate) |
| `/help` | List all commands |

Setup: create a bot via [@BotFather](https://t.me/BotFather), set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

### Alerts (18 events)

Automatic Telegram alerts for all critical events:

| Event | Severity |
|-------|----------|
| Bot startup / shutdown | INFO |
| Trade cycle summary (fills, equity, holdings) | INFO |
| Hourly heartbeat (silence = problem) | INFO |
| Stale data, order timeout, partial fill | WARNING |
| OCO bracket failure, risk violation, order rejection | WARNING |
| Caution mode (VIX/drawdown elevated) | WARNING |
| ML pipeline failure, Alpaca connection failure | CRITICAL |
| Halt mode, drawdown kill switch, liquidation | CRITICAL |

CRITICAL alerts bypass rate limiting. All others rate-limited to 20/5min.

Transport priority: Telegram (always) > Resend > SendGrid > SMTP for email.

### Dashboard

**Next.js A/B comparison dashboard** deployed on Vercel (`preferredRegion = "bom1"` Mumbai for low VPS latency).

Provides real-time side-by-side comparison of Bot A (main, LightGBM) vs Bot B (ensemble) during paper trading.

**Panels:**
1. **Header** — "SIGNUM" branding, market session badge (Pre-market/Open/After-hours/Closed) with countdown, Bot A/B tab switcher, triple clock (NY/IST/UTC), auto-refresh countdown (30s)
2. **Comparison Strip** — Side-by-side: health dot, equity, P&L vs $100K start, positions, regime, VIX, SPY drawdown
3. **Hero Metrics** — Portfolio equity/cash/buying power, market regime + exposure multiplier, bot state (online/offline + last shutdown)
4. **Dual Equity Chart** — Recharts AreaChart overlaying Bot A (solid) and Bot B (dashed) equity curves
5. **Risk Metrics** — Sharpe, Sortino, max drawdown, current drawdown, VaR/CVaR 95%, win rate, total trades
6. **Sector Exposure** — Horizontal bar chart showing GICS sector weights
7. **Positions Table** — Symbol, qty, avg entry, current price, market value, P&L ($/%); footer totals
8. **Logs / TCA / Drift** — Live bot logs (80 lines), IS bps + fill rate, drifted feature count

**Keyboard shortcuts:** `1` Bot A, `2` Bot B, `R` force refresh, `Space` pause/resume auto-refresh.

**Architecture:** Server-side API proxy (`app/api/bot/[bot]/[...path]/route.ts`) maps `bot-a`/`bot-b` to `BOT_A_URL`/`BOT_B_URL` env vars set in Vercel.

### JSON API (Python backend)

8 endpoints consumed by the Next.js dashboard, all return structured JSON:

| Endpoint | Returns |
|----------|---------|
| `GET /api/status` | Tick, regime, equity, VIX, SPY drawdown, position count |
| `GET /api/positions` | Open positions with entry/current prices and P&L |
| `GET /api/risk` | Sharpe, Sortino, VaR, CVaR, drawdowns, win rate |
| `GET /api/tca` | Avg IS bps, fill rate, trade count |
| `GET /api/drift` | Feature drift KS test + PSI results |
| `GET /api/equity` | Equity history time series |
| `GET /api/logs?lines=80` | Recent bot log output |
| `GET /healthz` | Health check (online/offline + last shutdown) |

## Risk Controls

### Trade-Level

| Check | Limit | Severity |
|-------|-------|----------|
| Max position weight | 30% | Critical (blocks trade) |
| Max sector weight | 25% | Critical |
| Max single trade size | 15% | Critical |
| Max leverage | 1.0x (long-only) | Critical |
| Max daily trades | 50 | Warning |
| Max daily turnover | 100% | Warning |

### Portfolio-Level

| Check | Limit | Action |
|-------|-------|--------|
| Max drawdown | 15% | Kill switch — liquidate all |
| Graduated drawdown | 10-20% | Linear deleveraging (exposure ramps from 100% to 0%) |
| Drawdown recovery | 5% | Hysteresis — must recover before re-leveraging |
| VaR (95%, daily) | 6% | Warning logged |
| Min Sharpe ratio | -0.5 | Warning logged |
| Max volatility | 30% annualized | Warning logged |

**Graduated drawdown control:** Linear interpolation of exposure between `max_dd` (10%, full exposure) and `hard_limit` (20%, zero exposure). Hysteresis prevents whipsawing — once deleveraging starts, drawdown must recover below 5% before re-leveraging. Also includes CPPI overlay (floor=90%, multiplier=3.0).

### Position Protection

- ATR-based stop-loss (2x ATR) and take-profit (3x ATR) via OCO orders
- Fallback to fixed 5% SL / 15% TP when ATR unavailable
- Orphaned order cleanup every cycle
- Duplicate execution prevention (checks `_has_traded_today` before trading)

## Project Structure

```
signum/
├── examples/
│   ├── live_bot.py              # Main trading bot (entry point)
│   ├── dry_run.py               # ML pipeline test without orders
│   └── paper_trading_tracker.py # CLI portfolio snapshot
├── python/
│   ├── alpha/
│   │   ├── features.py          # 27 alpha features (11 active) + winsorization
│   │   ├── model.py             # LightGBM (Huber loss) wrapper
│   │   ├── ensemble.py          # LightGBM + CatBoost + RF + Ridge stacking
│   │   ├── predict.py           # End-to-end: data → features → rank → optimize → confidence sizing
│   │   ├── train.py             # Purged walk-forward CV + SHAP + alpha decay + MLflow
│   │   └── explainability.py    # SHAP feature importance per CV fold + stability analysis
│   ├── portfolio/
│   │   ├── optimizer.py         # HRP, Min-CVaR, Black-Litterman, Risk Parity
│   │   ├── risk.py              # VaR, CVaR, Sharpe, Sortino, drawdowns
│   │   ├── risk_manager.py      # Real-time trade gating + graduated drawdown control
│   │   ├── risk_attribution.py  # Marginal/component risk, Brinson-Fachler
│   │   ├── tca.py               # Transaction cost analysis (IS bps, fill rates)
│   │   └── drawdown_control.py  # CPPI overlay, graduated deleveraging
│   ├── risk/
│   │   └── volatility.py        # Yang-Zhang, Parkinson, Garman-Klass, EWMA (7 estimators)
│   ├── bridge/
│   │   └── execution.py         # Order submission, position tracking, P&L
│   ├── brokers/
│   │   ├── base.py              # Abstract broker interface
│   │   └── alpaca_broker.py     # Alpaca Markets implementation
│   ├── data/
│   │   ├── ingestion.py         # S&P 500 scrape + yfinance OHLCV
│   │   └── sectors.py           # GICS sector map + dynamic yfinance lookup
│   ├── backtest/
│   │   ├── run.py               # Walk-forward backtest engine
│   │   ├── validation.py        # Purged k-fold CV + deflated Sharpe
│   │   ├── robustness.py        # Monte Carlo, block bootstrap, stress tests
│   │   └── regime_analysis.py   # Per-regime performance breakdown
│   └── monitoring/
│       ├── alerting.py          # Multi-channel alerts (Telegram, Resend, SendGrid, SMTP, webhook)
│       ├── telegram_cmd.py      # Telegram command handler (/status, /positions, /tca, etc.)
│       ├── dashboard.py         # Dash web UI + JSON API (8 endpoints + /healthz)
│       ├── drift.py             # KS test + PSI feature drift detection
│       ├── regime.py            # VIX/SPY-based threshold regime detector
│       └── hmm_regime.py        # Gaussian HMM regime detector (primary)
├── dashboard/                   # Next.js 16 A/B comparison dashboard (Vercel)
│   ├── app/
│   │   ├── page.tsx             # Single-page dashboard
│   │   ├── layout.tsx           # Root layout (dark mode, JetBrains Mono)
│   │   └── api/bot/[bot]/[...path]/route.ts  # Server-side API proxy
│   ├── lib/
│   │   ├── api.ts               # 8 fetch functions
│   │   └── types.ts             # TypeScript interfaces
│   └── components/ui/           # shadcn/ui primitives
├── deploy/
│   ├── signum-bot.service       # systemd service file (trading bot)
│   └── signum-dashboard.service # systemd service file (web dashboard)
├── tests/                       # 1443+ tests
├── rust/matching-engine/        # Lock-free order book (sub-microsecond)
├── run_live_bot.sh              # Bash wrapper with crash recovery
├── .env.example                 # Environment variable template
└── pyproject.toml               # Python 3.11, all dependencies
```

## Configuration

All parameters configurable via `.env`:

```bash
# Alpaca (required)
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret

# Strategy
TOP_N_STOCKS=10              # Stocks to hold
OPTIMIZER_METHOD=hrp         # hrp, min_cvar, risk_parity
REBALANCE_FREQUENCY=weekly   # daily or weekly
REBALANCE_DAY=2              # 0=Mon ... 4=Fri

# Risk
MAX_POSITION_WEIGHT=0.30     # 30% max per position
MAX_DRAWDOWN_LIMIT=0.15      # 15% kill switch
ATR_SL_MULTIPLIER=2.0        # Stop-loss at 2x ATR
ATR_TP_MULTIPLIER=3.0        # Take-profit at 3x ATR

# Alerts (recommended)
TELEGRAM_BOT_TOKEN=          # From @BotFather
TELEGRAM_CHAT_ID=            # Your chat ID
ALERT_WEBHOOK_URL=           # Slack/Discord webhook (optional)
```

## Backtest Results

Walk-forward backtest on S&P 500, LightGBM alpha (22 features), residual return target, top-20 portfolio, 5-day rebalancing, VIX scaling:

| Optimizer | Sharpe (net) | Ann. Return | Max DD | Avg Turnover |
|-----------|-------------|-------------|--------|-------------|
| Equal Weight | 1.66 | 24.6% | 50.0% | 36% |
| HRP | 1.28 | 13.9% | 40.8% | 39% |
| Black-Litterman | 0.99 | 10.5% | 31.8% | 44% |

**Known backtest limitations** (these don't affect live trading):
- Survivorship bias: uses current S&P 500 list for historical data (~1-3% annual return inflation)
- Forward return overlap: 5-day returns with 5-day rebalancing inflates Sharpe by ~sqrt(overlap)
- Feature leakage: backtest computes features on full dataset before train/test split (live pipeline correctly saves/loads bounds per training cycle)

## Audit History

Four audit rounds (113+ findings resolved) plus feature integration:

| Round | Findings | Focus |
|-------|----------|-------|
| 1 | 40 | Initial code review |
| 2 | 56 | Parallel audit by 6 agents across execution + ML pipeline |
| 3 | 37 | Final pre-paper-trading hardening |
| Post | — | yfinance circuit breaker, alerting module, Telegram commands, /healthz, structured logging |
| Feature | 28 | Bug fixes cherry-picked, ensemble + HMM + TCA + confidence sizing integrated |

Key fixes: OCO order construction, train/inference winsorization parity, date-space purged k-fold, geometric Sharpe standardization, Ledoit-Wolf covariance shrinkage, regime de-escalation logic, risk manager weight tracking, dynamic sector classification.

## Tests

```bash
uv run python -m pytest tests/ -x -q --tb=short
# 1443+ passed in ~140s
```

Coverage includes: ML pipeline (features, model, ensemble, predict, explainability), purged walk-forward CV, portfolio optimization, risk engine, risk manager, graduated drawdown control, TCA, execution bridge, broker integration, backtest validation, robustness analysis, HMM regime detection, live bot helpers, alerting (Telegram, SendGrid, SMTP, webhook), Telegram command handler, confidence sizing, alpha decay, and full integration tests.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | LightGBM, CatBoost, scikit-learn (Random Forest, Ridge) |
| Explainability | SHAP, MLflow |
| Portfolio | skfolio (HRP, CVaR, BL), Ledoit-Wolf shrinkage |
| Data | yfinance, pandas |
| Risk | scipy, numpy, hmmlearn (HMM regime) |
| Monitoring | Dash, Plotly, Flask (JSON API) |
| Dashboard | Next.js 16, React 19, Recharts, shadcn/ui, Tailwind CSS v4 |
| Alerting | Telegram Bot API, Resend, SendGrid, SMTP, webhooks |
| Broker | Alpaca Markets API (`alpaca-trade-api`) |
| Infra | VPS, systemd, Vercel (dashboard) |
| Matching Engine | Rust, BTreeMap, Criterion.rs |

## Design Decisions

**Residual return target**: The model predicts stock-specific alpha (returns minus cross-sectional mean), not absolute returns. This means the model learns which stocks outperform relative to the average — works well in bull/flat markets, but in broad selloffs all positions lose money (regime detection partially mitigates this).

**Weekly rebalancing**: Reduces transaction costs ~74% vs daily. The model predicts 5-day forward returns, matching the rebalance frequency.

**HRP over mean-variance**: HRP uses hierarchical clustering + recursive bisection — no covariance matrix inversion required. More stable with 10 stocks and noisy correlation estimates.

**Three-model ensemble with stacking**: LightGBM captures complex interactions, CatBoost adds ordered boosting diversity, RF provides bagging stability. Ridge stacking on OOS predictions avoids overfitting to any single model's biases. Falls back to IC-weighted averaging gracefully.

**Confidence-weighted sizing**: 70% HRP risk-based weights + 30% conviction from model prediction scores. Conservative blend (alpha=0.3) prevents overconcentration while allowing high-conviction signals to tilt allocation.

**Graduated drawdown control**: Linear deleveraging between 10-20% drawdown avoids the binary kill switch problem where a 14.9% drawdown does nothing but 15.1% liquidates everything. Hysteresis prevents whipsaw re-entry.

**Long-only with market-neutral model**: The model is trained on residual returns but the bot only takes long positions. This wastes half the model's discriminative power by design. A long-short structure would capture more alpha but adds complexity (margin, locate fees, short squeeze risk) that isn't warranted during paper trading validation.

**ATR-based brackets over fixed percentages**: 2x ATR stop-loss adapts to each stock's volatility. A volatile stock gets a wider stop, a stable stock gets a tighter one. Falls back to fixed 5%/15% when ATR data is unavailable.
