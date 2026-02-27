# Signum

[![CI](https://github.com/Codeptor/signum/actions/workflows/ci.yml/badge.svg)](https://github.com/Codeptor/signum/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-589%20passed-brightgreen)]()
[![Paper Trading](https://img.shields.io/badge/status-paper%20trading-blue)]()

Automated quantitative equity trading system. Trains a LightGBM model weekly on S&P 500 data, selects top 10 stocks by predicted 5-day return, optimizes portfolio weights via HRP, and executes through Alpaca with ATR-based stop-loss/take-profit brackets.

Currently paper trading on a DigitalOcean VPS. Collecting data for 3+ months before evaluating for real capital.

## How It Works

Every Wednesday at market open, the bot runs a full cycle:

```
Scrape S&P 500 tickers (Wikipedia)
        │
        ▼
Fetch 2yr daily OHLCV (yfinance, full ~500 S&P 500 universe)
        │
        ▼
Compute 22 alpha features (momentum, volatility, RSI, volume, cross-sectional ranks, VIX)
        │
        ▼
Train LightGBM (Huber loss, residual return target, 80/20 date-split with 5-day embargo)
        │
        ▼
Score all ~500 S&P 500 stocks using saved winsorization bounds
        │
        ▼
Select top 10 by predicted residual return
        │
        ▼
Optimize weights via HRP (Ledoit-Wolf covariance shrinkage)
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
Send trade summary via Telegram, persist state, sleep until next Wednesday
```

Between Wednesdays the bot sleeps. GTC stop-loss and take-profit orders sit on Alpaca's servers and fire automatically.

### Regime Detection

The bot monitors VIX and SPY drawdown continuously:

| Regime | Condition | Action |
|--------|-----------|--------|
| **Normal** | VIX < 25, SPY DD < 8% | Full exposure |
| **Caution** | VIX 25-35 or SPY DD 8-15% | 50% exposure (all weights halved) |
| **Halt** | VIX > 35 and SPY DD > 15% | Liquidate everything, wait 1 hour |

De-escalation uses OR logic (either VIX or drawdown clearing allows caution).

### Model Ensemble

Two-model ensemble with IC-weighted calibration:

- **LightGBM** (60%) — gradient boosting, captures non-linear feature interactions
- **Random Forest** (40%) — bagging, robust to outliers

Weights are dynamically recalibrated each training cycle using Spearman rank IC on a held-out validation set.

## Deployment

The bot runs on a VPS as two systemd services:

| Service | What | Port |
|---------|------|------|
| `signum-bot` | Trading bot (sleeps between Wednesdays) | — |
| `signum-dashboard` | Dash web UI + JSON API | 8050 (localhost only) |

Both auto-restart on crash and start on boot.

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

Public at `https://dashboard.bhanueso.dev` (nginx reverse proxy + Let's Encrypt SSL).

Two tabs:
- **Live** — account overview, open positions, regime beacon, equity curve, bot log viewer
- **Backtest** — historical performance, drawdown, rolling Sharpe

### JSON API

12 endpoints, all return structured JSON with CORS headers:

| Endpoint | Returns |
|----------|---------|
| `GET /api` | Index of all endpoints |
| `GET /api/status` | System overview (regime + account + bot state) |
| `GET /api/account` | Alpaca account (equity, cash, buying power) |
| `GET /api/positions` | Open positions with unrealized P&L |
| `GET /api/regime` | VIX, SPY drawdown, exposure multiplier |
| `GET /api/equity` | Equity history time-series |
| `GET /api/risk` | Full risk engine output (VaR, Sharpe, drawdowns) |
| `GET /api/drift` | Feature drift report (PSI per feature) |
| `GET /api/bot` | Bot state (last trade, shutdown reason) |
| `GET /api/backtest` | Backtest metrics and risk summary |
| `GET /api/logs` | Bot log lines (`?lines=N`, default 80, max 500) |
| `GET /healthz` | Health check (bot liveness, alerting, data freshness) |

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
| VaR (95%, daily) | 6% | Warning logged |
| Min Sharpe ratio | -0.5 | Warning logged |
| Max volatility | 30% annualized | Warning logged |

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
│   │   ├── features.py          # 22 alpha features + winsorization
│   │   ├── model.py             # LightGBM/CatBoost wrapper
│   │   ├── ensemble.py          # LightGBM + RF ensemble with IC calibration
│   │   ├── predict.py           # End-to-end: data → features → rank → optimize
│   │   └── train.py             # Training pipeline orchestrator
│   ├── portfolio/
│   │   ├── optimizer.py         # HRP, Min-CVaR, Black-Litterman, Risk Parity
│   │   ├── risk.py              # VaR, CVaR, Sharpe, Sortino, drawdowns
│   │   ├── risk_manager.py      # Real-time trade gating
│   │   └── risk_attribution.py  # Marginal/component risk, Brinson-Fachler
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
│       ├── telegram_cmd.py      # Telegram command handler (/status, /positions, etc.)
│       ├── dashboard.py         # Dash web UI + JSON API (12 endpoints + /healthz)
│       ├── drift.py             # KS test + PSI feature drift detection
│       └── regime.py            # VIX/SPY-based regime detector
├── deploy/
│   ├── signum-bot.service       # systemd service file (trading bot)
│   └── signum-dashboard.service # systemd service file (web dashboard)
├── tests/                       # 589 tests
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

Three rounds of code audit (113+ findings resolved):

| Round | Findings | Focus |
|-------|----------|-------|
| 1 | 40 | Initial code review |
| 2 | 56 | Parallel audit by 6 agents across execution + ML pipeline |
| 3 | 37 | Final pre-paper-trading hardening |

Key fixes: OCO order construction, train/inference winsorization parity, date-space purged k-fold, geometric Sharpe standardization, Ledoit-Wolf covariance shrinkage, regime de-escalation logic, risk manager weight tracking, dynamic sector classification.

Post-audit additions: yfinance circuit breaker, centralized alerting module (Telegram + email), Telegram command handler, /healthz endpoint, structured JSON logging.

## Tests

```bash
uv run python -m pytest tests/ -x -q --tb=short
# 589 passed in ~77s
```

Coverage includes: ML pipeline (features, model, ensemble, predict), portfolio optimization, risk engine, risk manager, execution bridge, broker integration, backtest validation, robustness analysis, live bot helpers, alerting (Telegram, SendGrid, SMTP, webhook), Telegram command handler, and full integration tests.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | LightGBM, scikit-learn (Random Forest) |
| Portfolio | skfolio (HRP, CVaR, BL), Ledoit-Wolf shrinkage |
| Data | yfinance, pandas |
| Risk | scipy, numpy |
| Monitoring | Dash, Plotly, Flask (JSON API) |
| Alerting | Telegram Bot API, Resend, SendGrid, SMTP, webhooks |
| Broker | Alpaca Markets API (`alpaca-trade-api`) |
| Infra | DigitalOcean VPS, systemd, nginx, Let's Encrypt |
| Matching Engine | Rust, BTreeMap, Criterion.rs |

## Design Decisions

**Residual return target**: The model predicts stock-specific alpha (returns minus cross-sectional mean), not absolute returns. This means the model learns which stocks outperform relative to the average — works well in bull/flat markets, but in broad selloffs all positions lose money (regime detection partially mitigates this).

**Weekly rebalancing**: Reduces transaction costs ~74% vs daily. The model predicts 5-day forward returns, matching the rebalance frequency.

**HRP over mean-variance**: HRP uses hierarchical clustering + recursive bisection — no covariance matrix inversion required. More stable with 10 stocks and noisy correlation estimates.

**Long-only with market-neutral model**: The model is trained on residual returns but the bot only takes long positions. This wastes half the model's discriminative power by design. A long-short structure would capture more alpha but adds complexity (margin, locate fees, short squeeze risk) that isn't warranted during paper trading validation.

**ATR-based brackets over fixed percentages**: 2x ATR stop-loss adapts to each stock's volatility. A volatile stock gets a wider stop, a stable stock gets a tighter one. Falls back to fixed 5%/15% when ATR data is unavailable.
