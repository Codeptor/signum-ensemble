# Signum — Agent Instructions

## What This Is

Signum is an automated quantitative equity trading bot. It trains a LightGBM model weekly on S&P 500 data, selects the top 10 stocks by predicted 5-day residual return, optimizes portfolio weights via HRP, and executes through Alpaca with ATR-based stop-loss/take-profit brackets.

**Status:** Live paper trading $100k on a DigitalOcean VPS. Collecting data for 3+ months before evaluating for real capital.

## Critical Rules

- **Use `uv run` for everything** — `uv run python -m pytest tests/ -x -q --tb=short`, never raw `python` or `pip`
- **Test command:** `uv run python -m pytest tests/ -x -q --tb=short` — must pass **589 tests**
- **Never commit secrets** — `.env`, API keys, SSH keys (`deploy/signum_ed25519`) are gitignored
- **The bot defaults to paper trading.** Only `LIVE_TRADING=true` env var activates real money. Do not set this.
- **LSP errors** about unresolved imports (numpy, pandas, pytest, etc.) are pre-existing venv-path issues — **ignore them**
- **Do NOT use rsync for VPS deploys** — use `scp -i deploy/signum_ed25519` instead
- **VPS SSH:** `ssh -i deploy/signum_ed25519 root@209.38.122.78`

## Architecture

```
examples/live_bot.py          Entry point — runs weekly on Wednesdays
    │
    ├── python/data/ingestion.py       Scrape S&P 500 tickers, fetch 2yr OHLCV
    ├── python/alpha/features.py       22 alpha features + winsorization
    ├── python/alpha/model.py          LightGBM (Huber loss) wrapper
    ├── python/alpha/predict.py        End-to-end: data → features → rank → optimize
    ├── python/alpha/ensemble.py       LightGBM + RF ensemble (research only, not wired into live)
    │
    ├── python/portfolio/optimizer.py  HRP, Min-CVaR, Black-Litterman, Risk Parity
    ├── python/portfolio/risk.py       VaR, CVaR, Sharpe, Sortino, drawdowns
    ├── python/portfolio/risk_manager.py  Real-time trade gating (position size, sector, leverage)
    │
    ├── python/bridge/execution.py     Order submission, position tracking, P&L
    ├── python/brokers/alpaca_broker.py  Alpaca Markets API implementation
    ├── python/brokers/base.py         Abstract broker interface + data classes
    │
    ├── python/monitoring/alerting.py     Multi-channel alerts (Telegram, Resend, SendGrid, SMTP, webhook)
    ├── python/monitoring/telegram_cmd.py Telegram command handler (/status, /positions, etc.)
    ├── python/monitoring/dashboard.py    Dash web UI + 12 JSON API endpoints + /healthz
    ├── python/monitoring/regime.py       VIX/SPY-based regime detector (normal/caution/halt)
    └── python/monitoring/drift.py        KS test + PSI feature drift detection
```

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Timezone | NY (`ZoneInfo("America/New_York")`) for trading, UTC for internal timestamps | Market hours are Eastern |
| Risk-free rate | `RISK_FREE_RATE = 0.05` in `python/data/config.py` | Centralized, used by all Sharpe calcs |
| Embargo | 22 business days | Matches longest feature lookback window |
| OCO brackets | 2x ATR SL / 3x ATR TP | Adapts to per-stock volatility |
| Training universe | Full ~500 S&P 500 tickers | No sampling |
| IC quality gate | `MIN_VALIDATION_IC = 0.02` | Falls back to equal-weight if model is weak |
| Covariance | Ledoit-Wolf shrinkage (unconditional) | Stable with 10 stocks |
| `get_ml_weights()` | Returns `(dict, bool)` tuple — `(weights, stale_data)` | All callers must destructure |

## VPS Details

| Item | Value |
|------|-------|
| IP | `209.38.122.78` |
| Spec | 2 vCPU, 4 GB RAM, 80 GB disk, Ubuntu 22.04 |
| SSH | `ssh -i deploy/signum_ed25519 root@209.38.122.78` |
| Services | `signum-bot.service` + `signum-dashboard.service` |
| Dashboard | `https://dashboard.bhanueso.dev` (nginx + Let's Encrypt) |
| Logs | `/var/log/signum/bot.log`, `/var/log/signum/bot-error.log` |
| Project path | `/opt/signum/` |
| uv path | `/root/.local/bin/uv` |

## Deploy Procedure

```bash
# 1. Deploy changed files (use scp, NOT rsync)
scp -i deploy/signum_ed25519 <file> root@209.38.122.78:/opt/signum/<file>

# 2. Restart the bot
ssh -i deploy/signum_ed25519 root@209.38.122.78 'systemctl restart signum-bot'

# 3. Verify it's running
ssh -i deploy/signum_ed25519 root@209.38.122.78 'systemctl status signum-bot --no-pager | head -10'
```

## Alerting

### Telegram Bot (@sigum_paperbot)

**Outbound alerts** (18 events) fire automatically:
- Bot startup, shutdown, trade cycle summary, heartbeat (hourly)
- Stale data, order timeout, partial fill, OCO failure, risk violation
- ML pipeline failure, Alpaca connect failure
- Caution mode, halt mode, drawdown kill switch, liquidation

**Interactive commands** (polling every 3s):
- `/status` `/positions` `/equity` `/regime` `/health` `/trades` `/logs` `/help`

**Transport priority for email:** Resend > SendGrid > SMTP (all use port 443 since DigitalOcean blocks SMTP ports).

### Config (in VPS `.env`)
```
TELEGRAM_BOT_TOKEN=<bot token from @BotFather>
TELEGRAM_CHAT_ID=<your chat ID>
```

## Trading Schedule

- **Rebalance:** Weekly on Wednesdays (configurable: `REBALANCE_DAY=2`)
- **Between rebalances:** Bot sleeps. GTC stop-loss and take-profit orders sit on Alpaca's servers.
- **Regime detection:** Continuous — VIX and SPY drawdown checked each cycle

| Regime | Condition | Action |
|--------|-----------|--------|
| Normal | VIX < 25, SPY DD < 8% | Full exposure |
| Caution | VIX 25-35 or SPY DD 8-15% | 50% exposure |
| Halt | VIX > 35 and SPY DD > 15% | Liquidate, wait |

## Risk Limits

| Check | Limit | Blocks trade? |
|-------|-------|--------------|
| Max position weight | 30% | Yes |
| Max sector weight | 25% | Yes |
| Max single trade size | 15% | Yes |
| Max leverage | 1.0x | Yes |
| Max drawdown | 15% | Kill switch — liquidates all |
| Max daily trades | 50 | Warning only |
| Max daily turnover | 100% | Warning only |

## Running Tests

```bash
# Full suite (should pass 589 tests in ~77s)
uv run python -m pytest tests/ -x -q --tb=short

# Specific modules
uv run python -m pytest tests/monitoring/ -x -q            # Alerting + Telegram + dashboard
uv run python -m pytest tests/alpha/ -x -q                 # ML pipeline
uv run python -m pytest tests/backtest/ -x -q              # Backtesting
uv run python -m pytest tests/portfolio/ -x -q             # Optimization + risk
uv run python -m pytest tests/bridge/ tests/brokers/ -x -q # Execution + brokers
uv run python -m pytest tests/test_live_bot_helpers.py tests/test_live_integration.py -x -q  # Live bot
```

## File Layout

```
signum/
├── AGENTS.md                      # This file
├── README.md                      # Project overview
├── .env.example                   # Environment variable template
├── pyproject.toml                 # Python 3.11, all dependencies
├── examples/
│   ├── live_bot.py                # Main trading bot (entry point)
│   ├── dry_run.py                 # ML pipeline test without orders
│   └── paper_trading_tracker.py   # CLI portfolio snapshot
├── python/
│   ├── alpha/                     # ML pipeline (features, model, ensemble, predict, train)
│   ├── portfolio/                 # Optimization (HRP, CVaR, BL) + risk engine + risk manager
│   ├── bridge/                    # Execution bridge (order submission, position tracking)
│   ├── brokers/                   # Broker interface (base + Alpaca implementation)
│   ├── data/                      # Data ingestion, config, sectors
│   ├── backtest/                  # Walk-forward backtest, purged CV, robustness, regime analysis
│   └── monitoring/                # Alerting, Telegram commands, dashboard, drift, regime
├── tests/                         # 589 tests mirroring python/ structure
├── deploy/
│   ├── signum_ed25519             # SSH private key (NOT in git)
│   ├── signum-bot.service         # systemd unit (trading bot)
│   └── signum-dashboard.service   # systemd unit (web dashboard)
├── docs/
│   ├── PAPER_TRADING_READINESS.md # Audit results + readiness verdict
│   ├── AUDIT_REPORT.md            # Original 45-finding audit
│   └── IMPROVEMENT_PLAN.md        # Improvement roadmap (all phases implemented)
└── rust/matching-engine/          # Lock-free order book (research, not used in live)
```

## Known Limitations (documented, not fixing)

- **Survivorship bias:** Uses current S&P 500 list for historical data
- **Ensemble not wired:** `ensemble.py` exists but live uses single LightGBM directly
- **Drift detection not in live loop:** `DriftDetector` exists but isn't called automatically
- **`alpaca-trade-api` deprecated:** Works fine but Alpaca recommends migration to `alpaca-py`
- **`rolling_beta` crash:** Only called from dashboard, never from live bot
- **VaR sign inconsistency:** `abs()` workaround in risk manager is safe

## Audit History

Three audit rounds (113+ findings resolved) + post-audit hardening:

| Round | Findings | Focus |
|-------|----------|-------|
| 1 | 45 | Full codebase review (14 P0, 24 P1, 7 P2) |
| 2 | 56 | Parallel audit by 6 agents (execution + ML pipeline) |
| 3 | 37 | Final pre-paper-trading hardening |
| Post | — | yfinance circuit breaker, alerting module, Telegram commands, /healthz, structured logging |

See `docs/AUDIT_REPORT.md` and `docs/PAPER_TRADING_READINESS.md` for details.
