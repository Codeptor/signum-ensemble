# Signum

[![CI](https://github.com/Codeptor/signum/actions/workflows/ci.yml/badge.svg)](https://github.com/Codeptor/signum/actions/workflows/ci.yml)

End-to-end quantitative equity platform: ML alpha generation, portfolio optimization, and a Rust matching engine.

## Architecture

```
ML Signals ──► Black-Litterman Bridge ──► Portfolio Optimizer ──► Risk Engine ──► Risk Manager
    │                                            │                           │
    ▼                                            ▼                           ▼
LightGBM (Alpha158)                    HRP / CVaR / BL              Position Sizing
TFT (PyTorch)                         Risk Parity                   Real-time Checks
    │
    ▼
Drift Detection (KS + PSI) ──► Dash Dashboard
```

**Rust Matching Engine** — Lock-free order book with price-time priority, 4 order types (Limit, Market, IOC, FOK), sub-microsecond latency.

## Live Backtest Results (S&P 500, 5yr history)

Walk-forward backtest on 503 S&P 500 constituents, LightGBM alpha (23 features incl. cross-sectional ranks), residual return target, top-20 portfolio, 5-day rebalancing, 10 bps transaction costs, VIX-based position scaling, liquidity filter:

| Optimizer | Sharpe (net) | Sharpe (gross) | Ann. Return | Max DD | Avg Turnover |
|-----------|-------------|----------------|-------------|--------|-------------|
| Equal Weight | **1.66** | 1.78 | 24.6% | 50.0% | 36% |
| HRP | **1.28** | 1.47 | 13.9% | 40.8% | 39% |
| Risk Parity | **1.28** | 1.47 | 13.9% | 40.8% | 39% |
| Black-Litterman | **0.99** | 1.20 | 10.5% | 31.8% | 44% |
| Min CVaR | **0.99** | 1.20 | 10.5% | 31.8% | 44% |

Alpha improvement loop achieved **+51% net Sharpe** (1.10 → 1.66) through cross-sectional feature engineering, residual return targeting, model regularization, and turnover dampening. Equal-weight top-20 remains the strongest risk-adjusted strategy, consistent with the "1/N puzzle" (DeMiguel et al., 2009). HRP provides the best risk-managed allocation among optimizers.

<details>
<summary>Improvement breakdown</summary>

| Step | Sharpe | Delta |
|------|--------|-------|
| Baseline (absolute features, raw target) | 1.10 | — |
| + Cross-sectional features + residual target | 1.29 | +17% |
| + Real OHLCV data, liquidity filter, VIX scaling | 1.44 | +31% |
| + Feature pruning (remove overfit macro features) | 1.60 | +46% |
| + Model regularization (min_child_samples=100) | 1.60 | +46% |
| + Turnover dampening (blend_alpha=0.3) | **1.66** | **+51%** |

Key insight: macro features (VIX, term spread) had the highest tree-split importance but near-zero cross-sectional IC — they were causing overfitting, not adding alpha. Cross-sectional volatility rank (`cs_vol_rank_20d`, IC=0.069) was the strongest new predictor.
</details>

## Benchmark Results (Criterion.rs)

| Operation | Median Latency |
|-----------|---------------|
| Limit order insert | **132 ns** |
| Market order match (10 levels) | **84 ns** |
| Market order match (100 levels) | **231 ns** |
| Market order match (1000 levels) | **1.92 µs** |
| Cancel order | **71 ns** |
| Mixed workload (1000 orders) | **249 µs** |

## Project Structure

```
quant-platform/
├── python/
│   ├── alpha/           # ML signal generation (LightGBM, TFT)
│   ├── portfolio/       # HRP, CVaR, Black-Litterman, risk engine + attribution
│   ├── data/            # yfinance ingestion + TimescaleDB storage
│   ├── backtest/        # Walk-forward CPCV + deflated Sharpe ratio
│   ├── monitoring/      # Drift detection + Dash dashboard
│   └── bridge/          # ML predictions → Black-Litterman views
├── rust/
│   └── matching-engine/ # Lock-free order book with Criterion benchmarks
├── infra/
│   └── docker-compose.yml  # TimescaleDB, Redis, MLflow
├── tests/               # 415+ tests
├── dvc.yaml             # Reproducible pipeline DAG
└── Makefile             # Build orchestration
```

## Quick Start

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[all]"

# Infrastructure
docker-compose -f infra/docker-compose.yml up -d

# Run pipeline
make ingest    # Fetch S&P 500 data via yfinance
make train     # Train LightGBM with MLflow tracking
make backtest  # Walk-forward backtest with CPCV
make dashboard # Launch Dash risk dashboard on :8050

# Tests
uv run pytest tests/ -q  # Python (415+ passed)
cargo test               # Rust (10 passed)
cargo bench              # Criterion benchmarks
```

## Paper Trading Setup

```bash
# 1. Install dependencies
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[all]"

# 2. Configure Alpaca credentials
#    Sign up at https://alpaca.markets (free paper trading account)
#    Get keys from: https://app.alpaca.markets/paper/dashboard/overview
cp .env.example .env
# Edit .env: set ALPACA_API_KEY and ALPACA_API_SECRET

# 3. Dry run (full ML pipeline, no orders submitted)
source .env && uv run python examples/dry_run.py

# 4. Start paper trading bot
source .env && ./run_live_bot.sh

# 5. Production deployment (systemd)
# See deploy/VPS_DEPLOYMENT.md for full guide
sudo cp deploy/signum-bot.service /etc/systemd/system/
sudo systemctl enable --now signum-bot
```

The bot rebalances weekly (Wednesdays by default), running the full ML pipeline each cycle:
scrape S&P 500 tickers, fetch 2yr OHLCV from yfinance, compute 23 alpha features,
train LightGBM with Huber loss, rank stocks by predicted 5-day return, select top 10,
optimize weights via HRP, execute orders via Alpaca with ATR-based SL/TP brackets.

See `docs/PAPER_TRADING_READINESS.md` for the full readiness assessment and known limitations.


## Key Components

### ML Alpha Generation
- **LightGBM/CatBoost** cross-sectional model with 23 features (Alpha158 technicals + cross-sectional ranks)
- **Cross-sectional features**: percentile ranks for momentum, volatility, and volume within each date
- **Residual return target**: model predicts stock-specific alpha (market-neutral), not absolute returns
- **Temporal Fusion Transformer** wrapper (requires `pip install 'quant-platform[ml]'`)
- MLflow experiment tracking with IC, Rank IC metrics

### Portfolio Optimization (skfolio)
- **Hierarchical Risk Parity (HRP)** — avoids covariance inversion
- **Minimum CVaR** — tail risk minimization via linear program
- **Black-Litterman with ML views** — ML confidence scores mapped to view uncertainties

### Risk Engine
- **VaR**: Parametric, Historical, and Cornish-Fisher (accounts for skewness & kurtosis)
- **CVaR / Expected Shortfall**: Historical simulation
- **Drawdown Analysis**: Max, average, duration statistics
- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar, and Omega ratios
- **Rolling Metrics**: 63-day rolling Sharpe, VaR, max drawdown, beta
- **Volatility Regime Detection**: Low/normal/high classification
- **Information Ratio**: Alpha per unit tracking error vs benchmark
- **Herfindahl-Hirschman concentration index**

### Risk Attribution
- **Marginal Risk Contribution (MRC)**: Risk added per unit weight
- **Component Risk**: Actual contribution to portfolio volatility
- **Risk Parity Optimization**: True equal risk contribution
- **Diversification Ratio**: Weighted avg vol / portfolio vol
- **Stress Correlation**: Correlation breakdown during drawdowns

### Stress Testing
- **Historical Scenarios**: 2008 crisis, 2020 COVID, 2022 hikes, dot-com bust, flash crash
- **Hypothetical Shocks**: "What if Tech drops 20%?" scenario analysis
- **Monte Carlo Stress**: Elevated volatility simulations (1.5x, 2.5x, 3x)
- **Correlation Breakdown**: How correlations spike during stress periods

### Risk Manager
- **Real-time Risk Checks**: Position size limits, daily trade limits
- **Risk/Reward Validation**: Minimum 2:1 ratio enforcement
- **Portfolio Monitoring**: VaR, drawdown, Sharpe checks
- **Position Sizing**: Kelly criterion, risk-based, volatility-adjusted sizing

### Execution Bridge
- **Order Validation**: Risk manager checks before execution
- **Position Tracking**: Real-time position and P&L monitoring
- **Paper Trading**: Simulate live trading without real capital
- **Portfolio Rebalancing**: Automated target weight reconciliation
- **Equity Curve**: Track performance over time
- **Position Sizing**: Kelly criterion, risk-based, volatility-adjusted sizing

### Rust Matching Engine
- `BTreeMap<Price, VecDeque<Order>>` for bid/ask levels
- Price-time priority matching (LMAX Disruptor pattern)
- Order types: Limit, Market, IOC, FOK
- All operations sub-microsecond at typical book depths

### Backtesting
- Walk-forward cross-validation with embargo period
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Transaction cost model (turnover-based, configurable bps)
- Position limits (max weight constraint)
- Turnover dampening (weight blending, optimized blend_alpha=0.3)
- VIX-based position scaling (reduce exposure in high-volatility regimes)
- Liquidity filter (exclude bottom 20% by dollar volume)
- Multi-strategy comparison

### Live Trading
- **Alpaca Integration**: Paper and live trading via Alpaca Markets API
- **Broker Abstraction**: Generic broker interface for multiple providers
- **Position Synchronization**: Sync local state with broker positions
- **Real-time Monitoring**: Live P&L and risk tracking
- **Paper Trading**: Risk-free testing environment

### Monitoring
- Feature drift detection (KS test + Population Stability Index)
- Dash dashboard: KPI cards, cumulative returns, drawdown, rolling Sharpe, turnover, concentration gauge

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | LightGBM, CatBoost, PyTorch Forecasting (TFT) |
| Portfolio | skfolio, cvxpy |
| Data | yfinance, SQLAlchemy, TimescaleDB, Parquet |
| Risk | scipy, numpy |
| Monitoring | Evidently-style drift (scipy), Dash + Plotly |
| Execution | Rust, Criterion.rs |
| Brokers | Alpaca Markets API |
| MLOps | MLflow, DVC |
| Infra | Docker Compose (TimescaleDB, Redis, MLflow) |

---

## Recent Improvements (February 2026)

### Enhanced Risk Analytics
- **15 new risk metrics** including Sortino, Calmar, Omega, Information Ratio
- **Cornish-Fisher VaR** for fat-tailed distributions
- **Rolling risk metrics** (63-day windows) for dynamic monitoring
- **Volatility regime detection** for adaptive strategies

### Risk Attribution System
- **Marginal Risk Contribution** analysis
- **Component risk breakdown** per asset
- **True Risk Parity** optimization (vs HRP)
- **Diversification ratio** tracking
- **Stress correlation** analysis

### Advanced Stress Testing
- **Historical scenarios**: 2008 crisis, COVID crash, rate hikes
- **Hypothetical shocks**: Custom scenario modeling
- **Monte Carlo stress** with elevated volatility
- **Correlation breakdown** during stress periods

### Risk Manager Integration
- **Real-time risk checks** in backtest loop
- **Position sizing** with Kelly criterion
- **Risk/Reward validation** (2:1 minimum)
- **Automated stop-losses** and limits

### Performance Attribution
- **Brinson-Fachler model**: Allocation, selection, and interaction effects
- **Sector-level decomposition**: Track where alpha comes from
- **Attribution reports**: Formatted analysis of performance drivers

### Execution Bridge
- **Order management**: Submit and track orders with risk validation
- **Position tracking**: Real-time P&L monitoring
- **Paper trading**: Full simulation without real capital
- **Portfolio reconciliation**: Automated rebalancing to target weights

### Live Trading
- **Alpaca Integration**: Paper and live trading support
- **Broker Abstraction**: Generic interface for multiple brokers
- **Position Sync**: Automatic reconciliation with broker
- **Real-time Data**: Live market data and execution

### Audit Round 2 (February 2026)

Two parallel audits (6 agents) identified 56 additional findings across execution and ML pipeline layers. All resolved:

**Execution layer** (28 fixes):
- OCO order construction, partial fill reconciliation, timeout handling
- Risk manager severity levels, weight initialization, sector exposure
- Liquidation safety, renorm-clamp stability, caution mode scaling
- Timezone consistency (NY for rebalance, UTC for order IDs)
- Batch price fetch, stale position close bypass, atomic state write

**ML pipeline** (28 fixes):
- Train/inference winsorization bounds consistency
- Target winsorization removed (Huber loss handles outliers)
- Date-space purged k-fold with calendar-day purge gap
- Spearman rank IC (not Pearson) for cross-sectional signals
- Geometric Sharpe with risk-free rate subtraction (centralized)
- Net-exposure gate (reduces longs when median prediction < 0)
- Ledoit-Wolf covariance shrinkage for HRP
- Stale VIX detection and OR-based halt de-escalation
- Stationary block bootstrap (preserves autocorrelation)

### Test Coverage
- **415+ tests** (up from 20 at project start)
- Integration tests for full pipeline (data -> signal -> portfolio -> risk)
- Live trading path tests (MockBroker -> ExecutionBridge -> trading cycle)
- ML orchestration tests (fetch -> features -> rank -> optimize)
- 8 audit-specific test coverage gaps filled (short positions, OCO top-up, atomic write, liquidation failure, has-traded-today, position flip, no-price skip, renorm-clamp)

**Next Steps**: Merge audit branches to `main`, begin paper trading, migrate from deprecated `alpaca-trade-api` to `alpaca-py`, add drift detection to live loop, implement point-in-time index membership for production backtests.
