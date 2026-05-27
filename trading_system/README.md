# Autonomous Quantitative Trading System

An institutional-grade autonomous trading bot for US equities + crypto.
Combines technical, microstructure, statistical-arbitrage, regime,
sentiment, on-chain, and ML signals into a layered risk-managed pipeline.

## Architecture

```
trading_system/
├── core/         config, logger, market clock, kill switch
├── data/         in-memory store + 6 feed clients (Alpaca, Binance, FRED,
│                 CoinMetrics, Reddit, Alpha Vantage)
├── signals/      technical, microstructure (LOB/Hawkes/CVD), stat-arb
│                 (OU/Kalman), regime (GARCH/HMM/macro), sentiment, on-chain
├── ml/           75-feature pipeline, LightGBM, LSTM, TFT, meta-stack
├── risk/         layered Kelly+CVaR sizer, circuit breakers, VaR, stress
├── execution/    Alpaca broker (live + sim), smart router, Avellaneda-Stoikov
├── portfolio/    state manager, Black-Litterman optimizer, analytics
├── backtest/     event-driven engine, walk-forward, Monte Carlo
├── db/           SQLite schema + repository
├── tests/        signals / risk / execution / ML / integration
└── main.py       async orchestrator (eternal main loop)
```

## Immutable Laws

1. Never crash. Every loop catches all exceptions.
2. Never trade stale data. Bars older than 45 seconds are rejected.
3. Never skip the kill switch. `touch /tmp/trading_kill_switch` halts trading.
4. Never trust one signal. Minimum 3 independent signals must agree.
5. Never ignore correlation. Two correlated bets count as one.
6. Never overtrade. 15-minute cooldown between entries per asset.
7. Never fight a circuit breaker. >8% drawdown = cash + 24-hour halt.

## Quick start

### 1. Install dependencies

```bash
cd trading_system
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
$EDITOR .env   # fill in ALPACA_API_KEY, ALPACA_SECRET_KEY (paper),
               # FRED_API_KEY, REDDIT_*, ALPHA_VANTAGE_KEY, TELEGRAM_*
```

### 3. Run unit tests

```bash
cd ..
PYTHONPATH=. pytest trading_system/tests/ -v
```

### 4. Paper trading

```bash
PYTHONPATH=. python -m trading_system.main
```

The bot prints structured logs to stderr and writes rotating files to
`trading_system/logs/`. SQLite DB lives at `trading_system/data/trading.db`.

### 5. Docker

```bash
cd trading_system
docker compose up -d
docker compose logs -f
```

## Mandatory validation protocol

Before flipping `PAPER_MODE=false`, satisfy ALL of these:

- 14 consecutive days of paper trading with no unhandled exceptions
- Sharpe (2-week annualised) > 0.8
- Max drawdown < 8%
- Win rate > 48%
- No Level-2+ circuit breaker triggered
- Slippage within model (actual < 2x estimated)
- ML model OOS AUC > 0.54 and OOS accuracy > 0.52 (gate via
  `ml/validator.py::walk_forward_validate`)

When you do go live, start with 25% of intended capital and re-validate at
the 30-day mark.

## Emergency stop

Two ways:

```bash
# 1. File-based
touch /tmp/trading_kill_switch

# 2. Signal
kill -SIGUSR1 $(pgrep -f trading_system.main)
```

The bot detects the kill switch within 1 second, refuses to enter new
trades, and writes a CRITICAL log line + Telegram alert if configured.

## Build sequence reference

See the build prompt for the complete 8-phase build sequence. The repo
follows that sequence: core → data → signals → ML → risk → execution →
portfolio → backtest → integration.
