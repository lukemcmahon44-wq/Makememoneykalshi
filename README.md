# Kalshi Edge Trading Bot

A production-ready Python bot that identifies and auto-executes trades on [Kalshi](https://kalshi.com) prediction markets using real external data (weather forecasts, sports odds, news sentiment, and polling data) to calculate a statistical edge before placing any order.

---

## How It Works

1. Every 60 seconds, the bot fetches all open Kalshi markets.
2. Each market is passed through four data providers (weather, sports, news, polling). The first matching provider returns a probability estimate (0–100).
3. **Edge = my_probability − Kalshi's YES price**. If edge ≥ 8%, the bot places a limit YES order for 1 contract.
4. Open positions are monitored continuously. Exits happen on: market resolution, stop-loss (−15¢), or edge flip (−10%).
5. All trades are logged to SQLite. Alerts fire via Telegram.

---

## Project Structure

```
.
├── main.py               # Entry point — logging setup, P&L banner, starts bot loop
├── bot.py                # Core scan/trade/exit loop
├── kalshi_client.py      # Kalshi REST API v2 wrapper
├── edge_calculator.py    # Runs providers, computes edge
├── providers/
│   ├── __init__.py       # Exports ALL_PROVIDERS list
│   ├── weather.py        # OpenWeatherMap forecast → probability
│   ├── sports.py         # The Odds API moneyline → implied probability
│   ├── news.py           # NewsAPI + TextBlob sentiment → probability
│   └── polling.py        # Configurable polling JSON endpoint → probability
├── db.py                 # SQLite: positions, trades, P&L summary
├── alerts.py             # Telegram entry/exit/halt notifications
├── pnl_report.py         # Standalone P&L table printer
├── requirements.txt
├── .env.example          # All env vars documented — copy to .env
├── supervisord.conf      # Auto-restart via supervisord
├── railway.toml          # One-click Railway deployment
└── README.md
```

---

## Quick Start (Local)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd Makememoneykalshi
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m textblob.download_corpora   # downloads NLTK data for sentiment
```

### 2. Configure

```bash
cp .env.example .env
nano .env   # fill in your API keys (see table below)
```

### 3. Run

```bash
python main.py
```

On first run you'll see an all-time P&L summary (empty to start), then the scan loop begins.

### 4. View P&L

```bash
python pnl_report.py               # full history
python pnl_report.py --provider news
python pnl_report.py --since 2024-06-01
python pnl_report.py --closed-only
```

---

## API Keys You Need

| Key | Where to get it | Free tier |
|-----|----------------|-----------|
| `KALSHI_API_KEY` | [kalshi.com](https://kalshi.com) → Settings → API | Yes |
| `TELEGRAM_BOT_TOKEN` | Message [@BotFather](https://t.me/BotFather) on Telegram | Yes |
| `TELEGRAM_CHAT_ID` | Message [@userinfobot](https://t.me/userinfobot) on Telegram | Yes |
| `OPENWEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) | Yes (1000 req/day) |
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) | Yes (500 req/month) |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) | Yes (100 req/day) |
| `POLLING_URL` | Your own endpoint or skip (polling provider will be inactive) | N/A |

---

## Environment Variables Reference

See [`.env.example`](.env.example) for the full annotated list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_THRESHOLD` | `8` | Minimum edge % to place a trade |
| `MAX_OPEN_POSITIONS` | `5` | Max simultaneous positions |
| `MAX_POSITION_SIZE` | `2` | Max dollars per trade |
| `MIN_BALANCE_HALT` | `5` | Halt trading if balance drops below this |
| `SCAN_INTERVAL` | `60` | Seconds between market scans |

---

## Deploy to Railway

Railway runs the bot 24/7 in the cloud for ~$5/month.

### Steps

1. Push this repo to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select your repo. Railway auto-detects `railway.toml` and uses `python main.py`.
4. Click **Variables** → **Add all variables from `.env`** (copy each key/value).
5. Click **Deploy**. The bot starts immediately.
6. View logs in Railway's dashboard → **Logs** tab.

> **Important**: Railway's free tier sleeps after inactivity. Use the **Hobby plan ($5/mo)** for always-on execution.

---

## Auto-Restart with supervisord (Linux VPS)

```bash
pip install supervisor
supervisord -c supervisord.conf
supervisorctl -c supervisord.conf status
```

The bot auto-restarts up to 10 times on crash with a 5-second grace period.

---

## Risk Controls Summary

| Control | Value |
|---------|-------|
| Max per trade | $2 (1 contract × capped price) |
| Max open positions | 5 |
| Minimum edge required | 8% |
| Stop-loss | Exit if YES drops ≥ 15¢ below entry |
| Edge-flip exit | Exit if edge turns −10% or worse |
| Low balance halt | Suspend new trades below $5 balance |
| Min open interest | Skip markets with < 100 contracts OI |
| Min time to expiry | Skip markets expiring in < 2 hours |

---

## Telegram Alerts

| Event | Message format |
|-------|---------------|
| New trade | `EDGE TRADE: TICKER \| Edge: +X% \| My prob: X% \| Kalshi: X¢ \| Provider: name` |
| Exit | `EXIT: TICKER \| Entry: X¢ \| Exit: X¢ \| P&L: ±X¢ \| Reason: resolved/stop/edge-flip` |
| Low balance | `HALT: Balance below $5. Trading suspended.` |

---

## Disclaimer

This bot is provided for educational and research purposes. Prediction market trading involves financial risk. Past performance does not guarantee future results. Always paper-trade first by setting `EDGE_THRESHOLD` very high (e.g., `99`) so no real orders are placed while you validate the setup.
