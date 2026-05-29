# Kalshi High-Probability Auto-Sniper

An automated bot that scans every open Kalshi market and buys YES contracts on
markets whose YES ask implies a **95–99% probability** (ask between `$0.95` and
`$0.99` inclusive), sizing each position as a fixed percentage of account
balance, with hard capital-safety rails.

> **It ships in DEMO + DRY-RUN.** On first launch it places **no** real orders —
> it logs the trades it *would* place. You flip to live trading deliberately,
> after you've validated it. Read the whole README before going live.

---

## ⚠️ Read this about the strategy

Buying at 95–99¢ to win 1–5¢ is a *"picking up pennies"* profile. The expected
value is only positive if Kalshi's prices are well-calibrated **and** you survive
variance: a single NO resolution at 97¢ wipes out the gains from ~32 winning 97¢
trades. This bot executes the strategy carefully; it does **not** prove the edge
is real. **Run it in demo for a couple of weeks and reconcile the fill log
against settlements** (see below) to check whether your realized win rate
actually matches the implied probabilities before risking real money.

---

## How it works

Each cycle (default every 60s) the bot:

1. Refreshes **balance** from the API (never sizes from a cached value).
2. Pulls **positions** and **resting orders** to build a dedup set and compute
   already-deployed capital.
3. Stops buying if balance is below the floor, if the **daily realized-loss
   limit** is hit, or if deployed capital is already at the cap.
4. Fetches **all open markets** (paginated).
5. Runs the **pure** `strategy.select_and_size()` to pick + size candidates.
6. Places orders one-by-one (fresh `client_order_id` UUID, `fill_or_kill`),
   stopping when the deployed-capital cap would be breached.
7. Logs a one-line status summary and the structured fill log, then sleeps.

`strategy.py` is pure (no network, no side effects) and fully unit-tested.

## File layout

```
kalshi-sniper/
├── config.py          # All tunable constants + env loading (Decimal money path)
├── kalshi_client.py   # RSA-PSS signed REST wrapper, typed methods
├── strategy.py        # Pure candidate selection + position sizing
├── trader.py          # Main loop: scan -> size -> execute -> log
├── state.py           # SQLite fill log + settlement P&L tracking
├── preflight.py       # One-shot signed auth/connectivity check (no orders)
├── test_strategy.py   # Unit tests for sizing + selection
├── test_trader.py     # Unit tests for the loop's guard rails
├── test_client.py     # Unit tests for the signed REST client contract
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd kalshi-sniper
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
```

### Generating API keys

1. Log into the Kalshi web app → **Settings → API Keys**.
2. Create a key. Kalshi gives you a **Key ID** and you generate/keep an **RSA
   private key** (PEM); the matching **public key** is uploaded to Kalshi.
3. Put the Key ID and the path to your private key in `.env`:

   ```
   KALSHI_API_KEY_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
   ```

   The private key file and `.env` are gitignored. **Never commit them.**
   Demo and production use **separate** keys — generate demo keys in the demo
   web app and use them only against the demo host.

## Running

First, verify your key, signing, and host with a single safe request (places no
orders, exits non-zero on any problem):

```bash
python preflight.py
```

Then run a single cycle to watch one scan, or start the continuous loop:

```bash
python trader.py --once   # one scan/size/place cycle, then exit
python trader.py          # continuous loop
```

The startup banner shows the active mode loudly. In the default DEMO + DRY-RUN
mode you'll see `[DRY RUN] WOULD BUY ...` lines and a `STATUS | ...` summary each
cycle; nothing is sent to the exchange.

### Switching environments and modes

Both flags default **ON** (the safe direction). Set them in `.env`:

| Goal                       | Setting          | Effect                                            |
|----------------------------|------------------|---------------------------------------------------|
| Stay safe (default)        | `DRY_RUN=true`   | Logs intended trades, places nothing.             |
| Place real orders          | `DRY_RUN=false`  | **Submits live orders.**                          |
| Use the sandbox (default)  | `USE_DEMO=true`  | Talks to `external-api.demo.kalshi.co`.           |
| Use production             | `USE_DEMO=false` | Talks to `external-api.kalshi.com`.               |

Recommended progression:

1. **Demo + dry-run** (default) — confirm candidates and sizing look sane.
2. **Demo + live** (`DRY_RUN=false`, `USE_DEMO=true`) — place real *demo* orders,
   then reconcile (below).
3. **Production + live** (`USE_DEMO=false`, `DRY_RUN=false`) — only after the
   demo fill log reconciles cleanly and you accept the risk. Start with a small
   balance, a low `POSITION_PCT`, and a low `MAX_DEPLOYED_PCT` in `config.py`.

## Configuration

All knobs live in `config.py` (with comments). The most important:

| Constant                  | Default     | Meaning                                            |
|---------------------------|-------------|----------------------------------------------------|
| `MIN/MAX_PROBABILITY_PRICE` | 0.95 / 0.99 | YES-ask band (inclusive). 1.00 never traded.     |
| `POSITION_PCT`            | 0.10        | Target position = 10% of balance per market.       |
| `MAX_DEPLOYED_PCT`        | 0.80        | Never tie up more than 80% of balance total.       |
| `MIN_BALANCE`             | 5.00        | Halt buying below this balance (USD).              |
| `MAX_POSITION_PER_MARKET` | 5.00        | Cap USD into any single market.                    |
| `MIN_ORDER_DOLLARS`       | 0.50        | Skip orders that round below this notional.        |
| `DAILY_LOSS_LIMIT`        | 5.00        | Halt all new buying after this realized daily loss.|
| `MIN_MINUTES_TO_CLOSE`    | 5           | Skip markets closing within N minutes.             |
| `MIN_LIQUIDITY_CONTRACTS` | 10          | Require this much resting size at the ask.         |
| `SCAN_INTERVAL_SECONDS`   | 60          | Loop cadence.                                      |

## Safety rails

- **Dry-run default ON** and **demo default ON**; ambiguous env values stay ON.
- **Idempotency** at two layers: a fresh UUID `client_order_id` per order, and an
  in-cycle touched-ticker set that blocks a second order on any market.
- **Deployed-capital cap** enforced order-by-order as a running total.
- **Balance floor** and **per-market cap** bound exposure.
- **Daily loss limit** halts new buying until manual restart.
- **Fill-or-kill** orders — no stale resting bids left at a moved market.
- **Exchange-hours gate**: a cycle is skipped when `trading_active` is false
  (fails open if the status can't be read — order placement rejects anyway).
- **Rate limiting** self-throttles below Basic-tier limits (18 reads/s, 8 writes/s).
- **Kill switch**: SIGINT/SIGTERM shut the loop down cleanly and report any
  resting orders (set `CANCEL_ON_EXIT=true` to also cancel them).
- **Single-instance lock**: a PID lock file (`LOCK_PATH`) prevents a second bot
  from trading the same account at once; a stale lock (dead PID) is reclaimed
  automatically.

## Tests

```bash
cd kalshi-sniper
python -m pytest -v
```

`test_strategy.py` covers the price-band edges (0.94/0.95/0.99/1.00), sizing +
round-down, dedup, time-to-close, liquidity, fractional handling, ordering, and
asserts no `float` ever appears in the order strings. `test_trader.py` covers the
loop's guard rails: the order-by-order deployed-capital cap, in-cycle dedup, the
balance floor and daily-loss halt, and correct deployed-capital accounting when a
fill-or-kill order is *killed* vs. *filled*. `test_client.py` pins the signed REST
contract: the exact order payload, the auth headers (13-digit ms timestamp), the
query-string-free signing path, balance parsing, pagination, and 401/400/429
handling. A GitHub Actions workflow (`.github/workflows/sniper-tests.yml`) runs
the whole suite on every push/PR that touches the bot.

## Reconciliation (do this before going live)

Every order intention and result is written to SQLite (`sniper_state.db`,
gitignored) in the `fills` table, and settlements are stored verbatim in the
`settlements` table. To check whether the edge is real:

```bash
sqlite3 sniper_state.db \
  "SELECT ticker, price_dollars, count, status, filled_count, ts_utc
   FROM fills ORDER BY id DESC LIMIT 50;"

sqlite3 sniper_state.db \
  "SELECT settled_time, ticker, market_result, revenue, cost, fee, pnl
   FROM settlements ORDER BY settled_time DESC;"
```

Compare your realized outcomes to the implied probabilities you paid. If the
realized win rate is below break-even for the price you paid, the edge isn't
there — fix that before risking real capital.

## Notes on the API

Built directly against the Kalshi `trade-api/v2` (no SDK dependency, so you fully
control signing). Auth is RSA-PSS: three headers (`KALSHI-ACCESS-KEY`,
`-SIGNATURE`, `-TIMESTAMP` in **milliseconds**), signing `timestamp + METHOD +
path` (no query string) with salt length = digest length. Prices are 4-decimal
dollar strings (`yes_price_dollars`), sizes are fixed-point strings (`count_fp`),
and the YES ask is read from `yes_ask_dollars` (or derived as `1 − best_no_bid`
from the bids-only order book). All money is handled with `Decimal`. Default
hosts are the docs-recommended `external-api.kalshi.com` (prod) /
`external-api.demo.kalshi.co` (demo); the legacy `api.elections.kalshi.com` /
`demo-api.kalshi.co` hosts also work and can be selected via `KALSHI_HOST`.
