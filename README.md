# Kalshi High-Probability Auto-Trader

A Python bot that scans Kalshi's open markets, picks Yes contracts priced
between 95¢ and 99¢, and buys them cheapest-first until the account can no
longer afford a single contract. Then it sleeps 4 hours and looks again.

> ### ⚠️ Read this before you flip live
> This strategy has **negative expected value after fees**. At 95–99¢ the
> Yes side is already near-certain; the spread plus Kalshi's fee makes the
> long-run EV negative even when you "win" most of the time. The bot is
> built correctly and safely, but the math of the strategy itself is not.
> The default config ships in dry-run mode (`LIVE_TRADING=False`) against
> Kalshi's demo environment for a reason. Keep it there until you've
> watched a few passes and you understand the trade-offs.

---

## What the bot does, in one pass

1. Read live cash balance from Kalshi.
2. Page through every open market.
3. Keep only markets whose best Yes ask is in `[95, 99]¢`, are still open,
   have enough resting size, and that you don't already hold.
4. Sort survivors cheapest-first (95 before 96 before … before 99).
   Ties broken by liquidity, then soonest close time, then ticker.
5. Walk the list, sizing each order with fees included:
   - `FIXED_DOLLAR` mode: spend up to `FIXED_TRADE_SIZE_USD` per market.
   - `ALL_IN_PER_MARKET` mode: dump the whole balance into the top market.
6. Submit a limit Yes buy at the current ask (never a naked market order).
7. Reconcile with `/portfolio/orders/{id}` + `/portfolio/fills` — no fill
   is assumed.
8. Stop when the working budget can't afford one contract at 95¢ + fee.
9. Sleep 4 hours, then start over.

---

## Repository layout

```
.
├── main.py                 entry point — boots, validates, runs forever
├── config.py               all configuration; reads .env, validates inputs
├── trader.py               run_pass() and run_forever() — the loop
├── executor.py             dry-run + live order placement, reconciliation
├── logging_setup.py        console + rotating-file logging
├── kalshi/
│   ├── __init__.py
│   ├── auth.py             RSA-PSS request signing (KALSHI-ACCESS-* headers)
│   └── client.py           httpx-based REST client + retry/backoff
├── strategy/
│   ├── fees.py             integer-cent fee math
│   ├── filter.py           keep only 95-99¢ Yes asks, open, not held, liquid
│   ├── ranker.py           cheapest first, deterministic tie-breaks
│   └── sizer.py            contract-count math for both sizing modes
├── tests/                  pytest suite (48 unit tests + 3 demo integration)
├── requirements.txt
├── .env.example
├── .gitignore              excludes .env, *.pem, *.key, logs
├── pytest.ini
└── README.md
```

---

## Getting Kalshi API keys

1. Sign up at https://kalshi.com (or use the demo URL listed below) and
   complete identity verification on the account you'll trade with.
2. Go to **Profile → API Keys → Create new key**.
3. Kalshi shows you a **Key ID** (UUID-ish) and downloads an **RSA private
   key** as a `.pem` file. **Save the PEM file outside the repo** (or
   inside, but make sure `.gitignore` is in effect — it is by default).
4. For demo, go to https://demo.kalshi.co, create a separate account, and
   create a separate API key there. Demo keys do not work in production
   and vice versa.

---

## Installation

```bash
git clone <your-repo-url>
cd Makememoneykalshi
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
$EDITOR .env                       # fill in real values
```

`.env`:

```
KALSHI_API_KEY_ID=<the UUID Kalshi gave you>
KALSHI_PRIVATE_KEY_PATH=/absolute/path/to/your_demo_private_key.pem
ENVIRONMENT=demo
```

Optional overrides (defaults shown):

```
LIVE_TRADING=false                 # set true to actually submit orders
SIZING_MODE=FIXED_DOLLAR           # or ALL_IN_PER_MARKET
FIXED_TRADE_SIZE_USD=1.00
RECHECK_INTERVAL_HOURS=4
MIN_LIQUIDITY_CONTRACTS=1
LOG_FILE=trader.log
LOG_LEVEL=INFO
```

---

## Running

### Dry run (default)

```bash
python main.py
```

You'll see a boot banner, then per-pass logs of: balance, filtered
candidates, considered orders, dry-run intent, and the 4-hour sleep.
Nothing is submitted to Kalshi.

### Go live (after testing!)

Open `.env` and change exactly one line:

```
LIVE_TRADING=true
```

Save, then `python main.py` again. The bot now submits limit Yes buys at
the current ask. Keep `ENVIRONMENT=demo` until you have watched several
live passes in demo and are happy with the behavior.

### Switch sizing mode

In `.env`, change exactly one line:

```
SIZING_MODE=ALL_IN_PER_MARKET
```

The bot now places one big order into the single top-ranked market each
pass instead of $1 slices across many markets.

---

## Tests

```bash
pip install pytest
pytest                                 # 48 unit tests, no network
```

Demo integration tests are gated on real credentials and `ENVIRONMENT=demo`:

```bash
ENVIRONMENT=demo pytest -m demo -v
```

The "place a tiny test order" step is *further* gated on
`KALSHI_INTEGRATION_PLACE_ORDER=1` so that a routine `pytest` run never
spends demo cash unless you explicitly opt in:

```bash
ENVIRONMENT=demo KALSHI_INTEGRATION_PLACE_ORDER=1 pytest -m demo -v
```

Production tests are not provided. Do not point integration tests at
production.

---

## API details verified against Kalshi's current docs (May 2026)

| What                  | Value                                                                  |
| --------------------- | ---------------------------------------------------------------------- |
| Demo base URL         | `https://demo-api.kalshi.co/trade-api/v2`                              |
| Production base URL   | `https://api.elections.kalshi.com/trade-api/v2`                        |
| Auth                  | API key ID + RSA private key, RSA-PSS / SHA-256, MGF1, salt 32 bytes   |
| Auth headers          | `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE` |
| Signed string         | `f"{timestamp_ms}{METHOD}{path}"` — path includes the `/trade-api/v2` prefix, query string excluded |
| Prices                | integer cents, 1..99; tick = 1¢ for most markets                       |
| Min order size        | 1 contract                                                             |
| Taker fee formula     | `fee = ceil(0.07 × contracts × price × (1 − price))` rounded up to the cent. Computed in integer cents in `strategy/fees.py` |

Override the URLs without a code change by setting `KALSHI_DEMO_BASE_URL`
and/or `KALSHI_PROD_BASE_URL` in `.env` — Kalshi has rotated hosts before
(`demo-api.kalshi.co`, `external-api.demo.kalshi.co`,
`trading-api.kalshi.com`, `api.elections.kalshi.com`) and will likely do
so again. If the docs list a different host than the defaults above, set
the env var rather than editing `config.py`.

### Discrepancies vs. the reference values in the spec

- **Demo base URL** — the spec listed `https://demo-api.kalshi.co/trade-api/v2`
  and that is the default we kept. Kalshi's current docs (May 2026) also
  show `https://external-api.demo.kalshi.co/trade-api/v2`; both have been
  served at various points. If one returns 404/host-unreachable, set
  `KALSHI_DEMO_BASE_URL` in `.env` to the other — no code change needed.
- **Production base URL** — the spec listed
  `https://api.elections.kalshi.com/trade-api/v2` (kept as default) and
  the older `https://trading-api.kalshi.com/trade-api/v2`. The "elections"
  hostname is the current canonical one; the trading-api hostname is the
  legacy alias and still resolves at the time of writing. Override via
  `KALSHI_PROD_BASE_URL` if that changes.
- **Authentication** — the spec said "if Kalshi's current auth requires
  API key ID + RSA signing". It does. The bot signs every request with
  RSA-PSS / SHA-256 / MGF1 / salt = 32 bytes and sends the three
  `KALSHI-ACCESS-*` headers. Bearer-token auth is not used.
- **Fee formula** — the spec's
  `ceil(0.07 × contracts × price × (1 − price))` is correct in shape.
  Kalshi's documented precision is technically centicents (1/10000 dollar)
  before rounding, but ceiling-to-cent gives the same answer for every
  input in the 95-99¢ band we trade in, so the integer-cent implementation
  in `strategy/fees.py` uses the simpler cent-ceiling.
- **No official Python client** — Kalshi does not publish a maintained
  first-party Python SDK as of May 2026, so we hand-roll the HTTP layer
  with `httpx`. Several community libraries exist; none were stable
  enough to take a dependency on.

---

## Safety design

- All money math is integer cents. Floats only show up at the log line.
- Live balance is read every pass; no cached number is used to size.
- Idempotent client_order_ids (`hp-<uuid>`); each market is bought at
  most once per pass, reconciled against live `/positions` and `/fills`.
- `MarketClosedError` is a distinct exception so a market that flips to
  settled between scan and submit is logged cleanly, not crashed on.
- HTTP retries with exponential backoff + jitter on 429/5xx and
  transport errors, capped at `HTTP_BACKOFF_CAP_SECONDS`.
- If the balance read fails, the pass is skipped entirely (no trading on
  unknown cash).
- `LIVE_TRADING=False` means the executor never calls `place_order` —
  unit-tested in `tests/test_executor.py`.
- `.gitignore` excludes `.env`, `*.pem`, `*.key`, and `logs/` so you
  can't accidentally commit credentials.

---

## Disclaimer

Not financial advice. Prediction-market trading carries real loss
potential, and as noted at the top this particular strategy is
negative-EV in expectation. Use the demo environment, keep your stake
small, and read the bot's logs.
