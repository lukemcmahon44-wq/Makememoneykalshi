"""
config.py — All tunable constants and environment loading for the Kalshi
high-probability auto-sniper.

Design notes
------------
* Every value in the money path is a `decimal.Decimal`. There is no `float`
  anywhere a price, size, or dollar amount is computed.
* This module has NO side effects beyond reading environment variables. It does
  NOT validate credentials or touch the network on import, so `strategy.py` and
  the unit tests can import it freely without API keys present.
* Safety defaults are biased toward *not* trading: `DRY_RUN` and `USE_DEMO`
  both default ON, and any ambiguous/garbage env value leaves them ON.
"""

from __future__ import annotations

import os
from decimal import Decimal

from dotenv import load_dotenv

# Load a local .env if present. Never commit real credentials (see .gitignore).
load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Mode flags (safety-critical)
# ──────────────────────────────────────────────────────────────────────────────
def _flag_default_true(name: str) -> bool:
    """Return a boolean env flag that defaults to True.

    To turn the flag OFF you must set it to one of the explicit falsey strings
    below. ANY other value — unset, empty, or unrecognized garbage — leaves the
    flag ON. This is deliberate: for DRY_RUN and USE_DEMO the safe direction is
    "on", so ambiguity must never silently enable live trading.
    """
    raw = os.getenv(name)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


# MUST default True. Logs intended trades and places nothing until explicitly
# set to a falsey value (e.g. DRY_RUN=false).
DRY_RUN: bool = _flag_default_true("DRY_RUN")

# MUST default True. Points at the demo endpoint until explicitly disabled.
USE_DEMO: bool = _flag_default_true("USE_DEMO")

# Optional: cancel any resting orders belonging to us on shutdown. Off by
# default — with fill-or-kill we should never have resting orders anyway.
CANCEL_ON_EXIT: bool = os.getenv("CANCEL_ON_EXIT", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints (verified against docs.kalshi.com, 2026)
# ──────────────────────────────────────────────────────────────────────────────
KALSHI_HOST_PROD = "https://api.elections.kalshi.com"
KALSHI_HOST_DEMO = "https://demo-api.kalshi.co"

# Path prefix that is part of EVERY request path AND part of the RSA-PSS signing
# string. The signature is computed over `timestamp + METHOD + path`, where path
# includes this prefix and the endpoint but NOT the query string.
API_PREFIX = "/trade-api/v2"

# Host may be overridden explicitly (e.g. an alternate prod host). Otherwise it
# is chosen by USE_DEMO.
_HOST = (os.getenv("KALSHI_HOST") or (KALSHI_HOST_DEMO if USE_DEMO else KALSHI_HOST_PROD)).rstrip("/")
HOST: str = _HOST
BASE_URL: str = HOST + API_PREFIX


# ──────────────────────────────────────────────────────────────────────────────
# Credentials (read here, validated lazily by the client / trader)
# ──────────────────────────────────────────────────────────────────────────────
KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "").strip()
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
# Optional passphrase if the PEM private key is encrypted.
KALSHI_PRIVATE_KEY_PASSWORD: str | None = os.getenv("KALSHI_PRIVATE_KEY_PASSWORD") or None


def require_credentials() -> None:
    """Crash loudly with a clear message if credentials are missing.

    Called by the client/trader at startup — NOT on import — so that the pure
    strategy code and unit tests do not need keys.
    """
    missing = []
    if not KALSHI_API_KEY_ID:
        missing.append("KALSHI_API_KEY_ID")
    if not KALSHI_PRIVATE_KEY_PATH:
        missing.append("KALSHI_PRIVATE_KEY_PATH")
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in your environment or a .env file. "
            "See .env.example and the README for how to generate API keys."
        )
    if not os.path.isfile(KALSHI_PRIVATE_KEY_PATH):
        raise RuntimeError(
            f"KALSHI_PRIVATE_KEY_PATH points to a file that does not exist: "
            f"{KALSHI_PRIVATE_KEY_PATH!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Strategy parameters
# ──────────────────────────────────────────────────────────────────────────────
# Inclusive lower bound on the YES ask price (implied probability floor).
MIN_PROBABILITY_PRICE = Decimal("0.95")
# Inclusive upper bound. We NEVER buy at 1.00 — there is no edge and no profit.
MAX_PROBABILITY_PRICE = Decimal("0.99")
# Fraction of TOTAL account balance to target per market (e.g. 0.10 of $10 -> $1).
POSITION_PCT = Decimal("0.10")


# ──────────────────────────────────────────────────────────────────────────────
# Capital safety rails
# ──────────────────────────────────────────────────────────────────────────────
# Never tie up more than this fraction of total balance across all positions
# plus resting buy orders.
MAX_DEPLOYED_PCT = Decimal("0.80")
# Halt all new buying when available balance falls below this floor (dollars).
MIN_BALANCE = Decimal("5.00")
# Hard cap on dollars deployed into any single market in one go.
MAX_POSITION_PER_MARKET = Decimal("5.00")
# Skip an order whose computed notional rounds below this (dollars). A single
# contract in the 0.95–0.99 band is always >= $0.95, so this only ever filters
# out degenerate zero-size results.
MIN_ORDER_DOLLARS = Decimal("0.50")
# Cumulative *realized* loss (dollars) in a single UTC day that triggers a full
# halt on new buying until manual restart.
DAILY_LOSS_LIMIT = Decimal("5.00")


# ──────────────────────────────────────────────────────────────────────────────
# Market filters
# ──────────────────────────────────────────────────────────────────────────────
# Skip markets that close within this many minutes (avoid settlement-edge risk).
MIN_MINUTES_TO_CLOSE = 5
# Require at least this many contracts of liquidity resting at the ask, AND at
# least as many as our intended order size. Lower this if you find it too strict
# for the thin books common in the 95–99¢ band.
MIN_LIQUIDITY_CONTRACTS = Decimal("10")


# ──────────────────────────────────────────────────────────────────────────────
# Operational
# ──────────────────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
# Self-throttle below the Basic-tier limits (20 reads/s, 10 writes/s).
READ_RATE_LIMIT = 18
WRITE_RATE_LIMIT = 8
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 1.5  # seconds; backoff = RETRY_BACKOFF_BASE ** attempt

# HTTP request timeout (seconds).
REQUEST_TIMEOUT = 15

# Optional: if a market lacks `yes_ask_dollars`, derive it from the order book
# (yes_ask = 1 - best_no_bid). OFF by default — a missing ask almost always means
# there is nothing to buy, and deriving for every market would flood the read
# rate limit. When enabled, the number of lookups per cycle is bounded below.
DERIVE_ASK_FROM_ORDERBOOK: bool = os.getenv("DERIVE_ASK_FROM_ORDERBOOK", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MAX_ORDERBOOK_LOOKUPS_PER_CYCLE = int(os.getenv("MAX_ORDERBOOK_LOOKUPS_PER_CYCLE", "50"))

# Logging
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "sniper.log")
# Persistent fill / settlement store (SQLite).
DB_PATH = os.getenv("DB_PATH", "sniper_state.db")
