"""
config.py — Single source of truth for trader configuration.

All credentials are loaded from environment variables / a .env file that
is gitignored. Nothing in this file is a secret on its own; the secrets
come in via the environment.

Edit the values below to tune the strategy. Do not edit the credential
constants — those are wired from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Credentials (loaded from .env, never hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "").strip()

# Either point to a PEM file on disk…
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
# …or paste the PEM contents inline (literal "\n" sequences in the env var
# are unescaped to real newlines by the auth loader).
KALSHI_PRIVATE_KEY: str = os.getenv("KALSHI_PRIVATE_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Environment selection
# ─────────────────────────────────────────────────────────────────────────────

# "demo" or "production". MUST stay "demo" for all testing.
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "demo").strip().lower()

# Base URLs verified against current Kalshi docs (May 2026). The demo host
# was historically demo-api.kalshi.co and is now served at
# external-api.demo.kalshi.co; production uses external-api.kalshi.com.
# These can be overridden via env to track future host changes without a
# code change.
BASE_URLS = {
    "demo": os.getenv(
        "KALSHI_DEMO_BASE_URL",
        "https://demo-api.kalshi.co/trade-api/v2",
    ),
    "production": os.getenv(
        "KALSHI_PROD_BASE_URL",
        "https://api.elections.kalshi.com/trade-api/v2",
    ),
}


def base_url() -> str:
    if ENVIRONMENT not in BASE_URLS:
        raise ValueError(
            f"ENVIRONMENT must be 'demo' or 'production', got {ENVIRONMENT!r}"
        )
    return BASE_URLS[ENVIRONMENT]


# ─────────────────────────────────────────────────────────────────────────────
# Safety switches
# ─────────────────────────────────────────────────────────────────────────────

# False  = dry run. The executor logs every intended order but submits none.
# True   = live trading. Orders are actually sent to Kalshi.
# READ THE README BEFORE FLIPPING THIS TO True.
LIVE_TRADING: bool = os.getenv("LIVE_TRADING", "false").strip().lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

PRICE_BAND_MIN_CENTS: int = 95
PRICE_BAND_MAX_CENTS: int = 99

# "ascending" -> consider 95¢ first, then 96, 97, 98, 99
# (No other modes are defined; this constant exists as documentation.)
PRICE_PRIORITY: str = "ascending"


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

# "FIXED_DOLLAR"       — spend at most FIXED_TRADE_SIZE_USD per market
# "ALL_IN_PER_MARKET"  — dump the entire balance into the single top market
SIZING_MODE: str = os.getenv("SIZING_MODE", "FIXED_DOLLAR").strip().upper()

FIXED_TRADE_SIZE_USD: float = float(os.getenv("FIXED_TRADE_SIZE_USD", "1.00"))


# ─────────────────────────────────────────────────────────────────────────────
# Account / loop
# ─────────────────────────────────────────────────────────────────────────────

# Informational only — the real balance is read from the API at every pass.
STARTING_BALANCE_USD: float = float(os.getenv("STARTING_BALANCE_USD", "10.00"))

RECHECK_INTERVAL_HOURS: float = float(os.getenv("RECHECK_INTERVAL_HOURS", "4"))

# Skip markets too thin to fill our intended size.
MIN_LIQUIDITY_CONTRACTS: int = int(os.getenv("MIN_LIQUIDITY_CONTRACTS", "1"))


# ─────────────────────────────────────────────────────────────────────────────
# HTTP behavior
# ─────────────────────────────────────────────────────────────────────────────

HTTP_TIMEOUT_SECONDS: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "5"))
HTTP_BACKOFF_BASE_SECONDS: float = float(os.getenv("HTTP_BACKOFF_BASE_SECONDS", "1.0"))
HTTP_BACKOFF_CAP_SECONDS: float = float(os.getenv("HTTP_BACKOFF_CAP_SECONDS", "60.0"))


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE: str = os.getenv("LOG_FILE", "trader.log")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated view used by main.py for the boot banner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Snapshot:
    environment: str
    base_url: str
    live_trading: bool
    sizing_mode: str
    fixed_trade_size_usd: float
    price_band: tuple[int, int]
    recheck_interval_hours: float
    log_file: str
    log_level: str
    api_key_id_set: bool
    private_key_source: str


def snapshot() -> Snapshot:
    if KALSHI_PRIVATE_KEY_PATH:
        src = f"file: {KALSHI_PRIVATE_KEY_PATH}"
    elif KALSHI_PRIVATE_KEY:
        src = "inline (KALSHI_PRIVATE_KEY env var)"
    else:
        src = "MISSING"
    return Snapshot(
        environment=ENVIRONMENT,
        base_url=base_url(),
        live_trading=LIVE_TRADING,
        sizing_mode=SIZING_MODE,
        fixed_trade_size_usd=FIXED_TRADE_SIZE_USD,
        price_band=(PRICE_BAND_MIN_CENTS, PRICE_BAND_MAX_CENTS),
        recheck_interval_hours=RECHECK_INTERVAL_HOURS,
        log_file=LOG_FILE,
        log_level=LOG_LEVEL,
        api_key_id_set=bool(KALSHI_API_KEY_ID),
        private_key_source=src,
    )


def validate() -> list[str]:
    """Return a list of human-readable problems; empty list = ready to run."""
    problems: list[str] = []
    if ENVIRONMENT not in BASE_URLS:
        problems.append(
            f"ENVIRONMENT must be 'demo' or 'production', got {ENVIRONMENT!r}"
        )
    if not KALSHI_API_KEY_ID:
        problems.append("KALSHI_API_KEY_ID is not set.")
    if not KALSHI_PRIVATE_KEY_PATH and not KALSHI_PRIVATE_KEY:
        problems.append(
            "Neither KALSHI_PRIVATE_KEY_PATH nor KALSHI_PRIVATE_KEY is set."
        )
    if KALSHI_PRIVATE_KEY_PATH and not Path(KALSHI_PRIVATE_KEY_PATH).is_file():
        problems.append(
            f"KALSHI_PRIVATE_KEY_PATH does not point to a file: "
            f"{KALSHI_PRIVATE_KEY_PATH}"
        )
    if SIZING_MODE not in ("FIXED_DOLLAR", "ALL_IN_PER_MARKET"):
        problems.append(
            f"SIZING_MODE must be FIXED_DOLLAR or ALL_IN_PER_MARKET, "
            f"got {SIZING_MODE!r}"
        )
    if FIXED_TRADE_SIZE_USD <= 0:
        problems.append("FIXED_TRADE_SIZE_USD must be > 0")
    if PRICE_BAND_MIN_CENTS < 1 or PRICE_BAND_MAX_CENTS > 99:
        problems.append("Price band must lie inside 1..99 cents.")
    if PRICE_BAND_MIN_CENTS > PRICE_BAND_MAX_CENTS:
        problems.append("PRICE_BAND_MIN_CENTS must be <= PRICE_BAND_MAX_CENTS")
    return problems
