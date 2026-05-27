"""
kalshi_client.py — Thin wrapper around Kalshi REST API v2.

Auth: API-key header  (KALSHI_API_KEY env var)
Docs: https://trading-api.kalshi.com/trade-api/v2
"""

import logging
import os
import time
import traceback
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://trading-api.kalshi.com/trade-api/v2")
_API_KEY = os.getenv("KALSHI_API_KEY", "")

# Exponential-backoff constants for 429 handling
_BACKOFF_START = 2   # seconds
_BACKOFF_MAX = 60    # seconds


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=0)  # We handle retries manually for full control
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return session


_session = _build_session()


def _request(method: str, path: str, **kwargs) -> Optional[dict]:
    """
    Execute an HTTP request with exponential back-off on 429s.
    Returns parsed JSON on success, None on unrecoverable error.
    """
    url = f"{_BASE_URL}{path}"
    backoff = _BACKOFF_START
    while True:
        try:
            resp = _session.request(method, url, timeout=15, **kwargs)
            if resp.status_code == 429:
                logger.warning("Rate-limited (429). Sleeping %ds before retry.", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            logger.error("HTTP error [%s %s]: %s", method, path, exc)
            return None
        except Exception:
            logger.error("Unexpected error [%s %s]:\n%s", method, path, traceback.format_exc())
            return None


# ── Market queries ────────────────────────────────────────────────────────────

def get_markets(status: str = "open", limit: int = 200, cursor: str = "") -> list:
    """
    Fetch a page of markets. Paginates automatically until all are fetched.
    Returns list of market dicts.
    """
    markets = []
    params: dict = {"status": status, "limit": limit}
    if cursor:
        params["cursor"] = cursor

    while True:
        data = _request("GET", "/markets", params=params)
        if not data:
            break
        batch = data.get("markets", [])
        markets.extend(batch)
        next_cursor = data.get("cursor")
        if not next_cursor or len(batch) < limit:
            break
        params["cursor"] = next_cursor

    logger.debug("Fetched %d markets", len(markets))
    return markets


def get_btc_markets(series_ticker: str) -> list:
    """
    Fetch open markets for a BTC hourly series (e.g. 'KXBTCU').
    Tries the series_ticker query param first; falls back to prefix-filtering
    the full market list if the API doesn't return results that way.
    """
    data = _request("GET", "/markets", params={
        "status": "open",
        "series_ticker": series_ticker,
        "limit": 200,
    })
    if data:
        markets = data.get("markets", [])
        if markets:
            logger.debug("Fetched %d BTC markets via series_ticker=%s", len(markets), series_ticker)
            return markets

    # Fallback: filter full market list by ticker prefix
    all_markets = get_markets(status="open")
    prefix = series_ticker.upper()
    filtered = [
        m for m in all_markets
        if (m.get("ticker") or "").upper().startswith(prefix)
        or (m.get("series_ticker") or "").upper() == prefix
    ]
    logger.debug("Fetched %d BTC markets via prefix filter (series=%s)", len(filtered), series_ticker)
    return filtered


def get_market(ticker: str) -> Optional[dict]:
    """Fetch a single market by ticker."""
    data = _request("GET", f"/markets/{ticker}")
    if data:
        return data.get("market")
    return None


def get_orderbook(ticker: str) -> Optional[dict]:
    """Return the order book for a market."""
    data = _request("GET", f"/markets/{ticker}/orderbook")
    if data:
        return data.get("orderbook")
    return None


# ── Account ───────────────────────────────────────────────────────────────────

def get_balance() -> Optional[float]:
    """
    Return available balance in dollars.
    Kalshi returns balance in cents; we convert to dollars.
    """
    data = _request("GET", "/portfolio/balance")
    if data is None:
        return None
    # Kalshi v2 returns {"balance": <cents>}
    cents = data.get("balance", 0)
    return cents / 100.0


# ── Orders ────────────────────────────────────────────────────────────────────

def place_order(ticker: str, side: str, count: int, price_cents: int) -> Optional[dict]:
    """
    Place a limit order.

    side       : "yes" or "no"
    count      : number of contracts
    price_cents: limit price in cents (1-99)

    Returns order dict on success, None on failure.
    """
    payload = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "type": "limit",
        "count": count,
        "yes_price": price_cents if side == "yes" else (100 - price_cents),
    }
    data = _request("POST", "/portfolio/orders", json=payload)
    if data:
        order = data.get("order")
        logger.info("Order placed: %s %s×%d @ %d¢ → id=%s",
                    ticker, side, count, price_cents,
                    order.get("order_id") if order else "?")
        return order
    return None


def cancel_order(order_id: str) -> bool:
    """Cancel an open order. Returns True on success."""
    data = _request("DELETE", f"/portfolio/orders/{order_id}")
    return data is not None


def get_order(order_id: str) -> Optional[dict]:
    """Fetch a single order by ID."""
    data = _request("GET", f"/portfolio/orders/{order_id}")
    if data:
        return data.get("order")
    return None


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions() -> list:
    """Return all open Kalshi positions from the API."""
    data = _request("GET", "/portfolio/positions")
    if data:
        return data.get("market_positions", [])
    return []


def get_fills() -> list:
    """Return recent fills (executions)."""
    data = _request("GET", "/portfolio/fills")
    if data:
        return data.get("fills", [])
    return []
