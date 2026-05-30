"""
HTTP client for Kalshi REST v2.

Endpoints (verified May 2026):
  GET    /markets                        list with status/limit/cursor
  GET    /markets/{ticker}               one market
  GET    /markets/{ticker}/orderbook     order book for one market
  GET    /portfolio/balance              cash balance, in cents
  GET    /portfolio/positions            current positions
  GET    /portfolio/fills                recent fills
  POST   /portfolio/orders               place limit/market order
  GET    /portfolio/orders/{order_id}    fetch one order
  DELETE /portfolio/orders/{order_id}    cancel
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import urlsplit

import httpx

from kalshi.auth import KalshiSigner

logger = logging.getLogger(__name__)


class KalshiAPIError(RuntimeError):
    """Anything that came back from Kalshi as a non-2xx after exhausting retries."""

    def __init__(self, status_code: int, body: str, method: str, path: str):
        super().__init__(f"{method} {path} -> {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body
        self.method = method
        self.path = path


class MarketClosedError(KalshiAPIError):
    """The market closed/settled between our scan and our order."""


class KalshiClient:
    def __init__(
        self,
        base_url: str,
        signer: KalshiSigner,
        *,
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
        http_client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # ── core request loop ────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        sign_path = _full_sign_path(self.base_url, path)
        attempt = 0
        while True:
            attempt += 1
            headers = self.signer.sign(method, sign_path)
            headers["Accept"] = "application/json"
            if json is not None:
                headers["Content-Type"] = "application/json"
            try:
                resp = self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=self.timeout,
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt > self.max_retries:
                    raise KalshiAPIError(0, f"transport: {exc}", method, path) from exc
                self._sleep_backoff(attempt, reason=f"transport {type(exc).__name__}")
                continue

            if 200 <= resp.status_code < 300:
                if not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    return {"raw": resp.text}

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt > self.max_retries:
                    raise KalshiAPIError(resp.status_code, resp.text, method, path)
                self._sleep_backoff(
                    attempt,
                    reason=f"http {resp.status_code}",
                    retry_after=resp.headers.get("Retry-After"),
                )
                continue

            body = resp.text or ""
            if _looks_like_market_closed(resp.status_code, body):
                raise MarketClosedError(resp.status_code, body, method, path)
            raise KalshiAPIError(resp.status_code, body, method, path)

    def _sleep_backoff(
        self,
        attempt: int,
        *,
        reason: str,
        retry_after: Optional[str] = None,
    ) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.backoff_base * (2 ** (attempt - 1))
        else:
            delay = self.backoff_base * (2 ** (attempt - 1))
        delay = min(delay, self.backoff_cap)
        delay += random.uniform(0, delay * 0.1)
        logger.warning(
            "Kalshi retry %d/%d after %.2fs (%s)",
            attempt, self.max_retries, delay, reason,
        )
        time.sleep(delay)

    # ── markets ──────────────────────────────────────────────────────────

    def list_markets(
        self,
        *,
        status: str = "open",
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> dict:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params)

    def iter_markets(self, *, status: str = "open", limit: int = 200) -> Iterator[dict]:
        """Yield every market across paginated responses."""
        cursor: Optional[str] = None
        seen = 0
        while True:
            page = self.list_markets(status=status, limit=limit, cursor=cursor)
            batch = page.get("markets") or []
            for m in batch:
                yield m
            seen += len(batch)
            cursor = page.get("cursor")
            if not cursor or not batch:
                return

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}").get("market", {})

    def get_orderbook(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}/orderbook").get("orderbook", {})

    # ── portfolio ────────────────────────────────────────────────────────

    def get_balance_cents(self) -> int:
        """Return available balance in INTEGER CENTS.

        Kalshi v2 historically returns `{"balance": <cents>}`.
        Some accounts also expose `payout` and `bonus` — we trust `balance`.
        """
        data = self._request("GET", "/portfolio/balance")
        if "balance" not in data:
            raise KalshiAPIError(
                200, f"unexpected balance payload: {data}", "GET", "/portfolio/balance"
            )
        return int(data["balance"])

    def get_positions(self) -> list[dict]:
        data = self._request("GET", "/portfolio/positions")
        return list(data.get("market_positions") or [])

    def get_fills(
        self,
        *,
        ticker: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        data = self._request("GET", "/portfolio/fills", params=params)
        return list(data.get("fills") or [])

    # ── orders ───────────────────────────────────────────────────────────

    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        count: int,
        yes_price_cents: int,
        client_order_id: str,
        action: str = "buy",
        order_type: str = "limit",
    ) -> dict:
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        if count <= 0:
            raise ValueError("count must be positive")
        if not (1 <= yes_price_cents <= 99):
            raise ValueError("yes_price_cents must be 1..99")

        payload: dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "action": action,
            "side": side,
            "type": order_type,
            "count": int(count),
        }
        if side == "yes":
            payload["yes_price"] = int(yes_price_cents)
        else:
            payload["no_price"] = int(100 - yes_price_cents)
        return self._request("POST", "/portfolio/orders", json=payload).get("order", {})

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/portfolio/orders/{order_id}").get("order", {})

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _full_sign_path(base_url: str, path: str) -> str:
    """Kalshi signs the path *including* the /trade-api/v2 prefix."""
    base_parts = urlsplit(base_url)
    base_path = base_parts.path.rstrip("/")
    if path.startswith("/"):
        return f"{base_path}{path}"
    return f"{base_path}/{path}"


def _looks_like_market_closed(status: int, body: str) -> bool:
    if status not in (400, 403, 409, 422):
        return False
    low = body.lower()
    return (
        "market" in low
        and ("closed" in low or "settled" in low or "not open" in low or "expired" in low)
    )
