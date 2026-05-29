"""
kalshi_client.py — Authenticated REST wrapper for the Kalshi trade-api v2.

Authentication (verified against docs.kalshi.com, 2026)
-------------------------------------------------------
Every request carries three headers:
    KALSHI-ACCESS-KEY        — the API key ID
    KALSHI-ACCESS-SIGNATURE  — base64( RSA-PSS sign( timestamp + METHOD + path ) )
    KALSHI-ACCESS-TIMESTAMP  — current Unix time in MILLISECONDS (not seconds)

The signature is computed over the exact string `timestamp + METHOD + path`,
where `path` includes the `/trade-api/v2` prefix and the endpoint but NOT the
query string. RSA-PSS uses SHA-256 for both the digest and MGF1, with a salt
length equal to the digest length (`padding.PSS.DIGEST_LENGTH`).

All prices, sizes, and dollar amounts are parsed to `decimal.Decimal`. No
`float` is ever used in the money path.
"""

from __future__ import annotations

import base64
import collections
import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

logger = logging.getLogger(__name__)


class KalshiAPIError(RuntimeError):
    """Raised for non-recoverable API responses (auth failure, bad request,
    or retries exhausted)."""


def to_decimal(value: Any) -> Optional[Decimal]:
    """Best-effort parse of an API value into a Decimal, or None.

    Accepts dollar strings ("0.9700"), fixed-point strings ("10.00"), ints, and
    Decimals. Never raises; returns None on anything unparseable.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


class _RateGate:
    """Simple sliding-window throttle: at most `max_per_sec` calls per rolling
    1-second window. Blocks (sleeps) when the window is full."""

    def __init__(self, max_per_sec: int) -> None:
        self.max = max(1, max_per_sec)
        self._calls: collections.deque[float] = collections.deque()

    def _prune(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= 1.0:
            self._calls.popleft()

    def wait(self) -> None:
        now = time.monotonic()
        self._prune(now)
        if len(self._calls) >= self.max:
            sleep_for = 1.0 - (now - self._calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._prune(time.monotonic())
        self._calls.append(time.monotonic())


class KalshiClient:
    """Thin, typed, signed wrapper around the Kalshi trade-api v2."""

    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        host: str,
        api_prefix: str,
        *,
        private_key_password: Optional[str] = None,
        read_rate_limit: int = 18,
        write_rate_limit: int = 8,
        max_retries: int = 5,
        backoff_base: float = 1.5,
        timeout: int = 15,
    ) -> None:
        if not api_key_id:
            raise ValueError("api_key_id is required")
        self.api_key_id = api_key_id
        self.host = host.rstrip("/")
        self.api_prefix = api_prefix
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self._private_key = self._load_private_key(private_key_path, private_key_password)
        self._session = requests.Session()
        self._read_gate = _RateGate(read_rate_limit)
        self._write_gate = _RateGate(write_rate_limit)

    # ── Auth ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load_private_key(path: str, password: Optional[str]) -> RSAPrivateKey:
        try:
            with open(path, "rb") as fh:
                key = serialization.load_pem_private_key(
                    fh.read(),
                    password=password.encode() if password else None,
                )
        except FileNotFoundError as exc:
            raise KalshiAPIError(f"Private key file not found: {path!r}") from exc
        except Exception as exc:  # malformed PEM, wrong password, etc.
            raise KalshiAPIError(f"Could not load RSA private key from {path!r}: {exc}") from exc
        if not isinstance(key, RSAPrivateKey):
            raise KalshiAPIError("The provided key is not an RSA private key.")
        return key

    def _sign(self, message: str) -> str:
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii")

    def _headers(self, method: str, path: str) -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts_ms + method + path),
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ── Core request with retry/backoff ──────────────────────────────────────
    def _backoff_sleep(self, attempt: int, reason: str) -> None:
        delay = min(self.backoff_base ** attempt, 60.0)
        logger.warning("Retrying after %s (attempt %d) in %.1fs", reason, attempt + 1, delay)
        time.sleep(delay)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        is_write: bool = False,
    ) -> dict:
        method = method.upper()
        # The signing path NEVER includes the query string.
        path = self.api_prefix + endpoint
        url = self.host + path
        gate = self._write_gate if is_write else self._read_gate
        last_err: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            gate.wait()
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=self._headers(method, path),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_err = f"network error: {exc}"
                self._backoff_sleep(attempt, last_err)
                continue

            status = resp.status_code

            # Auth errors: do NOT retry — the signing inputs are almost certainly wrong.
            if status in (401, 403):
                raise KalshiAPIError(
                    f"Authentication failed ({status}) on {method} {path}. Check that: "
                    "(1) KALSHI-ACCESS-TIMESTAMP is Unix MILLISECONDS; "
                    "(2) the signature signs exactly `timestamp+METHOD+path` with NO query string; "
                    "(3) the API key ID matches the public key uploaded to Kalshi for THIS private key; "
                    "(4) demo keys are used only against the demo host (and prod keys against prod). "
                    f"Response body: {resp.text[:300]}"
                )

            if status == 429:
                retry_after = resp.headers.get("Retry-After", "")
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = min(self.backoff_base ** (attempt + 1), 60.0)
                logger.warning("Rate limited (429) on %s %s; backing off %.1fs", method, path, wait)
                time.sleep(wait)
                continue

            if 500 <= status < 600:
                last_err = f"server error {status}: {resp.text[:200]}"
                self._backoff_sleep(attempt, last_err)
                continue

            if not resp.ok:
                # Other 4xx (e.g. 400 bad order) — not retryable.
                raise KalshiAPIError(f"{status} on {method} {path}: {resp.text[:500]}")

            if not resp.content:
                return {}
            try:
                return resp.json()
            except ValueError:
                return {}

        raise KalshiAPIError(
            f"Exhausted {self.max_retries} retries on {method} {path}: {last_err}"
        )

    def _paginate(self, endpoint: str, item_key: str, params: Optional[dict] = None) -> list[dict]:
        """Follow Kalshi's `cursor` pagination until exhausted."""
        params = dict(params or {})
        items: list[dict] = []
        seen_cursors: set[str] = set()
        for _ in range(1000):  # hard safety bound on pages
            data = self._request("GET", endpoint, params=params)
            batch = data.get(item_key) or []
            items.extend(batch)
            cursor = data.get("cursor")
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            params["cursor"] = cursor
        return items

    # ── Account / portfolio ──────────────────────────────────────────────────
    def get_balance(self) -> Decimal:
        """Available balance in DOLLARS as a Decimal.

        Prefers `balance_dollars` (dollar string); falls back to `balance`
        (integer cents) / 100.
        """
        data = self._request("GET", "/portfolio/balance")
        bal = to_decimal(data.get("balance_dollars"))
        if bal is not None:
            return bal
        cents = to_decimal(data.get("balance"))
        return (cents / Decimal(100)) if cents is not None else Decimal("0")

    def get_positions(self) -> list[dict]:
        """All market positions with a non-zero position (paginated)."""
        return self._paginate(
            "/portfolio/positions",
            item_key="market_positions",
            params={"count_filter": "position"},
        )

    def get_resting_orders(self) -> list[dict]:
        """All resting (open) orders (paginated)."""
        return self._paginate(
            "/portfolio/orders",
            item_key="orders",
            params={"status": "resting"},
        )

    def get_settlements(self, min_ts: Optional[int] = None, limit: int = 200) -> list[dict]:
        """Settlements (paginated). Pass `min_ts` (Unix seconds) to bound the
        query — e.g. start of today — so we don't refetch the whole settlement
        history every cycle."""
        params: dict = {"limit": limit}
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        return self._paginate(
            "/portfolio/settlements",
            item_key="settlements",
            params=params,
        )

    # ── Market data ───────────────────────────────────────────────────────────
    def get_markets(self, status: str = "open", limit: int = 200) -> list[dict]:
        """All markets matching `status` (paginated)."""
        return self._paginate(
            "/markets",
            item_key="markets",
            params={"status": status, "limit": limit},
        )

    def get_orderbook(self, ticker: str, depth: int = 1) -> dict:
        data = self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
        return data.get("orderbook") or data.get("orderbook_fp") or {}

    def derive_yes_ask(self, ticker: str) -> Optional[Decimal]:
        """Derive the best YES ask from the order book.

        Kalshi's book contains BIDS ONLY. A NO bid at price Y is equivalent to a
        YES ask at price (1.00 - Y), so the best YES ask = 1.00 - (highest NO bid).
        Handles both the new dollar-string arrays (`no_dollars`) and any legacy
        integer-cent arrays (`no`).
        """
        data = self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": 50})
        ob = data.get("orderbook_fp") or data.get("orderbook") or {}
        no_levels = ob.get("no_dollars") or ob.get("no") or []
        best_no: Optional[Decimal] = None
        for level in no_levels:
            if not level:
                continue
            price = to_decimal(level[0])
            if price is None:
                continue
            # Legacy arrays express price in integer cents (1..99); normalize.
            if price > 1:
                price = price / Decimal(100)
            if best_no is None or price > best_no:
                best_no = price
        if best_no is None:
            return None
        return Decimal(1) - best_no

    # ── Exchange ──────────────────────────────────────────────────────────────
    def get_exchange_status(self) -> dict:
        """Exchange status flags: {exchange_active, trading_active, ...}."""
        return self._request("GET", "/exchange/status")

    # ── Orders ────────────────────────────────────────────────────────────────
    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: Decimal,
        yes_price: Decimal,
        client_order_id: str,
        time_in_force: str = "fill_or_kill",
    ) -> dict:
        """Place a limit order.

        Sends `count_fp` (fixed-point string) and `yes_price_dollars`
        (4-decimal dollar string). We send only `count_fp` (not the legacy
        integer `count`) to avoid the "if both are provided they must match"
        rejection. `time_in_force=fill_or_kill` means the order fills fully and
        immediately or is canceled — we never leave a resting bid at the ask.
        """
        payload = {
            "ticker": ticker,
            "action": action,        # "buy"
            "side": side,            # "yes"
            "type": "limit",
            "count_fp": f"{count:.2f}",
            "yes_price_dollars": f"{yes_price:.4f}",
            "time_in_force": time_in_force,
            "client_order_id": client_order_id,
        }
        data = self._request("POST", "/portfolio/orders", json_body=payload, is_write=True)
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}", is_write=True)
            return True
        except KalshiAPIError as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False
