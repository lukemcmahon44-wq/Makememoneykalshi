"""
test_client.py — Locks the KalshiClient REST contract with a fake HTTP session.

These never hit the network. A real RSA key is used so the signing code actually
runs, but the client's session is swapped for a FakeSession that records the
outgoing request and returns canned responses. This pins down the exact order
payload, the auth headers, the query-string-free signing path, response parsing,
pagination, and error handling.

Run:  python -m pytest -v
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi_client import KalshiAPIError, KalshiClient


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text
        self.headers = headers or {}
        self.ok = 200 <= status_code < 400
        self.content = b"{}" if json_data is not None else b""

    def json(self):
        return self._json


class FakeSession:
    """Returns a single response, or pops through a list (the last one sticks)."""

    def __init__(self, responses):
        self.calls: list[dict] = []
        self._responses = responses

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append(dict(method=method, url=url, params=params, json=json, headers=headers))
        if isinstance(self._responses, list):
            return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return self._responses


@pytest.fixture(scope="module")
def key_path(tmp_path_factory):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path_factory.mktemp("keys") / "key.pem"
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return str(path)


def make_client(key_path, responses):
    client = KalshiClient(
        api_key_id="test-id",
        private_key_path=key_path,
        host="https://external-api.demo.kalshi.co",
        api_prefix="/trade-api/v2",
        max_retries=2,
        backoff_base=0.0,
    )
    client._session = FakeSession(responses)
    return client


# ── Order payload + auth headers ────────────────────────────────────────────--
def test_place_order_payload_and_headers(key_path):
    resp = FakeResponse(200, {"order": {"order_id": "abc", "status": "executed", "fill_count_fp": "5.00"}})
    client = make_client(key_path, resp)
    out = client.place_order(
        ticker="MKT-X",
        side="yes",
        action="buy",
        count=Decimal("5"),
        yes_price=Decimal("0.97"),
        client_order_id="cid-1",
    )
    assert out["order_id"] == "abc"

    call = client._session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://external-api.demo.kalshi.co/trade-api/v2/portfolio/orders"
    assert call["json"] == {
        "ticker": "MKT-X",
        "action": "buy",
        "side": "yes",
        "type": "limit",
        "count_fp": "5.00",
        "yes_price_dollars": "0.9700",
        "time_in_force": "fill_or_kill",
        "client_order_id": "cid-1",
    }
    h = call["headers"]
    assert h["KALSHI-ACCESS-KEY"] == "test-id"
    assert h.get("KALSHI-ACCESS-SIGNATURE")  # present and non-empty
    assert h["KALSHI-ACCESS-TIMESTAMP"].isdigit() and len(h["KALSHI-ACCESS-TIMESTAMP"]) == 13


def test_signed_path_excludes_query_string(key_path):
    client = make_client(key_path, FakeResponse(200, {"markets": [], "cursor": None}))
    client.get_markets(status="open", limit=10)
    call = client._session.calls[0]
    # The signed URL path carries no query string; query params are passed
    # separately (so they are never part of the signature).
    assert "?" not in call["url"]
    assert call["params"]["status"] == "open" and call["params"]["limit"] == 10


# ── Balance parsing ──────────────────────────────────────────────────────────-
def test_get_balance_prefers_dollars(key_path):
    client = make_client(key_path, FakeResponse(200, {"balance_dollars": "12.3456", "balance": 1}))
    assert client.get_balance() == Decimal("12.3456")


def test_get_balance_falls_back_to_cents(key_path):
    client = make_client(key_path, FakeResponse(200, {"balance": 1234}))
    assert client.get_balance() == Decimal("12.34")


# ── Pagination ────────────────────────────────────────────────────────────────
def test_get_markets_paginates(key_path):
    pages = [
        FakeResponse(200, {"markets": [{"ticker": "A"}], "cursor": "c1"}),
        FakeResponse(200, {"markets": [{"ticker": "B"}], "cursor": None}),
    ]
    client = make_client(key_path, pages)
    markets = client.get_markets()
    assert [m["ticker"] for m in markets] == ["A", "B"]
    assert len(client._session.calls) == 2
    assert client._session.calls[1]["params"]["cursor"] == "c1"


# ── Order-book ask derivation ────────────────────────────────────────────────-
def test_derive_yes_ask_from_no_bids(key_path):
    ob = {"orderbook_fp": {"no_dollars": [["0.02", "5.00"], ["0.03", "9.00"]]}}
    client = make_client(key_path, FakeResponse(200, ob))
    assert client.derive_yes_ask("MKT") == Decimal("0.97")  # 1 - best no bid (0.03)


# ── Error handling ────────────────────────────────────────────────────────────
def test_auth_error_is_not_retried(key_path):
    client = make_client(key_path, FakeResponse(401, text="nope"))
    with pytest.raises(KalshiAPIError):
        client.get_balance()
    assert len(client._session.calls) == 1  # 401 must not be retried


def test_bad_request_raises(key_path):
    client = make_client(key_path, FakeResponse(400, text="bad order"))
    with pytest.raises(KalshiAPIError):
        client.place_order(
            ticker="X", side="yes", action="buy",
            count=Decimal("1"), yes_price=Decimal("0.97"), client_order_id="c",
        )


def test_429_then_success_is_retried(key_path):
    responses = [
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(200, {"balance_dollars": "5.00"}),
    ]
    client = make_client(key_path, responses)
    assert client.get_balance() == Decimal("5.00")
    assert len(client._session.calls) == 2
