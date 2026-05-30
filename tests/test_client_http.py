"""HTTP client retry/backoff and pagination — exercised against httpx MockTransport."""

import json
from typing import Callable

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi.auth import KalshiSigner
from kalshi.client import KalshiAPIError, KalshiClient, MarketClosedError


BASE = "https://demo-api.kalshi.co/trade-api/v2"


def _signer() -> KalshiSigner:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return KalshiSigner("test-key-id", key)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> KalshiClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return KalshiClient(
        base_url=BASE, signer=_signer(),
        timeout=5, max_retries=3, backoff_base=0.0, backoff_cap=0.0,
        http_client=http,
    )


def test_get_balance_parses_cents():
    def handler(req):
        assert req.method == "GET"
        assert req.url.path.endswith("/portfolio/balance")
        assert "KALSHI-ACCESS-KEY" in req.headers
        return httpx.Response(200, json={"balance": 1234})
    c = _client(handler)
    assert c.get_balance_cents() == 1234


def test_iter_markets_paginates():
    pages = [
        {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "next1"},
        {"markets": [{"ticker": "C"}], "cursor": ""},
    ]
    calls = {"n": 0}
    def handler(req):
        i = calls["n"]
        calls["n"] += 1
        return httpx.Response(200, json=pages[i])
    c = _client(handler)
    out = list(c.iter_markets(status="open", limit=2))
    assert [m["ticker"] for m in out] == ["A", "B", "C"]
    assert calls["n"] == 2


def test_retries_on_429_then_succeeds():
    state = {"n": 0}
    def handler(req):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(200, json={"balance": 7})
    c = _client(handler)
    assert c.get_balance_cents() == 7
    assert state["n"] == 2


def test_retries_on_500_then_gives_up():
    def handler(req):
        return httpx.Response(500, text="kaboom")
    c = _client(handler)
    with pytest.raises(KalshiAPIError) as exc:
        c.get_balance_cents()
    assert exc.value.status_code == 500


def test_market_closed_error_is_distinguished():
    def handler(req):
        return httpx.Response(
            400, text='{"error":"market is closed for trading"}',
        )
    c = _client(handler)
    with pytest.raises(MarketClosedError):
        c.place_order(
            ticker="KX", side="yes", count=1, yes_price_cents=95,
            client_order_id="cid",
        )


def test_place_order_sends_correct_body_for_yes():
    captured = {}
    def handler(req):
        captured["method"] = req.method
        captured["path"] = req.url.path
        captured["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"order": {"order_id": "ord_1", "status": "resting"}})
    c = _client(handler)
    order = c.place_order(
        ticker="KX-EXAMPLE", side="yes", count=3, yes_price_cents=97,
        client_order_id="cid-1",
    )
    assert order["order_id"] == "ord_1"
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/portfolio/orders")
    assert captured["body"] == {
        "ticker": "KX-EXAMPLE",
        "client_order_id": "cid-1",
        "action": "buy",
        "side": "yes",
        "type": "limit",
        "count": 3,
        "yes_price": 97,
    }
