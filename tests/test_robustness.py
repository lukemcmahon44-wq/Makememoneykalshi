"""
Robustness — what happens at the edges of the HTTP and parsing paths.

Most of these exercise the retry/backoff loop and resilience to weird
upstream responses without ever touching Kalshi.
"""

import json
from typing import Callable

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi.auth import KalshiSigner
from kalshi.client import KalshiAPIError, KalshiClient, MarketClosedError

BASE = "https://demo-api.kalshi.co/trade-api/v2"


def _signer() -> KalshiSigner:
    return KalshiSigner("k", rsa.generate_private_key(public_exponent=65537, key_size=2048))


def _client(handler: Callable[[httpx.Request], httpx.Response],
            *, max_retries=3) -> KalshiClient:
    transport = httpx.MockTransport(handler)
    return KalshiClient(
        base_url=BASE, signer=_signer(),
        timeout=2, max_retries=max_retries, backoff_base=0.0, backoff_cap=0.0,
        http_client=httpx.Client(transport=transport),
    )


def test_retry_recovers_from_transport_error():
    state = {"calls": 0}
    def handler(req):
        state["calls"] += 1
        if state["calls"] < 3:
            raise httpx.ConnectError("temp")
        return httpx.Response(200, json={"balance": 42})
    c = _client(handler)
    assert c.get_balance_cents() == 42
    assert state["calls"] == 3


def test_transport_error_exhausts_retries_and_raises():
    def handler(req):
        raise httpx.ReadTimeout("slow")
    c = _client(handler, max_retries=2)
    with pytest.raises(KalshiAPIError):
        c.get_balance_cents()


def test_retry_after_seconds_is_honoured_and_finite():
    state = {"calls": 0}
    def handler(req):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="back off")
        return httpx.Response(200, json={"balance": 7})
    c = _client(handler)
    assert c.get_balance_cents() == 7
    assert state["calls"] == 2


def test_retry_after_non_numeric_falls_back_to_backoff():
    state = {"calls": 0}
    def handler(req):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
        return httpx.Response(200, json={"balance": 9})
    c = _client(handler)
    assert c.get_balance_cents() == 9


def test_4xx_other_than_429_does_not_retry():
    state = {"calls": 0}
    def handler(req):
        state["calls"] += 1
        return httpx.Response(401, text="bad key")
    c = _client(handler, max_retries=5)
    with pytest.raises(KalshiAPIError) as exc:
        c.get_balance_cents()
    assert exc.value.status_code == 401
    assert state["calls"] == 1  # no retry on auth errors


def test_missing_balance_field_raises():
    def handler(req):
        return httpx.Response(200, json={"something_else": 1})
    c = _client(handler)
    with pytest.raises(KalshiAPIError, match="unexpected balance payload"):
        c.get_balance_cents()


def test_empty_response_body_treated_as_empty_dict():
    def handler(req):
        return httpx.Response(204)  # no content
    c = _client(handler)
    # cancel_order returns the parsed body; an empty 204 should not crash.
    assert c.cancel_order("ord_x") == {}


def test_non_json_body_returns_raw_field():
    def handler(req):
        if req.url.path.endswith("/portfolio/orders/ord_y"):
            return httpx.Response(200, text="plain text not json", headers={"Content-Type": "text/plain"})
        return httpx.Response(404)
    c = _client(handler)
    fetched = c.get_order("ord_y")
    # `_request` returns {"raw": ...} on un-JSON-able 2xx; `get_order` then
    # tries to .get("order", {}) which yields {}.
    assert fetched == {}


def test_market_closed_recognised_in_multiple_phrasings():
    for phrase in (
        '{"error":"market is closed for trading"}',
        '{"error":"this market has settled"}',
        '{"error":"market is not open"}',
        '{"error":"market expired"}',
    ):
        def handler(req, _p=phrase):
            return httpx.Response(409, text=_p)
        c = _client(handler)
        with pytest.raises(MarketClosedError):
            c.place_order(ticker="K", side="yes", count=1,
                          yes_price_cents=95, client_order_id="cid")


def test_pagination_stops_on_empty_cursor():
    pages = [
        {"markets": [{"ticker": "A"}], "cursor": "p2"},
        {"markets": [], "cursor": ""},
    ]
    calls = {"n": 0}
    def handler(req):
        i = calls["n"]
        calls["n"] += 1
        return httpx.Response(200, json=pages[min(i, len(pages) - 1)])
    c = _client(handler)
    out = list(c.iter_markets(limit=1))
    assert [m["ticker"] for m in out] == ["A"]
    assert calls["n"] == 2


def test_pagination_stops_when_batch_is_empty_even_with_cursor():
    # Defensive: if Kalshi ever returns "cursor": "next" with an empty batch
    # we'd loop forever — verify the safety break works.
    def handler(req):
        return httpx.Response(200, json={"markets": [], "cursor": "stuck"})
    c = _client(handler)
    out = list(c.iter_markets(limit=1))
    assert out == []


def test_place_order_validates_price_band():
    def handler(req):
        return httpx.Response(200, json={"order": {}})
    c = _client(handler)
    with pytest.raises(ValueError):
        c.place_order(ticker="K", side="yes", count=1,
                      yes_price_cents=100, client_order_id="cid")
    with pytest.raises(ValueError):
        c.place_order(ticker="K", side="yes", count=1,
                      yes_price_cents=0, client_order_id="cid")
    with pytest.raises(ValueError):
        c.place_order(ticker="K", side="yes", count=0,
                      yes_price_cents=95, client_order_id="cid")
    with pytest.raises(ValueError):
        c.place_order(ticker="K", side="maybe", count=1,
                      yes_price_cents=95, client_order_id="cid")


def test_no_side_sends_no_price_for_no_orders():
    captured = {}
    def handler(req):
        captured["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"order": {"order_id": "o"}})
    c = _client(handler)
    c.place_order(ticker="K", side="no", count=2,
                  yes_price_cents=30, client_order_id="cid")
    assert captured["body"]["side"] == "no"
    assert captured["body"]["no_price"] == 70
    assert "yes_price" not in captured["body"]


def test_cancel_order_uses_delete_path():
    state = {}
    def handler(req):
        state["method"] = req.method
        state["path"] = req.url.path
        return httpx.Response(200, json={"order": {"order_id": "ord_z", "status": "canceled"}})
    c = _client(handler)
    c.cancel_order("ord_z")
    assert state["method"] == "DELETE"
    assert state["path"].endswith("/portfolio/orders/ord_z")


def test_get_orderbook_returns_payload():
    def handler(req):
        return httpx.Response(200, json={
            "orderbook": {"yes": [[95, 10], [94, 5]], "no": [[3, 20]]},
        })
    c = _client(handler)
    book = c.get_orderbook("KX")
    assert book["yes"][0] == [95, 10]


def test_large_pagination_terminates(monkeypatch):
    # 50 pages of 200 markets — make sure we don't accidentally break the
    # cursor loop on a long scan.
    pages_left = {"n": 50}
    def handler(req):
        n = pages_left["n"]
        pages_left["n"] -= 1
        return httpx.Response(200, json={
            "markets": [{"ticker": f"T{n}-{i}"} for i in range(200)],
            "cursor": "next" if n > 1 else "",
        })
    c = _client(handler)
    total = sum(1 for _ in c.iter_markets(limit=200))
    assert total == 50 * 200
