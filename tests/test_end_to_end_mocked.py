"""
End-to-end mock-network test: a full trader pass driven through a real
KalshiClient + RSA signer, with httpx.MockTransport faking Kalshi.

If any of the auth → scan → filter → rank → size → execute → reconcile
wiring is off, this catches it without ever touching Kalshi.
"""

from __future__ import annotations

import base64
import json
from typing import Callable

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi.auth import KalshiSigner
from kalshi.client import KalshiClient
from trader import run_pass

BASE = "https://demo-api.kalshi.co/trade-api/v2"


def _key_and_signer():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, KalshiSigner("test-key-id", key)


def _verify_signature(public_key, headers: dict, method: str, path: str):
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    msg = f"{ts}{method.upper()}{path}".encode("utf-8")
    public_key.verify(
        sig, msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256.digest_size,
        ),
        hashes.SHA256(),
    )


def _build_kalshi_market(ticker, *, ask, size=20, status="open",
                        close_time="2030-01-01T00:00:00Z"):
    return {
        "ticker": ticker,
        "status": status,
        "yes_ask": ask,
        "no_ask": 100 - ask,
        "yes_bid": ask - 1,
        "no_bid": 100 - ask - 1,
        "yes_ask_size": size,
        "close_time": close_time,
        "last_price": ask,
        "open_interest": 1000,
    }


class FakeKalshi:
    """In-memory Kalshi server good enough to drive a full pass."""

    def __init__(self, public_key, *, balance_cents: int, markets: list[dict],
                 positions: list[dict] | None = None):
        self.public_key = public_key
        self.balance_cents = balance_cents
        self.markets = markets
        self.positions = positions or []
        self.orders_placed: list[dict] = []
        self.fills_by_order: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        # Auth check — every call must carry valid signing headers.
        _verify_signature(
            self.public_key,
            {
                "KALSHI-ACCESS-KEY": request.headers["KALSHI-ACCESS-KEY"],
                "KALSHI-ACCESS-TIMESTAMP": request.headers["KALSHI-ACCESS-TIMESTAMP"],
                "KALSHI-ACCESS-SIGNATURE": request.headers["KALSHI-ACCESS-SIGNATURE"],
            },
            method, path,
        )

        if method == "GET" and path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance": self.balance_cents})
        if method == "GET" and path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"market_positions": self.positions})
        if method == "GET" and path.endswith("/markets"):
            # Paginate in two pages to exercise the cursor path.
            cursor = request.url.params.get("cursor")
            if cursor:
                return httpx.Response(200, json={"markets": self.markets[2:], "cursor": ""})
            return httpx.Response(200, json={"markets": self.markets[:2], "cursor": "p2"})
        if method == "POST" and path.endswith("/portfolio/orders"):
            body = json.loads(request.content.decode("utf-8"))
            oid = f"ord-{len(self.orders_placed) + 1}"
            order = {
                "order_id": oid,
                "client_order_id": body["client_order_id"],
                "ticker": body["ticker"],
                "yes_price": body["yes_price"],
                "count": body["count"],
                "status": "executed",
                "remaining_count": 0,
            }
            self.orders_placed.append(order)
            self.fills_by_order[oid] = [
                {"order_id": oid, "count": body["count"],
                 "yes_price": body["yes_price"], "fees": 1},
            ]
            return httpx.Response(200, json={"order": order})
        if method == "GET" and "/portfolio/orders/" in path:
            oid = path.rsplit("/", 1)[-1]
            for o in self.orders_placed:
                if o["order_id"] == oid:
                    return httpx.Response(200, json={"order": o})
            return httpx.Response(404, text="not found")
        if method == "GET" and path.endswith("/portfolio/fills"):
            oid = request.url.params.get("order_id")
            return httpx.Response(200, json={"fills": self.fills_by_order.get(oid, [])})

        return httpx.Response(404, text=f"unhandled {method} {path}")


def _build_client(fake: FakeKalshi, signer: KalshiSigner) -> KalshiClient:
    transport = httpx.MockTransport(fake)
    http = httpx.Client(transport=transport)
    return KalshiClient(
        base_url=BASE, signer=signer,
        timeout=2, max_retries=2, backoff_base=0.0, backoff_cap=0.0,
        http_client=http,
    )


def test_end_to_end_live_pass_buys_cheapest_first_through_real_http_stack():
    key, signer = _key_and_signer()
    fake = FakeKalshi(
        key.public_key(),
        balance_cents=500,                # $5
        markets=[
            _build_kalshi_market("KX-A-PRICEY", ask=99, size=99),
            _build_kalshi_market("KX-B-MID",    ask=97, size=99),
            _build_kalshi_market("KX-C-CHEAP",  ask=95, size=99),
            _build_kalshi_market("KX-D-CHEAPER_BUT_HELD", ask=95, size=99),
            _build_kalshi_market("KX-E-OUT_OF_BAND", ask=50, size=99),
        ],
        positions=[{"ticker": "KX-D-CHEAPER_BUT_HELD", "position": 4}],
    )
    client = _build_client(fake, signer)

    summary = run_pass(
        client,
        live=True,
        sizing_mode="FIXED_DOLLAR",
        fixed_trade_size_usd=1.0,
        price_band=(95, 99),
        min_liquidity=1,
    )

    # 5 markets scanned across two pages, 3 considered (95/97/99 not held),
    # all 3 should fill in dry-run order since each gets a $1 slice.
    assert summary.markets_scanned == 5
    assert summary.markets_considered == 3
    assert summary.orders_submitted == 3
    assert summary.orders_filled == 3
    placed_tickers = [o["ticker"] for o in fake.orders_placed]
    assert placed_tickers == ["KX-C-CHEAP", "KX-B-MID", "KX-A-PRICEY"]
    # Prices we limit-bought at MUST be the live asks.
    prices = [o["yes_price"] for o in fake.orders_placed]
    assert prices == [95, 97, 99]
    # Idempotency: every client_order_id is unique.
    cids = [o["client_order_id"] for o in fake.orders_placed]
    assert len(set(cids)) == len(cids)


def test_end_to_end_dry_run_pass_submits_nothing():
    key, signer = _key_and_signer()
    fake = FakeKalshi(
        key.public_key(),
        balance_cents=500,
        markets=[
            _build_kalshi_market("KX-CHEAP", ask=95, size=10),
            _build_kalshi_market("KX-MID",   ask=97, size=10),
            _build_kalshi_market("KX-HIGH",  ask=99, size=10),
            _build_kalshi_market("KX-X",     ask=50, size=10),
            _build_kalshi_market("KX-Y",     ask=10, size=10),
        ],
    )
    client = _build_client(fake, signer)
    summary = run_pass(
        client, live=False, sizing_mode="FIXED_DOLLAR",
        fixed_trade_size_usd=1.0, price_band=(95, 99), min_liquidity=1,
    )
    assert summary.markets_considered == 3
    assert summary.orders_submitted == 3   # counted as intent
    assert fake.orders_placed == []        # …but the network saw zero POSTs
    # Network calls in dry run: GET balance, GET markets x2, GET positions.
    assert ("POST", "/trade-api/v2/portfolio/orders") not in fake.calls


def test_end_to_end_all_in_lives_one_order_only():
    key, signer = _key_and_signer()
    fake = FakeKalshi(
        key.public_key(),
        balance_cents=10_00,  # $10
        markets=[
            _build_kalshi_market("KX-CHEAP", ask=95, size=1000),
            _build_kalshi_market("KX-MID",   ask=97, size=1000),
            _build_kalshi_market("KX-HIGH",  ask=99, size=1000),
            _build_kalshi_market("KX-EXTRA", ask=95, size=1000),
            _build_kalshi_market("KX-EXTRA2",ask=95, size=1000),
        ],
    )
    client = _build_client(fake, signer)

    summary = run_pass(
        client, live=True, sizing_mode="ALL_IN_PER_MARKET",
        fixed_trade_size_usd=1.0,   # ignored in all-in mode
        price_band=(95, 99), min_liquidity=1,
    )
    assert summary.orders_submitted == 1
    assert len(fake.orders_placed) == 1
    # Buys the cheapest market.
    assert fake.orders_placed[0]["yes_price"] == 95
    # At 95¢ with $10 budget, the integer-cent sizer fits 10 contracts
    # (cost 950, fee 4, outlay 954 <= 1000).
    assert fake.orders_placed[0]["count"] == 10


def test_end_to_end_held_positions_are_skipped_through_real_pipeline():
    key, signer = _key_and_signer()
    fake = FakeKalshi(
        key.public_key(),
        balance_cents=500,
        markets=[
            _build_kalshi_market("KX-A", ask=95, size=10),
            _build_kalshi_market("KX-B", ask=95, size=10),
        ],
        positions=[
            {"ticker": "KX-A", "position": 1},
            {"ticker": "KX-FLAT", "position": 0, "total_traded": 50},
        ],
    )
    client = _build_client(fake, signer)
    summary = run_pass(client, live=True, sizing_mode="FIXED_DOLLAR",
                       fixed_trade_size_usd=1.0, price_band=(95, 99), min_liquidity=1)
    placed = [o["ticker"] for o in fake.orders_placed]
    assert placed == ["KX-B"]      # KX-A held; KX-FLAT not in market list anyway
    assert summary.markets_considered == 1


def test_partial_fill_still_debits_full_intended_outlay():
    """A partial fill must NOT free up the resting portion's reserved cash
    within the same pass — Kalshi already deducts the resting reservation
    from /portfolio/balance, and the next pass will see it."""
    key, signer = _key_and_signer()

    # ALL_IN_PER_MARKET with $10 budget at 95¢ + fee fits 10 contracts
    # (cost 950, fee 4, outlay 954). We'll fake Kalshi filling only 4.
    class PartialFillKalshi(FakeKalshi):
        def __call__(self, request):
            method, path = request.method, request.url.path
            self.calls.append((method, path))
            if method == "POST" and path.endswith("/portfolio/orders"):
                body = json.loads(request.content.decode("utf-8"))
                oid = f"ord-{len(self.orders_placed) + 1}"
                order = {
                    "order_id": oid,
                    "client_order_id": body["client_order_id"],
                    "ticker": body["ticker"],
                    "yes_price": body["yes_price"],
                    "count": body["count"],
                    "status": "resting",
                    "remaining_count": body["count"] - 4,
                }
                self.orders_placed.append(order)
                self.fills_by_order[oid] = [
                    {"order_id": oid, "count": 4,
                     "yes_price": body["yes_price"], "fees": 1},
                ]
                return httpx.Response(200, json={"order": order})
            return super().__call__(request)

    fake = PartialFillKalshi(
        key.public_key(),
        balance_cents=1000,
        markets=[
            _build_kalshi_market("KX-PARTIAL", ask=95, size=10),
            _build_kalshi_market("KX-NEXT",    ask=95, size=10),
        ],
    )
    client = _build_client(fake, signer)
    summary = run_pass(
        client, live=True, sizing_mode="ALL_IN_PER_MARKET",
        fixed_trade_size_usd=1.0, price_band=(95, 99), min_liquidity=1,
    )
    # All-in mode places exactly one order — the partial-fill behaviour just
    # has to leave the trader in a self-consistent state.
    assert summary.orders_submitted == 1
    assert summary.orders_filled == 0          # not fully filled
    # Budget after the (partial) order: 1000 - 954 (intended outlay) = 46¢
    # — well below 95¢, so the pass halts on all-in completion regardless,
    # but the *accounting* must show the resting portion as committed.
    assert summary.ending_balance_cents == 46


def test_two_consecutive_passes_do_not_leak_state():
    """Running run_pass twice in a row must produce independent results;
    nothing from the first pass should bleed into the second's planning."""
    key, signer = _key_and_signer()
    fake = FakeKalshi(
        key.public_key(),
        balance_cents=200,
        markets=[
            _build_kalshi_market("KX-1", ask=95, size=10),
            _build_kalshi_market("KX-2", ask=97, size=10),
        ],
    )
    client = _build_client(fake, signer)

    s1 = run_pass(client, live=True, sizing_mode="FIXED_DOLLAR",
                  fixed_trade_size_usd=1.0, price_band=(95, 99), min_liquidity=1)
    assert s1.orders_submitted >= 1

    # Reset the mock balance and positions to simulate fresh state after the sleep.
    fake.balance_cents = 200
    fake.positions = [{"ticker": "KX-1", "position": 1}]   # now held from pass 1
    # Rewind markets iterator state — FakeKalshi uses page-1/page-2 derived
    # from self.markets so a fresh call works as-is.

    s2 = run_pass(client, live=True, sizing_mode="FIXED_DOLLAR",
                  fixed_trade_size_usd=1.0, price_band=(95, 99), min_liquidity=1)
    # KX-1 should now be skipped (held); KX-2 should be the only candidate.
    placed_in_pass2 = [o["ticker"] for o in fake.orders_placed[len(fake.orders_placed) - s2.orders_submitted:]]
    assert "KX-1" not in placed_in_pass2
