"""
Integration tests against Kalshi's DEMO environment.

These are skipped unless KALSHI_API_KEY_ID + a key source are present in
the environment AND ENVIRONMENT=demo. They MUST NOT be pointed at
production.

To run locally:
    cp .env.example .env       # then fill in real demo credentials
    ENVIRONMENT=demo pytest -m demo -v

The "place a tiny test order" stage is further gated on the
KALSHI_INTEGRATION_PLACE_ORDER=1 environment variable so a routine
`pytest` run never spends demo cash unless you really mean it.
"""

import os
import time
import uuid

import pytest

import config

pytestmark = pytest.mark.demo


_have_creds = bool(
    config.KALSHI_API_KEY_ID
    and (config.KALSHI_PRIVATE_KEY_PATH or config.KALSHI_PRIVATE_KEY)
)

requires_demo = pytest.mark.skipif(
    not _have_creds or config.ENVIRONMENT != "demo",
    reason="Demo credentials not set or ENVIRONMENT != 'demo'",
)


@pytest.fixture(scope="module")
def client():
    from kalshi import KalshiClient
    from kalshi.auth import KalshiSigner
    if config.KALSHI_PRIVATE_KEY_PATH:
        signer = KalshiSigner.from_pem_file(
            config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH,
        )
    else:
        signer = KalshiSigner.from_pem_string(
            config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY,
        )
    c = KalshiClient(base_url=config.base_url(), signer=signer)
    yield c
    c.close()


@requires_demo
def test_demo_authenticated_balance_read(client):
    # Just proves auth works end-to-end.
    cents = client.get_balance_cents()
    assert isinstance(cents, int)
    assert cents >= 0


@requires_demo
def test_demo_markets_paginate(client):
    seen = 0
    for _ in client.iter_markets(status="open", limit=50):
        seen += 1
        if seen >= 100:
            break
    assert seen > 0


@requires_demo
@pytest.mark.skipif(
    os.getenv("KALSHI_INTEGRATION_PLACE_ORDER") != "1",
    reason="Set KALSHI_INTEGRATION_PLACE_ORDER=1 to exercise order placement.",
)
def test_demo_round_trip_order(client):
    # Find any open market with a Yes ask we can post a deep-below-market
    # limit on — guaranteed to rest, not fill, so we can cancel cleanly.
    target = None
    for m in client.iter_markets(status="open", limit=100):
        if m.get("yes_ask") and 5 < int(m["yes_ask"]) < 95:
            target = m
            break
    assert target is not None, "No suitable demo market found"

    cid = f"itest-{uuid.uuid4().hex[:16]}"
    order = client.place_order(
        ticker=target["ticker"],
        side="yes",
        count=1,
        yes_price_cents=1,        # 1¢ — won't fill, will rest
        client_order_id=cid,
    )
    order_id = str(order.get("order_id") or order.get("id"))
    assert order_id
    time.sleep(1.0)
    fetched = client.get_order(order_id)
    assert fetched
    client.cancel_order(order_id)
