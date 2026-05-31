"""
Verify the bot never leaks the private key, signature, or PEM body into
logs. The auth signer's public surface is the headers dict; we should
treat any other path as a regression.
"""

import io
import logging
import re
from unittest.mock import MagicMock

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from executor import Executor
from kalshi.auth import KalshiSigner
from kalshi.client import KalshiClient
from strategy.sizer import SizeDecision
from trader import run_pass


def _signer_with_real_pem() -> tuple[KalshiSigner, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    return KalshiSigner("api-key-id-XYZ", key), pem


def _capture_root_logs() -> tuple[logging.Handler, io.StringIO]:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.DEBUG)
    return h, buf


def _remove_handler(h: logging.Handler) -> None:
    logging.getLogger().removeHandler(h)


def test_dry_run_does_not_log_signature_or_pem_body():
    signer, pem = _signer_with_real_pem()
    # The signer also exposes its api_key_id, which IS logged (intentionally —
    # not a secret on its own without the matching private key).
    h, buf = _capture_root_logs()
    try:
        client = MagicMock()
        ex = Executor(client, live=False)
        decision = SizeDecision(
            contracts=1, price_cents=95, cost_cents=95, fee_cents=1,
            total_outlay_cents=96, reason="test",
        )
        ex.execute("KX-LOG-TEST", decision)
    finally:
        _remove_handler(h)
    output = buf.getvalue()
    assert "BEGIN RSA PRIVATE KEY" not in output
    assert "BEGIN PRIVATE KEY" not in output
    # No raw PEM body lines.
    for line in pem.splitlines():
        if line.startswith("-----"):
            continue
        # the inner b64 body is high-entropy strings; verify none leak.
        if len(line) > 20:
            assert line not in output


def test_live_path_does_not_log_signature_header(monkeypatch):
    signer, _pem = _signer_with_real_pem()
    captured_sigs: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_sigs.append(req.headers.get("KALSHI-ACCESS-SIGNATURE", ""))
        path = req.url.path
        if path.endswith("/portfolio/balance"):
            return httpx.Response(200, json={"balance": 200})
        if path.endswith("/portfolio/positions"):
            return httpx.Response(200, json={"market_positions": []})
        if path.endswith("/markets"):
            return httpx.Response(200, json={
                "markets": [{
                    "ticker": "KX-L", "status": "open", "yes_ask": 95,
                    "yes_ask_size": 10,
                }],
                "cursor": "",
            })
        if path.endswith("/portfolio/orders") and req.method == "POST":
            return httpx.Response(200, json={"order": {
                "order_id": "ord_log_test", "status": "executed",
                "remaining_count": 0,
            }})
        if "/portfolio/orders/" in path:
            return httpx.Response(200, json={"order": {
                "order_id": "ord_log_test", "status": "executed",
                "remaining_count": 0,
            }})
        if path.endswith("/portfolio/fills"):
            return httpx.Response(200, json={"fills": [
                {"order_id": "ord_log_test", "count": 1, "yes_price": 95, "fees": 1},
            ]})
        return httpx.Response(404)

    client = KalshiClient(
        base_url="https://demo-api.kalshi.co/trade-api/v2",
        signer=signer,
        timeout=2, max_retries=1, backoff_base=0.0, backoff_cap=0.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    h, buf = _capture_root_logs()
    try:
        summary = run_pass(
            client, live=True, sizing_mode="FIXED_DOLLAR",
            fixed_trade_size_usd=1.0, price_band=(95, 99), min_liquidity=1,
        )
    finally:
        _remove_handler(h)
    output = buf.getvalue()
    # Sigs were generated…
    assert any(captured_sigs)
    # …but none of them appear in the log buffer.
    for sig in captured_sigs:
        assert sig not in output, "signature header leaked into logs"
    # Sanity: we did at least log something for the pass.
    assert "BALANCE" in output
    assert summary.orders_submitted == 1


def test_signer_repr_does_not_expose_private_key():
    signer, _pem = _signer_with_real_pem()
    rendered = repr(signer)
    assert "BEGIN" not in rendered
    assert "PRIVATE KEY" not in rendered
