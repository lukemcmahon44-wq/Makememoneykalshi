"""RSA-PSS request signing — verify headers and that the signature roundtrips."""

import base64
import time

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi.auth import KalshiSigner, _path_for_signing


def _gen_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def test_headers_present_and_signature_verifies():
    key = _gen_key()
    signer = KalshiSigner("api-key-id-123", key)
    ts = int(time.time() * 1000)
    headers = signer.sign("GET", "/trade-api/v2/markets", timestamp_ms=ts)
    assert headers["KALSHI-ACCESS-KEY"] == "api-key-id-123"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == str(ts)
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    message = f"{ts}GET/trade-api/v2/markets".encode("utf-8")
    # Should not raise.
    key.public_key().verify(
        sig,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256.digest_size,
        ),
        hashes.SHA256(),
    )


def test_path_for_signing_strips_query_and_host():
    assert _path_for_signing("/portfolio/orders?cursor=abc") == "/portfolio/orders"
    assert _path_for_signing("https://x.example.com/trade-api/v2/markets?x=1") == "/trade-api/v2/markets"
    assert _path_for_signing("/trade-api/v2/markets") == "/trade-api/v2/markets"


def test_method_is_uppercased_in_signed_message():
    key = _gen_key()
    signer = KalshiSigner("k", key)
    ts = 1700000000000
    upper = signer.sign("post", "/trade-api/v2/portfolio/orders", timestamp_ms=ts)
    explicit = signer.sign("POST", "/trade-api/v2/portfolio/orders", timestamp_ms=ts)
    # Same key + timestamp + path + (effective) method must produce a verifiable
    # signature for the same message. Verify the lowercase-input one was signed
    # as if it were POST.
    sig = base64.b64decode(upper["KALSHI-ACCESS-SIGNATURE"])
    msg = f"{ts}POST/trade-api/v2/portfolio/orders".encode("utf-8")
    key.public_key().verify(
        sig, msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256.digest_size,
        ),
        hashes.SHA256(),
    )
    # And confirm both signatures verify against that same message
    sig2 = base64.b64decode(explicit["KALSHI-ACCESS-SIGNATURE"])
    key.public_key().verify(
        sig2, msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256.digest_size,
        ),
        hashes.SHA256(),
    )
