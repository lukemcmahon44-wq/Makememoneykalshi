"""KalshiSigner.from_pem_file / from_pem_string accept real PEM bytes."""

import textwrap

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi.auth import KalshiSigner


def _pem_text() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def test_load_from_pem_file(tmp_path):
    p = tmp_path / "k.pem"
    p.write_text(_pem_text())
    signer = KalshiSigner.from_pem_file("aki", str(p))
    headers = signer.sign("GET", "/trade-api/v2/markets")
    assert headers["KALSHI-ACCESS-KEY"] == "aki"
    assert headers["KALSHI-ACCESS-SIGNATURE"]


def test_load_from_pem_string_literal_newlines():
    signer = KalshiSigner.from_pem_string("aki", _pem_text())
    headers = signer.sign("GET", "/trade-api/v2/markets")
    assert headers["KALSHI-ACCESS-SIGNATURE"]


def test_load_from_pem_string_escaped_newlines():
    # Many users paste PEM into a .env var with literal "\n" sequences.
    escaped = _pem_text().replace("\n", "\\n")
    signer = KalshiSigner.from_pem_string("aki", escaped)
    headers = signer.sign("GET", "/trade-api/v2/markets")
    assert headers["KALSHI-ACCESS-SIGNATURE"]


def test_rejects_non_rsa_key(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key = ed25519.Ed25519PrivateKey.generate()
    p = tmp_path / "ed.pem"
    p.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    with pytest.raises(ValueError, match="RSA"):
        KalshiSigner.from_pem_file("aki", str(p))


def test_empty_api_key_id_rejected():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(ValueError):
        KalshiSigner("", key)


def test_garbage_pem_raises_clear_error(tmp_path):
    p = tmp_path / "junk.pem"
    p.write_text("not a real pem")
    with pytest.raises(Exception):
        KalshiSigner.from_pem_file("aki", str(p))
