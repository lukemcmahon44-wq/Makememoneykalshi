"""
Kalshi RSA-PSS request signing.

Per Kalshi's current docs (verified May 2026):
  - Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
  - Signed string: f"{timestamp_ms}{METHOD}{path}"
  - path is the URL path *including* the /trade-api/v2 prefix, query string excluded.
  - Algorithm: RSA-PSS, SHA-256 hash, MGF1 with SHA-256, salt length = 32 bytes
  - Signature is base64-encoded.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiSigner:
    """Holds a parsed RSA private key and signs requests for it."""

    def __init__(self, api_key_id: str, private_key: rsa.RSAPrivateKey):
        if not api_key_id:
            raise ValueError("api_key_id is required")
        self.api_key_id = api_key_id
        self._key = private_key

    @classmethod
    def from_pem_file(cls, api_key_id: str, pem_path: str) -> "KalshiSigner":
        data = Path(pem_path).read_bytes()
        return cls(api_key_id, _load_private_key(data))

    @classmethod
    def from_pem_string(cls, api_key_id: str, pem_text: str) -> "KalshiSigner":
        text = pem_text.replace("\\n", "\n")
        return cls(api_key_id, _load_private_key(text.encode("utf-8")))

    def sign(self, method: str, path: str, timestamp_ms: Optional[int] = None) -> dict:
        """Return the three KALSHI-ACCESS-* headers for one request.

        `path` may be a full URL or just a path. The signed value is the path
        portion only (query string stripped).
        """
        method = method.upper()
        signed_path = _path_for_signing(path)
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        message = f"{ts}{method}{signed_path}".encode("utf-8")

        signature = self._key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
        }


def _path_for_signing(path: str) -> str:
    """Strip scheme/host/query — Kalshi signs the path component only."""
    parts = urlsplit(path)
    return parts.path or path


def _load_private_key(pem_bytes: bytes) -> rsa.RSAPrivateKey:
    key = serialization.load_pem_private_key(pem_bytes, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("Kalshi requires an RSA private key.")
    return key
