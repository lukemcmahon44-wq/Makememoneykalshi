"""
preflight.py — one-shot connectivity + auth check.

Run this BEFORE the trading loop to confirm your API key, RSA signing, and host
are all working. It makes a couple of signed GET requests (exchange status +
balance) and places NO orders.

    python preflight.py

Exits 0 on success; non-zero with a clear message on failure. This is the safe
way to make your very first authenticated call, rather than launching the loop.
"""

from __future__ import annotations

import sys

import config
from kalshi_client import KalshiAPIError
from trader import build_client


def main() -> int:
    print(f"Host : {config.BASE_URL}")
    print(f"Mode : DRY_RUN={config.DRY_RUN}  USE_DEMO={config.USE_DEMO}")

    try:
        config.require_credentials()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 2

    client = build_client()
    try:
        status = client.get_exchange_status()
        print(
            f"Exchange: exchange_active={status.get('exchange_active')} "
            f"trading_active={status.get('trading_active')}"
        )
        balance = client.get_balance()
        print(f"Balance : ${balance:.2f}")
    except KalshiAPIError as exc:
        print("FAIL: an authenticated request was rejected:")
        print(f"  {exc}")
        return 1
    except Exception as exc:  # DNS/TLS/connection errors, etc.
        print(f"FAIL: could not reach {config.BASE_URL}: {exc}")
        return 1

    print("OK: credentials, signing, and host all working. Safe to run trader.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
