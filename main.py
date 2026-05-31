"""
main.py — Entry point for the Kalshi high-probability auto-trader.

Run:
    python main.py
"""

from __future__ import annotations

import logging
import sys

import config
import logging_setup
from kalshi import KalshiClient
from kalshi.auth import KalshiSigner
from trader import run_forever


def _build_signer() -> KalshiSigner:
    if config.KALSHI_PRIVATE_KEY_PATH:
        return KalshiSigner.from_pem_file(
            config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH
        )
    if config.KALSHI_PRIVATE_KEY:
        return KalshiSigner.from_pem_string(
            config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY
        )
    raise RuntimeError("No Kalshi private key configured")


def _banner(logger: logging.Logger) -> None:
    snap = config.snapshot()
    logger.info("=" * 64)
    logger.info("  Kalshi High-Probability Auto-Trader")
    logger.info("=" * 64)
    logger.info("  environment            : %s", snap.environment)
    logger.info("  base url               : %s", snap.base_url)
    logger.info("  LIVE_TRADING           : %s", snap.live_trading)
    logger.info("  sizing mode            : %s", snap.sizing_mode)
    logger.info("  fixed trade size       : $%.2f", snap.fixed_trade_size_usd)
    logger.info("  price band             : %d..%d¢", *snap.price_band)
    logger.info("  re-eval interval (hrs) : %.2f", snap.recheck_interval_hours)
    logger.info("  api key id set         : %s", snap.api_key_id_set)
    logger.info("  private key source     : %s", snap.private_key_source)
    logger.info("  log file               : %s", snap.log_file)
    logger.info("=" * 64)
    if not snap.live_trading:
        logger.info("  DRY-RUN MODE — no orders will be submitted.")
        logger.info("  Flip LIVE_TRADING=True in your .env to trade for real.")
        logger.info("=" * 64)


def main() -> int:
    logging_setup.setup(config.LOG_LEVEL, config.LOG_FILE)
    logger = logging.getLogger("main")

    problems = config.validate()
    if problems:
        for p in problems:
            logger.error("CONFIG ERROR  | %s", p)
        logger.error("Fix the above in your .env and re-run.")
        return 2

    _banner(logger)

    for w in config.warnings():
        logger.warning("CONFIG WARN   | %s", w)

    try:
        signer = _build_signer()
    except Exception as exc:
        logger.error("AUTH ERROR    | could not load private key: %s", exc)
        return 2

    client = KalshiClient(
        base_url=config.base_url(),
        signer=signer,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        max_retries=config.HTTP_MAX_RETRIES,
        backoff_base=config.HTTP_BACKOFF_BASE_SECONDS,
        backoff_cap=config.HTTP_BACKOFF_CAP_SECONDS,
    )
    try:
        run_forever(client, recheck_interval_hours=config.RECHECK_INTERVAL_HOURS)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Goodbye.")
        return 0
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
