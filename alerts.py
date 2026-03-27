"""
alerts.py — Telegram notification helpers.
All functions are fire-and-forget; failures are logged but never raised.
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_BASE_URL = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"


def _send(text: str) -> None:
    """Send a plain-text message. Silently swallows all errors."""
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.debug("Telegram not configured — skipping alert")
        return
    try:
        resp = requests.post(
            _BASE_URL,
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("Telegram error (non-fatal): %s", exc)


def alert_entry(ticker: str, edge: float, my_prob: float,
                kalshi_price: float, provider: str) -> None:
    msg = (
        f"<b>EDGE TRADE</b>: {ticker}\n"
        f"Edge: +{edge:.1f}%  |  My prob: {my_prob:.1f}%  |  "
        f"Kalshi: {kalshi_price:.1f}¢  |  Provider: {provider}"
    )
    _send(msg)
    logger.info("Telegram entry alert sent: %s", ticker)


def alert_exit(ticker: str, entry_price: float, exit_price: float,
               pnl_cents: float, reason: str) -> None:
    sign = "+" if pnl_cents >= 0 else ""
    msg = (
        f"<b>EXIT</b>: {ticker}\n"
        f"Entry: {entry_price:.1f}¢  |  Exit: {exit_price:.1f}¢  |  "
        f"P&L: {sign}{pnl_cents:.1f}¢  |  Reason: {reason}"
    )
    _send(msg)
    logger.info("Telegram exit alert sent: %s | reason=%s", ticker, reason)


def alert_low_balance(balance: float) -> None:
    msg = (
        f"<b>HALT</b>: Balance below $5 (${balance:.2f}). "
        f"Trading suspended."
    )
    _send(msg)
    logger.warning("Low balance alert sent: $%.2f", balance)


def alert_error(context: str, error: str) -> None:
    """Generic error ping — used sparingly for critical failures."""
    msg = f"<b>BOT ERROR</b> [{context}]\n<code>{error[:500]}</code>"
    _send(msg)
