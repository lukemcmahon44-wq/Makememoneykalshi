"""
Telegram alert helper. Silent when token/chat_id are unset.
"""

from __future__ import annotations

from typing import Optional

import aiohttp

from .core.logger import get_logger

log = get_logger(__name__)


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, message: str) -> None:
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json={"chat_id": self.chat_id,
                                              "text": message[:4000]},
                                    timeout=10) as r:
                    if r.status != 200:
                        log.warning(f"Telegram send returned {r.status}")
        except Exception as e:
            log.error(f"Telegram send error: {e}")


_alerter: Optional[TelegramAlerter] = None


def init_alerter(config) -> None:
    global _alerter
    _alerter = TelegramAlerter(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)


async def send_telegram_alert(message: str) -> None:
    if _alerter is not None:
        await _alerter.send(message)
