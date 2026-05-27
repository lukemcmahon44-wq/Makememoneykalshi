"""
Hardware-style emergency stop. Touching the KILL_SWITCH_FILE halts all trading.

Two mechanisms:
1. File-based:  touch /tmp/trading_kill_switch  → halts everything immediately
2. Process signal: SIGUSR1 → kill switch engages
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from .logger import get_logger

log = get_logger(__name__)

KILL_SWITCH_FILE = Path(os.getenv("KILL_SWITCH_FILE", "/tmp/trading_kill_switch"))


class KillSwitch:
    def __init__(self, config=None):
        self.config = config
        self._engaged = False
        self._reason = ""
        self._install_signal_handler()

    def _install_signal_handler(self) -> None:
        try:
            signal.signal(signal.SIGUSR1, self._on_signal)
        except (ValueError, AttributeError):
            pass

    def _on_signal(self, signum, frame) -> None:
        self.engage(f"SIGUSR1 received ({signum})")

    def engage(self, reason: str) -> None:
        if not self._engaged:
            self._engaged = True
            self._reason = reason
            log.critical(f"KILL SWITCH ENGAGED: {reason}")

    def disengage(self) -> None:
        if self._engaged:
            log.warning(f"KILL SWITCH DISENGAGED (was: {self._reason})")
        self._engaged = False
        self._reason = ""
        try:
            if KILL_SWITCH_FILE.exists():
                KILL_SWITCH_FILE.unlink()
        except OSError as e:
            log.error(f"Failed to remove kill switch file: {e}")

    def is_engaged(self) -> bool:
        if self._engaged:
            return True
        if KILL_SWITCH_FILE.exists():
            self.engage(f"Kill switch file detected: {KILL_SWITCH_FILE}")
            return True
        return False

    @property
    def reason(self) -> str:
        return self._reason

    async def monitor(self, poll_interval_secs: float = 1.0) -> None:
        """Background task: poll for kill switch file."""
        log.info(f"Kill switch monitor active. File: {KILL_SWITCH_FILE}")
        while True:
            try:
                self.is_engaged()
            except Exception as e:
                log.error(f"Kill switch monitor error: {e}")
            await asyncio.sleep(poll_interval_secs)
