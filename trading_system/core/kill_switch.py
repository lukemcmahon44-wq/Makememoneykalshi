"""
Hardware kill switch for the trading system.

The kill switch is implemented as a sentinel file (``./kill.flag``) on disk so
that it survives process restarts and can be triggered from outside the process
(e.g. a shell script, a monitoring daemon, or a human operator).

Usage
-----
    from trading_system.core.kill_switch import KillSwitch

    ks = KillSwitch()

    # In your main async loop:
    asyncio.create_task(ks.monitor())

    # To trigger programmatically:
    ks.engage(reason="max drawdown exceeded")

    # To check inside hot path:
    ks.check_and_raise()

    # External trigger (shell):
    #   echo "manual stop" > ./kill.flag
"""

from __future__ import annotations

import asyncio
import datetime
import os
from pathlib import Path
from typing import Optional

from trading_system.core.logger import get_logger

log = get_logger(__name__)

# Default location for the sentinel file (relative to cwd)
_DEFAULT_FLAG_PATH = Path("./kill.flag")

# Polling interval for the async monitor (seconds)
_MONITOR_INTERVAL_SECONDS: float = 5.0


class KillSwitch:
    """
    File-backed hardware kill switch.

    The kill switch is considered *active* when ``flag_path`` exists on disk.
    The file contents are human-readable (ISO timestamp + reason) so operators
    can inspect why the kill switch fired.

    Attributes
    ----------
    flag_path : Path
        Path to the sentinel file.
    kill_event : asyncio.Event
        Asyncio event that is set when the kill switch becomes active.
        Coroutines can ``await kill_event.wait()`` to block until the switch
        fires.
    """

    def __init__(self, flag_path: Optional[Path | str] = None) -> None:
        self.flag_path: Path = (
            Path(flag_path) if flag_path is not None else _DEFAULT_FLAG_PATH
        )
        self.kill_event: asyncio.Event = asyncio.Event()

        # If the flag already exists at startup, set the event immediately so
        # any waiting coroutines unblock without requiring a poll cycle.
        if self.flag_path.exists():
            log.critical(
                "Kill switch flag detected at startup — system is halted",
                flag_path=str(self.flag_path),
            )
            # asyncio.Event is not yet attached to a running loop at import
            # time; we set it lazily inside is_active() / monitor() instead.

    # ------------------------------------------------------------------
    # Core state accessors
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """
        Return ``True`` if the kill switch is currently engaged.

        The check is a simple file-existence test; no caching is performed so
        the result always reflects the current filesystem state.
        """
        active = self.flag_path.exists()
        if active and not self.kill_event.is_set():
            # Lazily synchronise the asyncio event in case the flag was
            # created externally between poll cycles.
            try:
                self.kill_event.set()
            except RuntimeError:
                # No running event loop — safe to ignore; monitor() will set
                # it once the loop is running.
                pass
        return active

    def engage(self, reason: str = "manual trigger") -> None:
        """
        Engage the kill switch by writing the sentinel file.

        Parameters
        ----------
        reason:
            Human-readable explanation written into the flag file.
        """
        timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        content = f"KILL SWITCH ENGAGED\ntimestamp: {timestamp}\nreason: {reason}\n"

        try:
            self.flag_path.parent.mkdir(parents=True, exist_ok=True)
            self.flag_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            log.error(
                "Failed to write kill switch flag — proceeding with in-memory flag only",
                error=str(exc),
                flag_path=str(self.flag_path),
            )

        try:
            self.kill_event.set()
        except RuntimeError:
            pass

        log.critical(
            "Kill switch ENGAGED",
            reason=reason,
            timestamp=timestamp,
            flag_path=str(self.flag_path),
        )

    def disengage(self) -> None:
        """
        Disengage the kill switch by deleting the sentinel file.

        This allows the system to resume normal operation.  Note that any
        in-flight ``check_and_raise()`` calls that already observed an active
        kill switch will have raised ``SystemExit``; this method is primarily
        useful for resetting state between test runs or after a controlled halt.
        """
        if self.flag_path.exists():
            try:
                os.remove(self.flag_path)
                log.info(
                    "Kill switch disengaged — sentinel file removed",
                    flag_path=str(self.flag_path),
                )
            except OSError as exc:
                log.error(
                    "Failed to remove kill switch flag",
                    error=str(exc),
                    flag_path=str(self.flag_path),
                )
        else:
            log.debug(
                "Kill switch disengage called but flag file does not exist",
                flag_path=str(self.flag_path),
            )

        # Clear the asyncio event so coroutines no longer unblock immediately
        self.kill_event.clear()

    # ------------------------------------------------------------------
    # Async monitor
    # ------------------------------------------------------------------

    async def monitor(self) -> None:
        """
        Async coroutine that polls the sentinel file every
        ``_MONITOR_INTERVAL_SECONDS`` seconds.

        When the kill switch becomes active this method:

        1. Logs a CRITICAL message.
        2. Sets ``self.kill_event`` so all awaiting coroutines unblock.

        The coroutine runs indefinitely until the enclosing task is cancelled.
        It should be launched as a background task at application startup::

            asyncio.create_task(kill_switch.monitor())
        """
        log.debug(
            "Kill switch monitor started",
            interval_s=_MONITOR_INTERVAL_SECONDS,
            flag_path=str(self.flag_path),
        )

        was_active = self.flag_path.exists()
        if was_active:
            log.critical(
                "Kill switch already active at monitor start",
                flag_path=str(self.flag_path),
            )
            self.kill_event.set()

        while True:
            try:
                await asyncio.sleep(_MONITOR_INTERVAL_SECONDS)
                currently_active = self.flag_path.exists()

                if currently_active and not was_active:
                    # Transition: inactive → active
                    flag_content = "(unreadable)"
                    try:
                        flag_content = self.flag_path.read_text(encoding="utf-8").strip()
                    except OSError:
                        pass

                    log.critical(
                        "Kill switch ACTIVATED — all trading halted",
                        flag_path=str(self.flag_path),
                        flag_content=flag_content,
                    )
                    self.kill_event.set()

                elif not currently_active and was_active:
                    # Transition: active → inactive (operator reset)
                    log.warning(
                        "Kill switch flag removed — system may resume if explicitly restarted",
                        flag_path=str(self.flag_path),
                    )
                    # We do NOT clear the event here; a deliberate restart
                    # should call disengage() explicitly.

                was_active = currently_active

            except asyncio.CancelledError:
                log.info("Kill switch monitor task cancelled — shutting down")
                raise
            except Exception as exc:  # noqa: BLE001
                # Never let a monitoring exception crash the loop; log and continue
                log.error(
                    "Unexpected error in kill switch monitor loop",
                    error=str(exc),
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Synchronous guard
    # ------------------------------------------------------------------

    def check_and_raise(self) -> None:
        """
        Raise ``SystemExit(1)`` if the kill switch is currently active.

        Call this at the top of every order-submission path or risk-check loop
        to ensure no trades are placed while the system is halted.

        Raises
        ------
        SystemExit
            With exit code 1 and a descriptive message if the kill switch is
            active.
        """
        if self.is_active():
            reason = "(unknown)"
            try:
                reason = self.flag_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass

            log.critical(
                "check_and_raise(): kill switch is active — raising SystemExit",
                flag_path=str(self.flag_path),
                reason=reason,
            )
            raise SystemExit(
                f"Trading system halted by kill switch. "
                f"Flag: {self.flag_path}. Content: {reason}"
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KillSwitch(flag_path={str(self.flag_path)!r}, "
            f"active={self.is_active()}, "
            f"event_set={self.kill_event.is_set()})"
        )
