"""
Structured logging setup using Loguru.

All stdlib ``logging`` output is intercepted and re-routed through Loguru so
that third-party libraries (SQLAlchemy, asyncio, httpx, …) appear in the same
stream with the same format.

Usage
-----
    from trading_system.core.logger import setup_logger, get_logger

    setup_logger("INFO", "./logs/trading.log")
    log = get_logger(__name__)
    log.info("System started", version="1.0")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# stdlib → loguru bridge
# ---------------------------------------------------------------------------

class _InterceptHandler(logging.Handler):
    """
    Redirect all stdlib ``logging`` records to Loguru.

    The *depth* calculation ensures that Loguru reports the *original* call
    site (inside the third-party library) rather than this handler.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        # Map stdlib level to Loguru level name
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk up the call stack to find the frame that issued the log call
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_SETUP_DONE = False


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "50 MB",
    retention: int = 10,
    serialize: bool = False,
) -> None:
    """
    Configure Loguru sinks and intercept stdlib logging.

    Parameters
    ----------
    log_level:
        Minimum log level for both the console and the file sink
        (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    log_file:
        Absolute or relative path for the rotating log file.
        If *None*, only the console sink is installed.
    rotation:
        Loguru rotation spec (default ``"50 MB"``).
    retention:
        Maximum number of rotated files to keep (default 10).
    serialize:
        If *True*, write JSON-serialised records to the file sink.
        The console sink always uses a human-readable format.
    """
    global _SETUP_DONE

    # Remove default Loguru handler so we start clean on repeated calls
    logger.remove()

    # ------------------------------------------------------------------
    # Console sink — human-readable with colour
    # ------------------------------------------------------------------
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
        "{extra}"
    )

    logger.add(
        sys.stdout,
        level=log_level.upper(),
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,          # thread-safe async queue
    )

    # ------------------------------------------------------------------
    # File sink — structured (JSON-like) with rotation
    # ------------------------------------------------------------------
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{extra}"
        )

        logger.add(
            str(file_path),
            level=log_level.upper(),
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=serialize,   # True → pure JSON lines
            backtrace=True,
            diagnose=False,        # no sensitive local vars in files
            enqueue=True,
        )

    # ------------------------------------------------------------------
    # Intercept stdlib logging
    # ------------------------------------------------------------------
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Silence noisy libraries unless you really want their DEBUG output
    for noisy in ("urllib3", "asyncio", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _SETUP_DONE = True

    logger.info(
        "Logging initialised",
        log_level=log_level,
        log_file=log_file or "stdout only",
    )


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def get_logger(name: str) -> "logger":  # type: ignore[valid-type]
    """
    Return a Loguru logger bound with ``module=name``.

    The returned object is the global Loguru ``logger`` bound with a context
    key so that every message emitted via the returned logger carries the
    module name in its ``extra`` dict.

    If ``setup_logger`` has not been called yet, a minimal console-only setup
    is performed automatically so the system stays operable in tests.

    Example
    -------
    ::

        log = get_logger(__name__)
        log.info("Position opened", asset="AAPL", qty=100)
    """
    if not _SETUP_DONE:
        setup_logger()

    return logger.bind(module=name)
