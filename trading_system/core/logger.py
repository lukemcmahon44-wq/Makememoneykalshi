"""
Structured logging with loguru. Writes to console + rotating file.

Use as: from trading_system.core.logger import get_logger; log = get_logger(__name__)
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru

_configured = False


def configure_logger(log_dir: str = "./trading_system/logs/", level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _loguru.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    _loguru.add(sys.stderr, format=fmt, level=level, enqueue=True)
    _loguru.add(
        Path(log_dir) / "trading.log",
        rotation="50 MB",
        retention="14 days",
        compression="gz",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    _loguru.add(
        Path(log_dir) / "errors.log",
        rotation="10 MB",
        retention="30 days",
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True,
    )
    _configured = True


def get_logger(name: str | None = None):
    if not _configured:
        configure_logger()
    return _loguru.bind(component=name or "system")
