"""Console + rotating-file logging for the trader."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup(level_name: str, log_file: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    # Replace handlers on re-entry (e.g. tests run setup multiple times).
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = Path(log_file)
    if log_path.parent and str(log_path.parent) not in ("", "."):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    file_h = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    # httpx is chatty at DEBUG; pin it to WARNING regardless of root level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
