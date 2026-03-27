"""
main.py — Entry point for the Kalshi Edge Trading Bot.

Usage:
    python main.py

Performs startup checks, prints all-time P&L summary, then
hands off to the blocking bot loop.
"""

import logging
import logging.handlers
import os
import sys
import traceback

from dotenv import load_dotenv

# Load .env before importing anything that reads os.getenv
load_dotenv()

import db
import bot
from db import get_pnl_summary


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging() -> None:
    log_file  = os.getenv("LOG_FILE",  "bot.log")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    level = getattr(logging, log_level, logging.INFO)

    # Rotating file handler (10 MB, keep 3 files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )

    # Console handler with colour via colorlog if available
    try:
        import colorlog
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


# ── Startup validation ─────────────────────────────────────────────────────────

def check_required_env() -> None:
    required = ["KALSHI_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)


def print_pnl_banner() -> None:
    summary = get_pnl_summary()
    print("=" * 55)
    print("  Kalshi Edge Bot — All-Time P&L Summary")
    print("=" * 55)
    print(f"  Total trades  : {summary['total_trades']}")
    print(f"  Wins          : {summary['wins']}")
    print(f"  Losses        : {summary['losses']}")
    print(f"  Win rate      : {summary['win_rate']:.1f}%")
    total_pnl = summary['total_pnl_cents']
    sign = "+" if total_pnl >= 0 else ""
    print(f"  Total P&L     : {sign}{total_pnl:.1f}¢  (${total_pnl/100:.2f})")
    print("=" * 55)
    print()


# ── Entry ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 55)
    logger.info("  Kalshi Edge Trading Bot starting up")
    logger.info("=" * 55)

    check_required_env()

    try:
        db.init_db()
    except Exception:
        logger.critical("Database initialisation failed:\n%s", traceback.format_exc())
        sys.exit(1)

    print_pnl_banner()

    logger.info(
        "Config — EDGE_THRESHOLD=%.1f%% | MAX_POSITIONS=%d | MAX_SIZE=$%.2f | "
        "HALT_BELOW=$%.2f | SCAN=%ds",
        float(os.getenv("EDGE_THRESHOLD", "8")),
        int(os.getenv("MAX_OPEN_POSITIONS", "5")),
        float(os.getenv("MAX_POSITION_SIZE", "2")),
        float(os.getenv("MIN_BALANCE_HALT", "5")),
        int(os.getenv("SCAN_INTERVAL", "60")),
    )

    try:
        bot.run_loop()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
    except Exception:
        logger.critical("Fatal error in bot loop:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
