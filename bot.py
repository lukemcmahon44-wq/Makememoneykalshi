"""
bot.py — Core scan-and-trade loop.

Responsibilities:
  1. Fetch all open Kalshi markets every SCAN_INTERVAL seconds
  2. Filter by risk rules (open interest, time-to-expiry)
  3. Calculate edge via edge_calculator
  4. Place orders when edge >= EDGE_THRESHOLD
  5. Monitor open positions and exit on stop / edge-flip / resolution
  6. Halt trading when balance is low
"""

import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import alerts
import db
import kalshi_client as kalshi
from edge_calculator import calculate_edge

logger = logging.getLogger(__name__)

# ── Config from env ───────────────────────────────────────────────────────────
EDGE_THRESHOLD      = float(os.getenv("EDGE_THRESHOLD",      "8"))
MAX_POSITION_SIZE   = float(os.getenv("MAX_POSITION_SIZE",   "2"))    # dollars
MAX_OPEN_POSITIONS  = int(os.getenv("MAX_OPEN_POSITIONS",    "5"))
MIN_BALANCE_HALT    = float(os.getenv("MIN_BALANCE_HALT",    "5"))
SCAN_INTERVAL       = int(os.getenv("SCAN_INTERVAL",         "60"))

# Exit thresholds
STOP_LOSS_CENTS     = 15      # exit if YES price drops > 15¢ below entry
EDGE_FLIP_THRESHOLD = 10.0    # exit if edge flips negative by > 10%

# Market filters
MIN_OPEN_INTEREST   = 100
MIN_HOURS_TO_EXPIRY = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_expiry(market: dict) -> Optional[datetime]:
    raw = market.get("close_time") or market.get("expiration_time")
    if not raw:
        return None
    try:
        # Kalshi uses ISO-8601 with Z suffix
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _hours_to_expiry(market: dict) -> float:
    expiry = _parse_expiry(market)
    if not expiry:
        return float("inf")
    now = datetime.now(timezone.utc)
    delta = (expiry - now).total_seconds() / 3600
    return delta


def _open_interest(market: dict) -> int:
    return int(market.get("open_interest", 0) or 0)


def _market_resolved(market: dict) -> bool:
    status = (market.get("status") or "").lower()
    return status in ("resolved", "finalized", "settled", "closed")


def _yes_price(market: dict) -> Optional[float]:
    p = market.get("yes_ask") or market.get("last_price")
    return float(p) if p is not None else None


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once() -> dict:
    """
    Execute one full scan cycle.
    Returns a summary dict for logging.
    """
    summary = {
        "scanned": 0,
        "evaluated": 0,
        "edges_found": 0,
        "trades_placed": 0,
        "positions_exited": 0,
        "errors": 0,
    }

    # ── Balance check ─────────────────────────────────────────────────────────
    try:
        balance = kalshi.get_balance()
    except Exception:
        logger.error("Failed to fetch balance:\n%s", traceback.format_exc())
        balance = None

    if balance is not None and balance < MIN_BALANCE_HALT:
        alerts.alert_low_balance(balance)
        logger.warning("Balance $%.2f below halt threshold $%.2f — skipping new trades",
                       balance, MIN_BALANCE_HALT)
        _check_exits(summary)
        return summary

    # ── Fetch markets ─────────────────────────────────────────────────────────
    try:
        markets = kalshi.get_markets(status="open")
    except Exception:
        logger.error("Failed to fetch markets:\n%s", traceback.format_exc())
        summary["errors"] += 1
        return summary

    summary["scanned"] = len(markets)

    # ── Check existing positions for exits ────────────────────────────────────
    _check_exits(summary)

    open_count = db.count_open_positions()
    if open_count >= MAX_OPEN_POSITIONS:
        logger.info("Max positions (%d) reached — skipping entry scan", MAX_OPEN_POSITIONS)
        return summary

    # ── Scan for new edges ────────────────────────────────────────────────────
    for market in markets:
        if open_count >= MAX_OPEN_POSITIONS:
            break

        ticker = market.get("ticker", "")

        # Skip already-held positions
        if db.is_position_open(ticker):
            continue

        # Basic filters
        if _open_interest(market) < MIN_OPEN_INTEREST:
            continue
        if _hours_to_expiry(market) < MIN_HOURS_TO_EXPIRY:
            continue

        # Edge calculation
        try:
            result = calculate_edge(market)
        except Exception:
            logger.error("Edge calculation error for %s:\n%s", ticker, traceback.format_exc())
            summary["errors"] += 1
            continue

        if result is None:
            continue

        summary["evaluated"] += 1

        if result.edge < EDGE_THRESHOLD:
            logger.debug("Edge %.1f%% below threshold for %s", result.edge, ticker)
            continue

        summary["edges_found"] += 1
        logger.info(
            "EDGE FOUND: %s | edge=+%.1f%% | my_prob=%.1f%% | kalshi=%.1f¢ | provider=%s",
            ticker, result.edge, result.my_probability, result.kalshi_price, result.provider_name
        )

        # ── Place order ───────────────────────────────────────────────────────
        try:
            order = kalshi.place_order(
                ticker=ticker,
                side="yes",
                count=1,
                price_cents=int(round(result.kalshi_price)),
            )
        except Exception:
            logger.error("Order placement error for %s:\n%s", ticker, traceback.format_exc())
            summary["errors"] += 1
            continue

        if order is None:
            logger.warning("Order failed for %s — logged and skipping (no retry)", ticker)
            summary["errors"] += 1
            continue

        order_id = order.get("order_id", "")
        db.open_position(
            ticker=ticker,
            question=result.question,
            provider=result.provider_name,
            entry_price=result.kalshi_price,
            my_probability=result.my_probability,
            edge_score=result.edge,
            contracts=1,
            order_id=order_id,
        )
        alerts.alert_entry(
            ticker=ticker,
            edge=result.edge,
            my_prob=result.my_probability,
            kalshi_price=result.kalshi_price,
            provider=result.provider_name,
        )
        summary["trades_placed"] += 1
        open_count += 1

    return summary


def _check_exits(summary: dict) -> None:
    """
    Review all open positions and exit if any exit condition is met.
    Mutates summary["positions_exited"].
    """
    positions = db.get_open_positions()
    if not positions:
        return

    # Fetch current market data in bulk
    for pos in positions:
        ticker = pos["ticker"]
        try:
            market = kalshi.get_market(ticker)
        except Exception:
            logger.error("Failed to fetch market data for %s:\n%s",
                         ticker, traceback.format_exc())
            continue

        if market is None:
            logger.warning("Market %s returned None — skipping exit check", ticker)
            continue

        current_price = _yes_price(market)
        if current_price is None:
            continue

        entry_price    = pos["entry_price"]
        my_probability = pos["my_probability"]
        exit_reason    = None

        # ── Condition A: market resolved ──────────────────────────────────────
        if _market_resolved(market):
            exit_reason = "resolved"

        # ── Condition B: stop-loss ────────────────────────────────────────────
        elif (entry_price - current_price) >= STOP_LOSS_CENTS:
            exit_reason = "stop"

        # ── Condition C: edge flip ────────────────────────────────────────────
        else:
            try:
                result = calculate_edge(market)
                if result is not None and result.edge <= -EDGE_FLIP_THRESHOLD:
                    exit_reason = "edge-flip"
                elif current_price < (my_probability - EDGE_FLIP_THRESHOLD):
                    exit_reason = "edge-flip"
            except Exception:
                logger.error("Edge check during exit scan failed for %s:\n%s",
                             ticker, traceback.format_exc())

        if exit_reason:
            pnl = db.close_position(ticker, current_price, exit_reason)
            alerts.alert_exit(
                ticker=ticker,
                entry_price=entry_price,
                exit_price=current_price,
                pnl_cents=pnl or 0.0,
                reason=exit_reason,
            )
            logger.info(
                "EXITED %s | reason=%s | entry=%.1f¢ | exit=%.1f¢ | pnl=%.1f¢",
                ticker, exit_reason, entry_price, current_price, pnl or 0.0
            )
            summary["positions_exited"] += 1


# ── Blocking run loop (called from main.py) ───────────────────────────────────

def run_loop() -> None:
    logger.info("Bot loop starting. Scan interval: %ds", SCAN_INTERVAL)
    while True:
        cycle_start = time.time()
        try:
            summary = run_once()
            logger.info(
                "CYCLE COMPLETE | scanned=%d | evaluated=%d | edges=%d | "
                "trades=%d | exits=%d | errors=%d",
                summary["scanned"], summary["evaluated"], summary["edges_found"],
                summary["trades_placed"], summary["positions_exited"], summary["errors"],
            )
        except Exception:
            logger.critical("Unhandled exception in run_once:\n%s", traceback.format_exc())

        elapsed = time.time() - cycle_start
        sleep_for = max(0, SCAN_INTERVAL - elapsed)
        logger.debug("Sleeping %.1fs until next scan", sleep_for)
        time.sleep(sleep_for)
