"""
btc_bot.py — BTC Hourly Market Bot ("Cowshed Robot")

Strategy:
  Each scan cycle, find open BTC hourly price markets (KXBTCU series).
  - If YES ask price is in [BTC_BUY_MIN, BTC_BUY_MAX] cents, buy YES.
  - If NO ask price is in [BTC_BUY_MIN, BTC_BUY_MAX] cents, buy NO.
  - Skip the hour if nothing qualifies.

Positions resolve automatically within the hour — no manual exit needed.
P&L = (100 - entry_price) * contracts on a win, -entry_price * contracts on a loss.
"""

import logging
import math
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import alerts
import db
import kalshi_client as kalshi

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BTC_SERIES_TICKER  = os.getenv("BTC_SERIES_TICKER",     "KXBTCU")
BTC_BUY_MIN        = int(os.getenv("BTC_BUY_MIN",       "95"))   # min cents to qualify
BTC_BUY_MAX        = int(os.getenv("BTC_BUY_MAX",       "99"))   # max cents to qualify
TRADE_SIZE_DOLLARS = float(os.getenv("BTC_TRADE_SIZE",  "10"))   # target spend per trade
MIN_BALANCE_HALT   = float(os.getenv("MIN_BALANCE_HALT", "5"))
SCAN_INTERVAL      = int(os.getenv("SCAN_INTERVAL",     "300"))  # seconds between scans
MIN_MINS_TO_EXPIRY = int(os.getenv("BTC_MIN_MINS_TO_EXPIRY", "5"))  # skip near-closing markets


# ── Market helpers ────────────────────────────────────────────────────────────

def _minutes_to_expiry(market: dict) -> float:
    raw = market.get("close_time") or market.get("expiration_time")
    if not raw:
        return float("inf")
    try:
        expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (expiry - datetime.now(timezone.utc)).total_seconds() / 60
    except Exception:
        return float("inf")


def _get_prices(market: dict) -> tuple:
    """Return (yes_ask, no_ask) in cents. Derives missing side from the other."""
    yes_ask = market.get("yes_ask")
    no_ask  = market.get("no_ask")

    if yes_ask is not None and no_ask is not None:
        return float(yes_ask), float(no_ask)
    if yes_ask is not None:
        return float(yes_ask), 100.0 - float(yes_ask)
    if no_ask is not None:
        return 100.0 - float(no_ask), float(no_ask)

    # Last-resort: use last traded price
    last = market.get("last_price")
    if last is not None:
        yes = float(last)
        return yes, 100.0 - yes

    return None, None


def _qualifying_side(market: dict) -> Optional[tuple]:
    """
    Return (side, price_cents) if YES or NO qualifies, else None.
    YES takes priority when both would qualify (shouldn't happen in practice).
    """
    yes_ask, no_ask = _get_prices(market)
    if yes_ask is None:
        return None

    if BTC_BUY_MIN <= yes_ask <= BTC_BUY_MAX:
        return ("yes", int(round(yes_ask)))
    if no_ask is not None and BTC_BUY_MIN <= no_ask <= BTC_BUY_MAX:
        return ("no", int(round(no_ask)))
    return None


def _contracts_for_size(price_cents: int) -> int:
    """Number of contracts to buy to spend approximately TRADE_SIZE_DOLLARS."""
    if price_cents <= 0:
        return 1
    return max(1, math.floor(TRADE_SIZE_DOLLARS * 100 / price_cents))


def _market_result(market: dict) -> Optional[str]:
    """
    Return 'yes' or 'no' if market is definitively resolved, else None.
    Tries the result/winner field first, then infers from the final price.
    """
    status = (market.get("status") or "").lower()
    if status not in ("resolved", "finalized", "settled", "closed"):
        return None

    result = (market.get("result") or market.get("winner") or "").lower()
    if result in ("yes", "no"):
        return result

    # Infer from price: a resolved YES market has yes_ask ~100, NO has yes_ask ~0
    yes_ask, _ = _get_prices(market)
    if yes_ask is not None:
        return "yes" if yes_ask >= 50 else "no"

    return None


# ── Exit checker ──────────────────────────────────────────────────────────────

def _check_exits(summary: dict) -> None:
    """Close positions where the market has already resolved."""
    for pos in db.get_open_positions():
        # Only handle BTC positions placed by this bot
        if pos.get("provider") != "btc_hourly":
            continue

        ticker = pos["ticker"]
        side   = pos.get("side", "yes")

        try:
            market = kalshi.get_market(ticker)
        except Exception:
            logger.error("Failed to fetch %s:\n%s", ticker, traceback.format_exc())
            continue

        if market is None:
            continue

        result = _market_result(market)
        if result is None:
            continue  # not resolved yet

        entry_price = pos["entry_price"]
        contracts   = pos["contracts"]

        # exit_price = 100 if our side won, 0 if lost
        exit_price = 100.0 if (result == side) else 0.0
        pnl        = db.close_position(ticker, exit_price, "resolved")

        pnl_cents  = pnl or 0.0
        pnl_sign   = "+" if pnl_cents >= 0 else ""
        logger.info(
            "RESOLVED %s | side=%s | result=%s | entry=%d¢ × %d = %s%.1f¢",
            ticker, side, result, entry_price, contracts, pnl_sign, pnl_cents,
        )
        alerts.alert_exit(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_cents=pnl_cents,
            reason=f"resolved-{result}",
        )
        summary["positions_exited"] += 1


# ── Main scan-and-trade ───────────────────────────────────────────────────────

def run_once() -> dict:
    summary = {
        "scanned":          0,
        "qualified":        0,
        "trades_placed":    0,
        "positions_exited": 0,
        "errors":           0,
    }

    # Balance guard
    try:
        balance = kalshi.get_balance()
    except Exception:
        logger.error("Balance fetch failed:\n%s", traceback.format_exc())
        balance = None

    if balance is not None and balance < MIN_BALANCE_HALT:
        alerts.alert_low_balance(balance)
        logger.warning("Balance $%.2f below halt threshold — pausing new trades", balance)
        _check_exits(summary)
        return summary

    # Fetch BTC markets
    try:
        markets = kalshi.get_btc_markets(BTC_SERIES_TICKER)
    except Exception:
        logger.error("Failed to fetch BTC markets:\n%s", traceback.format_exc())
        summary["errors"] += 1
        return summary

    summary["scanned"] = len(markets)
    logger.info("Fetched %d BTC markets", len(markets))

    # Resolve existing positions before entering new ones
    _check_exits(summary)

    # Scan for qualifying entries
    for market in markets:
        ticker = market.get("ticker", "")

        if db.is_position_open(ticker):
            continue

        mins = _minutes_to_expiry(market)
        if mins < MIN_MINS_TO_EXPIRY:
            logger.debug("Skipping %s — %.1f min to expiry", ticker, mins)
            continue

        qual = _qualifying_side(market)
        if qual is None:
            continue

        side, price_cents = qual
        contracts = _contracts_for_size(price_cents)
        cost_dollars = price_cents * contracts / 100
        win_dollars  = (100 - price_cents) * contracts / 100
        title = market.get("title") or market.get("question") or ticker

        logger.info(
            "QUALIFYING: %s | side=%s | %d¢ × %d contracts | "
            "cost=$%.2f | win=+$%.2f | %.0f min left",
            ticker, side, price_cents, contracts, cost_dollars, win_dollars, mins,
        )

        summary["qualified"] += 1

        try:
            order = kalshi.place_order(
                ticker=ticker,
                side=side,
                count=contracts,
                price_cents=price_cents,
            )
        except Exception:
            logger.error("Order error for %s:\n%s", ticker, traceback.format_exc())
            summary["errors"] += 1
            continue

        if order is None:
            logger.warning("Order returned None for %s — skipping", ticker)
            summary["errors"] += 1
            continue

        order_id = order.get("order_id", "")
        db.open_position(
            ticker=ticker,
            question=title,
            provider="btc_hourly",
            entry_price=float(price_cents),
            my_probability=float(price_cents),
            edge_score=0.0,
            contracts=contracts,
            order_id=order_id,
            side=side,
        )
        alerts.alert_btc_entry(
            ticker=ticker,
            side=side,
            price_cents=price_cents,
            contracts=contracts,
            cost_dollars=cost_dollars,
            win_dollars=win_dollars,
            minutes_left=mins,
        )
        summary["trades_placed"] += 1

    return summary


# ── Blocking run loop ─────────────────────────────────────────────────────────

def run_loop() -> None:
    logger.info(
        "Cowshed Robot starting | series=%s | threshold=%d–%d¢ | "
        "size=$%.2f | scan=%ds",
        BTC_SERIES_TICKER, BTC_BUY_MIN, BTC_BUY_MAX, TRADE_SIZE_DOLLARS, SCAN_INTERVAL,
    )
    while True:
        cycle_start = time.time()
        try:
            summary = run_once()
            logger.info(
                "CYCLE | scanned=%d | qualified=%d | trades=%d | exits=%d | errors=%d",
                summary["scanned"], summary["qualified"],
                summary["trades_placed"], summary["positions_exited"], summary["errors"],
            )
        except Exception:
            logger.critical("Unhandled error in BTC bot:\n%s", traceback.format_exc())

        sleep_for = max(0, SCAN_INTERVAL - (time.time() - cycle_start))
        logger.debug("Sleeping %.0fs until next scan", sleep_for)
        time.sleep(sleep_for)
