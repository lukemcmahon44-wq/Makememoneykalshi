"""
state.py — Persistent fill log and realized-P&L tracking (SQLite).

Two responsibilities:
  1. Record every order INTENTION and RESULT to a durable store, with enough
     detail to reconcile later against GET /portfolio/settlements (for backtest
     calibration and taxes). The raw API response is stored verbatim.
  2. Track realized P&L from settlements so the trader can enforce a daily loss
     limit.

All monetary values are stored as TEXT (the str() of a Decimal) to preserve
exact precision; they are read back as Decimal. No float touches the store.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _money(obj: dict, dollar_key: str, cents_key: str) -> Optional[Decimal]:
    """Prefer a `*_dollars` string field; fall back to an integer-cents field."""
    dollars = _to_decimal(obj.get(dollar_key))
    if dollars is not None:
        return dollars
    cents = _to_decimal(obj.get(cents_key))
    return (cents / Decimal(100)) if cents is not None else None


class Store:
    """Thin SQLite wrapper for fills and settlements."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc          TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                side            TEXT NOT NULL,
                action          TEXT NOT NULL,
                price_dollars   TEXT NOT NULL,
                count           TEXT NOT NULL,
                dollar_amount   TEXT NOT NULL,
                client_order_id TEXT NOT NULL,
                order_id        TEXT,
                status          TEXT,
                filled_count    TEXT,
                dry_run         INTEGER NOT NULL DEFAULT 0,
                raw             TEXT
            );

            CREATE TABLE IF NOT EXISTS settlements (
                settlement_key TEXT PRIMARY KEY,
                ticker         TEXT,
                settled_time   TEXT,
                market_result  TEXT,
                revenue        TEXT,
                cost           TEXT,
                fee            TEXT,
                pnl            TEXT,
                raw            TEXT
            );
            """
        )
        self._conn.commit()

    # ── Fills ────────────────────────────────────────────────────────────────
    def log_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        price: Decimal,
        count: Decimal,
        dollar_amount: Decimal,
        client_order_id: str,
        result: Optional[dict],
        dry_run: bool,
    ) -> None:
        """Persist one order intention + its result (or dry-run marker)."""
        order_id = None
        status = "dry_run" if dry_run else None
        filled_count = None
        if result:
            order_id = result.get("order_id")
            status = result.get("status", status)
            filled_count = (
                result.get("fill_count_fp")
                or result.get("fill_count")
                or result.get("filled_count")
            )
        self._conn.execute(
            """
            INSERT INTO fills (ts_utc, ticker, side, action, price_dollars, count,
                               dollar_amount, client_order_id, order_id, status,
                               filled_count, dry_run, raw)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                ticker,
                side,
                action,
                str(price),
                str(count),
                str(dollar_amount),
                client_order_id,
                order_id,
                status,
                str(filled_count) if filled_count is not None else None,
                1 if dry_run else 0,
                json.dumps(result) if result is not None else None,
            ),
        )
        self._conn.commit()

    def fills_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM fills WHERE substr(ts_utc, 1, 10) = ?", (today,)
        )
        return int(cur.fetchone()[0])

    # ── Settlements / realized P&L ────────────────────────────────────────────
    def record_settlements(self, settlements: list[dict]) -> None:
        """Upsert settlements and compute per-settlement realized P&L.

        Units are handled defensively: `*_dollars` string fields are preferred,
        with integer-cent fields (revenue, fee_cost) divided by 100 as a
        fallback. P&L = revenue - (yes_cost + no_cost) - fees. The raw payload is
        always stored for authoritative reconciliation regardless.
        """
        for s in settlements:
            ticker = s.get("ticker")
            settled_time = s.get("settled_time") or s.get("settlement_time") or ""
            key = f"{ticker}|{settled_time}"

            revenue = _money(s, "revenue_dollars", "revenue")
            yes_cost = _money(s, "yes_total_cost_dollars", "yes_total_cost")
            no_cost = _money(s, "no_total_cost_dollars", "no_total_cost")
            fee = _money(s, "fee_cost_dollars", "fee_cost")

            cost = (yes_cost or Decimal(0)) + (no_cost or Decimal(0))
            pnl: Optional[Decimal] = None
            if revenue is not None:
                pnl = revenue - cost - (fee or Decimal(0))

            self._conn.execute(
                """
                INSERT INTO settlements
                    (settlement_key, ticker, settled_time, market_result,
                     revenue, cost, fee, pnl, raw)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(settlement_key) DO UPDATE SET
                    market_result=excluded.market_result,
                    revenue=excluded.revenue,
                    cost=excluded.cost,
                    fee=excluded.fee,
                    pnl=excluded.pnl,
                    raw=excluded.raw
                """,
                (
                    key,
                    ticker,
                    settled_time,
                    s.get("market_result"),
                    str(revenue) if revenue is not None else None,
                    str(cost),
                    str(fee) if fee is not None else None,
                    str(pnl) if pnl is not None else None,
                    json.dumps(s),
                ),
            )
        self._conn.commit()

    def realized_pnl_today(self, today: Optional[date] = None) -> Decimal:
        """Net realized P&L (signed dollars) for settlements settled today UTC.
        Negative means a net loss."""
        day = (today or datetime.now(timezone.utc).date()).isoformat()
        cur = self._conn.execute(
            "SELECT pnl FROM settlements WHERE substr(settled_time, 1, 10) = ? AND pnl IS NOT NULL",
            (day,),
        )
        total = Decimal(0)
        for row in cur.fetchall():
            val = _to_decimal(row["pnl"])
            if val is not None:
                total += val
        return total

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
