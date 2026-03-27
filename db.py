"""
db.py — SQLite persistence layer for trades and open positions.
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "trades.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                question        TEXT    NOT NULL,
                provider        TEXT    NOT NULL,
                my_probability  REAL    NOT NULL,
                kalshi_price    REAL    NOT NULL,
                edge_score      REAL    NOT NULL,
                entry_price     REAL    NOT NULL,
                exit_price      REAL,
                contracts       INTEGER NOT NULL DEFAULT 1,
                order_id        TEXT,
                status          TEXT    NOT NULL DEFAULT 'open',
                exit_reason     TEXT,
                pnl_cents       REAL,
                opened_at       TEXT    NOT NULL,
                closed_at       TEXT
            );

            CREATE TABLE IF NOT EXISTS positions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL UNIQUE,
                question        TEXT    NOT NULL,
                provider        TEXT    NOT NULL,
                entry_price     REAL    NOT NULL,
                my_probability  REAL    NOT NULL,
                edge_score      REAL    NOT NULL,
                contracts       INTEGER NOT NULL DEFAULT 1,
                order_id        TEXT,
                opened_at       TEXT    NOT NULL
            );
        """)
    logger.info("Database initialised at %s", DB_PATH)


# ── Position management ───────────────────────────────────────────────────────

def open_position(ticker: str, question: str, provider: str,
                  entry_price: float, my_probability: float,
                  edge_score: float, contracts: int, order_id: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO positions
                (ticker, question, provider, entry_price, my_probability,
                 edge_score, contracts, order_id, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, question, provider, entry_price, my_probability,
              edge_score, contracts, order_id, now))

        conn.execute("""
            INSERT INTO trades
                (ticker, question, provider, my_probability, kalshi_price,
                 edge_score, entry_price, contracts, order_id, status, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """, (ticker, question, provider, my_probability, entry_price,
              edge_score, entry_price, contracts, order_id, now))
    logger.debug("Opened position: %s", ticker)


def close_position(ticker: str, exit_price: float, exit_reason: str) -> Optional[float]:
    """
    Close an open position. Returns PnL in cents (positive = profit).
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE ticker = ?", (ticker,)
        ).fetchone()

        if not row:
            logger.warning("close_position called for unknown ticker: %s", ticker)
            return None

        entry_price = row["entry_price"]
        contracts = row["contracts"]
        pnl_cents = (exit_price - entry_price) * contracts

        now = datetime.utcnow().isoformat()
        conn.execute(
            "DELETE FROM positions WHERE ticker = ?", (ticker,)
        )
        conn.execute("""
            UPDATE trades
            SET exit_price  = ?,
                status      = 'closed',
                exit_reason = ?,
                pnl_cents   = ?,
                closed_at   = ?
            WHERE ticker = ? AND status = 'open'
        """, (exit_price, exit_reason, pnl_cents, now, ticker))

    logger.debug("Closed position: %s | exit=%.2f | pnl=%.2f¢", ticker, exit_price, pnl_cents)
    return pnl_cents


def get_open_positions() -> list:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY opened_at").fetchall()
    return [dict(r) for r in rows]


def is_position_open(ticker: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM positions WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row is not None


def count_open_positions() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM positions").fetchone()
    return row["cnt"]


# ── P&L helpers ───────────────────────────────────────────────────────────────

def get_all_trades() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY opened_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_pnl_summary() -> dict:
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                AS total_trades,
                SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) AS wins,
                COALESCE(SUM(pnl_cents), 0)             AS total_pnl_cents
            FROM trades
            WHERE status = 'closed'
        """).fetchone()
    total = row["total_trades"] or 0
    wins = row["wins"] or 0
    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": (wins / total * 100) if total else 0.0,
        "total_pnl_cents": row["total_pnl_cents"] or 0.0,
    }
