"""
DB repository. All SQL lives here - never raw SQL in other modules.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from ..core.logger import get_logger
from .schema import init_db

log = get_logger(__name__)


class DatabaseRepository:
    def __init__(self, config):
        self.config = config
        self.db_path = config.DB_PATH
        self._lock = threading.RLock()
        self.conn = init_db(self.db_path)
        log.info(f"DB initialised at {self.db_path}")

    # ── candles ──────────────────────────────────────────────────────────
    def write_candle(self, asset: str, timeframe: str, ts: int,
                     o: float, h: float, l: float, c: float, v: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO candles (asset, timeframe, timestamp, "
                "open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
                (asset, timeframe, ts, o, h, l, c, v),
            )

    def get_candles(self, asset: str, timeframe: str, limit: int = 500) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM candles WHERE asset=? AND timeframe=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (asset, timeframe, limit),
            ))

    # ── trades ───────────────────────────────────────────────────────────
    def open_trade(self, asset: str, side: str, direction: str, qty: float,
                   entry_price: float, signal_strength: float = 0,
                   composite_signal: float = 0, notes: str = "") -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO trades (asset, side, direction, qty, entry_price, "
                "entry_time, signal_strength, composite_signal, notes, status) "
                "VALUES (?,?,?,?,?,?,?,?,?, 'OPEN')",
                (asset, side, direction, qty, entry_price,
                 int(time.time()), signal_strength, composite_signal, notes),
            )
            return cur.lastrowid

    def close_trade(self, trade_id: int, exit_price: float, pnl: float,
                    pnl_pct: float, commission: float = 0,
                    slippage_bps: float = 0) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?, "
                "commission=?, slippage_bps=?, status='CLOSED' WHERE id=?",
                (exit_price, int(time.time()), pnl, pnl_pct, commission,
                 slippage_bps, trade_id),
            )

    def open_trades(self, asset: Optional[str] = None) -> List[sqlite3.Row]:
        with self._lock:
            if asset:
                return list(self.conn.execute(
                    "SELECT * FROM trades WHERE status='OPEN' AND asset=?", (asset,)))
            return list(self.conn.execute(
                "SELECT * FROM trades WHERE status='OPEN'"))

    def closed_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE status='CLOSED' "
                "ORDER BY exit_time DESC LIMIT ?", (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── signals ──────────────────────────────────────────────────────────
    def write_signal(self, asset: str, category: str, value: float,
                     composite: float = 0, decision: str = "") -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO signal_history (asset, timestamp, category, value, "
                "composite, decision) VALUES (?,?,?,?,?,?)",
                (asset, int(time.time()), category, value, composite, decision),
            )

    def recent_signals(self, asset: str, category: str, limit: int = 500
                       ) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM signal_history WHERE asset=? AND category=? "
                "ORDER BY timestamp DESC LIMIT ?", (asset, category, limit),
            ))

    # ── risk events ──────────────────────────────────────────────────────
    def write_risk_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO risk_events (timestamp, event_type, asset, severity, "
                "message, portfolio_value) VALUES (?,?,?,?,?,?)",
                (payload.get("timestamp", int(time.time())),
                 payload["event_type"], payload.get("asset"),
                 payload.get("severity", "INFO"), payload["message"],
                 payload.get("portfolio_value")),
            )

    def recent_risk_events(self, limit: int = 50) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM risk_events ORDER BY timestamp DESC LIMIT ?", (limit,)))

    # ── orders ───────────────────────────────────────────────────────────
    def log_order(self, payload: Dict[str, Any]) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO orders (broker_order_id, asset, side, qty, order_type, "
                "limit_price, submitted_at, status, parent_strategy) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (payload.get("broker_order_id"), payload["asset"], payload["side"],
                 payload["qty"], payload["order_type"], payload.get("limit_price"),
                 payload.get("submitted_at", int(time.time())),
                 payload.get("status", "SUBMITTED"),
                 payload.get("parent_strategy", "")),
            )
            return cur.lastrowid

    def update_order_fill(self, order_id: int, fill_price: float,
                          fill_qty: float, status: str = "FILLED") -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE orders SET fill_price=?, fill_qty=?, filled_at=?, status=? "
                "WHERE id=?",
                (fill_price, fill_qty, int(time.time()), status, order_id),
            )

    # ── NAV ──────────────────────────────────────────────────────────────
    def write_nav(self, ts: int, portfolio_value: float, cash: float,
                  positions_value: float, n_positions: int,
                  daily_pnl: float = 0, drawdown: float = 0,
                  peak_value: float = 0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO nav_history (timestamp, portfolio_value, "
                "cash, positions_value, n_positions, daily_pnl, drawdown, peak_value) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, portfolio_value, cash, positions_value, n_positions,
                 daily_pnl, drawdown, peak_value),
            )

    def get_portfolio_value(self) -> float:
        with self._lock:
            row = self.conn.execute(
                "SELECT portfolio_value FROM nav_history ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return float(row[0]) if row else 0.0

    def nav_history(self, limit: int = 500) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM nav_history ORDER BY timestamp DESC LIMIT ?", (limit,)))

    # ── fill quality ─────────────────────────────────────────────────────
    def log_fill_quality(self, asset: str, expected: float, actual: float,
                         side: str, qty: float, order_type: str) -> None:
        slippage_bps = ((actual - expected) / expected) * 10_000 if expected else 0
        if side == "sell":
            slippage_bps = -slippage_bps
        with self._lock:
            self.conn.execute(
                "INSERT INTO fill_quality (asset, timestamp, expected_price, "
                "actual_price, slippage_bps, side, qty, order_type) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (asset, int(time.time()), expected, actual, slippage_bps,
                 side, qty, order_type),
            )

    # ── models ───────────────────────────────────────────────────────────
    def register_model(self, name: str, version: str, oos_accuracy: float,
                       oos_auc: float, oos_sharpe: float, accepted: bool,
                       path: str, metadata: Dict[str, Any]) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO models (name, version, trained_at, oos_accuracy, "
                "oos_auc, oos_sharpe, accepted, path, metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (name, version, int(time.time()), oos_accuracy, oos_auc,
                 oos_sharpe, 1 if accepted else 0, path, json.dumps(metadata)),
            )
            return cur.lastrowid

    def latest_accepted_model(self, name: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM models WHERE name=? AND accepted=1 "
                "ORDER BY trained_at DESC LIMIT 1", (name,),
            ).fetchone()

    # ── system state ─────────────────────────────────────────────────────
    def write_system_state(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO system_state (timestamp, event, payload_json) "
                "VALUES (?,?,?)",
                (int(time.time()), payload.get("event", "STATE"),
                 json.dumps(payload, default=str)),
            )

    def test_write(self) -> bool:
        try:
            with self._lock:
                self.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
