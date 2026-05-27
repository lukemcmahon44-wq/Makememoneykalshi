"""
SQLite schema. All tables defined here. Run `init_db(path)` to create.

Use Decimal-string for monetary values to avoid float drift.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- Closed candles for each asset/timeframe
CREATE TABLE IF NOT EXISTS candles (
    asset      TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    timestamp  INTEGER NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    PRIMARY KEY (asset, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_asset_tf_ts ON candles(asset, timeframe, timestamp);

-- Closed trades (executions)
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    side            TEXT NOT NULL,           -- 'buy' or 'sell'
    direction       TEXT NOT NULL,           -- 'long' or 'short'
    qty             REAL NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    entry_time      INTEGER NOT NULL,
    exit_time       INTEGER,
    signal_strength REAL,
    composite_signal REAL,
    pnl             REAL,
    pnl_pct         REAL,
    commission      REAL DEFAULT 0,
    slippage_bps    REAL DEFAULT 0,
    notes           TEXT,
    status          TEXT NOT NULL DEFAULT 'OPEN'
);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);

-- Per-bar signal snapshots (for IC analysis)
CREATE TABLE IF NOT EXISTS signal_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    category    TEXT NOT NULL,
    value       REAL NOT NULL,
    composite   REAL,
    decision    TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_asset_ts ON signal_history(asset, timestamp);

-- Risk and circuit breaker events
CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    asset           TEXT,
    severity        TEXT,
    message         TEXT NOT NULL,
    portfolio_value REAL
);
CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_type ON risk_events(event_type);

-- Order lifecycle log (audit trail)
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id TEXT,
    asset           TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    submitted_at    INTEGER NOT NULL,
    filled_at       INTEGER,
    fill_price      REAL,
    fill_qty        REAL,
    status          TEXT NOT NULL,
    rejection_reason TEXT,
    parent_strategy TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_broker_id ON orders(broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_asset ON orders(asset);

-- Portfolio NAV snapshots
CREATE TABLE IF NOT EXISTS nav_history (
    timestamp       INTEGER PRIMARY KEY,
    portfolio_value REAL NOT NULL,
    cash            REAL NOT NULL,
    positions_value REAL NOT NULL,
    n_positions     INTEGER NOT NULL,
    daily_pnl       REAL,
    drawdown        REAL,
    peak_value      REAL
);

-- ML model registry
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    trained_at      INTEGER NOT NULL,
    oos_accuracy    REAL,
    oos_auc         REAL,
    oos_sharpe      REAL,
    accepted        INTEGER DEFAULT 0,
    path            TEXT,
    metadata_json   TEXT
);

-- Slippage / fill quality log
CREATE TABLE IF NOT EXISTS fill_quality (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    timestamp       INTEGER NOT NULL,
    expected_price  REAL NOT NULL,
    actual_price    REAL NOT NULL,
    slippage_bps    REAL NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    order_type      TEXT NOT NULL
);

-- System state checkpoints
CREATE TABLE IF NOT EXISTS system_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       INTEGER NOT NULL,
    event           TEXT NOT NULL,
    payload_json    TEXT
);
"""


def init_db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn
