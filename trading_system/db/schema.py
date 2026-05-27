"""
SQLAlchemy Core schema definitions for the trading system.

All tables are defined using SQLAlchemy Core (``Table`` / ``Column``) rather
than the ORM so that callers interact with plain Python dicts and SQL
expressions.  This keeps the data layer simple, fast, and easy to migrate.

WAL mode is enabled on the SQLite engine so that concurrent readers do not
block the writer (important for a system where the main trading loop and the
dashboard/reporting threads all read from the same DB).

Usage
-----
    from trading_system.db.schema import get_engine, create_all_tables

    engine = get_engine("./data/trading.db")
    create_all_tables(engine)
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Engine

from trading_system.core.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared metadata registry
# ---------------------------------------------------------------------------

metadata = MetaData()

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

# 1. candles — OHLCV price data for every asset / timeframe combination
candles = Table(
    "candles",
    metadata,
    Column("id",         Integer,  primary_key=True, autoincrement=True),
    Column("asset",      String(32), nullable=False),
    Column("timeframe",  String(16), nullable=False),   # e.g. "1m", "5m", "1h", "1d"
    Column("timestamp",  String(32), nullable=False),   # ISO-8601 string (UTC)
    Column("open",       Float,    nullable=False),
    Column("high",       Float,    nullable=False),
    Column("low",        Float,    nullable=False),
    Column("close",      Float,    nullable=False),
    Column("volume",     Float,    nullable=False),
    Column("vwap",       Float,    nullable=True),
    Column("created_at", String(32), nullable=False),   # ISO-8601 UTC insert time
)

# 2. trades — individual round-trip trade records
trades = Table(
    "trades",
    metadata,
    Column("id",               Integer,  primary_key=True, autoincrement=True),
    Column("asset",            String(32), nullable=False),
    Column("direction",        String(8),  nullable=False),  # "long" | "short"
    Column("entry_price",      Float,    nullable=False),
    Column("exit_price",       Float,    nullable=True),
    Column("entry_time",       String(32), nullable=False),  # ISO-8601 UTC
    Column("exit_time",        String(32), nullable=True),
    Column("qty",              Float,    nullable=False),
    Column("pnl",              Float,    nullable=True),
    Column("pnl_pct",          Float,    nullable=True),
    Column("holding_bars",     Integer,  nullable=True),
    Column("signal_value",     Float,    nullable=True),
    Column("signal_breakdown", Text,     nullable=True),     # JSON text
    Column("commission",       Float,    nullable=True, default=0.0),
    Column("slippage",         Float,    nullable=True, default=0.0),
    Column("status",           String(16), nullable=False, default="open"),
    # status: "open" | "closed" | "cancelled"
)

# 3. signals — raw signal snapshots at each bar
signals = Table(
    "signals",
    metadata,
    Column("id",               Integer,  primary_key=True, autoincrement=True),
    Column("asset",            String(32), nullable=False),
    Column("timestamp",        String(32), nullable=False),  # ISO-8601 UTC
    Column("signal_value",     Float,    nullable=False),    # aggregate score [−1, 1]
    Column("signal_breakdown", Text,     nullable=True),     # JSON
    Column("regime_data",      Text,     nullable=True),     # JSON
    Column("created_at",       String(32), nullable=False),
)

# 4. risk_events — risk-limit breaches and system alerts
risk_events = Table(
    "risk_events",
    metadata,
    Column("id",              Integer,  primary_key=True, autoincrement=True),
    Column("timestamp",       String(32), nullable=False),  # ISO-8601 UTC
    Column("event_type",      String(64), nullable=False),
    # e.g. "MAX_DRAWDOWN", "DAILY_LOSS_LIMIT", "KILL_SWITCH", "POSITION_LIMIT"
    Column("asset",           String(32), nullable=True),
    Column("message",         Text,     nullable=True),
    Column("portfolio_value", Float,    nullable=True),
    Column("resolved_at",     String(32), nullable=True),   # NULL while open
)

# 5. portfolio_snapshots — periodic portfolio state snapshots
portfolio_snapshots = Table(
    "portfolio_snapshots",
    metadata,
    Column("id",              Integer,  primary_key=True, autoincrement=True),
    Column("timestamp",       String(32), nullable=False),  # ISO-8601 UTC
    Column("portfolio_value", Float,    nullable=False),
    Column("cash",            Float,    nullable=False),
    Column("positions",       Text,     nullable=True),     # JSON: {asset: qty}
    Column("drawdown",        Float,    nullable=True),
    Column("daily_pnl",       Float,    nullable=True),
)

# 6. system_state — key/value store for durable system metadata
system_state = Table(
    "system_state",
    metadata,
    Column("id",         Integer,  primary_key=True, autoincrement=True),
    Column("key",        String(128), nullable=False, unique=True),
    Column("value",      Text,     nullable=True),
    Column("updated_at", String(32), nullable=False),       # ISO-8601 UTC
)

# 7. model_metrics — training / evaluation results for ML models
model_metrics = Table(
    "model_metrics",
    metadata,
    Column("id",          Integer,  primary_key=True, autoincrement=True),
    Column("model_name",  String(128), nullable=False),
    Column("fold",        Integer,  nullable=True),         # cross-validation fold
    Column("train_start", String(32), nullable=True),       # ISO-8601 date
    Column("test_end",    String(32), nullable=True),       # ISO-8601 date
    Column("accuracy",    Float,    nullable=True),
    Column("auc",         Float,    nullable=True),
    Column("sharpe",      Float,    nullable=True),
    Column("created_at",  String(32), nullable=False),
)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def get_engine(db_path: str = "./data/trading.db") -> Engine:
    """
    Create and return a SQLAlchemy ``Engine`` for the given SQLite path.

    WAL mode is enabled via a ``connect`` event listener so it is applied on
    every new connection, including those created by connection pooling.

    The parent directory for *db_path* is created automatically if it does
    not exist.

    Parameters
    ----------
    db_path:
        Filesystem path (absolute or relative) to the SQLite database file.

    Returns
    -------
    Engine
        Configured SQLAlchemy engine ready for use.
    """
    resolved = Path(db_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{resolved}",
        connect_args={
            "check_same_thread": False,     # allow multi-threaded access
            "timeout": 30,                  # busy-wait up to 30 s for a write lock
        },
        pool_pre_ping=True,                 # discard stale connections
    )

    @event.listens_for(engine, "connect")
    def _set_wal_mode(dbapi_connection, connection_record):  # noqa: ANN001
        """Enable WAL and optimise SQLite for concurrent read/write."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")    # safe with WAL
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA cache_size=-65536")     # 64 MB page cache
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

    log.debug("SQLite engine created", db_path=str(resolved))
    return engine


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_all_tables(engine: Engine) -> None:
    """
    Create all tables defined in this module if they do not already exist.

    This function is idempotent — calling it on an existing database with
    existing tables is safe and leaves the data intact.

    Parameters
    ----------
    engine:
        A SQLAlchemy engine, typically from :func:`get_engine`.
    """
    try:
        metadata.create_all(engine, checkfirst=True)
        log.info(
            "Database schema initialised",
            tables=list(metadata.tables.keys()),
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Failed to create database tables",
            error=str(exc),
            exc_info=True,
        )
        raise
