"""
Database repository layer for the trading system.

All persistence operations go through ``DatabaseRepository``.  Every public
method catches its own exceptions so that a transient database error never
crashes the trading loop — instead it is logged and the method returns a safe
default value (``None``, ``[]``, ``False``, etc.).

The repository is *not* a singleton.  Instantiate it once at startup and pass
it around (or use dependency injection).  It is thread-safe because SQLAlchemy
manages the connection pool.

Usage
-----
    from trading_system.db.repository import DatabaseRepository
    from trading_system.core.config import get_config

    repo = DatabaseRepository(get_config())

    # Write a candle
    repo.write_candle("AAPL", "1m", {
        "timestamp": "2025-01-15T14:30:00Z",
        "open": 235.10, "high": 235.50,
        "low": 234.80, "close": 235.30,
        "volume": 12_500, "vwap": 235.15,
    })

    # Read recent trades
    trades = repo.get_recent_trades(asset="AAPL", limit=20)
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from trading_system.core.logger import get_logger
from trading_system.db.schema import (
    candles,
    create_all_tables,
    get_engine,
    model_metrics,
    portfolio_snapshots,
    risk_events,
    signals,
    system_state,
    trades,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _row_to_dict(row) -> Dict[str, Any]:  # noqa: ANN001
    """Convert a SQLAlchemy ``Row`` object to a plain Python dict."""
    return dict(row._mapping)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class DatabaseRepository:
    """
    Thin persistence layer backed by a SQLite database via SQLAlchemy Core.

    Parameters
    ----------
    config:
        A :class:`~trading_system.core.config.TradingConfig` instance.  Only
        ``config.DB_PATH`` is used; everything else is ignored by this class.
    """

    def __init__(self, config) -> None:  # noqa: ANN001  (avoid circular import)
        self._engine: Engine = get_engine(config.DB_PATH)
        create_all_tables(self._engine)
        log.info("DatabaseRepository initialised", db_path=config.DB_PATH)

    # ------------------------------------------------------------------
    # Internal connection helper
    # ------------------------------------------------------------------

    def _connect(self):  # noqa: ANN201
        """Return a new SQLAlchemy connection from the pool."""
        return self._engine.connect()

    # ==================================================================
    # Candles
    # ==================================================================

    def write_candle(
        self,
        asset: str,
        timeframe: str,
        ohlcv_dict: Dict[str, Any],
    ) -> Optional[int]:
        """
        Insert a single OHLCV candle row.

        Parameters
        ----------
        asset:
            Ticker symbol, e.g. ``"AAPL"`` or ``"BTCUSD"``.
        timeframe:
            Bar timeframe, e.g. ``"1m"``, ``"5m"``, ``"1d"``.
        ohlcv_dict:
            Dict with keys: ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``.  Optional key: ``vwap``.

        Returns
        -------
        int | None
            The inserted row ``id``, or ``None`` on failure.
        """
        try:
            row = {
                "asset":      asset,
                "timeframe":  timeframe,
                "timestamp":  str(ohlcv_dict["timestamp"]),
                "open":       float(ohlcv_dict["open"]),
                "high":       float(ohlcv_dict["high"]),
                "low":        float(ohlcv_dict["low"]),
                "close":      float(ohlcv_dict["close"]),
                "volume":     float(ohlcv_dict["volume"]),
                "vwap":       float(ohlcv_dict["vwap"]) if ohlcv_dict.get("vwap") is not None else None,
                "created_at": _now_utc(),
            }
            with self._connect() as conn:
                result = conn.execute(insert(candles).values(**row))
                conn.commit()
                return result.inserted_primary_key[0]
        except Exception as exc:  # noqa: BLE001
            log.error(
                "write_candle failed",
                asset=asset,
                timeframe=timeframe,
                error=str(exc),
            )
            return None

    # ==================================================================
    # Trades
    # ==================================================================

    def write_trade(self, trade_dict: Dict[str, Any]) -> Optional[int]:
        """
        Insert a new trade record (status defaults to ``"open"``).

        Expected keys: ``asset``, ``direction``, ``entry_price``,
        ``entry_time``, ``qty``.  All other columns are optional.

        Returns
        -------
        int | None
            Inserted trade id, or ``None`` on failure.
        """
        try:
            row = {
                "asset":            str(trade_dict["asset"]),
                "direction":        str(trade_dict.get("direction", "long")),
                "entry_price":      float(trade_dict["entry_price"]),
                "exit_price":       trade_dict.get("exit_price"),
                "entry_time":       str(trade_dict.get("entry_time", _now_utc())),
                "exit_time":        trade_dict.get("exit_time"),
                "qty":              float(trade_dict["qty"]),
                "pnl":              trade_dict.get("pnl"),
                "pnl_pct":          trade_dict.get("pnl_pct"),
                "holding_bars":     trade_dict.get("holding_bars"),
                "signal_value":     trade_dict.get("signal_value"),
                "signal_breakdown": (
                    json.dumps(trade_dict["signal_breakdown"])
                    if isinstance(trade_dict.get("signal_breakdown"), (dict, list))
                    else trade_dict.get("signal_breakdown")
                ),
                "commission":       float(trade_dict.get("commission", 0.0)),
                "slippage":         float(trade_dict.get("slippage", 0.0)),
                "status":           str(trade_dict.get("status", "open")),
            }
            with self._connect() as conn:
                result = conn.execute(insert(trades).values(**row))
                conn.commit()
                trade_id = result.inserted_primary_key[0]
                log.debug(
                    "Trade written",
                    trade_id=trade_id,
                    asset=row["asset"],
                    direction=row["direction"],
                )
                return trade_id
        except Exception as exc:  # noqa: BLE001
            log.error("write_trade failed", error=str(exc), trade=trade_dict)
            return None

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: str,
        pnl: float,
        pnl_pct: float,
        holding_bars: Optional[int] = None,
    ) -> bool:
        """
        Update a trade record with exit information and mark it ``"closed"``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        try:
            values: Dict[str, Any] = {
                "exit_price": float(exit_price),
                "exit_time":  str(exit_time),
                "pnl":        float(pnl),
                "pnl_pct":    float(pnl_pct),
                "status":     "closed",
            }
            if holding_bars is not None:
                values["holding_bars"] = int(holding_bars)

            with self._connect() as conn:
                conn.execute(
                    update(trades).where(trades.c.id == trade_id).values(**values)
                )
                conn.commit()
                log.debug(
                    "Trade exit recorded",
                    trade_id=trade_id,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                return True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "update_trade_exit failed",
                trade_id=trade_id,
                error=str(exc),
            )
            return False

    # ==================================================================
    # Signals
    # ==================================================================

    def write_signal(
        self,
        asset: str,
        timestamp: str,
        signal_value: float,
        signal_breakdown: Optional[Dict[str, Any]] = None,
        regime_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        Persist a signal snapshot.

        Returns
        -------
        int | None
            Inserted signal id, or ``None`` on failure.
        """
        try:
            row = {
                "asset":            asset,
                "timestamp":        str(timestamp),
                "signal_value":     float(signal_value),
                "signal_breakdown": json.dumps(signal_breakdown) if signal_breakdown is not None else None,
                "regime_data":      json.dumps(regime_data) if regime_data is not None else None,
                "created_at":       _now_utc(),
            }
            with self._connect() as conn:
                result = conn.execute(insert(signals).values(**row))
                conn.commit()
                return result.inserted_primary_key[0]
        except Exception as exc:  # noqa: BLE001
            log.error(
                "write_signal failed",
                asset=asset,
                timestamp=timestamp,
                error=str(exc),
            )
            return None

    # ==================================================================
    # Risk events
    # ==================================================================

    def write_risk_event(self, event_dict: Dict[str, Any]) -> Optional[int]:
        """
        Insert a risk event record.

        Expected keys: ``event_type``, ``message``.
        Optional keys: ``asset``, ``portfolio_value``, ``timestamp``.

        Returns
        -------
        int | None
            Inserted risk event id, or ``None`` on failure.
        """
        try:
            row = {
                "timestamp":       str(event_dict.get("timestamp", _now_utc())),
                "event_type":      str(event_dict["event_type"]),
                "asset":           event_dict.get("asset"),
                "message":         str(event_dict.get("message", "")),
                "portfolio_value": event_dict.get("portfolio_value"),
                "resolved_at":     event_dict.get("resolved_at"),
            }
            with self._connect() as conn:
                result = conn.execute(insert(risk_events).values(**row))
                conn.commit()
                log.warning(
                    "Risk event recorded",
                    event_type=row["event_type"],
                    asset=row["asset"],
                    message=row["message"],
                )
                return result.inserted_primary_key[0]
        except Exception as exc:  # noqa: BLE001
            log.error("write_risk_event failed", error=str(exc), event=event_dict)
            return None

    # ==================================================================
    # Portfolio snapshots
    # ==================================================================

    def write_portfolio_snapshot(self, snapshot_dict: Dict[str, Any]) -> Optional[int]:
        """
        Insert a portfolio state snapshot.

        Expected keys: ``portfolio_value``, ``cash``.
        Optional keys: ``positions`` (dict), ``drawdown``, ``daily_pnl``,
        ``timestamp``.

        Returns
        -------
        int | None
            Inserted snapshot id, or ``None`` on failure.
        """
        try:
            positions = snapshot_dict.get("positions")
            row = {
                "timestamp":       str(snapshot_dict.get("timestamp", _now_utc())),
                "portfolio_value": float(snapshot_dict["portfolio_value"]),
                "cash":            float(snapshot_dict["cash"]),
                "positions":       json.dumps(positions) if positions is not None else None,
                "drawdown":        snapshot_dict.get("drawdown"),
                "daily_pnl":       snapshot_dict.get("daily_pnl"),
            }
            with self._connect() as conn:
                result = conn.execute(insert(portfolio_snapshots).values(**row))
                conn.commit()
                return result.inserted_primary_key[0]
        except Exception as exc:  # noqa: BLE001
            log.error("write_portfolio_snapshot failed", error=str(exc))
            return None

    # ==================================================================
    # Read operations
    # ==================================================================

    def get_recent_trades(
        self,
        asset: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent *limit* trades, optionally filtered by *asset*.

        The ``signal_breakdown`` column is automatically decoded from JSON.

        Returns
        -------
        list[dict]
            List of trade dicts ordered by id descending (most recent first).
            Returns ``[]`` on failure.
        """
        try:
            stmt = select(trades).order_by(trades.c.id.desc()).limit(limit)
            if asset is not None:
                stmt = stmt.where(trades.c.asset == asset)

            with self._connect() as conn:
                rows = conn.execute(stmt).fetchall()

            result = []
            for row in rows:
                d = _row_to_dict(row)
                if isinstance(d.get("signal_breakdown"), str):
                    try:
                        d["signal_breakdown"] = json.loads(d["signal_breakdown"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                result.append(d)
            return result
        except Exception as exc:  # noqa: BLE001
            log.error("get_recent_trades failed", asset=asset, error=str(exc))
            return []

    def get_portfolio_snapshots(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Return portfolio snapshots from the last *days* days.

        The ``positions`` column is automatically decoded from JSON.

        Returns
        -------
        list[dict]
            Ordered by timestamp ascending.  Returns ``[]`` on failure.
        """
        try:
            cutoff = (
                datetime.datetime.utcnow() - datetime.timedelta(days=days)
            ).isoformat(timespec="milliseconds") + "Z"

            stmt = (
                select(portfolio_snapshots)
                .where(portfolio_snapshots.c.timestamp >= cutoff)
                .order_by(portfolio_snapshots.c.timestamp.asc())
            )

            with self._connect() as conn:
                rows = conn.execute(stmt).fetchall()

            result = []
            for row in rows:
                d = _row_to_dict(row)
                if isinstance(d.get("positions"), str):
                    try:
                        d["positions"] = json.loads(d["positions"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                result.append(d)
            return result
        except Exception as exc:  # noqa: BLE001
            log.error("get_portfolio_snapshots failed", days=days, error=str(exc))
            return []

    def get_risk_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Return risk events from the last *hours* hours.

        Returns
        -------
        list[dict]
            Ordered by timestamp descending (most recent first).
            Returns ``[]`` on failure.
        """
        try:
            cutoff = (
                datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
            ).isoformat(timespec="milliseconds") + "Z"

            stmt = (
                select(risk_events)
                .where(risk_events.c.timestamp >= cutoff)
                .order_by(risk_events.c.timestamp.desc())
            )

            with self._connect() as conn:
                rows = conn.execute(stmt).fetchall()

            return [_row_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            log.error("get_risk_events failed", hours=hours, error=str(exc))
            return []

    # ==================================================================
    # System state (key/value store)
    # ==================================================================

    def get_system_state(self, key: str) -> Optional[str]:
        """
        Retrieve the value associated with *key* from the system_state table.

        Returns
        -------
        str | None
            The stored value string, or ``None`` if the key does not exist or
            on failure.
        """
        try:
            stmt = select(system_state.c.value).where(system_state.c.key == key)
            with self._connect() as conn:
                row = conn.execute(stmt).fetchone()
            if row is None:
                return None
            return row[0]
        except Exception as exc:  # noqa: BLE001
            log.error("get_system_state failed", key=key, error=str(exc))
            return None

    def set_system_state(self, key: str, value: str) -> bool:
        """
        Upsert a key/value pair into the system_state table.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        try:
            now = _now_utc()
            with self._connect() as conn:
                # Check for existing row
                existing = conn.execute(
                    select(system_state.c.id).where(system_state.c.key == key)
                ).fetchone()

                if existing is not None:
                    conn.execute(
                        update(system_state)
                        .where(system_state.c.key == key)
                        .values(value=str(value), updated_at=now)
                    )
                else:
                    conn.execute(
                        insert(system_state).values(
                            key=key, value=str(value), updated_at=now
                        )
                    )
                conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("set_system_state failed", key=key, error=str(exc))
            return False

    # ==================================================================
    # Portfolio value convenience accessor
    # ==================================================================

    def get_portfolio_value(self) -> float:
        """
        Return the portfolio value from the most recent snapshot.

        Returns
        -------
        float
            Most recent ``portfolio_value``, or ``0.0`` if no snapshots exist
            or on failure.
        """
        try:
            stmt = (
                select(portfolio_snapshots.c.portfolio_value)
                .order_by(portfolio_snapshots.c.id.desc())
                .limit(1)
            )
            with self._connect() as conn:
                row = conn.execute(stmt).fetchone()
            if row is None:
                return 0.0
            return float(row[0])
        except Exception as exc:  # noqa: BLE001
            log.error("get_portfolio_value failed", error=str(exc))
            return 0.0

    # ==================================================================
    # Health check
    # ==================================================================

    def test_write(self) -> bool:
        """
        Smoke-test the database by writing a test key to ``system_state`` and
        reading it back.

        Returns
        -------
        bool
            ``True`` if the write+read round-trip succeeds and the value
            matches, ``False`` otherwise.
        """
        test_key = "__db_test__"
        test_val = f"ok:{_now_utc()}"
        try:
            ok = self.set_system_state(test_key, test_val)
            if not ok:
                return False
            retrieved = self.get_system_state(test_key)
            if retrieved != test_val:
                log.error(
                    "test_write round-trip mismatch",
                    expected=test_val,
                    got=retrieved,
                )
                return False
            log.debug("Database test_write passed")
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("test_write raised an exception", error=str(exc))
            return False

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:  # pragma: no cover
        return f"DatabaseRepository(engine={self._engine.url!r})"
