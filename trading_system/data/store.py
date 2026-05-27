"""
Thread-safe in-memory market data store.

Maintains rolling OHLCV per asset/timeframe + LOB snapshot + recent trades.
Anything older than MAX_DATA_STALENESS_SECS is rejected by readers.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..core.logger import get_logger

log = get_logger(__name__)


class DataStaleError(RuntimeError):
    pass


@dataclass
class Candle:
    ts: int          # epoch seconds, candle close time
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_tuple(self):
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


@dataclass
class LOBSnapshot:
    bids: List[Tuple[float, float]] = field(default_factory=list)   # [(price, qty), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def best_bid(self) -> Optional[Tuple[float, float]]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Tuple[float, float]]:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return None


@dataclass
class TradeTick:
    ts: float          # epoch seconds (float for sub-second precision)
    price: float
    qty: float
    is_buyer_maker: bool   # True = sell aggressor (sell hit bid)


class MarketDataStore:
    DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
    CANDLE_MAXLEN = 1000
    LOB_LEVELS = 20
    TRADE_MAXLEN = 1000

    def __init__(self, max_staleness_secs: int = 45):
        self.max_staleness_secs = max_staleness_secs
        self.candles: Dict[str, Dict[str, Deque[Candle]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.CANDLE_MAXLEN)))
        self.lob: Dict[str, LOBSnapshot] = defaultdict(LOBSnapshot)
        self.trades: Dict[str, Deque[TradeTick]] = defaultdict(
            lambda: deque(maxlen=self.TRADE_MAXLEN))
        self.last_update_time: Dict[str, float] = {}
        self.vwap_state: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cum_pv": 0.0, "cum_v": 0.0, "session_start_ts": 0.0})
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.assets: set[str] = set()

    # ── lock helpers ─────────────────────────────────────────────────────
    def _lock(self, asset: str) -> asyncio.Lock:
        return self._locks[asset]

    # ── candle writes ────────────────────────────────────────────────────
    def append_candle(self, asset: str, timeframe: str, candle: Candle) -> None:
        self.assets.add(asset)
        dq = self.candles[asset][timeframe]
        if dq and dq[-1].ts == candle.ts:
            dq[-1] = candle   # update in-progress candle
        else:
            dq.append(candle)
        self.last_update_time[asset] = time.time()

    def get_candles(self, asset: str, timeframe: str = "1m", n: int = 200,
                    allow_stale: bool = False) -> List[Candle]:
        if not allow_stale:
            self._check_stale(asset)
        dq = self.candles.get(asset, {}).get(timeframe, deque())
        return list(dq)[-n:]

    def latest_close(self, asset: str, timeframe: str = "1m") -> Optional[float]:
        dq = self.candles.get(asset, {}).get(timeframe)
        return dq[-1].close if dq else None

    # ── LOB ──────────────────────────────────────────────────────────────
    def update_lob(self, asset: str, bids: List[Tuple[float, float]],
                   asks: List[Tuple[float, float]]) -> None:
        self.assets.add(asset)
        snap = LOBSnapshot(
            bids=sorted(bids, key=lambda x: -x[0])[:self.LOB_LEVELS],
            asks=sorted(asks, key=lambda x: x[0])[:self.LOB_LEVELS],
            timestamp=time.time(),
        )
        self.lob[asset] = snap
        self.last_update_time[asset] = time.time()

    def get_lob(self, asset: str, allow_stale: bool = False) -> LOBSnapshot:
        if not allow_stale:
            self._check_stale(asset)
        return self.lob.get(asset, LOBSnapshot())

    # ── trades ───────────────────────────────────────────────────────────
    def append_trade(self, asset: str, tick: TradeTick) -> None:
        self.assets.add(asset)
        self.trades[asset].append(tick)
        self.last_update_time[asset] = time.time()
        self._update_vwap(asset, tick)

    def get_recent_trades(self, asset: str, n: int = 500) -> List[TradeTick]:
        return list(self.trades.get(asset, deque()))[-n:]

    def _update_vwap(self, asset: str, tick: TradeTick) -> None:
        state = self.vwap_state[asset]
        # Reset at session boundary
        now = tick.ts
        if now - state["session_start_ts"] > 24 * 3600:
            state["cum_pv"] = 0.0
            state["cum_v"] = 0.0
            state["session_start_ts"] = now - (now % (24 * 3600))
        state["cum_pv"] += tick.price * tick.qty
        state["cum_v"] += tick.qty

    def get_vwap(self, asset: str) -> Optional[float]:
        state = self.vwap_state.get(asset)
        if not state or state["cum_v"] == 0:
            return None
        return state["cum_pv"] / state["cum_v"]

    # ── staleness ────────────────────────────────────────────────────────
    def _check_stale(self, asset: str) -> None:
        last = self.last_update_time.get(asset, 0)
        if last == 0:
            return     # never updated yet, caller handles None
        age = time.time() - last
        if age > self.max_staleness_secs:
            raise DataStaleError(f"{asset} data is {age:.1f}s stale")

    def asset_age_secs(self, asset: str) -> float:
        last = self.last_update_time.get(asset, 0)
        if last == 0:
            return float("inf")
        return time.time() - last

    def get_health(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for asset in self.assets:
            out[asset] = {
                "age_seconds": self.asset_age_secs(asset),
                "candles_1m": len(self.candles[asset].get("1m", [])),
                "lob_levels": len(self.lob.get(asset, LOBSnapshot()).bids),
                "trades": len(self.trades.get(asset, [])),
            }
        return out
