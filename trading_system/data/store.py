"""
Thread-safe in-memory MarketDataStore backed by asyncio locks.
Stores candles, LOB snapshots, recent trades, and session VWAP for all assets.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STALENESS_THRESHOLD_SECONDS = 45.0


class DataStaleError(Exception):
    """Raised when requested data has not been updated within the staleness threshold."""
    pass


class MarketDataStore:
    """
    Central in-memory store for all market data.

    Data layout:
      candles[asset][timeframe]  -> deque(maxlen=1000) of OHLCV dicts
      lob[asset]                 -> {'bids': [(price, qty)], 'asks': [(price, qty)], 'timestamp': float}
      trades[asset]              -> deque(maxlen=500) of trade dicts
      vwap_session[asset]        -> {'cum_tp_vol': float, 'cum_vol': float, 'vwap': float}
      last_update_time[asset]    -> float (unix timestamp)
    """

    def __init__(self) -> None:
        # candles[asset][timeframe] = deque of OHLCV dicts
        self._candles: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))

        # lob[asset] = {'bids': list[tuple], 'asks': list[tuple], 'timestamp': float}
        self._lob: Dict[str, dict] = {}

        # trades[asset] = deque of trade dicts
        self._trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

        # session VWAP accumulator
        self._vwap_session: Dict[str, dict] = defaultdict(lambda: {
            'cum_tp_vol': 0.0,
            'cum_vol': 0.0,
            'vwap': 0.0,
        })

        # last update timestamp per asset
        self._last_update_time: Dict[str, float] = {}

        # per-asset asyncio locks to serialise concurrent writes
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # global lock for structural changes (e.g. adding new assets)
        self._global_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _touch(self, asset: str) -> None:
        """Record the current time as the last update time for an asset."""
        self._last_update_time[asset] = time.time()

    def _check_staleness(self, asset: str) -> None:
        """Raise DataStaleError if the asset has not been updated recently."""
        last = self._last_update_time.get(asset)
        if last is None:
            raise DataStaleError(f"No data ever received for asset '{asset}'")
        age = time.time() - last
        if age > STALENESS_THRESHOLD_SECONDS:
            raise DataStaleError(
                f"Data for '{asset}' is {age:.1f}s stale (threshold={STALENESS_THRESHOLD_SECONDS}s)"
            )

    # ------------------------------------------------------------------ #
    #  Write methods (all async, acquire per-asset lock)                  #
    # ------------------------------------------------------------------ #

    async def update_candle(self, asset: str, timeframe: str, ohlcv: dict) -> None:
        """
        Append or update the latest candle for (asset, timeframe).

        ohlcv must contain keys: open, high, low, close, volume, vwap, timestamp
        If a candle with the same timestamp already exists at the tail, it is
        replaced (in-place bar update).  Otherwise a new bar is appended.
        """
        required = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        missing = required - ohlcv.keys()
        if missing:
            logger.warning("update_candle: missing keys %s for %s/%s", missing, asset, timeframe)
            return

        async with self._locks[asset]:
            dq = self._candles[asset][timeframe]
            if dq and dq[-1]['timestamp'] == ohlcv['timestamp']:
                # Replace tail (live bar update)
                dq[-1] = ohlcv
            else:
                dq.append(ohlcv)
            self._touch(asset)

    async def update_lob(
        self,
        asset: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        """
        Replace the full LOB snapshot for an asset.

        Bids and asks are lists of (price, quantity) tuples.
        Bids are stored best-bid first (descending price).
        Asks are stored best-ask first (ascending price).
        """
        async with self._locks[asset]:
            self._lob[asset] = {
                'bids': sorted(bids, key=lambda x: x[0], reverse=True),
                'asks': sorted(asks, key=lambda x: x[0]),
                'timestamp': time.time(),
            }
            self._touch(asset)

    async def update_trade(self, asset: str, trade_dict: dict) -> None:
        """
        Append a single trade to the ring buffer.

        trade_dict must contain: price, qty, is_buyer_maker, timestamp
        Also updates the session VWAP incrementally.
        """
        required = {'price', 'qty', 'timestamp'}
        missing = required - trade_dict.keys()
        if missing:
            logger.warning("update_trade: missing keys %s for %s", missing, asset)
            return

        async with self._locks[asset]:
            self._trades[asset].append(trade_dict)
            # Incremental VWAP update
            price = float(trade_dict['price'])
            qty = float(trade_dict['qty'])
            self._vwap_session[asset]['cum_tp_vol'] += price * qty
            self._vwap_session[asset]['cum_vol'] += qty
            cum_vol = self._vwap_session[asset]['cum_vol']
            if cum_vol > 0:
                self._vwap_session[asset]['vwap'] = (
                    self._vwap_session[asset]['cum_tp_vol'] / cum_vol
                )
            self._touch(asset)

    async def update_vwap(self, asset: str, price: float, volume: float) -> None:
        """
        Manually update session VWAP accumulators (e.g. from bar data).
        Called when trade-level data is unavailable.
        """
        async with self._locks[asset]:
            self._vwap_session[asset]['cum_tp_vol'] += price * volume
            self._vwap_session[asset]['cum_vol'] += volume
            cum_vol = self._vwap_session[asset]['cum_vol']
            if cum_vol > 0:
                self._vwap_session[asset]['vwap'] = (
                    self._vwap_session[asset]['cum_tp_vol'] / cum_vol
                )
            self._touch(asset)

    # ------------------------------------------------------------------ #
    #  Read methods (return copies, check staleness)                      #
    # ------------------------------------------------------------------ #

    def get_candles(self, asset: str, timeframe: str, n: int = 100) -> List[dict]:
        """
        Return the last *n* closed candles for (asset, timeframe).

        Raises DataStaleError if the asset data is older than STALENESS_THRESHOLD_SECONDS.
        Returns a list ordered oldest-first.
        """
        self._check_staleness(asset)
        dq = self._candles[asset][timeframe]
        result = list(dq)
        return result[-n:] if len(result) > n else result

    def get_lob(self, asset: str) -> dict:
        """
        Return the latest LOB snapshot for an asset.

        Raises DataStaleError if stale.
        Returns {'bids': [...], 'asks': [...], 'timestamp': float} or empty dict.
        """
        self._check_staleness(asset)
        lob = self._lob.get(asset)
        if lob is None:
            return {'bids': [], 'asks': [], 'timestamp': 0.0}
        # Return a shallow copy to prevent external mutation
        return {
            'bids': list(lob['bids']),
            'asks': list(lob['asks']),
            'timestamp': lob['timestamp'],
        }

    def get_recent_trades(self, asset: str, n: int = 100) -> List[dict]:
        """
        Return the last *n* trades for an asset, oldest-first.

        Raises DataStaleError if stale.
        """
        self._check_staleness(asset)
        dq = self._trades[asset]
        result = list(dq)
        return result[-n:] if len(result) > n else result

    def get_vwap(self, asset: str) -> float:
        """
        Return the current session VWAP for an asset.

        Returns 0.0 if no trades have been recorded.
        """
        return self._vwap_session[asset]['vwap']

    def get_health(self) -> dict:
        """
        Return a health snapshot for all assets that have received data.

        Returns::

            {
                asset: {
                    'age_seconds': float,
                    'has_lob': bool,
                    'candle_counts': {timeframe: int},
                    'trade_count': int,
                    'vwap': float,
                }
            }
        """
        now = time.time()
        health = {}
        for asset, last in self._last_update_time.items():
            health[asset] = {
                'age_seconds': now - last,
                'has_lob': asset in self._lob,
                'candle_counts': {
                    tf: len(dq)
                    for tf, dq in self._candles[asset].items()
                },
                'trade_count': len(self._trades[asset]),
                'vwap': self._vwap_session[asset]['vwap'],
            }
        return health

    async def reset_session_vwap(self, asset: str) -> None:
        """
        Reset the session VWAP accumulators for an asset.
        Call at market open or session reset.
        """
        async with self._locks[asset]:
            self._vwap_session[asset] = {
                'cum_tp_vol': 0.0,
                'cum_vol': 0.0,
                'vwap': 0.0,
            }
            logger.info("Session VWAP reset for %s", asset)

    # ------------------------------------------------------------------ #
    #  Convenience / bulk helpers                                          #
    # ------------------------------------------------------------------ #

    def all_assets(self) -> List[str]:
        """Return the list of all assets that have received at least one update."""
        return list(self._last_update_time.keys())

    def get_mid_price(self, asset: str) -> Optional[float]:
        """Return the current mid price from the LOB, or None if unavailable."""
        try:
            lob = self.get_lob(asset)
            if lob['bids'] and lob['asks']:
                return (lob['bids'][0][0] + lob['asks'][0][0]) / 2.0
        except DataStaleError:
            pass
        return None

    def get_spread(self, asset: str) -> Optional[float]:
        """Return the current bid-ask spread, or None if unavailable."""
        try:
            lob = self.get_lob(asset)
            if lob['bids'] and lob['asks']:
                return lob['asks'][0][0] - lob['bids'][0][0]
        except DataStaleError:
            pass
        return None
