"""
Binance public WebSocket data feed (no authentication required).

Streams per symbol:
  <symbol>@depth20@100ms  — 20-level LOB snapshot at 100 ms cadence
  <symbol>@aggTrade        — aggregated trades
  <symbol>@kline_1m        — 1-minute OHLCV bars (published on close)

Uses the Binance combined-stream endpoint so a single WebSocket connection
carries all streams for all symbols.

Auto-reconnects with exponential backoff on any disconnect or error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from trading_system.data.store import MarketDataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"

BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0
BACKOFF_MULT = 2.0

# Canonical Binance symbols we track and their normalised asset names in the store
DEFAULT_SYMBOLS: Dict[str, str] = {
    "btcusdt": "BTCUSD",
    "ethusdt": "ETHUSD",
    "solusdt": "SOLUSD",
    "bnbusdt": "BNBUSD",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_stream_url(binance_symbols: List[str]) -> str:
    """
    Build the combined-stream WebSocket URL for the given list of lower-case
    Binance symbols (e.g. ['btcusdt', 'ethusdt']).
    """
    streams: List[str] = []
    for sym in binance_symbols:
        s = sym.lower()
        streams.append(f"{s}@depth20@100ms")
        streams.append(f"{s}@aggTrade")
        streams.append(f"{s}@kline_1m")
    return BINANCE_WS_BASE + "?streams=" + "/".join(streams)


# ---------------------------------------------------------------------------
# Main feed class
# ---------------------------------------------------------------------------

class BinanceFeed:
    """
    Binance public WebSocket feed.

    Subscribes to depth20, aggTrade, and kline_1m for each configured symbol.
    Updates the shared :class:`~trading_system.data.store.MarketDataStore` on
    every message.

    Usage::

        feed = BinanceFeed(store=store)
        await feed.start()   # runs indefinitely

    Parameters
    ----------
    store:
        The shared in-memory market-data store.
    symbol_map:
        Mapping from lower-case Binance symbol (e.g. ``"btcusdt"``) to the
        normalised asset key used in the store (e.g. ``"BTCUSD"``).
        Defaults to :data:`DEFAULT_SYMBOLS`.
    """

    def __init__(
        self,
        store: MarketDataStore,
        symbol_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.store = store
        self.symbol_map: Dict[str, str] = symbol_map if symbol_map is not None else dict(DEFAULT_SYMBOLS)
        self._running = False
        self._backoff = BACKOFF_BASE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run forever, reconnecting on any error or disconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_run()
                logger.warning("[binance] Connection closed cleanly; reconnecting…")
            except asyncio.CancelledError:
                logger.info("[binance] Cancelled, shutting down.")
                self._running = False
                return
            except Exception as exc:
                logger.error("[binance] Connection error: %s", exc, exc_info=True)

            if self._running:
                logger.info("[binance] Waiting %.1fs before reconnect…", self._backoff)
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * BACKOFF_MULT, BACKOFF_MAX)

    async def stop(self) -> None:
        """Signal the feed to stop after the current reconnect cycle."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_run(self) -> None:
        import websockets  # type: ignore

        symbols = list(self.symbol_map.keys())
        url = _build_stream_url(symbols)
        logger.info(
            "[binance] Connecting to combined stream for %d symbols: %s",
            len(symbols),
            url,
        )

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=2 ** 23,
        ) as ws:
            # Reset backoff on successful connection
            self._backoff = BACKOFF_BASE
            logger.info("[binance] Connected. Streaming %d symbols.", len(symbols))

            async for raw in ws:
                try:
                    envelope = json.loads(raw)
                    await self._dispatch(envelope)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("[binance] Dispatch error: %s | raw=%s", exc, raw[:200])

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, envelope: dict) -> None:
        """
        The combined stream wraps each payload as::

            {"stream": "<name>@<type>", "data": {...}}
        """
        stream: str = envelope.get("stream", "")
        data: dict = envelope.get("data", {})
        if not stream or not data:
            return

        # Derive the Binance symbol from the stream name (first segment)
        # e.g. "btcusdt@depth20@100ms" -> "btcusdt"
        parts = stream.split("@")
        binance_sym = parts[0].lower()
        asset = self.symbol_map.get(binance_sym)
        if asset is None:
            return  # unknown symbol – ignore

        stream_type = parts[1] if len(parts) > 1 else ""

        if stream_type == "depth20":
            await self._handle_depth(asset, data)
        elif stream_type == "aggTrade":
            await self._handle_agg_trade(asset, data)
        elif stream_type == "kline_1m" or stream_type == "kline":
            await self._handle_kline(asset, data)

    # ------------------------------------------------------------------
    # Handler: depth (LOB snapshot)
    # ------------------------------------------------------------------

    async def _handle_depth(self, asset: str, data: dict) -> None:
        """
        Parse a depth20 snapshot and push to store.

        Binance sends::

            {
              "lastUpdateId": 123,
              "bids": [["price_str", "qty_str"], ...],
              "asks": [["price_str", "qty_str"], ...]
            }
        """
        try:
            bids = [
                (float(level[0]), float(level[1]))
                for level in data.get("bids", [])
                if float(level[1]) > 0
            ]
            asks = [
                (float(level[0]), float(level[1]))
                for level in data.get("asks", [])
                if float(level[1]) > 0
            ]
            await self.store.update_lob(asset, bids, asks)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            logger.debug("[binance] Bad depth message for %s: %s", asset, exc)

    # ------------------------------------------------------------------
    # Handler: aggTrade
    # ------------------------------------------------------------------

    async def _handle_agg_trade(self, asset: str, data: dict) -> None:
        """
        Parse an aggregated-trade message.

        Binance sends::

            {
              "e": "aggTrade",
              "E": 1672531200000,   // event time (ms)
              "s": "BTCUSDT",
              "a": 26129,           // agg trade ID
              "p": "0.01633102",
              "q": "4.70443515",
              "f": 27781,           // first trade ID
              "l": 27781,           // last trade ID
              "T": 1672531200000,   // trade time (ms)
              "m": true             // is buyer the market maker?
            }
        """
        try:
            trade = {
                "price": float(data["p"]),
                "qty": float(data["q"]),
                "is_buyer_maker": bool(data.get("m", False)),
                "timestamp": float(data["T"]) / 1000.0,
            }
            await self.store.update_trade(asset, trade)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("[binance] Bad aggTrade for %s: %s", asset, exc)

    # ------------------------------------------------------------------
    # Handler: kline (OHLCV bar)
    # ------------------------------------------------------------------

    async def _handle_kline(self, asset: str, data: dict) -> None:
        """
        Parse a kline message.  Only emit a candle when the bar is *closed*
        (``kline.x == True``).

        Binance sends::

            {
              "e": "kline",
              "E": 123456789,
              "s": "BNBBTC",
              "k": {
                "t": 123400000,   // open time (ms)
                "T": 123460000,   // close time (ms)
                "s": "BNBBTC",
                "i": "1m",
                "o": "0.0010",
                "c": "0.0020",
                "h": "0.0025",
                "l": "0.0015",
                "v": "1000",
                "x": false        // is this kline closed?
              }
            }
        """
        try:
            k = data.get("k", {})
            # Update the live (open) bar in store on every tick
            ts_open = float(k["t"]) / 1000.0
            ohlcv = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "vwap": float(k.get("q", 0.0)) / float(k["v"]) if float(k["v"]) > 0 else float(k["c"]),
                "timestamp": ts_open,
            }
            # Always push live bar update; store deduplicates by timestamp
            await self.store.update_candle(asset, "1m", ohlcv)

            # On close, also update session VWAP accumulators
            if k.get("x", False):
                await self.store.update_vwap(asset, ohlcv["vwap"], ohlcv["volume"])

        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            logger.debug("[binance] Bad kline for %s: %s", asset, exc)
