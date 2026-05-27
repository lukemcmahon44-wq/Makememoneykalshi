"""
Binance public WebSocket feed - no auth needed.

Streams subscribed:
  <sym>@bookTicker      best bid/ask
  <sym>@depth20@100ms   top 20 LOB levels every 100 ms
  <sym>@aggTrade        every aggregated trade with isBuyerMaker
  <sym>@kline_1m        1m OHLCV
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List

from ..core.logger import get_logger
from .store import Candle, MarketDataStore, TradeTick

log = get_logger(__name__)

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None


BASE_WS = "wss://stream.binance.com:9443/stream?streams="


class BinanceFeed:
    """Combined-stream client for multiple Binance symbols."""

    def __init__(self, store: MarketDataStore, assets: List[str]):
        self.store = store
        self.assets = [a.lower() for a in assets]
        self._stop = False
        self._build_stream_url()

    def _build_stream_url(self) -> None:
        parts: List[str] = []
        for a in self.assets:
            parts.append(f"{a}@kline_1m")
            parts.append(f"{a}@aggTrade")
            parts.append(f"{a}@depth20@100ms")
            parts.append(f"{a}@bookTicker")
        self.url = BASE_WS + "/".join(parts)

    async def run(self) -> None:
        if websockets is None:
            log.error("websockets package missing - Binance feed disabled")
            return
        backoff = 1.0
        while not self._stop:
            try:
                await self._connect()
                backoff = 1.0
            except Exception as e:
                log.warning(f"Binance feed dropped: {e} (retry in {backoff}s)")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect(self) -> None:
        log.info(f"Connecting Binance feed for {self.assets}")
        async with websockets.connect(self.url, ping_interval=20,
                                      ping_timeout=20, max_size=2**22) as ws:
            async for raw in ws:
                try:
                    payload = json.loads(raw)
                    self._handle(payload)
                except Exception as e:
                    log.error(f"Binance parse error: {e}")

    def _handle(self, payload: dict) -> None:
        stream = payload.get("stream", "")
        data = payload.get("data", {})
        if not stream or not data:
            return

        sym_lower, _, suffix = stream.partition("@")
        sym = sym_lower.upper()

        if suffix.startswith("kline"):
            k = data.get("k", {})
            cd = Candle(
                ts=int(k.get("t", time.time() * 1000) / 1000),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
            )
            self.store.append_candle(sym, "1m", cd)
        elif suffix == "aggTrade":
            tick = TradeTick(
                ts=float(data.get("T", time.time() * 1000)) / 1000,
                price=float(data["p"]),
                qty=float(data["q"]),
                is_buyer_maker=bool(data.get("m", False)),
            )
            self.store.append_trade(sym, tick)
        elif suffix.startswith("depth"):
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
            if bids and asks:
                self.store.update_lob(sym, bids, asks)
        elif suffix == "bookTicker":
            try:
                bid_px = float(data["b"])
                bid_sz = float(data["B"])
                ask_px = float(data["a"])
                ask_sz = float(data["A"])
            except (KeyError, ValueError):
                return
            existing = self.store.get_lob(sym, allow_stale=True)
            if not existing.bids:
                self.store.update_lob(sym, [(bid_px, bid_sz)], [(ask_px, ask_sz)])

    def stop(self) -> None:
        self._stop = True
