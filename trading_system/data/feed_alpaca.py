"""
Alpaca WebSocket feed for equities (IEX) and crypto.

Endpoints:
  wss://stream.data.alpaca.markets/v2/iex                   (equity free tier)
  wss://stream.data.alpaca.markets/v1beta3/crypto/us        (crypto)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional

from ..core.logger import get_logger
from .store import Candle, MarketDataStore, TradeTick

log = get_logger(__name__)

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None


EQUITY_WS = "wss://stream.data.alpaca.markets/v2/iex"
CRYPTO_WS = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"


class AlpacaFeed:
    """One feed per stream (equity or crypto). Auto-reconnect on failure."""

    def __init__(self, store: MarketDataStore, api_key: str, secret_key: str,
                 assets: List[str], stream: str = "equity"):
        self.store = store
        self.api_key = api_key
        self.secret_key = secret_key
        self.assets = [a.upper() for a in assets]
        self.stream = stream
        self.ws_url = EQUITY_WS if stream == "equity" else CRYPTO_WS
        self._stop = False

    async def run(self) -> None:
        if websockets is None:
            log.error("websockets package not installed - Alpaca feed disabled")
            return
        backoff = 1.0
        while not self._stop:
            try:
                await self._connect_and_stream()
                backoff = 1.0
            except Exception as e:
                log.warning(f"Alpaca {self.stream} feed dropped: {e} (retry in {backoff}s)")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect_and_stream(self) -> None:
        log.info(f"Connecting Alpaca {self.stream} feed: {self.ws_url}")
        async with websockets.connect(self.ws_url, ping_interval=20,
                                      ping_timeout=20) as ws:
            await ws.send(json.dumps({
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key,
            }))
            auth_resp = await ws.recv()
            log.debug(f"Auth resp: {auth_resp}")

            await ws.send(json.dumps({
                "action": "subscribe",
                "trades": self.assets,
                "quotes": self.assets,
                "bars": self.assets,
            }))
            sub_resp = await ws.recv()
            log.info(f"Alpaca {self.stream} subscribed: {self.assets}")

            async for raw in ws:
                try:
                    msgs = json.loads(raw)
                    if isinstance(msgs, list):
                        for m in msgs:
                            self._handle_message(m)
                    else:
                        self._handle_message(msgs)
                except Exception as e:
                    log.error(f"Alpaca msg parse error: {e}")

    def _handle_message(self, m: dict) -> None:
        msg_type = m.get("T") or m.get("type")
        sym = m.get("S") or m.get("symbol")
        if not sym:
            return
        if msg_type == "b":
            ts = _parse_ts(m.get("t"))
            cd = Candle(
                ts=int(ts),
                open=float(m["o"]),
                high=float(m["h"]),
                low=float(m["l"]),
                close=float(m["c"]),
                volume=float(m.get("v", 0)),
            )
            self.store.append_candle(sym, "1m", cd)
        elif msg_type == "t":
            tick = TradeTick(
                ts=_parse_ts(m.get("t")),
                price=float(m["p"]),
                qty=float(m["s"]),
                is_buyer_maker=False,   # Alpaca doesn't expose maker side
            )
            self.store.append_trade(sym, tick)
        elif msg_type == "q":
            bid_px = float(m.get("bp", 0))
            ask_px = float(m.get("ap", 0))
            bid_sz = float(m.get("bs", 0))
            ask_sz = float(m.get("as", 0))
            if bid_px and ask_px:
                self.store.update_lob(sym, [(bid_px, bid_sz)], [(ask_px, ask_sz)])

    def stop(self) -> None:
        self._stop = True


def _parse_ts(t) -> float:
    if t is None:
        return time.time()
    if isinstance(t, (int, float)):
        # nanoseconds → seconds
        if t > 1e15:
            return t / 1e9
        if t > 1e12:
            return t / 1e3
        return float(t)
    # ISO 8601 string
    try:
        import datetime as dt
        # Handle 'Z' suffix
        s = t.rstrip("Z")
        if "." in s:
            s = s[: s.index(".") + 7]   # truncate to microseconds
        d = dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    except Exception:
        return time.time()
