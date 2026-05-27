"""
Alpaca WebSocket data feed.

Connects to:
  - wss://stream.data.alpaca.markets/v2/iex          (equities)
  - wss://stream.data.alpaca.markets/v1beta3/crypto/us (crypto)

Subscribes to trades, quotes, and 1-minute bars for all assets in the universe.
Updates the shared MarketDataStore on every message.
Auto-reconnects with exponential backoff on any disconnect or error.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Set

from trading_system.data.store import MarketDataStore

logger = logging.getLogger(__name__)

# Reconnect settings
BACKOFF_BASE = 1.0          # seconds
BACKOFF_MAX = 60.0          # seconds
BACKOFF_MULTIPLIER = 2.0

# Alpaca WS endpoints
EQUITY_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
CRYPTO_WS_URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"

# Asset classification helpers
_CRYPTO_SUFFIXES = ("/USD", "/USDT", "/BTC", "/ETH")
_CRYPTO_COINS = {"BTC", "ETH", "SOL", "BNB", "DOGE", "ADA", "AVAX", "MATIC"}


def _is_crypto(symbol: str) -> bool:
    upper = symbol.upper()
    if upper in _CRYPTO_COINS:
        return True
    for sfx in _CRYPTO_SUFFIXES:
        if upper.endswith(sfx):
            return True
    return False


def _normalise_symbol(symbol: str) -> str:
    """Return a canonical asset key (e.g. 'BTC/USD' -> 'BTCUSD')."""
    return symbol.replace("/", "").upper()


class _AlpacaStreamWorker:
    """
    Manages a single authenticated Alpaca WebSocket connection for one endpoint
    (either equity or crypto).  Subscribes to bars, trades, and quotes.
    Parses incoming messages and writes to the MarketDataStore.
    Reconnects automatically on failure.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        store: MarketDataStore,
        worker_name: str = "alpaca",
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.store = store
        self.worker_name = worker_name
        self._running = False
        self._backoff = BACKOFF_BASE

    async def start(self) -> None:
        """Run forever, reconnecting on any error."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_run()
                # If _connect_and_run exits cleanly (shouldn't happen), wait before retry
                logger.warning("[%s] Connection closed cleanly, reconnecting…", self.worker_name)
            except asyncio.CancelledError:
                logger.info("[%s] Cancelled, shutting down.", self.worker_name)
                self._running = False
                return
            except Exception as exc:
                logger.error("[%s] Connection error: %s", self.worker_name, exc, exc_info=True)

            if self._running:
                logger.info(
                    "[%s] Waiting %.1fs before reconnect…", self.worker_name, self._backoff
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX)

    async def stop(self) -> None:
        self._running = False

    async def _connect_and_run(self) -> None:
        """Open the WebSocket, authenticate, subscribe, and process messages."""
        # Import here to avoid hard dependency at module load time
        import websockets  # type: ignore

        logger.info("[%s] Connecting to %s", self.worker_name, self.url)
        async with websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=2 ** 23,  # 8 MB
        ) as ws:
            # Reset backoff on successful connection
            self._backoff = BACKOFF_BASE

            # Step 1: receive connection confirmation
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            msg = self._parse(raw)
            if not self._check_connected(msg):
                raise ConnectionError(f"[{self.worker_name}] Unexpected greeting: {msg}")

            # Step 2: authenticate
            await ws.send(self._auth_payload())
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            msg = self._parse(raw)
            if not self._check_authenticated(msg):
                raise ConnectionError(f"[{self.worker_name}] Auth failed: {msg}")

            # Step 3: subscribe
            await ws.send(self._subscribe_payload())
            logger.info(
                "[%s] Subscribed to %d symbols (bars, trades, quotes)",
                self.worker_name,
                len(self.symbols),
            )

            # Step 4: message loop
            async for raw in ws:
                try:
                    await self._dispatch(self._parse(raw))
                except Exception as exc:
                    logger.debug("[%s] Dispatch error: %s", self.worker_name, exc)

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw) -> list:
        import json
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        return data  # Alpaca sends arrays of messages

    def _auth_payload(self) -> str:
        import json
        return json.dumps({"action": "auth", "key": self.api_key, "secret": self.api_secret})

    def _subscribe_payload(self) -> str:
        import json
        return json.dumps({
            "action": "subscribe",
            "bars": self.symbols,
            "trades": self.symbols,
            "quotes": self.symbols,
        })

    @staticmethod
    def _check_connected(msgs: list) -> bool:
        for m in msgs:
            if isinstance(m, dict) and m.get("T") == "success" and m.get("msg") == "connected":
                return True
        return False

    @staticmethod
    def _check_authenticated(msgs: list) -> bool:
        for m in msgs:
            if isinstance(m, dict) and m.get("T") == "success" and m.get("msg") == "authenticated":
                return True
        return False

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msgs: list) -> None:
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            t = msg.get("T")
            if t == "b":
                await self._handle_bar(msg)
            elif t == "t":
                await self._handle_trade(msg)
            elif t == "q":
                await self._handle_quote(msg)
            elif t == "error":
                logger.warning("[%s] Server error: %s", self.worker_name, msg)
            # subscription confirmations and heartbeats are silently ignored

    async def _handle_bar(self, msg: dict) -> None:
        """Process a bar (1m OHLCV) message."""
        symbol = _normalise_symbol(msg.get("S", ""))
        if not symbol:
            return
        try:
            ts = self._parse_timestamp(msg.get("t"))
            ohlcv = {
                "open": float(msg["o"]),
                "high": float(msg["h"]),
                "low": float(msg["l"]),
                "close": float(msg["c"]),
                "volume": float(msg["v"]),
                "vwap": float(msg.get("vw", 0.0)),
                "timestamp": ts,
            }
            await self.store.update_candle(symbol, "1m", ohlcv)
            # Also push a VWAP update from bar data
            await self.store.update_vwap(symbol, ohlcv["vwap"] or ohlcv["close"], ohlcv["volume"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("[%s] Bad bar message %s: %s", self.worker_name, msg, exc)

    async def _handle_trade(self, msg: dict) -> None:
        """Process a trade tick message."""
        symbol = _normalise_symbol(msg.get("S", ""))
        if not symbol:
            return
        try:
            ts = self._parse_timestamp(msg.get("t"))
            trade = {
                "price": float(msg["p"]),
                "qty": float(msg["s"]),
                "is_buyer_maker": False,  # Alpaca doesn't expose this; default to False
                "timestamp": ts,
                "conditions": msg.get("c", []),
                "exchange": msg.get("x", ""),
            }
            await self.store.update_trade(symbol, trade)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("[%s] Bad trade message %s: %s", self.worker_name, msg, exc)

    async def _handle_quote(self, msg: dict) -> None:
        """
        Process a quote (NBBO) message and update the LOB with a 1-level snapshot.
        Full depth is obtained from Binance for crypto; Alpaca quotes give NBBO.
        """
        symbol = _normalise_symbol(msg.get("S", ""))
        if not symbol:
            return
        try:
            bid_price = float(msg.get("bp", 0.0))
            ask_price = float(msg.get("ap", 0.0))
            bid_size = float(msg.get("bs", 0.0))
            ask_size = float(msg.get("as", 0.0))
            if bid_price > 0 and ask_price > 0:
                await self.store.update_lob(
                    symbol,
                    bids=[(bid_price, bid_size)],
                    asks=[(ask_price, ask_size)],
                )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("[%s] Bad quote message %s: %s", self.worker_name, msg, exc)

    @staticmethod
    def _parse_timestamp(ts_str: Optional[str]) -> float:
        """Parse an RFC3339 timestamp string to a Unix timestamp float."""
        if ts_str is None:
            return time.time()
        from datetime import datetime, timezone
        # Strip trailing 'Z' and handle offset
        ts_str = ts_str.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ts_str)
            return dt.timestamp()
        except ValueError:
            return time.time()


class AlpacaFeed:
    """
    Top-level Alpaca feed that spawns two workers:
      - one for equities (IEX feed)
      - one for crypto (crypto/us feed)

    Usage::

        feed = AlpacaFeed(store=store, equity_symbols=['AAPL', 'TSLA'], crypto_symbols=['BTC/USD'])
        await feed.start()   # runs forever
    """

    def __init__(
        self,
        store: MarketDataStore,
        equity_symbols: Optional[List[str]] = None,
        crypto_symbols: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self.store = store
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")

        self.equity_symbols: List[str] = equity_symbols or []
        self.crypto_symbols: List[str] = crypto_symbols or []

        if not self.api_key or not self.api_secret:
            logger.warning(
                "AlpacaFeed: ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
                "WebSocket connections will fail authentication."
            )

        self._workers: List[_AlpacaStreamWorker] = []
        self._tasks: List[asyncio.Task] = []

    def add_symbols(self, symbols: List[str]) -> None:
        """
        Classify and add symbols to the appropriate list before calling start().
        Symbols containing '/' or matching known crypto tickers go to crypto;
        everything else goes to equities.
        """
        for sym in symbols:
            if _is_crypto(sym):
                if sym not in self.crypto_symbols:
                    self.crypto_symbols.append(sym)
            else:
                if sym not in self.equity_symbols:
                    self.equity_symbols.append(sym)

    async def start(self) -> None:
        """
        Start both equity and crypto streams.  Runs until cancelled.
        """
        tasks = []

        if self.equity_symbols:
            eq_worker = _AlpacaStreamWorker(
                url=EQUITY_WS_URL,
                api_key=self.api_key,
                api_secret=self.api_secret,
                symbols=self.equity_symbols,
                store=self.store,
                worker_name="alpaca-equity",
            )
            self._workers.append(eq_worker)
            tasks.append(asyncio.create_task(eq_worker.start(), name="alpaca-equity"))

        if self.crypto_symbols:
            cr_worker = _AlpacaStreamWorker(
                url=CRYPTO_WS_URL,
                api_key=self.api_key,
                api_secret=self.api_secret,
                symbols=self.crypto_symbols,
                store=self.store,
                worker_name="alpaca-crypto",
            )
            self._workers.append(cr_worker)
            tasks.append(asyncio.create_task(cr_worker.start(), name="alpaca-crypto"))

        self._tasks = tasks

        if not tasks:
            logger.warning("AlpacaFeed.start(): no symbols configured, nothing to do.")
            return

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("AlpacaFeed: shutting down all workers.")
            for w in self._workers:
                await w.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Gracefully stop all workers."""
        for w in self._workers:
            await w.stop()
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("AlpacaFeed stopped.")
