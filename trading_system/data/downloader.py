"""
Historical data bootstrap. Uses yfinance for OHLCV history.

Used to populate the in-memory store + DB with enough history for
indicators to be valid on startup (200 bars minimum per timeframe).
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import pandas as pd

from ..core.logger import get_logger
from .store import Candle, MarketDataStore

log = get_logger(__name__)


_YF_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "60m",       # yfinance has no 4h; resample below
    "1d": "1d",
}

_YF_PERIOD = {
    "1m": "5d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "5y",
}


def _yf_symbol(asset: str) -> str:
    """Convert internal symbol to yfinance symbol."""
    if asset.endswith("USD") and len(asset) == 6:
        return f"{asset[:3]}-USD"
    if asset.endswith("USDT"):
        return f"{asset[:-4]}-USD"
    return asset


def fetch_history_sync(asset: str, timeframe: str = "1m"
                       ) -> Optional[pd.DataFrame]:
    """Synchronous historical fetch using yfinance."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        log.error("yfinance not installed - history bootstrap disabled")
        return None

    symbol = _yf_symbol(asset)
    period = _YF_PERIOD.get(timeframe, "60d")
    interval = _YF_INTERVAL.get(timeframe, "1m")

    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=False, prepost=False,
                         threads=False)
        if df is None or df.empty:
            return None
        # Flatten potential MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.lower)
        # Resample 4h from 1h
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        log.warning(f"yfinance fetch failed for {symbol}@{timeframe}: {e}")
        return None


async def bootstrap_history(store: MarketDataStore, assets: List[str],
                            timeframes: List[str] = ("1m", "5m", "15m", "1h", "1d"),
                            bars_required: int = 250) -> Dict[str, Dict[str, int]]:
    """Pull history for each asset+timeframe and populate the store."""
    loaded: Dict[str, Dict[str, int]] = {}
    loop = asyncio.get_event_loop()
    for asset in assets:
        loaded[asset] = {}
        for tf in timeframes:
            df = await loop.run_in_executor(None, fetch_history_sync, asset, tf)
            if df is None or df.empty:
                loaded[asset][tf] = 0
                continue
            count = 0
            for ts, row in df.iterrows():
                try:
                    cd = Candle(
                        ts=int(ts.timestamp()),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0) or 0),
                    )
                    store.append_candle(asset, tf, cd)
                    count += 1
                except (KeyError, ValueError):
                    continue
            loaded[asset][tf] = count
            log.info(f"Bootstrap {asset}@{tf}: {count} bars loaded")
    return loaded
