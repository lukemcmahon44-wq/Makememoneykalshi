"""
FRED API client for macro indicators.

Refreshes every 15 minutes. Caches results in memory.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp

from ..core.logger import get_logger

log = get_logger(__name__)


FRED_SERIES = {
    "VIXCLS": "VIXCLS",
    "FEDFUNDS": "FEDFUNDS",
    "DGS10": "DGS10",
    "T10Y2Y": "T10Y2Y",
    "DXY": "DTWEXBGS",
    "UNRATE": "UNRATE",
    "CPIAUCSL": "CPIAUCSL",
    "T5YIE": "T5YIE",
}

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDFeed:
    def __init__(self, api_key: str, refresh_secs: int = 900):
        self.api_key = api_key
        self.refresh_secs = refresh_secs
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._stop = False

    async def fetch_series(self, session: aiohttp.ClientSession, series_id: str
                           ) -> Optional[float]:
        if not self.api_key:
            return None
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            async with session.get(BASE_URL, params=params, timeout=10) as r:
                if r.status != 200:
                    log.warning(f"FRED {series_id} returned {r.status}")
                    return None
                data = await r.json()
                obs = data.get("observations", [])
                if not obs:
                    return None
                val_str = obs[0].get("value", ".")
                if val_str in (".", ""):
                    return None
                return float(val_str)
        except Exception as e:
            log.warning(f"FRED fetch error {series_id}: {e}")
            return None

    async def run(self) -> None:
        if not self.api_key:
            log.warning("FRED_API_KEY not set - macro feed disabled")
            return
        async with aiohttp.ClientSession() as session:
            while not self._stop:
                try:
                    await self._refresh_all(session)
                except Exception as e:
                    log.error(f"FRED refresh error: {e}")
                await asyncio.sleep(self.refresh_secs)

    async def _refresh_all(self, session: aiohttp.ClientSession) -> None:
        results: Dict[str, Any] = {}
        for label, series_id in FRED_SERIES.items():
            val = await self.fetch_series(session, series_id)
            if val is not None:
                results[label] = val
            await asyncio.sleep(0.5)   # respect rate limits
        if results:
            self.cache.update(results)
            self.cache["last_refresh"] = time.time()
            log.info(f"FRED refresh: {len(results)} metrics updated")

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.cache)

    def stop(self) -> None:
        self._stop = True
