"""
CoinMetrics Community API — free on-chain BTC/ETH metrics.

Refresh interval: 20 min. Metrics returned per asset are listed below.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from ..core.logger import get_logger

log = get_logger(__name__)

ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

METRICS_BTC = [
    "AdrActCnt",         # active addresses
    "TxCnt",             # transaction count
    "FlowInExNtv",       # exchange inflow native
    "FlowOutExNtv",      # exchange outflow native
    "SplyAct1yr",        # supply active in last year
    "FeeTotNtv",         # total fees
    "VtyDayRet30d",      # 30d volatility of daily returns
]
METRICS_ETH = METRICS_BTC


class OnChainFeed:
    def __init__(self, refresh_secs: int = 1200):
        self.refresh_secs = refresh_secs
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.last_refresh = 0.0
        self._stop = False

    async def fetch(self, session: aiohttp.ClientSession, asset: str,
                    metrics: List[str], lookback_days: int = 120
                    ) -> Optional[pd.DataFrame]:
        end = pd.Timestamp.utcnow().normalize()
        start = (end - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        params = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": start,
            "end_time": end.strftime("%Y-%m-%d"),
            "page_size": "500",
        }
        try:
            async with session.get(ENDPOINT, params=params, timeout=20) as r:
                if r.status != 200:
                    log.warning(f"CoinMetrics {asset} returned {r.status}")
                    return None
                payload = await r.json()
        except Exception as e:
            log.warning(f"CoinMetrics fetch error {asset}: {e}")
            return None

        rows = payload.get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["time"])
        for m in metrics:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")
        df = df.set_index("date").sort_index()
        return df

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self._stop:
                try:
                    btc = await self.fetch(session, "btc", METRICS_BTC)
                    eth = await self.fetch(session, "eth", METRICS_ETH)
                    if btc is not None:
                        self.dataframes["BTC"] = btc
                    if eth is not None:
                        self.dataframes["ETH"] = eth
                    self.last_refresh = time.time()
                    log.info(f"OnChain refresh: BTC={btc is not None} ETH={eth is not None}")
                except Exception as e:
                    log.error(f"OnChain refresh error: {e}")
                await asyncio.sleep(self.refresh_secs)

    def snapshot(self) -> Dict[str, pd.DataFrame]:
        return dict(self.dataframes)

    def stop(self) -> None:
        self._stop = True
