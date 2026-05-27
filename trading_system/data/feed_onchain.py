"""
CoinMetrics community API on-chain data feed.

Base URL: https://community-api.coinmetrics.io/v4/timeseries/asset-metrics

Fetches the last 90 days of daily on-chain metrics for BTC and ETH and
exposes the data as pandas DataFrames.  Refreshes every 20 minutes.

Metrics fetched
---------------
  AdrActCnt       — Active addresses
  TxCnt           — Transaction count
  FlowInExNtv     — Exchange inflows (native units)
  FlowOutExNtv    — Exchange outflows (native units)
  SplyAct1yr      — Supply active in last 1 year
  FeeTotNtv       — Total fees (native units)
  VtyDayRet30d    — 30-day daily-return volatility
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COINMETRICS_BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

ASSETS = ["btc", "eth"]

METRICS = [
    "AdrActCnt",
    "TxCnt",
    "FlowInExNtv",
    "FlowOutExNtv",
    "SplyAct1yr",
    "FeeTotNtv",
    "VtyDayRet30d",
]

HISTORY_DAYS = 90
REFRESH_INTERVAL_SECONDS = 1200  # 20 minutes
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class OnChainDataStore:
    """
    Thread-safe (asyncio) container for on-chain DataFrames.

    Attributes
    ----------
    frames : Dict[str, pd.DataFrame]
        ``{asset_name: DataFrame}`` — asset name is ``"btc"`` or ``"eth"``.
    last_updated : Dict[str, float]
        Unix timestamp of the last successful refresh per asset.
    """

    def __init__(self) -> None:
        self.frames: Dict[str, Optional[pd.DataFrame]] = {a: None for a in ASSETS}
        self.last_updated: Dict[str, float] = {a: 0.0 for a in ASSETS}
        self._lock = asyncio.Lock()

    async def set(self, asset: str, df: pd.DataFrame) -> None:
        async with self._lock:
            self.frames[asset] = df
            self.last_updated[asset] = time.time()

    def get(self, asset: str) -> Optional[pd.DataFrame]:
        """Return the DataFrame for *asset* (or ``None`` if not yet loaded)."""
        return self.frames.get(asset)

    def age_seconds(self, asset: str) -> Optional[float]:
        ts = self.last_updated.get(asset, 0.0)
        return time.time() - ts if ts else None


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

class OnChainFeed:
    """
    CoinMetrics community API feed.

    Parameters
    ----------
    store :
        Shared :class:`OnChainDataStore`.  A new one is created if *None*.
    assets :
        List of CoinMetrics asset identifiers to fetch (default: btc, eth).
    metrics :
        List of metric names to fetch (default: the seven listed above).
    history_days :
        How many calendar days of history to fetch (default 90).
    refresh_interval :
        Seconds between full refreshes (default 1200 = 20 min).
    """

    def __init__(
        self,
        store: Optional[OnChainDataStore] = None,
        assets: Optional[list] = None,
        metrics: Optional[list] = None,
        history_days: int = HISTORY_DAYS,
        refresh_interval: int = REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self.store = store or OnChainDataStore()
        self.assets = assets or ASSETS
        self.metrics = metrics or METRICS
        self.history_days = history_days
        self.refresh_interval = refresh_interval
        self._running = False

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_btc_data(self) -> Optional[pd.DataFrame]:
        """Return the latest BTC on-chain DataFrame, or ``None`` if not loaded."""
        return self.store.get("btc")

    def get_eth_data(self) -> Optional[pd.DataFrame]:
        """Return the latest ETH on-chain DataFrame, or ``None`` if not loaded."""
        return self.store.get("eth")

    def get_asset_data(self, asset: str) -> Optional[pd.DataFrame]:
        """Return data for the given asset string (e.g. 'btc' or 'eth')."""
        return self.store.get(asset.lower())

    # ------------------------------------------------------------------
    # Refresh lifecycle
    # ------------------------------------------------------------------

    async def refresh_once(self) -> None:
        """Fetch fresh data for all assets and update the store."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        ) as session:
            tasks = [self._fetch_asset(session, asset) for asset in self.assets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for asset, result in zip(self.assets, results):
            if isinstance(result, Exception):
                logger.error(
                    "[onchain] Failed to refresh %s: %s", asset, result, exc_info=False
                )
            elif result is not None and not result.empty:
                await self.store.set(asset, result)
                logger.info(
                    "[onchain] Refreshed %s: %d rows, cols=%s",
                    asset, len(result), list(result.columns),
                )

    async def refresh_loop(self) -> None:
        """
        Continuously call :meth:`refresh_once` every :attr:`refresh_interval`
        seconds.  Runs until cancelled.
        """
        self._running = True
        while self._running:
            await self.refresh_once()
            try:
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                logger.info("[onchain] Refresh loop cancelled.")
                self._running = False
                return

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal fetch helpers
    # ------------------------------------------------------------------

    async def _fetch_asset(
        self, session: aiohttp.ClientSession, asset: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch all metrics for one asset with retry logic.
        Returns a DataFrame indexed by datetime.
        """
        end_date = datetime.now(tz=timezone.utc)
        start_date = end_date - timedelta(days=self.history_days)

        params = {
            "assets": asset,
            "metrics": ",".join(self.metrics),
            "start_time": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frequency": "1d",
            "page_size": "10000",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(COINMETRICS_BASE_URL, params=params) as resp:
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", RETRY_BACKOFF * attempt))
                        logger.warning("[onchain] Rate-limited; sleeping %.1fs", retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    payload = await resp.json()
                return self._parse_response(payload, asset)
            except asyncio.CancelledError:
                raise
            except aiohttp.ClientResponseError as exc:
                logger.warning(
                    "[onchain] HTTP %d for %s (attempt %d/%d): %s",
                    exc.status, asset, attempt, MAX_RETRIES, exc.message,
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(
                    "[onchain] Request error for %s (attempt %d/%d): %s",
                    asset, attempt, MAX_RETRIES, exc,
                )
                last_exc = exc

            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF * attempt)

        raise last_exc or RuntimeError(f"[onchain] All retries exhausted for {asset}")

    @staticmethod
    def _parse_response(payload: dict, asset: str) -> pd.DataFrame:
        """
        Convert the CoinMetrics v4 JSON response to a pandas DataFrame.

        Expected payload structure::

            {
              "data": [
                {"asset": "btc", "time": "2024-01-01T00:00:00.000000000Z",
                 "AdrActCnt": "123456", ...},
                ...
              ]
            }
        """
        data = payload.get("data", [])
        if not data:
            logger.warning("[onchain] Empty data payload for %s", asset)
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Parse the time column to a DatetimeIndex
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"])
            df = df.set_index("time").sort_index()
        else:
            logger.warning("[onchain] No 'time' column in response for %s", asset)

        # Drop the 'asset' column if present
        df = df.drop(columns=["asset"], errors="ignore")

        # Convert all metric columns to float, coercing errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
