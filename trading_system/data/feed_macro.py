"""
FRED API macro data feed.

Fetches a curated set of macroeconomic time series from FRED, computes
derived indicators, and exposes the results via a simple in-memory store.
Refreshes every 15 minutes.

FRED series fetched:
  VIXCLS    — CBOE Volatility Index (daily)
  FEDFUNDS  — Effective Federal Funds Rate (monthly)
  DGS10     — 10-Year Treasury Constant Maturity Rate (daily)
  T10Y2Y    — 10-Year minus 2-Year Treasury yield spread (daily)
  DTWEXBGS  — Broad Dollar Index (daily)
  UNRATE    — Civilian Unemployment Rate (monthly)
  CPIAUCSL  — CPI All Urban Consumers (monthly)
  T5YIE     — 5-Year Breakeven Inflation Rate (daily)

Derived indicators:
  CPI_MOM        — CPI month-over-month % change
  UNRATE_CHANGE_3M — 3-month change in unemployment rate

Falls back gracefully if the FRED key is missing or the API is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFRESH_INTERVAL_SECONDS = 900  # 15 minutes

FRED_SERIES = [
    "VIXCLS",
    "FEDFUNDS",
    "DGS10",
    "T10Y2Y",
    "DTWEXBGS",
    "UNRATE",
    "CPIAUCSL",
    "T5YIE",
]

# Sensible defaults returned when the API is unavailable
_DEFAULTS: Dict[str, Optional[float]] = {
    "VIXCLS": 20.0,
    "FEDFUNDS": 5.25,
    "DGS10": 4.5,
    "T10Y2Y": 0.2,
    "DTWEXBGS": 120.0,
    "UNRATE": 4.0,
    "CPIAUCSL": 310.0,
    "T5YIE": 2.3,
    "CPI_MOM": 0.0,
    "UNRATE_CHANGE_3M": 0.0,
}


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class MacroDataStore:
    """
    Thread-safe (asyncio) container for the latest macro values.

    Attributes
    ----------
    data : dict
        ``{series_name: value}`` mapping, updated by :class:`MacroFeed`.
    last_updated : float
        Unix timestamp of the last successful refresh.
    """

    def __init__(self) -> None:
        self.data: Dict[str, Optional[float]] = dict(_DEFAULTS)
        self.last_updated: float = 0.0
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Optional[float]) -> None:
        async with self._lock:
            self.data[key] = value

    async def bulk_set(self, updates: Dict[str, Optional[float]]) -> None:
        async with self._lock:
            self.data.update(updates)
            self.last_updated = time.time()

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.data.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current macro data dict with metadata."""
        snap = dict(self.data)
        snap["_last_updated"] = self.last_updated
        snap["_age_seconds"] = time.time() - self.last_updated if self.last_updated else None
        return snap


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

class MacroFeed:
    """
    Fetches FRED macro series, computes derived indicators, and stores
    results in a :class:`MacroDataStore`.

    Parameters
    ----------
    store :
        The shared macro data store.  If *None*, a new one is created.
    api_key :
        FRED API key.  Falls back to the ``FRED_API_KEY`` environment variable.
    refresh_interval :
        Seconds between refreshes (default 900).
    """

    def __init__(
        self,
        store: Optional[MacroDataStore] = None,
        api_key: Optional[str] = None,
        refresh_interval: int = REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self.store = store or MacroDataStore()
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.refresh_interval = refresh_interval
        self._fred: Any = None  # lazy-initialised fredapi.Fred instance
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_data(self) -> Dict[str, Any]:
        """Return the current macro snapshot (non-blocking)."""
        return self.store.snapshot()

    async def refresh_once(self) -> None:
        """Perform a single fetch-and-update cycle."""
        if not self.api_key:
            logger.warning("[macro] FRED_API_KEY not set; using default macro values.")
            await self.store.bulk_set(dict(_DEFAULTS))
            return

        try:
            raw = await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)
            derived = self.compute_derived(raw)
            raw.update(derived)
            await self.store.bulk_set(raw)
            logger.info("[macro] Refreshed %d FRED series + %d derived.", len(FRED_SERIES), len(derived))
        except Exception as exc:
            logger.error("[macro] Refresh failed: %s — keeping previous values.", exc, exc_info=True)

    async def refresh_loop(self) -> None:
        """
        Run :meth:`refresh_once` every :attr:`refresh_interval` seconds.
        Runs indefinitely until cancelled.
        """
        self._running = True
        while self._running:
            await self.refresh_once()
            try:
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                logger.info("[macro] Refresh loop cancelled.")
                self._running = False
                return

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Synchronous FRED fetch (runs in thread executor)
    # ------------------------------------------------------------------

    def _get_fred(self):
        """Lazy-initialise the fredapi.Fred client."""
        if self._fred is None:
            try:
                from fredapi import Fred  # type: ignore
                self._fred = Fred(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError("fredapi is not installed. Run: pip install fredapi") from exc
        return self._fred

    def _fetch_series(self, series_id: str) -> Optional[float]:
        """Return the most-recent non-NaN value for a FRED series."""
        try:
            fred = self._get_fred()
            # fetch last 90 days to ensure we have a non-NaN recent value
            import pandas as pd
            data = fred.get_series_latest_release(series_id)
            if data is None or len(data) == 0:
                return _DEFAULTS.get(series_id)
            # drop NaN and take last value
            clean = data.dropna()
            if len(clean) == 0:
                return _DEFAULTS.get(series_id)
            return float(clean.iloc[-1])
        except Exception as exc:
            logger.warning("[macro] Failed to fetch %s: %s", series_id, exc)
            return _DEFAULTS.get(series_id)

    def _fetch_series_history(self, series_id: str, observation_limit: int = 6):
        """Return a pandas Series with the last *observation_limit* values."""
        try:
            fred = self._get_fred()
            data = fred.get_series_latest_release(series_id)
            if data is None or len(data) == 0:
                return None
            return data.dropna().iloc[-observation_limit:]
        except Exception as exc:
            logger.warning("[macro] Failed to fetch history for %s: %s", series_id, exc)
            return None

    def _fetch_all(self) -> Dict[str, Optional[float]]:
        """Fetch all FRED series synchronously and return as a dict."""
        result: Dict[str, Optional[float]] = {}
        for series_id in FRED_SERIES:
            result[series_id] = self._fetch_series(series_id)
        return result

    # ------------------------------------------------------------------
    # Derived indicators
    # ------------------------------------------------------------------

    def compute_derived(self, raw: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Compute derived macro indicators from the raw FRED data.

        Returns a dict with:
        - ``CPI_MOM``            — month-over-month % change in CPI
        - ``UNRATE_CHANGE_3M``   — change in unemployment rate over 3 months
        """
        derived: Dict[str, Optional[float]] = {}

        # CPI month-over-month — requires history; compute in thread-safe way
        try:
            cpi_hist = self._fetch_series_history("CPIAUCSL", 3)
            if cpi_hist is not None and len(cpi_hist) >= 2:
                cpi_now = float(cpi_hist.iloc[-1])
                cpi_prev = float(cpi_hist.iloc[-2])
                derived["CPI_MOM"] = round((cpi_now / cpi_prev - 1.0) * 100.0, 4) if cpi_prev != 0 else 0.0
            else:
                derived["CPI_MOM"] = _DEFAULTS["CPI_MOM"]
        except Exception as exc:
            logger.debug("[macro] CPI_MOM computation failed: %s", exc)
            derived["CPI_MOM"] = _DEFAULTS["CPI_MOM"]

        # Unemployment 3-month change
        try:
            unrate_hist = self._fetch_series_history("UNRATE", 4)
            if unrate_hist is not None and len(unrate_hist) >= 4:
                unrate_now = float(unrate_hist.iloc[-1])
                unrate_3m_ago = float(unrate_hist.iloc[-4])
                derived["UNRATE_CHANGE_3M"] = round(unrate_now - unrate_3m_ago, 4)
            else:
                derived["UNRATE_CHANGE_3M"] = _DEFAULTS["UNRATE_CHANGE_3M"]
        except Exception as exc:
            logger.debug("[macro] UNRATE_CHANGE_3M computation failed: %s", exc)
            derived["UNRATE_CHANGE_3M"] = _DEFAULTS["UNRATE_CHANGE_3M"]

        return derived
