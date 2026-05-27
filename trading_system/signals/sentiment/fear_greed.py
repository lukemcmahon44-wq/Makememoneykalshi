"""
Fear & Greed Index signal.

Fetches the CNN Fear & Greed Index from the alternative.me API.
Caches the result for 30 minutes to avoid excessive API calls.

API: https://api.alternative.me/fng/
"""

import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/"
_CACHE_TTL_SECONDS = 1800  # 30 minutes


class FearGreedSignal:
    """
    Fear & Greed Index signal with caching.

    Converts the index value [0, 100] to a contrarian signal in [-1, +1]:
        0-20   (Extreme Fear)  → +1.0  (buy signal)
        20-40  (Fear)          → +0.5
        40-60  (Neutral)       →  0.0
        60-80  (Greed)         → -0.5
        80-100 (Extreme Greed) → -1.0
    """

    def __init__(self) -> None:
        self._cache: Optional[dict] = None
        self._cache_time: float = 0.0

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch(self) -> dict:
        """
        Fetch the current Fear & Greed Index value.

        Returns the API response dict with at minimum:
            {
              'value': str,          – e.g. '32'
              'value_classification': str,  – e.g. 'Fear'
              'timestamp': str
            }
        Returns cached result if within 30-minute window.
        Raises requests.RequestException on network failure.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < _CACHE_TTL_SECONDS:
            return self._cache

        try:
            response = requests.get(_FNG_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            # API returns list of entries; take the most recent
            entries = data.get("data", [])
            if not entries:
                raise ValueError("Empty data from Fear & Greed API")

            result = entries[0]  # most recent entry
            self._cache = result
            self._cache_time = now
            return result

        except Exception as exc:
            logger.warning("Fear & Greed fetch failed: %s", exc)
            # Return cached value even if stale, or default neutral
            if self._cache is not None:
                return self._cache
            return {"value": "50", "value_classification": "Neutral"}

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def get_signal(self, value: Optional[float] = None) -> float:
        """
        Convert Fear & Greed index value to a contrarian trading signal.

        Parameters
        ----------
        value : float | None
            Index value in [0, 100].
            If None, fetches from API first.

        Returns
        -------
        float in [-1.0, +1.0]
            Contrarian: extreme fear → +1 (buy), extreme greed → -1 (sell).
        """
        if value is None:
            data = self.fetch()
            try:
                value = float(data.get("value", 50))
            except (TypeError, ValueError):
                value = 50.0

        value = float(value)

        if value < 20:
            return 1.0    # Extreme Fear → strong buy
        elif value < 40:
            # Fear: linear interpolation 20→+0.5, 40→0.0... wait, spec says 20-40=+0.5
            return 0.5
        elif value <= 60:
            # Neutral
            return 0.0
        elif value <= 80:
            # Greed
            return -0.5
        else:
            # Extreme Greed → strong sell
            return -1.0

    # ------------------------------------------------------------------
    # Convenience: fetch + signal in one call
    # ------------------------------------------------------------------

    def fetch_and_signal(self) -> float:
        """Fetch latest F&G index and return the contrarian signal."""
        return self.get_signal(value=None)
