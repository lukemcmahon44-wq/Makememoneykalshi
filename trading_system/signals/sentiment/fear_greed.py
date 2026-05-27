"""
Alternative.me Crypto Fear & Greed + VIX-based equity fear gauge.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import aiohttp
import numpy as np

from ...core.logger import get_logger
from .._common import tanh_clip

log = get_logger(__name__)


async def fetch_crypto_fear_greed(session: aiohttp.ClientSession
                                   ) -> Optional[Dict[str, Any]]:
    try:
        async with session.get("https://api.alternative.me/fng/", timeout=10) as r:
            if r.status != 200:
                return None
            data = await r.json()
            if not data.get("data"):
                return None
            entry = data["data"][0]
            return {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "neutral"),
                "timestamp": int(entry.get("timestamp", 0)),
            }
    except Exception as e:
        log.warning(f"Fear & Greed fetch error: {e}")
        return None


def fear_greed_signal(value: int) -> float:
    """Contrarian signal: extreme fear = bullish, extreme greed = bearish."""
    centered = (value - 50) / 50.0   # → [-1, +1]
    return float(np.clip(-centered, -1.0, 1.0))


def vix_based_signal(vix: float) -> float:
    """For equities: scaled inverse of VIX deviation from 18."""
    if vix is None:
        return 0.0
    centered = (float(vix) - 18.0) / 12.0   # 18 = neutral, ±12 = extreme
    return float(np.clip(-tanh_clip(centered, 1.0), -1.0, 1.0))
