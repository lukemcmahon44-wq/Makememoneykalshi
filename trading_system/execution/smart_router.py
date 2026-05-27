"""
Smart order router: market / adaptive limit / TWAP based on size vs ADV.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.logger import get_logger
from ..data.store import LOBSnapshot

log = get_logger(__name__)


class SmartOrderRouter:
    MARKET_ORDER_THRESHOLD = 0.001    # < 0.1% ADV
    TWAP_THRESHOLD = 0.005             # > 0.5% ADV
    LIMIT_CHASE_MAX = 3
    LIMIT_CHASE_INTERVAL = 30
    DEFAULT_TICK_FRAC = 10

    def route(self, asset: str, qty_usd: float, side: str,
              portfolio_value: float, current_lob: Optional[LOBSnapshot],
              adv_usd: float) -> Dict[str, Any]:
        if adv_usd <= 0:
            return self._market_order(asset, qty_usd, side)
        pct_adv = qty_usd / adv_usd
        if pct_adv < self.MARKET_ORDER_THRESHOLD:
            return self._market_order(asset, qty_usd, side)
        if pct_adv < self.TWAP_THRESHOLD:
            return self._adaptive_limit(asset, qty_usd, side, current_lob)
        n_slices = max(3, int(np.ceil(pct_adv / 0.001)))
        return self._twap_schedule(asset, qty_usd, side, n_slices, 45)

    def _market_order(self, asset: str, qty_usd: float, side: str) -> Dict[str, Any]:
        return {"type": "market", "asset": asset, "qty_usd": qty_usd, "side": side}

    def _adaptive_limit(self, asset: str, qty_usd: float, side: str,
                        lob: Optional[LOBSnapshot]) -> Dict[str, Any]:
        if lob is None or not lob.bids or not lob.asks:
            return self._market_order(asset, qty_usd, side)
        best_bid = lob.bids[0][0]
        best_ask = lob.asks[0][0]
        tick = (best_ask - best_bid) / self.DEFAULT_TICK_FRAC
        limit_price = best_bid + tick if side == "buy" else best_ask - tick
        return {
            "type": "adaptive_limit",
            "asset": asset, "qty_usd": qty_usd, "side": side,
            "limit_price": limit_price,
            "max_chases": self.LIMIT_CHASE_MAX,
            "chase_interval": self.LIMIT_CHASE_INTERVAL,
            "chase_increment": tick,
            "fallback": "market",
        }

    def _twap_schedule(self, asset: str, total_usd: float, side: str,
                       n_slices: int, interval_secs: int) -> Dict[str, Any]:
        rng = np.random.default_rng()
        slice_usd = total_usd / n_slices
        schedule: List[Dict[str, Any]] = []
        for i in range(n_slices):
            jitter = float(rng.uniform(0.90, 1.10))
            schedule.append({
                "slice_usd": slice_usd * jitter,
                "delay_secs": i * interval_secs + float(rng.uniform(-5, 5)),
                "type": "adaptive_limit",
            })
        return {"type": "twap", "asset": asset, "side": side,
                "schedule": schedule, "total_usd": total_usd}
