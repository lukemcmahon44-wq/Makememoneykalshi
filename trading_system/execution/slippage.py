"""
Slippage tracker. Logs every fill, computes rolling slippage stats per asset.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import numpy as np


class SlippageTracker:
    def __init__(self, db=None, window: int = 200):
        self.db = db
        self.window = window
        self.history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window))

    def log_fill(self, asset: str, expected_price: float, actual_price: float,
                  side: str, qty: float, order_type: str) -> float:
        if expected_price <= 0:
            return 0.0
        slip_bps = ((actual_price - expected_price) / expected_price) * 10_000
        if side == "sell":
            slip_bps = -slip_bps
        self.history[asset].append(slip_bps)
        if self.db is not None:
            try:
                self.db.log_fill_quality(asset, expected_price, actual_price,
                                           side, qty, order_type)
            except Exception:
                pass
        return float(slip_bps)

    def stats(self, asset: str) -> Dict[str, float]:
        arr = list(self.history.get(asset, []))
        if not arr:
            return {"n": 0, "mean_bps": 0.0, "p95_bps": 0.0, "max_bps": 0.0}
        a = np.asarray(arr)
        return {
            "n": len(arr),
            "mean_bps": float(np.mean(a)),
            "p95_bps": float(np.quantile(a, 0.95)),
            "max_bps": float(np.max(a)),
        }

    def is_degraded(self, asset: str, threshold_bps: float = 25.0) -> bool:
        """True if recent slippage exceeds threshold consistently."""
        arr = list(self.history.get(asset, []))
        if len(arr) < 5:
            return False
        recent = arr[-10:]
        return float(np.mean(recent)) > threshold_bps
