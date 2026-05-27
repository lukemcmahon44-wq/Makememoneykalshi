"""
Avellaneda-Stoikov (2008) optimal market making model.

Used for limit-order exit pricing on crypto. Entries use market orders.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class AvellanedaStoikovExecutor:
    def __init__(self, risk_aversion: float = 0.1, k: float = 1.5,
                 T: float = 1.0):
        self.gamma = risk_aversion
        self.k = k
        self.T = T

    def reservation_price(self, mid: float, inventory_qty: float,
                          total_capital: float, asset_vol: float,
                          time_remaining: float) -> float:
        max_pos = total_capital * 0.15
        q = inventory_qty / max(max_pos, 1e-8)
        return float(mid - q * self.gamma * (asset_vol ** 2) * time_remaining)

    def optimal_spread(self, asset_vol: float, time_remaining: float
                       ) -> float:
        inventory_component = self.gamma * (asset_vol ** 2) * time_remaining
        selection_component = (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return float(inventory_component + selection_component)

    def optimal_quotes(self, mid: float, inventory_qty: float,
                       total_capital: float, asset_vol: float,
                       time_remaining: float) -> Dict[str, float]:
        r = self.reservation_price(mid, inventory_qty, total_capital,
                                    asset_vol, time_remaining)
        half = self.optimal_spread(asset_vol, time_remaining) / 2
        return {
            "reservation_price": r,
            "bid": r - half,
            "ask": r + half,
            "half_spread": half,
        }
