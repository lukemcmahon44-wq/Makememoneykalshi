"""
Monte Carlo stress tests + scenario shocks.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def bootstrap_drawdown(returns: np.ndarray, n_sims: int = 1000,
                       horizon: int = 252) -> Dict[str, float]:
    if len(returns) < 60:
        return {"mean_dd": 0.0, "p95_dd": 0.0, "p99_dd": 0.0}
    rng = np.random.default_rng(42)
    arr = np.asarray(returns, dtype=float)
    dds: List[float] = []
    for _ in range(n_sims):
        sample = rng.choice(arr, size=horizon, replace=True)
        cum = np.cumprod(1 + sample)
        peak = np.maximum.accumulate(cum)
        dd = float(np.max((peak - cum) / peak))
        dds.append(dd)
    arr_dds = np.asarray(dds)
    return {
        "mean_dd": float(np.mean(arr_dds)),
        "p95_dd": float(np.quantile(arr_dds, 0.95)),
        "p99_dd": float(np.quantile(arr_dds, 0.99)),
    }


def scenario_shock(positions: Dict[str, dict],
                   shock_pct: Dict[str, float]) -> Dict[str, float]:
    """Apply per-asset shock; return P&L impact per position."""
    impacts: Dict[str, float] = {}
    for asset, pos in positions.items():
        size_usd = float(pos.get("size_usd", 0))
        direction = 1 if pos.get("direction") == "long" else -1
        shock = float(shock_pct.get(asset, 0.0))
        impacts[asset] = size_usd * shock * direction
    return impacts
