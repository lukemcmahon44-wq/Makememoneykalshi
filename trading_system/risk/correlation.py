"""
Rolling correlation matrix + crowding detection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_correlation_matrix(returns: pd.DataFrame, window: int = 60
                                ) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if len(returns) < window:
        return out
    df = returns.tail(window)
    corr = df.corr().fillna(0.0)
    assets = corr.columns.tolist()
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if i >= j:
                continue
            out[(a, b)] = float(corr.iloc[i, j])
    return out


def is_overcrowded(asset: str, open_positions: Dict[str, dict],
                    corr_matrix: Dict[Tuple[str, str], float],
                    threshold: float = 0.7) -> Tuple[bool, List[str]]:
    """Return (True, conflicting_assets) if too correlated to existing book."""
    conflicts: List[str] = []
    for pos_asset in open_positions:
        key = (asset, pos_asset) if (asset, pos_asset) in corr_matrix else (pos_asset, asset)
        c = float(corr_matrix.get(key, 0.0))
        if abs(c) >= threshold:
            conflicts.append(pos_asset)
    return (len(conflicts) > 0, conflicts)
