"""
Macro overlay - deterministic risk regime gate (NOT a directional signal).

Multiplier is applied to position sizes, never to signal value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def compute_macro_multiplier(fred_data: Dict[str, Any]
                              ) -> Tuple[float, List[str]]:
    """
    Returns (multiplier in [0.2, 1.1], list_of_active_flags).
    """
    vix = float(fred_data.get("VIXCLS", 20))
    t10y2y = float(fred_data.get("T10Y2Y", 0.5))
    cpi_mom = float(fred_data.get("CPI_MOM", 0.2))

    multiplier = 1.0
    flags: List[str] = []

    if vix > 45:
        multiplier *= 0.20; flags.append("VIX_EXTREME")
    elif vix > 30:
        multiplier *= 0.50; flags.append("VIX_HIGH")
    elif vix > 22:
        multiplier *= 0.80; flags.append("VIX_ELEVATED")
    elif vix < 12:
        multiplier *= 1.10; flags.append("VIX_COMPLACENCY")

    if t10y2y < -0.5:
        multiplier *= 0.60; flags.append("CURVE_DEEPLY_INVERTED")
    elif t10y2y < 0:
        multiplier *= 0.85; flags.append("CURVE_INVERTED")

    if cpi_mom > 0.5:
        multiplier *= 0.80; flags.append("INFLATION_SHOCK")

    return float(np.clip(multiplier, 0.20, 1.10)), flags


def equity_signal_gate(fred_data: Dict[str, Any]) -> bool:
    """Return False = equity longs disabled (classic recession setup)."""
    t10y2y = float(fred_data.get("T10Y2Y", 0.5))
    unrate_change_3m = float(fred_data.get("UNRATE_CHANGE_3M", 0.0))
    if t10y2y < -0.3 and unrate_change_3m > 0.3:
        return False
    return True
