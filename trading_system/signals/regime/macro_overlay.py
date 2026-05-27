"""
Macro overlay signal: adjusts position sizing based on macro regime indicators.

fred_data format: dict with keys (all optional):
    'VIX'     : float  – CBOE Volatility Index
    'T10Y2Y'  : float  – 10Y-2Y Treasury yield spread (%)
    'CPI_MOM' : float  – CPI month-over-month change (%)
    'UNRATE'  : float  – Unemployment rate (%) [optional]
"""

import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# VIX regime thresholds
# ---------------------------------------------------------------------------
_VIX_THRESHOLDS = [
    (45.0, 0.20),   # VIX > 45 → extreme fear → 20%
    (30.0, 0.50),   # VIX > 30 → high fear    → 50%
    (22.0, 0.80),   # VIX > 22 → elevated     → 80%
    (12.0, 1.00),   # VIX ≤ 22 → normal       → 100%
    (0.0,  1.10),   # VIX < 12 → complacency  → 110%
]


def _vix_scalar(vix: float) -> Tuple[float, str]:
    """Map VIX level to position size scalar and flag description."""
    for threshold, scalar in _VIX_THRESHOLDS:
        if vix > threshold:
            return scalar, f"VIX={vix:.1f}"
    return 1.10, f"VIX={vix:.1f} (low)"


# ---------------------------------------------------------------------------
# Yield curve regime
# ---------------------------------------------------------------------------

def _yield_curve_scalar(t10y2y: float) -> Tuple[float, str]:
    """Map T10Y2Y spread to position size scalar."""
    if t10y2y < -0.5:
        return 0.60, f"T10Y2Y={t10y2y:.2f} (deeply inverted)"
    elif t10y2y < 0.0:
        return 0.85, f"T10Y2Y={t10y2y:.2f} (inverted)"
    else:
        return 1.00, f"T10Y2Y={t10y2y:.2f} (normal)"


# ---------------------------------------------------------------------------
# CPI shock
# ---------------------------------------------------------------------------

def _cpi_scalar(cpi_mom: float) -> Tuple[float, str]:
    """Map CPI MoM change to position size scalar."""
    if cpi_mom > 0.5:
        return 0.80, f"CPI_MoM={cpi_mom:.2f}% (shock)"
    else:
        return 1.00, f"CPI_MoM={cpi_mom:.2f}%"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_macro_multiplier(fred_data: dict) -> Tuple[float, List[str]]:
    """
    Compute macro-adjusted position size multiplier.

    Parameters
    ----------
    fred_data : dict
        Macro indicator values. Supported keys:
        'VIX', 'T10Y2Y', 'CPI_MOM'

    Returns
    -------
    (scalar, flags)
        scalar : float [0.20, 1.10] – position size multiplier
        flags  : list[str]          – human-readable regime flags
    """
    scalars = []
    flags: List[str] = []

    # VIX
    if "VIX" in fred_data and fred_data["VIX"] is not None:
        vix = float(fred_data["VIX"])
        s, flag = _vix_scalar(vix)
        scalars.append(s)
        flags.append(flag)

    # Yield curve
    if "T10Y2Y" in fred_data and fred_data["T10Y2Y"] is not None:
        t10y2y = float(fred_data["T10Y2Y"])
        s, flag = _yield_curve_scalar(t10y2y)
        scalars.append(s)
        flags.append(flag)

    # CPI shock
    if "CPI_MOM" in fred_data and fred_data["CPI_MOM"] is not None:
        cpi_mom = float(fred_data["CPI_MOM"])
        s, flag = _cpi_scalar(cpi_mom)
        scalars.append(s)
        flags.append(flag)

    if not scalars:
        return (1.0, ["no_macro_data"])

    # Combine: take the minimum scalar (most conservative)
    # This prevents taking large positions when any single indicator is alarming
    combined = float(min(scalars))
    combined = float(np.clip(combined, 0.20, 1.10))

    return (combined, flags)


def equity_signal_gate(fred_data: dict) -> bool:
    """
    Binary gate: should we trade equity signals at all given macro conditions?

    Returns False (block trading) if:
    - VIX > 45 (extreme fear / market dislocation)
    - Yield curve deeply inverted AND VIX elevated (recession risk)
    - CPI shock AND yield curve inverted (stagflation)

    Returns True otherwise (allow trading).
    """
    vix = float(fred_data.get("VIX", 20.0) or 20.0)
    t10y2y = float(fred_data.get("T10Y2Y", 0.5) or 0.5)
    cpi_mom = float(fred_data.get("CPI_MOM", 0.2) or 0.2)

    # Hard block conditions
    if vix > 45:
        return False

    if t10y2y < -0.5 and vix > 25:
        return False

    if cpi_mom > 0.5 and t10y2y < 0.0:
        return False

    return True
