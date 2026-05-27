"""
Cointegration tests: Engle-Granger and Johansen.

Use Engle-Granger as default (single pair). Use Johansen for >2 assets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .._common import EPS


def engle_granger(y: np.ndarray, x: np.ndarray, significance: float = 0.05
                  ) -> Tuple[bool, float, float, np.ndarray]:
    """
    Returns: (is_cointegrated, p_value, hedge_ratio, residuals)
    """
    try:
        from statsmodels.tsa.stattools import coint  # type: ignore
        import statsmodels.api as sm  # type: ignore
    except ImportError:
        return False, 1.0, 1.0, np.zeros_like(y)
    if len(y) < 30 or len(x) < 30:
        return False, 1.0, 1.0, np.zeros_like(y)
    try:
        t_stat, p_value, _ = coint(y, x)
        x_c = sm.add_constant(x)
        ols = sm.OLS(y, x_c).fit()
        hedge_ratio = float(ols.params[1])
        intercept = float(ols.params[0])
        residuals = y - (hedge_ratio * x + intercept)
        return p_value < significance, float(p_value), hedge_ratio, residuals
    except Exception:
        return False, 1.0, 1.0, np.zeros_like(y)


def half_life(spread: np.ndarray) -> Optional[float]:
    """Estimate mean-reversion half-life via OLS on Δspread vs spread_lag."""
    if len(spread) < 30:
        return None
    s = np.asarray(spread, dtype=float)
    s_lag = s[:-1]
    ds = np.diff(s)
    A = np.column_stack([np.ones_like(s_lag), s_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, ds, rcond=None)
    except np.linalg.LinAlgError:
        return None
    b = float(coeffs[1])
    if b >= 0:
        return None
    return float(-np.log(2) / b)


def cointegrated_pair_score(y: np.ndarray, x: np.ndarray
                            ) -> dict:
    cointegrated, p, h, resid = engle_granger(y, x)
    hl = half_life(resid)
    return {
        "cointegrated": cointegrated,
        "p_value": p,
        "hedge_ratio": h,
        "half_life": hl,
        "residuals": resid,
        "tradeable": (cointegrated and hl is not None and 1 < hl < 168),
    }
