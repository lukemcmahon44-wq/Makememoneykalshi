"""
Historical and parametric VaR / CVaR.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def historical_var(returns: np.ndarray, confidence: float = 0.95
                    ) -> Tuple[float, float]:
    """
    Returns (VaR, CVaR) at the given confidence level.
    Negative numbers mean expected loss.
    """
    if len(returns) < 30:
        return 0.0, 0.0
    arr = np.asarray(returns, dtype=float)
    alpha = 1 - confidence
    var = float(np.quantile(arr, alpha))
    tail = arr[arr <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return var, cvar


def parametric_var(returns: np.ndarray, confidence: float = 0.95
                    ) -> Tuple[float, float]:
    """Gaussian VaR/CVaR. Underestimates tail risk."""
    if len(returns) < 30:
        return 0.0, 0.0
    arr = np.asarray(returns, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=0))
    # Normal inverse CDF approximation at common confidence levels
    z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(round(confidence, 2), 1.645)
    var = mu - z * sd
    # CVaR closed form for normal
    from math import exp, pi, sqrt
    pdf_z = exp(-(z ** 2) / 2) / sqrt(2 * pi)
    cvar = mu - sd * (pdf_z / (1 - confidence))
    return float(var), float(cvar)


def portfolio_var(positions_value: dict, returns: dict, confidence: float = 0.95
                   ) -> Tuple[float, float]:
    """Aggregate VaR over portfolio assets (assume independent)."""
    total = 0.0
    for asset, weight in positions_value.items():
        if asset not in returns:
            continue
        var, cvar = historical_var(returns[asset], confidence)
        total += weight * var
    return float(total), float(total)
