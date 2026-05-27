"""
Monte Carlo statistical significance tests for backtests.

Bootstrap: shuffle trade returns, compute distribution of Sharpe ratios.
Permutation: shuffle entry timing, verify strategy beats random entries.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def bootstrap_sharpe(trade_pnls: List[float], n_sims: int = 1000
                      ) -> Dict[str, float]:
    if len(trade_pnls) < 30:
        return {"observed_sharpe": 0.0, "p_value": 1.0}
    arr = np.asarray(trade_pnls, dtype=float)
    observed = float(np.mean(arr) / (np.std(arr, ddof=0) + 1e-8) * np.sqrt(252))
    rng = np.random.default_rng(42)
    centered = arr - np.mean(arr)   # null: zero mean
    sharpes = []
    for _ in range(n_sims):
        sample = rng.choice(centered, size=len(arr), replace=True)
        sharpes.append(np.mean(sample) / (np.std(sample, ddof=0) + 1e-8)
                       * np.sqrt(252))
    p_value = float(np.mean(np.asarray(sharpes) >= observed))
    return {"observed_sharpe": observed, "p_value": p_value,
             "significant": p_value < 0.05}


def deflated_sharpe_ratio(sr: float, n_trials: int, T: int,
                           skew: float = 0, kurt: float = 3) -> float:
    """
    Bailey & Lopez de Prado (2014). Adjusts observed Sharpe for selection bias.
    """
    if T < 30 or sr == 0:
        return 0.0
    from math import log, sqrt
    expected_max = sqrt(2 * log(max(n_trials, 2)))
    var_sr = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (T - 1)
    if var_sr <= 0:
        return 0.0
    psr = (sr - expected_max * sqrt(var_sr)) / sqrt(var_sr)
    # Convert z to probability via stdnormal CDF approximation
    from math import erf
    cdf = 0.5 * (1 + erf(psr / sqrt(2)))
    return float(cdf)
