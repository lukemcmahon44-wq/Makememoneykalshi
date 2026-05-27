"""
Cointegration tests for statistical arbitrage signal generation.

Uses statsmodels for Engle-Granger and Johansen tests.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ---------------------------------------------------------------------------
# Engle-Granger test
# ---------------------------------------------------------------------------

def engle_granger_test(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Engle-Granger two-step cointegration test.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Price series (should be I(1)).

    Returns
    -------
    dict with keys:
        cointegrated : bool   – p_value < 0.05
        p_value      : float
        test_stat    : float
        critical_values : dict {1%, 5%, 10%}
    """
    if len(series_a) != len(series_b):
        min_len = min(len(series_a), len(series_b))
        series_a = series_a.iloc[-min_len:]
        series_b = series_b.iloc[-min_len:]

    # Drop NaN
    combined = pd.concat([series_a, series_b], axis=1).dropna()
    if len(combined) < 30:
        return {
            "cointegrated": False,
            "p_value": 1.0,
            "test_stat": 0.0,
            "critical_values": {"1%": None, "5%": None, "10%": None},
        }

    a = combined.iloc[:, 0].values
    b = combined.iloc[:, 1].values

    test_stat, p_value, crit_values = coint(a, b)

    return {
        "cointegrated": bool(p_value < 0.05),
        "p_value": float(p_value),
        "test_stat": float(test_stat),
        "critical_values": {
            "1%": float(crit_values[0]),
            "5%": float(crit_values[1]),
            "10%": float(crit_values[2]),
        },
    }


# ---------------------------------------------------------------------------
# Johansen test
# ---------------------------------------------------------------------------

def johansen_test(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Johansen cointegration test for a multivariate system.

    Parameters
    ----------
    df         : pd.DataFrame  – columns = asset price series
    det_order  : int           – deterministic term: -1=none, 0=constant, 1=trend
    k_ar_diff  : int           – number of lagged differences

    Returns
    -------
    dict with keys:
        n_cointegrating_vectors : int
        trace_stats             : list[float]
        crit_values             : list[list[float]]  – [90%, 95%, 99%] per rank
        eigenvectors            : list[list[float]]
    """
    clean = df.dropna()
    if len(clean) < 30 or clean.shape[1] < 2:
        return {
            "n_cointegrating_vectors": 0,
            "trace_stats": [],
            "crit_values": [],
            "eigenvectors": [],
        }

    result = coint_johansen(clean.values, det_order, k_ar_diff)

    # trace statistics and critical values
    trace_stats = result.lr1.tolist()  # trace test statistic
    crit_values = result.cvt.tolist()  # critical values [90%, 95%, 99%]

    # Count cointegrating vectors at 95% confidence
    n_coint = 0
    for i, (stat, cvs) in enumerate(zip(trace_stats, crit_values)):
        if stat > cvs[1]:  # cvs[1] = 95% critical value
            n_coint += 1

    return {
        "n_cointegrating_vectors": n_coint,
        "trace_stats": [float(x) for x in trace_stats],
        "crit_values": [[float(c) for c in row] for row in crit_values],
        "eigenvectors": result.evec.tolist(),
    }


# ---------------------------------------------------------------------------
# Find cointegrated pairs
# ---------------------------------------------------------------------------

def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    p_threshold: float = 0.05,
) -> List[Tuple[str, str, float]]:
    """
    Test all pairs in prices_df for cointegration.

    Parameters
    ----------
    prices_df   : pd.DataFrame  – columns = asset tickers, rows = time
    p_threshold : float         – maximum p-value to include a pair

    Returns
    -------
    List of (asset_a, asset_b, p_value) sorted by p_value ascending.
    """
    columns = list(prices_df.columns)
    n = len(columns)
    pairs: List[Tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            col_a = columns[i]
            col_b = columns[j]
            result = engle_granger_test(prices_df[col_a], prices_df[col_b])
            if result["p_value"] <= p_threshold:
                pairs.append((col_a, col_b, result["p_value"]))

    pairs.sort(key=lambda x: x[2])
    return pairs
