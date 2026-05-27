"""
Momentum signal generators.
All public functions return float in [-1.0, +1.0] via tanh normalization.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public signal functions
# ---------------------------------------------------------------------------

def roc_signal(close: pd.Series, period: int = 12) -> float:
    """
    Rate of Change momentum signal.

    ROC = (close / close[-period] - 1) * 100
    signal = tanh(ROC / 5.0)
    """
    if len(close) < period + 1:
        return 0.0

    close_now = float(close.iloc[-1])
    close_ago = float(close.iloc[-(period + 1)])

    if np.isnan(close_now) or np.isnan(close_ago) or close_ago == 0:
        return 0.0

    roc = (close_now / close_ago - 1.0) * 100.0
    return float(np.tanh(roc / 5.0))


def chande_momentum_signal(close: pd.Series, period: int = 14) -> float:
    """
    Chande Momentum Oscillator signal.

    CMO = (sum_up - sum_down) / (sum_up + sum_down + 1e-8) * 100
    Range: [-100, +100]; normalize with tanh(CMO / 50).
    """
    if len(close) < period + 2:
        return 0.0

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    sum_up = gain.rolling(period).sum()
    sum_down = loss.rolling(period).sum()

    last_up = float(sum_up.iloc[-1])
    last_down = float(sum_down.iloc[-1])

    if np.isnan(last_up) or np.isnan(last_down):
        return 0.0

    cmo = (last_up - last_down) / (last_up + last_down + 1e-8) * 100.0
    return float(np.tanh(cmo / 50.0))


def cmf_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> float:
    """
    Chaikin Money Flow signal.

    MFM = ((close - low) - (high - close)) / (high - low + 1e-8)
    MFV = MFM * volume
    CMF = sum(MFV, period) / sum(volume, period)
    Range: [-1, +1]; signal = tanh(CMF * 3).
    """
    if len(close) < period + 1:
        return 0.0

    hl_range = high - low
    mfm = ((close - low) - (high - close)) / (hl_range.replace(0, np.nan) + 1e-8)
    mfv = mfm * volume

    sum_mfv = mfv.rolling(period).sum()
    sum_vol = volume.rolling(period).sum()

    cmf_val = sum_mfv / (sum_vol.replace(0, np.nan) + 1e-8)
    last_cmf = float(cmf_val.iloc[-1])

    if np.isnan(last_cmf):
        return 0.0

    return float(np.tanh(last_cmf * 3.0))


def composite_momentum_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> float:
    """
    Composite momentum signal: weighted average of ROC, CMO, CMF.

    Weights: ROC=0.35, CMO=0.35, CMF=0.30.
    """
    roc = roc_signal(close, period=12)
    cmo = chande_momentum_signal(close, period=14)
    cmf = cmf_signal(high, low, close, volume, period=20)

    # If any signal is unavailable (0.0), reduce its weight
    # Use fixed weights and normalize
    weights = {"roc": 0.35, "cmo": 0.35, "cmf": 0.30}
    signals = {"roc": roc, "cmo": cmo, "cmf": cmf}

    total_weight = 0.0
    weighted_sum = 0.0

    for name, sig in signals.items():
        w = weights[name]
        weighted_sum += w * sig
        total_weight += w

    if total_weight == 0:
        return 0.0

    composite = weighted_sum / total_weight
    return float(np.clip(composite, -1.0, 1.0))
