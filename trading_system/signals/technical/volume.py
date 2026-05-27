"""
Volume signals: OBV, VWAP deviation, volume surge.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .._common import EPS, closes, tanh_clip, volumes, zscore
from ...data.store import Candle


def obv(candles: Sequence[Candle]) -> np.ndarray:
    if len(candles) < 2:
        return np.zeros(len(candles))
    c = closes(candles); v = volumes(candles)
    out = np.zeros_like(c)
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - v[i]
        else:
            out[i] = out[i - 1]
    return out


def obv_zscore(candles: Sequence[Candle], window: int = 20) -> float:
    if len(candles) < window + 2:
        return 0.0
    o = obv(candles)
    z = zscore(o, window)
    return float(tanh_clip(z, 2.0))


def volume_surge(candles: Sequence[Candle], window: int = 20) -> float:
    if len(candles) < window + 1:
        return 0.0
    v = volumes(candles)
    avg = float(np.mean(v[-window - 1: -1]))
    if avg < EPS:
        return 0.0
    ratio = v[-1] / avg
    return float(tanh_clip(ratio - 1.0, 2.0))


def money_flow_index(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    from .._common import highs, lows
    h = highs(candles); l = lows(candles); c = closes(candles); v = volumes(candles)
    tp = (h + l + c) / 3
    raw_mf = tp * v
    pos_mf = 0.0; neg_mf = 0.0
    for i in range(len(tp) - period, len(tp)):
        if i == 0:
            continue
        if tp[i] > tp[i - 1]:
            pos_mf += raw_mf[i]
        elif tp[i] < tp[i - 1]:
            neg_mf += raw_mf[i]
    if neg_mf < EPS:
        return 1.0 if pos_mf > 0 else 0.0
    mfi = 100 - (100 / (1 + pos_mf / neg_mf))
    return float(-tanh_clip(mfi - 50, 20))


def composite_volume(candles: Sequence[Candle]) -> float:
    if len(candles) < 25:
        return 0.0
    sig = [
        obv_zscore(candles),
        volume_surge(candles),
        money_flow_index(candles),
    ]
    return float(np.clip(np.mean(sig), -1.0, 1.0))
