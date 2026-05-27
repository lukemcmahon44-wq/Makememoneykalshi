"""
Pure momentum signals: ROC, Williams %R, CMF, Chande momentum.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .._common import EPS, closes, highs, lows, tanh_clip, volumes
from ...data.store import Candle


def rate_of_change(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    c = closes(candles)
    roc = (c[-1] - c[-period - 1]) / (c[-period - 1] + EPS)
    return float(tanh_clip(roc, 0.05))


def williams_r(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    h = highs(candles); l = lows(candles); c = closes(candles)
    hh = float(np.max(h[-period:]))
    ll = float(np.min(l[-period:]))
    if hh - ll < EPS:
        return 0.0
    wr = -100 * (hh - c[-1]) / (hh - ll)
    # %R ranges -100 (oversold) to 0 (overbought); fade extremes
    return float(np.clip(-tanh_clip(wr + 50, 30), -1.0, 1.0))


def chaikin_money_flow(candles: Sequence[Candle], period: int = 20) -> float:
    if len(candles) < period + 1:
        return 0.0
    h = highs(candles); l = lows(candles); c = closes(candles); v = volumes(candles)
    rng = h - l
    rng[rng < EPS] = EPS
    mfm = ((c - l) - (h - c)) / rng
    mfv = mfm * v
    cmf = float(np.sum(mfv[-period:]) / (np.sum(v[-period:]) + EPS))
    return float(tanh_clip(cmf, 0.15))


def chande_momentum(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 2:
        return 0.0
    c = closes(candles)
    deltas = np.diff(c[-period - 1:])
    gains = float(np.sum(np.where(deltas > 0, deltas, 0)))
    losses = float(np.sum(np.where(deltas < 0, -deltas, 0)))
    denom = gains + losses
    if denom < EPS:
        return 0.0
    cmo = 100 * (gains - losses) / denom
    return float(tanh_clip(cmo, 50))


def composite_momentum(candles: Sequence[Candle]) -> float:
    if len(candles) < 20:
        return 0.0
    signals = [
        rate_of_change(candles),
        williams_r(candles),
        chaikin_money_flow(candles),
        chande_momentum(candles),
    ]
    return float(np.clip(np.mean(signals), -1.0, 1.0))
