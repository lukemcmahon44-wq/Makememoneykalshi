"""
Mean-reversion signals: Bollinger, RSI w/ divergence, VWAP deviation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .._common import (EPS, atr, closes, highs, lows, rolling_std, rsi,
                       rsi_series, sma, tanh_clip, volumes)
from ...data.store import Candle


def bollinger_signal(candles: Sequence[Candle], period: int = 20) -> float:
    if len(candles) < period + 1:
        return 0.0
    c = closes(candles)
    mu = float(np.mean(c[-period:]))
    sd = float(np.std(c[-period:], ddof=0))
    if sd < EPS:
        return 0.0
    z = (c[-1] - mu) / sd
    signal = -tanh_clip(z, 2.5)

    # Squeeze suppression
    bw_series = rolling_std(c, period) * 4 / (sma(c, period) + EPS)
    if len(bw_series) >= 100:
        valid = bw_series[~np.isnan(bw_series)]
        if len(valid) > 30:
            current_bw = bw_series[-1]
            pct = float(np.mean(valid < current_bw))
            if pct < 0.20:    # in squeeze
                signal *= 0.3
    return float(np.clip(signal, -1.0, 1.0))


def rsi_signal(candles: Sequence[Candle]) -> float:
    if len(candles) < 25:
        return 0.0
    c = closes(candles)
    rsi_arr = rsi_series(c, 14)
    rsi_val = float(rsi_arr[-1])
    base = -tanh_clip(rsi_val - 50.0, 20.0)

    # Divergence checks
    new_price_low = c[-1] < float(np.min(c[-20:-1])) if len(c) >= 20 else False
    new_rsi_low = rsi_arr[-1] < float(np.min(rsi_arr[-20:-1])) if len(rsi_arr) >= 20 else False
    new_price_high = c[-1] > float(np.max(c[-20:-1])) if len(c) >= 20 else False
    new_rsi_high = rsi_arr[-1] > float(np.max(rsi_arr[-20:-1])) if len(rsi_arr) >= 20 else False

    bull_div = new_price_low and not new_rsi_low
    bear_div = new_price_high and not new_rsi_high
    if bull_div:
        return float(max(0.7, base))
    if bear_div:
        return float(min(-0.7, base))
    return float(np.clip(base, -1.0, 1.0))


def vwap_signal(candles: Sequence[Candle], session_vwap: float | None = None
                ) -> float:
    if len(candles) < 25 or session_vwap is None:
        return 0.0
    c = closes(candles)
    a = atr(candles, 14) or 1.0
    dev = (c[-1] - session_vwap) / (0.5 * a + EPS)
    signal = -tanh_clip(dev, 2.0)

    # Bands: amplify on 2-sigma deviation
    typical = (closes(candles) + highs(candles) + lows(candles)) / 3
    diff = typical[-min(20, len(typical)):] - session_vwap
    vwap_std = float(np.std(diff, ddof=0)) if len(diff) > 1 else 0.0
    if vwap_std > 0:
        sigma_dev = abs(c[-1] - session_vwap) / vwap_std
        if sigma_dev > 2.0:
            signal *= 1.5
    return float(np.clip(signal, -1.0, 1.0))


def stochastic_signal(candles: Sequence[Candle], k_period: int = 14
                      ) -> float:
    if len(candles) < k_period + 5:
        return 0.0
    h = highs(candles); l = lows(candles); c = closes(candles)
    highest = float(np.max(h[-k_period:]))
    lowest = float(np.min(l[-k_period:]))
    if highest - lowest < EPS:
        return 0.0
    k = 100 * (c[-1] - lowest) / (highest - lowest)
    base = -tanh_clip(k - 50.0, 20.0)
    return float(np.clip(base, -1.0, 1.0))


def composite_reversion(candles: Sequence[Candle],
                        session_vwap: float | None = None) -> float:
    if len(candles) < 30:
        return 0.0
    signals = [
        bollinger_signal(candles),
        rsi_signal(candles),
        vwap_signal(candles, session_vwap),
        stochastic_signal(candles),
    ]
    return float(np.clip(np.mean(signals), -1.0, 1.0))
