"""
Trend signals: EMA crossover, triple EMA, MACD, ADX regime, Ichimoku.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .._common import (EPS, adx, atr, closes, ema, highs, lows, macd,
                       rolling_std, tanh_clip)
from ...data.store import Candle


def dual_ema_cross(candles: Sequence[Candle], fast: int = 9, slow: int = 21
                   ) -> float:
    if len(candles) < slow + 5:
        return 0.0
    c = closes(candles)
    fast_ema = ema(c, fast)
    slow_ema = ema(c, slow)
    a = atr(candles, 14) or 1.0
    raw_cross = (fast_ema[-1] - slow_ema[-1]) / max(a, EPS)

    fast_slope = (fast_ema[-1] - fast_ema[-3]) / (fast_ema[-3] + EPS)
    slow_slope = (slow_ema[-1] - slow_ema[-3]) / (slow_ema[-3] + EPS)
    slope_aligned = (np.sign(fast_slope) == np.sign(slow_slope) ==
                     np.sign(raw_cross))
    signal = tanh_clip(raw_cross, 2.0) * (1.0 if slope_aligned else 0.3)
    return float(np.clip(signal, -1.0, 1.0))


def triple_ema_alignment(candles: Sequence[Candle]) -> float:
    if len(candles) < 55:
        return 0.0
    c = closes(candles)
    e5 = ema(c, 5)[-1]
    e13 = ema(c, 13)[-1]
    e50 = ema(c, 50)[-1]

    fully_bullish = e5 > e13 > e50
    fully_bearish = e5 < e13 < e50
    partial_bull = (e5 > e13) and (e13 < e50)
    partial_bear = (e5 < e13) and (e13 > e50)

    if fully_bullish:
        strength = 1.0
    elif fully_bearish:
        strength = -1.0
    elif partial_bull:
        strength = 0.4
    elif partial_bear:
        strength = -0.4
    else:
        strength = 0.0

    a = atr(candles, 14) or 1.0
    magnitude = tanh_clip(abs(e5 - e50) / (2 * max(a, EPS)), 1.0)
    return float(np.clip(strength * magnitude, -1.0, 1.0))


def macd_signal(candles: Sequence[Candle]) -> float:
    if len(candles) < 35:
        return 0.0
    c = closes(candles)
    _, _, hist = macd(c)
    hist_std = float(np.std(hist[-20:], ddof=0)) if len(hist) >= 20 else 1.0
    if hist_std < EPS:
        return 0.0
    hist_slope = float(hist[-1] - hist[-3])
    signal = tanh_clip(hist_slope, hist_std)

    # Divergence amplification
    recent_close = c[-5:]
    recent_hist = hist[-5:]
    if len(recent_close) == 5 and len(recent_hist) == 5:
        price_falling = recent_close[-1] < recent_close[0]
        price_rising = recent_close[-1] > recent_close[0]
        hist_rising = recent_hist[-1] > recent_hist[0]
        hist_falling = recent_hist[-1] < recent_hist[0]
        if price_falling and hist_rising:
            signal = max(0.6, signal)
        elif price_rising and hist_falling:
            signal = min(-0.6, signal)
    return float(np.clip(signal, -1.0, 1.0))


def adx_regime(candles: Sequence[Candle]) -> Dict[str, float | str]:
    if len(candles) < 30:
        return {"regime": "transition", "multiplier": 1.0, "direction": 0.0,
                "adx": 0.0}
    a, p, m = adx(candles, 14)
    direction_strength = (p - m) / (p + m + EPS)
    if a > 25:
        regime = "trending"
    elif a < 20:
        regime = "ranging"
    else:
        regime = "transition"
    multiplier = float(np.clip(a / 30.0, 0.5, 1.5))
    return {"regime": regime, "multiplier": multiplier,
            "direction": float(direction_strength), "adx": float(a)}


def ichimoku(candles: Sequence[Candle]) -> float:
    if len(candles) < 60:
        return 0.0
    h, l, c = highs(candles), lows(candles), closes(candles)

    def _range_mid(arr_h, arr_l, w, offset=0):
        end = len(arr_h) - offset
        start = end - w
        if start < 0:
            return None
        return (np.max(arr_h[start:end]) + np.min(arr_l[start:end])) / 2

    tenkan = _range_mid(h, l, 9)
    kijun = _range_mid(h, l, 26)
    if tenkan is None or kijun is None:
        return 0.0

    senkou_a_now = _range_mid(h[:-26], l[:-26], 9) if len(h) > 35 else None
    senkou_b_now = _range_mid(h[:-26], l[:-26], 52) if len(h) > 78 else None
    if senkou_a_now is None or senkou_b_now is None:
        return 0.0

    cloud_top = max(senkou_a_now, senkou_b_now)
    above_cloud = c[-1] > cloud_top
    below_cloud = c[-1] < min(senkou_a_now, senkou_b_now)
    tk_cross = tenkan > kijun

    chikou_confirm = c[-1] > c[-27] if len(c) >= 27 else False

    bullish_count = sum([above_cloud, tk_cross, chikou_confirm])
    bearish_count = sum([below_cloud, not tk_cross, not chikou_confirm])

    if above_cloud or below_cloud:
        signal = (bullish_count - bearish_count) / 3.0
    else:
        signal = 0.0
    return float(np.clip(signal, -1.0, 1.0))


def composite_trend(candles: Sequence[Candle]) -> float:
    """Equal-weighted composite of all trend signals."""
    if len(candles) < 60:
        return 0.0
    signals = [
        dual_ema_cross(candles),
        triple_ema_alignment(candles),
        macd_signal(candles),
        ichimoku(candles),
    ]
    return float(np.clip(np.mean(signals), -1.0, 1.0))
