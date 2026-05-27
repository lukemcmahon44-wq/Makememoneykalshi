"""
Common signal utilities. Provides robust indicator math operating on numpy.
All functions accept a 1-D numpy array of closes or a Candle deque.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..data.store import Candle

EPS = 1e-10


def closes(candles: Sequence[Candle]) -> np.ndarray:
    return np.array([c.close for c in candles], dtype=float)


def highs(candles: Sequence[Candle]) -> np.ndarray:
    return np.array([c.high for c in candles], dtype=float)


def lows(candles: Sequence[Candle]) -> np.ndarray:
    return np.array([c.low for c in candles], dtype=float)


def volumes(candles: Sequence[Candle]) -> np.ndarray:
    return np.array([c.volume for c in candles], dtype=float)


def opens(candles: Sequence[Candle]) -> np.ndarray:
    return np.array([c.open for c in candles], dtype=float)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def sma(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    csum = np.cumsum(np.insert(values, 0, 0))
    out = (csum[period:] - csum[:-period]) / period
    return np.concatenate([np.full(period - 1, np.nan), out])


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(period - 1, len(values)):
        out[i] = float(np.std(values[i - period + 1: i + 1], ddof=0))
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    if len(high) < 2:
        return np.full_like(high, np.nan, dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    return np.maximum.reduce([high - low, np.abs(high - prev_close),
                              np.abs(low - prev_close)])


def atr(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    h, l, c = highs(candles), lows(candles), closes(candles)
    tr = true_range(h, l, c)
    return float(np.mean(tr[-period:]))


def rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss < EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(values: np.ndarray, period: int = 14) -> np.ndarray:
    if len(values) < period + 1:
        return np.full(len(values), 50.0)
    out = np.full(len(values), 50.0)
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    for i in range(period, len(values)):
        ag = float(np.mean(gains[i - period: i]))
        al = float(np.mean(losses[i - period: i]))
        if al < EPS:
            out[i] = 100.0
        else:
            rs = ag / al
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(values: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)
    macd_line = fast_ema - slow_ema
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def adx(candles: Sequence[Candle], period: int = 14):
    if len(candles) < period + 2:
        return 0.0, 0.0, 0.0
    h, l, c = highs(candles), lows(candles), closes(candles)
    up_move = h[1:] - h[:-1]
    dn_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = true_range(h, l, c)[1:]

    atr_arr = ema(tr, period)
    plus_di = 100 * ema(plus_dm, period) / (atr_arr + EPS)
    minus_di = 100 * ema(minus_dm, period) / (atr_arr + EPS)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + EPS)
    adx_val = ema(dx, period)
    return float(adx_val[-1]), float(plus_di[-1]), float(minus_di[-1])


def tanh_clip(x: float, scale: float = 1.0) -> float:
    return float(np.tanh(x / scale))


def safe_div(a, b):
    return a / (b + EPS)


def zscore(values: np.ndarray, window: int = 20) -> float:
    if len(values) < window:
        return 0.0
    sample = values[-window:]
    mu = float(np.mean(sample))
    sd = float(np.std(sample, ddof=0))
    if sd < EPS:
        return 0.0
    return (float(values[-1]) - mu) / sd


def percentile_rank(arr: np.ndarray, value: float) -> float:
    if len(arr) == 0:
        return 0.5
    return float(np.mean(arr < value))
