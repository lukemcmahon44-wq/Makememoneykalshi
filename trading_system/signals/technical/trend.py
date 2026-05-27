"""
Trend-following signal generators.
All public functions return float in [-1.0, +1.0] via tanh normalization.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential moving average using pandas ewm (span = period)."""
    return prices.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing via ewm with com = period-1)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ---------------------------------------------------------------------------
# Public signal functions
# ---------------------------------------------------------------------------

def dual_ema_signal(close: pd.Series, high: pd.Series, low: pd.Series) -> float:
    """
    EMA9 vs EMA21 crossover with slope confirmation.

    raw_cross = (EMA9 - EMA21) / ATR(14)
    slope_aligned = sign(fast_slope) == sign(slow_slope) == sign(raw_cross)
    return tanh(raw_cross / 2.0) * (1.0 if slope_aligned else 0.3)
    """
    if len(close) < 30:
        return 0.0

    fast = ema(close, 9)
    slow = ema(close, 21)
    atr14 = atr(high, low, close, 14)

    last_atr = atr14.iloc[-1]
    if last_atr <= 0:
        return 0.0

    raw_cross = (fast.iloc[-1] - slow.iloc[-1]) / last_atr

    # Slopes over last 3 bars
    fast_slope = fast.iloc[-1] - fast.iloc[-3]
    slow_slope = slow.iloc[-1] - slow.iloc[-3]

    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    slope_aligned = (
        _sign(fast_slope) == _sign(raw_cross) and _sign(slow_slope) == _sign(raw_cross)
    )

    multiplier = 1.0 if slope_aligned else 0.3
    return float(np.tanh(raw_cross / 2.0) * multiplier)


def triple_ema_signal(close: pd.Series, high: pd.Series, low: pd.Series) -> float:
    """
    EMA5 / EMA13 / EMA50 full alignment strength.

    Returns tanh of a weighted position score relative to ATR.
    """
    if len(close) < 55:
        return 0.0

    e5 = ema(close, 5)
    e13 = ema(close, 13)
    e50 = ema(close, 50)
    atr14 = atr(high, low, close, 14)

    last_atr = atr14.iloc[-1]
    if last_atr <= 0:
        return 0.0

    v5, v13, v50 = e5.iloc[-1], e13.iloc[-1], e50.iloc[-1]

    # Normalised gap between each consecutive pair
    gap_fast_mid = (v5 - v13) / last_atr       # > 0 = bullish
    gap_mid_slow = (v13 - v50) / last_atr      # > 0 = bullish

    # Slopes (3-bar)
    slope5  = (e5.iloc[-1]  - e5.iloc[-3])  / last_atr
    slope13 = (e13.iloc[-1] - e13.iloc[-3]) / last_atr
    slope50 = (e50.iloc[-1] - e50.iloc[-3]) / last_atr

    # Full alignment: all three gaps same sign + all slopes same sign
    gaps_aligned = (np.sign(gap_fast_mid) == np.sign(gap_mid_slow))
    slopes_aligned = (np.sign(slope5) == np.sign(slope13) == np.sign(slope50))

    # Strength = average of the two gap magnitudes, then penalise misalignment
    raw = (gap_fast_mid + gap_mid_slow) / 2.0
    alignment_factor = 1.0 if (gaps_aligned and slopes_aligned) else 0.25

    return float(np.tanh(raw / 1.5) * alignment_factor)


def macd_signal(close: pd.Series) -> float:
    """
    MACD histogram divergence signal.

    Histogram = MACD - Signal (EMA9 of MACD).
    Divergence: price falling last 5 bars but histogram rising.
    Returns tanh(hist_slope / (hist_std + 1e-8)), amplified on divergence.
    """
    if len(close) < 35:
        return 0.0

    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    histogram = macd_line - signal_line

    if len(histogram) < 10:
        return 0.0

    hist_slice = histogram.dropna()
    if len(hist_slice) < 6:
        return 0.0

    # Slope over last 5 bars via linear regression coefficient
    y = hist_slice.iloc[-5:].values
    x = np.arange(len(y))
    hist_slope = float(np.polyfit(x, y, 1)[0])
    hist_std = float(hist_slice.rolling(20).std().iloc[-1] or 1e-8)

    base_signal = float(np.tanh(hist_slope / (hist_std + 1e-8)))

    # Divergence detection
    price_window = close.iloc[-5:]
    price_falling = price_window.iloc[-1] < price_window.iloc[0]
    hist_rising = hist_slice.iloc[-1] > hist_slice.iloc[-5]

    # Bearish divergence: price rising, histogram falling
    price_rising = price_window.iloc[-1] > price_window.iloc[0]
    hist_falling = hist_slice.iloc[-1] < hist_slice.iloc[-5]

    divergence_amplifier = 1.0
    if price_falling and hist_rising:
        # Bullish divergence → amplify positive histogram signal
        divergence_amplifier = 1.5
    elif price_rising and hist_falling:
        # Bearish divergence → amplify negative histogram signal
        divergence_amplifier = 1.5

    raw = base_signal * divergence_amplifier
    return float(np.clip(raw, -1.0, 1.0))


def adx_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict:
    """
    ADX directional regime.

    Returns:
        {
            'regime': 'trending' | 'ranging' | 'transition',
            'multiplier': float,
            'direction': float,   # DI+ vs DI- strength in [-1, +1]
        }
    """
    if len(close) < period + 5:
        return {"regime": "ranging", "multiplier": 0.5, "direction": 0.0}

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # True Range
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    pos_dm_s = pd.Series(pos_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
    neg_dm_s = pd.Series(neg_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
    tr_s = tr.ewm(com=period - 1, adjust=False).mean()

    dip = 100.0 * pos_dm_s / (tr_s + 1e-8)
    dim = 100.0 * neg_dm_s / (tr_s + 1e-8)

    dx = 100.0 * (dip - dim).abs() / (dip + dim + 1e-8)
    adx_series = dx.ewm(com=period - 1, adjust=False).mean()

    adx_val = float(adx_series.iloc[-1])
    dip_val = float(dip.iloc[-1])
    dim_val = float(dim.iloc[-1])

    # Regime classification
    if adx_val > 25:
        regime = "trending"
        multiplier = 1.0 + (adx_val - 25) / 100.0  # up to ~1.5 at ADX=75
        multiplier = min(multiplier, 1.5)
    elif adx_val < 20:
        regime = "ranging"
        multiplier = 0.4
    else:
        regime = "transition"
        # Linear interpolation 20→25 maps to 0.4→1.0
        multiplier = 0.4 + (adx_val - 20) / 5.0 * 0.6

    direction_strength = (dip_val - dim_val) / (dip_val + dim_val + 1e-8)

    return {
        "regime": regime,
        "multiplier": float(np.clip(multiplier, 0.0, 1.5)),
        "direction": float(np.clip(direction_strength, -1.0, 1.0)),
    }


def ichimoku_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """
    Ichimoku Kinko Hyo signal.

    Components: tenkan(9), kijun(26), senkou_a, senkou_b(52), chikou.
    Bullish conditions:
      1. close above cloud (both senkou_a and senkou_b)
      2. tenkan > kijun (cross confirmation)
      3. chikou above price 26 bars ago
    signal = (bullish_count - 1.5) / 1.5  → [-1, +1] range
    """
    if len(close) < 52:
        return 0.0

    def _mid(s_h: pd.Series, s_l: pd.Series, n: int) -> pd.Series:
        return (s_h.rolling(n).max() + s_l.rolling(n).min()) / 2.0

    tenkan = _mid(high, low, 9)
    kijun  = _mid(high, low, 26)

    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = _mid(high, low, 52).shift(26)

    # Chikou: current close plotted 26 bars back → compare to price 26 bars ago
    chikou_price_ref = close.shift(26)  # price 26 bars ago

    last_close = close.iloc[-1]
    last_sa = senkou_a.iloc[-1]
    last_sb = senkou_b.iloc[-1]
    last_tenkan = tenkan.iloc[-1]
    last_kijun  = kijun.iloc[-1]
    last_chikou_ref = chikou_price_ref.iloc[-1]

    if any(np.isnan([last_sa, last_sb, last_tenkan, last_kijun, last_chikou_ref])):
        return 0.0

    cloud_top = max(last_sa, last_sb)
    cloud_bot = min(last_sa, last_sb)

    # Conditions (each worth 1 point)
    above_cloud     = 1 if last_close > cloud_top else (-1 if last_close < cloud_bot else 0)
    tenkan_vs_kijun = 1 if last_tenkan > last_kijun else (-1 if last_tenkan < last_kijun else 0)
    chikou_confirm  = 1 if last_close > last_chikou_ref else (-1 if last_close < last_chikou_ref else 0)

    # bullish_count ∈ {0,1,2,3}; convert to signed sum ∈ {-3,...,+3}
    signed_sum = float(above_cloud + tenkan_vs_kijun + chikou_confirm)
    # Map [-3, +3] → [-1, +1]
    return float(np.clip(signed_sum / 3.0, -1.0, 1.0))
