"""
Mean-reversion signal generators.
All public functions return float in [-1.0, +1.0] via tanh normalization.
"""

import numpy as np
import pandas as pd

from .trend import atr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's EMA smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Public signal functions
# ---------------------------------------------------------------------------

def bollinger_signal(close: pd.Series) -> float:
    """
    Bollinger Band mean-reversion signal.

    SMA20, std20, z = (close - SMA20) / std20
    signal = -tanh(z / 2.5)
    Suppress 70% in a squeeze (std < 0.5 * rolling_std_mean).
    """
    if len(close) < 25:
        return 0.0

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    last_sma = sma20.iloc[-1]
    last_std = std20.iloc[-1]

    if np.isnan(last_sma) or np.isnan(last_std) or last_std <= 0:
        return 0.0

    z = (close.iloc[-1] - last_sma) / last_std
    raw = -float(np.tanh(z / 2.5))

    # Squeeze detection: current std vs rolling mean of std over last 50 bars
    std_history = std20.dropna()
    if len(std_history) >= 20:
        std_mean = float(std_history.rolling(20).mean().iloc[-1])
        if not np.isnan(std_mean) and std_mean > 0:
            # Squeeze: current std is less than 50% of its mean → suppress 70%
            if last_std < 0.5 * std_mean:
                raw *= 0.3

    return float(np.clip(raw, -1.0, 1.0))


def rsi_signal(close: pd.Series) -> float:
    """
    RSI signal combining RSI14 and RSI7 with divergence detection.

    Bull divergence: price makes new 20-bar low but RSI does NOT make new RSI low.
    Bear divergence: price makes new 20-bar high but RSI does NOT make new RSI high.
    """
    if len(close) < 30:
        return 0.0

    rsi14 = rsi(close, 14)
    rsi7 = rsi(close, 7)

    r14 = rsi14.iloc[-1]
    r7 = rsi7.iloc[-1]

    if np.isnan(r14) or np.isnan(r7):
        return 0.0

    # Combined RSI: weighted average
    combined_rsi = 0.6 * r14 + 0.4 * r7

    # Base signal: map RSI [0,100] → [-1,+1] with mean-reversion logic
    # RSI=50 → 0, RSI=30 → +1, RSI=70 → -1
    base = -float(np.tanh((combined_rsi - 50.0) / 20.0))

    # Divergence detection over 20-bar lookback
    lookback = min(20, len(close) - 1)
    price_window = close.iloc[-(lookback + 1):]
    rsi14_window = rsi14.iloc[-(lookback + 1):]

    divergence_factor = 1.0

    if len(price_window) >= lookback:
        # Bull divergence: new price low but NOT new RSI low
        price_low_now = close.iloc[-1] <= price_window.min()
        rsi_low_now = rsi14.iloc[-1] <= rsi14_window.dropna().min()

        price_high_now = close.iloc[-1] >= price_window.max()
        rsi_high_now = rsi14.iloc[-1] >= rsi14_window.dropna().max()

        if price_low_now and not rsi_low_now:
            # Bullish divergence: amplify positive signal
            divergence_factor = 1.5
        elif price_high_now and not rsi_high_now:
            # Bearish divergence: amplify negative signal
            divergence_factor = 1.5

    raw = base * divergence_factor
    return float(np.clip(raw, -1.0, 1.0))


def stochastic_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
) -> float:
    """
    Stochastic Oscillator signal.

    %K = (close - lowest_low(k)) / (highest_high(k) - lowest_low(k)) * 100
    %D = SMA(%K, d)
    signal = -tanh((%K - 50) / 20) with oversold/overbought boost.
    """
    if len(close) < k + d + 2:
        return 0.0

    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom = highest_high - lowest_low

    pct_k = (close - lowest_low) / (denom.replace(0, np.nan) + 1e-8) * 100.0
    pct_d = pct_k.rolling(d).mean()

    last_k = pct_k.iloc[-1]
    last_d = pct_d.iloc[-1]

    if np.isnan(last_k) or np.isnan(last_d):
        return 0.0

    # Base mean-reversion signal
    raw = -float(np.tanh((last_k - 50.0) / 20.0))

    # Boost when in extreme zones
    if last_k < 20:  # Oversold → boost bullish signal
        oversold_boost = 1.0 + (20.0 - last_k) / 20.0  # up to 2.0 at K=0
        raw = abs(raw) * min(oversold_boost, 1.5)
    elif last_k > 80:  # Overbought → boost bearish signal
        overbought_boost = 1.0 + (last_k - 80.0) / 20.0  # up to 2.0 at K=100
        raw = -abs(raw) * min(overbought_boost, 1.5)

    # %K crossing %D as confirmation
    if len(pct_k) >= 2 and len(pct_d) >= 2:
        prev_k = pct_k.iloc[-2]
        prev_d = pct_d.iloc[-2]
        if not np.isnan(prev_k) and not np.isnan(prev_d):
            # Bullish cross: K crossed above D
            if prev_k < prev_d and last_k >= last_d and last_k < 40:
                raw = abs(raw) * 1.2
            # Bearish cross: K crossed below D
            elif prev_k > prev_d and last_k <= last_d and last_k > 60:
                raw = -abs(raw) * 1.2

    return float(np.clip(raw, -1.0, 1.0))


def vwap_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    session_vwap: float,
) -> float:
    """
    VWAP deviation mean-reversion signal.

    dev = (close - vwap) / (0.5 * ATR(14))
    signal = -tanh(dev / 2.0)
    Amplify at 2-sigma deviation from VWAP.
    """
    if len(close) < 20:
        return 0.0

    atr14 = atr(high, low, close, 14)
    last_atr = float(atr14.iloc[-1])

    if np.isnan(last_atr) or last_atr <= 0:
        return 0.0

    last_close = float(close.iloc[-1])

    if np.isnan(session_vwap) or session_vwap <= 0:
        # Fall back to rolling VWAP if session_vwap is not valid
        typical_price = (high + low + close) / 3.0
        cumulative_tpv = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        rolling_vwap = cumulative_tpv / (cumulative_vol + 1e-8)
        session_vwap = float(rolling_vwap.iloc[-1])

    dev = (last_close - session_vwap) / (0.5 * last_atr + 1e-8)
    raw = -float(np.tanh(dev / 2.0))

    # Amplify at 2-sigma deviation
    abs_dev = abs(dev)
    if abs_dev >= 2.0:
        amplifier = 1.0 + min((abs_dev - 2.0) * 0.3, 0.5)
        raw = float(np.clip(raw * amplifier, -1.0, 1.0))

    return float(np.clip(raw, -1.0, 1.0))


def williams_r_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float:
    """
    Williams %R mean-reversion signal.

    WR = (highest_high - close) / (highest_high - lowest_low) * -100
    Range: [-100, 0]; -100 = oversold, 0 = overbought
    signal = tanh((WR + 50) / 20)
    """
    if len(close) < period + 1:
        return 0.0

    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    hh = float(highest_high.iloc[-1])
    ll = float(lowest_low.iloc[-1])

    if np.isnan(hh) or np.isnan(ll):
        return 0.0

    denom = hh - ll
    if denom <= 0:
        return 0.0

    last_close = float(close.iloc[-1])
    wr = (hh - last_close) / denom * -100.0  # in [-100, 0]

    # WR + 50 centers around 0 → -50 (oversold) to +50 (overbought)
    # tanh((WR+50)/20): oversold (WR≈-100) → tanh(-50/20)≈-1 → buy
    # But WR is mean-reversion: oversold = positive signal
    signal = float(np.tanh((wr + 50.0) / 20.0))
    # WR near -100 → (−100+50)/20 = −2.5 → tanh = −0.987 (oversold = SELL?)
    # Correct mean-reversion: oversold → BUY → invert
    signal = -signal

    return float(np.clip(signal, -1.0, 1.0))
