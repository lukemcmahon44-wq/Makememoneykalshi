"""
Volume-based signal generators.
All public functions return float in [-1.0, +1.0] via tanh normalization.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public signal functions
# ---------------------------------------------------------------------------

def obv_signal(close: pd.Series, volume: pd.Series) -> float:
    """
    On-Balance Volume z-score signal.

    OBV = cumulative sum of volume (positive when close up, negative when down).
    signal = tanh(z_score / 2.0) where z_score is OBV vs 20-period rolling mean/std.
    """
    if len(close) < 22:
        return 0.0

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()

    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std()

    last_obv = float(obv.iloc[-1])
    last_mean = float(obv_mean.iloc[-1])
    last_std = float(obv_std.iloc[-1])

    if np.isnan(last_mean) or np.isnan(last_std) or last_std <= 0:
        return 0.0

    z = (last_obv - last_mean) / last_std
    return float(np.tanh(z / 2.0))


def volume_surge_signal(volume: pd.Series) -> float:
    """
    Volume surge signal: current volume vs 20-bar average.

    ratio = current_volume / avg_volume_20
    signal = tanh((ratio - 1) * 2)
    Surge = bullish if price is also rising; standalone surge is neutral direction.
    """
    if len(volume) < 21:
        return 0.0

    avg_vol = volume.rolling(20).mean()
    last_vol = float(volume.iloc[-1])
    last_avg = float(avg_vol.iloc[-1])

    if np.isnan(last_avg) or last_avg <= 0:
        return 0.0

    ratio = last_vol / last_avg
    # tanh((ratio-1)*2): ratio=1 → 0, ratio=2 → tanh(2)≈0.96, ratio=0 → tanh(-2)≈-0.96
    return float(np.tanh((ratio - 1.0) * 2.0))


def mfi_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> float:
    """
    Money Flow Index signal (volume-weighted RSI).

    Typical price = (high + low + close) / 3
    Raw money flow = typical_price * volume
    Money ratio = positive_flow / negative_flow
    MFI = 100 - (100 / (1 + money_ratio))
    signal = -tanh((MFI - 50) / 20)  [mean-reversion: overbought → short]
    """
    if len(close) < period + 2:
        return 0.0

    typical = (high + low + close) / 3.0
    raw_flow = typical * volume

    delta_tp = typical.diff()
    pos_flow = raw_flow.where(delta_tp > 0, 0.0)
    neg_flow = raw_flow.where(delta_tp < 0, 0.0)

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.abs().rolling(period).sum()

    money_ratio = pos_sum / (neg_sum + 1e-8)
    mfi = 100.0 - (100.0 / (1.0 + money_ratio))

    last_mfi = float(mfi.iloc[-1])

    if np.isnan(last_mfi):
        return 0.0

    # Mean-reversion: MFI > 80 = overbought → negative, < 20 = oversold → positive
    raw = -float(np.tanh((last_mfi - 50.0) / 20.0))

    # Boost at extremes
    if last_mfi > 80:
        boost = 1.0 + (last_mfi - 80.0) / 20.0
        raw = -abs(raw) * min(boost, 1.5)
    elif last_mfi < 20:
        boost = 1.0 + (20.0 - last_mfi) / 20.0
        raw = abs(raw) * min(boost, 1.5)

    return float(np.clip(raw, -1.0, 1.0))


def vwap_deviation_signal(close: pd.Series, volume: pd.Series) -> float:
    """
    Session VWAP deviation z-score signal.

    Computes cumulative VWAP from the series start (treated as session start).
    z_score = (close - vwap) / rolling_std(close - vwap, 20)
    signal = -tanh(z_score / 2.0)  [mean-reversion toward VWAP]
    """
    if len(close) < 22:
        return 0.0

    # Cumulative VWAP
    cum_vol = volume.cumsum()
    cum_tpv = (close * volume).cumsum()
    vwap = cum_tpv / (cum_vol + 1e-8)

    deviation = close - vwap
    dev_std = deviation.rolling(20).std()

    last_dev = float(deviation.iloc[-1])
    last_std = float(dev_std.iloc[-1])

    if np.isnan(last_dev) or np.isnan(last_std) or last_std <= 0:
        return 0.0

    z = last_dev / last_std
    return float(np.tanh(-z / 2.0))
