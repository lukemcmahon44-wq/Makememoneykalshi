"""
Order flow signals derived from aggregated trade data.

Trade record format (dict or object with attributes):
  - price        : float
  - qty          : float   (trade size)
  - isBuyerMaker : bool    (True if the buyer is the maker → sell aggressor)
"""

import math
import numpy as np
from typing import List, Union


# Type alias: a trade can be a dict or any object with the above attributes.
Trade = dict  # {'price': float, 'qty': float, 'isBuyerMaker': bool}


def _get(trade: Trade, key: str):
    """Retrieve field from dict or object."""
    if isinstance(trade, dict):
        return trade[key]
    return getattr(trade, key)


# ---------------------------------------------------------------------------
# Cumulative Volume Delta
# ---------------------------------------------------------------------------

def cvd_signal(recent_trades: List[Trade], lookback: int = 50) -> float:
    """
    Cumulative Volume Delta signal.

    buy_volume  = volume of trades where NOT isBuyerMaker (taker is buyer = buy aggression)
    sell_volume = volume of trades where isBuyerMaker     (taker is seller = sell aggression)

    CVD = buy_volume - sell_volume  (over the full series)

    cvd_zscore = (cvd - mean(historical_cvds)) / (std(historical_cvds) + 1e-8)
    signal = tanh(cvd_zscore / 2.0)

    Also amplifies on CVD/price divergence:
      - price rising but CVD falling → amplify bearish signal
      - price falling but CVD rising → amplify bullish signal
    """
    if len(recent_trades) < 2:
        return 0.0

    trades = recent_trades[-max(lookback * 2, len(recent_trades)):]
    n = len(trades)

    # Compute rolling CVD at each point
    cvd_series = []
    running_cvd = 0.0
    for trade in trades:
        qty = float(_get(trade, "qty"))
        is_buyer_maker = bool(_get(trade, "isBuyerMaker"))
        if is_buyer_maker:
            running_cvd -= qty   # sell aggression
        else:
            running_cvd += qty   # buy aggression
        cvd_series.append(running_cvd)

    current_cvd = cvd_series[-1]

    # Historical statistics for z-score
    # Use rolling windows of `lookback` bars
    if len(cvd_series) >= lookback:
        hist = np.array(cvd_series[-(lookback + 1):-1])
        hist_mean = float(np.mean(hist))
        hist_std = float(np.std(hist))
    else:
        hist = np.array(cvd_series[:-1]) if len(cvd_series) > 1 else np.array([0.0])
        hist_mean = float(np.mean(hist))
        hist_std = float(np.std(hist))

    cvd_zscore = (current_cvd - hist_mean) / (hist_std + 1e-8)
    signal = math.tanh(cvd_zscore / 2.0)

    # Divergence detection
    price_series = [float(_get(t, "price")) for t in trades[-lookback:]]
    cvd_window = cvd_series[-lookback:]

    if len(price_series) >= 5 and len(cvd_window) >= 5:
        price_slope = price_series[-1] - price_series[0]
        cvd_slope = cvd_window[-1] - cvd_window[0]

        # Divergence: price and CVD moving in opposite directions
        if price_slope > 0 and cvd_slope < 0:
            # Bearish divergence: price up, CVD down → amplify bearish
            signal = -abs(signal) * 1.4
        elif price_slope < 0 and cvd_slope > 0:
            # Bullish divergence: price down, CVD up → amplify bullish
            signal = abs(signal) * 1.4

    return float(np.clip(signal, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Aggressor Ratio
# ---------------------------------------------------------------------------

def aggressor_ratio(recent_trades: List[Trade]) -> float:
    """
    Buy aggressor ratio normalized to [-1, +1].

    buy_vol  = total volume where taker is buyer (isBuyerMaker = False)
    sell_vol = total volume where taker is seller (isBuyerMaker = True)

    ratio = buy_vol / (buy_vol + sell_vol)   → [0, 1]
    signal = (ratio - 0.5) * 2               → [-1, +1]
    """
    if not recent_trades:
        return 0.0

    buy_vol = 0.0
    sell_vol = 0.0

    for trade in recent_trades:
        qty = float(_get(trade, "qty"))
        is_buyer_maker = bool(_get(trade, "isBuyerMaker"))
        if is_buyer_maker:
            sell_vol += qty
        else:
            buy_vol += qty

    total = buy_vol + sell_vol
    if total <= 0:
        return 0.0

    ratio = buy_vol / total   # [0, 1]
    return float((ratio - 0.5) * 2.0)   # [-1, +1]
