"""
Limit Order Book microstructure signals.
All public functions return float in [-1.0, +1.0] unless otherwise noted.

Inputs:
  bids / asks: list of (price, quantity) tuples, sorted by price descending (bids)
               and ascending (asks).
"""

import math
from typing import List, Tuple


# Type alias
Level = Tuple[float, float]  # (price, quantity)


# ---------------------------------------------------------------------------
# Order Book Imbalance
# ---------------------------------------------------------------------------

def order_book_imbalance(
    bids: List[Level],
    asks: List[Level],
    levels: int = 5,
) -> float:
    """
    Simple order book imbalance across top N levels.

    OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
    Returns float in [-1, +1].
    """
    bid_vol = sum(qty for _, qty in bids[:levels])
    ask_vol = sum(qty for _, qty in asks[:levels])
    total = bid_vol + ask_vol + 1e-8
    return float((bid_vol - ask_vol) / total)


def weighted_obi(
    bids: List[Level],
    asks: List[Level],
    levels: int = 10,
    decay: float = 0.7,
) -> float:
    """
    Exponentially decay-weighted Order Book Imbalance.

    Weight of level i = decay^i (level 0 = best bid/ask gets weight 1.0).
    OBI = (weighted_bid_vol - weighted_ask_vol) / (total_weighted_vol + 1e-8)
    Returns float in [-1, +1].
    """
    weighted_bid = 0.0
    weighted_ask = 0.0
    total_weight = 0.0

    for i in range(min(levels, max(len(bids), len(asks)))):
        w = decay ** i
        if i < len(bids):
            weighted_bid += w * bids[i][1]
        if i < len(asks):
            weighted_ask += w * asks[i][1]
        total_weight += w

    total_vol = weighted_bid + weighted_ask + 1e-8
    return float((weighted_bid - weighted_ask) / total_vol)


def obi_vector(
    bids: List[Level],
    asks: List[Level],
    levels: int = 20,
) -> List[float]:
    """
    Per-level OBI vector.

    Returns list of floats of length `levels`, each entry is the imbalance
    at that level: (bid_qty_i - ask_qty_i) / (bid_qty_i + ask_qty_i + 1e-8).
    """
    result = []
    for i in range(levels):
        bid_qty = bids[i][1] if i < len(bids) else 0.0
        ask_qty = asks[i][1] if i < len(asks) else 0.0
        total = bid_qty + ask_qty + 1e-8
        result.append(float((bid_qty - ask_qty) / total))
    return result


# ---------------------------------------------------------------------------
# Micro Price
# ---------------------------------------------------------------------------

def micro_price(
    best_bid_price: float,
    best_bid_qty: float,
    best_ask_price: float,
    best_ask_qty: float,
) -> float:
    """
    Weighted mid-price (micro price).

    micro_price = (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty)
                  / (best_bid_qty + best_ask_qty + 1e-8)

    Skews toward the side with more quantity.
    """
    total = best_bid_qty + best_ask_qty + 1e-8
    return float(
        (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / total
    )


def micro_price_signal(bids: List[Level], asks: List[Level]) -> float:
    """
    Signal based on micro price deviation from mid price.

    mid = (best_bid + best_ask) / 2
    micro = micro_price(...)
    dev = (micro - mid) / (spread / 2 + 1e-8)
    signal = tanh(dev * 3)
    """
    if not bids or not asks:
        return 0.0

    best_bid_price, best_bid_qty = bids[0]
    best_ask_price, best_ask_qty = asks[0]

    mid = (best_bid_price + best_ask_price) / 2.0
    spread = best_ask_price - best_bid_price
    half_spread = spread / 2.0 + 1e-8

    mp = micro_price(best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
    dev = (mp - mid) / half_spread
    return float(math.tanh(dev * 3.0))


# ---------------------------------------------------------------------------
# VAMP – Volume-weighted Average Mid Price
# ---------------------------------------------------------------------------

def vamp(bids: List[Level], asks: List[Level], depth: int = 10) -> float:
    """
    Volume-weighted Average Mid Price across `depth` levels.

    For each level i, compute mid_i = (bid_price_i + ask_price_i) / 2
    and weight by (bid_qty_i + ask_qty_i).

    Returns the VAMP value (an absolute price level, not a signal).
    If you need a signal, compare to current mid.
    """
    total_weight = 0.0
    weighted_mid = 0.0

    for i in range(min(depth, min(len(bids), len(asks)))):
        bp, bq = bids[i]
        ap, aq = asks[i]
        mid_i = (bp + ap) / 2.0
        weight = bq + aq
        weighted_mid += mid_i * weight
        total_weight += weight

    if total_weight <= 0:
        if bids and asks:
            return (bids[0][0] + asks[0][0]) / 2.0
        return 0.0

    return float(weighted_mid / total_weight)


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

def spread_bps(bids: List[Level], asks: List[Level]) -> float:
    """
    Bid-ask spread in basis points.

    spread_bps = (ask_price - bid_price) / mid_price * 10_000
    """
    if not bids or not asks:
        return float("nan")

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0

    if mid <= 0:
        return float("nan")

    return float((best_ask - best_bid) / mid * 10_000.0)
