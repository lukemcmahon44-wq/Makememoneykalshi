"""
Limit order book signals: OBI, weighted OBI, micro-price, VAMP.

Reference: Kolm, Turiel, Westray (2021); Stoikov (2018).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .._common import EPS, tanh_clip
from ...data.store import LOBSnapshot


def order_book_imbalance(bids: List[Tuple[float, float]],
                         asks: List[Tuple[float, float]],
                         levels: int = 5) -> float:
    if not bids or not asks:
        return 0.0
    bid_vol = sum(q for _, q in bids[:levels])
    ask_vol = sum(q for _, q in asks[:levels])
    total = bid_vol + ask_vol
    if total < EPS:
        return 0.0
    return float((bid_vol - ask_vol) / total)


def weighted_obi(bids: List[Tuple[float, float]],
                 asks: List[Tuple[float, float]],
                 levels: int = 10, decay: float = 0.7) -> float:
    if not bids or not asks:
        return 0.0
    weights = [decay ** i for i in range(levels)]
    bid_vol = sum(w * q for w, (_, q) in zip(weights, bids[:levels]))
    ask_vol = sum(w * q for w, (_, q) in zip(weights, asks[:levels]))
    total = bid_vol + ask_vol
    if total < EPS:
        return 0.0
    return float((bid_vol - ask_vol) / total)


def obi_vector(bids: List[Tuple[float, float]],
               asks: List[Tuple[float, float]]) -> List[float]:
    return [order_book_imbalance(bids, asks, n) for n in (1, 2, 5, 10, 20)]


def micro_price(best_bid_price: float, best_bid_qty: float,
                best_ask_price: float, best_ask_qty: float) -> float:
    total = best_bid_qty + best_ask_qty
    if total < EPS:
        return (best_bid_price + best_ask_price) / 2
    return ((best_ask_price * best_bid_qty + best_bid_price * best_ask_qty)
            / total)


def micro_price_signal(lob: LOBSnapshot) -> float:
    if not lob.bids or not lob.asks:
        return 0.0
    bp, bq = lob.bids[0]; ap, aq = lob.asks[0]
    mid = (bp + ap) / 2
    spread = ap - bp
    if spread < EPS:
        return 0.0
    micro = micro_price(bp, bq, ap, aq)
    dev = (micro - mid) / (0.5 * spread)
    return float(tanh_clip(dev * 3, 1.0))


def vamp(lob: LOBSnapshot, depth: int = 10) -> float:
    if not lob.bids or not lob.asks:
        return 0.0
    mid = (lob.bids[0][0] + lob.asks[0][0]) / 2
    weighted_bid = 0.0; weighted_ask = 0.0
    for p, q in lob.bids[:depth]:
        weighted_bid += q / max(abs(mid - p), EPS)
    for p, q in lob.asks[:depth]:
        weighted_ask += q / max(abs(p - mid), EPS)
    denom = weighted_bid + weighted_ask
    if denom < EPS:
        return mid
    return ((lob.bids[0][0] * weighted_ask + lob.asks[0][0] * weighted_bid)
            / denom)


def vamp_signal(lob: LOBSnapshot, depth: int = 10) -> float:
    if not lob.bids or not lob.asks:
        return 0.0
    v = vamp(lob, depth)
    mid = (lob.bids[0][0] + lob.asks[0][0]) / 2
    spread = lob.asks[0][0] - lob.bids[0][0]
    if spread < EPS:
        return 0.0
    dev = (v - mid) / (0.5 * spread)
    return float(tanh_clip(dev, 1.0))


def lob_composite(lob: LOBSnapshot) -> float:
    if not lob.bids or not lob.asks:
        return 0.0
    obi_5 = order_book_imbalance(lob.bids, lob.asks, 5)
    obi_w = weighted_obi(lob.bids, lob.asks, 10)
    mp_sig = micro_price_signal(lob)
    vamp_sig = vamp_signal(lob)
    return float(np.clip(0.30 * obi_5 + 0.20 * obi_w + 0.30 * mp_sig
                         + 0.20 * vamp_sig, -1.0, 1.0))
