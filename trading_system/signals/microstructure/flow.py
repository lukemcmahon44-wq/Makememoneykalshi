"""
Order flow signals: CVD, aggressor ratio, trade count surge.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .._common import EPS, tanh_clip
from ...data.store import TradeTick


def cumulative_volume_delta(trades: Sequence[TradeTick]) -> float:
    if not trades:
        return 0.0
    cvd = 0.0
    for t in trades:
        if t.is_buyer_maker:
            cvd -= t.qty   # sell aggressor
        else:
            cvd += t.qty   # buy aggressor
    return float(cvd)


def cvd_zscore_signal(trades: Sequence[TradeTick], window_secs: float = 300,
                      bucket_n: int = 30) -> float:
    """Compute rolling CVD per bucket; z-score the latest bucket."""
    if len(trades) < 30:
        return 0.0
    sorted_trades = sorted(trades, key=lambda t: t.ts)
    end = sorted_trades[-1].ts
    start = end - window_secs * bucket_n
    bucket_size = max(window_secs, 1.0)
    buckets: list[float] = []
    cursor = start
    i = 0
    while cursor < end and i < len(sorted_trades) * 2:
        bucket_cvd = 0.0
        bucket_end = cursor + bucket_size
        while i < len(sorted_trades) and sorted_trades[i].ts < bucket_end:
            t = sorted_trades[i]
            bucket_cvd += -t.qty if t.is_buyer_maker else t.qty
            i += 1
        buckets.append(bucket_cvd)
        cursor = bucket_end
    if len(buckets) < 5:
        return 0.0
    arr = np.asarray(buckets, dtype=float)
    mu = float(np.mean(arr[:-1]))
    sd = float(np.std(arr[:-1], ddof=0))
    if sd < EPS:
        return 0.0
    z = (arr[-1] - mu) / sd
    return float(tanh_clip(z, 2.0))


def aggressor_ratio(trades: Sequence[TradeTick]) -> float:
    if not trades:
        return 0.0
    buys = 0.0; sells = 0.0
    for t in trades:
        if t.is_buyer_maker:
            sells += t.qty
        else:
            buys += t.qty
    total = buys + sells
    if total < EPS:
        return 0.0
    return float(np.clip((buys - sells) / total, -1.0, 1.0))


def trade_intensity_signal(trades: Sequence[TradeTick],
                           recent_secs: float = 60,
                           baseline_secs: float = 600) -> float:
    """Trade count surge: recent vs longer baseline rate."""
    if len(trades) < 50:
        return 0.0
    now = trades[-1].ts
    recent_count = sum(1 for t in trades if now - t.ts < recent_secs)
    baseline_count = sum(1 for t in trades if now - t.ts < baseline_secs)
    if baseline_count == 0:
        return 0.0
    expected_recent = baseline_count * (recent_secs / baseline_secs)
    if expected_recent < 1:
        return 0.0
    ratio = recent_count / expected_recent
    return float(tanh_clip(ratio - 1.0, 2.0))


def composite_flow(trades: Sequence[TradeTick]) -> float:
    if not trades:
        return 0.0
    sig = [
        cvd_zscore_signal(trades),
        aggressor_ratio(trades),
        trade_intensity_signal(trades),
    ]
    return float(np.clip(np.mean(sig), -1.0, 1.0))
