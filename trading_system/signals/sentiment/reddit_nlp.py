"""
Reddit sentiment signal: VADER scores weighted by upvotes^0.5 * recency_decay.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from .._common import EPS, tanh_clip


def reddit_signal_for_asset(posts: List[Dict[str, Any]],
                            hot_decay_secs: float = 3600 * 4) -> float:
    """Compute weighted sentiment score in [-1, +1]."""
    if not posts:
        return 0.0
    now = time.time()
    weighted_sum = 0.0
    weights = 0.0
    for p in posts:
        vader = float(p.get("vader", 0))
        if vader == 0:
            continue
        score = max(int(p.get("score", 1)), 1)
        age = max(now - float(p.get("created_utc", now)), 1)
        decay = float(np.exp(-age / hot_decay_secs))
        weight = (score ** 0.5) * decay
        weighted_sum += vader * weight
        weights += weight
    if weights < EPS:
        return 0.0
    raw = weighted_sum / weights
    return float(tanh_clip(raw, 0.5))


def reddit_momentum(posts_hist_1h: List[Dict[str, Any]],
                    posts_hist_4h: List[Dict[str, Any]]) -> float:
    """1h sentiment vs 4h baseline."""
    s1h = reddit_signal_for_asset(posts_hist_1h)
    s4h = reddit_signal_for_asset(posts_hist_4h)
    return float(np.clip(s1h - s4h, -1.0, 1.0))


def mention_surge(posts_recent: List, baseline_count: float) -> float:
    """Surge of new mentions vs baseline → attention signal."""
    n = len(posts_recent)
    if baseline_count < 1:
        return 0.0
    ratio = n / baseline_count
    return float(tanh_clip(ratio - 1.0, 3.0))
