"""
providers/polling.py — Probability estimates for political / polling markets.

Fetches JSON from POLLING_URL env var.  Supports two response shapes:

  Shape A (single object):
      {"probability": 0.62}

  Shape B (array of objects with a "question" / "title" field):
      [{"question": "Will Biden win PA?", "probability": 0.58}, ...]

For Shape B, finds the best-matching entry by fuzzy keyword overlap
with the Kalshi market title.

Political keywords trigger this provider.
"""

import logging
import os
import re
import traceback
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_POLLING_URL = os.getenv("POLLING_URL", "")

_POLITICAL_KEYWORDS = [
    "election", "president", "senate", "house", "congress", "governor",
    "mayor", "vote", "poll", "approval", "democrat", "republican",
    "gop", "primary", "candidate", "ballot", "party", "legislation",
    "impeach", "cabinet", "secretary", "justice", "supreme court",
    "biden", "trump", "harris", "political",
]


def _is_political_market(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _POLITICAL_KEYWORDS)


def _keyword_overlap(a: str, b: str) -> float:
    """Return fraction of words in `a` that appear in `b`."""
    words_a = set(re.findall(r"[a-z]+", a.lower()))
    words_b = set(re.findall(r"[a-z]+", b.lower()))
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def _fetch_polling_data() -> Optional[object]:
    if not _POLLING_URL:
        logger.debug("POLLING_URL not configured")
        return None
    try:
        resp = requests.get(_POLLING_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.error("PollingProvider fetch error:\n%s", traceback.format_exc())
        return None


class PollingProvider:
    name = "polling"

    def estimate(self, market: dict) -> Optional[float]:
        title = market.get("title", "") or market.get("question", "")
        if not _is_political_market(title):
            return None

        data = _fetch_polling_data()
        if data is None:
            return None

        # ── Shape A: single object ────────────────────────────────────────────
        if isinstance(data, dict):
            raw = data.get("probability") or data.get("prob")
            if raw is not None:
                prob = float(raw) * 100 if float(raw) <= 1.0 else float(raw)
                logger.debug("PollingProvider (single): %s → %.1f%%", title, prob)
                return round(prob, 1)
            return None

        # ── Shape B: array ────────────────────────────────────────────────────
        if isinstance(data, list):
            best_score = 0.0
            best_prob: Optional[float] = None
            for entry in data:
                entry_title = entry.get("question") or entry.get("title") or ""
                score = _keyword_overlap(title, entry_title)
                if score > best_score:
                    raw = entry.get("probability") or entry.get("prob")
                    if raw is not None:
                        best_score = score
                        best_prob = float(raw) * 100 if float(raw) <= 1.0 else float(raw)

            if best_prob is not None and best_score >= 0.3:
                logger.debug("PollingProvider (array): %s → %.1f%% (match=%.2f)",
                             title, best_prob, best_score)
                return round(best_prob, 1)

        return None
