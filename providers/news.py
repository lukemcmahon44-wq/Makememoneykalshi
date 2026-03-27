"""
providers/news.py — Probability estimate via news sentiment analysis.

Uses NewsAPI (https://newsapi.org) to fetch headlines related to the
market question, then runs TextBlob sentiment analysis to derive a
rough probability signal.

Logic:
  - Fetch top 20 recent headlines matching the market title keywords
  - Score each headline: positive sentiment → event more likely
  - Map average compound score (-1..+1) to a probability (0..100)
  - Applies only to markets with event-language keywords
"""

import logging
import os
import re
import traceback
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_NEWS_KEY = os.getenv("NEWS_API_KEY", "")
_NEWS_URL = "https://newsapi.org/v2/everything"

_EVENT_KEYWORDS = [
    "will", "launch", "release", "announce", "pass", "approve", "sign",
    "ban", "regulation", "bill", "law", "deal", "merger", "acquisition",
    "ipo", "bankruptcy", "election", "win", "lose", "resign", "appoint",
    "confirm", "vote", "ruling", "verdict", "rate", "hike", "cut",
]

_NOISE_WORDS = {"the", "a", "an", "in", "on", "at", "to", "for", "of",
                "and", "or", "is", "are", "will", "be", "by", "with",
                "this", "that", "it", "its", "as", "from", "up", "down"}


def _is_event_market(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _EVENT_KEYWORDS)


def _build_query(title: str) -> str:
    """Strip stop-words and build a 3–5 word NewsAPI query."""
    words = re.findall(r"[a-z]+", title.lower())
    content_words = [w for w in words if w not in _NOISE_WORDS and len(w) > 2]
    return " ".join(content_words[:5])


def _fetch_headlines(query: str) -> list:
    if not _NEWS_KEY:
        logger.warning("NEWS_API_KEY not set")
        return []
    try:
        resp = requests.get(
            _NEWS_URL,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": _NEWS_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            (a.get("title", "") + " " + (a.get("description") or ""))
            for a in articles
        ]
    except Exception:
        logger.error("NewsProvider fetch error:\n%s", traceback.format_exc())
        return []


def _sentiment_to_prob(headlines: list) -> float:
    """
    Average TextBlob polarity across headlines, then map [-1, 1] → [20, 80].

    We cap at 20/80 because sentiment alone is a weak signal — it nudges the
    probability but doesn't make extreme claims.
    """
    try:
        from textblob import TextBlob
    except ImportError:
        logger.warning("textblob not installed; NewsProvider disabled")
        return 50.0

    if not headlines:
        return 50.0

    scores = []
    for h in headlines:
        if h.strip():
            blob = TextBlob(h)
            scores.append(blob.sentiment.polarity)

    if not scores:
        return 50.0

    avg = sum(scores) / len(scores)
    # Map [-1, 1] → [20, 80]
    prob = 50.0 + avg * 30.0
    return round(max(20.0, min(80.0, prob)), 1)


class NewsProvider:
    name = "news"

    def estimate(self, market: dict) -> Optional[float]:
        title = market.get("title", "") or market.get("question", "")
        if not _is_event_market(title):
            return None

        query = _build_query(title)
        if not query:
            return None

        headlines = _fetch_headlines(query)
        if not headlines:
            return None

        prob = _sentiment_to_prob(headlines)
        logger.debug("NewsProvider: '%s' | query='%s' | headlines=%d | prob=%.1f",
                     title, query, len(headlines), prob)
        return prob
