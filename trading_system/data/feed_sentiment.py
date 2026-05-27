"""
Sentiment data feed.

Three sources:
  1. Reddit (PRAW) — hot posts from financial/crypto subreddits, scored
     with VADER, weighted by upvotes and recency, aggregated per ticker.
  2. Alpha Vantage — NEWS_SENTIMENT endpoint; budgeted to 25 calls/day.
  3. Alternative.me Fear & Greed index.

All results land in a shared :class:`SentimentDataStore`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "investing",
    "stocks",
    "Bitcoin",
    "CryptoCurrency",
    "algotrading",
]
REDDIT_POST_LIMIT = 50          # hot posts per subreddit
REDDIT_REFRESH_SECONDS = 600    # 10 minutes

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
AV_DAILY_BUDGET = 25            # max API calls per day
AV_REFRESH_SECONDS = 86400 // AV_DAILY_BUDGET  # spread calls over 24h

FEAR_GREED_URL = "https://api.alternative.me/fng/"
FEAR_GREED_REFRESH_SECONDS = 1800  # 30 minutes

# Ticker extraction patterns
_TICKER_RE = re.compile(r'\b([A-Z]{2,5})\b')
_DOLLAR_TICKER_RE = re.compile(r'\$([A-Z]{2,5})\b')

# Words that are common English words and should not be treated as tickers
_STOPWORDS = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "HAD", "HIM", "HIS", "HOW", "ITS",
    "NOW", "OLD", "SEE", "TWO", "WHO", "BOY", "DID", "ITS", "LET", "PUT",
    "SAY", "SHE", "TOO", "USE", "ETF", "CEO", "USD", "IPO", "OTC", "ATH",
    "ATL", "IMO", "FUD", "WSB", "YOY", "QOQ", "MOM", "GDP", "FED", "IMF",
    "SEC", "IRS", "VAT", "TAX", "NAV", "AUM", "SPX", "NDX", "VIX", "DOW",
    "OIL", "GAS", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "HKD", "CNY",
    "BRL", "RUB", "INR", "MXN", "KRW", "IDR", "TRY", "EPS", "ROE", "ROA",
    "P&L", "YTD", "QTD", "MTD", "ALL", "NEW", "BUY", "SELL", "HOLD", "PUT",
    "CALL", "BULL", "BEAR", "PUTS", "CASH", "RISK",
})

# Sentiment VWAP: 4-hour half-life
HALF_LIFE_HOURS = 4.0
DECAY_LAMBDA = math.log(2) / HALF_LIFE_HOURS


# ---------------------------------------------------------------------------
# Sentiment data store
# ---------------------------------------------------------------------------

class SentimentDataStore:
    """
    Thread-safe (asyncio) container for per-asset sentiment scores.

    Structure::

        {
          asset: {
            "reddit": float [-1, +1],
            "news":   float [-1, +1],
            "fear_greed": float [-1, +1],
          }
        }
    """

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Optional[float]]] = defaultdict(
            lambda: {"reddit": None, "news": None, "fear_greed": None}
        )
        self._lock = asyncio.Lock()
        self._fear_greed: Optional[float] = None
        self._fear_greed_updated: float = 0.0

    async def set_reddit(self, asset: str, score: float) -> None:
        async with self._lock:
            self._data[asset]["reddit"] = score

    async def set_news(self, asset: str, score: float) -> None:
        async with self._lock:
            self._data[asset]["news"] = score

    async def set_fear_greed(self, score: float) -> None:
        async with self._lock:
            self._fear_greed = score
            self._fear_greed_updated = time.time()
            # Propagate to every asset already in the store
            for d in self._data.values():
                d["fear_greed"] = score

    def get(self, asset: str) -> Dict[str, Optional[float]]:
        d = self._data.get(asset, {})
        return {
            "reddit": d.get("reddit"),
            "news": d.get("news"),
            "fear_greed": self._fear_greed,
        }

    def get_fear_greed(self) -> Optional[float]:
        return self._fear_greed

    def all_assets(self) -> List[str]:
        return list(self._data.keys())

    def snapshot(self) -> Dict[str, Dict[str, Optional[float]]]:
        result = {}
        for asset, d in self._data.items():
            result[asset] = {
                "reddit": d.get("reddit"),
                "news": d.get("news"),
                "fear_greed": self._fear_greed,
            }
        return result


# ---------------------------------------------------------------------------
# Reddit sentiment helpers
# ---------------------------------------------------------------------------

def _extract_tickers(text: str) -> List[str]:
    """Extract uppercase ticker-like tokens from *text*, filtering stopwords."""
    found = set()
    # $TICKER pattern — high confidence
    for m in _DOLLAR_TICKER_RE.finditer(text):
        t = m.group(1).upper()
        if t not in _STOPWORDS:
            found.add(t)
    # Bare uppercase words
    for m in _TICKER_RE.finditer(text):
        t = m.group(1).upper()
        if t not in _STOPWORDS:
            found.add(t)
    return list(found)


def _decay_weight(created_utc: float) -> float:
    """Return the recency weight [0, 1] using exponential decay."""
    age_hours = (time.time() - created_utc) / 3600.0
    return math.exp(-DECAY_LAMBDA * age_hours)


def _post_weight(score: int, created_utc: float) -> float:
    """Combined weight: upvote^0.5 * recency_decay."""
    upvote_weight = math.sqrt(max(score, 1))
    return upvote_weight * _decay_weight(created_utc)


class _RedditScorer:
    """
    Pulls PRAW posts from configured subreddits, scores with VADER, and
    maintains a rolling 24-hour buffer per ticker for baseline computation.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddits: List[str] = REDDIT_SUBREDDITS,
        post_limit: int = REDDIT_POST_LIMIT,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddits = subreddits
        self.post_limit = post_limit

        # ticker -> deque of (weighted_score, weight, timestamp)
        self._history: Dict[str, Deque[Tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=2000)
        )
        self._vader = None  # lazy init

    def _get_vader(self):
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    def _get_reddit(self):
        import praw  # type: ignore
        return praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

    def fetch_and_score(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Pull hot posts from all subreddits, extract tickers, and score.

        Returns dict of ``ticker -> [(weighted_score, weight, timestamp)]``
        for this batch.
        """
        vader = self._get_vader()
        reddit = self._get_reddit()
        batch: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

        for sub_name in self.subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.hot(limit=self.post_limit):
                    text = f"{post.title} {post.selftext or ''}"
                    tickers = _extract_tickers(text)
                    if not tickers:
                        continue
                    compound = vader.polarity_scores(text)["compound"]  # [-1, +1]
                    weight = _post_weight(post.score, float(post.created_utc))
                    ts = float(post.created_utc)
                    for ticker in tickers:
                        batch[ticker].append((compound, weight, ts))
                        self._history[ticker].append((compound, weight, ts))
            except Exception as exc:
                logger.warning("[sentiment/reddit] Error reading r/%s: %s", sub_name, exc)

        return batch

    def compute_reddit_sentiment(self, ticker: str) -> float:
        """
        Compute a z-score comparing the current 1-hour weighted average to the
        24-hour baseline.

        Returns a float in ``[-1, +1]`` (clipped).
        """
        now = time.time()
        window_1h = now - 3600
        window_24h = now - 86400

        history = list(self._history.get(ticker, []))

        def weighted_avg(entries: List[Tuple[float, float, float]]) -> Optional[float]:
            total_w = sum(w for _, w, _ in entries)
            if total_w == 0:
                return None
            return sum(s * w for s, w, _ in entries) / total_w

        entries_1h = [(s, w, t) for s, w, t in history if t >= window_1h]
        entries_24h = [(s, w, t) for s, w, t in history if t >= window_24h]

        avg_1h = weighted_avg(entries_1h)
        avg_24h = weighted_avg(entries_24h)

        if avg_1h is None:
            return 0.0

        if avg_24h is None or len(entries_24h) < 3:
            # Not enough baseline — return raw 1h score
            return float(max(-1.0, min(1.0, avg_1h)))

        # Simple z-score: (current - baseline) / std of 24h scores
        scores_24h = [s for s, _, _ in entries_24h]
        std_24h = stdev(scores_24h) if len(scores_24h) >= 2 else 0.1
        if std_24h < 1e-6:
            std_24h = 0.1
        z = (avg_1h - avg_24h) / std_24h
        # Clip z-score to [-3, +3] then normalise to [-1, +1]
        z_clipped = max(-3.0, min(3.0, z))
        return float(z_clipped / 3.0)


# ---------------------------------------------------------------------------
# Alpha Vantage news sentiment
# ---------------------------------------------------------------------------

class _AlphaVantageNewsFetcher:
    """
    Fetches news sentiment from Alpha Vantage NEWS_SENTIMENT endpoint.
    Budgeted to at most AV_DAILY_BUDGET calls per day.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._call_timestamps: Deque[float] = deque(maxlen=AV_DAILY_BUDGET)

    def _budget_ok(self) -> bool:
        """Return True if we have remaining daily budget."""
        now = time.time()
        cutoff = now - 86400
        recent = [t for t in self._call_timestamps if t >= cutoff]
        return len(recent) < AV_DAILY_BUDGET

    async def fetch_news_sentiment(self, ticker: str, session: aiohttp.ClientSession) -> float:
        """
        Fetch and average the news sentiment score for *ticker*.

        Returns a float in ``[-1, +1]``.
        """
        if not self.api_key:
            return 0.0

        if not self._budget_ok():
            logger.warning("[sentiment/av] Daily budget exhausted; skipping %s.", ticker)
            return 0.0

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self.api_key,
            "limit": "50",
            "sort": "RELEVANCE",
        }
        try:
            async with session.get(
                ALPHA_VANTAGE_BASE,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            self._call_timestamps.append(time.time())

            feed_items = data.get("feed", [])
            if not feed_items:
                return 0.0

            scores: List[float] = []
            for item in feed_items:
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker", "").upper() == ticker.upper():
                        try:
                            scores.append(float(ts["ticker_sentiment_score"]))
                        except (KeyError, ValueError):
                            pass

            if not scores:
                return 0.0

            # Average sentiment score (already in -1..+1 range from AV)
            avg = mean(scores)
            return float(max(-1.0, min(1.0, avg)))

        except Exception as exc:
            logger.warning("[sentiment/av] Error fetching news for %s: %s", ticker, exc)
            return 0.0


# ---------------------------------------------------------------------------
# Fear & Greed fetcher
# ---------------------------------------------------------------------------

async def _fetch_fear_greed(session: aiohttp.ClientSession) -> Optional[float]:
    """
    Fetch the current Alternative.me Fear & Greed index value.

    Returns a float in ``[-1, +1]``:
      - raw value < 20  → mapped to +1  (extreme fear = buy signal)
      - raw value > 80  → mapped to -1  (extreme greed = sell signal)
      - linear scaling in between
    """
    try:
        async with session.get(
            FEAR_GREED_URL,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        raw_value = int(data["data"][0]["value"])  # 0..100
        # Normalise: 0..100 -> +1..-1 (low fear/greed index = buy, high = sell)
        # We invert so that extreme fear (<20) gives +1, extreme greed (>80) gives -1
        normalised = (50.0 - raw_value) / 50.0   # 100 -> -1, 0 -> +1
        return float(max(-1.0, min(1.0, normalised)))
    except Exception as exc:
        logger.warning("[sentiment/fg] Failed to fetch fear & greed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main SentimentFeed class
# ---------------------------------------------------------------------------

class SentimentFeed:
    """
    Orchestrates Reddit, Alpha Vantage, and Fear & Greed sentiment collection.

    Parameters
    ----------
    store :
        Shared :class:`SentimentDataStore`.  Created if not provided.
    reddit_client_id, reddit_client_secret, reddit_user_agent :
        PRAW credentials.  Fall back to env vars ``REDDIT_CLIENT_ID``,
        ``REDDIT_CLIENT_SECRET``, ``REDDIT_USER_AGENT``.
    av_api_key :
        Alpha Vantage API key.  Falls back to ``ALPHA_VANTAGE_KEY`` env var.
    assets :
        List of asset/ticker strings to track for AV news sentiment.
    """

    def __init__(
        self,
        store: Optional[SentimentDataStore] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None,
        av_api_key: Optional[str] = None,
        assets: Optional[List[str]] = None,
    ) -> None:
        self.store = store or SentimentDataStore()

        self._reddit_client_id = reddit_client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self._reddit_client_secret = reddit_client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self._reddit_user_agent = reddit_user_agent or os.environ.get("REDDIT_USER_AGENT", "trading_system/1.0")

        self._av_key = av_api_key or os.environ.get("ALPHA_VANTAGE_KEY", "")

        self.assets: List[str] = assets or []

        self._reddit_scorer: Optional[_RedditScorer] = None
        self._av_fetcher: Optional[_AlphaVantageNewsFetcher] = None
        self._running = False

        # AV asset rotation cursor (spread budget across assets)
        self._av_cursor = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment(self, asset: str) -> Dict[str, Optional[float]]:
        """Return the latest {reddit, news, fear_greed} for *asset*."""
        return self.store.get(asset)

    def compute_reddit_sentiment(self, ticker: str) -> float:
        """Compute the weighted Reddit sentiment z-score for *ticker*."""
        if self._reddit_scorer is None:
            return 0.0
        return self._reddit_scorer.compute_reddit_sentiment(ticker)

    async def fetch_news_sentiment(self, ticker: str) -> float:
        """Fetch AV news sentiment for *ticker* (budgeted)."""
        if self._av_fetcher is None or not self._av_key:
            return 0.0
        async with aiohttp.ClientSession() as session:
            score = await self._av_fetcher.fetch_news_sentiment(ticker, session)
        return score

    def get_fear_greed(self) -> Optional[float]:
        """Return the last Fear & Greed index value mapped to [-1, +1]."""
        return self.store.get_fear_greed()

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------

    async def refresh_loop(self) -> None:
        """
        Launch three concurrent refresh loops:
        - Reddit (every 10 min)
        - Alpha Vantage news (budgeted, rotated across assets)
        - Fear & Greed (every 30 min)
        """
        self._running = True
        self._init_scorers()

        tasks = [
            asyncio.create_task(self._reddit_loop(), name="sentiment-reddit"),
            asyncio.create_task(self._av_loop(), name="sentiment-av"),
            asyncio.create_task(self._fear_greed_loop(), name="sentiment-fg"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("[sentiment] Refresh loop cancelled.")
            self._running = False
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Sub-loops
    # ------------------------------------------------------------------

    async def _reddit_loop(self) -> None:
        while self._running:
            try:
                await self._refresh_reddit()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[sentiment/reddit] Loop error: %s", exc, exc_info=True)
            try:
                await asyncio.sleep(REDDIT_REFRESH_SECONDS)
            except asyncio.CancelledError:
                return

    async def _av_loop(self) -> None:
        """Rotate across all assets, respecting the daily budget."""
        while self._running:
            if self._av_fetcher and self.assets:
                ticker = self.assets[self._av_cursor % len(self.assets)]
                self._av_cursor += 1
                try:
                    async with aiohttp.ClientSession() as session:
                        score = await self._av_fetcher.fetch_news_sentiment(ticker, session)
                    await self.store.set_news(ticker, score)
                    logger.debug("[sentiment/av] %s news score = %.3f", ticker, score)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[sentiment/av] Error for %s: %s", ticker, exc)
            try:
                await asyncio.sleep(AV_REFRESH_SECONDS)
            except asyncio.CancelledError:
                return

    async def _fear_greed_loop(self) -> None:
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    score = await _fetch_fear_greed(session)
                if score is not None:
                    await self.store.set_fear_greed(score)
                    logger.debug("[sentiment/fg] Fear & Greed = %.3f", score)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[sentiment/fg] Loop error: %s", exc, exc_info=True)
            try:
                await asyncio.sleep(FEAR_GREED_REFRESH_SECONDS)
            except asyncio.CancelledError:
                return

    # ------------------------------------------------------------------
    # Reddit refresh (sync PRAW call offloaded to executor)
    # ------------------------------------------------------------------

    async def _refresh_reddit(self) -> None:
        if self._reddit_scorer is None:
            logger.debug("[sentiment/reddit] No PRAW credentials; skipping.")
            return

        loop = asyncio.get_event_loop()
        try:
            batch = await loop.run_in_executor(
                None, self._reddit_scorer.fetch_and_score
            )
        except Exception as exc:
            logger.error("[sentiment/reddit] fetch_and_score error: %s", exc)
            return

        # Update sentiment for every ticker we know about
        known_assets = set(self.assets)
        for ticker, entries in batch.items():
            score = self._reddit_scorer.compute_reddit_sentiment(ticker)
            await self.store.set_reddit(ticker, score)
            if ticker not in known_assets:
                logger.debug("[sentiment/reddit] Discovered ticker %s (score=%.3f)", ticker, score)

        logger.info(
            "[sentiment/reddit] Processed %d tickers from Reddit batch.",
            len(batch),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_scorers(self) -> None:
        if self._reddit_client_id and self._reddit_client_secret:
            self._reddit_scorer = _RedditScorer(
                client_id=self._reddit_client_id,
                client_secret=self._reddit_client_secret,
                user_agent=self._reddit_user_agent,
            )
        else:
            logger.warning(
                "[sentiment] PRAW credentials missing; Reddit sentiment disabled."
            )

        if self._av_key:
            self._av_fetcher = _AlphaVantageNewsFetcher(api_key=self._av_key)
        else:
            logger.warning(
                "[sentiment] ALPHA_VANTAGE_KEY missing; news sentiment disabled."
            )
