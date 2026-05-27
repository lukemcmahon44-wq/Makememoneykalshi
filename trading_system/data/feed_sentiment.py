"""
Sentiment data feed: Reddit (PRAW) + Alpha Vantage news.

Provides:
  reddit_posts[asset]   recent posts with VADER scores
  news_items[asset]     news with headline + summary
Consumers (signals.sentiment.*) handle FinBERT scoring.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from typing import Any, Dict, List

import aiohttp

from ..core.logger import get_logger

log = get_logger(__name__)

SUBREDDITS = [
    "wallstreetbets", "investing", "stocks", "options",
    "Bitcoin", "ethereum", "CryptoCurrency", "algotrading",
]

TICKER_RE = re.compile(r"\$?([A-Z]{2,6})\b")


class SentimentFeed:
    def __init__(self, reddit_client_id: str, reddit_secret: str,
                 reddit_user_agent: str, alpha_vantage_key: str,
                 universe: List[str], refresh_secs: int = 600):
        self.reddit_client_id = reddit_client_id
        self.reddit_secret = reddit_secret
        self.reddit_user_agent = reddit_user_agent
        self.alpha_vantage_key = alpha_vantage_key
        self.universe = [u.upper() for u in universe]
        self.refresh_secs = refresh_secs
        self.reddit_posts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.news_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._reddit = None
        self._vader = None
        self._stop = False

    def _init_clients(self) -> None:
        try:
            import praw  # type: ignore
            self._reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_secret,
                user_agent=self.reddit_user_agent,
                check_for_async=False,
            )
            self._reddit.read_only = True
        except Exception as e:
            log.warning(f"Reddit client init failed: {e}")
            self._reddit = None
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
            self._vader = SentimentIntensityAnalyzer()
        except Exception as e:
            log.warning(f"VADER init failed (need nltk + vader_lexicon): {e}")
            self._vader = None

    async def run(self) -> None:
        self._init_clients()
        async with aiohttp.ClientSession() as session:
            while not self._stop:
                try:
                    await self._refresh_reddit()
                    await self._refresh_news(session)
                except Exception as e:
                    log.error(f"Sentiment refresh error: {e}")
                await asyncio.sleep(self.refresh_secs)

    async def _refresh_reddit(self) -> None:
        if self._reddit is None:
            return
        loop = asyncio.get_event_loop()
        try:
            posts = await loop.run_in_executor(None, self._pull_reddit_posts)
            scored = []
            for p in posts:
                text = f"{p['title']}. {p['selftext'][:500]}"
                vader_score = self._score_vader(text)
                p["vader"] = vader_score
                scored.append(p)
            # Map posts to tickers
            new_posts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for p in scored:
                tickers = TICKER_RE.findall(p["title"] + " " + p["selftext"][:200])
                tickers = [t for t in tickers if t in self.universe]
                for t in tickers:
                    new_posts[t].append(p)
            self.reddit_posts = new_posts
            log.info(f"Reddit: scored {len(scored)} posts, mapped to "
                     f"{len(new_posts)} tickers")
        except Exception as e:
            log.warning(f"Reddit pull failed: {e}")

    def _pull_reddit_posts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sub in SUBREDDITS:
            try:
                for post in self._reddit.subreddit(sub).hot(limit=25):
                    out.append({
                        "title": post.title or "",
                        "selftext": (post.selftext or "")[:1000],
                        "score": int(post.score or 0),
                        "subreddit": sub,
                        "created_utc": float(post.created_utc or time.time()),
                    })
            except Exception as e:
                log.warning(f"Reddit subreddit {sub} pull failed: {e}")
        return out

    def _score_vader(self, text: str) -> float:
        if self._vader is None:
            return 0.0
        try:
            return float(self._vader.polarity_scores(text).get("compound", 0))
        except Exception:
            return 0.0

    async def _refresh_news(self, session: aiohttp.ClientSession) -> None:
        if not self.alpha_vantage_key:
            return
        # 25 calls/day - rotate one ticker per refresh
        try:
            idx = int(time.time() // 3600) % max(1, len(self.universe))
            ticker = self.universe[idx]
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "apikey": self.alpha_vantage_key,
                "limit": "30",
            }
            async with session.get("https://www.alphavantage.co/query",
                                    params=params, timeout=15) as r:
                if r.status != 200:
                    return
                data = await r.json()
            feed = data.get("feed", [])
            items: List[Dict[str, Any]] = []
            for item in feed:
                items.append({
                    "headline": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "timestamp": _parse_av_time(item.get("time_published", "")),
                    "av_sentiment": float(item.get("overall_sentiment_score", 0) or 0),
                })
            if items:
                self.news_items[ticker] = items
                log.info(f"Alpha Vantage news: {ticker} -> {len(items)} items")
        except Exception as e:
            log.warning(f"News refresh error: {e}")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "reddit_posts": dict(self.reddit_posts),
            "news_items": dict(self.news_items),
        }

    def stop(self) -> None:
        self._stop = True


def _parse_av_time(t: str) -> float:
    if not t:
        return time.time()
    try:
        import datetime as dt
        d = dt.datetime.strptime(t, "%Y%m%dT%H%M%S")
        return d.replace(tzinfo=dt.timezone.utc).timestamp()
    except Exception:
        return time.time()
