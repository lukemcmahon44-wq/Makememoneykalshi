"""
Reddit NLP sentiment signals using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Post format (dict):
    'title'      : str   – post title
    'body'       : str   – post selftext (may be empty)
    'upvotes'    : int   – net upvotes
    'created_utc': float – Unix timestamp of post creation
"""

import math
import numpy as np
from typing import List, Tuple

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# VADER scorer
# ---------------------------------------------------------------------------

class VaderScorer:
    """
    Wrapper around vaderSentiment SentimentIntensityAnalyzer.
    """

    def __init__(self) -> None:
        if _VADER_AVAILABLE:
            self._analyzer = SentimentIntensityAnalyzer()
        else:
            self._analyzer = None

    def score(self, text: str) -> float:
        """
        Score a single text string.

        Returns
        -------
        float in [-1, +1]  – compound VADER sentiment score.
        """
        if not self._analyzer or not text:
            return 0.0
        scores = self._analyzer.polarity_scores(str(text))
        return float(scores["compound"])

    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score a list of texts.

        Returns
        -------
        list[float] in [-1, +1] per text.
        """
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# Reddit signal
# ---------------------------------------------------------------------------

class RedditSignal:
    """
    Reddit-based sentiment signal for a specific asset ticker.
    """

    def __init__(self) -> None:
        self._scorer = VaderScorer()

    # ------------------------------------------------------------------
    # Compute asset sentiment
    # ------------------------------------------------------------------

    def compute_asset_sentiment(
        self,
        posts: List[dict],
        ticker: str,
        now: float,
    ) -> Tuple[float, int]:
        """
        Compute time-decay weighted sentiment for a ticker from Reddit posts.

        Parameters
        ----------
        posts  : list[dict]  – Reddit posts (see module docstring for format)
        ticker : str         – ticker symbol to filter by (case-insensitive)
        now    : float       – current Unix timestamp

        Returns
        -------
        (signal, n_posts)
            signal  : float in [-1, +1]
            n_posts : int  – number of matching posts used
        """
        ticker_upper = ticker.upper()
        matching = []

        for post in posts:
            title = str(post.get("title", ""))
            body = str(post.get("body", ""))
            full_text = f"{title} {body}"

            # Case-insensitive ticker mention check
            if ticker_upper not in full_text.upper():
                continue

            matching.append(post)

        if not matching:
            return (0.0, 0)

        weighted_scores = []
        weights = []

        for post in matching:
            title = str(post.get("title", ""))
            body = str(post.get("body", ""))
            full_text = f"{title} {body}".strip()

            # Sentiment score
            sentiment = self._scorer.score(full_text)

            # Upvote weight: upvotes^0.5 (sqrt to reduce impact of viral posts)
            upvotes = max(float(post.get("upvotes", 1)), 1.0)
            upvote_weight = math.sqrt(upvotes)

            # Time decay: half-life of 4 hours
            created = float(post.get("created_utc", now))
            age_hours = max((now - created) / 3600.0, 0.0)
            time_decay = math.exp(-0.693 * age_hours / 4.0)  # half-life = 4h

            weight = upvote_weight * time_decay
            weighted_scores.append(sentiment * weight)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight <= 0:
            return (0.0, len(matching))

        weighted_avg = sum(weighted_scores) / total_weight
        return (float(np.clip(weighted_avg, -1.0, 1.0)), len(matching))

    # ------------------------------------------------------------------
    # Z-score signal
    # ------------------------------------------------------------------

    def compute_zscore_signal(
        self,
        current_score: float,
        historical_scores: List[float],
    ) -> float:
        """
        Normalize current sentiment score as a z-score against historical baseline.

        Parameters
        ----------
        current_score      : float       – current sentiment score
        historical_scores  : list[float] – recent historical sentiment scores

        Returns
        -------
        float in [-1, +1]
        """
        if len(historical_scores) < 3:
            return float(np.clip(current_score, -1.0, 1.0))

        hist = np.array(historical_scores, dtype=float)
        mu = float(np.mean(hist))
        sigma = float(np.std(hist))

        if sigma <= 0:
            return float(np.clip(current_score - mu, -1.0, 1.0))

        z = (current_score - mu) / sigma
        return float(np.tanh(z / 2.0))
