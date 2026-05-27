"""
News NLP sentiment signals using FinBERT (financial BERT).

Falls back gracefully to VADER if transformers is not available.

News item format (dict):
    'headline'    : str   – article headline
    'body'        : str   – article body text (optional)
    'ticker'      : str   – associated ticker (optional)
    'published_at': float – Unix timestamp of publication
"""

import math
import numpy as np
from typing import List, Tuple, Optional

# Attempt to import transformers (FinBERT)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# Fallback to VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

_FINBERT_MODEL = "ProsusAI/finbert"
_HALF_LIFE_HOURS = 4.0


class FinBERTScorer:
    """
    Financial sentiment scorer using FinBERT.

    If transformers is not installed, falls back to VADER compound scores.
    Labels returned by FinBERT: 'positive', 'neutral', 'negative'
    Mapping: positive → +1, neutral → 0, negative → -1
    """

    _LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(self) -> None:
        self._pipeline = None
        self._backend = "none"
        self._vader = None
        self._load()

    def _load(self) -> None:
        """Load FinBERT pipeline or fall back to VADER."""
        if _TRANSFORMERS_AVAILABLE:
            try:
                self._pipeline = pipeline(
                    "text-classification",
                    model=_FINBERT_MODEL,
                    tokenizer=_FINBERT_MODEL,
                    top_k=None,
                    device=-1,  # CPU
                    truncation=True,
                    max_length=512,
                )
                self._backend = "finbert"
                return
            except Exception:
                pass  # Fall through to VADER

        if _VADER_AVAILABLE:
            self._vader = SentimentIntensityAnalyzer()
            self._backend = "vader"

    def _score_single_finbert(self, text: str) -> float:
        """Score single text with FinBERT pipeline."""
        try:
            result = self._pipeline(text[:512])[0]  # list of label/score dicts
            # result is list of {'label': ..., 'score': ...}
            label_scores = {r["label"].lower(): r["score"] for r in result}
            # Weighted sum: positive*1 + neutral*0 + negative*-1
            score = (
                label_scores.get("positive", 0.0)
                - label_scores.get("negative", 0.0)
            )
            return float(np.clip(score, -1.0, 1.0))
        except Exception:
            return 0.0

    def _score_single_vader(self, text: str) -> float:
        """Score single text with VADER."""
        if self._vader is None:
            return 0.0
        result = self._vader.polarity_scores(text)
        return float(result["compound"])

    def score_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[float]:
        """
        Score a batch of texts.

        Parameters
        ----------
        texts      : list[str]  – texts to score
        batch_size : int        – batch size for inference

        Returns
        -------
        list[float] in [-1, +1] per text.
        """
        if not texts:
            return []

        results = []

        if self._backend == "finbert":
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                for text in batch:
                    results.append(self._score_single_finbert(text))
        elif self._backend == "vader":
            for text in texts:
                results.append(self._score_single_vader(text))
        else:
            # No backend available; return zeros
            results = [0.0] * len(texts)

        return results

    # ------------------------------------------------------------------
    # Asset-level sentiment with recency decay
    # ------------------------------------------------------------------

    def compute_asset_sentiment(
        self,
        news_items: List[dict],
        asset_ticker: str,
        now: Optional[float] = None,
    ) -> Tuple[float, int]:
        """
        Compute recency-decay weighted sentiment for an asset.

        Parameters
        ----------
        news_items   : list[dict]   – news items (see module docstring)
        asset_ticker : str          – ticker to filter by
        now          : float|None   – current Unix timestamp (defaults to import time)

        Returns
        -------
        (signal, n_articles)
            signal     : float in [-1, +1]
            n_articles : int
        """
        import time as _time

        if now is None:
            now = _time.time()

        ticker_upper = asset_ticker.upper()
        relevant = []

        for item in news_items:
            headline = str(item.get("headline", ""))
            body = str(item.get("body", ""))
            item_ticker = str(item.get("ticker", "")).upper()
            full_text = f"{headline} {body}"

            if ticker_upper not in full_text.upper() and item_ticker != ticker_upper:
                continue
            relevant.append(item)

        if not relevant:
            return (0.0, 0)

        texts = []
        timestamps = []
        for item in relevant:
            headline = str(item.get("headline", ""))
            body = str(item.get("body", ""))
            # Use headline + first 200 chars of body for scoring
            text = f"{headline}. {body[:200]}".strip()
            texts.append(text)
            ts = float(item.get("published_at", now))
            timestamps.append(ts)

        scores = self.score_batch(texts)

        # Apply recency decay: half-life of 4 hours
        weighted_scores = []
        weights = []

        for score, ts in zip(scores, timestamps):
            age_hours = max((now - ts) / 3600.0, 0.0)
            decay = math.exp(-0.693 * age_hours / _HALF_LIFE_HOURS)
            weighted_scores.append(score * decay)
            weights.append(decay)

        total_weight = sum(weights)
        if total_weight <= 0:
            return (0.0, len(relevant))

        weighted_avg = sum(weighted_scores) / total_weight
        return (float(np.clip(weighted_avg, -1.0, 1.0)), len(relevant))
