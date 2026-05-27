"""
News sentiment via FinBERT (ProsusAI/finbert). Falls back to AV sentiment.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.logger import get_logger
from .._common import tanh_clip

log = get_logger(__name__)


class FinBERTScorer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = None
        self.model = None
        self._cache: Dict[str, float] = {}
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            import torch  # noqa: F401
            from transformers import (BertForSequenceClassification,  # type: ignore
                                       BertTokenizer)
            self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
            self._loaded = True
            return True
        except Exception as e:
            log.warning(f"FinBERT load failed (falling back to AV scores): {e}")
            return False

    def score_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        if not self.load() or not texts:
            return [0.0] * len(texts)
        import torch  # type: ignore
        out: List[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                hashed = [hashlib.md5(t.encode()).hexdigest()[:16] for t in batch]
                cached = [self._cache.get(h) for h in hashed]
                missing_idx = [j for j, c in enumerate(cached) if c is None]
                if missing_idx:
                    inputs = self.tokenizer([batch[j] for j in missing_idx],
                                             return_tensors="pt", padding=True,
                                             truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).numpy()
                    for k, j in enumerate(missing_idx):
                        # FinBERT label order: positive=0, negative=1, neutral=2
                        score = float(probs[k, 0] - probs[k, 1])
                        self._cache[hashed[j]] = score
                        cached[j] = score
                out.extend([float(c or 0.0) for c in cached])
        return out


def compute_news_sentiment(news_items: List[Dict[str, Any]],
                           asset_ticker: str,
                           scorer: Optional[FinBERTScorer] = None,
                           half_life_secs: float = 14400) -> float:
    if not news_items:
        return 0.0
    ticker_l = asset_ticker.lower()
    relevant = [item for item in news_items
                if ticker_l in item.get("headline", "").lower()
                or ticker_l in item.get("summary", "").lower()]
    if not relevant:
        return 0.0
    if scorer is not None and scorer.load():
        texts = [f"{i.get('headline', '')}. {i.get('summary', '')[:200]}"
                  for i in relevant]
        raw_scores = scorer.score_batch(texts)
    else:
        raw_scores = [float(i.get("av_sentiment", 0)) for i in relevant]
    now = time.time()
    decays = [float(np.exp(-0.693 * (now - float(i.get("timestamp", now)))
                            / half_life_secs)) for i in relevant]
    total_w = sum(decays)
    if total_w <= 0:
        return 0.0
    weighted = sum(s * w for s, w in zip(raw_scores, decays)) / total_w
    return float(tanh_clip(weighted, 0.5))
