"""
2-state Hidden Markov Model for bull/bear regime classification.

Features: daily return, 5d realized vol, vol ratio, volume ratio.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class MarketHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 200):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self.scaler = None
        self.bull_state: Optional[int] = None
        self.bear_state: Optional[int] = None

    def _build(self):
        from hmmlearn import hmm  # type: ignore
        return hmm.GaussianHMM(n_components=self.n_states,
                                covariance_type="full",
                                n_iter=self.n_iter,
                                random_state=42)

    def _features(self, closes: np.ndarray, volumes: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        rets = np.diff(np.log(closes + 1e-10))
        vol_5 = _rolling_std(rets, 5)
        vol_20 = _rolling_std(rets, 20)
        vol_ratio = vol_5 / (vol_20 + 1e-10)
        vol_pct = volumes[1:] / (_rolling_mean(volumes, 20)[1:] + 1e-10)
        n = min(len(rets), len(vol_5), len(vol_ratio), len(vol_pct))
        X = np.column_stack([
            rets[-n:], vol_5[-n:], vol_ratio[-n:], vol_pct[-n:],
        ])
        mask = np.all(np.isfinite(X), axis=1)
        return X[mask], rets[-n:][mask]

    def fit(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError:
            return False
        try:
            X, rets = self._features(closes, volumes)
            if len(X) < 100:
                return False
            self.scaler = StandardScaler().fit(X)
            Xn = self.scaler.transform(X)
            self.model = self._build()
            self.model.fit(Xn)
            states = self.model.predict(Xn)
            mean_by_state = {s: float(np.mean(rets[states == s]))
                              for s in range(self.n_states)}
            self.bull_state = max(mean_by_state, key=mean_by_state.get)
            self.bear_state = min(mean_by_state, key=mean_by_state.get)
            return True
        except Exception:
            return False

    def predict_current(self, recent_features: np.ndarray
                        ) -> Tuple[str, float]:
        if self.model is None or self.scaler is None:
            return "neutral", 0.5
        try:
            Xn = self.scaler.transform(recent_features.reshape(1, -1))
            probs = self.model.predict_proba(Xn)[0]
            bull_prob = float(probs[self.bull_state]) if self.bull_state is not None else 0.5
            state = "bull" if bull_prob > 0.5 else "bear"
            return state, bull_prob
        except Exception:
            return "neutral", 0.5

    def signal_multiplier(self, bull_prob: float) -> float:
        return float(np.clip(0.5 + bull_prob, 0.3, 1.5))


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    if len(arr) < w:
        return np.full(len(arr), np.nan)
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = float(np.std(arr[i - w + 1: i + 1], ddof=0))
    return out


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if len(arr) < w:
        return np.full(len(arr), np.nan)
    csum = np.cumsum(np.insert(arr, 0, 0))
    out = (csum[w:] - csum[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), out])
