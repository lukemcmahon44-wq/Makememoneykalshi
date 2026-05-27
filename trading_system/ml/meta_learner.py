"""
Meta-learner / level-2 stacking ensemble.

Takes the base-model predictions (LGBM + LSTM + TFT) as inputs and learns
the best combination via logistic regression with isotonic calibration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger

log = get_logger(__name__)


class StackingMetaLearner:
    def __init__(self):
        self.model = None
        self.calibrator = None
        self.feature_cols: List[str] = []

    def fit(self, base_preds: pd.DataFrame, y: pd.Series) -> bool:
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.isotonic import IsotonicRegression  # type: ignore
        except ImportError:
            log.error("sklearn missing - meta-learner disabled")
            return False
        if len(base_preds) < 100 or base_preds.shape[1] < 2:
            return False
        X = base_preds.fillna(0.5).values
        self.feature_cols = list(base_preds.columns)
        self.model = LogisticRegression(max_iter=500, C=1.0)
        self.model.fit(X, y.values)
        probs = self.model.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(probs, y.values)
        return True

    def predict_proba(self, base_preds: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(base_preds), 0.5)
        X = base_preds.reindex(columns=self.feature_cols).fillna(0.5).values
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def signal(self, base_preds_row: Dict[str, float]) -> float:
        """Convert calibrated probability to signal in [-1, +1]."""
        df = pd.DataFrame([base_preds_row])
        p = float(self.predict_proba(df)[0])
        return float(np.clip((p - 0.5) * 2, -1.0, 1.0))
