"""
Real-time inference wrapper. Loads accepted models from disk and serves
prediction requests with < 50 ms latency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .lgbm_trainer import LGBMTrainer

log = get_logger(__name__)


class MLPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.lgbm: Optional[LGBMTrainer] = None
        self.calibration: Optional[Any] = None
        self.feature_cols: list = []
        self.last_inference_ms: float = 0.0
        self.accepted: bool = False

    def load(self, model_name: str = "lgbm_v1") -> bool:
        path = self.model_dir / f"{model_name}.txt"
        feat_path = self.model_dir / f"{model_name}.features.txt"
        if not path.exists():
            log.info(f"ML model not found at {path} - predictor disabled")
            return False
        try:
            self.lgbm = LGBMTrainer()
            self.lgbm.load(str(path))
            if feat_path.exists():
                with open(feat_path) as fh:
                    self.feature_cols = [ln.strip() for ln in fh if ln.strip()]
            self.accepted = True
            log.info(f"ML predictor loaded {model_name} ({len(self.feature_cols)} features)")
            return True
        except Exception as e:
            log.error(f"ML load failed: {e}")
            return False

    def signal(self, features_row: pd.DataFrame) -> Dict[str, float]:
        """Returns signal in [-1, +1] plus probability + confidence."""
        if not self.accepted or self.lgbm is None:
            return {"signal": 0.0, "probability": 0.5, "confidence": 0.0}
        start = time.time()
        try:
            row = features_row.reindex(columns=self.feature_cols).fillna(0)
            proba = float(self.lgbm.predict(row)[0])
        except Exception as e:
            log.warning(f"ML inference error: {e}")
            return {"signal": 0.0, "probability": 0.5, "confidence": 0.0}
        self.last_inference_ms = (time.time() - start) * 1000
        signal_val = float(np.clip((proba - 0.5) * 2, -1.0, 1.0))
        confidence = float(abs(proba - 0.5) * 2)
        return {"signal": signal_val, "probability": proba,
                 "confidence": confidence, "latency_ms": self.last_inference_ms}

    def is_ready(self) -> bool:
        return self.accepted and self.lgbm is not None
