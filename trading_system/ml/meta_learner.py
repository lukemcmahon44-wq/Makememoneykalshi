"""
Level-2 stacking ensemble: combines LightGBM, LSTM, and TFT predictions
via a regularized logistic regression meta-model.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MetaLearner:
    """
    Stacking meta-learner that combines base model predictions.

    Architecture
    ------------
    Level-1 models: LightGBM, LSTM, TFT → each produces P(up) in [0, 1]
    Level-2: LogisticRegression(C=0.1) trained on base model outputs
    Optional: probability calibration via Platt scaling

    Attributes
    ----------
    meta_model : LogisticRegression
    scaler : StandardScaler
    last_calibration_date : datetime
    calibration_data : list of (features, labels) for monthly recalibration
    """

    def __init__(self, regularization_C: float = 0.1, max_iter: int = 1000):
        self.regularization_C = regularization_C
        self.max_iter = max_iter
        self.meta_model: Optional[LogisticRegression] = None
        self.scaler = StandardScaler()
        self.last_calibration_date: Optional[datetime] = None
        self.calibration_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self._calibrator: Optional[CalibratedClassifierCV] = None
        self._use_calibrated: bool = False

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_meta_features(
        self,
        lgbm_preds: np.ndarray,
        lstm_preds: np.ndarray,
        tft_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Construct meta-feature matrix from base model predictions.

        Features include:
          - Raw predictions from each model
          - Pairwise differences (disagreement)
          - Mean prediction
          - Variance across models
          - Interaction terms
        """
        lgbm = np.array(lgbm_preds, dtype=np.float32).reshape(-1)
        lstm = np.array(lstm_preds, dtype=np.float32).reshape(-1)
        tft = np.array(tft_preds, dtype=np.float32).reshape(-1)

        n = len(lgbm)

        # Raw predictions
        f1 = lgbm
        f2 = lstm
        f3 = tft

        # Pairwise differences
        f4 = lgbm - lstm
        f5 = lgbm - tft
        f6 = lstm - tft

        # Ensemble statistics
        stack = np.stack([lgbm, lstm, tft], axis=1)
        f7 = stack.mean(axis=1)  # mean
        f8 = stack.std(axis=1)   # std / disagreement
        f9 = stack.max(axis=1)   # max
        f10 = stack.min(axis=1)  # min

        # Interaction terms (product = confidence amplifier)
        f11 = lgbm * lstm
        f12 = lgbm * tft
        f13 = lstm * tft

        # Agreement indicators (all above 0.6 or all below 0.4)
        f14 = (stack > 0.6).all(axis=1).astype(np.float32)
        f15 = (stack < 0.4).all(axis=1).astype(np.float32)

        X_meta = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                                   f11, f12, f13, f14, f15])
        return X_meta

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        lgbm_preds: np.ndarray,
        lstm_preds: np.ndarray,
        tft_preds: np.ndarray,
        y: np.ndarray,
    ) -> LogisticRegression:
        """
        Train the meta-model.

        Parameters
        ----------
        lgbm_preds : np.ndarray of shape (n,) with probabilities from LightGBM
        lstm_preds : np.ndarray of shape (n,) with probabilities from LSTM
        tft_preds : np.ndarray of shape (n,) with probabilities from TFT
        y : np.ndarray of shape (n,) with binary labels

        Returns
        -------
        LogisticRegression (fitted)
        """
        X_meta = self._build_meta_features(lgbm_preds, lstm_preds, tft_preds)
        y = np.array(y, dtype=np.float32)

        # Normalize inputs
        X_scaled = self.scaler.fit_transform(X_meta)

        self.meta_model = LogisticRegression(
            C=self.regularization_C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self.meta_model.fit(X_scaled, y.astype(int))

        # Store calibration data
        self.calibration_data.append((X_meta.copy(), y.copy()))
        self.last_calibration_date = datetime.utcnow()

        return self.meta_model

    def recalibrate(
        self,
        lgbm_preds: np.ndarray,
        lstm_preds: np.ndarray,
        tft_preds: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Re-calibrate the meta-model using recent data (monthly).
        Accumulates calibration data across calls.
        """
        self.fit(lgbm_preds, lstm_preds, tft_preds, y)
        self.last_calibration_date = datetime.utcnow()

    def needs_recalibration(self) -> bool:
        """Returns True if more than 30 days since last calibration."""
        if self.last_calibration_date is None:
            return True
        return (datetime.utcnow() - self.last_calibration_date) > timedelta(days=30)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        lgbm_pred: float,
        lstm_pred: float,
        tft_pred: float,
    ) -> float:
        """
        Ensemble prediction.

        Parameters
        ----------
        lgbm_pred : float probability from LightGBM
        lstm_pred : float probability from LSTM
        tft_pred : float probability from TFT

        Returns
        -------
        float
            Ensemble probability in [0, 1]. Falls back to simple average
            if meta-model is not trained.
        """
        if self.meta_model is None:
            # Fallback: weighted average with equal weights
            return float(np.clip((lgbm_pred + lstm_pred + tft_pred) / 3.0, 0.0, 1.0))

        X_meta = self._build_meta_features(
            np.array([lgbm_pred]),
            np.array([lstm_pred]),
            np.array([tft_pred]),
        )

        try:
            X_scaled = self.scaler.transform(X_meta)
            proba = self.meta_model.predict_proba(X_scaled)[0]
            # Class 1 probability
            if len(proba) >= 2:
                return float(np.clip(proba[1], 0.0, 1.0))
            return float(np.clip(proba[0], 0.0, 1.0))
        except Exception:
            return float(np.clip((lgbm_pred + lstm_pred + tft_pred) / 3.0, 0.0, 1.0))

    def get_signal(self, probability: float) -> float:
        """
        Convert probability to trading signal.

        Maps [0, 1] → [-1, +1] via (p - 0.5) * 2, clamped to [-1, +1].

        Parameters
        ----------
        probability : float in [0, 1]

        Returns
        -------
        float in [-1, +1]
            > 0 → long bias, < 0 → short bias, 0 → neutral.
        """
        signal = (probability - 0.5) * 2.0
        return float(np.clip(signal, -1.0, 1.0))

    def predict_signal(
        self,
        lgbm_pred: float,
        lstm_pred: float,
        tft_pred: float,
    ) -> float:
        """
        Convenience: predict and convert to signal in one call.

        Returns
        -------
        float in [-1, +1]
        """
        prob = self.predict(lgbm_pred, lstm_pred, tft_pred)
        return self.get_signal(prob)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save meta-learner to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "meta_model": self.meta_model,
            "scaler": self.scaler,
            "last_calibration_date": self.last_calibration_date,
            "regularization_C": self.regularization_C,
        }
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load meta-learner state from disk."""
        with open(f"{path}.meta.pkl", "rb") as f:
            state = pickle.load(f)
        self.meta_model = state.get("meta_model")
        self.scaler = state.get("scaler", StandardScaler())
        self.last_calibration_date = state.get("last_calibration_date")
        self.regularization_C = state.get("regularization_C", 0.1)
