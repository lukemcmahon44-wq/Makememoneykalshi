"""
Hidden Markov Model for market regime detection.

Identifies bull/bear regimes using a 2-state Gaussian HMM fitted to
daily returns, log-volume, and volatility features.
"""

import numpy as np
import pandas as pd
from typing import Tuple

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False


class MarketHMM:
    """
    2-state Gaussian HMM for market regime classification.

    States: 'bull' and 'bear', identified by mean daily return.

    Features per bar:
        1. Daily log return
        2. Log-volume change (normalized)
        3. 5-day rolling volatility (annualised)
    """

    def __init__(self) -> None:
        self._model = None
        self._bull_state: int = 0
        self._bear_state: int = 1
        self._fitted: bool = False
        self._feature_mean: np.ndarray = np.zeros(3)
        self._feature_std: np.ndarray = np.ones(3)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        daily_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Prepare feature matrix from daily OHLCV data.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Must contain 'close' and 'volume' columns.

        Returns
        -------
        (features, index)
            features : np.ndarray of shape (N, 3)
            index    : DatetimeIndex or RangeIndex
        """
        close = daily_df["close"].astype(float)
        volume = daily_df["volume"].astype(float)

        # 1. Daily log return
        log_ret = np.log(close / close.shift(1))

        # 2. Log-volume change
        log_vol_chg = np.log(volume / volume.shift(1).replace(0, np.nan))

        # 3. 5-day rolling vol (annualised)
        rolling_vol = log_ret.rolling(5).std() * np.sqrt(252)

        features_df = pd.concat(
            [log_ret, log_vol_chg, rolling_vol], axis=1
        )
        features_df.columns = ["log_ret", "log_vol_chg", "rolling_vol"]
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

        return features_df.values, features_df.index

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, daily_df: pd.DataFrame) -> None:
        """
        Fit 2-state Gaussian HMM and identify bull/bear states.

        Bull state = state with higher mean log return.
        Bear state = state with lower mean log return.
        """
        if not _HMMLEARN_AVAILABLE:
            self._fitted = False
            return

        features, _ = self.prepare_features(daily_df)
        if len(features) < 60:
            return

        # Standardize features
        self._feature_mean = features.mean(axis=0)
        self._feature_std = features.std(axis=0) + 1e-8
        X = (features - self._feature_mean) / self._feature_std

        model = hmmlearn_hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        try:
            model.fit(X)
        except Exception:
            return

        self._model = model

        # Identify bull/bear by mean return of state 0 vs state 1
        # Model means are in standardized space; retrieve original-space return means
        means_std = model.means_  # shape (2, 3)
        # De-standardize only the return dimension (index 0)
        means_orig = means_std[:, 0] * self._feature_std[0] + self._feature_mean[0]

        if means_orig[0] >= means_orig[1]:
            self._bull_state = 0
            self._bear_state = 1
        else:
            self._bull_state = 1
            self._bear_state = 0

        self._fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_current(
        self,
        recent_features: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Predict current market regime and bull probability.

        Parameters
        ----------
        recent_features : np.ndarray of shape (N, 3)
            Feature matrix for recent bars (same format as prepare_features output).

        Returns
        -------
        (regime_str, bull_probability)
            regime_str      : 'bull' or 'bear'
            bull_probability: float [0, 1]
        """
        if not self._fitted or self._model is None:
            return ("bull", 0.5)

        if len(recent_features) < 5:
            return ("bull", 0.5)

        X = (recent_features - self._feature_mean) / self._feature_std

        try:
            # Posterior probabilities for each state at each timestep
            posteriors = self._model.predict_proba(X)
            bull_prob = float(posteriors[-1, self._bull_state])
        except Exception:
            return ("bull", 0.5)

        regime = "bull" if bull_prob >= 0.5 else "bear"
        return (regime, bull_prob)

    # ------------------------------------------------------------------
    # Signal multiplier
    # ------------------------------------------------------------------

    def signal_multiplier(self, bull_prob: float) -> float:
        """
        Convert bull probability to a signal scaling factor.

        bull_prob = 1.0 → multiplier = 1.5  (strong bull; amplify long signals)
        bull_prob = 0.5 → multiplier = 1.0  (neutral)
        bull_prob = 0.0 → multiplier = 0.5  (strong bear; dampen long signals)

        Linear interpolation:
            multiplier = 0.5 + bull_prob * 1.0  → [0.5, 1.5]
        """
        bull_prob = float(np.clip(bull_prob, 0.0, 1.0))
        return float(0.5 + bull_prob * 1.0)
