"""
Kalman filter dynamic hedge ratio for pairs trading.

State vector: [hedge_ratio, intercept]
Observation: price_A = h * price_B + intercept + noise
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from .._common import EPS


class KalmanHedge:
    def __init__(self, delta: float = 1e-4, R: float = 0.001):
        self.delta = delta
        self.R = R
        self.C = np.zeros((2, 2))
        self.W = delta / (1 - delta) * np.eye(2)
        self.theta = np.zeros(2)
        self.initialized = False
        self.spread_history: Deque[float] = deque(maxlen=500)

    def update(self, price_A: float, price_B: float
               ) -> Tuple[float, float, float, float]:
        x = np.array([price_B, 1.0])
        if not self.initialized:
            self.theta = np.array([1.0, 0.0])
            self.C = np.eye(2)
            self.initialized = True

        C_pred = self.C + self.W
        y_hat = float(x @ self.theta)
        innovation = price_A - y_hat
        S = float(x @ C_pred @ x.T + self.R)
        if abs(S) < EPS:
            S = EPS
        K = (C_pred @ x.T) / S
        self.theta = self.theta + K * innovation
        self.C = (np.eye(2) - np.outer(K, x)) @ C_pred

        hedge_ratio = float(self.theta[0])
        intercept = float(self.theta[1])
        spread = price_A - hedge_ratio * price_B - intercept
        self.spread_history.append(spread)
        return hedge_ratio, intercept, spread, innovation / np.sqrt(abs(S))

    def signal(self, current_spread: Optional[float] = None,
               entry_z: float = 2.0, exit_z: float = 0.5) -> Tuple[Optional[int], float]:
        if len(self.spread_history) < 20:
            return 0, 0.0
        if current_spread is None:
            current_spread = self.spread_history[-1]
        recent = list(self.spread_history)[-60:]
        mu = float(np.mean(recent))
        sd = float(np.std(recent, ddof=0)) + EPS
        z = (current_spread - mu) / sd
        if z < -entry_z:
            return 1, z
        if z > entry_z:
            return -1, z
        if abs(z) < exit_z:
            return 0, z
        return None, z

    def continuous_signal(self) -> float:
        if len(self.spread_history) < 20:
            return 0.0
        recent = list(self.spread_history)[-60:]
        mu = float(np.mean(recent))
        sd = float(np.std(recent, ddof=0)) + EPS
        z = (recent[-1] - mu) / sd
        return float(np.clip(-np.tanh(z / 2), -1.0, 1.0))
