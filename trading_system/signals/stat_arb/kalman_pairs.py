"""
Kalman filter hedge ratio estimation for pairs trading.

State vector: [beta, alpha] where
    price_A_t ≈ beta_t * price_B_t + alpha_t + noise

Uses a random walk model for the state (local linear model).
"""

import math
import numpy as np
import pandas as pd
from typing import List, Tuple


class KalmanHedge:
    """
    Online Kalman filter for dynamic hedge ratio estimation.

    Parameters
    ----------
    delta : float  – process noise covariance scale (controls how fast β can change)
    R     : float  – observation noise variance

    State: θ = [beta, alpha]  (2-dimensional)
    Observation model: y_t = [price_B, 1] @ θ + ε_t  → y_t = price_A
    State model: θ_t = θ_{t-1} + w_t,  w_t ~ N(0, Q)
    """

    def __init__(self, delta: float = 1e-4, R: float = 0.001) -> None:
        self.delta = delta
        self.R = R

        # State estimate: [beta, alpha]
        self._theta = np.zeros(2)

        # State covariance P (2x2)
        self._P = np.eye(2) * 1.0

        # Process noise covariance Q (2x2)
        self._Q = (delta / (1.0 - delta)) * np.eye(2)

        self._spread_history: List[float] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Kalman update step
    # ------------------------------------------------------------------

    def update(
        self,
        price_A: float,
        price_B: float,
    ) -> Tuple[float, float, float, float]:
        """
        Update Kalman filter with new (price_A, price_B) observation.

        Returns
        -------
        (hedge_ratio, intercept, spread, z_score)
            hedge_ratio : β (hedge units of B per unit of A)
            intercept   : α
            spread      : price_A - β * price_B - α
            z_score     : spread / rolling_std_of_spread
        """
        # Observation vector H = [price_B, 1]
        H = np.array([price_B, 1.0])

        # Predict step
        # θ_pred = θ (random walk)
        theta_pred = self._theta.copy()
        P_pred = self._P + self._Q

        # Innovation (residual)
        y_pred = float(H @ theta_pred)
        innovation = price_A - y_pred

        # Innovation covariance
        S = float(H @ P_pred @ H) + self.R

        # Kalman gain
        K = P_pred @ H / (S + 1e-12)

        # Update state
        self._theta = theta_pred + K * innovation
        self._P = (np.eye(2) - np.outer(K, H)) @ P_pred

        hedge_ratio = float(self._theta[0])
        intercept = float(self._theta[1])
        spread = price_A - hedge_ratio * price_B - intercept

        self._spread_history.append(spread)

        # Compute z-score from recent history
        window = min(len(self._spread_history), 20)
        recent = self._spread_history[-window:]
        spread_mean = float(np.mean(recent))
        spread_std = float(np.std(recent))

        if spread_std > 0:
            z = (spread - spread_mean) / spread_std
        else:
            z = 0.0

        self._initialized = True
        return (hedge_ratio, intercept, spread, z)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def get_signal(
        self,
        spread_history: List[float],
        current_spread: float,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> Tuple[int, float]:
        """
        Generate mean-reversion entry/exit signal.

        Parameters
        ----------
        spread_history : list[float]  – historical spread values for z-score calc
        current_spread : float        – current spread value
        entry_z        : float        – z-score magnitude to trigger entry
        exit_z         : float        – z-score magnitude to trigger exit

        Returns
        -------
        (action, z_score)
          action: +1 = buy spread, -1 = sell spread, 0 = hold/flat
        """
        all_spreads = list(spread_history) + [current_spread]
        if len(all_spreads) < 5:
            return (0, 0.0)

        mu = float(np.mean(all_spreads))
        sigma = float(np.std(all_spreads))

        if sigma <= 0:
            return (0, 0.0)

        z = (current_spread - mu) / sigma

        if z <= -entry_z:
            return (1, float(z))   # buy spread
        elif z >= entry_z:
            return (-1, float(z))  # sell spread
        elif abs(z) <= exit_z:
            return (0, float(z))   # exit / flat
        else:
            return (0, float(z))   # hold
