"""
Ornstein-Uhlenbeck process estimation for pairs trading.

dX_t = θ(μ - X_t)dt + σ dW_t

OLS estimator: ΔX_t = a + b * X_{t-1} + ε_t
θ = -b/Δt; μ = a/-b; σ_eq = σ / √(2θ)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .._common import EPS


class OrnsteinUhlenbeck:
    def __init__(self):
        self.theta: Optional[float] = None
        self.mu: Optional[float] = None
        self.sigma: Optional[float] = None
        self.sigma_eq: Optional[float] = None
        self.half_life: Optional[float] = None

    def fit(self, spread: np.ndarray, dt: float = 1.0) -> "OrnsteinUhlenbeck":
        x = np.asarray(spread, dtype=float)
        if len(x) < 30:
            raise ValueError("Need >= 30 observations to fit OU")
        x_lag = x[:-1]
        dx = np.diff(x)
        A = np.column_stack([np.ones_like(x_lag), x_lag])
        coeffs, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        if b >= 0:
            raise ValueError("Spread is not mean-reverting (b >= 0)")
        self.theta = -b / dt
        self.mu = a / (-b)
        residuals = dx - (a + b * x_lag)
        sigma_eps = float(np.std(residuals, ddof=0))
        self.sigma = sigma_eps / np.sqrt(dt)
        self.sigma_eq = self.sigma / np.sqrt(2 * self.theta + EPS)
        self.half_life = float(np.log(2) / self.theta) if self.theta > 0 else None
        return self

    def z_score(self, current_spread: float) -> float:
        if self.mu is None or self.sigma_eq is None or self.sigma_eq < EPS:
            return 0.0
        return (current_spread - self.mu) / self.sigma_eq

    def entry_signal(self, current_spread: float,
                     entry_threshold: float = 2.0,
                     exit_threshold: float = 0.5,
                     stop_threshold: float = 3.5
                     ) -> Tuple[Optional[int], float]:
        z = self.z_score(current_spread)
        if z < -entry_threshold:
            return 1, z
        if z > entry_threshold:
            return -1, z
        if abs(z) < exit_threshold:
            return 0, z
        if abs(z) > stop_threshold:
            return 0, z
        return None, z

    def tradeable(self) -> bool:
        return self.half_life is not None and 1.0 < self.half_life < 168.0

    def signal(self, current_spread: float) -> float:
        """Return continuous signal in [-1, +1]."""
        z = self.z_score(current_spread)
        return float(np.clip(-np.tanh(z / 2.0), -1.0, 1.0))
