"""
Hawkes Process for trade arrival intensity modelling.

Fits a uni-variate Hawkes process:
    λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))

Parameters estimated via Maximum Likelihood Estimation (MLE).
"""

import math
import numpy as np
from typing import List, Optional
from scipy.optimize import minimize


class HawkesProcess:
    """
    Uni-variate Hawkes process for trade-arrival modelling.

    Attributes
    ----------
    mu    : float  – baseline intensity (events / second)
    alpha : float  – excitation amplitude
    beta  : float  – decay rate (1/second)
    """

    def __init__(self) -> None:
        self.mu: float = 0.01
        self.alpha: float = 0.3
        self.beta: float = 1.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # MLE Fitting
    # ------------------------------------------------------------------

    def _log_likelihood(self, params: np.ndarray, timestamps: np.ndarray) -> float:
        """Negative log-likelihood for uni-variate Hawkes process."""
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
            return 1e12  # constraint: branching ratio < 1

        n = len(timestamps)
        if n == 0:
            return 1e12

        T = float(timestamps[-1] - timestamps[0])
        if T <= 0:
            return 1e12

        # Normalise timestamps to start at 0
        t = timestamps - timestamps[0]

        # Log-likelihood (Ozaki 1979 recursive formula)
        # L = -μ*T - α/β * Σ (1 - exp(-β*(T-t_i)))
        #         + Σ log(μ + α * R_i)
        # where R_i = Σ_{j<i} exp(-β*(t_i - t_j))

        R = 0.0
        log_sum = 0.0
        for i in range(n):
            if i == 0:
                R = 0.0
            else:
                R = math.exp(-beta * (t[i] - t[i - 1])) * (1.0 + R)
            intensity = mu + alpha * R
            if intensity <= 0:
                return 1e12
            log_sum += math.log(intensity)

        # Compensator term
        compensator = mu * T
        for i in range(n):
            compensator += (alpha / beta) * (1.0 - math.exp(-beta * (T - t[i])))

        return -(log_sum - compensator)

    def fit_mle(self, trade_timestamps: List[float]) -> None:
        """
        Fit Hawkes process parameters via MLE.

        Parameters
        ----------
        trade_timestamps : list of float
            Unix timestamps (seconds) of trade events.
        """
        if len(trade_timestamps) < 10:
            # Not enough data; keep defaults
            return

        t = np.sort(np.array(trade_timestamps, dtype=float))

        # Initial guess based on data statistics
        T = t[-1] - t[0]
        n = len(t)
        mu0 = n / (T + 1e-8) * 0.5
        alpha0 = 0.3
        beta0 = 1.5

        x0 = np.array([mu0, alpha0, beta0])

        bounds = [(1e-6, None), (1e-6, None), (1e-4, None)]

        result = minimize(
            self._log_likelihood,
            x0,
            args=(t,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )

        if result.success or result.fun < 1e11:
            mu, alpha, beta = result.x
            # Enforce stability: branching ratio < 1
            if 0 < alpha < beta:
                self.mu = float(mu)
                self.alpha = float(alpha)
                self.beta = float(beta)
                self._fitted = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def branching_ratio(self) -> float:
        """
        Branching ratio n = α / β.

        n < 1 → process is stationary.
        n > 0.7 → strong momentum / clustering.
        n < 0.3 → rapidly mean-reverting / fading.
        """
        return float(self.alpha / (self.beta + 1e-12))

    def current_intensity(
        self,
        trade_timestamps: List[float],
        now: float,
    ) -> float:
        """
        Compute current conditional intensity λ(now).

        λ(now) = μ + Σ_{t_i ≤ now} α * exp(-β * (now - t_i))
        """
        lam = self.mu
        for t_i in trade_timestamps:
            dt = now - t_i
            if dt >= 0:
                lam += self.alpha * math.exp(-self.beta * dt)
        return float(lam)

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def momentum_signal(
        self,
        trade_timestamps: List[float],
        now: float,
    ) -> float:
        """
        Trade flow momentum signal in [-1, +1].

        br > 0.7 → momentum  → signal = +tanh(intensity_norm)
        br < 0.3 → fade      → signal = -tanh(intensity_norm)
        else     → neutral   → signal = 0.0

        intensity_norm = (intensity - μ) / (μ + 1e-8)
        """
        br = self.branching_ratio()

        intensity = self.current_intensity(trade_timestamps, now)
        # Normalise against baseline
        intensity_norm = (intensity - self.mu) / (self.mu + 1e-8)
        intensity_norm = float(np.clip(intensity_norm, 0.0, 10.0))

        if br > 0.7:
            return float(math.tanh(intensity_norm))
        elif br < 0.3:
            return float(-math.tanh(intensity_norm))
        else:
            return 0.0
