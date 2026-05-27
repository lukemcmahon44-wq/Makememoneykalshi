"""
Hawkes self-exciting point process for trade clustering.

Reference: Neural Hawkes (2025), Bitcoin Trade Arrival (Heusser 2013).

λ(t) = μ + α * Σ exp(-β(t - t_i))
Branching ratio α/β:
    > 0.7 → self-exciting regime, momentum likely to continue
    < 0.3 → trades exogenous, fade moves
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .._common import EPS, tanh_clip


class HawkesProcess:
    def __init__(self, mu: float = 1.0, alpha: float = 0.3, beta: float = 1.0,
                 lookback_seconds: float = 600):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lookback = lookback_seconds

    def fit_mle(self, trade_timestamps: Sequence[float]) -> bool:
        """Fit parameters via MLE. Returns True on success."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            return False
        if len(trade_timestamps) < 30:
            return False
        times = np.asarray(sorted(trade_timestamps), dtype=float)
        t0 = times[0]
        times = times - t0
        T = times[-1]
        if T <= 0:
            return False

        def neg_ll(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0:
                return 1e10
            if alpha >= beta:
                return 1e10
            n = len(times)
            R = np.zeros(n)
            for i in range(1, n):
                R[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1 + R[i - 1])
            intensities = mu + alpha * R
            ll = (-mu * T
                  + (alpha / beta) * float(np.sum(np.exp(-beta * (T - times)) - 1))
                  + float(np.sum(np.log(intensities + 1e-10))))
            return -ll

        try:
            result = minimize(
                neg_ll, x0=[self.mu, self.alpha, self.beta],
                bounds=[(1e-4, 100), (1e-4, 10), (1e-4, 100)],
                method="L-BFGS-B",
            )
        except Exception:
            return False
        if result.success and np.all(np.isfinite(result.x)):
            self.mu, self.alpha, self.beta = float(result.x[0]), float(result.x[1]), float(result.x[2])
            return True
        return False

    def branching_ratio(self) -> float:
        return self.alpha / max(self.beta, EPS)

    def current_intensity(self, trade_timestamps: Sequence[float],
                          now: float) -> float:
        recent = [t for t in trade_timestamps if now - t < self.lookback]
        if not recent:
            return self.mu
        return self.mu + self.alpha * float(np.sum(
            [np.exp(-self.beta * (now - t)) for t in recent]))

    def momentum_signal(self, trade_timestamps: Sequence[float],
                        now: float) -> float:
        br = self.branching_ratio()
        intensity = self.current_intensity(trade_timestamps, now)
        intensity_norm = (intensity - self.mu) / (self.mu + EPS)
        if br > 0.7:
            return float(tanh_clip(intensity_norm, 1.0))
        if br < 0.3:
            return float(-tanh_clip(intensity_norm, 1.0))
        return 0.0


class HawkesCache:
    """Per-asset Hawkes processes that periodically refit."""

    def __init__(self, refit_every_n_trades: int = 100):
        self.refit_every = refit_every_n_trades
        self.processes: dict[str, HawkesProcess] = {}
        self.last_fit_count: dict[str, int] = {}

    def update_and_signal(self, asset: str, trade_timestamps: List[float],
                          now: float) -> float:
        if not trade_timestamps:
            return 0.0
        proc = self.processes.setdefault(asset, HawkesProcess())
        last = self.last_fit_count.get(asset, 0)
        if len(trade_timestamps) - last >= self.refit_every:
            ok = proc.fit_mle(trade_timestamps[-500:])
            if ok:
                self.last_fit_count[asset] = len(trade_timestamps)
        return proc.momentum_signal(trade_timestamps, now)

    def branching_ratio(self, asset: str) -> float:
        proc = self.processes.get(asset)
        return proc.branching_ratio() if proc else 0.0
