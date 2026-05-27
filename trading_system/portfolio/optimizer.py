"""
Black-Litterman portfolio optimization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class BlackLittermanOptimizer:
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.delta = risk_aversion
        self.tau = tau

    def equilibrium_returns(self, weights: np.ndarray, cov: np.ndarray
                            ) -> np.ndarray:
        return self.delta * cov @ weights

    def combine_views(self, pi: np.ndarray, cov: np.ndarray, P: np.ndarray,
                       Q: np.ndarray, omega: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        tau_sigma = self.tau * cov
        tau_sigma_inv = np.linalg.pinv(tau_sigma)
        if omega is None:
            omega = np.diag(np.diag(P @ tau_sigma @ P.T)) + 1e-6 * np.eye(P.shape[0])
        omega_inv = np.linalg.pinv(omega)
        M_inv = tau_sigma_inv + P.T @ omega_inv @ P
        M = np.linalg.pinv(M_inv)
        mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        cov_bl = cov + M
        return mu_bl, cov_bl

    def optimize_weights(self, mu_bl: np.ndarray, cov_bl: np.ndarray,
                         max_weight: float = 0.20,
                         min_position: float = 0.02) -> np.ndarray:
        try:
            from scipy.optimize import minimize  # type: ignore
        except ImportError:
            n = len(mu_bl)
            return np.ones(n) / n
        n = len(mu_bl)

        def neg_sharpe(w):
            r = w @ mu_bl
            v = np.sqrt(max(w @ cov_bl @ w, 1e-12))
            return -(r / v)

        bounds = [(0.0, max_weight)] * n
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        try:
            res = minimize(neg_sharpe, x0=np.ones(n) / n,
                            bounds=bounds, constraints=[cons],
                            method="SLSQP",
                            options={"maxiter": 200, "disp": False})
        except Exception:
            return np.ones(n) / n
        if not res.success:
            return np.ones(n) / n
        w = res.x.copy()
        w[w < min_position] = 0
        if w.sum() <= 0:
            return np.ones(n) / n
        return w / w.sum()

    def signal_to_views(self, signals: Dict[str, float], assets: List[str]
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(assets)
        P = np.eye(n)
        Q = np.array([float(signals.get(a, 0)) * 0.15 for a in assets])
        certainty = np.array([abs(float(signals.get(a, 0))) + 0.1 for a in assets])
        omega = np.diag((1.0 / certainty) * 0.05)
        return P, Q, omega
