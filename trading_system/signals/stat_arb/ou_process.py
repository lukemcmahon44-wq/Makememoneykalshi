"""
Ornstein-Uhlenbeck process for spread modelling in statistical arbitrage.

dS = θ(μ - S)dt + σ dW

Parameters estimated via OLS regression on the discrete form:
  ΔS_t = a + b * S_{t-1} + ε
  → θ = -b, μ = a / θ, σ_eq = std(ε) / sqrt(-2b)
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process fitted to a spread series.

    Parameters (fitted)
    -------------------
    theta   : float  – mean-reversion speed (per bar)
    mu      : float  – long-run mean of the spread
    sigma   : float  – instantaneous volatility of the spread
    sigma_eq: float  – equilibrium standard deviation = σ / sqrt(2θ)
    half_life: float – half-life of mean reversion (in bars)
    """

    def __init__(self) -> None:
        self.theta: float = 0.1
        self.mu: float = 0.0
        self.sigma: float = 1.0
        self.sigma_eq: float = 1.0
        self.half_life: float = math.log(2) / 0.1
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, spread_series: pd.Series) -> None:
        """
        Fit OU parameters via OLS regression.

        Discrete model: ΔS_t = a + b * S_{t-1} + ε
          a = α, b = β in OLS notation
          θ = -β  (mean-reversion speed per bar)
          μ = α / θ
          σ_eq = std(residuals) / sqrt(-2β + 1e-8)
        """
        s = spread_series.dropna()
        if len(s) < 30:
            return

        s_lag = s.shift(1).dropna()
        s_diff = s.diff().dropna()

        # Align
        idx = s_lag.index.intersection(s_diff.index)
        s_lag = s_lag.loc[idx].values
        s_diff = s_diff.loc[idx].values

        # OLS: s_diff = a + b * s_lag
        X = np.column_stack([np.ones(len(s_lag)), s_lag])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, s_diff, rcond=None)
        except np.linalg.LinAlgError:
            return

        a, b = float(coeffs[0]), float(coeffs[1])

        # Stability check: b must be negative for mean reversion
        if b >= 0:
            # No mean reversion; set very slow speed
            b = -1e-4

        theta = -b
        mu = a / (theta + 1e-12)

        # Residual std
        predicted = a + b * s_lag
        resid = s_diff - predicted
        sigma_resid = float(np.std(resid))

        # Equilibrium std
        denom = math.sqrt(max(2.0 * theta, 1e-12))
        sigma_eq = sigma_resid / denom

        half_life = math.log(2.0) / (theta + 1e-12)

        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma_resid)
        self.sigma_eq = float(sigma_eq)
        self.half_life = float(half_life)
        self._fitted = True

    # ------------------------------------------------------------------
    # Z-score
    # ------------------------------------------------------------------

    def z_score(self, current_spread: float) -> float:
        """
        Z-score of the spread relative to the OU equilibrium.

        z = (spread - μ) / σ_eq
        """
        if self.sigma_eq <= 0:
            return 0.0
        return float((current_spread - self.mu) / (self.sigma_eq + 1e-12))

    # ------------------------------------------------------------------
    # Entry signal
    # ------------------------------------------------------------------

    def entry_signal(
        self,
        current_spread: float,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_threshold: float = 3.5,
    ) -> Tuple[int, float]:
        """
        Generate entry/exit signal based on z-score thresholds.

        Returns
        -------
        (action, z_score)
          action: +1 = buy spread (spread too low → expect mean reversion up)
                  -1 = sell spread (spread too high → expect mean reversion down)
                   0 = hold / exit
        """
        z = self.z_score(current_spread)

        if abs(z) >= stop_threshold:
            # Stop loss zone – exit or avoid
            return (0, z)

        if z <= -entry_threshold:
            # Spread is far below mean → buy
            return (1, z)
        elif z >= entry_threshold:
            # Spread is far above mean → sell
            return (-1, z)
        elif abs(z) <= exit_threshold:
            # Near mean → exit/flat
            return (0, z)
        else:
            # Between exit and entry thresholds → hold existing position
            return (0, z)

    # ------------------------------------------------------------------
    # Tradability
    # ------------------------------------------------------------------

    def tradeable(self) -> bool:
        """
        Check whether the OU process has a half-life amenable to trading.

        Tradeable window: 1 hour to 7 days (in bars).
        Here bars are assumed to be 1-hour bars, so:
          min half-life = 1 bar, max half-life = 168 bars (7 * 24).
        """
        if not self._fitted:
            return False
        return 1.0 <= self.half_life <= 168.0
