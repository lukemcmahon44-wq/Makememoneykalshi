"""
GARCH(1,1) volatility regime estimator.

Falls back to realized volatility if `arch` library is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .._common import EPS


class GARCHRegime:
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.fitted = None
        self.use_arch = True

    def fit(self, returns_pct: np.ndarray) -> bool:
        """returns_pct: array of percentage daily returns (e.g. 1.5 = 1.5%)."""
        try:
            from arch import arch_model  # type: ignore
        except ImportError:
            self.use_arch = False
            return False
        if len(returns_pct) < 100:
            return False
        try:
            model = arch_model(returns_pct * 100, vol="GARCH",
                               p=self.p, q=self.q, mean="Constant",
                               dist="SkewStudent")
            result = model.fit(disp="off", show_warning=False)
            self.fitted = result
            return True
        except Exception:
            return False

    def forecast_variance(self) -> Optional[float]:
        if self.fitted is None:
            return None
        try:
            f = self.fitted.forecast(horizon=1, reindex=False)
            v = float(f.variance.values[-1, 0]) / 10000  # back to decimal
            return v
        except Exception:
            return None

    def forecast_vol(self) -> Optional[float]:
        v = self.forecast_variance()
        return float(np.sqrt(v)) if v is not None else None

    def realized_fallback(self, returns: np.ndarray) -> float:
        if len(returns) < 5:
            return 0.02
        return float(np.std(returns[-30:], ddof=0))

    def volatility_regime(self, current_vol: float, hist_vols: np.ndarray,
                          window: int = 30) -> Tuple[str, float]:
        if len(hist_vols) < 5:
            return "normal", 1.0
        avg_vol = float(np.mean(hist_vols[-window:]))
        ratio = current_vol / (avg_vol + EPS)
        if ratio < 0.6:
            return "low", 1.3
        if ratio < 1.3:
            return "normal", 1.0
        if ratio < 2.0:
            return "high", 0.7
        return "extreme", 0.3

    def position_size_scalar(self, target_annual_vol: float = 0.20,
                             returns: Optional[np.ndarray] = None) -> float:
        v = self.forecast_variance()
        if v is None and returns is not None:
            daily = self.realized_fallback(returns)
        elif v is not None:
            daily = float(np.sqrt(v))
        else:
            return 1.0
        annual = daily * np.sqrt(252)
        scalar = target_annual_vol / (annual + EPS)
        return float(np.clip(scalar, 0.3, 2.5))
