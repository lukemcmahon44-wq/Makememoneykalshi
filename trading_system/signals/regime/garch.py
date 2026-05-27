"""
GARCH volatility regime modelling.

Uses the arch library with a GARCH(1,1) model and Skewed Student-t distribution.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


class GARCHRegime:
    """
    GARCH(1,1) volatility regime classifier.

    Fits a GARCH(1,1) model with Skewed Student-t innovations to percentage returns.
    Provides volatility forecasting and regime classification.
    """

    def __init__(self) -> None:
        self._model = None
        self._result = None
        self._fitted: bool = False
        self._last_vol: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns_pct: pd.Series) -> None:
        """
        Fit GARCH(1,1) model with Skewed Student-t distribution.

        Parameters
        ----------
        returns_pct : pd.Series
            Percentage returns (e.g. 1.5 for 1.5%).
        """
        r = returns_pct.dropna()
        if len(r) < 50:
            return

        if not _ARCH_AVAILABLE:
            # Fallback: simple EWMA volatility estimate
            self._last_vol = float(r.ewm(span=20).std().iloc[-1])
            self._fitted = True
            return

        try:
            model = arch_model(
                r,
                vol="Garch",
                p=1,
                q=1,
                dist="skewt",
                rescale=True,
            )
            result = model.fit(
                disp="off",
                show_warning=False,
                options={"maxiter": 200},
            )
            self._model = model
            self._result = result
            # Store current conditional volatility
            cond_vol = result.conditional_volatility
            if cond_vol is not None and len(cond_vol) > 0:
                self._last_vol = float(cond_vol.iloc[-1])
            self._fitted = True
        except Exception:
            # Fallback to EWMA
            self._last_vol = float(r.ewm(span=20).std().iloc[-1])
            self._fitted = True

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast_variance(self, horizon: int = 1) -> float:
        """
        Forecast conditional variance h steps ahead.

        Returns
        -------
        float : forecasted variance (not volatility; take sqrt for vol)
        """
        if not self._fitted:
            return float("nan")

        if self._result is not None and _ARCH_AVAILABLE:
            try:
                fc = self._result.forecast(horizon=horizon, reindex=False)
                # variance forecast: shape (1, horizon)
                var_fc = fc.variance.iloc[-1, horizon - 1]
                return float(var_fc)
            except Exception:
                pass

        # Fallback: use last vol squared
        return float(self._last_vol ** 2)

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def volatility_regime(
        self,
        current_vol: float,
        hist_vols: List[float],
        window: int = 30,
    ) -> Tuple[str, float]:
        """
        Classify current volatility into a regime.

        Parameters
        ----------
        current_vol : float       – annualised volatility (e.g. 0.25 = 25%)
        hist_vols   : list[float] – recent history of volatility estimates
        window      : int         – lookback for percentile calculation

        Returns
        -------
        (regime_str, percentile_rank)
          regime_str : 'low_vol' | 'normal_vol' | 'high_vol' | 'extreme_vol'
          percentile : float [0, 1] – where current vol sits in recent history
        """
        if len(hist_vols) < 5:
            return ("normal_vol", 0.5)

        recent = np.array(hist_vols[-window:])
        percentile = float(np.mean(recent <= current_vol))

        if percentile >= 0.90:
            regime = "extreme_vol"
        elif percentile >= 0.70:
            regime = "high_vol"
        elif percentile <= 0.20:
            regime = "low_vol"
        else:
            regime = "normal_vol"

        return (regime, percentile)

    # ------------------------------------------------------------------
    # Position size scalar
    # ------------------------------------------------------------------

    def position_size_scalar(self, target_vol: float = 0.20) -> float:
        """
        Inverse-vol position sizing scalar.

        scalar = target_vol / current_vol
        Clipped to [0.1, 3.0] to prevent extreme sizing.

        Parameters
        ----------
        target_vol : float  – annualised target volatility (default 20%)

        Returns
        -------
        float : position size multiplier
        """
        if not self._fitted or self._last_vol <= 0:
            return 1.0

        # Convert conditional vol (percentage daily) to annualised
        # Assuming daily data and 252 trading days
        annualised_vol = self._last_vol * math.sqrt(252) / 100.0

        if annualised_vol <= 0:
            return 1.0

        scalar = target_vol / annualised_vol
        return float(np.clip(scalar, 0.1, 3.0))


# Need math for sqrt
import math
