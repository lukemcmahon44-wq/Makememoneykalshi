"""
Signal ensemble. Dynamically reweighs categories by rolling IC.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from ._common import EPS


class SignalEnsemble:
    SIGNAL_CATEGORIES = (
        "technical_trend",
        "technical_reversion",
        "microstructure",
        "stat_arb",
        "sentiment",
        "onchain",
        "ml_prediction",
    )

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "technical_trend": 0.10,
        "technical_reversion": 0.10,
        "microstructure": 0.20,
        "stat_arb": 0.15,
        "sentiment": 0.10,
        "onchain": 0.10,
        "ml_prediction": 0.25,
    }
    MIN_WEIGHT = 0.05

    def __init__(self, ic_lookback: int = 50):
        self.weights: Dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self.signal_history: Deque[Dict[str, float]] = deque(maxlen=500)
        self.return_history: Deque[float] = deque(maxlen=500)
        self.ic_lookback = ic_lookback
        self.ml_enabled = False    # gated off until OOS metrics pass

    def record_observation(self, signals: Dict[str, float], realized_return: float
                            ) -> None:
        self.signal_history.append(dict(signals))
        self.return_history.append(float(realized_return))

    def compute_rolling_ic(self) -> Dict[str, float]:
        ics: Dict[str, float] = {}
        if len(self.signal_history) < 10:
            return {c: 0.0 for c in self.SIGNAL_CATEGORIES}
        rets = list(self.return_history)[-self.ic_lookback:]
        for cat in self.SIGNAL_CATEGORIES:
            sig = []
            ret = []
            for h, r in zip(list(self.signal_history)[-self.ic_lookback:], rets):
                v = h.get(cat)
                if v is None:
                    continue
                sig.append(v)
                ret.append(r)
            if len(sig) > 10:
                try:
                    ic = float(np.corrcoef(sig, ret)[0, 1])
                except Exception:
                    ic = 0.0
                ics[cat] = max(0.0, ic if np.isfinite(ic) else 0.0)
            else:
                ics[cat] = self.DEFAULT_WEIGHTS[cat]
        return ics

    def update_weights(self) -> Dict[str, float]:
        ics = self.compute_rolling_ic()
        total = sum(ics.values()) + EPS
        raw = {c: max(self.MIN_WEIGHT, v / total) for c, v in ics.items()}
        norm = sum(raw.values())
        self.weights = {c: v / norm for c, v in raw.items()}
        return dict(self.weights)

    def compute_composite(self, signals_dict: Dict[str, float],
                          regime_data: Dict[str, float], asset_type: str = "equity"
                          ) -> Tuple[float, Dict[str, Any]]:
        weights = dict(self.weights)

        # Disable ML category if not yet enabled
        if not self.ml_enabled:
            weights["ml_prediction"] = 0.0

        # Equity: no onchain - redistribute
        if asset_type == "equity":
            onchain_w = weights.pop("onchain", 0.0)
            if onchain_w > 0:
                weights["microstructure"] = weights.get("microstructure", 0) + onchain_w

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        composite = 0.0
        for cat, w in weights.items():
            v = signals_dict.get(cat)
            if v is None:
                continue
            composite += w * float(v)
        composite = float(np.clip(composite, -1.0, 1.0))

        # Apply regime multipliers (size scalars are computed separately; here
        # ADX regime can amplify the directional confidence)
        adx_regime = regime_data.get("adx_regime", "transition")
        if adx_regime == "trending":
            composite = float(np.clip(composite * 1.2, -1.0, 1.0))
        elif adx_regime == "ranging":
            composite = float(np.clip(composite * 0.8, -1.0, 1.0))

        garch_scalar = float(regime_data.get("garch_size_scalar", 1.0))
        hmm_mult = float(regime_data.get("hmm_multiplier", 1.0))
        macro_mult = float(regime_data.get("macro_multiplier", 1.0))
        size_scalar = float(np.clip(garch_scalar * hmm_mult * macro_mult, 0.05, 2.5))

        return composite, {
            "raw_composite": composite,
            "weights_used": weights,
            "garch_scalar": garch_scalar,
            "hmm_mult": hmm_mult,
            "macro_mult": macro_mult,
            "effective_size_scalar": size_scalar,
        }

    @staticmethod
    def threshold_check(composite: float,
                        entry: float = 0.55,
                        high_conviction: float = 0.75) -> Tuple[str, str]:
        abs_c = abs(composite)
        if abs_c >= high_conviction:
            return "ENTER", "HIGH"
        if abs_c >= entry:
            return "ENTER", "NORMAL"
        if abs_c >= 0.40:
            return "WAIT", "WEAK"
        return "NO_SIGNAL", "NONE"

    def enable_ml(self) -> None:
        self.ml_enabled = True

    def disable_ml(self) -> None:
        self.ml_enabled = False
