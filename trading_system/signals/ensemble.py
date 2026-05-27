"""
Signal ensemble: combines signals from all categories into a composite,
using IC-weighted dynamic weighting and regime-aware scaling.

Signal return convention: float in [-1.0, +1.0].
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Signal categories and default weights
# ---------------------------------------------------------------------------

SIGNAL_CATEGORIES = [
    "trend",
    "reversion",
    "momentum",
    "volume",
    "microstructure",
    "stat_arb",
    "regime",
    "sentiment",
    "onchain",
]

# Default weights per category (must sum to ≤ 1.0; not required to sum to 1 exactly)
DEFAULT_WEIGHTS: Dict[str, float] = {
    # Technical
    "dual_ema":          0.07,
    "triple_ema":        0.05,
    "macd":              0.06,
    "ichimoku":          0.04,
    # Reversion
    "bollinger":         0.05,
    "rsi":               0.05,
    "stochastic":        0.04,
    "vwap_rev":          0.04,
    "williams_r":        0.03,
    # Momentum
    "roc":               0.04,
    "chande_momentum":   0.03,
    "cmf":               0.03,
    "composite_momentum": 0.04,
    # Volume
    "obv":               0.03,
    "volume_surge":      0.02,
    "mfi":               0.03,
    "vwap_dev":          0.03,
    # Microstructure
    "obi":               0.05,
    "hawkes":            0.04,
    "cvd":               0.04,
    # On-chain (only for crypto assets)
    "btc_onchain":       0.04,
    "eth_onchain":       0.04,
    # Sentiment
    "reddit":            0.03,
    "news":              0.03,
    "fear_greed":        0.02,
}


# ---------------------------------------------------------------------------
# Information Coefficient utilities
# ---------------------------------------------------------------------------

def compute_rolling_ic(
    signals_history: Dict[str, List[float]],
    returns_history: List[float],
    lookback: int = 50,
) -> Dict[str, float]:
    """
    Compute rolling Spearman rank IC for each signal vs forward returns.

    Parameters
    ----------
    signals_history : dict[str, list[float]]
        Historical signal values per signal name.
    returns_history : list[float]
        Historical forward returns (same length as each signal list).
    lookback : int
        Number of bars to use for IC calculation.

    Returns
    -------
    dict[str, float]  – IC per signal (Spearman correlation with returns)
    """
    from scipy.stats import spearmanr

    ics: Dict[str, float] = {}
    ret_arr = np.array(returns_history[-lookback:], dtype=float)

    for name, sig_list in signals_history.items():
        sig_arr = np.array(sig_list[-lookback:], dtype=float)

        # Align lengths
        min_len = min(len(sig_arr), len(ret_arr))
        if min_len < 10:
            ics[name] = 0.0
            continue

        s = sig_arr[-min_len:]
        r = ret_arr[-min_len:]

        # Remove NaN pairs
        mask = ~(np.isnan(s) | np.isnan(r))
        if mask.sum() < 5:
            ics[name] = 0.0
            continue

        try:
            corr, pval = spearmanr(s[mask], r[mask])
            # Penalize by p-value: if not significant, scale toward 0
            significance = max(0.0, 1.0 - pval * 2.0)  # 0.05 → 0.9, 0.5 → 0
            ics[name] = float(corr * significance)
        except Exception:
            ics[name] = 0.0

    return ics


def optimize_weights(
    ics: Dict[str, float],
    min_weight: float = 0.05,
) -> Dict[str, float]:
    """
    Compute IC-proportional weights with a minimum weight floor.

    Parameters
    ----------
    ics        : dict[str, float]  – IC per signal
    min_weight : float             – minimum weight floor for any signal

    Returns
    -------
    dict[str, float]  – normalized weights summing to 1.0
    """
    if not ics:
        return {}

    names = list(ics.keys())
    # Use positive IC only; negative IC → use min_weight (signal is anti-predictive)
    positive_ic = {k: max(v, 0.0) for k, v in ics.items()}

    total_ic = sum(positive_ic.values())

    raw_weights: Dict[str, float] = {}
    if total_ic > 0:
        for name in names:
            raw_weights[name] = positive_ic[name] / total_ic
    else:
        # All ICs zero or negative → equal weights
        n = len(names)
        for name in names:
            raw_weights[name] = 1.0 / n

    # Apply minimum weight floor
    for name in names:
        if raw_weights[name] < min_weight:
            raw_weights[name] = min_weight

    # Re-normalize to sum to 1.0
    total = sum(raw_weights.values())
    normalized = {k: v / total for k, v in raw_weights.items()}

    return normalized


# ---------------------------------------------------------------------------
# Composite signal computation
# ---------------------------------------------------------------------------

def compute_composite(
    signals_dict: Dict[str, float],
    weights: Dict[str, float],
    regime_data: Optional[dict] = None,
    asset_type: str = "crypto",
) -> Tuple[float, dict]:
    """
    Compute weighted composite signal with regime adjustment.

    Parameters
    ----------
    signals_dict : dict[str, float]
        Current signal values keyed by signal name.
    weights      : dict[str, float]
        Weight per signal (should sum approximately to 1.0).
    regime_data  : dict | None
        Optional regime context. Supported keys:
          'adx_regime' : dict from adx_regime() → {'regime', 'multiplier', 'direction'}
          'hmm_bull_prob' : float [0,1]
          'macro_scalar'  : float [0.2, 1.1]
          'garch_vol_regime' : str
    asset_type   : str
        'crypto' | 'equity' | 'fx'  – adjusts which signals are active.

    Returns
    -------
    (composite, metadata)
        composite : float in [-1, +1]
        metadata  : dict with component contributions and regime info
    """
    if not signals_dict:
        return (0.0, {"error": "no signals provided"})

    # Use provided weights or fall back to defaults
    active_weights = weights if weights else DEFAULT_WEIGHTS

    # Exclude on-chain signals for non-crypto assets
    if asset_type in ("equity", "fx"):
        active_weights = {
            k: v for k, v in active_weights.items()
            if k not in ("btc_onchain", "eth_onchain")
        }

    # Compute weighted sum over signals that are available
    contributions: Dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for name, signal_val in signals_dict.items():
        w = active_weights.get(name, 0.0)
        if w <= 0:
            continue
        if np.isnan(signal_val):
            signal_val = 0.0
        contribution = w * float(signal_val)
        contributions[name] = contribution
        weighted_sum += contribution
        total_weight += w

    if total_weight <= 0:
        return (0.0, {"error": "no matching weights"})

    raw_composite = weighted_sum / total_weight

    # ------------------------------------------------------------------
    # Regime adjustments
    # ------------------------------------------------------------------
    regime_multiplier = 1.0
    regime_info: Dict[str, object] = {}

    if regime_data:
        # ADX regime adjustment
        adx = regime_data.get("adx_regime")
        if adx and isinstance(adx, dict):
            adx_regime_str = adx.get("regime", "ranging")
            adx_mult = float(adx.get("multiplier", 1.0))
            adx_dir = float(adx.get("direction", 0.0))

            if adx_regime_str == "trending":
                # In trending regime, amplify if signal aligns with ADX direction
                alignment = np.sign(raw_composite) == np.sign(adx_dir)
                if alignment and adx_dir != 0:
                    regime_multiplier *= min(adx_mult, 1.4)
                elif not alignment:
                    regime_multiplier *= 0.6  # counter-trend: reduce
            elif adx_regime_str == "ranging":
                # In ranging regime, favour reversion signals; dampen trend signals
                regime_multiplier *= 0.7
            regime_info["adx"] = adx_regime_str

        # HMM bull probability
        hmm_prob = regime_data.get("hmm_bull_prob")
        if hmm_prob is not None:
            hmm_mult = 0.5 + float(hmm_prob) * 1.0  # [0.5, 1.5]
            # HMM multiplier modulates direction, not magnitude
            if float(hmm_prob) < 0.35:
                # Bear regime: dampen bullish signals
                if raw_composite > 0:
                    regime_multiplier *= 0.6
            elif float(hmm_prob) > 0.65:
                # Bull regime: amplify bullish signals
                if raw_composite > 0:
                    regime_multiplier *= 1.2
            regime_info["hmm_bull_prob"] = float(hmm_prob)

        # Macro scalar
        macro_scalar = regime_data.get("macro_scalar")
        if macro_scalar is not None:
            regime_multiplier *= float(macro_scalar)
            regime_info["macro_scalar"] = float(macro_scalar)

        # GARCH vol regime
        garch_regime = regime_data.get("garch_vol_regime")
        if garch_regime == "extreme_vol":
            regime_multiplier *= 0.3
        elif garch_regime == "high_vol":
            regime_multiplier *= 0.6
        regime_info["garch_vol_regime"] = garch_regime

    # Apply regime multiplier (preserves sign)
    adjusted = raw_composite * regime_multiplier

    # Final tanh normalization to ensure [-1, +1]
    final = float(np.tanh(adjusted * 2.0))
    final = float(np.clip(final, -1.0, 1.0))

    metadata = {
        "raw_composite": float(raw_composite),
        "regime_multiplier": float(regime_multiplier),
        "adjusted_pre_tanh": float(adjusted),
        "final": final,
        "contributions": contributions,
        "n_signals": len(contributions),
        "regime_info": regime_info,
    }

    return (final, metadata)


# ---------------------------------------------------------------------------
# Decision threshold check
# ---------------------------------------------------------------------------

_CONVICTION_LEVELS = [
    (0.80, "extreme"),
    (0.60, "high"),
    (0.40, "medium"),
    (0.20, "low"),
    (0.00, "none"),
]


def threshold_check(composite: float) -> Tuple[str, str]:
    """
    Translate composite signal to a trading decision and conviction level.

    Parameters
    ----------
    composite : float in [-1, +1]

    Returns
    -------
    (decision, conviction)
        decision   : 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell'
        conviction : 'extreme' | 'high' | 'medium' | 'low' | 'none'
    """
    abs_val = abs(composite)
    direction = 1 if composite >= 0 else -1

    # Determine conviction level
    conviction = "none"
    for threshold, level in _CONVICTION_LEVELS:
        if abs_val >= threshold:
            conviction = level
            break

    # Determine decision
    if abs_val >= 0.70:
        decision = "strong_buy" if direction > 0 else "strong_sell"
    elif abs_val >= 0.40:
        decision = "buy" if direction > 0 else "sell"
    elif abs_val >= 0.15:
        decision = "buy" if direction > 0 else "sell"
    else:
        decision = "hold"

    return (decision, conviction)
