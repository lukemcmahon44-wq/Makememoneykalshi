"""
Layered position sizing: Kelly + volatility scaling + CVaR + conviction
+ correlation penalty + hard caps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PositionSizer:
    def __init__(self, config):
        self.config = config
        self.target_vol = config.TARGET_ANNUAL_VOL
        self.kelly_fraction = config.KELLY_FRACTION
        self.max_position_pct = config.MAX_POSITION_PCT
        self.max_total_long = config.MAX_TOTAL_LONG
        self.cvar_limit = config.CVAR_LIMIT

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float
                   ) -> float:
        if avg_loss < 1e-8:
            return 0.02
        p = float(np.clip(win_rate, 0.48, 0.65))
        q = 1 - p
        b = avg_win / avg_loss
        f_full = max(0.0, (b * p - q) / b)
        f_kelly = self.kelly_fraction * f_full
        return float(np.clip(f_kelly, 0.01, 0.20))

    def volatility_scalar(self, asset_annual_vol: float) -> float:
        return float(np.clip(self.target_vol / (asset_annual_vol + 0.01), 0.3, 3.0))

    def cvar_constraint(self, kelly_size: float, asset_annual_vol: float
                        ) -> float:
        daily_vol = asset_annual_vol / np.sqrt(252)
        # CVaR_95 ≈ 2.33 σ; size so |size * 2.33 σ_daily| ≤ cvar_limit
        cvar_size = self.cvar_limit / (2.33 * daily_vol + 1e-8)
        return float(min(kelly_size, cvar_size))

    def correlation_penalty(self, new_asset: str,
                            open_positions: Dict[str, Any],
                            corr_matrix: Dict[Tuple[str, str], float]
                            ) -> float:
        if not open_positions:
            return 1.0
        max_corr = 0.0
        for pos_asset in open_positions:
            key = (new_asset, pos_asset) if (new_asset, pos_asset) in corr_matrix else (pos_asset, new_asset)
            corr = abs(float(corr_matrix.get(key, 0.0)))
            max_corr = max(max_corr, corr)
        if max_corr > 0.8:
            return 0.40
        if max_corr > 0.6:
            return 0.65
        if max_corr > 0.4:
            return 0.80
        return 1.0

    def compute_final_size(self,
                            asset: str,
                            signal: float,
                            portfolio_value: float,
                            trade_history: List[Dict[str, Any]],
                            open_positions: Dict[str, Any],
                            corr_matrix: Dict[Tuple[str, str], float],
                            asset_annual_vol: float,
                            macro_multiplier: float = 1.0,
                            garch_scalar: float = 1.0,
                            ) -> Tuple[float, float, Dict[str, float]]:
        """
        Returns (dollar_size, pct_size, breakdown).
        """
        # Base Kelly from trade history
        if len(trade_history) < 20:
            base_kelly = 0.03
        else:
            recent = trade_history[-50:]
            wins = [t for t in recent if (t.get("pnl") or 0) > 0]
            losses = [t for t in recent if (t.get("pnl") or 0) <= 0]
            win_rate = len(wins) / max(len(recent), 1)
            avg_win = float(np.mean([t.get("pnl_pct", 0) for t in wins])) if wins else 0.01
            avg_loss = float(abs(np.mean([t.get("pnl_pct", 0) for t in losses]))) if losses else 0.01
            base_kelly = self.kelly_size(win_rate, avg_win, avg_loss)

        size = base_kelly * self.volatility_scalar(asset_annual_vol)
        size = self.cvar_constraint(size, asset_annual_vol)

        abs_signal = abs(float(signal))
        if abs_signal >= 0.75:
            size *= 1.25
        elif abs_signal < 0.60:
            size *= 0.75

        size *= self.correlation_penalty(asset, open_positions, corr_matrix)
        size *= float(macro_multiplier) * float(garch_scalar)
        size = float(np.clip(size, 0.005, self.max_position_pct))

        current_long_exposure = sum(
            float(p.get("size_pct", 0)) for p in open_positions.values()
            if p.get("direction") == "long"
        )
        if current_long_exposure + size > self.max_total_long:
            size = max(0.0, self.max_total_long - current_long_exposure)

        dollar_size = size * portfolio_value
        breakdown = {
            "kelly_base": base_kelly,
            "vol_scaled": size,
            "macro_applied": macro_multiplier,
            "garch_applied": garch_scalar,
        }
        return dollar_size, size, breakdown
