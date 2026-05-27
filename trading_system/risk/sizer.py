"""
Position sizing with 7-layer risk framework:
Kelly → Vol Scale → CVaR → Conviction → Correlation → Macro+GARCH → Hard Caps
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading_system.core.logger import get_logger

log = get_logger(__name__)

# Hard cap constants
_MAX_SINGLE_POSITION_PCT = 0.15      # 15% of portfolio per position
_MAX_TOTAL_LONG_PCT = 0.80           # 80% total gross long exposure
_MIN_POSITION_PCT = 0.001            # 0.1% minimum (below = skip)
_KELLY_FRACTION = 0.25               # Fractional Kelly (25%)
_TARGET_ANNUAL_VOL = 0.20            # 20% target annual vol
_CVAR_LIMIT = 0.02                   # 2% max CVaR contribution per position


class PositionSizer:
    """
    Multi-layer position sizing with risk controls.

    Layers applied in sequence:
        1. Kelly criterion (base size)
        2. Volatility scaling (target vol budget)
        3. CVaR constraint (tail risk limit)
        4. Conviction adjustment (signal strength)
        5. Correlation penalty (portfolio concentration)
        6. Macro + GARCH volatility adjustments
        7. Hard caps (max position, max total exposure)
    """

    def __init__(
        self,
        kelly_fraction: float = _KELLY_FRACTION,
        target_annual_vol: float = _TARGET_ANNUAL_VOL,
        cvar_limit: float = _CVAR_LIMIT,
        max_position_pct: float = _MAX_SINGLE_POSITION_PCT,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.target_annual_vol = target_annual_vol
        self.cvar_limit = cvar_limit
        self.max_position_pct = max_position_pct

    # ------------------------------------------------------------------
    # Layer 1: Kelly Criterion
    # ------------------------------------------------------------------

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Compute fractional Kelly position size.

        f* = (win_rate / |avg_loss|) - ((1 - win_rate) / avg_win)
        Returns fractional_kelly * f* clamped to [0, 1].

        Parameters
        ----------
        win_rate : float in [0, 1]
        avg_win : float (positive, average winning trade P&L as fraction)
        avg_loss : float (positive magnitude of average losing trade)

        Returns
        -------
        float in [0, 1]
        """
        avg_win = abs(avg_win) + 1e-8
        avg_loss = abs(avg_loss) + 1e-8
        win_rate = float(np.clip(win_rate, 0.0, 1.0))

        # Kelly formula: f* = p/b - q/1 where b = avg_win/avg_loss, q = 1-p
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        f_star = (win_rate / avg_loss) - (q / avg_win)

        # Apply fractional Kelly and floor at 0
        kelly = self.kelly_fraction * f_star
        return float(np.clip(kelly, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Layer 2: Volatility Scalar
    # ------------------------------------------------------------------

    def volatility_scalar(self, asset_annual_vol: float) -> float:
        """
        Compute volatility scalar to normalize to target annual volatility.

        scalar = target_annual_vol / asset_annual_vol
        Clamped to [0.1, 3.0] to prevent extreme adjustments.

        Parameters
        ----------
        asset_annual_vol : float (e.g. 0.30 = 30% annual vol)

        Returns
        -------
        float in [0.1, 3.0]
        """
        if asset_annual_vol <= 0:
            return 1.0
        scalar = self.target_annual_vol / asset_annual_vol
        return float(np.clip(scalar, 0.1, 3.0))

    # ------------------------------------------------------------------
    # Layer 3: CVaR Constraint
    # ------------------------------------------------------------------

    def cvar_constraint(
        self,
        kelly_size: float,
        asset_annual_vol: float,
        portfolio_value: float,
    ) -> float:
        """
        Constrain position so its CVaR contribution does not exceed limit.

        CVaR contribution (approx) = position_dollar * asset_daily_vol * 2.326
        (2.326 is the 99% normal quantile)
        Constraint: CVaR / portfolio_value <= cvar_limit

        Parameters
        ----------
        kelly_size : float in [0, 1] (fraction of portfolio)
        asset_annual_vol : float (annualized vol, e.g. 0.30)
        portfolio_value : float (total portfolio value)

        Returns
        -------
        float
            Constrained position fraction (may be smaller than kelly_size).
        """
        if portfolio_value <= 0 or asset_annual_vol <= 0:
            return kelly_size

        # Convert to daily vol
        daily_vol = asset_annual_vol / math.sqrt(252)
        # 99% CVaR multiplier for normal distribution
        cvar_multiplier = 2.326

        # Max fraction: limit * portfolio / (dollar * multiplier)
        # dollar = fraction * portfolio_value
        # Solve: fraction * portfolio_value * daily_vol * cvar_multiplier / portfolio_value <= limit
        # fraction <= limit / (daily_vol * cvar_multiplier)
        max_fraction = self.cvar_limit / (daily_vol * cvar_multiplier + 1e-8)

        return float(min(kelly_size, max_fraction))

    # ------------------------------------------------------------------
    # Layer 4: Conviction (signal strength)
    # ------------------------------------------------------------------

    def conviction_scalar(self, signal_strength: float) -> float:
        """
        Scale position by signal conviction.

        signal_strength in [0, 1] where 1 = maximum conviction.
        scalar = signal_strength ** 0.5  (square root to prevent overconcentration)
        Clamped to [0.2, 1.0].
        """
        signal_strength = float(abs(np.clip(signal_strength, 0.0, 1.0)))
        scalar = signal_strength ** 0.5
        return float(np.clip(scalar, 0.2, 1.0))

    # ------------------------------------------------------------------
    # Layer 5: Correlation Penalty
    # ------------------------------------------------------------------

    def correlation_penalty(
        self,
        new_asset: str,
        open_positions: Dict[str, float],
        corr_matrix: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Apply penalty for portfolio concentration from correlated positions.

        For each open position, fetch the correlation to new_asset.
        penalty = 1 - sum(max_corr_contribution per existing position)
        Minimum penalty factor: 0.3 (max 70% size reduction from correlation).

        Parameters
        ----------
        new_asset : str
        open_positions : dict {asset: position_size_pct}
        corr_matrix : pd.DataFrame or None

        Returns
        -------
        float in [0.3, 1.0]
        """
        if not open_positions or corr_matrix is None:
            return 1.0

        total_corr_drag = 0.0
        for existing_asset, pos_size in open_positions.items():
            try:
                corr = self._get_corr(new_asset, existing_asset, corr_matrix)
                # Contribution: correlation * size of existing position
                corr_contribution = abs(corr) * abs(pos_size) * 0.5
                total_corr_drag += corr_contribution
            except Exception:
                continue

        penalty = 1.0 - min(total_corr_drag, 0.7)
        return float(np.clip(penalty, 0.3, 1.0))

    def _get_corr(
        self,
        asset_a: str,
        asset_b: str,
        corr_matrix: pd.DataFrame,
    ) -> float:
        """Get correlation between two assets, return 0 if not found."""
        if asset_a == asset_b:
            return 1.0
        try:
            if asset_a in corr_matrix.index and asset_b in corr_matrix.columns:
                return float(corr_matrix.loc[asset_a, asset_b])
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Final computation: 7 layers combined
    # ------------------------------------------------------------------

    def compute_final_size(
        self,
        asset: str,
        signal: float,
        portfolio_value: float,
        trade_history: List[Dict],
        open_positions: Dict[str, float],
        corr_matrix: Optional[pd.DataFrame],
        asset_annual_vol: float,
        macro_multiplier: float = 1.0,
        garch_scalar: float = 1.0,
    ) -> Tuple[float, float, Dict]:
        """
        Compute final position size through all 7 risk layers.

        Parameters
        ----------
        asset : str
        signal : float in [-1, +1] (trading signal strength; sign = direction)
        portfolio_value : float
        trade_history : list of completed trade dicts with 'pnl' and 'entry_price'
        open_positions : dict {asset: size_pct}
        corr_matrix : pd.DataFrame or None
        asset_annual_vol : float (annualized vol)
        macro_multiplier : float (macro regime scalar, typically 0.5-1.2)
        garch_scalar : float (GARCH vol forecast scalar, typically 0.5-1.5)

        Returns
        -------
        (dollar_size, size_pct, breakdown_dict)
            dollar_size : float (USD position size)
            size_pct : float (fraction of portfolio)
            breakdown_dict : dict with each layer's output for auditing
        """
        breakdown: Dict[str, float] = {}

        # Layer 1: Kelly
        win_rate, avg_win, avg_loss = self._compute_trade_stats(trade_history)
        layer1_kelly = self.kelly_size(win_rate, avg_win, avg_loss)
        breakdown["kelly_raw"] = layer1_kelly

        # Layer 2: Volatility scaling
        vol_scalar = self.volatility_scalar(asset_annual_vol)
        layer2 = layer1_kelly * vol_scalar
        breakdown["vol_scalar"] = vol_scalar
        breakdown["after_vol_scale"] = layer2

        # Layer 3: CVaR constraint
        layer3 = self.cvar_constraint(layer2, asset_annual_vol, portfolio_value)
        breakdown["cvar_constraint"] = layer3

        # Layer 4: Conviction scalar
        conviction = self.conviction_scalar(abs(signal))
        layer4 = layer3 * conviction
        breakdown["conviction_scalar"] = conviction
        breakdown["after_conviction"] = layer4

        # Layer 5: Correlation penalty
        corr_factor = self.correlation_penalty(asset, open_positions, corr_matrix)
        layer5 = layer4 * corr_factor
        breakdown["correlation_penalty"] = corr_factor
        breakdown["after_correlation"] = layer5

        # Layer 6: Macro + GARCH adjustment
        # Low vol regime: can increase size; high vol regime: reduce size
        macro_garch_scalar = macro_multiplier / max(garch_scalar, 0.1)
        macro_garch_scalar = float(np.clip(macro_garch_scalar, 0.1, 2.0))
        layer6 = layer5 * macro_garch_scalar
        breakdown["macro_multiplier"] = macro_multiplier
        breakdown["garch_scalar"] = garch_scalar
        breakdown["macro_garch_scalar"] = macro_garch_scalar
        breakdown["after_macro_garch"] = layer6

        # Layer 7: Hard caps
        # a) Single position cap
        layer7 = min(layer6, self.max_position_pct)
        # b) Total long cap: check remaining budget
        current_total_long = sum(abs(v) for v in open_positions.values())
        remaining_budget = max(_MAX_TOTAL_LONG_PCT - current_total_long, 0.0)
        layer7 = min(layer7, remaining_budget)
        # c) Minimum position floor
        if layer7 < _MIN_POSITION_PCT:
            layer7 = 0.0

        breakdown["hard_cap_max_pct"] = self.max_position_pct
        breakdown["remaining_long_budget"] = remaining_budget
        breakdown["final_size_pct"] = layer7

        size_pct = float(layer7)
        dollar_size = size_pct * portfolio_value

        log.debug(
            "Position size computed",
            asset=asset,
            signal=round(signal, 3),
            size_pct=round(size_pct, 4),
            dollar_size=round(dollar_size, 2),
        )

        return dollar_size, size_pct, breakdown

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_trade_stats(
        self,
        trade_history: List[Dict],
    ) -> Tuple[float, float, float]:
        """
        Compute win rate, average win, and average loss from trade history.

        Returns (win_rate, avg_win, avg_loss) with safe defaults.
        """
        if not trade_history:
            # No history: assume 52% win rate, 1:1 win/loss ratio
            return 0.52, 0.01, 0.01

        pnl_pcts = []
        for t in trade_history:
            pnl = t.get("pnl_pct") or t.get("pnl", 0.0)
            try:
                pnl_pcts.append(float(pnl))
            except (TypeError, ValueError):
                continue

        if len(pnl_pcts) < 5:
            return 0.52, 0.01, 0.01

        wins = [p for p in pnl_pcts if p > 0]
        losses = [abs(p) for p in pnl_pcts if p < 0]

        win_rate = len(wins) / len(pnl_pcts)
        avg_win = float(np.mean(wins)) if wins else 0.01
        avg_loss = float(np.mean(losses)) if losses else 0.01

        # Floor both at 0.001 to prevent division by zero
        avg_win = max(avg_win, 0.001)
        avg_loss = max(avg_loss, 0.001)

        return win_rate, avg_win, avg_loss
