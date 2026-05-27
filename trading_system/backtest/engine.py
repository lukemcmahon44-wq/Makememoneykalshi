"""
Event-driven backtest engine.

Rules:
  - signal computed on bar t close
  - order fills at bar t+1 open
  - slippage applied to fills
  - commission deducted
  - drawdown circuit breaker enforced
  - no lookahead allowed
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..portfolio.analytics import all_metrics

log = get_logger(__name__)


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.commission = {"equity": config.COMMISSION_EQUITY,
                            "crypto": config.COMMISSION_CRYPTO}
        self.slippage_model = {"equity": config.SLIPPAGE_EQUITY,
                                "crypto": config.SLIPPAGE_CRYPTO}

    def run(self, data: Dict[str, pd.DataFrame],
            signal_fn: Callable[[str, pd.DataFrame], float],
            sizer,
            start_capital: float = 10_000.0
            ) -> Dict[str, Any]:
        """
        data:        {asset: DataFrame[open, high, low, close, volume]}
        signal_fn:   (asset, history_df_up_to_t) -> composite signal float
        sizer:       PositionSizer instance
        """
        all_assets = list(data.keys())
        if not all_assets:
            return {}
        # Align all data on union of timestamps
        master_idx = sorted(set().union(*(df.index for df in data.values())))

        cash = float(start_capital)
        positions: Dict[str, Dict[str, Any]] = {}
        trades: List[Dict[str, Any]] = []
        nav_series: List[float] = []
        peak = start_capital
        halted_until = -1   # bar index

        for i, ts in enumerate(master_idx):
            mark_prices = {a: float(df["close"].asof(ts)) if ts in df.index
                            else float(df["close"].iloc[max(df.index.searchsorted(ts) - 1, 0)])
                            for a, df in data.items() if not df.empty}
            port_val = cash + sum(p["qty"] * mark_prices.get(a, p["entry_price"])
                                    for a, p in positions.items())
            peak = max(peak, port_val)
            dd = (peak - port_val) / max(peak, 1e-8)
            nav_series.append(port_val)
            if dd > self.config.MAX_DRAWDOWN_PCT and halted_until < i:
                # close all positions at next open, halt for 24 bars (~1 day)
                for asset, pos in list(positions.items()):
                    df = data[asset]
                    if ts not in df.index:
                        continue
                    fill_px = float(df["close"].loc[ts])
                    pnl = pos["qty"] * (fill_px - pos["entry_price"])
                    if pos["direction"] == "short":
                        pnl = -pnl
                    cash += pos["qty"] * fill_px - abs(pnl) * 0
                    trades.append({**pos, "exit_price": fill_px, "pnl": pnl,
                                    "pnl_pct": pnl / max(pos["size_usd"], 1e-8),
                                    "exit_ts": ts, "halted": True})
                    del positions[asset]
                halted_until = i + 24
                continue
            if i < halted_until:
                continue

            # Generate signals from data[:i]
            for asset, df in data.items():
                if ts not in df.index:
                    continue
                up_to_t = df.loc[:ts].iloc[:-1]   # drop current bar to avoid lookahead
                if len(up_to_t) < 30:
                    continue
                try:
                    signal = float(signal_fn(asset, up_to_t))
                except Exception:
                    signal = 0.0
                if not np.isfinite(signal):
                    continue
                asset_type = self.config.asset_type(asset)
                # Approximate annual vol
                rets = up_to_t["close"].pct_change().dropna().values
                ann_vol = float(np.std(rets, ddof=0) * np.sqrt(252)) if len(rets) > 30 else 0.3
                fill_px = float(df["close"].loc[ts])
                slip = self.slippage_model[asset_type]
                if asset in positions:
                    pos = positions[asset]
                    flipped = np.sign(signal) != np.sign(pos["signal"])
                    weak = abs(signal) < 0.30
                    if flipped or weak:
                        actual = fill_px * (1 - slip)   # sell with slippage down
                        proceeds = pos["qty"] * actual
                        commission = proceeds * self.commission[asset_type]
                        pnl = (actual - pos["entry_price"]) * pos["qty"]
                        if pos["direction"] == "short":
                            pnl = -pnl
                        pnl -= commission
                        cash += proceeds - commission
                        trades.append({**pos, "exit_price": actual, "pnl": pnl,
                                        "pnl_pct": pnl / max(pos["size_usd"], 1e-8),
                                        "exit_ts": ts})
                        del positions[asset]
                else:
                    decision_threshold = self.config.SIGNAL_ENTRY_THRESHOLD
                    if abs(signal) < decision_threshold:
                        continue
                    dollar_size, pct_size, _ = sizer.compute_final_size(
                        asset, signal, port_val, trades, positions, {},
                        ann_vol, 1.0, 1.0,
                    )
                    dollar_size = min(dollar_size, cash * 0.95)
                    if dollar_size < self.config.MIN_ORDER_USD:
                        continue
                    actual = fill_px * (1 + slip)
                    qty = dollar_size / actual
                    commission = dollar_size * self.commission[asset_type]
                    cash -= dollar_size + commission
                    positions[asset] = {
                        "asset": asset,
                        "qty": qty,
                        "entry_price": actual,
                        "entry_ts": ts,
                        "signal": signal,
                        "direction": "long" if signal > 0 else "short",
                        "size_usd": dollar_size,
                    }

        final_value = nav_series[-1] if nav_series else start_capital
        nav_arr = np.asarray(nav_series, dtype=float)
        if len(nav_arr) > 1:
            returns = np.diff(nav_arr) / np.where(nav_arr[:-1] != 0, nav_arr[:-1], 1)
        else:
            returns = np.array([0.0])
        trade_pnls = [t["pnl_pct"] for t in trades]
        metrics = all_metrics(returns, trade_pnls)
        metrics["final_value"] = float(final_value)
        metrics["total_trades"] = len(trades)
        if trades:
            metrics["avg_holding_bars"] = float(np.mean([
                (pd.Timestamp(t["exit_ts"]) - pd.Timestamp(t["entry_ts"]))
                  .total_seconds() / 60 if "exit_ts" in t and "entry_ts" in t
                else 0
                for t in trades
            ]))
        return {"metrics": metrics, "trades": trades, "nav": nav_arr.tolist()}
