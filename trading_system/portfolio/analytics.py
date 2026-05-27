"""
Performance analytics: Sharpe, Sortino, Calmar, Omega, max drawdown, etc.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        return returns.dropna().values
    return np.asarray([r for r in returns if r is not None and np.isfinite(r)],
                       dtype=float)


def sharpe_ratio(returns, periods_per_year: int = 252) -> float:
    arr = _to_array(returns)
    if len(arr) < 2:
        return 0.0
    sd = float(np.std(arr, ddof=0))
    if sd < 1e-10:
        return 0.0
    return float(np.mean(arr) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year: int = 252) -> float:
    arr = _to_array(returns)
    if len(arr) < 2:
        return 0.0
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float(np.mean(arr) * np.sqrt(periods_per_year) / 1e-8)
    dd_std = float(np.std(downside, ddof=0))
    if dd_std < 1e-10:
        return 0.0
    return float(np.mean(arr) / dd_std * np.sqrt(periods_per_year))


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    arr = _to_array(returns)
    if len(arr) < 2:
        return 0.0
    annual = float(np.mean(arr) * periods_per_year)
    mdd = max_drawdown(arr)
    return annual / max(mdd, 1e-8)


def max_drawdown(returns) -> float:
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    eq = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return float(np.max(dd))


def win_rate(pnls) -> float:
    arr = _to_array(pnls)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr > 0))


def profit_factor(pnls) -> float:
    arr = _to_array(pnls)
    if len(arr) == 0:
        return 0.0
    wins = float(arr[arr > 0].sum())
    losses = float(arr[arr < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / abs(losses)


def avg_win_loss(pnls) -> Dict[str, float]:
    arr = _to_array(pnls)
    if len(arr) == 0:
        return {"avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    win_rt = float(np.mean(arr > 0))
    expectancy = win_rt * avg_win + (1 - win_rt) * avg_loss
    return {"avg_win": avg_win, "avg_loss": avg_loss, "expectancy": expectancy}


def all_metrics(returns, trade_pnls=None) -> Dict[str, float]:
    arr = _to_array(returns)
    out: Dict[str, float] = {
        "total_return": float(np.prod(1 + arr) - 1) if len(arr) else 0.0,
        "sharpe": sharpe_ratio(arr),
        "sortino": sortino_ratio(arr),
        "calmar": calmar_ratio(arr),
        "max_drawdown": max_drawdown(arr),
        "volatility": float(np.std(arr, ddof=0) * np.sqrt(252)) if len(arr) else 0.0,
    }
    if trade_pnls is not None:
        out.update(avg_win_loss(trade_pnls))
        out["win_rate"] = win_rate(trade_pnls)
        out["profit_factor"] = profit_factor(trade_pnls)
        out["n_trades"] = int(len(_to_array(trade_pnls)))
    return out
