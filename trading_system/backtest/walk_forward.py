"""
Walk-forward optimization for strategy parameters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .engine import BacktestEngine


def walk_forward_optimize(config, data: Dict[str, pd.DataFrame],
                           signal_fn_factory: Callable,
                           sizer,
                           param_grid: List[Dict[str, Any]],
                           train_bars: int = 30 * 390,
                           test_bars: int = 10 * 390,
                           ) -> Dict[str, Any]:
    """
    signal_fn_factory(params): returns a signal_fn closure for those params.
    Each fold: pick best params on train segment, evaluate on test segment.
    """
    engine = BacktestEngine(config)
    first_asset = next(iter(data.values()))
    n = len(first_asset)
    results: List[Dict[str, Any]] = []
    start = train_bars
    while start + test_bars <= n:
        train_slice = {a: df.iloc[start - train_bars: start] for a, df in data.items()}
        test_slice = {a: df.iloc[start: start + test_bars] for a, df in data.items()}
        best_params = None
        best_sharpe = -np.inf
        for params in param_grid:
            fn = signal_fn_factory(params)
            res = engine.run(train_slice, fn, sizer)
            s = res["metrics"].get("sharpe", 0)
            if s > best_sharpe:
                best_sharpe = s
                best_params = params
        if best_params is not None:
            test_fn = signal_fn_factory(best_params)
            test_res = engine.run(test_slice, test_fn, sizer)
            results.append({
                "fold": len(results),
                "best_params": best_params,
                "train_sharpe": best_sharpe,
                "test_metrics": test_res["metrics"],
            })
        start += test_bars
    profitable_pct = float(np.mean([r["test_metrics"].get("sharpe", 0) > 0
                                       for r in results])) if results else 0.0
    return {"folds": results, "profitable_pct": profitable_pct,
             "passed": profitable_pct > 0.6}
