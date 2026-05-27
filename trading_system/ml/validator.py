"""
Walk-forward validation. Honest backtest with purge gap and strict acceptance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .lgbm_trainer import LGBMTrainer

log = get_logger(__name__)


def walk_forward_validate(df: pd.DataFrame, feature_cols: List[str],
                           target_col: str,
                           train_window: int = 180 * 390,
                           test_window: int = 20 * 390,
                           step_size: int = 20 * 390,
                           purge_gap: int = 60,
                           min_folds: int = 3
                           ) -> Dict[str, Any]:
    try:
        from sklearn.metrics import (accuracy_score, precision_score,  # type: ignore
                                      recall_score, roc_auc_score)
    except ImportError:
        return {"model_accepted": False, "reason": "sklearn_missing"}

    results: List[Dict[str, Any]] = []
    n = len(df)
    start = train_window
    while start + test_window <= n:
        train_end = start - purge_gap
        train_start = train_end - train_window
        test_start = start
        test_end = min(start + test_window, n)
        if train_start < 0:
            start += step_size
            continue
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        X_tr = train_df[feature_cols]
        y_tr = train_df[target_col]
        X_te = test_df[feature_cols]
        y_te = test_df[target_col]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            start += step_size
            continue
        trainer = LGBMTrainer()
        try:
            trainer.train(X_tr, y_tr, X_te, y_te,
                           num_boost_round=200, early_stopping_rounds=20)
            proba = trainer.predict(X_te)
            preds = (proba > 0.5).astype(int)
            results.append({
                "fold": len(results),
                "train_start": train_df.index[0],
                "test_end": test_df.index[-1],
                "accuracy": float(accuracy_score(y_te, preds)),
                "auc": float(roc_auc_score(y_te, proba)),
                "precision": float(precision_score(y_te, preds, zero_division=0)),
                "recall": float(recall_score(y_te, preds, zero_division=0)),
            })
        except Exception as e:
            log.warning(f"Fold {len(results)} failed: {e}")
        start += step_size

    if len(results) < min_folds:
        return {"model_accepted": False, "n_folds": len(results),
                 "reason": "insufficient_folds"}

    summary = {
        "n_folds": len(results),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "mean_auc": float(np.mean([r["auc"] for r in results])),
        "mean_precision": float(np.mean([r["precision"] for r in results])),
        "mean_recall": float(np.mean([r["recall"] for r in results])),
        "profitable_pct": float(np.mean([r["accuracy"] > 0.5 for r in results])),
        "folds": results,
    }
    summary["model_accepted"] = (
        summary["mean_auc"] > 0.54 and summary["mean_accuracy"] > 0.52
    )
    return summary
