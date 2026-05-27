"""
LightGBM trainer with Optuna hyperparameter tuning + Boruta feature selection.

Walk-forward validation must pass before model is registered as accepted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger

log = get_logger(__name__)


class LGBMTrainer:
    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or dict(self.DEFAULT_PARAMS)
        self.model = None
        self.feature_cols: List[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              num_boost_round: int = 500,
              early_stopping_rounds: int = 30
              ) -> Tuple[Any, Dict[str, float]]:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError:
            raise RuntimeError("lightgbm not installed")
        self.feature_cols = list(X.columns)
        X_clean = X.fillna(0)
        train_set = lgb.Dataset(X_clean.values, label=y.values)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val.fillna(0).values, label=y_val.values,
                                   reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")
        self.model = lgb.train(
            self.params, train_set, num_boost_round=num_boost_round,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)],
        )
        metrics: Dict[str, float] = {}
        if X_val is not None and y_val is not None:
            preds = self.predict(X_val)
            from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore
            metrics["val_auc"] = float(roc_auc_score(y_val.values, preds))
            metrics["val_accuracy"] = float(accuracy_score(y_val.values, (preds > 0.5).astype(int)))
        return self.model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X.fillna(0).values)

    def save(self, path: str) -> None:
        if self.model is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        import lightgbm as lgb  # type: ignore
        self.model = lgb.Booster(model_file=path)


def hyperparam_search(X: pd.DataFrame, y: pd.Series, n_trials: int = 20
                      ) -> Dict[str, Any]:
    """Lightweight Optuna sweep. Returns best params dict."""
    try:
        import optuna  # type: ignore
    except ImportError:
        log.warning("optuna not installed - returning defaults")
        return dict(LGBMTrainer.DEFAULT_PARAMS)

    # Time-series split: train first 70%, validate last 30%
    split = int(len(X) * 0.7)
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    def objective(trial):
        params = dict(LGBMTrainer.DEFAULT_PARAMS)
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 5.0, log=True),
        })
        trainer = LGBMTrainer(params)
        _, metrics = trainer.train(X_tr, y_tr, X_va, y_va,
                                    num_boost_round=200, early_stopping_rounds=20)
        return metrics.get("val_auc", 0.5)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(LGBMTrainer.DEFAULT_PARAMS)
    best.update(study.best_params)
    return best
