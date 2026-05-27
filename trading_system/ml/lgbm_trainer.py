"""
LightGBM trainer with Optuna hyperparameter optimization.
Includes feature selection, model persistence, and probability prediction.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional, Tuple

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMTrainer:
    """
    LightGBM binary classifier with Optuna hyperparameter optimization.

    Training flow:
        1. Run 50 Optuna trials to maximize AUC on validation set.
        2. Retrain final model on combined train+val with best hyperparams.
        3. Select features via importance threshold (top 80%).
    """

    def __init__(self, n_trials: int = 50, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params: Optional[dict] = None
        self.feature_names: Optional[list] = None
        self.selected_feature_indices: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> lgb.Booster:
        """
        Train LightGBM with Optuna hyperopt.

        Parameters
        ----------
        X_train, y_train : training features and labels.
        X_val, y_val : validation features and labels (for Optuna objective).

        Returns
        -------
        lgb.Booster
            Trained LightGBM model on combined train+val data with best params.
        """
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # Step 1: Hyperparameter optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        best_auc = study.best_value

        # Step 2: Feature selection on training set with best params
        feature_model = self._build_model(X_train, y_train, X_val, y_val, self.best_params)
        self.selected_feature_indices = self._select_features(feature_model, X_train.shape[1])

        # Step 3: Retrain on train+val with selected features
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        X_combined_sel = X_combined[:, self.selected_feature_indices]

        # Use a 20% split from combined for early stopping reference
        n_total = len(X_combined)
        n_val_new = max(int(n_total * 0.2), 1)
        X_final_train = X_combined_sel[:-n_val_new]
        y_final_train = y_combined[:-n_val_new]
        X_final_val = X_combined_sel[-n_val_new:]
        y_final_val = y_combined[-n_val_new:]

        final_model = self._build_model(
            X_final_train, y_final_train, X_final_val, y_final_val, self.best_params
        )

        return final_model

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Optuna objective function: maximize AUC on validation set."""
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        try:
            model = self._build_model(X_train, y_train, X_val, y_val, params)
            preds = self._predict_proba(model, X_val)
            # Handle binary or probabilistic labels
            if len(np.unique(y_val)) < 2:
                return 0.5
            auc = roc_auc_score(y_val, preds)
            return float(auc)
        except Exception:
            return 0.5

    def _build_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict,
    ) -> lgb.Booster:
        """Build and train a LightGBM model with given params."""
        n_estimators = params.pop("n_estimators", 200)
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "n_jobs": -1,
            "seed": self.random_state,
            **params,
        }
        # Restore n_estimators to params dict
        params["n_estimators"] = n_estimators

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False),
                     lgb.log_evaluation(period=-1)]

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        return model

    def _select_features(self, model: lgb.Booster, n_features: int) -> np.ndarray:
        """
        Select top 80% of features by importance.

        Returns array of selected feature indices.
        """
        importance = model.feature_importance(importance_type="gain")
        if len(importance) == 0 or importance.sum() == 0:
            return np.arange(n_features)

        # Sort by importance descending
        sorted_idx = np.argsort(importance)[::-1]
        cumulative = np.cumsum(importance[sorted_idx])
        total = cumulative[-1]

        # Keep features that account for 80% of total importance
        threshold = total * 0.80
        n_keep = np.searchsorted(cumulative, threshold) + 1
        n_keep = max(n_keep, 1)

        selected = np.sort(sorted_idx[:n_keep])
        return selected

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, model: lgb.Booster, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        model : lgb.Booster
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with probabilities in [0, 1].
        """
        X = np.array(X, dtype=np.float32)
        if self.selected_feature_indices is not None:
            if X.shape[1] > len(self.selected_feature_indices):
                X = X[:, self.selected_feature_indices]
        return self._predict_proba(model, X)

    def _predict_proba(self, model: lgb.Booster, X: np.ndarray) -> np.ndarray:
        """Raw prediction without feature selection filtering."""
        preds = model.predict(X)
        # Ensure shape is (n_samples,)
        if preds.ndim > 1:
            preds = preds[:, 1]
        return np.clip(preds, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model: lgb.Booster, path: str) -> None:
        """
        Save model and metadata to disk.

        Saves:
          - {path}.lgb : LightGBM booster model
          - {path}.meta.pkl : trainer metadata (params, feature indices)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        model.save_model(f"{path}.lgb")
        meta = {
            "best_params": self.best_params,
            "selected_feature_indices": self.selected_feature_indices,
            "feature_names": self.feature_names,
        }
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load_model(self, path: str) -> lgb.Booster:
        """
        Load model and metadata from disk.

        Parameters
        ----------
        path : str
            Base path (without extension) used in save_model.

        Returns
        -------
        lgb.Booster
        """
        model = lgb.Booster(model_file=f"{path}.lgb")
        meta_path = f"{path}.meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.best_params = meta.get("best_params")
            self.selected_feature_indices = meta.get("selected_feature_indices")
            self.feature_names = meta.get("feature_names")
        return model
