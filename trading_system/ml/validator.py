"""
Walk-forward validation for trading models.
Implements purged walk-forward cross-validation with LightGBM.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_window: int,
    test_window: int,
    step_size: int,
    purge_gap: int = 60,
) -> Dict:
    """
    Purged walk-forward validation using LightGBM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target column.
    feature_cols : list of str
        Feature column names.
    target_col : str
        Binary target column name.
    train_window : int
        Number of bars in each training window.
    test_window : int
        Number of bars in each test window.
    step_size : int
        Number of bars to step forward between folds.
    purge_gap : int
        Number of bars between training end and test start (embargo).
        Prevents leakage from short-term autocorrelation.

    Returns
    -------
    dict with keys:
        n_folds : int
        mean_accuracy : float
        mean_auc : float
        profitable_pct : float
        model_accepted : bool
        folds : list of fold result dicts
    """
    import lightgbm as lgb

    df = df.copy().reset_index(drop=True)
    n_rows = len(df)

    # Validate columns
    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        return _empty_result()

    if target_col not in df.columns:
        return _empty_result()

    folds = []
    fold_start = 0

    while True:
        train_end = fold_start + train_window
        test_start = train_end + purge_gap
        test_end = test_start + test_window

        if test_end > n_rows:
            break

        # Extract train/test splits
        train_idx = list(range(fold_start, train_end))
        test_idx = list(range(test_start, test_end))

        X_train = df.loc[train_idx, available_features].values.astype(np.float32)
        y_train = df.loc[train_idx, target_col].values.astype(np.float32)
        X_test = df.loc[test_idx, available_features].values.astype(np.float32)
        y_test = df.loc[test_idx, target_col].values.astype(np.float32)

        # Skip fold if labels are all one class
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            fold_start += step_size
            continue

        # Replace NaN with median
        X_train = _fill_nan(X_train)
        X_test = _fill_nan(X_test)

        # Train LightGBM (fast, minimal tuning)
        try:
            n_val = max(int(len(X_train) * 0.2), 1)
            X_tr = X_train[:-n_val]
            y_tr = y_train[:-n_val]
            X_val = X_train[-n_val:]
            y_val = y_train[-n_val:]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            params = {
                "objective": "binary",
                "metric": "auc",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }

            callbacks = [
                lgb.early_stopping(stopping_rounds=10, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=200,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba >= 0.5).astype(int)

            fold_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
            fold_auc = float(roc_auc_score(y_test.astype(int), y_pred_proba))

            folds.append({
                "fold": len(folds) + 1,
                "train_start": fold_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "accuracy": fold_accuracy,
                "auc": fold_auc,
                "profitable": fold_accuracy > 0.50,
            })

        except Exception as e:
            folds.append({
                "fold": len(folds) + 1,
                "train_start": fold_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "error": str(e),
                "accuracy": 0.5,
                "auc": 0.5,
                "profitable": False,
            })

        fold_start += step_size

    if not folds:
        return _empty_result()

    # Aggregate results
    accuracies = [f["accuracy"] for f in folds if "accuracy" in f]
    aucs = [f["auc"] for f in folds if "auc" in f]
    profitable = [f["profitable"] for f in folds if "profitable" in f]

    mean_accuracy = float(np.mean(accuracies)) if accuracies else 0.5
    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    profitable_pct = float(np.mean(profitable)) if profitable else 0.0

    # Acceptance criteria: AUC > 0.54 AND accuracy > 0.52
    model_accepted = bool(mean_auc > 0.54 and mean_accuracy > 0.52)

    return {
        "n_folds": len(folds),
        "mean_accuracy": mean_accuracy,
        "mean_auc": mean_auc,
        "profitable_pct": profitable_pct,
        "model_accepted": model_accepted,
        "folds": folds,
    }


def _fill_nan(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with column medians."""
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            col[mask] = median_val
    return X


def _empty_result() -> Dict:
    return {
        "n_folds": 0,
        "mean_accuracy": 0.5,
        "mean_auc": 0.5,
        "profitable_pct": 0.0,
        "model_accepted": False,
        "folds": [],
    }


def check_no_lookahead(feature_df: pd.DataFrame, target_col: str) -> bool:
    """
    Verify that no feature has lookahead bias by checking correlations
    between features and the TARGET at the SAME timestep.

    In a correctly constructed feature matrix (shift(1) applied), features
    at time t should not strongly predict returns at time t+1.

    We verify by checking if any feature correlates suspiciously with
    contemporaneous target values (which would indicate lookahead).

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix (should already have shift(1) applied).
    target_col : str
        Target column name (in the same DataFrame or as a separate series).

    Returns
    -------
    bool
        True if no significant lookahead is detected (all |corr| < 0.1).
    """
    if target_col not in feature_df.columns:
        return True

    target = feature_df[target_col]
    feature_cols = [c for c in feature_df.columns if c != target_col]

    suspicious = []
    for col in feature_cols:
        try:
            feat = feature_df[col].dropna()
            tgt = target.reindex(feat.index).dropna()
            common = feat.index.intersection(tgt.index)
            if len(common) < 30:
                continue
            corr = abs(float(np.corrcoef(feat.loc[common].values, tgt.loc[common].values)[0, 1]))
            if corr > 0.1:
                suspicious.append((col, corr))
        except Exception:
            continue

    if suspicious:
        import logging
        logger = logging.getLogger(__name__)
        for col, corr in suspicious:
            logger.warning(f"Potential lookahead: {col} corr={corr:.3f}")
        return False

    return True
