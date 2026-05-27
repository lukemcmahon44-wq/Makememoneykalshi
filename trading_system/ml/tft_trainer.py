"""
Temporal Fusion Transformer (TFT) trainer using pytorch-forecasting.

Heavy dependency footprint - guarded so the rest of the system runs without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger

log = get_logger(__name__)


class TFTTrainer:
    def __init__(self, max_encoder_length: int = 96,
                 max_prediction_length: int = 12,
                 hidden_size: int = 64, attention_head_size: int = 4,
                 dropout: float = 0.15, hidden_continuous_size: int = 32,
                 learning_rate: float = 3e-4):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.model: Any = None
        self.dataset: Any = None

    def _have_deps(self) -> bool:
        try:
            import pytorch_forecasting  # type: ignore  # noqa: F401
            import lightning  # type: ignore  # noqa: F401
            return True
        except ImportError:
            log.warning("pytorch-forecasting / lightning not installed - TFT disabled")
            return False

    def prepare_dataset(self, df: pd.DataFrame, target_col: str,
                         time_idx_col: str = "time_idx",
                         group_col: str = "asset_id"
                         ) -> Optional[Any]:
        if not self._have_deps():
            return None
        from pytorch_forecasting import TimeSeriesDataSet  # type: ignore

        required = {time_idx_col, group_col, target_col}
        if not required.issubset(df.columns):
            log.error(f"TFT prepare_dataset missing cols: {required - set(df.columns)}")
            return None

        time_varying_unknown = [c for c in df.columns
                                 if c not in {time_idx_col, group_col, target_col}
                                 and pd.api.types.is_numeric_dtype(df[c])]
        ds = TimeSeriesDataSet(
            df,
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[group_col],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=[group_col],
            time_varying_unknown_reals=time_varying_unknown,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        self.dataset = ds
        return ds

    def train(self, df: pd.DataFrame, target_col: str,
              max_epochs: int = 20, val_frac: float = 0.2
              ) -> Dict[str, Any]:
        if not self._have_deps():
            return {"trained": False, "reason": "deps_missing"}
        from pytorch_forecasting import TemporalFusionTransformer  # type: ignore
        from pytorch_forecasting.metrics import QuantileLoss  # type: ignore
        import lightning as pl  # type: ignore

        train_size = int(len(df) * (1 - val_frac))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]

        train_ds = self.prepare_dataset(train_df, target_col)
        if train_ds is None:
            return {"trained": False, "reason": "dataset_failed"}
        from pytorch_forecasting import TimeSeriesDataSet  # type: ignore
        val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df, predict=False)

        train_loader = train_ds.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=50,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator="auto",
            gradient_clip_val=0.1, logger=False, enable_progress_bar=False,
            callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=4)],
        )
        trainer.fit(tft, train_loader, val_loader)
        self.model = tft
        return {"trained": True}

    def predict_signal(self, recent_df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None or self.dataset is None:
            return {"signal": 0.0, "point_forecast": 0.0, "uncertainty": 1.0,
                     "confidence": 0.0}
        try:
            preds = self.model.predict(recent_df, mode="quantiles")
            q10 = float(preds[..., 1][-1])
            q50 = float(preds[..., 3][-1])
            q90 = float(preds[..., 5][-1])
            uncertainty = (q90 - q10) / 2 + 1e-6
            signal = float(np.tanh(q50 / uncertainty))
            confidence = float(1 - min(1, uncertainty / (abs(q50) + 1e-6)))
            return {"signal": signal, "point_forecast": q50,
                     "uncertainty": uncertainty, "confidence": confidence,
                     "q10": q10, "q50": q50, "q90": q90}
        except Exception as e:
            log.warning(f"TFT inference error: {e}")
            return {"signal": 0.0, "point_forecast": 0.0, "uncertainty": 1.0,
                     "confidence": 0.0}

    def save(self, path: str) -> None:
        if self.model is None:
            return
        import torch  # type: ignore
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
