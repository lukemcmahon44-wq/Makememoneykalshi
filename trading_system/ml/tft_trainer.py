"""
Temporal Fusion Transformer trainer using pytorch-forecasting.
Provides quantile forecasts and uncertainty estimation.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# pytorch-forecasting is optional; wrap imports
try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    _TFT_AVAILABLE = True
except ImportError:
    _TFT_AVAILABLE = False
    TimeSeriesDataSet = None  # type: ignore
    TemporalFusionTransformer = None  # type: ignore


from trading_system.ml.features import ALL_FEATURES


class TFTTrainer:
    """
    Temporal Fusion Transformer for multi-horizon probabilistic forecasting.

    Configuration
    -------------
    max_encoder_length : int
        Number of past timesteps to encode (context window). Default 96.
    max_prediction_length : int
        Number of future timesteps to predict. Default 12.
    hidden_size : int
        TFT hidden layer size. Default 64.
    attention_heads : int
        Number of multi-head attention heads. Default 4.
    """

    def __init__(
        self,
        max_encoder_length: int = 96,
        max_prediction_length: int = 12,
        hidden_size: int = 64,
        attention_heads: int = 4,
        quantiles: Optional[List[float]] = None,
    ) -> None:
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self._dataset_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        group_col: str = "group_id",
        time_col: str = "time_idx",
    ) -> Any:
        """
        Prepare a TimeSeriesDataSet from a feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ALL_FEATURES columns plus target_col, group_col, time_col.
        target_col : str
        group_col : str
        time_col : str

        Returns
        -------
        TimeSeriesDataSet
        """
        if not _TFT_AVAILABLE:
            raise ImportError(
                "pytorch-forecasting is not installed. "
                "Install with: pip install pytorch-forecasting pytorch-lightning"
            )

        df = df.copy()

        # Ensure required columns
        if group_col not in df.columns:
            df[group_col] = "asset"
        if time_col not in df.columns:
            df[time_col] = np.arange(len(df))
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Select feature columns that exist
        available_features = [f for f in ALL_FEATURES if f in df.columns]

        # Determine static and time-varying features
        calendar_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                             "minutes_to_close_norm", "days_to_options_expiry",
                             "is_fomc_day", "is_earnings_week"]
        static_features = [group_col]
        time_varying_known = [f for f in calendar_features if f in available_features]
        time_varying_unknown = [f for f in available_features if f not in calendar_features]

        # Fill NaN with 0
        for col in available_features + [target_col]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        dataset = TimeSeriesDataSet(
            df,
            time_idx=time_col,
            target=target_col,
            group_ids=[group_col],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=[group_col],
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=GroupNormalizer(groups=[group_col]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        self._dataset_params = dataset.get_parameters()
        return dataset

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, dataset: Any) -> Any:
        """
        Build TemporalFusionTransformer from a prepared dataset.

        Parameters
        ----------
        dataset : TimeSeriesDataSet

        Returns
        -------
        TemporalFusionTransformer
        """
        if not _TFT_AVAILABLE:
            raise ImportError("pytorch-forecasting is not installed.")

        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=1e-3,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_heads,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=len(self.quantiles),
            loss=QuantileLoss(quantiles=self.quantiles),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Any,
        max_epochs: int = 50,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> Any:
        """
        Train the TFT model.

        Parameters
        ----------
        train_dataset : TimeSeriesDataSet
        val_dataset : TimeSeriesDataSet
        max_epochs : int
        batch_size : int
        num_workers : int

        Returns
        -------
        Trained TemporalFusionTransformer model.
        """
        if not _TFT_AVAILABLE:
            raise ImportError("pytorch-forecasting is not installed.")

        try:
            from torch.utils.data import DataLoader as TorchDataLoader
            train_loader = train_dataset.to_dataloader(
                train=True, batch_size=batch_size, num_workers=num_workers
            )
            val_loader = val_dataset.to_dataloader(
                train=False, batch_size=batch_size, num_workers=num_workers
            )

            model = self.build_model(train_dataset)

            early_stop = pl.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=5,
                verbose=False,
                mode="min",
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                gradient_clip_val=0.1,
                callbacks=[early_stop],
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            return model

        except Exception as e:
            raise RuntimeError(f"TFT training failed: {e}") from e

    # ------------------------------------------------------------------
    # Prediction / Signal
    # ------------------------------------------------------------------

    def predict_signal(
        self,
        tft: Any,
        recent_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate trading signal from TFT predictions.

        Returns
        -------
        dict with keys:
            signal : float in [-1, +1]
                tanh(q50 / ((q90 - q10) / 2))
            point_forecast : float (q50 median forecast)
            uncertainty : float (q90 - q10 inter-quantile range)
            confidence : float in [0, 1]
            quantiles : dict {quantile: forecast_value}
        """
        if not _TFT_AVAILABLE:
            return {
                "signal": 0.0,
                "point_forecast": 0.0,
                "uncertainty": 1.0,
                "confidence": 0.0,
                "quantiles": {str(q): 0.0 for q in self.quantiles},
            }

        try:
            tft.eval()
            raw_predictions = tft.predict(recent_data, mode="quantiles", return_x=False)

            if hasattr(raw_predictions, "numpy"):
                preds = raw_predictions.numpy()
            else:
                preds = np.array(raw_predictions)

            # preds shape: (n_samples, prediction_horizon, n_quantiles)
            # Take the first prediction horizon step, average over samples
            if preds.ndim == 3:
                step_preds = preds[:, 0, :]  # (n_samples, n_quantiles)
                mean_preds = step_preds.mean(axis=0)  # (n_quantiles,)
            elif preds.ndim == 2:
                mean_preds = preds.mean(axis=0)
            else:
                mean_preds = preds

            quantile_vals = {str(q): float(mean_preds[i]) for i, q in enumerate(self.quantiles)}

            q10 = quantile_vals.get("0.1", mean_preds[0] if len(mean_preds) > 0 else 0.0)
            q50 = quantile_vals.get("0.5", mean_preds[len(mean_preds) // 2])
            q90 = quantile_vals.get("0.9", mean_preds[-1] if len(mean_preds) > 0 else 0.0)

            iqr = float(q90) - float(q10)
            if iqr > 1e-8:
                signal = float(np.tanh(float(q50) / (iqr / 2.0)))
            else:
                signal = float(np.tanh(float(q50) * 10.0))

            # Confidence: inverse of normalized uncertainty
            uncertainty = iqr
            confidence = float(1.0 / (1.0 + iqr))

            return {
                "signal": float(np.clip(signal, -1.0, 1.0)),
                "point_forecast": float(q50),
                "uncertainty": float(uncertainty),
                "confidence": float(confidence),
                "quantiles": quantile_vals,
            }

        except Exception as e:
            return {
                "signal": 0.0,
                "point_forecast": 0.0,
                "uncertainty": 1.0,
                "confidence": 0.0,
                "quantiles": {str(q): 0.0 for q in self.quantiles},
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model: Any, path: str) -> None:
        """Save TFT model checkpoint."""
        if not _TFT_AVAILABLE:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        import torch
        checkpoint = {
            "state_dict": model.state_dict(),
            "hparams": model.hparams if hasattr(model, "hparams") else {},
            "dataset_params": self._dataset_params,
            "config": {
                "max_encoder_length": self.max_encoder_length,
                "max_prediction_length": self.max_prediction_length,
                "hidden_size": self.hidden_size,
                "attention_heads": self.attention_heads,
                "quantiles": self.quantiles,
            },
        }
        torch.save(checkpoint, f"{path}.tft.pt")

    def load_model(self, path: str, dataset: Optional[Any] = None) -> Any:
        """
        Load TFT model from checkpoint.

        Parameters
        ----------
        path : str
            Base path (without .tft.pt extension).
        dataset : TimeSeriesDataSet or None
            If provided, used to reconstruct model architecture.

        Returns
        -------
        TemporalFusionTransformer in eval mode.
        """
        if not _TFT_AVAILABLE:
            raise ImportError("pytorch-forecasting is not installed.")

        import torch
        checkpoint = torch.load(f"{path}.tft.pt", map_location="cpu")

        config = checkpoint.get("config", {})
        self.max_encoder_length = config.get("max_encoder_length", self.max_encoder_length)
        self.max_prediction_length = config.get("max_prediction_length", self.max_prediction_length)
        self.hidden_size = config.get("hidden_size", self.hidden_size)
        self.attention_heads = config.get("attention_heads", self.attention_heads)
        self.quantiles = config.get("quantiles", self.quantiles)
        self._dataset_params = checkpoint.get("dataset_params")

        if dataset is not None:
            model = self.build_model(dataset)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Reconstruct from stored dataset params
            if self._dataset_params is not None:
                dataset = TimeSeriesDataSet.from_parameters(self._dataset_params, pd.DataFrame())
                model = self.build_model(dataset)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                raise ValueError("Cannot load model without dataset or stored dataset_params")

        model.eval()
        return model
