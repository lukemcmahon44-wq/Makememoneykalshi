"""
Real-time inference engine: loads all trained models at startup and
provides fast (<50ms) predictions with graceful model fallback.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np

from trading_system.core.logger import get_logger

log = get_logger(__name__)

# Minimum OOS accuracy required to enable a model
_OOS_ACCURACY_GATE = 0.52


class ModelPredictor:
    """
    Real-time inference engine for LightGBM, LSTM, and TFT models.

    All models are loaded into memory at startup to eliminate disk I/O
    during inference. Falls back gracefully if any model is missing or
    fails accuracy gate.

    Parameters
    ----------
    model_dir : str
        Directory containing saved model files.
    oos_accuracy_threshold : float
        Minimum OOS accuracy required to enable model (default 0.52).
    """

    def __init__(
        self,
        model_dir: str = "./models",
        oos_accuracy_threshold: float = _OOS_ACCURACY_GATE,
    ) -> None:
        self.model_dir = model_dir
        self.oos_accuracy_threshold = oos_accuracy_threshold

        # Loaded model objects (in-memory, no disk reads during inference)
        self._lgbm_model: Optional[Any] = None
        self._lstm_model: Optional[Any] = None
        self._tft_model: Optional[Any] = None
        self._meta_learner: Optional[Any] = None

        # Trainer instances (hold feature selection state, etc.)
        self._lgbm_trainer: Optional[Any] = None
        self._lstm_trainer: Optional[Any] = None
        self._tft_trainer: Optional[Any] = None

        # OOS accuracy metrics for gate checks
        self._oos_accuracy: Dict[str, float] = {
            "lgbm": 0.0,
            "lstm": 0.0,
            "tft": 0.0,
        }

        # Sequence cache for LSTM (stores recent feature vectors)
        self._sequence_cache: Dict[str, list] = {}
        self._seq_len = 60  # LSTM sequence length

    # ------------------------------------------------------------------
    # Startup loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """
        Load all available models from model_dir into memory.
        Missing models are silently skipped.
        """
        self._load_lgbm()
        self._load_lstm()
        self._load_tft()
        self._load_meta()
        log.info(
            "ModelPredictor loaded",
            lgbm_available=self._lgbm_model is not None,
            lstm_available=self._lstm_model is not None,
            tft_available=self._tft_model is not None,
            meta_available=self._meta_learner is not None,
        )

    def _load_lgbm(self) -> None:
        try:
            from trading_system.ml.lgbm_trainer import LGBMTrainer
            path = os.path.join(self.model_dir, "lgbm")
            if os.path.exists(f"{path}.lgb"):
                trainer = LGBMTrainer()
                self._lgbm_model = trainer.load_model(path)
                self._lgbm_trainer = trainer
                # Load stored OOS accuracy
                meta_acc = self._read_oos_accuracy("lgbm")
                self._oos_accuracy["lgbm"] = meta_acc
                log.info("LightGBM model loaded", oos_accuracy=meta_acc)
        except Exception as e:
            log.warning("Failed to load LightGBM model", error=str(e))

    def _load_lstm(self) -> None:
        try:
            from trading_system.ml.lstm_trainer import LSTMTrainer
            path = os.path.join(self.model_dir, "lstm")
            if os.path.exists(f"{path}.pt"):
                trainer = LSTMTrainer()
                self._lstm_model = trainer.load_model(path)
                self._lstm_trainer = trainer
                meta_acc = self._read_oos_accuracy("lstm")
                self._oos_accuracy["lstm"] = meta_acc
                log.info("LSTM model loaded", oos_accuracy=meta_acc)
        except Exception as e:
            log.warning("Failed to load LSTM model", error=str(e))

    def _load_tft(self) -> None:
        try:
            from trading_system.ml.tft_trainer import TFTTrainer
            path = os.path.join(self.model_dir, "tft")
            if os.path.exists(f"{path}.tft.pt"):
                trainer = TFTTrainer()
                # TFT needs dataset to reconstruct; skip if no dataset params
                self._tft_trainer = trainer
                # Store path for lazy loading
                self._tft_path = path
                meta_acc = self._read_oos_accuracy("tft")
                self._oos_accuracy["tft"] = meta_acc
                log.info("TFT trainer prepared", oos_accuracy=meta_acc)
        except Exception as e:
            log.warning("Failed to load TFT model", error=str(e))

    def _load_meta(self) -> None:
        try:
            from trading_system.ml.meta_learner import MetaLearner
            path = os.path.join(self.model_dir, "meta")
            meta_file = f"{path}.meta.pkl"
            if os.path.exists(meta_file):
                learner = MetaLearner()
                learner.load(path)
                self._meta_learner = learner
                log.info("Meta-learner loaded")
        except Exception as e:
            log.warning("Failed to load meta-learner", error=str(e))

    def _read_oos_accuracy(self, model_name: str) -> float:
        """Read stored OOS accuracy from model directory."""
        try:
            acc_file = os.path.join(self.model_dir, f"{model_name}_oos_accuracy.txt")
            if os.path.exists(acc_file):
                with open(acc_file) as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return 0.0

    def save_oos_accuracy(self, model_name: str, accuracy: float) -> None:
        """Persist OOS accuracy to disk and update in-memory value."""
        self._oos_accuracy[model_name] = accuracy
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            acc_file = os.path.join(self.model_dir, f"{model_name}_oos_accuracy.txt")
            with open(acc_file, "w") as f:
                f.write(str(accuracy))
        except Exception as e:
            log.warning("Failed to save OOS accuracy", model=model_name, error=str(e))

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model passes the OOS accuracy gate.

        Parameters
        ----------
        model_name : str
            One of 'lgbm', 'lstm', 'tft'.

        Returns
        -------
        bool
            True if OOS accuracy >= threshold (default 52%).
        """
        acc = self._oos_accuracy.get(model_name, 0.0)
        model_loaded = {
            "lgbm": self._lgbm_model is not None,
            "lstm": self._lstm_model is not None,
            "tft": self._tft_model is not None,
        }.get(model_name, False)
        return model_loaded and acc >= self.oos_accuracy_threshold

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        asset: str,
        feature_vector: np.ndarray,
        recent_candles: Any = None,
    ) -> Dict[str, Any]:
        """
        Generate trading signals from all available models.

        Parameters
        ----------
        asset : str
            Asset ticker (used for sequence caching).
        feature_vector : np.ndarray of shape (n_features,)
            Current feature vector (output of compute_features for latest bar).
        recent_candles : pd.DataFrame or None
            Recent OHLCV data for TFT context window.

        Returns
        -------
        dict with keys:
            lgbm_signal : float or None
            lstm_signal : float or None
            tft_signal : float or None
            meta_signal : float or None
            active_models : list of str
        """
        result: Dict[str, Any] = {
            "lgbm_signal": None,
            "lstm_signal": None,
            "tft_signal": None,
            "meta_signal": None,
            "active_models": [],
        }

        feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # --- LightGBM ---
        lgbm_prob = None
        if self._lgbm_model is not None and self.is_model_enabled("lgbm"):
            try:
                preds = self._lgbm_trainer.predict(self._lgbm_model, feature_vector)
                lgbm_prob = float(preds[0])
                lgbm_signal = float(np.tanh((lgbm_prob - 0.5) * 4.0))
                result["lgbm_signal"] = lgbm_signal
                result["active_models"].append("lgbm")
            except Exception as e:
                log.debug("LightGBM prediction failed", error=str(e))

        # --- LSTM ---
        lstm_prob = None
        if self._lstm_model is not None and self.is_model_enabled("lstm"):
            try:
                # Update sequence cache
                if asset not in self._sequence_cache:
                    self._sequence_cache[asset] = []
                self._sequence_cache[asset].append(feature_vector.flatten())
                # Keep only last seq_len bars
                if len(self._sequence_cache[asset]) > self._seq_len:
                    self._sequence_cache[asset] = self._sequence_cache[asset][-self._seq_len:]

                cache = self._sequence_cache[asset]
                if len(cache) >= self._seq_len:
                    seq = np.array(cache[-self._seq_len:], dtype=np.float32)
                    seq = seq.reshape(1, self._seq_len, -1)
                    preds = self._lstm_trainer.predict(self._lstm_model, seq)
                    lstm_prob = float(preds[0])
                    lstm_signal = float(np.tanh((lstm_prob - 0.5) * 4.0))
                    result["lstm_signal"] = lstm_signal
                    result["active_models"].append("lstm")
            except Exception as e:
                log.debug("LSTM prediction failed", error=str(e))

        # --- TFT ---
        tft_prob = None
        if self._tft_model is not None and self.is_model_enabled("tft") and recent_candles is not None:
            try:
                tft_result = self._tft_trainer.predict_signal(self._tft_model, recent_candles)
                tft_signal = tft_result.get("signal", 0.0)
                tft_prob = (tft_signal + 1.0) / 2.0  # convert [-1,1] to [0,1] for meta
                result["tft_signal"] = tft_signal
                result["active_models"].append("tft")
            except Exception as e:
                log.debug("TFT prediction failed", error=str(e))

        # --- Meta-ensemble ---
        if (
            self._meta_learner is not None
            and lgbm_prob is not None
            and lstm_prob is not None
            and tft_prob is not None
        ):
            try:
                meta_prob = self._meta_learner.predict(lgbm_prob, lstm_prob, tft_prob)
                meta_signal = self._meta_learner.get_signal(meta_prob)
                result["meta_signal"] = meta_signal
                result["active_models"].append("meta")
            except Exception as e:
                log.debug("Meta-learner prediction failed", error=str(e))

        # If meta not available, fall back to available model average
        if result["meta_signal"] is None:
            available_signals = [
                s for s in [result["lgbm_signal"], result["lstm_signal"], result["tft_signal"]]
                if s is not None
            ]
            if available_signals:
                result["meta_signal"] = float(np.mean(available_signals))

        return result

    # ------------------------------------------------------------------
    # Latency benchmarking
    # ------------------------------------------------------------------

    def benchmark_latency(self, n_trials: int = 100) -> float:
        """
        Measure average inference latency.

        Parameters
        ----------
        n_trials : int
            Number of prediction calls to average over.

        Returns
        -------
        float
            Average latency in milliseconds.
        """
        from trading_system.ml.features import ALL_FEATURES
        dummy_features = np.random.randn(len(ALL_FEATURES)).astype(np.float32)

        # Warm up
        for _ in range(5):
            self.predict("BENCH", dummy_features)

        latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()
            self.predict("BENCH", dummy_features)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        avg_ms = float(np.mean(latencies))
        p99_ms = float(np.percentile(latencies, 99))
        log.info(
            "Inference latency benchmark",
            avg_ms=round(avg_ms, 2),
            p99_ms=round(p99_ms, 2),
            n_trials=n_trials,
        )
        return avg_ms
