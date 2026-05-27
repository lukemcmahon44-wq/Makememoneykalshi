"""
LSTM with attention for short-horizon return prediction.

Skeletal but functional - call .train() with a DataFrame of features and target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger

log = get_logger(__name__)


class LSTMTrainer:
    def __init__(self, seq_len: int = 60, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.feature_cols: list = []
        self.device = "cpu"

    def _build_model(self, n_features: int):
        import torch.nn as nn  # type: ignore
        import torch  # type: ignore

        class LSTMAttn(nn.Module):
            def __init__(self_, n_feat, hidden, layers, drop):
                super().__init__()
                self_.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                                      batch_first=True, dropout=drop if layers > 1 else 0)
                self_.attn = nn.Linear(hidden, 1)
                self_.head = nn.Sequential(
                    nn.Linear(hidden, 32), nn.ReLU(),
                    nn.Dropout(drop), nn.Linear(32, 1), nn.Sigmoid())

            def forward(self_, x):
                out, _ = self_.lstm(x)
                attn_scores = torch.softmax(self_.attn(out).squeeze(-1), dim=1)
                ctx = (out * attn_scores.unsqueeze(-1)).sum(dim=1)
                return self_.head(ctx).squeeze(-1)

        return LSTMAttn(n_features, self.hidden_size, self.num_layers, self.dropout)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        seqs, targets = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len: i])
            targets.append(y[i])
        return np.asarray(seqs, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def train(self, X: pd.DataFrame, y: pd.Series, val_split: float = 0.2,
              epochs: int = 10) -> dict:
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError as e:
            log.error(f"LSTM dependencies missing: {e}")
            return {"trained": False}

        self.feature_cols = list(X.columns)
        Xv = X.fillna(0).values.astype(np.float32)
        self.scaler = StandardScaler().fit(Xv)
        Xv = self.scaler.transform(Xv).astype(np.float32)
        yv = y.values.astype(np.float32)

        split = int(len(Xv) * (1 - val_split))
        X_tr_seq, y_tr_seq = self._make_sequences(Xv[:split], yv[:split])
        X_va_seq, y_va_seq = self._make_sequences(Xv[split:], yv[split:])

        if len(X_tr_seq) < 32 or len(X_va_seq) < 16:
            log.warning(f"LSTM training: too few sequences (train={len(X_tr_seq)})")
            return {"trained": False}

        self.model = self._build_model(Xv.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        X_tr_t = torch.from_numpy(X_tr_seq)
        y_tr_t = torch.from_numpy(y_tr_seq)
        X_va_t = torch.from_numpy(X_va_seq)
        y_va_t = torch.from_numpy(y_va_seq)

        n = len(X_tr_t)
        best_va = float("inf"); patience = 3; bad = 0
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                pv = self.model(X_va_t)
                va_loss = float(loss_fn(pv, y_va_t).item())
            log.info(f"LSTM ep{ep}: val_loss={va_loss:.4f}")
            if va_loss < best_va:
                best_va = va_loss; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        return {"trained": True, "val_loss": best_va}

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            return np.zeros(len(X_seq))
        import torch  # type: ignore
        Xs = self.scaler.transform(X_seq).astype(np.float32)
        if len(Xs) < self.seq_len:
            return np.zeros(1)
        seq = Xs[-self.seq_len:][None, :, :]
        self.model.eval()
        with torch.no_grad():
            p = self.model(torch.from_numpy(seq)).numpy()
        return p

    def save(self, path: str) -> None:
        if self.model is None:
            return
        import torch  # type: ignore
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "seq_len": self.seq_len,
        }, path)
