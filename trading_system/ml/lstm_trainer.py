"""
PyTorch LSTM with self-attention for sequential signal prediction.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """
    LSTM with multi-head self-attention and linear output.

    Architecture:
        LSTM (input_size → hidden_size, num_layers, dropout)
        → Self-Attention over hidden states
        → Linear (hidden_size → 1)
        → Sigmoid

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        LSTM hidden dimension (default 128).
    num_layers : int
        Number of LSTM layers (default 2).
    dropout : float
        Dropout between LSTM layers (default 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Self-attention: query, key, value projections
        self.attention_q = nn.Linear(hidden_size, hidden_size)
        self.attention_k = nn.Linear(hidden_size, hidden_size)
        self.attention_v = nn.Linear(hidden_size, hidden_size)
        self.attention_scale = hidden_size ** -0.5

        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor of shape (batch, 1) with values in [0, 1].
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Self-attention over all timesteps
        q = self.attention_q(lstm_out)  # (batch, seq_len, hidden_size)
        k = self.attention_k(lstm_out)
        v = self.attention_v(lstm_out)

        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) * self.attention_scale  # (batch, seq, seq)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.bmm(attn_weights, v)  # (batch, seq_len, hidden_size)

        # Residual connection + layer norm
        attn_out = self.layer_norm(attn_out + lstm_out)

        # Use the last timestep representation
        final_hidden = attn_out[:, -1, :]  # (batch, hidden_size)
        final_hidden = self.dropout_layer(final_hidden)

        out = self.output_linear(final_hidden)  # (batch, 1)
        return self.sigmoid(out)


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


class LSTMTrainer:
    """
    Trainer for LSTMModel with early stopping and GPU support.

    Usage
    -----
        trainer = LSTMTrainer()
        model = trainer.train(X_sequences, y, epochs=100)
        probs = trainer.predict(model, X_new)
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_sequences: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 3e-4,
        batch_size: int = 64,
        val_split: float = 0.2,
        patience: int = 10,
    ) -> LSTMModel:
        """
        Train LSTM on sequence data.

        Parameters
        ----------
        X_sequences : np.ndarray of shape (n, seq_len=60, n_features)
        y : np.ndarray of shape (n,) with binary labels [0, 1]
        epochs : int
        lr : float
        batch_size : int
        val_split : float
        patience : int

        Returns
        -------
        LSTMModel (best checkpoint by validation loss)
        """
        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        n_total = len(X)
        n_val = max(int(n_total * val_split), 1)
        n_train = n_total - n_val

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:]
        y_val = y[n_train:]

        input_size = X.shape[2]
        model = LSTMModel(input_size=input_size).to(self.device)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val).unsqueeze(1),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.BCELoss()
        early_stopper = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= max(n_train, 1)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * len(X_batch)
            val_loss /= max(n_val, 1)

            scheduler.step(val_loss)

            if early_stopper(val_loss, model):
                break

        # Restore best checkpoint
        if early_stopper.best_state is not None:
            model.load_state_dict(early_stopper.best_state)

        model.eval()
        return model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, model: LSTMModel, X_sequences: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions.

        Parameters
        ----------
        model : LSTMModel
        X_sequences : np.ndarray of shape (n, seq_len, n_features)

        Returns
        -------
        np.ndarray of shape (n,) with probabilities in [0, 1].
        """
        model.eval()
        X = torch.from_numpy(np.array(X_sequences, dtype=np.float32))
        all_preds = []

        with torch.no_grad():
            # Process in chunks to avoid OOM
            chunk_size = 512
            for i in range(0, len(X), chunk_size):
                chunk = X[i : i + chunk_size].to(self.device)
                preds = model(chunk).cpu().numpy()
                all_preds.append(preds.flatten())

        return np.concatenate(all_preds)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model: LSTMModel, path: str) -> None:
        """
        Save model state dict, architecture config, and optimizer state.

        Parameters
        ----------
        model : LSTMModel
        path : str
            File path (will save as {path}.pt)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "input_size": model.lstm.input_size,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
            },
        }
        torch.save(checkpoint, f"{path}.pt")

    def load_model(self, path: str) -> LSTMModel:
        """
        Load model from checkpoint.

        Parameters
        ----------
        path : str
            Base path (without .pt extension).

        Returns
        -------
        LSTMModel in eval mode.
        """
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        config = checkpoint["config"]
        model = LSTMModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model
