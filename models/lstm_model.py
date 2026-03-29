"""
LSTM model for learning temporal patterns from OHLCV price data.

Requires: torch>=2.0.0
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── lazy torch import so the rest of the bot works without GPU deps ───────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTMModel will not be functional")


class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    """Internal PyTorch LSTM network."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for LSTMNet")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last time-step
        return self.fc(out)


class LSTMModel:
    """
    LSTM-based model for classifying next-candle direction (BUY / SELL / HOLD)
    and optionally for multi-step price forecasting.

    Parameters
    ----------
    config : LSTMConfig (or dict-compatible object with attributes)
    """

    MODEL_FILE = "lstm_model.pt"

    def __init__(self, config=None):
        if config is None:
            from config.model_config import LSTMConfig
            config = LSTMConfig()
        self.config = config
        self.model: Optional[Any] = None
        self.input_size: Optional[int] = None
        self._is_trained = False
        self.train_losses: list = []
        self.val_losses: list = []

    # ── feature / data helpers ────────────────────────────────────────────────

    def _prepare_data(
        self,
        data: pd.DataFrame,
        add_strategy_features: bool = False,
    ) -> tuple:
        """Build features, scale them, create sequences."""
        from models.feature_engineering import build_features, get_feature_columns
        from models.data_processor import DataProcessor

        feat_df = build_features(data, add_target=True, add_strategy_features=add_strategy_features)
        feature_cols = get_feature_columns(feat_df)
        self._feature_cols = feature_cols
        X_raw = feat_df[feature_cols].values
        y_raw = feat_df["target"].values

        processor = DataProcessor(scaler="minmax")
        X_scaled = processor.fit_transform(pd.DataFrame(X_raw, columns=feature_cols))
        self._processor = processor

        X_seq, y_seq = processor.make_sequences(
            X_scaled, y_raw, sequence_length=self.config.sequence_length
        )
        return X_seq, y_seq

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, data: pd.DataFrame, add_strategy_features: bool = False) -> Dict[str, Any]:
        """
        Train the LSTM on historical OHLCV data.

        Parameters
        ----------
        data : DataFrame with OHLCV columns
        add_strategy_features : when True, include walk-forward strategy signals
            as additional features (improves quality; slower to prepare)

        Returns
        -------
        dict with 'train_loss', 'val_loss', 'epochs_trained'
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required — install with: pip install torch>=2.0.0")

        self._add_strategy_features = add_strategy_features
        logger.info("LSTM: preparing data …")
        X_seq, y_seq = self._prepare_data(data, add_strategy_features=add_strategy_features)

        n = len(X_seq)
        n_val = max(1, int(n * 0.15))
        n_train = n - n_val

        X_tr, X_val = X_seq[:n_train], X_seq[n_train:]
        y_tr, y_val = y_seq[:n_train], y_seq[n_train:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        self.input_size = X_seq.shape[2]
        num_classes = len(np.unique(y_seq))
        self._num_classes = num_classes

        self.model = _LSTMNet(
            input_size=self.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=num_classes,
            dropout=self.config.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()

            avg_train_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)

            if epoch % 10 == 0:
                logger.info(f"LSTM epoch {epoch}/{self.config.epochs} — train: {avg_train_loss:.4f}, val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"LSTM early stopping at epoch {epoch}")
                    break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        self._is_trained = True
        logger.info(f"LSTM training complete — best val loss: {best_val_loss:.4f}")
        return {
            "train_loss": self.train_losses[-1],
            "val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses),
        }

    def predict(self, data: pd.DataFrame, strategy_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict direction for the most recent candle.

        Parameters
        ----------
        data : OHLCV DataFrame (at least ``sequence_length`` rows)
        strategy_signals : optional dict mapping strategy feature column names to
            their current values.  When the model was trained with
            ``add_strategy_features=True``, these values are broadcast across
            every step of the input sequence so the model receives the same
            feature set it was trained on.

        Returns
        -------
        dict with 'signal' (0=HOLD,1=BUY,2=SELL), 'confidence', 'probabilities'
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required")
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        from models.feature_engineering import build_features, get_feature_columns

        feat_df = build_features(data, add_target=False)

        # Resolve the expected feature columns (saved during training)
        feature_cols = getattr(self, "_feature_cols", None) or get_feature_columns(feat_df)

        # Populate strategy feature columns if the model expects them
        strategy_cols = [c for c in feature_cols if c.startswith("strategy_")]
        for col in strategy_cols:
            if col not in feat_df.columns:
                feat_df[col] = strategy_signals.get(col, 0) if strategy_signals else 0

        available = [c for c in feature_cols if c in feat_df.columns]
        X_raw = feat_df[available].values
        X_scaled = self._processor.transform(pd.DataFrame(X_raw, columns=available))

        if len(X_scaled) < self.config.sequence_length:
            raise ValueError(
                f"Need at least {self.config.sequence_length} rows for prediction"
            )

        seq = X_scaled[-self.config.sequence_length:]
        x_t = torch.tensor(seq[np.newaxis], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return {"signal": pred_class, "confidence": confidence, "probabilities": probs.tolist()}

    def forecast_prices(self, data: pd.DataFrame, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-step price forecasting using the trained LSTM.

        Returns
        -------
        dict with 'predicted_prices' (list of length steps) and 'confidence'
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required")
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        n_steps = steps or self.config.forecast_steps
        from models.feature_engineering import build_features, get_feature_columns

        feat_df = build_features(data, add_target=False)
        feature_cols = get_feature_columns(feat_df)
        X_raw = feat_df[feature_cols].values
        X_scaled = self._processor.transform(pd.DataFrame(X_raw, columns=feature_cols))
        close_prices = data["close"].values[-self.config.sequence_length:]

        seq = X_scaled[-self.config.sequence_length:].copy()
        predicted = []

        all_probs = []
        self.model.eval()
        for _ in range(n_steps):
            x_t = torch.tensor(seq[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(x_t)
                probs = torch.softmax(logits, dim=-1).numpy()[0]
            all_probs.append(probs)
            # direction-based forecast: shift last known price
            direction = int(np.argmax(probs))
            last_price = close_prices[-1] if not predicted else predicted[-1]
            # use per-step volatility that grows with forecast horizon
            base_vol = feat_df["volatility_20"].iloc[-1]
            step_vol = base_vol * (1 + len(predicted) * 0.05)
            delta = step_vol * last_price
            if direction == 1:
                next_price = last_price + delta
            elif direction == 2:
                next_price = last_price - delta
            else:
                next_price = last_price
            predicted.append(float(next_price))
            # roll the sequence forward (simple copy of last row)
            seq = np.roll(seq, -1, axis=0)

        # confidence: mean of max-class probability across all steps
        mean_max_prob = float(np.mean([p.max() for p in all_probs]))
        confidence = min(max(mean_max_prob, 0.0), 1.0)
        return {"predicted_prices": predicted, "steps": n_steps, "confidence": confidence}

    def save_model(self, path: str):
        """Save model weights and processor to *path* directory."""
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required")
        if not self._is_trained or self.model is None:
            raise RuntimeError("Nothing to save — model is not trained")
        import joblib

        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, self.MODEL_FILE))
        joblib.dump(self._processor, os.path.join(path, "lstm_processor.pkl"))
        joblib.dump(
            {
                "input_size": self.input_size,
                "num_classes": getattr(self, "_num_classes", 3),
                "config": self.config,
                "feature_cols": getattr(self, "_feature_cols", []),
                "add_strategy_features": getattr(self, "_add_strategy_features", False),
            },
            os.path.join(path, "lstm_meta.pkl"),
        )
        logger.info(f"LSTM model saved to {path}")

    def load_model(self, path: str):
        """Load model weights and processor from *path* directory."""
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required")
        import joblib

        meta = joblib.load(os.path.join(path, "lstm_meta.pkl"))
        self.config = meta["config"]
        self.input_size = meta["input_size"]
        self._num_classes = meta.get("num_classes", 3)
        self._feature_cols = meta.get("feature_cols", [])
        self._add_strategy_features = meta.get("add_strategy_features", False)
        self._processor = joblib.load(os.path.join(path, "lstm_processor.pkl"))

        num_classes = self._num_classes
        self.model = _LSTMNet(
            input_size=self.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=num_classes,
            dropout=self.config.dropout,
        )
        self.model.load_state_dict(torch.load(os.path.join(path, self.MODEL_FILE), map_location="cpu"))
        self.model.eval()
        self._is_trained = True
        logger.info(f"LSTM model loaded from {path}")
