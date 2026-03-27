"""
Data preprocessing utilities for AI/ML models.

Handles:
- Normalisation / scaling
- Train / validation / test splitting
- Sequence creation for LSTM (sliding window)
- Missing data handling
- Basic data validation
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Prepares raw OHLCV DataFrames for use by training and prediction models.

    Usage
    -----
    processor = DataProcessor(scaler="minmax", test_size=0.15, validation_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split(feature_df)
    X_seq, y_seq = processor.make_sequences(X_train, y_train, sequence_length=60)
    """

    SCALER_MAP = {"minmax": MinMaxScaler, "standard": StandardScaler}

    def __init__(
        self,
        scaler: str = "minmax",
        test_size: float = 0.15,
        validation_size: float = 0.15,
        handle_missing: str = "ffill",
    ):
        if scaler not in self.SCALER_MAP:
            raise ValueError(f"scaler must be one of {list(self.SCALER_MAP)}")
        self.scaler_name = scaler
        self.test_size = test_size
        self.validation_size = validation_size
        self.handle_missing = handle_missing

        self.feature_scaler = self.SCALER_MAP[scaler]()
        self.target_scaler = MinMaxScaler()  # always 0-1 for regression targets
        self._fitted = False

    # ── public API ─────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic sanity checks; returns cleaned DataFrame."""
        if df.empty:
            raise ValueError("Empty DataFrame")
        if df.isnull().values.any():
            if self.handle_missing == "ffill":
                df = df.ffill().bfill()
            else:
                df = df.dropna()
        if len(df) < 100:
            logger.warning(f"DataFrame only has {len(df)} rows — may not be enough for training")
        return df

    def split(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series, pd.Series, pd.Series,
    ]:
        """
        Chronological train / validation / test split.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        df = self.validate(df)
        feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else pd.Series(dtype=float)

        n = len(df)
        n_test = max(1, int(n * self.test_size))
        n_val = max(1, int(n * self.validation_size))
        n_train = n - n_val - n_test

        if n_train < 1:
            raise ValueError("Not enough data for train/val/test split")

        X_train = X.iloc[:n_train]
        X_val = X.iloc[n_train: n_train + n_val]
        X_test = X.iloc[n_train + n_val:]

        y_train = y.iloc[:n_train] if len(y) > 0 else pd.Series(dtype=float)
        y_val = y.iloc[n_train: n_train + n_val] if len(y) > 0 else pd.Series(dtype=float)
        y_test = y.iloc[n_train + n_val:] if len(y) > 0 else pd.Series(dtype=float)

        logger.info(
            f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler on X and return scaled array."""
        scaled = self.feature_scaler.fit_transform(X)
        self._fitted = True
        return scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform X using the already-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first")
        return self.feature_scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse-scale features."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first")
        return self.feature_scaler.inverse_transform(X)

    def make_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sequence_length: int = 60,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert a 2-D feature array into overlapping sequences for LSTM.

        Returns
        -------
        X_seq : shape (n_samples, sequence_length, n_features)
        y_seq : shape (n_samples,) aligned to last step in each window, or None
        """
        if len(X) <= sequence_length:
            raise ValueError(
                f"Need more than {sequence_length} rows to build sequences "
                f"(got {len(X)})"
            )

        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length: i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        logger.debug(f"Sequences shape: X={X_seq.shape}, y={y_seq.shape if y_seq is not None else None}")
        return X_seq, y_seq

    def scale_prices(self, prices: np.ndarray) -> np.ndarray:
        """Scale a 1-D price array using the target scaler."""
        reshaped = prices.reshape(-1, 1)
        return self.target_scaler.fit_transform(reshaped).flatten()

    def inverse_scale_prices(self, prices: np.ndarray) -> np.ndarray:
        """Inverse-scale prices."""
        return self.target_scaler.inverse_transform(prices.reshape(-1, 1)).flatten()
