"""
Feature engineering utilities shared by all AI/ML models.

Computes technical indicators, returns, volatility and volume features
from a pandas DataFrame with OHLCV columns.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── required OHLCV columns ───────────────────────────────────────────────────
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has OHLCV columns (case-insensitive) and return normalised copy."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return df


# ── individual indicator helpers ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"bb_upper": upper, "bb_mid": mid, "bb_lower": lower, "bb_width": width, "bb_pct_b": pct_b}
    )


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ── main feature builder ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    lag_periods: Optional[list] = None,
    add_target: bool = True,
    target_horizon: int = 1,
) -> pd.DataFrame:
    """
    Build a feature matrix from raw OHLCV data.

    Parameters
    ----------
    df : DataFrame with OHLCV columns
    lag_periods : list of integers — which lag periods to include (default [1,2,3,5])
    add_target : whether to append a target column for classification
    target_horizon : how many candles ahead to look for the target

    Returns
    -------
    DataFrame with feature columns (and optionally a 'target' column).
    Rows with NaN values are dropped.
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3, 5]

    df = validate_ohlcv(df)
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]

    # ── price returns ─────────────────────────────────────────────────────────
    feat["return_1"] = close.pct_change(1)
    feat["return_3"] = close.pct_change(3)
    feat["return_5"] = close.pct_change(5)
    feat["log_return_1"] = np.log(close / close.shift(1))

    # ── moving averages & ratios ──────────────────────────────────────────────
    for period in [5, 10, 20, 50]:
        feat[f"sma_{period}"] = close.rolling(period).mean()
        feat[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    feat["price_to_sma20"] = close / feat["sma_20"].replace(0, np.nan)
    feat["price_to_ema20"] = close / feat["ema_20"].replace(0, np.nan)
    feat["sma5_sma20_ratio"] = feat["sma_5"] / feat["sma_20"].replace(0, np.nan)
    feat["ema5_ema20_ratio"] = feat["ema_5"] / feat["ema_20"].replace(0, np.nan)

    # ── volatility ────────────────────────────────────────────────────────────
    feat["volatility_10"] = feat["return_1"].rolling(10).std()
    feat["volatility_20"] = feat["return_1"].rolling(20).std()
    feat["high_low_ratio"] = (df["high"] - df["low"]) / close.replace(0, np.nan)

    # ── ATR ───────────────────────────────────────────────────────────────────
    feat["atr_14"] = compute_atr(df, 14)
    feat["atr_ratio"] = feat["atr_14"] / close.replace(0, np.nan)

    # ── RSI ───────────────────────────────────────────────────────────────────
    feat["rsi_14"] = compute_rsi(close, 14)
    feat["rsi_7"] = compute_rsi(close, 7)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_df = compute_macd(close)
    feat["macd"] = macd_df["macd"]
    feat["macd_signal"] = macd_df["macd_signal"]
    feat["macd_hist"] = macd_df["macd_hist"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_df = compute_bollinger_bands(close)
    feat["bb_width"] = bb_df["bb_width"]
    feat["bb_pct_b"] = bb_df["bb_pct_b"]

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_df = compute_stochastic(df)
    feat["stoch_k"] = stoch_df["stoch_k"]
    feat["stoch_d"] = stoch_df["stoch_d"]

    # ── Volume features ───────────────────────────────────────────────────────
    feat["volume_sma_20"] = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / feat["volume_sma_20"].replace(0, np.nan)
    feat["obv"] = compute_obv(df)
    feat["obv_ema"] = feat["obv"].ewm(span=20, adjust=False).mean()

    # ── Lag features ──────────────────────────────────────────────────────────
    for lag in lag_periods:
        feat[f"close_lag_{lag}"] = close.shift(lag)
        feat[f"return_lag_{lag}"] = feat["return_1"].shift(lag)
        feat[f"rsi_lag_{lag}"] = feat["rsi_14"].shift(lag)

    # ── Candlestick pattern helpers ───────────────────────────────────────────
    feat["body_size"] = (df["close"] - df["open"]).abs() / close.replace(0, np.nan)
    feat["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / close.replace(0, np.nan)
    feat["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / close.replace(0, np.nan)
    feat["candle_direction"] = np.sign(df["close"] - df["open"])

    # ── Target variable ───────────────────────────────────────────────────────
    if add_target:
        future_return = close.shift(-target_horizon) / close - 1
        # 0 = HOLD, 1 = BUY, 2 = SELL
        threshold = 0.001  # 0.1 % dead-zone
        feat["target"] = 0
        feat.loc[future_return > threshold, "target"] = 1
        feat.loc[future_return < -threshold, "target"] = 2

    feat = feat.dropna()
    logger.debug(f"Feature matrix shape: {feat.shape}")
    return feat


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes 'target')."""
    return [c for c in df.columns if c != "target"]
