"""
Feature engineering utilities shared by all AI/ML models.

Computes technical indicators, returns, volatility and volume features
from a pandas DataFrame with OHLCV columns.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

# ── strategy name → feature column suffix mapping ───────────────────────────
# Used by _compute_strategy_signals() and by backtest.py to build the
# strategy_signals dict passed to model.predict().
STRATEGY_NAME_MAP = {
    "MA_Crossover":   "ma_crossover",
    "EMA_Crossover":  "ema_crossover",
    "RSI":            "rsi",
    "MACD":           "macd",
    "Bollinger":      "bollinger",
    "MeanReversion":  "mean_reversion",
    "Breakout":       "breakout",
    "Stochastic":     "stochastic",
    "SMC_ICT":        "smc_ict",
    "ITS8OS":         "its8os",
}

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

def _compute_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward computation of strategy signals for every row in *df*.

    Creates fresh strategy instances, iterates through all rows calling
    each strategy's ``on_bar()`` once per row (O(N * 10)), and returns a
    DataFrame with the following columns:

      strategy_ma_crossover, strategy_ema_crossover, strategy_rsi,
      strategy_macd, strategy_bollinger, strategy_mean_reversion,
      strategy_breakout, strategy_stochastic, strategy_smc_ict,
      strategy_its8os, strategy_consensus, strategy_consensus_confidence,
      strategy_buy_count, strategy_sell_count

    Signal values: 1 = BUY, -1 = SELL, 0 = HOLD / no signal.

    Only called during training (controlled by ``add_strategy_features``).
    """
    from strategies import (
        StrategyConfig, SignalType,
        MovingAverageCrossover, EMAcrossoverStrategy, RSIStrategy,
        MACDStrategy, BollingerBandsStrategy, MeanReversionStrategy,
        BreakoutStrategy, StochasticStrategy, SMCICTStrategy, ITS8OSStrategy,
    )

    symbol = "XAUUSD"
    timeframe = "M15"

    # Map from config name → (feature suffix, strategy instance)
    strategy_instances = {
        "MA_Crossover":  MovingAverageCrossover(StrategyConfig("MA_Crossover",  symbol, timeframe)),
        "EMA_Crossover": EMAcrossoverStrategy(  StrategyConfig("EMA_Crossover", symbol, timeframe)),
        "RSI":           RSIStrategy(           StrategyConfig("RSI",           symbol, timeframe)),
        "MACD":          MACDStrategy(          StrategyConfig("MACD",          symbol, timeframe)),
        "Bollinger":     BollingerBandsStrategy(StrategyConfig("Bollinger",     symbol, timeframe)),
        "MeanReversion": MeanReversionStrategy( StrategyConfig("MeanReversion", symbol, timeframe)),
        "Breakout":      BreakoutStrategy(      StrategyConfig("Breakout",      symbol, timeframe)),
        "Stochastic":    StochasticStrategy(    StrategyConfig("Stochastic",    symbol, timeframe)),
        "SMC_ICT":       SMCICTStrategy(        StrategyConfig("SMC_ICT",       symbol, timeframe)),
        "ITS8OS":        ITS8OSStrategy(        StrategyConfig("ITS8OS",        symbol, timeframe)),
    }
    for s in strategy_instances.values():
        s.start()

    n = len(df)
    # Pre-build bar records (faster than per-row dict construction)
    cols = ["open", "high", "low", "close"]
    sub = df[cols].copy()
    if "volume" in df.columns:
        sub["volume"] = df["volume"].values
    else:
        sub["volume"] = 0.0
    bar_records = sub[["open", "high", "low", "close", "volume"]].astype(float).to_dict("records")

    # Output arrays
    individual = {suf: np.zeros(n, dtype=np.int8) for suf in STRATEGY_NAME_MAP.values()}
    consensus_arr = np.zeros(n, dtype=np.int8)
    conf_arr = np.zeros(n, dtype=np.float32)
    buy_cnt_arr = np.zeros(n, dtype=np.int8)
    sell_cnt_arr = np.zeros(n, dtype=np.int8)

    logger.info("Computing strategy signals walk-forward (this may take a minute) …")

    for i in range(n):
        bar_dict = {
            **bar_records[i],
            "symbol": symbol,
            "prices": bar_records[: i + 1],
        }
        buy_count = 0
        sell_count = 0
        for config_name, strategy in strategy_instances.items():
            feat_key = STRATEGY_NAME_MAP[config_name]
            try:
                signal = strategy.on_bar(bar_dict)
            except Exception:
                signal = None
            if signal is not None:
                if signal.signal_type == SignalType.BUY:
                    individual[feat_key][i] = 1
                    buy_count += 1
                elif signal.signal_type == SignalType.SELL:
                    individual[feat_key][i] = -1
                    sell_count += 1

        buy_cnt_arr[i] = buy_count
        sell_cnt_arr[i] = sell_count

        # Simple majority consensus (threshold 60 %)
        total = buy_count + sell_count
        if total > 0:
            buy_ratio = buy_count / total
            sell_ratio = sell_count / total
            if buy_ratio >= 0.6:
                consensus_arr[i] = 1
                conf_arr[i] = buy_ratio
            elif sell_ratio >= 0.6:
                consensus_arr[i] = -1
                conf_arr[i] = sell_ratio

        if (i + 1) % 10_000 == 0:
            logger.info(f"  … processed {i + 1:,}/{n:,} rows")

    logger.info("Strategy signal computation complete.")

    result = pd.DataFrame(index=df.index)
    for suf in STRATEGY_NAME_MAP.values():
        result[f"strategy_{suf}"] = individual[suf]
    result["strategy_consensus"] = consensus_arr
    result["strategy_consensus_confidence"] = conf_arr
    result["strategy_buy_count"] = buy_cnt_arr
    result["strategy_sell_count"] = sell_cnt_arr
    return result


def build_features(
    df: pd.DataFrame,
    lag_periods: Optional[list] = None,
    add_target: bool = True,
    target_horizon: int = 1,
    add_strategy_features: bool = False,
) -> pd.DataFrame:
    """
    Build a feature matrix from raw OHLCV data.

    Parameters
    ----------
    df : DataFrame with OHLCV columns
    lag_periods : list of integers — which lag periods to include (default [1,2,3,5])
    add_target : whether to append a target column for classification
    target_horizon : how many candles ahead to look for the target
    add_strategy_features : when True, pre-compute signals from all 10 strategies
        via walk-forward simulation and add them as extra feature columns.
        Only used during training; slower but improves model quality.

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

    # ── Strategy signal features (optional, training only) ────────────────────
    if add_strategy_features:
        try:
            sig_df = _compute_strategy_signals(df)
            for col in sig_df.columns:
                feat[col] = sig_df[col].values
        except Exception as e:
            logger.warning(f"Strategy feature computation failed: {e}")

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
