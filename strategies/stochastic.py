# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
Stochastic Oscillator Trading Strategy

This strategy uses the Stochastic Oscillator to identify overbought/oversold conditions.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator trading strategy.

    Uses %K and %D lines to identify momentum and reversals.
    """

    def __init__(
        self,
        config: StrategyConfig,
        *_args,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20,
        overbought: float = 80,
    ):
        super().__init__(config)
        params = config.parameters or {}
        self.k_period = params.get("k_period", k_period)
        self.d_period = params.get("d_period", d_period)
        self.oversold = params.get("oversold", oversold)
        self.overbought = params.get("overbought", overbought)
        self.ohlcv_history = []
        logger.info(
            f"Stochastic Strategy initialized: k={self.k_period}, d={self.d_period}, "
            f"oversold={self.oversold}, overbought={self.overbought}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Stochastic from data dict."""
        prices = data.get("prices")
        if prices and isinstance(prices[0], dict):
            self.ohlcv_history.extend(prices)
        else:
            close = data.get("close", 0.0)
            high = data.get("high", close)
            low = data.get("low", close)
            self.ohlcv_history.append({"high": high, "low": low, "close": close})

        keep = (self.k_period + self.d_period) * 3
        if len(self.ohlcv_history) > keep:
            self.ohlcv_history = self.ohlcv_history[-keep:]

        min_length = self.k_period + self.d_period
        if len(self.ohlcv_history) < min_length:
            return {"k_percent": None, "d_percent": None, "price": self.ohlcv_history[-1]["close"] if self.ohlcv_history else None}

        df = pd.DataFrame(self.ohlcv_history)
        k_percent, d_percent = self._calculate_stochastic(df)
        current_k = float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else None
        current_d = float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else None
        prev_k = float(k_percent.iloc[-2]) if len(k_percent) > 1 and not pd.isna(k_percent.iloc[-2]) else current_k
        prev_d = float(d_percent.iloc[-2]) if len(d_percent) > 1 and not pd.isna(d_percent.iloc[-2]) else current_d

        return {
            "k_percent": current_k,
            "d_percent": current_d,
            "prev_k": prev_k,
            "prev_d": prev_d,
            "price": float(df["close"].iloc[-1]),
        }

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        current_k = analysis.get("k_percent")
        current_d = analysis.get("d_percent")
        prev_k = analysis.get("prev_k", current_k)
        prev_d = analysis.get("prev_d", current_d)
        price = analysis.get("price", 0.0)

        if current_k is None or current_d is None:
            return None

        # Bullish crossover in oversold region
        if current_k < self.oversold and prev_k <= prev_d and current_k > current_d:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.85,
                metadata={"reason": f"Bullish crossover in oversold: %K={current_k:.1f}", "k": current_k, "d": current_d},
            )

        # Rising from oversold
        if current_k < self.oversold and current_k > prev_k:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.70,
                metadata={"reason": f"Rising from oversold: %K={current_k:.1f}", "k": current_k, "d": current_d},
            )

        # Exiting oversold zone
        if prev_k < self.oversold and current_k > self.oversold:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.75,
                metadata={"reason": f"Exiting oversold zone: %K={current_k:.1f}", "k": current_k, "d": current_d},
            )

        # Bearish crossover in overbought region
        if current_k > self.overbought and prev_k >= prev_d and current_k < current_d:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.85,
                metadata={"reason": f"Bearish crossover in overbought: %K={current_k:.1f}", "k": current_k, "d": current_d},
            )

        # Falling from overbought
        if current_k > self.overbought and current_k < prev_k:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.70,
                metadata={"reason": f"Falling from overbought: %K={current_k:.1f}", "k": current_k, "d": current_d},
            )

        # Exiting overbought zone
        if prev_k > self.overbought and current_k < self.overbought:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.75,
                metadata={"reason": f"Exiting overbought zone: %K={current_k:.1f}", "k": current_k, "d": current_d},
            )

        return None

    def _calculate_stochastic(self, df: pd.DataFrame):
        lowest_low = df["low"].rolling(window=self.k_period).min()
        highest_high = df["high"].rolling(window=self.k_period).max()
        k_percent = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low + 1e-9))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        return k_percent, d_percent
