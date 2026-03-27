# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
EMA Crossover Trading Strategy

This strategy uses Exponential Moving Average crossovers for signals.
Similar to MA Crossover but more responsive to recent price changes.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class EMAcrossoverStrategy(BaseStrategy):
    """
    EMA Crossover trading strategy.

    Uses exponential moving averages which give more weight to recent prices.
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        params = config.parameters or {}
        self.fast_period = params.get("fast_period", 12)
        self.slow_period = params.get("slow_period", 26)

        self.price_history = []

        logger.info(
            f"EMA Crossover Strategy initialized: "
            f"fast={self.fast_period}, slow={self.slow_period}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute EMA crossover from OHLCV data dict."""
        close_price = data.get("close", 0.0)
        self.price_history.append(close_price)

        max_history = self.slow_period * 3
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]

        if len(self.price_history) < self.slow_period:
            return {
                "close": close_price,
                "fast_ema": None,
                "slow_ema": None,
                "crossover": None,
                "trend": "NEUTRAL",
            }

        series = pd.Series(self.price_history)
        fast_ema = series.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=self.slow_period, adjust=False).mean()

        current_fast = float(fast_ema.iloc[-1])
        current_slow = float(slow_ema.iloc[-1])
        prev_fast = float(fast_ema.iloc[-2]) if len(fast_ema) > 1 else current_fast
        prev_slow = float(slow_ema.iloc[-2]) if len(slow_ema) > 1 else current_slow

        crossover = None
        if prev_fast <= prev_slow and current_fast > current_slow:
            crossover = "BULLISH"
        elif prev_fast >= prev_slow and current_fast < current_slow:
            crossover = "BEARISH"

        trend = "NEUTRAL"
        if current_fast > current_slow:
            trend = "BULLISH"
        elif current_fast < current_slow:
            trend = "BEARISH"

        return {
            "close": close_price,
            "fast_ema": current_fast,
            "slow_ema": current_slow,
            "prev_fast": prev_fast,
            "prev_slow": prev_slow,
            "crossover": crossover,
            "trend": trend,
            "ema_diff": abs(current_fast - current_slow) / (close_price + 1e-9),
        }

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        crossover = analysis.get("crossover")
        close_price = analysis.get("close", 0.0)
        fast_ema = analysis.get("fast_ema")
        slow_ema = analysis.get("slow_ema")
        ema_diff = analysis.get("ema_diff", 0.0)

        if fast_ema is None or slow_ema is None:
            return None

        if crossover == "BULLISH":
            confidence = 0.80
            if ema_diff < 0.001:
                confidence = min(0.95, confidence + 0.10)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=self.config.symbol,
                price=close_price,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
                metadata={
                    "fast_ema": fast_ema,
                    "slow_ema": slow_ema,
                    "crossover": "BULLISH",
                    "strategy": self.config.name,
                },
            )
        elif crossover == "BEARISH":
            confidence = 0.80
            if ema_diff < 0.001:
                confidence = min(0.95, confidence + 0.10)
            return Signal(
                signal_type=SignalType.SELL,
                symbol=self.config.symbol,
                price=close_price,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
                metadata={
                    "fast_ema": fast_ema,
                    "slow_ema": slow_ema,
                    "crossover": "BEARISH",
                    "strategy": self.config.name,
                },
            )

        # Trend continuation signals
        current_fast = fast_ema
        current_slow = slow_ema
        prev_fast = analysis.get("prev_fast", current_fast)
        prev_slow = analysis.get("prev_slow", current_slow)

        if current_fast > current_slow:
            if current_fast > prev_fast and current_slow > prev_slow:
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=self.config.symbol,
                    price=close_price,
                    timestamp=datetime.now(timezone.utc),
                    confidence=0.60,
                    metadata={"reason": "Strong uptrend continuation", "strategy": self.config.name},
                )
        elif current_fast < current_slow:
            if current_fast < prev_fast and current_slow < prev_slow:
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=self.config.symbol,
                    price=close_price,
                    timestamp=datetime.now(timezone.utc),
                    confidence=0.60,
                    metadata={"reason": "Strong downtrend continuation", "strategy": self.config.name},
                )

        return None
