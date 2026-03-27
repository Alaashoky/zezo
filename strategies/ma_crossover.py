# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
Moving Average Crossover Strategy

A simple trend-following strategy based on moving average crossovers.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Generates signals when:
    - BUY: Fast MA crosses above slow MA
    - SELL: Fast MA crosses below slow MA

    Parameters:
    - fast_period: Fast MA period (default: 10)
    - slow_period: Slow MA period (default: 30)
    - min_confidence: Minimum confidence threshold (default: 0.6)
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        params = config.parameters or {}
        self.fast_period = params.get("fast_period", 10)
        self.slow_period = params.get("slow_period", 30)
        self.min_confidence = params.get("min_confidence", 0.6)

        self.price_history = []
        self.fast_ma = None
        self.slow_ma = None
        self.prev_fast_ma = None
        self.prev_slow_ma = None

        logger.info(
            f"MA Crossover Strategy initialized: "
            f"fast={self.fast_period}, slow={self.slow_period}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        close_price = data.get("close", 0.0)

        self.price_history.append(close_price)

        max_period = max(self.fast_period, self.slow_period)
        if len(self.price_history) > max_period * 2:
            self.price_history = self.price_history[-max_period * 2:]

        analysis = {
            "close": close_price,
            "fast_ma": None,
            "slow_ma": None,
            "crossover": None,
            "trend": "NEUTRAL",
        }

        if len(self.price_history) >= self.slow_period:
            fast_ma = self._calculate_ma(self.fast_period)
            slow_ma = self._calculate_ma(self.slow_period)

            analysis["fast_ma"] = fast_ma
            analysis["slow_ma"] = slow_ma

            if self.prev_fast_ma is not None and self.prev_slow_ma is not None:
                if fast_ma > slow_ma and self.prev_fast_ma <= self.prev_slow_ma:
                    analysis["crossover"] = "BULLISH"
                    analysis["trend"] = "BULLISH"
                elif fast_ma < slow_ma and self.prev_fast_ma >= self.prev_slow_ma:
                    analysis["crossover"] = "BEARISH"
                    analysis["trend"] = "BEARISH"
                elif fast_ma > slow_ma:
                    analysis["trend"] = "BULLISH"
                elif fast_ma < slow_ma:
                    analysis["trend"] = "BEARISH"

            self.prev_fast_ma = fast_ma
            self.prev_slow_ma = slow_ma

        return analysis

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        crossover = analysis.get("crossover")
        close_price = analysis.get("close", 0.0)

        if not crossover:
            return None

        fast_ma = analysis.get("fast_ma", 0.0)
        slow_ma = analysis.get("slow_ma", 0.0)

        if slow_ma > 0:
            ma_distance = abs(fast_ma - slow_ma) / slow_ma
            confidence = min(0.5 + ma_distance * 10, 1.0)
        else:
            confidence = 0.5

        if confidence < self.min_confidence:
            return None

        if crossover == "BULLISH":
            signal_type = SignalType.BUY
        elif crossover == "BEARISH":
            signal_type = SignalType.SELL
        else:
            return None

        return Signal(
            signal_type=signal_type,
            symbol=self.config.symbol,
            price=close_price,
            timestamp=datetime.now(timezone.utc),
            confidence=confidence,
            metadata={
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "crossover": crossover,
                "strategy": self.config.name,
            },
        )

    def _calculate_ma(self, period: int) -> float:
        if len(self.price_history) < period:
            return 0.0
        return sum(self.price_history[-period:]) / period
