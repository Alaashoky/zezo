# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
Breakout/Momentum Trading Strategy

This strategy identifies and trades breakouts from consolidation periods.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Breakout momentum trading strategy.

    Identifies support/resistance levels and trades breakouts.
    """

    def __init__(
        self,
        config: StrategyConfig,
        *_args,
        lookback_period: int = 20,
        breakout_threshold: float = 0.02,
    ):
        super().__init__(config)
        params = config.parameters or {}
        self.lookback_period = params.get("lookback_period", lookback_period)
        self.breakout_threshold = params.get("breakout_threshold", breakout_threshold)
        self.ohlcv_history = []
        logger.info(
            f"Breakout Strategy initialized: lookback={self.lookback_period}, "
            f"threshold={self.breakout_threshold}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify support/resistance and breakout conditions from data dict."""
        # Accept bar dict with ohlcv or a prices list
        prices = data.get("prices")
        if prices and isinstance(prices[0], dict):
            self.ohlcv_history.extend(prices)
        else:
            # Build OHLCV from scalar values
            close = data.get("close", 0.0)
            high = data.get("high", close)
            low = data.get("low", close)
            volume = data.get("volume", 0)
            self.ohlcv_history.append({"open": close, "high": high, "low": low, "close": close, "volume": volume})

        keep = self.lookback_period * 3
        if len(self.ohlcv_history) > keep:
            self.ohlcv_history = self.ohlcv_history[-keep:]

        if len(self.ohlcv_history) < self.lookback_period:
            return {"support": None, "resistance": None, "current_price": None}

        window = self.ohlcv_history[-self.lookback_period:]
        resistance = max(p["high"] for p in window)
        support = min(p["low"] for p in window)
        current = self.ohlcv_history[-1]
        current_price = current["close"]
        current_high = current["high"]
        current_low = current["low"]
        current_volume = current.get("volume", 0)
        avg_volume = sum(p.get("volume", 0) for p in window) / len(window) if window else 0

        return {
            "support": support,
            "resistance": resistance,
            "current_price": current_price,
            "current_high": current_high,
            "current_low": current_low,
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "range_size": resistance - support,
        }

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        support = analysis.get("support")
        resistance = analysis.get("resistance")
        current_price = analysis.get("current_price")

        if any(v is None for v in (support, resistance, current_price)):
            return None

        current_high = analysis.get("current_high", current_price)
        current_low = analysis.get("current_low", current_price)
        current_volume = analysis.get("current_volume", 0)
        avg_volume = analysis.get("avg_volume", 0)
        range_size = analysis.get("range_size", 0)
        high_volume = current_volume > avg_volume * 1.2 if avg_volume > 0 else False

        breakout_threshold_price = range_size * self.breakout_threshold

        if current_high > resistance:
            breakout_distance = current_high - resistance
            if breakout_distance >= breakout_threshold_price:
                confidence = 0.70
                if high_volume:
                    confidence = min(0.90, confidence + 0.15)
                if current_price > resistance:
                    confidence = min(0.95, confidence + 0.10)
                return Signal(
                    SignalType.BUY,
                    self.config.symbol,
                    current_price,
                    datetime.now(timezone.utc),
                    confidence=confidence,
                    metadata={"reason": "Bullish breakout above resistance", "resistance": resistance, "support": support},
                )

        if current_low < support:
            breakout_distance = support - current_low
            if breakout_distance >= breakout_threshold_price:
                confidence = 0.70
                if high_volume:
                    confidence = min(0.90, confidence + 0.15)
                if current_price < support:
                    confidence = min(0.95, confidence + 0.10)
                return Signal(
                    SignalType.SELL,
                    self.config.symbol,
                    current_price,
                    datetime.now(timezone.utc),
                    confidence=confidence,
                    metadata={"reason": "Bearish breakout below support", "resistance": resistance, "support": support},
                )

        return None
