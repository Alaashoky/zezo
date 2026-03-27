# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
Mean Reversion Trading Strategy

This strategy trades when price deviates significantly from its mean,
expecting it to revert back to the average.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands.

    Buys when price is below lower band (oversold).
    Sells when price is above upper band (overbought).
    """

    def __init__(
        self,
        config: StrategyConfig,
        *_args,
        period: int = 20,
        std_dev: float = 2.0,
    ):
        super().__init__(config)
        params = config.parameters or {}
        self.period = params.get("period", period)
        self.std_dev = params.get("std_dev", std_dev)
        self.price_history = []
        logger.info(
            f"Mean Reversion Strategy initialized: period={self.period}, std_dev={self.std_dev}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Bollinger Bands for mean reversion from data dict."""
        prices = data.get("prices") or data.get("close")
        if prices is None:
            return {"upper": None, "lower": None, "sma": None, "price": None}

        if isinstance(prices, (int, float)):
            self.price_history.append(float(prices))
            prices = self.price_history
        elif isinstance(prices, list) and prices and isinstance(prices[0], (int, float)):
            self.price_history.extend(prices)
            prices = self.price_history

        series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        if len(series) < self.period:
            return {"upper": None, "lower": None, "sma": None, "price": float(series.iloc[-1]) if not series.empty else None}

        sma = series.rolling(window=self.period).mean()
        std = series.rolling(window=self.period).std()
        upper = sma + std * self.std_dev
        lower = sma - std * self.std_dev
        price = float(series.iloc[-1])

        return {
            "upper": float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
            "lower": float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None,
            "sma": float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
            "price": price,
        }

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        upper = analysis.get("upper")
        lower = analysis.get("lower")
        sma = analysis.get("sma")
        price = analysis.get("price")

        if any(v is None for v in (upper, lower, price)):
            return None

        band_width = upper - lower
        if band_width == 0:
            return None

        distance_from_lower = (price - lower) / band_width
        distance_from_upper = (upper - price) / band_width

        if price <= lower:
            confidence = min(0.9, 0.5 + abs(distance_from_lower) * 0.4)
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=confidence,
                metadata={"reason": "Price below lower band (oversold)", "sma": sma},
            )
        elif price >= upper:
            confidence = min(0.9, 0.5 + abs(distance_from_upper) * 0.4)
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=confidence,
                metadata={"reason": "Price above upper band (overbought)", "sma": sma},
            )
        return None
