# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
RSI (Relative Strength Index) Trading Strategy

This strategy uses RSI to identify overbought and oversold conditions.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI-based trading strategy.

    Buys when RSI is oversold (below lower threshold).
    Sells when RSI is overbought (above upper threshold).
    """

    def __init__(
        self,
        config: StrategyConfig,
        *_args,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
    ):
        super().__init__(config)
        params = config.parameters or {}
        self.period = params.get("period", period)
        self.oversold = params.get("oversold", oversold)
        self.overbought = params.get("overbought", overbought)
        self.price_history = []
        logger.info(
            f"RSI Strategy initialized: period={self.period}, "
            f"oversold={self.oversold}, overbought={self.overbought}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute RSI from OHLCV data dict."""
        prices = data.get("prices") or data.get("close")
        if prices is None:
            return {"rsi": None, "error": "no price data"}

        if isinstance(prices, (int, float)):
            self.price_history.append(float(prices))
            prices = self.price_history
        elif isinstance(prices, list):
            self.price_history.extend(prices if isinstance(prices[0], (int, float)) else [])
            prices = self.price_history if not prices or isinstance(prices[0], (int, float)) else prices

        series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        rsi = self.calculate_rsi(series)
        current = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
        price = float(series.iloc[-1]) if not series.empty else data.get("price", 0.0)
        return {
            "rsi": current,
            "price": price,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def generate_signal(self, data: Any) -> Optional[Signal]:
        """Dual-dispatch: DataFrame → Optional[Signal], dict → Optional[Signal]."""
        if isinstance(data, pd.DataFrame):
            return self._generate_from_dataframe(data)
        analysis = data
        rsi = analysis.get("rsi")
        if rsi is None:
            return None
        price = analysis.get("price", 0.0)
        if rsi < self.oversold:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=min(0.95, 0.5 + (self.oversold - rsi) / self.oversold * 0.4),
            )
        if rsi > self.overbought:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=min(
                    0.95,
                    0.5 + (rsi - self.overbought) / (100 - self.overbought) * 0.4,
                ),
            )
        return None

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _generate_from_dataframe(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from OHLCV DataFrame."""
        if len(market_data) < self.period + 1:
            return None
        close = market_data["close"]
        rsi = self.calculate_rsi(close)
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return None
        price = float(close.iloc[-1])
        if current_rsi < self.oversold:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=min(0.95, 0.5 + (self.oversold - current_rsi) / self.oversold * 0.4),
            )
        if current_rsi > self.overbought:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=min(0.95, 0.5 + (current_rsi - self.overbought) / (100 - self.overbought) * 0.4),
            )
        return None
