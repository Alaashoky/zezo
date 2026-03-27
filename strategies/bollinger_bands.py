# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
Bollinger Bands Trading Strategy

This strategy uses Bollinger Bands for identifying trend strength
and potential reversals.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands trading strategy.

    Combines band touches with band squeeze for signal generation.
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
            f"Bollinger Bands Strategy initialized: period={self.period}, std_dev={self.std_dev}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Bollinger Bands from OHLCV data dict."""
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
        sma = series.rolling(window=self.period).mean()
        std = series.rolling(window=self.period).std()
        upper = sma + std * self.std_dev
        lower = sma - std * self.std_dev
        price = float(series.iloc[-1])
        prev_price = float(series.iloc[-2]) if len(series) > 1 else price
        return {
            "upper": float(upper.iloc[-1]) if not upper.empty and not pd.isna(upper.iloc[-1]) else None,
            "lower": float(lower.iloc[-1]) if not lower.empty and not pd.isna(lower.iloc[-1]) else None,
            "sma": float(sma.iloc[-1]) if not sma.empty and not pd.isna(sma.iloc[-1]) else None,
            "price": price,
            "prev_price": prev_price,
            "prev_upper": float(upper.iloc[-2]) if len(upper) > 1 else None,
            "prev_lower": float(lower.iloc[-2]) if len(lower) > 1 else None,
        }

    def generate_signal(self, data: Any) -> Optional[Signal]:
        """
        Dual-dispatch: accepts either a dict (from analyze()) or a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return self._generate_from_dataframe(data)
        analysis = data
        upper = analysis.get("upper")
        lower = analysis.get("lower")
        price = analysis.get("price")
        prev_price = analysis.get("prev_price", price)
        if any(v is None for v in (upper, lower, price)):
            return None
        if price < lower:
            conf = 0.85 if price > prev_price else 0.70
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=conf,
            )
        if price > upper:
            conf = 0.85 if price < prev_price else 0.70
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=conf,
            )
        return None

    def _generate_from_dataframe(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from OHLCV DataFrame."""
        if len(market_data) < self.period:
            return None
        close = market_data["close"]
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        current_price = float(close.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        prev_price = float(close.iloc[-2])
        if pd.isna(current_upper) or pd.isna(current_lower):
            return None
        if current_price < current_lower:
            conf = 0.85 if current_price > prev_price else 0.70
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                current_price,
                datetime.now(timezone.utc),
                confidence=conf,
            )
        if current_price > current_upper:
            conf = 0.85 if current_price < prev_price else 0.70
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                current_price,
                datetime.now(timezone.utc),
                confidence=conf,
            )
        return None
