# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
MACD (Moving Average Convergence Divergence) Trading Strategy

This strategy uses MACD indicator for trend-following signals.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MACDStrategy(BaseStrategy):
    """
    MACD-based trading strategy.

    Generates signals based on MACD line crossing signal line.
    """

    def __init__(
        self,
        config: StrategyConfig,
        *_args,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        super().__init__(config)
        params = config.parameters or {}
        self.fast_period = params.get("fast_period", fast_period)
        self.slow_period = params.get("slow_period", slow_period)
        self.signal_period = params.get("signal_period", signal_period)
        self.price_history = []
        logger.info(
            f"MACD Strategy initialized: fast={self.fast_period}, "
            f"slow={self.slow_period}, signal={self.signal_period}",
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute MACD from OHLCV data dict."""
        prices = data.get("close") or data.get("prices")
        if prices is None:
            return {"macd": None, "signal_line": None, "histogram": None}

        if isinstance(prices, (int, float)):
            self.price_history.append(float(prices))
            prices = self.price_history
        elif isinstance(prices, list) and prices and isinstance(prices[0], (int, float)):
            self.price_history.extend(prices)
            prices = self.price_history

        series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        macd_line, signal_line, histogram = self.calculate_macd(series)
        return {
            "macd": float(macd_line.iloc[-1]) if not macd_line.empty else None,
            "signal_line": float(signal_line.iloc[-1]) if not signal_line.empty else None,
            "histogram": float(histogram.iloc[-1]) if not histogram.empty else None,
            "prev_macd": float(macd_line.iloc[-2]) if len(macd_line) > 1 else None,
            "prev_signal": float(signal_line.iloc[-2]) if len(signal_line) > 1 else None,
            "price": float(series.iloc[-1]),
        }

    def generate_signal(self, data: Any) -> Optional[Signal]:
        """Dual-dispatch: DataFrame → Optional[Signal], dict → Optional[Signal]."""
        if isinstance(data, pd.DataFrame):
            return self._generate_from_dataframe(data)
        analysis = data
        macd = analysis.get("macd")
        sig = analysis.get("signal_line")
        prev_macd = analysis.get("prev_macd")
        prev_sig = analysis.get("prev_signal")
        price = analysis.get("price", 0.0)
        if any(v is None for v in (macd, sig, prev_macd, prev_sig)):
            return None
        if prev_macd <= prev_sig and macd > sig:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.85 if macd < 0 else 0.75,
            )
        if prev_macd >= prev_sig and macd < sig:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.85 if macd > 0 else 0.75,
            )
        return None

    def calculate_macd(self, prices: pd.Series):
        ema_fast = prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _generate_from_dataframe(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from OHLCV DataFrame."""
        min_length = self.slow_period + self.signal_period
        if len(market_data) < min_length:
            return None
        close = market_data["close"]
        macd_line, signal_line, histogram = self.calculate_macd(close)
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        if pd.isna(current_macd) or pd.isna(current_signal):
            return None
        price = float(close.iloc[-1])
        if prev_macd <= prev_signal and current_macd > current_signal:
            return Signal(
                SignalType.BUY,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.85 if current_macd < 0 else 0.75,
            )
        if prev_macd >= prev_signal and current_macd < current_signal:
            return Signal(
                SignalType.SELL,
                self.config.symbol,
                price,
                datetime.now(timezone.utc),
                confidence=0.85 if current_macd > 0 else 0.75,
            )
        return None
