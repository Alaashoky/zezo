# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
HOPEFX Strategy Manager
Multi-strategy system with regime detection and performance tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"


@dataclass
class ManagerSignal:
    """Trading signal used internally by StrategyManager."""

    symbol: str
    action: str  # buy, sell, close
    strength: float  # 0.0 to 1.0
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    timestamp: float = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "strength": self.strength,
            "strategy": self.strategy,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class StrategyManager:
    """
    Central strategy management system.

    Features:
    - Multiple strategy registration
    - Regime-based strategy selection
    - Signal aggregation and deduplication
    - Performance tracking per strategy
    """

    def __init__(self, preload_defaults: bool = False):
        self.strategies: Dict[str, Any] = {}
        if preload_defaults:
            self._initialize_default_strategies()

    def _initialize_default_strategies(self):
        """Initialize default strategies using the full strategy classes."""
        from .base import StrategyConfig
        from .ma_crossover import MovingAverageCrossover
        from .mean_reversion import MeanReversionStrategy
        from .breakout import BreakoutStrategy

        for cls, name, params in [
            (MovingAverageCrossover, "TrendFollowing", {"fast_period": 20, "slow_period": 50}),
            (MeanReversionStrategy, "MeanReversion", {"period": 20, "std_dev": 2.0}),
            (BreakoutStrategy, "Breakout", {"lookback_period": 20}),
        ]:
            cfg = StrategyConfig(name=name, symbol="XAUUSD", timeframe="1h", parameters=params)
            self.register_strategy(cls(cfg))

    def register_strategy(self, strategy: Any):
        """Register a strategy (accepts BaseStrategy or any object with .name)."""
        name = getattr(strategy, "name", None) or getattr(strategy, "config", {})
        if hasattr(name, "name"):
            name = name.name
        self.strategies[name] = strategy
        logger.info(f"Registered strategy: {name}")

    def unregister_strategy(self, name: str) -> bool:
        if name in self.strategies:
            del self.strategies[name]
            return True
        return False

    def start_strategy(self, name: str) -> bool:
        if name not in self.strategies:
            return False
        s = self.strategies[name]
        if hasattr(s, "start"):
            s.start()
        return True

    def stop_strategy(self, name: str) -> bool:
        if name not in self.strategies:
            return False
        s = self.strategies[name]
        if hasattr(s, "stop"):
            s.stop()
        return True

    def start_all(self) -> None:
        for s in self.strategies.values():
            if hasattr(s, "start"):
                s.start()

    def stop_all(self) -> None:
        for s in self.strategies.values():
            if hasattr(s, "stop"):
                s.stop()

    def enable_strategy(self, name: str):
        if name in self.strategies:
            s = self.strategies[name]
            if hasattr(s, "enabled"):
                s.enabled = True
            logger.info(f"Enabled strategy: {name}")

    def disable_strategy(self, name: str):
        if name in self.strategies:
            s = self.strategies[name]
            if hasattr(s, "enabled"):
                s.enabled = False
            logger.info(f"Disabled strategy: {name}")

    def get_strategy_performance(self, name: Optional[str] = None) -> dict:
        if name:
            s = self.strategies.get(name)
            return getattr(s, "performance_metrics", {}) if s else {}
        return {n: getattr(s, "performance_metrics", {}) for n, s in self.strategies.items()}

    @property
    def performance_summary(self) -> dict:
        """Aggregate performance across all registered strategies."""
        from .base import StrategyStatus
        total = len(self.strategies)
        active = sum(
            1 for s in self.strategies.values()
            if getattr(s, "status", None) == StrategyStatus.RUNNING
        )
        total_pnl = 0.0
        for s in self.strategies.values():
            metrics = getattr(s, "performance_metrics", {})
            total_pnl += metrics.get("total_pnl", 0.0)
        return {
            "total_strategies": total,
            "active_strategies": active,
            "total_pnl": total_pnl,
        }

    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update performance for a strategy."""
        if strategy_name in self.strategies:
            s = self.strategies[strategy_name]
            if hasattr(s, "update_performance"):
                pnl = trade_result.get("pnl", 0.0)
                s.update_performance(pnl=pnl)

    async def generate_signals(
        self,
        market_regimes: Dict[str, Any],
        price_engine: Any,
    ) -> List[Dict]:
        """
        Generate signals from all enabled strategies.

        Args:
            market_regimes: Dict of symbol -> market regime
            price_engine: Price data source with get_ohlcv method
        """
        all_signals = []

        for symbol, regime in market_regimes.items():
            regime_value = regime.value if hasattr(regime, "value") else str(regime)

            try:
                ohlcv = price_engine.get_ohlcv(symbol, "1h", limit=100)
                if not ohlcv or len(ohlcv) < 50:
                    continue
            except Exception as e:
                logger.warning(f"Could not get data for {symbol}: {e}")
                continue

            for strategy in self.strategies.values():
                if not getattr(strategy, "enabled", True):
                    continue

                try:
                    if hasattr(strategy, "generate_signals"):
                        signals = await strategy.generate_signals(
                            symbol=symbol,
                            price_data=ohlcv,
                            market_regime=regime_value,
                        )
                        for signal in signals:
                            all_signals.append(
                                signal.to_dict() if hasattr(signal, "to_dict") else signal
                            )
                except Exception as e:
                    name = getattr(strategy, "name", str(strategy))
                    logger.error(f"Strategy {name} error for {symbol}: {e}")

        deduplicated = self._deduplicate_signals(all_signals)
        deduplicated.sort(key=lambda x: x.get("strength", 0), reverse=True)
        return deduplicated

    def _deduplicate_signals(self, signals: List[Dict]) -> List[Dict]:
        seen = {}
        for signal in signals:
            key = (signal.get("symbol"), signal.get("action"))
            if key not in seen or signal.get("strength", 0) > seen[key].get("strength", 0):
                seen[key] = signal
        return list(seen.values())
