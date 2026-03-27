"""
Base Strategy Classes
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class StrategyStatus(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    price: float
    timestamp: datetime
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class StrategyConfig:
    name: str
    symbol: str
    timeframe: str
    enabled: bool = True
    risk_per_trade: float = 1.0
    max_positions: int = 3
    parameters: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.status = StrategyStatus.IDLE
        self.positions = []
        self.signals_history = []
        self.performance_metrics = {
            "total_signals": 0,
            "winning_signals": 0,
            "losing_signals": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }
        logger.info(f"Initialized strategy: {config.name} for {config.symbol}")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def symbol(self) -> str:
        return self.config.symbol

    @property
    def is_active(self) -> bool:
        return self.status == StrategyStatus.RUNNING

    @property
    def performance(self) -> dict:
        return self.performance_metrics

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        pass

    def on_bar(self, bar: Dict[str, Any]) -> Optional[Signal]:
        try:
            analysis = self.analyze(bar)
            signal = self.generate_signal(analysis)
            if signal:
                self._record_signal(signal)
                logger.info(
                    f"{self.config.name}: {signal.signal_type.value} "
                    f"at {signal.price} (confidence={{signal.confidence:.2f}})"
                )
            return signal
        except Exception as e:
            logger.error(f"Error in {self.config.name}.on_bar: {{e}}")[n]    self.status = StrategyStatus.ERROR
            return None

    def start(self):
        self.status = StrategyStatus.RUNNING
        logger.info(f"Started: {self.config.name}")

    def stop(self):
        self.status = StrategyStatus.STOPPED

    def pause(self):
        self.status = StrategyStatus.PAUSED

    def resume(self):
        self.status = StrategyStatus.RUNNING

    def _record_signal(self, signal: Signal):
        self.signals_history.append(signal)
        self.performance_metrics["total_signals"] += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        metrics = self.performance_metrics.copy()
        total = metrics["total_signals"]
        if total > 0:
            metrics["win_rate"] = metrics["winning_signals"] / total * 100
        return metrics

    def update_performance(self, pnl: float, is_winner: Optional[bool] = None):
        if is_winner is None:
            is_winner = pnl > 0
        self.performance_metrics["total_pnl"] += pnl
        if is_winner:
            self.performance_metrics["winning_signals"] += 1
            self.performance_metrics["winning_trades"] += 1
        else:
            self.performance_metrics["losing_signals"] += 1
            self.performance_metrics["losing_trades"] += 1
        wins = self.performance_metrics["winning_trades"]
        losses = self.performance_metrics["losing_trades"]
        total = wins + losses
        self.performance_metrics["win_rate"] = (wins / total * 100.0) if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"{{self.__class__.__name__}}("
            f"name={{self.config.name}}, "
            f"symbol={{self.config.symbol}}, "
            f"status={{self.status.value}})"
        )
