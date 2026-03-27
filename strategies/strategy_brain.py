# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
Strategy Brain - Multi-Strategy Joint Analysis Core

This module provides the central intelligence for combining signals from
multiple strategies into unified, high-confidence trading decisions.

Features:
- Multi-strategy signal aggregation
- Confidence weighting system
- Consensus-based decision making
- Signal correlation analysis
- Risk-adjusted signal combining
- Real-time strategy coordination
- Performance-weighted voting
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType, StrategyStatus

logger = logging.getLogger(__name__)


class StrategyBrain:
    """
    Central intelligence for multi-strategy coordination and joint analysis.

    Combines signals from multiple strategies using:
    - Weighted voting based on historical performance
    - Confidence score aggregation
    - Correlation analysis between strategies
    - Risk-adjusted position sizing recommendations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.min_strategies_required = self.config.get("min_strategies_required", 2)
        self.consensus_threshold = self.config.get("consensus_threshold", 0.6)
        self.performance_weight = self.config.get("performance_weight", 0.4)
        self.confidence_weight = self.config.get("confidence_weight", 0.6)

        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.strategy_weights: Dict[str, float] = {}

        self.signal_history: List[Dict[str, Any]] = []
        self.consensus_signals: List[Signal] = []

        self.stats = {
            "total_analyses": 0,
            "consensus_reached": 0,
            "bullish_consensus": 0,
            "bearish_consensus": 0,
            "neutral_count": 0,
            "average_confidence": 0.0,
        }

        logger.info("Strategy Brain initialized")

    def register_strategy(self, strategy: BaseStrategy):
        strategy_name = strategy.config.name
        self.strategies[strategy_name] = strategy

        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "total_signals": 0,
                "correct_signals": 0,
                "win_rate": 0.5,
                "average_confidence": 0.5,
                "total_pnl": 0.0,
            }

        self._recalculate_weights()
        logger.info(f"Strategy Brain: Registered {strategy_name}")

    def unregister_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self._recalculate_weights()
            logger.info(f"Strategy Brain: Unregistered {strategy_name}")

    def analyze_joint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.stats["total_analyses"] += 1

        try:
            strategy_signals = {}

            for name, strategy in self.strategies.items():
                if strategy.status != StrategyStatus.RUNNING:
                    continue
                try:
                    signal = strategy.on_bar(data)
                    if signal:
                        strategy_signals[name] = signal
                except Exception as e:
                    logger.error(f"Error getting signal from {name}: {e}")

            if len(strategy_signals) < self.min_strategies_required:
                self.stats["neutral_count"] += 1
                return {
                    "consensus_reached": False,
                    "reason": f"Insufficient strategies ({len(strategy_signals)} < {self.min_strategies_required})",
                    "signal": None,
                }

            consensus_result = self._calculate_consensus(strategy_signals, data)

            if consensus_result["consensus_reached"]:
                self.stats["consensus_reached"] += 1
                if consensus_result["consensus_signal"].signal_type == SignalType.BUY:
                    self.stats["bullish_consensus"] += 1
                elif consensus_result["consensus_signal"].signal_type == SignalType.SELL:
                    self.stats["bearish_consensus"] += 1

                self.stats["average_confidence"] = (
                    self.stats["average_confidence"] * (self.stats["total_analyses"] - 1)
                    + consensus_result["consensus_signal"].confidence
                ) / self.stats["total_analyses"]

                self.consensus_signals.append(consensus_result["consensus_signal"])
            else:
                self.stats["neutral_count"] += 1

            self.signal_history.append({
                "timestamp": datetime.now(timezone.utc),
                "strategy_signals": strategy_signals,
                "consensus": consensus_result,
                "data_snapshot": data.get("prices", [])[-1] if data.get("prices") else {},
            })

            return consensus_result

        except Exception as e:
            logger.error(f"Error in joint analysis: {e}")
            return {"consensus_reached": False, "reason": "Analysis error", "signal": None}

    def _calculate_consensus(
        self,
        strategy_signals: Dict[str, Signal],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            buy_signals = []
            sell_signals = []

            for strategy_name, signal in strategy_signals.items():
                weight = self.strategy_weights.get(
                    strategy_name,
                    1.0 / len(self.strategies),
                )

                weighted_confidence = (
                    signal.confidence * self.confidence_weight
                    + self.strategy_performance[strategy_name]["win_rate"] * self.performance_weight
                ) * weight

                if signal.signal_type == SignalType.BUY:
                    buy_signals.append({
                        "strategy": strategy_name,
                        "signal": signal,
                        "weight": weight,
                        "weighted_confidence": weighted_confidence,
                    })
                elif signal.signal_type == SignalType.SELL:
                    sell_signals.append({
                        "strategy": strategy_name,
                        "signal": signal,
                        "weight": weight,
                        "weighted_confidence": weighted_confidence,
                    })

            total_buy_confidence = sum(s["weighted_confidence"] for s in buy_signals)
            total_sell_confidence = sum(s["weighted_confidence"] for s in sell_signals)
            total_confidence = total_buy_confidence + total_sell_confidence

            if total_confidence == 0:
                return {"consensus_reached": False, "reason": "No directional signals", "signal": None}

            buy_ratio = total_buy_confidence / total_confidence
            sell_ratio = total_sell_confidence / total_confidence

            consensus_signal = None
            consensus_reached = False
            reason = ""

            if buy_ratio >= self.consensus_threshold:
                consensus_reached = True
                avg_price = np.mean([s["signal"].price for s in buy_signals])
                consensus_confidence = total_buy_confidence / len(buy_signals) if buy_signals else 0

                consensus_signal = Signal(
                    signal_type=SignalType.BUY,
                    symbol=list(strategy_signals.values())[0].symbol,
                    price=avg_price,
                    timestamp=datetime.now(timezone.utc),
                    confidence=min(consensus_confidence, 1.0),
                    metadata={
                        "type": "consensus",
                        "agreeing_strategies": [s["strategy"] for s in buy_signals],
                        "total_strategies": len(strategy_signals),
                        "buy_ratio": buy_ratio,
                        "weighted_confidence": total_buy_confidence,
                        "strategy_details": {
                            s["strategy"]: {
                                "confidence": s["signal"].confidence,
                                "weight": s["weight"],
                                "metadata": s["signal"].metadata,
                            }
                            for s in buy_signals
                        },
                    },
                )
                reason = f"Bullish consensus: {len(buy_signals)}/{len(strategy_signals)} strategies agree"

            elif sell_ratio >= self.consensus_threshold:
                consensus_reached = True
                avg_price = np.mean([s["signal"].price for s in sell_signals])
                consensus_confidence = total_sell_confidence / len(sell_signals) if sell_signals else 0

                consensus_signal = Signal(
                    signal_type=SignalType.SELL,
                    symbol=list(strategy_signals.values())[0].symbol,
                    price=avg_price,
                    timestamp=datetime.now(timezone.utc),
                    confidence=min(consensus_confidence, 1.0),
                    metadata={
                        "type": "consensus",
                        "agreeing_strategies": [s["strategy"] for s in sell_signals],
                        "total_strategies": len(strategy_signals),
                        "sell_ratio": sell_ratio,
                        "weighted_confidence": total_sell_confidence,
                        "strategy_details": {
                            s["strategy"]: {
                                "confidence": s["signal"].confidence,
                                "weight": s["weight"],
                                "metadata": s["signal"].metadata,
                            }
                            for s in sell_signals
                        },
                    },
                )
                reason = f"Bearish consensus: {len(sell_signals)}/{len(strategy_signals)} strategies agree"

            else:
                reason = f"No consensus: Buy {buy_ratio:.1%}, Sell {sell_ratio:.1%}"

            return {
                "consensus_reached": consensus_reached,
                "consensus_signal": consensus_signal,
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals),
                "total_signals": len(strategy_signals),
                "buy_ratio": buy_ratio,
                "sell_ratio": sell_ratio,
                "reason": reason,
                "analysis_details": {
                    "buy_confidence": total_buy_confidence,
                    "sell_confidence": total_sell_confidence,
                    "buy_strategies": [s["strategy"] for s in buy_signals],
                    "sell_strategies": [s["strategy"] for s in sell_signals],
                },
            }

        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return {"consensus_reached": False, "reason": "Consensus calculation error", "signal": None}

    def update_strategy_performance(
        self,
        strategy_name: str,
        signal_correct: bool,
        pnl: float,
    ):
        if strategy_name not in self.strategy_performance:
            return

        perf = self.strategy_performance[strategy_name]
        perf["total_signals"] += 1
        if signal_correct:
            perf["correct_signals"] += 1
        perf["total_pnl"] += pnl
        perf["win_rate"] = perf["correct_signals"] / perf["total_signals"]
        self._recalculate_weights()

        logger.info(
            f"Updated performance for {strategy_name}: "
            f"Win rate: {perf['win_rate']:.2%}, PnL: ${perf['total_pnl']:.2f}",
        )

    def _recalculate_weights(self):
        if not self.strategies:
            return

        total_performance = 0.0
        strategy_scores = {}

        for name in self.strategies.keys():
            if name in self.strategy_performance:
                perf = self.strategy_performance[name]
                score = (
                    perf["win_rate"] * 0.7
                    + min(perf["total_pnl"] / 1000, 1.0) * 0.3
                )
            else:
                score = 0.5
            strategy_scores[name] = max(score, 0.1)
            total_performance += strategy_scores[name]

        if total_performance > 0:
            self.strategy_weights = {
                name: score / total_performance
                for name, score in strategy_scores.items()
            }
        else:
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {name: equal_weight for name in self.strategies.keys()}

    def get_statistics(self) -> Dict[str, Any]:
        consensus_rate = (
            self.stats["consensus_reached"] / self.stats["total_analyses"]
            if self.stats["total_analyses"] > 0
            else 0
        )
        return {
            "total_analyses": self.stats["total_analyses"],
            "consensus_reached": self.stats["consensus_reached"],
            "consensus_rate": consensus_rate,
            "bullish_consensus": self.stats["bullish_consensus"],
            "bearish_consensus": self.stats["bearish_consensus"],
            "neutral_count": self.stats["neutral_count"],
            "average_confidence": self.stats["average_confidence"],
            "registered_strategies": len(self.strategies),
            "strategy_weights": self.strategy_weights.copy(),
            "strategy_performance": self.strategy_performance.copy(),
        }

    def get_strategy_correlations(self) -> Dict[str, Dict[str, float]]:
        correlations = {}
        strategy_names = list(self.strategies.keys())

        for strategy1 in strategy_names:
            correlations[strategy1] = {}
            for strategy2 in strategy_names:
                if strategy1 == strategy2:
                    correlations[strategy1][strategy2] = 1.0
                else:
                    agreements = 0
                    total_comparisons = 0
                    for history_entry in self.signal_history:
                        signals = history_entry["strategy_signals"]
                        if strategy1 in signals and strategy2 in signals:
                            total_comparisons += 1
                            if signals[strategy1].signal_type == signals[strategy2].signal_type:
                                agreements += 1
                    correlation = agreements / total_comparisons if total_comparisons > 0 else 0.5
                    correlations[strategy1][strategy2] = correlation

        return correlations

    def analyze_with_ai(
        self,
        data: Dict[str, Any],
        ai_prediction: Optional[Dict[str, Any]] = None,
        ai_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Combine strategy consensus with an AI model prediction.

        Parameters
        ----------
        data          : OHLCV bar dict passed to every strategy
        ai_prediction : output dict from MarketPredictor.predict()
                        (must contain 'signal_type' and 'confidence' keys)
        ai_weight     : how much weight to give the AI signal vs strategy consensus
                        (0 = ignore AI, 1 = ignore strategies)

        Returns
        -------
        Same structure as analyze_joint(), with an extra 'ai_contribution' key.
        Falls back to strategy-only consensus if ai_prediction is None or fails.
        """
        strategy_result = self.analyze_joint(data)

        if ai_prediction is None:
            strategy_result["ai_contribution"] = None
            return strategy_result

        try:
            ai_signal_type = ai_prediction.get("signal_type")
            ai_confidence = float(ai_prediction.get("confidence", 0.0))

            if ai_signal_type is None:
                ai_signal = ai_prediction.get("signal")
                ai_signal_type = getattr(ai_signal, "signal_type", None) if ai_signal else None

            # If strategy consensus was reached, blend the confidence
            if strategy_result.get("consensus_reached") and strategy_result.get("consensus_signal"):
                cs = strategy_result["consensus_signal"]
                strategy_weight = 1.0 - ai_weight

                if cs.signal_type == ai_signal_type:
                    # agreement — boost confidence
                    blended_confidence = min(
                        cs.confidence * strategy_weight + ai_confidence * ai_weight, 1.0
                    )
                    cs_dict = cs.__dict__.copy() if hasattr(cs, "__dict__") else {}
                    cs_dict["confidence"] = blended_confidence
                    # rebuild Signal with updated confidence
                    from strategies.base import Signal
                    from datetime import datetime, timezone

                    blended_signal = Signal(
                        signal_type=cs.signal_type,
                        symbol=cs.symbol,
                        price=cs.price,
                        timestamp=datetime.now(timezone.utc),
                        confidence=blended_confidence,
                        metadata={
                            **(cs.metadata or {}),
                            "ai_confidence": ai_confidence,
                            "ai_signal_type": str(ai_signal_type),
                            "ai_weight": ai_weight,
                            "blend": "agreement",
                        },
                    )
                    strategy_result["consensus_signal"] = blended_signal
                    strategy_result["ai_contribution"] = {
                        "ai_signal_type": str(ai_signal_type),
                        "ai_confidence": ai_confidence,
                        "blend": "agreement",
                        "blended_confidence": blended_confidence,
                    }
                else:
                    # disagreement — reduce confidence
                    reduced_confidence = max(
                        cs.confidence * strategy_weight - ai_confidence * ai_weight * 0.5, 0.0
                    )
                    from strategies.base import Signal
                    from datetime import datetime, timezone

                    reduced_signal = Signal(
                        signal_type=cs.signal_type,
                        symbol=cs.symbol,
                        price=cs.price,
                        timestamp=datetime.now(timezone.utc),
                        confidence=reduced_confidence,
                        metadata={
                            **(cs.metadata or {}),
                            "ai_confidence": ai_confidence,
                            "ai_signal_type": str(ai_signal_type),
                            "ai_weight": ai_weight,
                            "blend": "disagreement",
                        },
                    )
                    strategy_result["consensus_signal"] = reduced_signal
                    # suppress consensus if confidence drops too low
                    if reduced_confidence < self.consensus_threshold * 0.5:
                        strategy_result["consensus_reached"] = False
                        strategy_result["reason"] = (
                            f"Consensus suppressed by AI disagreement "
                            f"(confidence reduced to {reduced_confidence:.2f})"
                        )
                    strategy_result["ai_contribution"] = {
                        "ai_signal_type": str(ai_signal_type),
                        "ai_confidence": ai_confidence,
                        "blend": "disagreement",
                        "reduced_confidence": reduced_confidence,
                    }
            else:
                # no strategy consensus — let AI signal stand alone if confident enough
                if ai_confidence >= self.consensus_threshold and ai_signal_type is not None:
                    from strategies.base import Signal, SignalType
                    from datetime import datetime, timezone

                    current_price = data.get("close") or (
                        data["prices"][-1].get("close") if data.get("prices") else 0.0
                    )
                    ai_only_signal = Signal(
                        signal_type=ai_signal_type,
                        symbol=data.get("symbol", "UNKNOWN"),
                        price=float(current_price),
                        timestamp=datetime.now(timezone.utc),
                        confidence=ai_confidence * ai_weight,
                        metadata={
                            "source": "ai_only",
                            "ai_confidence": ai_confidence,
                            "ai_weight": ai_weight,
                        },
                    )
                    strategy_result["consensus_reached"] = True
                    strategy_result["consensus_signal"] = ai_only_signal
                    strategy_result["reason"] = "AI-only signal (no strategy consensus)"
                    strategy_result["ai_contribution"] = {
                        "ai_signal_type": str(ai_signal_type),
                        "ai_confidence": ai_confidence,
                        "blend": "ai_only",
                    }
                else:
                    strategy_result["ai_contribution"] = {
                        "ai_signal_type": str(ai_signal_type) if ai_signal_type else None,
                        "ai_confidence": ai_confidence,
                        "blend": "none",
                    }

        except Exception as e:
            logger.error(f"Error blending AI prediction: {e}")
            strategy_result["ai_contribution"] = {"error": str(e)}

        return strategy_result

    def __repr__(self) -> str:
        return (
            f"StrategyBrain("
            f"strategies={len(self.strategies)}, "
            f"consensus_rate={self.stats['consensus_reached']}/{self.stats['total_analyses']})"
        )
