# HOPEFX-AI-TRADING
# Copyright (c) 2025-2026
# Licensed under GNU Affero General Public License v3.0 (AGPL-3.0)
# All modifications must be shared under the same license.
# No commercial use without explicit permission.
"""
ICT Trading System - 8 Optimal Setups (ITS-8-OS)

This strategy implements the 8 core optimal setups from the
Inner Circle Trader methodology:
1. AMD (Accumulation, Manipulation, Distribution)
2. Power of 3
3. Judas Swing
4. Kill Zones (London, New York, Asian)
5. ICT Turtle Soup
6. Silver Bullet Setup
7. Optimal Trade Entry
8. Session-based Analysis
"""

import logging
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class ITS8OSStrategy(BaseStrategy):
    """
    ICT 8 Optimal Setups Strategy.

    Implements the complete ICT trading system with all 8 setups:
    - AMD pattern recognition
    - Power of 3 structure
    - Judas Swing detection
    - Kill Zone timing
    - Turtle Soup reversals
    - Silver Bullet entries
    - OTE precision
    - Session-based logic
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        params = config.parameters or {}
        self.enabled_setups = params.get("enabled_setups", list(range(1, 9)))
        self.min_setup_score = params.get("min_setup_score", 0.6)
        self.confluence_required = params.get("confluence_required", 2)

        self.kill_zones = {
            "asian": {"start": time(0, 0), "end": time(3, 0)},
            "london": {"start": time(2, 0), "end": time(5, 0)},
            "new_york": {"start": time(8, 30), "end": time(11, 0)},
            "london_close": {"start": time(10, 0), "end": time(12, 0)},
        }

        self.current_session = None
        self.session_high = None
        self.session_low = None
        self.manipulation_detected = False
        self.amd_phase = "accumulation"

        logger.info(f"ITS-8-OS Strategy initialized for {config.symbol}")

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prices = data.get("prices", [])
            if len(prices) < 50:
                return {"error": "Insufficient data"}

            current_price = prices[-1].get("close", 0)
            current_time = data.get("timestamp", datetime.now(timezone.utc))

            setup_results = {}

            if 1 in self.enabled_setups:
                setup_results["amd"] = self._analyze_amd_pattern(prices)
            if 2 in self.enabled_setups:
                setup_results["power_of_3"] = self._analyze_power_of_3(prices)
            if 3 in self.enabled_setups:
                setup_results["judas_swing"] = self._analyze_judas_swing(prices)
            if 4 in self.enabled_setups:
                setup_results["kill_zone"] = self._analyze_kill_zones(current_time)
            if 5 in self.enabled_setups:
                setup_results["turtle_soup"] = self._analyze_turtle_soup(prices)
            if 6 in self.enabled_setups:
                setup_results["silver_bullet"] = self._analyze_silver_bullet(prices, current_time)
            if 7 in self.enabled_setups:
                setup_results["ote"] = self._analyze_ote(prices)
            if 8 in self.enabled_setups:
                setup_results["session"] = self._analyze_session(prices, current_time)

            confluence = self._calculate_confluence(setup_results)

            return {
                "current_price": current_price,
                "timestamp": current_time,
                "setup_results": setup_results,
                "confluence": confluence,
                "active_kill_zone": setup_results.get("kill_zone", {}).get("active_zone"),
            }

        except Exception as e:
            logger.error(f"Error in ITS-8-OS analysis: {e}")
            return {"error": str(e)}

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        if "error" in analysis:
            return None

        try:
            current_price = analysis["current_price"]
            confluence = analysis["confluence"]
            setup_results = analysis["setup_results"]

            if confluence["agreeing_setups"] < self.confluence_required:
                return None

            signal_type = SignalType.HOLD
            confidence = 0.0
            metadata = {}

            if confluence["bullish_score"] > confluence["bearish_score"]:
                signal_type = SignalType.BUY
                confidence = min(confluence["bullish_score"], 1.0)
                metadata = {
                    "reason": "ITS-8-OS Bullish Confluence",
                    "agreeing_setups": confluence["agreeing_setups"],
                    "bullish_setups": confluence["bullish_setups"],
                    "active_kill_zone": analysis.get("active_kill_zone"),
                    "setup_details": self._extract_signal_details(setup_results, "bullish"),
                }
            elif confluence["bearish_score"] > confluence["bullish_score"]:
                signal_type = SignalType.SELL
                confidence = min(confluence["bearish_score"], 1.0)
                metadata = {
                    "reason": "ITS-8-OS Bearish Confluence",
                    "agreeing_setups": confluence["agreeing_setups"],
                    "bearish_setups": confluence["bearish_setups"],
                    "active_kill_zone": analysis.get("active_kill_zone"),
                    "setup_details": self._extract_signal_details(setup_results, "bearish"),
                }

            if signal_type != SignalType.HOLD and confidence >= self.min_setup_score:
                return Signal(
                    signal_type=signal_type,
                    symbol=self.config.symbol,
                    price=current_price,
                    timestamp=analysis["timestamp"],
                    confidence=confidence,
                    metadata=metadata,
                )

            return None

        except Exception as e:
            logger.error(f"Error generating ITS-8-OS signal: {e}")
            return None

    def _analyze_amd_pattern(self, prices: List[Dict]) -> Dict[str, Any]:
        try:
            recent_prices = [p["close"] for p in prices[-30:]]
            recent_highs = [p["high"] for p in prices[-30:]]
            recent_lows = [p["low"] for p in prices[-30:]]
            avg_price = np.mean(recent_prices)
            volatility = np.std(recent_prices) / avg_price

            if volatility < 0.005:
                return {"phase": "accumulation", "signal": "neutral", "score": 0.3, "volatility": volatility}

            if len(prices) > 5:
                last_move = abs(prices[-1]["close"] - prices[-5]["close"]) / prices[-5]["close"]
                if last_move > 0.01:
                    signal = "bullish" if prices[-1]["close"] < prices[-5]["close"] else "bearish"
                    return {"phase": "manipulation", "signal": signal, "score": 0.7, "volatility": volatility}
                return {"phase": "distribution", "signal": "neutral", "score": 0.4, "volatility": volatility}

            return {"phase": "unknown", "signal": "neutral", "score": 0.0, "volatility": volatility}
        except Exception as e:
            logger.error(f"Error analyzing AMD: {e}")
            return {"phase": "unknown", "signal": "neutral", "score": 0.0}

    def _analyze_power_of_3(self, prices: List[Dict]) -> Dict[str, Any]:
        try:
            if len(prices) < 3:
                return {"detected": False, "signal": "neutral", "score": 0.0}

            consolidation = all(
                abs(prices[i]["close"] - prices[i]["open"]) < abs(prices[-1]["close"] - prices[-1]["open"])
                for i in range(-10, -1)
                if i + len(prices) > 0
            )
            expansion = (
                abs(prices[-1]["close"] - prices[-1]["open"]) / prices[-1]["open"] > 0.005
            )

            if consolidation and expansion:
                signal = "bullish" if prices[-1]["close"] > prices[-1]["open"] else "bearish"
                return {"detected": True, "signal": signal, "score": 0.7}

            return {"detected": False, "signal": "neutral", "score": 0.0}
        except Exception as e:
            logger.error(f"Error analyzing Power of 3: {e}")
            return {"detected": False, "signal": "neutral", "score": 0.0}

    def _analyze_judas_swing(self, prices: List[Dict]) -> Dict[str, Any]:
        try:
            if len(prices) < 20:
                return {"detected": False, "signal": "neutral", "score": 0.0}

            recent_high = max(p["high"] for p in prices[-20:-1])
            recent_low = min(p["low"] for p in prices[-20:-1])
            current = prices[-1]

            if current["low"] < recent_low and current["close"] > recent_low:
                return {"detected": True, "signal": "bullish", "score": 0.8}
            if current["high"] > recent_high and current["close"] < recent_high:
                return {"detected": True, "signal": "bearish", "score": 0.8}

            return {"detected": False, "signal": "neutral", "score": 0.0}
        except Exception as e:
            logger.error(f"Error analyzing Judas Swing: {e}")
            return {"detected": False, "signal": "neutral", "score": 0.0}

    def _analyze_kill_zones(self, current_time: datetime) -> Dict[str, Any]:
        try:
            current_time_only = current_time.time()
            active_zone = None
            score = 0.0

            for zone_name, zone_times in self.kill_zones.items():
                if zone_times["start"] <= current_time_only <= zone_times["end"]:
                    active_zone = zone_name
                    score = 0.8 if zone_name in ["london", "new_york"] else 0.5
                    break

            return {
                "in_kill_zone": active_zone is not None,
                "active_zone": active_zone,
                "score": score,
                "signal": "neutral",
            }
        except Exception as e:
            logger.error(f"Error analyzing kill zones: {e}")
            return {"in_kill_zone": False, "active_zone": None, "score": 0.0, "signal": "neutral"}

    def _analyze_turtle_soup(self, prices: List[Dict]) -> Dict[str, Any]:
        try:
            if len(prices) < 20:
                return {"detected": False, "signal": "neutral", "score": 0.0}

            twenty_day_high = max(p["high"] for p in prices[-21:-1])
            twenty_day_low = min(p["low"] for p in prices[-21:-1])
            current = prices[-1]

            if current["low"] < twenty_day_low and current["close"] > twenty_day_low:
                return {"detected": True, "signal": "bullish", "score": 0.75}
            if current["high"] > twenty_day_high and current["close"] < twenty_day_high:
                return {"detected": True, "signal": "bearish", "score": 0.75}

            return {"detected": False, "signal": "neutral", "score": 0.0}
        except Exception as e:
            logger.error(f"Error analyzing Turtle Soup: {e}")
            return {"detected": False, "signal": "neutral", "score": 0.0}

    def _analyze_silver_bullet(self, prices: List[Dict], current_time: datetime) -> Dict[str, Any]:
        try:
            current_time_only = current_time.time()
            london_sb = time(3, 0) <= current_time_only <= time(4, 0)
            ny_sb = time(9, 0) <= current_time_only <= time(10, 0)
            in_sb_window = london_sb or ny_sb

            if not in_sb_window:
                return {"detected": False, "signal": "neutral", "score": 0.0}

            if len(prices) >= 5:
                initial_move = (
                    prices[-5]["close"] - prices[-10]["close"] if len(prices) >= 10 else 0
                )
                recent_pullback = prices[-1]["close"] - prices[-5]["close"]

                if initial_move > 0 and recent_pullback < 0:
                    return {"detected": True, "signal": "bullish", "score": 0.85,
                            "window": "london" if london_sb else "new_york"}
                elif initial_move < 0 and recent_pullback > 0:
                    return {"detected": True, "signal": "bearish", "score": 0.85,
                            "window": "london" if london_sb else "new_york"}

            return {"detected": False, "signal": "neutral", "score": 0.0}
        except Exception as e:
            logger.error(f"Error analyzing Silver Bullet: {e}")
            return {"detected": False, "signal": "neutral", "score": 0.0}

    def _analyze_ote(self, prices: List[Dict]) -> Dict[str, Any]:
        try:
            swing_high = max(p["high"] for p in prices[-50:])
            swing_low = min(p["low"] for p in prices[-50:])
            swing_range = swing_high - swing_low
            current_price = prices[-1]["close"]

            ote_low = swing_low + (swing_range * 0.62)
            ote_high = swing_low + (swing_range * 0.79)
            in_ote_zone = ote_low <= current_price <= ote_high

            if in_ote_zone:
                recent_trend = prices[-1]["close"] - prices[-20]["close"] if len(prices) >= 20 else 0
                signal = "bullish" if recent_trend > 0 else "bearish"
                return {"in_ote_zone": True, "signal": signal, "score": 0.7, "ote_range": (ote_low, ote_high)}

            return {"in_ote_zone": False, "signal": "neutral", "score": 0.0}
        except Exception as e:
            logger.error(f"Error analyzing OTE: {e}")
            return {"in_ote_zone": False, "signal": "neutral", "score": 0.0}

    def _analyze_session(self, prices: List[Dict], current_time: datetime) -> Dict[str, Any]:
        try:
            current_hour = current_time.hour
            if 0 <= current_hour < 8:
                session = "asian"
                return {"session": session, "signal": "neutral", "score": 0.3, "bias": "range"}
            elif 8 <= current_hour < 16:
                session = "london"
                if len(prices) >= 10:
                    trend = prices[-1]["close"] - prices[-10]["close"]
                    signal = "bullish" if trend > 0 else "bearish"
                    return {"session": session, "signal": signal, "score": 0.6, "bias": "trending"}
                return {"session": session, "signal": "neutral", "score": 0.3, "bias": "trending"}
            else:
                session = "new_york"
                return {"session": session, "signal": "neutral", "score": 0.4, "bias": "reversal"}
        except Exception as e:
            logger.error(f"Error analyzing session: {e}")
            return {"session": "unknown", "signal": "neutral", "score": 0.0}

    def _calculate_confluence(self, setup_results: Dict[str, Dict]) -> Dict[str, Any]:
        try:
            bullish_count = 0
            bearish_count = 0
            bullish_score = 0.0
            bearish_score = 0.0
            bullish_setups = []
            bearish_setups = []

            for setup_name, result in setup_results.items():
                signal = result.get("signal", "neutral")
                score = result.get("score", 0.0)
                if signal == "bullish":
                    bullish_count += 1
                    bullish_score += score
                    bullish_setups.append(setup_name)
                elif signal == "bearish":
                    bearish_count += 1
                    bearish_score += score
                    bearish_setups.append(setup_name)

            total_setups = len(setup_results)
            if total_setups > 0:
                bullish_score = bullish_score / total_setups
                bearish_score = bearish_score / total_setups

            return {
                "bullish_score": bullish_score,
                "bearish_score": bearish_score,
                "bullish_setups": bullish_setups,
                "bearish_setups": bearish_setups,
                "agreeing_setups": max(bullish_count, bearish_count),
            }
        except Exception as e:
            logger.error(f"Error calculating confluence: {e}")
            return {"bullish_score": 0.0, "bearish_score": 0.0, "bullish_setups": [], "bearish_setups": [], "agreeing_setups": 0}

    def _extract_signal_details(self, setup_results: Dict, direction: str) -> Dict[str, Any]:
        return {
            setup_name: {"score": result.get("score", 0.0), "detected": result.get("detected", False)}
            for setup_name, result in setup_results.items()
            if result.get("signal") == direction
        }
