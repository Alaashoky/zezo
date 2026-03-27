"""Smart Money Concepts (SMC) / ICT Strategy"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class SMCICTStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        params = config.parameters or {}
        self.ob_lookback = params.get("ob_lookback", 20)
        self.fvg_min_gap = params.get("fvg_min_gap", 0.001)
        self.structure_lookback = params.get("structure_lookback", 50)
        self.ote_fibonacci = params.get("ote_fibonacci", [0.62, 0.705, 0.79])
        self.ohlcv_history = []
        logger.info(f"SMC ICT initialized for {config.symbol}")

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prices = data.get("prices", [])
            if not prices or not isinstance(prices[0], dict):
                close_list = data.get("close", [])
                if isinstance(close_list, (int, float)):
                    close_list = [close_list]
                prices = [{"open": c, "high": c, "low": c, "close": c, "volume": 0}
                          for c in close_list]

            self.ohlcv_history.extend(prices)
            keep = self.structure_lookback * 3
            if len(self.ohlcv_history) > keep:
                self.ohlcv_history = self.ohlcv_history[-keep:]

            if len(self.ohlcv_history) < self.structure_lookback:
                return {"error": "Insufficient data"}

            current_price = self.ohlcv_history[-1].get("close", 0)
            market_structure = self._analyze_market_structure(self.ohlcv_history)
            order_blocks = self._identify_order_blocks(self.ohlcv_history)
            fair_value_gaps = self._identify_fair_value_gaps(self.ohlcv_history)
            liquidity_zones = self._analyze_liquidity(self.ohlcv_history)
            premium_discount = self._calculate_premium_discount(self.ohlcv_history)
            ote_levels = self._calculate_ote_levels(self.ohlcv_history, market_structure)

            return {
                "current_price": current_price,
                "market_structure": market_structure,
                "order_blocks": order_blocks,
                "fair_value_gaps": fair_value_gaps,
                "liquidity_zones": liquidity_zones,
                "premium_discount": premium_discount,
                "ote_levels": ote_levels,
                "timestamp": datetime.now(timezone.utc),
            }
        except Exception as e:
            logger.error(f"SMC analyze error: {e}")
            return {"error": str(e)}

    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        if "error" in analysis:
            return None
        try:
            current_price = analysis["current_price"]
            market_structure = analysis["market_structure"]
            order_blocks = analysis["order_blocks"]
            fair_value_gaps = analysis["fair_value_gaps"]
            liquidity_zones = analysis["liquidity_zones"]
            premium_discount = analysis["premium_discount"]
            ote_levels = analysis["ote_levels"]

            if market_structure.get("trend") == "bullish":
                score = 0.0
                if self._price_near_level(current_price, order_blocks.get("bullish", [])):
                    score += 0.25
                if self._price_in_fvg(current_price, fair_value_gaps.get("bullish", [])):
                    score += 0.25
                if premium_discount.get("zone") == "discount":
                    score += 0.20
                if self._price_near_level(current_price, ote_levels.get("bullish", [])):
                    score += 0.20
                if liquidity_zones.get("swept_below", False):
                    score += 0.10

                if score >= 0.50:
                    return Signal(SignalType.BUY, self.config.symbol, current_price,
                                  analysis["timestamp"], min(score, 1.0),
                                  metadata={"reason": "SMC Bullish Setup", "score": score,
                                            "structure": market_structure.get("type")})

            elif market_structure.get("trend") == "bearish":
                score = 0.0
                if self._price_near_level(current_price, order_blocks.get("bearish", [])):
                    score += 0.25
                if self._price_in_fvg(current_price, fair_value_gaps.get("bearish", [])):
                    score += 0.25
                if premium_discount.get("zone") == "premium":
                    score += 0.20
                if self._price_near_level(current_price, ote_levels.get("bearish", [])):
                    score += 0.20
                if liquidity_zones.get("swept_above", False):
                    score += 0.10

                if score >= 0.50:
                    return Signal(SignalType.SELL, self.config.symbol, current_price,
                                  analysis["timestamp"], min(score, 1.0),
                                  metadata={"reason": "SMC Bearish Setup", "score": score,
                                            "structure": market_structure.get("type")})

            return None
        except Exception as e:
            logger.error(f"SMC generate_signal error: {e}")
            return None

    def _analyze_market_structure(self, prices: List[Dict]) -> Dict:
        window = prices[-self.structure_lookback:]
        highs = [p["high"] for p in window]
        lows = [p["low"] for p in window]
        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
        hl = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
        lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
        ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
        if hh > lh and hl > ll:
            return {"trend": "bullish", "type": "HH_HL"}
        elif lh > hh and ll > hl:
            return {"trend": "bearish", "type": "LH_LL"}
        return {"trend": "neutral", "type": "consolidation"}

    def _identify_order_blocks(self, prices: List[Dict]) -> Dict:
        bullish, bearish = [], []
        window = prices[max(0, len(prices) - self.ob_lookback):]
        for i in range(len(window) - 1):
            c, n = window[i], window[i + 1]
            if (c["close"] < c["open"] and n["close"] > n["open"] and n["close"] > c["high"]):
                bullish.append(c["low"])
            if (c["close"] > c["open"] and n["close"] < n["open"] and n["close"] < c["low"]):
                bearish.append(c["high"])
        return {"bullish": bullish[-5:], "bearish": bearish[-5:]}

    def _identify_fair_value_gaps(self, prices: List[Dict]) -> Dict:
        bullish, bearish = [], []
        for i in range(2, len(prices)):
            if prices[i]["low"] > prices[i - 2]["high"]:
                gap = (prices[i]["low"] - prices[i - 2]["high"]) / (prices[i - 2]["high"] + 1e-10)
                if gap >= self.fvg_min_gap:
                    bullish.append({"top": prices[i]["low"], "bottom": prices[i - 2]["high"]})
            if prices[i]["high"] < prices[i - 2]["low"]:
                gap = (prices[i - 2]["low"] - prices[i]["high"]) / (prices[i]["high"] + 1e-10)
                if gap >= self.fvg_min_gap:
                    bearish.append({"top": prices[i - 2]["low"], "bottom": prices[i]["high"]})
        return {"bullish": bullish[-3:], "bearish": bearish[-3:]}

    def _analyze_liquidity(self, prices: List[Dict]) -> Dict:
        recent = prices[-20:]
        highs = [p["high"] for p in recent]
        lows = [p["low"] for p in recent]
        return {
            "swept_above": recent[-1]["high"] > max(highs[:-1]) if len(highs) > 1 else False,
            "swept_below": recent[-1]["low"] < min(lows[:-1]) if len(lows) > 1 else False,
        }

    def _calculate_premium_discount(self, prices: List[Dict]) -> Dict:
        window = prices[-50:] if len(prices) >= 50 else prices
        high = max(p["high"] for p in window)
        low = min(p["low"] for p in window)
        mid = low + (high - low) * 0.5
        price = prices[-1]["close"]
        return {"zone": "premium" if price > mid else "discount", "mid_point": mid,
                "high": high, "low": low}

    def _calculate_ote_levels(self, prices: List[Dict], structure: Dict) -> Dict:
        window = prices[-50:] if len(prices) >= 50 else prices
        high = max(p["high"] for p in window)
        low = min(p["low"] for p in window)
        r = high - low
        bullish = [high - r * f for f in self.ote_fibonacci] if structure.get("trend") == "bullish" else []
        bearish = [low + r * f for f in self.ote_fibonacci] if structure.get("trend") == "bearish" else []
        return {"bullish": bullish, "bearish": bearish}

    def _price_near_level(self, price: float, levels: List[float], threshold: float = 0.002) -> bool:
        return any(abs(price - lvl) / (lvl + 1e-10) <= threshold for lvl in levels)

    def _price_in_fvg(self, price: float, fvgs: List[Dict]) -> bool:
        return any(fvg["bottom"] <= price <= fvg["top"] for fvg in fvgs)