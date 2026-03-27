"""
RiskManager — position sizing and risk checks for ZEZO live bot.

Key responsibilities
--------------------
• Calculate appropriate lot size for a given risk % and SL distance.
• Calculate SL / TP price levels (ATR-based or fixed points).
• Gate new trades (max open positions, daily loss limit).
• Validate final trade parameters before execution.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False


class RiskManager:
    """Compute position sizes and enforce risk limits."""

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : TradingConfig, optional
            Provides default_sl_points, default_tp_points, min_lot, max_lot, lot_step, etc.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Lot size calculation
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        symbol: str,
        account_balance: float,
        risk_percent: float,
        sl_points: int,
        symbol_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate lot size so that a full SL hit costs exactly risk_percent of balance.

        Formula
        -------
        risk_amount  = account_balance * risk_percent
        point_value  = trade_tick_value * (point / trade_tick_size)
        lot          = risk_amount / (sl_points * point_value)

        The result is clamped to [min_lot, max_lot] and rounded to lot_step.
        """
        min_lot  = self.config.min_lot  if self.config else 0.01
        max_lot  = self.config.max_lot  if self.config else 0.10
        lot_step = self.config.lot_step if self.config else 0.01

        risk_amount = account_balance * risk_percent

        # Use symbol_info if provided; otherwise fall back to MT5 query
        if symbol_info is None and MT5_AVAILABLE:
            try:
                info = mt5.symbol_info(symbol)
                if info:
                    symbol_info = {
                        "point":               info.point,
                        "trade_tick_size":     info.trade_tick_size,
                        "trade_tick_value":    info.trade_tick_value,
                    }
            except Exception as exc:
                logger.warning(f"Could not get symbol info for lot calc: {exc}")

        if symbol_info:
            point          = symbol_info.get("point", 0.00001)
            tick_size      = symbol_info.get("trade_tick_size", 0.00001)
            tick_value     = symbol_info.get("trade_tick_value", 1.0)
            # point_value = monetary value of 1 point move on 1 lot
            point_value = tick_value * (point / tick_size) if tick_size else tick_value
        else:
            # Fallback — assume standard gold pip value ≈ $1 per 0.01 lot per point
            point_value = 1.0

        sl_points = max(sl_points, 1)

        try:
            lot = risk_amount / (sl_points * point_value)
        except ZeroDivisionError:
            logger.warning("ZeroDivisionError in lot calc — using min_lot")
            lot = min_lot

        # Clamp and round to step
        lot = max(min_lot, min(max_lot, lot))
        steps = round(lot / lot_step)
        lot = steps * lot_step
        lot = max(min_lot, min(max_lot, lot))

        logger.debug(
            f"Lot calc: balance={account_balance} risk={risk_percent*100:.1f}% "
            f"sl_pts={sl_points} → lot={lot}"
        )
        # Floating-point cleanup — keep precision matching lot_step decimals
        decimals = len(str(lot_step).rstrip("0").split(".")[-1]) if "." in str(lot_step) else 0
        return round(lot, decimals)

    # ------------------------------------------------------------------
    # SL / TP calculation
    # ------------------------------------------------------------------

    def calculate_sl_tp(
        self,
        symbol: str,
        signal_type: str,           # "BUY" or "SELL"
        entry_price: float,
        atr: Optional[float] = None,
        symbol_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Return {"sl": <price>, "tp": <price>} levels.

        When ATR is supplied the SL distance = 1.5 × ATR, ensuring ≥ 2:1 R:R.
        Otherwise fall back to fixed points from config.
        """
        sl_pts = self.config.default_sl_points if self.config else 200
        tp_pts = self.config.default_tp_points if self.config else 400

        # Determine point size
        point = 0.01  # XAUUSD default (price in USD, 0.01 per point)
        if symbol_info:
            point = symbol_info.get("point", point)
        elif MT5_AVAILABLE:
            try:
                info = mt5.symbol_info(symbol)
                if info:
                    point = info.point
            except Exception:
                pass

        if atr is not None and atr > 0:
            sl_distance = atr * 1.5
            tp_distance = sl_distance * 2.0   # 2:1 R:R
        else:
            sl_distance = sl_pts * point
            tp_distance = tp_pts * point

        if signal_type.upper() == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        # Round to symbol digits
        digits = 5
        if symbol_info:
            digits = symbol_info.get("digits", digits)
        elif MT5_AVAILABLE:
            try:
                info = mt5.symbol_info(symbol)
                if info:
                    digits = info.digits
            except Exception:
                pass

        sl = round(sl, digits)
        tp = round(tp, digits)

        logger.debug(
            f"SL/TP calc: {signal_type} entry={entry_price} "
            f"sl={sl} tp={tp} (atr={atr})"
        )
        return {"sl": sl, "tp": tp}

    # ------------------------------------------------------------------
    # Trade gate checks
    # ------------------------------------------------------------------

    def can_open_trade(
        self, current_positions: List[Any], max_positions: Optional[int] = None
    ) -> bool:
        """Return True when we have capacity to open a new position."""
        max_pos = max_positions
        if max_pos is None and self.config:
            max_pos = self.config.max_open_positions
        if max_pos is None:
            max_pos = 2

        open_count = len(current_positions)
        allowed = open_count < max_pos
        if not allowed:
            logger.info(
                f"Max positions reached ({open_count}/{max_pos}) — skipping new trade."
            )
        return allowed

    def check_daily_loss_limit(
        self,
        account_info: Dict[str, Any],
        daily_limit_percent: float = 5.0,
    ) -> bool:
        """
        Return True if it is safe to continue trading (daily loss within limit).

        Parameters
        ----------
        account_info : dict from MT5Connection.get_account_info()
        daily_limit_percent : float — percentage of balance as max daily loss
        """
        balance = account_info.get("balance", 0)
        equity  = account_info.get("equity", balance)
        if balance <= 0:
            return True  # can't compute, allow trading

        daily_loss_pct = (balance - equity) / balance * 100.0
        limit_hit = daily_loss_pct >= daily_limit_percent

        if limit_hit:
            logger.warning(
                f"Daily loss limit reached: {daily_loss_pct:.2f}% ≥ {daily_limit_percent}% "
                "— trading disabled for today."
            )
        return not limit_hit

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    def validate_trade(
        self,
        symbol: str,
        lot: float,
        sl: float,
        tp: float,
        symbol_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate trade parameters.

        Returns {"valid": True} or {"valid": False, "reason": "..."}
        """
        if lot <= 0:
            return {"valid": False, "reason": f"Invalid lot: {lot}"}

        min_lot = 0.01
        max_lot = 100.0
        if symbol_info:
            min_lot = symbol_info.get("volume_min", min_lot)
            max_lot = symbol_info.get("volume_max", max_lot)

        if lot < min_lot:
            return {"valid": False, "reason": f"Lot {lot} < min_lot {min_lot}"}
        if lot > max_lot:
            return {"valid": False, "reason": f"Lot {lot} > max_lot {max_lot}"}
        if sl <= 0:
            return {"valid": False, "reason": f"Invalid SL: {sl}"}
        if tp <= 0:
            return {"valid": False, "reason": f"Invalid TP: {tp}"}

        return {"valid": True}
