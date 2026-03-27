"""
TradeExecutor — places, modifies and closes MetaTrader 5 orders.

All public methods return a dict:
  {"success": bool, "ticket": int|None, "error": str|None, ...}
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed — TradeExecutor will run in stub mode.")


class TradeExecutor:
    """Sends trade requests to MetaTrader 5."""

    def __init__(self, connection=None, dry_run: bool = False):
        """
        Parameters
        ----------
        connection : MT5Connection
            Active MT5Connection instance (used only for connectivity checks).
        dry_run : bool
            When True every method logs the intended action and returns a
            synthetic success result WITHOUT sending any order to MT5.
        """
        self.connection = connection
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Open orders
    # ------------------------------------------------------------------

    def open_buy(
        self,
        symbol: str,
        lot: float,
        sl: float,
        tp: float,
        magic: int = 0,
        comment: str = "ZEZO_BUY",
    ) -> Dict[str, Any]:
        """Place a market buy order."""
        return self._open_order(
            symbol=symbol, lot=lot, sl=sl, tp=tp,
            order_type=mt5.ORDER_TYPE_BUY if MT5_AVAILABLE else 0,
            magic=magic, comment=comment,
        )

    def open_sell(
        self,
        symbol: str,
        lot: float,
        sl: float,
        tp: float,
        magic: int = 0,
        comment: str = "ZEZO_SELL",
    ) -> Dict[str, Any]:
        """Place a market sell order."""
        return self._open_order(
            symbol=symbol, lot=lot, sl=sl, tp=tp,
            order_type=mt5.ORDER_TYPE_SELL if MT5_AVAILABLE else 1,
            magic=magic, comment=comment,
        )

    # ------------------------------------------------------------------
    # Close orders
    # ------------------------------------------------------------------

    def close_position(self, position) -> Dict[str, Any]:
        """Close a single open position (passed as MT5 TradePosition)."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would close position #{position.ticket}")
            return {"success": True, "ticket": position.ticket, "dry_run": True}

        if not MT5_AVAILABLE:
            return {"success": False, "error": "MetaTrader5 not installed"}

        try:
            # Opposite order type to close
            close_type = (
                mt5.ORDER_TYPE_SELL
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"success": False, "error": "Cannot get current price"}

            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action":   mt5.TRADE_ACTION_DEAL,
                "symbol":   position.symbol,
                "volume":   position.volume,
                "type":     close_type,
                "position": position.ticket,
                "price":    price,
                "deviation": 10,
                "magic":    position.magic,
                "comment":  "ZEZO_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            return self._process_result(result, "close_position")
        except Exception as exc:
            logger.error(f"close_position exception: {exc}")
            return {"success": False, "error": str(exc)}

    def close_all_positions(
        self, symbol: Optional[str] = None, magic: Optional[int] = None
    ) -> Dict[str, Any]:
        """Close all open positions matching symbol and/or magic."""
        if not MT5_AVAILABLE and not self.dry_run:
            return {"success": False, "error": "MetaTrader5 not installed"}

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would close all positions — symbol={symbol}, magic={magic}")
            return {"success": True, "closed": 0, "dry_run": True}

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                positions = []

            if magic is not None:
                positions = [p for p in positions if p.magic == magic]

            closed = 0
            errors = []
            for pos in positions:
                res = self.close_position(pos)
                if res.get("success"):
                    closed += 1
                else:
                    errors.append(res.get("error"))

            return {"success": len(errors) == 0, "closed": closed, "errors": errors}
        except Exception as exc:
            logger.error(f"close_all_positions exception: {exc}")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Modify order
    # ------------------------------------------------------------------

    def modify_position(
        self, position, new_sl: float, new_tp: float
    ) -> Dict[str, Any]:
        """Modify the SL and TP of an open position."""
        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would modify #{position.ticket}: sl={new_sl}, tp={new_tp}"
            )
            return {"success": True, "ticket": position.ticket, "dry_run": True}

        if not MT5_AVAILABLE:
            return {"success": False, "error": "MetaTrader5 not installed"}

        try:
            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   position.symbol,
                "position": position.ticket,
                "sl":       new_sl,
                "tp":       new_tp,
            }
            result = mt5.order_send(request)
            return self._process_result(result, "modify_position")
        except Exception as exc:
            logger.error(f"modify_position exception: {exc}")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_order(
        self,
        symbol: str,
        lot: float,
        sl: float,
        tp: float,
        order_type: int,
        magic: int,
        comment: str,
    ) -> Dict[str, Any]:
        """Common logic for buy and sell market orders."""
        direction = "BUY" if order_type == (mt5.ORDER_TYPE_BUY if MT5_AVAILABLE else 0) else "SELL"

        # Validate parameters
        validation = self._validate_order(symbol, lot, sl, tp)
        if not validation["valid"]:
            logger.error(f"Order validation failed: {validation['reason']}")
            return {"success": False, "error": validation["reason"]}

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would open {direction} {lot} lots {symbol} "
                f"sl={sl:.5f} tp={tp:.5f} magic={magic}"
            )
            return {
                "success": True, "ticket": 0, "direction": direction,
                "lot": lot, "sl": sl, "tp": tp, "dry_run": True,
            }

        if not MT5_AVAILABLE:
            return {"success": False, "error": "MetaTrader5 not installed"}

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": f"Cannot get price for {symbol}"}

            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

            request = {
                "action":     mt5.TRADE_ACTION_DEAL,
                "symbol":     symbol,
                "volume":     lot,
                "type":       order_type,
                "price":      price,
                "sl":         sl,
                "tp":         tp,
                "deviation":  10,
                "magic":      magic,
                "comment":    comment,
                "type_time":  mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            return self._process_result(result, f"open_{direction.lower()}")
        except Exception as exc:
            logger.error(f"_open_order exception: {exc}")
            return {"success": False, "error": str(exc)}

    def _validate_order(
        self, symbol: str, lot: float, sl: float, tp: float
    ) -> Dict[str, Any]:
        """Basic parameter validation (does not require MT5 connection)."""
        if lot <= 0:
            return {"valid": False, "reason": f"Invalid lot size: {lot}"}
        if sl < 0 or tp < 0:
            return {"valid": False, "reason": f"SL/TP must be non-negative: sl={sl}, tp={tp}"}
        if not symbol:
            return {"valid": False, "reason": "Symbol is empty"}
        return {"valid": True}

    @staticmethod
    def _process_result(result, action: str) -> Dict[str, Any]:
        """Convert an mt5.order_send result into a standardised dict."""
        if result is None:
            error = mt5.last_error() if MT5_AVAILABLE else "unknown"
            logger.error(f"{action} — order_send returned None: {error}")
            return {"success": False, "error": str(error)}

        if result.retcode == (mt5.TRADE_RETCODE_DONE if MT5_AVAILABLE else 10009):
            logger.info(f"{action} succeeded — ticket #{result.order}")
            return {
                "success": True,
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume,
                "comment": result.comment,
            }
        else:
            logger.error(
                f"{action} failed — retcode={result.retcode} comment={result.comment}"
            )
            return {
                "success": False,
                "retcode": result.retcode,
                "error": result.comment,
            }
