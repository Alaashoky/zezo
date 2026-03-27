"""
MT5Connection — wrapper around the MetaTrader5 Python API.

Handles:
  • Initialisation and login
  • Account / symbol information
  • OHLCV data retrieval
  • Current bid/ask prices
  • Open position queries
  • Automatic reconnection on failure
"""

import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MT5 import — allow the module to load even when MetaTrader5 is
# not installed (e.g. CI, dry-run on Linux).
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not found — MT5Connection will run in stub mode.")


class MT5Connection:
    """Manages a connection to a MetaTrader 5 terminal."""

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise MT5 and optionally log in."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 library is not installed.")
            return False

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {}
                if self.path:
                    kwargs["path"] = self.path
                if self.login:
                    kwargs["login"] = self.login
                if self.password:
                    kwargs["password"] = self.password
                if self.server:
                    kwargs["server"] = self.server

                if not mt5.initialize(**kwargs):
                    error = mt5.last_error()
                    logger.warning(f"MT5 init attempt {attempt} failed: {error}")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                    continue

                self._connected = True
                info = mt5.terminal_info()
                logger.info(f"Connected to MT5 terminal: {info.name if info else 'unknown'}")
                return True

            except Exception as exc:
                logger.error(f"Exception during MT5 connect (attempt {attempt}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        self._connected = False
        return False

    def disconnect(self) -> None:
        """Shutdown the MT5 connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def is_connected(self) -> bool:
        """Return True if the terminal is connected and ready."""
        if not MT5_AVAILABLE or not self._connected:
            return False
        try:
            info = mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    def reconnect(self) -> bool:
        """Attempt to restore a lost connection."""
        logger.info("Attempting MT5 reconnection …")
        self.disconnect()
        time.sleep(self.retry_delay)
        return self.connect()

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Return account balance, equity, margin and free margin."""
        if not self._check_connection():
            return None
        try:
            info = mt5.account_info()
            if info is None:
                logger.error(f"get_account_info failed: {mt5.last_error()}")
                return None
            return {
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "free_margin": info.margin_free,
                "profit": info.profit,
                "currency": info.currency,
                "leverage": info.leverage,
                "login": info.login,
                "server": info.server,
            }
        except Exception as exc:
            logger.error(f"get_account_info exception: {exc}")
            return None

    # ------------------------------------------------------------------
    # Symbol information
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return symbol details required for lot/SL/TP calculations."""
        if not self._check_connection():
            return None
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Symbol '{symbol}' not found: {mt5.last_error()}")
                return None
            return {
                "name": info.name,
                "point": info.point,
                "digits": info.digits,
                "trade_tick_size": info.trade_tick_size,
                "trade_tick_value": info.trade_tick_value,
                "trade_contract_size": info.trade_contract_size,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step,
                "spread": info.spread,
            }
        except Exception as exc:
            logger.error(f"get_symbol_info exception: {exc}")
            return None

    # ------------------------------------------------------------------
    # OHLCV data
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: int,
        count: int = 100,
    ) -> Optional[Dict[str, List[float]]]:
        """
        Fetch the last *count* closed candles for *symbol* / *timeframe*.

        Returns a dict with keys: open, high, low, close, volume, prices
        where *prices* is an alias for close (kept for strategy compatibility).
        Returns None on failure.
        """
        if not self._check_connection():
            return None
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.error(f"copy_rates_from_pos failed: {mt5.last_error()}")
                return None

            # rates is a numpy structured array
            closes = [float(r["close"]) for r in rates]
            return {
                "open":   [float(r["open"])   for r in rates],
                "high":   [float(r["high"])   for r in rates],
                "low":    [float(r["low"])    for r in rates],
                "close":  closes,
                "volume": [float(r["tick_volume"]) for r in rates],
                "prices": closes,               # strategy compatibility alias
                "time":   [int(r["time"])      for r in rates],
            }
        except Exception as exc:
            logger.error(f"get_ohlcv exception: {exc}")
            return None

    # ------------------------------------------------------------------
    # Current price
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Return the current bid and ask for *symbol*."""
        if not self._check_connection():
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"symbol_info_tick failed: {mt5.last_error()}")
                return None
            return {"bid": tick.bid, "ask": tick.ask, "last": tick.last, "time": tick.time}
        except Exception as exc:
            logger.error(f"get_current_price exception: {exc}")
            return None

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> List[Any]:
        """
        Return a list of open positions filtered by *symbol* and/or *magic*.
        Each element is a raw MT5 TradePosition named-tuple.
        """
        if not self._check_connection():
            return []
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            if magic is not None:
                positions = [p for p in positions if p.magic == magic]

            return list(positions)
        except Exception as exc:
            logger.error(f"get_open_positions exception: {exc}")
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_connection(self) -> bool:
        """Ensure we are connected, try to reconnect once if not."""
        if self.is_connected():
            return True
        logger.warning("MT5 not connected — attempting reconnect …")
        return self.reconnect()
