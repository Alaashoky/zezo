#!/usr/bin/env python3
"""
live_bot.py — ZEZO Live Trading Bot
====================================
Connects to MetaTrader 5, streams XAUUSD M15 data, runs the multi-strategy
brain + optional AI models, and executes trades with proper risk management.

Usage
-----
  python live_bot.py               # live mode
  python live_bot.py --dry-run     # observe without placing orders
  python live_bot.py --symbol XAUUSD --timeframe M15 --dry-run
"""

from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Logging — console + rotating file
# ---------------------------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(log_dir, f"live_bot_{datetime.now():%Y%m%d}.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("live_bot")

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from config.trading_config import TradingConfig
from mt5.connector import MT5Connection
from mt5.executor import TradeExecutor
from mt5.risk_manager import RiskManager
from strategies import (
    StrategyBrain, StrategyConfig, SignalType,
    MovingAverageCrossover, EMAcrossoverStrategy, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, MeanReversionStrategy, BreakoutStrategy,
    StochasticStrategy, SMCICTStrategy, ITS8OSStrategy,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a formatted section banner to the console."""
    line = "─" * 60
    logger.info(line)
    logger.info(f"  {text}")
    logger.info(line)


def _fmt_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# LiveTradingBot
# ---------------------------------------------------------------------------

class LiveTradingBot:
    """Main live trading bot that ties all components together."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self._running = False

        # --- MT5 connectivity ---------------------------------------------
        self.connector = MT5Connection(
            login=None,     # set via env vars MT5_LOGIN / MT5_PASSWORD / MT5_SERVER
            password=None,
            server=None,
        )

        # Override with environment variables when present
        login = os.environ.get("MT5_LOGIN")
        password = os.environ.get("MT5_PASSWORD")
        server = os.environ.get("MT5_SERVER")
        if login:
            self.connector.login = int(login)
        if password:
            self.connector.password = password
        if server:
            self.connector.server = server

        # --- Trade execution ----------------------------------------------
        self.executor = TradeExecutor(
            connection=self.connector,
            dry_run=config.dry_run,
        )

        # --- Risk management ----------------------------------------------
        self.risk_manager = RiskManager(config=config)

        # --- Strategy Brain -----------------------------------------------
        self.brain = StrategyBrain(config={
            "min_strategies_required": 2,
            "consensus_threshold": 0.6,
            "performance_weight": 0.4,
            "confidence_weight": 0.6,
        })
        self._register_strategies()

        # --- Optional AI predictor ----------------------------------------
        self.ai_predictor = self._load_ai_predictor()

        # --- Symbol info cache --------------------------------------------
        self._symbol_info: Optional[Dict[str, Any]] = None

        logger.info(
            f"LiveTradingBot initialised — symbol={config.symbol}, "
            f"dry_run={config.dry_run}"
        )

    # ------------------------------------------------------------------
    # Strategy registration (mirrors bot.py)
    # ------------------------------------------------------------------

    def _register_strategies(self) -> None:
        symbol    = self.config.symbol
        timeframe = "M15"

        strategies = [
            MovingAverageCrossover(StrategyConfig(name="MA_Crossover",   symbol=symbol, timeframe=timeframe)),
            EMAcrossoverStrategy( StrategyConfig(name="EMA_Crossover",  symbol=symbol, timeframe=timeframe)),
            RSIStrategy(          StrategyConfig(name="RSI",             symbol=symbol, timeframe=timeframe)),
            MACDStrategy(         StrategyConfig(name="MACD",            symbol=symbol, timeframe=timeframe)),
            BollingerBandsStrategy(StrategyConfig(name="Bollinger",      symbol=symbol, timeframe=timeframe)),
            MeanReversionStrategy(StrategyConfig(name="MeanReversion",   symbol=symbol, timeframe=timeframe)),
            BreakoutStrategy(     StrategyConfig(name="Breakout",        symbol=symbol, timeframe=timeframe)),
            StochasticStrategy(   StrategyConfig(name="Stochastic",      symbol=symbol, timeframe=timeframe)),
            SMCICTStrategy(       StrategyConfig(name="SMC_ICT",         symbol=symbol, timeframe=timeframe)),
            ITS8OSStrategy(       StrategyConfig(name="ITS8OS",          symbol=symbol, timeframe=timeframe)),
        ]

        for s in strategies:
            s.start()
            self.brain.register_strategy(s)

        logger.info(f"Registered {len(strategies)} strategies.")

    # ------------------------------------------------------------------
    # AI predictor
    # ------------------------------------------------------------------

    def _load_ai_predictor(self):
        model_dir = os.environ.get("MODEL_DIR", "saved_models")
        try:
            from models.market_predictor import MarketPredictor
            from config.model_config import ModelConfig
            predictor = MarketPredictor(config=ModelConfig(), model_dir=model_dir)
            predictor.load_models(model_dir)
            logger.info("AI MarketPredictor loaded ✓")
            return predictor
        except Exception as exc:
            logger.warning(f"AI models not available ({exc}) — strategy-only mode.")
            return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main trading loop."""
        _banner("ZEZO Live Trading Bot — Starting Up")

        # 1. Connect to MT5
        if not self.config.dry_run:
            if not self.connector.connect():
                logger.error("Failed to connect to MT5 — aborting.")
                return
        else:
            logger.info("[DRY-RUN] Skipping MT5 connection.")

        self._running = True

        # Prefetch symbol info
        if not self.config.dry_run:
            self._symbol_info = self.connector.get_symbol_info(self.config.symbol)

        _banner(f"Watching {self.config.symbol} — M15  [{'DRY-RUN' if self.config.dry_run else 'LIVE'}]")

        try:
            while self._running:
                self._tick()
                self._wait_for_new_candle(timeframe_minutes=15)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop()

    def _tick(self) -> None:
        """Execute one analysis/trade cycle."""
        logger.info(f"[{_fmt_now()}] ── New candle cycle ─────────────────────────────")

        # 1. Log account status
        self._log_status()

        # 2. Fetch OHLCV
        ohlcv = self._fetch_ohlcv()
        if ohlcv is None:
            logger.warning("No OHLCV data — skipping this cycle.")
            return

        # 3. Format data for strategies
        market_data = self._format_market_data(ohlcv)

        # 4. Run Strategy Brain
        result = self._analyze(market_data)

        # 5. Act on consensus
        if result and result.get("consensus_reached"):
            signal = result["consensus_signal"]
            self._on_signal(signal, market_data)
        else:
            reason = result.get("reason", "N/A") if result else "analysis failed"
            logger.info(f"No consensus this cycle: {reason}")

        # 6. Manage existing positions
        self._manage_positions()

    # ------------------------------------------------------------------
    # Data fetching & formatting
    # ------------------------------------------------------------------

    def _fetch_ohlcv(self) -> Optional[Dict[str, Any]]:
        if self.config.dry_run:
            # Return synthetic data in dry-run / CI so logic can be exercised
            n = self.config.candle_count
            base = 2000.0
            import random
            closes = [base + random.uniform(-10, 10) for _ in range(n)]
            highs  = [c + random.uniform(0, 5)  for c in closes]
            lows   = [c - random.uniform(0, 5)  for c in closes]
            return {
                "open":   closes,
                "high":   highs,
                "low":    lows,
                "close":  closes,
                "volume": [random.uniform(100, 500) for _ in range(n)],
                "prices": closes,
                "time":   [int(time.time()) - (n - i) * 900 for i in range(n)],
            }
        return self.connector.get_ohlcv(
            self.config.symbol,
            self.config.timeframe,
            self.config.candle_count,
        )

    def _format_market_data(self, ohlcv: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MT5 OHLCV dict to the format expected by BaseStrategy."""
        return {
            "prices": ohlcv.get("close", []),
            "close":  ohlcv.get("close", []),
            "open":   ohlcv.get("open",  []),
            "high":   ohlcv.get("high",  []),
            "low":    ohlcv.get("low",   []),
            "volume": ohlcv.get("volume",[]),
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run strategy brain (+ AI if available)."""
        try:
            if self.ai_predictor is not None:
                ai_result = None
                try:
                    import pandas as pd
                    df = pd.DataFrame(market_data)
                    ai_raw = self.ai_predictor.predict(df, symbol=self.config.symbol)
                    if ai_raw:
                        ai_result = ai_raw
                except Exception as exc:
                    logger.warning(f"AI prediction failed: {exc}")

                if ai_result:
                    return self.brain.analyze_with_ai(market_data, ai_result)

            return self.brain.analyze_joint(market_data)
        except Exception as exc:
            logger.error(f"Analysis error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _on_signal(self, signal, market_data: Dict[str, Any]) -> None:
        """Handle a consensus signal: risk checks → lot calc → execute."""
        sig_type = signal.signal_type

        logger.info(
            f"CONSENSUS SIGNAL: {sig_type.value}  "
            f"price={signal.price}  confidence={signal.confidence:.2f}"
        )

        if sig_type == SignalType.HOLD:
            return

        # --- Risk gate 1: max positions -----------------------------------
        positions = self._get_open_positions()
        if not self.risk_manager.can_open_trade(positions):
            return

        # --- Risk gate 2: daily loss limit --------------------------------
        account_info = self._get_account_info()
        if account_info and not self.risk_manager.check_daily_loss_limit(account_info):
            return

        # --- Lot size -----------------------------------------------------
        balance = account_info["balance"] if account_info else self.config.account_balance_ref
        lot = self.risk_manager.calculate_lot_size(
            symbol=self.config.symbol,
            account_balance=balance,
            risk_percent=self.config.risk_per_trade,
            sl_points=self.config.default_sl_points,
            symbol_info=self._symbol_info,
        )

        # --- SL / TP calculation ------------------------------------------
        entry_price = signal.price or self._get_current_price()
        if entry_price is None:
            logger.error("Cannot determine entry price — skipping signal.")
            return

        # Simple ATR from last 14 candles if available
        atr = self._calc_atr(market_data)

        levels = self.risk_manager.calculate_sl_tp(
            symbol=self.config.symbol,
            signal_type=sig_type.value,
            entry_price=entry_price,
            atr=atr,
            symbol_info=self._symbol_info,
        )
        sl = levels["sl"]
        tp = levels["tp"]

        # --- Validate trade -----------------------------------------------
        validation = self.risk_manager.validate_trade(
            self.config.symbol, lot, sl, tp, self._symbol_info
        )
        if not validation["valid"]:
            logger.error(f"Trade validation failed: {validation['reason']}")
            return

        # --- Execute ------------------------------------------------------
        if sig_type == SignalType.BUY:
            result = self.executor.open_buy(
                symbol=self.config.symbol,
                lot=lot, sl=sl, tp=tp,
                magic=self.config.magic_number,
                comment="ZEZO_BUY",
            )
        elif sig_type == SignalType.SELL:
            result = self.executor.open_sell(
                symbol=self.config.symbol,
                lot=lot, sl=sl, tp=tp,
                magic=self.config.magic_number,
                comment="ZEZO_SELL",
            )
        else:
            return

        if result.get("success"):
            logger.info(
                f"✅ Trade executed: {sig_type.value} {lot} lots {self.config.symbol}  "
                f"sl={sl}  tp={tp}  ticket={result.get('ticket')}"
            )
        else:
            logger.error(f"❌ Trade failed: {result.get('error')}")

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_positions(self) -> None:
        """Check open positions — apply basic trailing stop when in profit."""
        positions = self._get_open_positions()
        if not positions:
            return

        for pos in positions:
            try:
                current_tick = None
                if not self.config.dry_run:
                    current_tick = self.connector.get_current_price(pos.symbol)

                if current_tick is None:
                    continue

                point = (
                    self._symbol_info.get("point", 0.01)
                    if self._symbol_info else 0.01
                )
                trail_pts = self.config.default_sl_points * 0.5  # trail at 50% of SL
                trail_distance = trail_pts * point

                if pos.type == 0:  # BUY
                    price = current_tick["bid"]
                    new_sl = price - trail_distance
                    if new_sl > pos.sl + point:
                        self.executor.modify_position(pos, new_sl=new_sl, new_tp=pos.tp)
                        logger.info(f"Trailing stop moved to {new_sl:.5f} for ticket #{pos.ticket}")
                else:              # SELL
                    price = current_tick["ask"]
                    new_sl = price + trail_distance
                    if new_sl < pos.sl - point:
                        self.executor.modify_position(pos, new_sl=new_sl, new_tp=pos.tp)
                        logger.info(f"Trailing stop moved to {new_sl:.5f} for ticket #{pos.ticket}")
            except Exception as exc:
                logger.warning(f"Error managing position #{getattr(pos, 'ticket', '?')}: {exc}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_status(self) -> None:
        account_info = self._get_account_info()
        if account_info:
            logger.info(
                f"Account  balance={account_info['balance']:.2f}  "
                f"equity={account_info['equity']:.2f}  "
                f"free_margin={account_info['free_margin']:.2f}"
            )

        positions = self._get_open_positions()
        logger.info(f"Open positions: {len(positions)}")
        for p in positions:
            logger.info(
                f"  #{p.ticket}  {p.symbol}  "
                f"{'BUY' if p.type == 0 else 'SELL'}  "
                f"vol={p.volume}  profit={p.profit:.2f}"
            )

        stats = self.brain.get_statistics() if hasattr(self.brain, "get_statistics") else {}
        if stats:
            logger.info(f"Brain stats: {stats}")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _wait_for_new_candle(self, timeframe_minutes: int = 15) -> None:
        """Sleep until the start of the next M15 candle."""
        now = datetime.now(timezone.utc)
        minutes_past = now.minute % timeframe_minutes
        seconds_past = minutes_past * 60 + now.second

        # Add a small buffer (5 s) to ensure the candle is fully formed
        wait_seconds = (timeframe_minutes * 60 - seconds_past) + 5

        logger.info(f"Waiting {wait_seconds:.0f}s for next candle …")

        # Sleep in 10-second chunks so we can detect a stop request
        slept = 0
        while slept < wait_seconds and self._running:
            chunk = min(10, wait_seconds - slept)
            time.sleep(chunk)
            slept += chunk

            # Periodic connection check (every 5 minutes)
            if slept % 300 == 0 and not self.config.dry_run:
                if not self.connector.is_connected():
                    logger.warning("MT5 connection lost — reconnecting …")
                    self.connector.reconnect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_account_info(self) -> Optional[Dict[str, Any]]:
        if self.config.dry_run:
            return {
                "balance":     self.config.account_balance_ref,
                "equity":      self.config.account_balance_ref,
                "margin":      0.0,
                "free_margin": self.config.account_balance_ref,
                "profit":      0.0,
                "currency":    "USD",
            }
        return self.connector.get_account_info()

    def _get_open_positions(self):
        if self.config.dry_run:
            return []
        return self.connector.get_open_positions(
            symbol=self.config.symbol,
            magic=self.config.magic_number,
        )

    def _get_current_price(self) -> Optional[float]:
        if self.config.dry_run:
            return 2000.0
        tick = self.connector.get_current_price(self.config.symbol)
        return tick["ask"] if tick else None

    @staticmethod
    def _calc_atr(market_data: Dict[str, Any], period: int = 14) -> Optional[float]:
        """Simple ATR calculation from OHLCV dicts."""
        highs  = market_data.get("high",  [])
        lows   = market_data.get("low",   [])
        closes = market_data.get("close", [])
        if len(highs) < period + 1:
            return None
        trs = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i]  - closes[i - 1])
            trs.append(max(hl, hc, lc))
        return sum(trs[-period:]) / period if trs else None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Gracefully stop the bot and disconnect from MT5."""
        self._running = False
        if not self.config.dry_run:
            self.connector.disconnect()
        _banner("ZEZO Live Trading Bot — Stopped")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZEZO Live Trading Bot — XAUUSD M15"
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Trading symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--timeframe",
        default="M15",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        help="Timeframe (default: M15)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without placing real orders (log only)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = TradingConfig(
        symbol=args.symbol,
        dry_run=args.dry_run,
    )

    bot = LiveTradingBot(config)

    # Graceful Ctrl+C handling
    def _sigint_handler(sig, frame):
        logger.info("SIGINT received — stopping bot …")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    bot.run()


if __name__ == "__main__":
    main()
