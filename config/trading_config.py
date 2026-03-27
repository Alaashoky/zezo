"""
Trading configuration for the ZEZO live trading bot.
"""
from dataclasses import dataclass, field


@dataclass
class TradingConfig:
    # ── Symbol / Timeframe ──────────────────────────────────────────────────
    symbol: str = "XAUUSD"
    timeframe: int = 16388          # mt5.TIMEFRAME_M15  (filled at runtime)

    # ── Risk Management ─────────────────────────────────────────────────────
    risk_per_trade: float = 0.01    # 1 % of account balance per trade
    max_open_positions: int = 2
    account_balance_ref: float = 400.0  # reference balance ($400 demo)
    daily_loss_limit_percent: float = 5.0   # stop trading if daily loss > 5 %

    # ── Default SL / TP (points) ────────────────────────────────────────────
    default_sl_points: int = 200    # 200 points SL for gold
    default_tp_points: int = 400    # 400 points TP  → 2:1 reward/risk

    # ── Lot Size Limits ─────────────────────────────────────────────────────
    min_lot: float = 0.01
    max_lot: float = 0.10
    lot_step: float = 0.01

    # ── Order Parameters ────────────────────────────────────────────────────
    magic_number: int = field(default_factory=lambda: 20260327)
    slippage: int = 10              # maximum slippage in points

    # ── Bot Control ─────────────────────────────────────────────────────────
    trading_enabled: bool = True
    dry_run: bool = False           # when True: log only, no real orders

    # ── Data Feed ───────────────────────────────────────────────────────────
    candle_count: int = 100         # how many historical candles to fetch

    def __post_init__(self):
        # Import mt5 timeframe constant at runtime so the module can be imported
        # even when MetaTrader5 is not installed (e.g. during CI / dry-run).
        try:
            import MetaTrader5 as mt5
            self.timeframe = mt5.TIMEFRAME_M15
        except ImportError:
            self.timeframe = 16388  # numeric value of TIMEFRAME_M15
