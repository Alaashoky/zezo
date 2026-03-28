#!/usr/bin/env python3
"""
training/train_and_backtest.py — Full Training + Backtest Pipeline
====================================================================
Loads historical OHLCV data, splits it into train/validation/backtest
periods, trains all AI models, validates them, and then runs the
multi-strategy StrategyBrain over the out-of-sample backtest window.

Data split (XAUUSD M15, 2020 → present)
-----------------------------------------
  Training   : 2020-01-01 → 2024-06-30  (4.5 years)
  Validation : 2024-07-01 → 2025-06-30  (1 year)
  Backtest   : 2025-07-01 → present      (~9 months, out-of-sample)

Usage
-----
    # Use a pre-downloaded CSV
    python training/train_and_backtest.py --csv data/XAUUSD_M15_historical.csv

    # Download directly from MT5 (requires .env credentials)
    python training/train_and_backtest.py --from-mt5

    # Skip LSTM (faster, no GPU required)
    python training/train_and_backtest.py --csv data/XAUUSD_M15_historical.csv --skip-lstm
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_and_backtest")

# ---------------------------------------------------------------------------
# Data split boundaries
# ---------------------------------------------------------------------------
TRAIN_END = "2024-06-30"
VALID_END = "2025-06-30"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI models and run strategy backtest on historical data"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        metavar="FILE",
        help="Path to CSV file produced by download_mt5_data.py",
    )
    source.add_argument(
        "--from-mt5",
        action="store_true",
        help="Download data directly from MetaTrader 5 (requires .env credentials)",
    )
    parser.add_argument(
        "--symbol", default="XAUUSD", help="Symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--timeframe", default="M15", help="Timeframe (default: M15)"
    )
    parser.add_argument(
        "--start", default="2020-01-01", help="Start date when using --from-mt5"
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("MODEL_DIR", "saved_models"),
        help="Directory to save trained models (default: saved_models)",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training (faster; useful without a GPU)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Simulated starting capital for backtest (default: 10000)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV produced by download_mt5_data.py."""
    logger.info(f"Loading data from {path} …")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df.sort_index(inplace=True)
    logger.info(f"Loaded {len(df):,} rows — {df.index[0]} → {df.index[-1]}")
    return df


def download_from_mt5(symbol: str, timeframe: str, start: str) -> pd.DataFrame:
    """Download data directly from MT5 (reuses logic from download_mt5_data.py)."""
    # Import here to avoid hard dependency when using --csv
    from training.download_mt5_data import connect_mt5, download_data

    if not connect_mt5():
        sys.exit(1)

    try:
        import MetaTrader5 as mt5

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.now()
        df = download_data(symbol, timeframe, start_dt, end_dt)

        if df.empty:
            logger.error("No data downloaded from MT5.")
            sys.exit(1)

        return df
    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed.")


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, backtest) DataFrames."""
    train = df[df.index <= TRAIN_END]
    valid = df[(df.index > TRAIN_END) & (df.index <= VALID_END)]
    backtest = df[df.index > VALID_END]

    logger.info(
        f"Data split:\n"
        f"  Training   : {len(train):>7,} rows  {train.index[0].date()} → {train.index[-1].date()}\n"
        f"  Validation : {len(valid):>7,} rows  {valid.index[0].date()} → {valid.index[-1].date()}\n"
        f"  Backtest   : {len(backtest):>7,} rows  {backtest.index[0].date()} → {backtest.index[-1].date()}"
    )
    return train, valid, backtest


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    train_df: pd.DataFrame,
    model_dir: str,
    skip_lstm: bool,
) -> Tuple[Any, Dict[str, Any]]:
    """Train all models on *train_df* and save them. Returns (trainer, results)."""
    from training.trainer import Trainer

    logger.info("=" * 60)
    logger.info("PHASE 1 — TRAINING")
    logger.info("=" * 60)

    trainer = Trainer(model_dir=model_dir)
    results = trainer.train_all_models(train_df, skip_lstm=skip_lstm)
    trainer.save_all_models()
    trainer.print_summary()
    return trainer, results


# ---------------------------------------------------------------------------
# Validation (AI models)
# ---------------------------------------------------------------------------

def run_validation(
    trainer: Any,
    valid_df: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, Any]:
    """Run the trained predictor over the validation period."""
    from training.backtester import Backtester

    logger.info("=" * 60)
    logger.info("PHASE 2 — VALIDATION (AI models)")
    logger.info("=" * 60)

    predictor = trainer.get_market_predictor()
    backtester = Backtester(predictor=predictor, initial_capital=initial_capital)

    try:
        report = backtester.run(valid_df, symbol="XAUUSD")
        logger.info(backtester.format_report(report))
        return report
    except Exception as e:
        logger.warning(f"Validation backtest failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Strategy backtest (out-of-sample)
# ---------------------------------------------------------------------------

def run_strategy_backtest(
    backtest_df: pd.DataFrame,
    symbol: str,
    initial_capital: float,
) -> Tuple[Dict[str, Any], list]:
    """
    Run all 10 strategies via StrategyBrain on the out-of-sample backtest data
    and simulate a simple equity curve based on consensus signals.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 — STRATEGY BACKTEST (out-of-sample)")
    logger.info("=" * 60)

    from strategies import (
        StrategyBrain,
        StrategyConfig,
        SignalType,
        MovingAverageCrossover,
        EMAcrossoverStrategy,
        RSIStrategy,
        MACDStrategy,
        BollingerBandsStrategy,
        MeanReversionStrategy,
        BreakoutStrategy,
        StochasticStrategy,
        SMCICTStrategy,
        ITS8OSStrategy,
    )

    timeframe = "M15"
    brain = StrategyBrain(config={
        "min_strategies_required": 2,
        "consensus_threshold": 0.6,
        "performance_weight": 0.4,
        "confidence_weight": 0.6,
    })

    strategies = [
        MovingAverageCrossover(StrategyConfig(name="MA_Crossover", symbol=symbol, timeframe=timeframe)),
        EMAcrossoverStrategy(StrategyConfig(name="EMA_Crossover", symbol=symbol, timeframe=timeframe)),
        RSIStrategy(StrategyConfig(name="RSI", symbol=symbol, timeframe=timeframe)),
        MACDStrategy(StrategyConfig(name="MACD", symbol=symbol, timeframe=timeframe)),
        BollingerBandsStrategy(StrategyConfig(name="Bollinger", symbol=symbol, timeframe=timeframe)),
        MeanReversionStrategy(StrategyConfig(name="MeanReversion", symbol=symbol, timeframe=timeframe)),
        BreakoutStrategy(StrategyConfig(name="Breakout", symbol=symbol, timeframe=timeframe)),
        StochasticStrategy(StrategyConfig(name="Stochastic", symbol=symbol, timeframe=timeframe)),
        SMCICTStrategy(StrategyConfig(name="SMC_ICT", symbol=symbol, timeframe=timeframe)),
        ITS8OSStrategy(StrategyConfig(name="ITS8OS", symbol=symbol, timeframe=timeframe)),
    ]

    for s in strategies:
        s.start()
        brain.register_strategy(s)

    # Walk-forward simulation over the backtest period
    close = backtest_df["close"].values
    equity = initial_capital
    equity_curve = [equity]
    trade_returns = []
    signals_log = []

    min_history = 50  # warm-up candles

    for i in range(min_history, len(backtest_df) - 1):
        window = backtest_df.iloc[: i + 1]
        try:
            result = brain.analyze(window)
        except Exception:
            result = None

        direction = 0
        signal_label = "HOLD"

        if result and result.get("signal"):
            sig = result.get("signal_type")
            if sig == SignalType.BUY or str(sig) in ("BUY", "SignalType.BUY"):
                direction = 1
                signal_label = "BUY"
            elif sig == SignalType.SELL or str(sig) in ("SELL", "SignalType.SELL"):
                direction = -1
                signal_label = "SELL"

        actual_return = (close[i + 1] - close[i]) / close[i]
        trade_ret = direction * actual_return
        trade_returns.append(trade_ret)
        equity *= 1 + trade_ret
        equity_curve.append(equity)

        signals_log.append(
            {
                "datetime": backtest_df.index[i],
                "signal": signal_label,
                "close": close[i],
                "actual_return": actual_return,
                "trade_return": trade_ret,
                "equity": equity,
            }
        )

    equity_arr = np.array(equity_curve)
    trade_arr = np.array(trade_returns)

    # win / loss
    trades = [t for t in trade_returns if t != 0]
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_dd = float(drawdown.min()) * 100

    total_return = (equity_arr[-1] - initial_capital) / initial_capital * 100
    sharpe = (
        float(np.mean(trade_arr) / np.std(trade_arr) * np.sqrt(252 * 96))
        if np.std(trade_arr) > 0
        else 0.0
    )  # 96 M15 candles per day

    # per-strategy performance (from brain stats)
    brain_stats = brain.get_stats()

    report = {
        "symbol": symbol,
        "period_start": str(backtest_df.index[min_history].date()),
        "period_end": str(backtest_df.index[-1].date()),
        "total_candles": len(backtest_df),
        "simulated_candles": len(trade_returns),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "total_return_pct": float(total_return),
        "initial_capital": initial_capital,
        "final_equity": float(equity_arr[-1]),
        "sharpe_ratio": sharpe,
        "brain_stats": brain_stats,
    }

    return report, signals_log


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_strategy_report(report: Dict[str, Any]) -> str:
    pf = report["profit_factor"]
    pf_str = f"{pf:.3f}" if pf != float("inf") else "∞"
    lines = [
        "",
        "=" * 60,
        f"STRATEGY BACKTEST REPORT — {report['symbol']}",
        "=" * 60,
        f"Period          : {report['period_start']} → {report['period_end']}",
        f"Candles tested  : {report['simulated_candles']:,} / {report['total_candles']:,}",
        "",
        "Trade statistics:",
        f"  Total trades  : {report['total_trades']:,}",
        f"  Wins / Losses : {report['wins']} / {report['losses']}",
        f"  Win rate      : {report['win_rate']:.2%}",
        f"  Profit factor : {pf_str}",
        "",
        "Portfolio performance:",
        f"  Initial capital : ${report['initial_capital']:,.2f}",
        f"  Final equity    : ${report['final_equity']:,.2f}",
        f"  Total return    : {report['total_return_pct']:+.2f}%",
        f"  Sharpe ratio    : {report['sharpe_ratio']:.3f}",
        f"  Max drawdown    : {report['max_drawdown_pct']:.2f}%",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Step 1: Load data ────────────────────────────────────────────────
    if args.from_mt5:
        df = download_from_mt5(args.symbol, args.timeframe, args.start)
    else:
        df = load_csv(args.csv)

    # ── Step 2: Split data ───────────────────────────────────────────────
    train_df, valid_df, backtest_df = split_data(df)

    if len(train_df) < 200:
        logger.error("Training set too small (< 200 candles). Check your data.")
        sys.exit(1)

    # ── Step 3: Train AI models ──────────────────────────────────────────
    trainer, train_results = run_training(train_df, args.model_dir, args.skip_lstm)

    # ── Step 4: Validate on validation period ────────────────────────────
    if len(valid_df) >= 102:
        val_report = run_validation(trainer, valid_df, args.initial_capital)
    else:
        logger.warning("Validation set too small — skipping validation phase.")
        val_report = {}

    # ── Step 5: Strategy backtest (out-of-sample) ────────────────────────
    if len(backtest_df) >= 52:
        strategy_report, signals = run_strategy_backtest(
            backtest_df, args.symbol, args.initial_capital
        )
        logger.info(format_strategy_report(strategy_report))
    else:
        logger.warning("Backtest set too small — skipping strategy backtest phase.")
        strategy_report = {}
        signals = []

    # ── Step 6: Summary ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    if val_report:
        logger.info(
            f"Validation  — accuracy: {val_report.get('accuracy', 0):.2%}, "
            f"return: {val_report.get('total_return_pct', 0):+.2f}%"
        )
    if strategy_report:
        logger.info(
            f"Backtest    — win rate: {strategy_report.get('win_rate', 0):.2%}, "
            f"return: {strategy_report.get('total_return_pct', 0):+.2f}%, "
            f"max DD: {strategy_report.get('max_drawdown_pct', 0):.2f}%"
        )
    logger.info(f"Models saved → {args.model_dir}/")


if __name__ == "__main__":
    main()
