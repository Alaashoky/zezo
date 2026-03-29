#!/usr/bin/env python3
"""
training/backtest.py — Strategy & AI Backtesting Pipeline
==========================================================
Loads saved AI models and runs walk-forward backtests over the
out-of-sample backtest period.  Supports four modes:

  strategy-only  — 10 strategies via StrategyBrain only (no AI)
  ai-only        — AI ensemble (LSTM + RF + XGB) only
  combined       — AI + strategies blended via brain.analyze_with_ai()
  compare        — Run all three modes and print a side-by-side table

Usage
-----
    # Strategy-only mode
    python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \\
        --mode strategy-only --initial-capital 400

    # AI-only mode
    python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \\
        --mode ai-only --initial-capital 400

    # Combined AI + strategies (default mode)
    python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \\
        --mode combined --initial-capital 400 --ai-weight 0.5

    # Compare all three modes side by side
    python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \\
        --mode compare --initial-capital 400
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("training.backtest")

# Mapping: strategy config name → feature engineering column suffix
# (mirrors models/feature_engineering.py STRATEGY_NAME_MAP)
_STRATEGY_NAME_MAP = {
    "MA_Crossover":  "ma_crossover",
    "EMA_Crossover": "ema_crossover",
    "RSI":           "rsi",
    "MACD":          "macd",
    "Bollinger":     "bollinger",
    "MeanReversion": "mean_reversion",
    "Breakout":      "breakout",
    "Stochastic":    "stochastic",
    "SMC_ICT":       "smc_ict",
    "ITS8OS":        "its8os",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest AI models and/or strategies on historical data"
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
    parser.add_argument("--symbol",    default="XAUUSD", help="Symbol (default: XAUUSD)")
    parser.add_argument("--timeframe", default="M15",    help="Timeframe (default: M15)")
    parser.add_argument("--start",     default="2020-01-01",
                        help="Start date when using --from-mt5")
    parser.add_argument(
        "--mode",
        choices=["strategy-only", "ai-only", "combined", "compare"],
        default="combined",
        help="Backtest mode (default: combined)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Simulated starting capital (default: 10000)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("MODEL_DIR", "saved_models"),
        help="Directory with saved models (default: saved_models)",
    )
    parser.add_argument(
        "--ai-weight",
        type=float,
        default=0.3,
        help="AI signal weight in combined mode (default: 0.3)",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.6,
        help="StrategyBrain consensus threshold (default: 0.6)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum AI confidence to enter a trade (default: 0.5)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# StrategyBrain builder (shared between modes)
# ---------------------------------------------------------------------------

def _build_brain(symbol: str, timeframe: str, consensus_threshold: float):
    """Create and start a fresh StrategyBrain with all 10 strategies."""
    from strategies import (
        StrategyBrain, StrategyConfig,
        MovingAverageCrossover, EMAcrossoverStrategy, RSIStrategy,
        MACDStrategy, BollingerBandsStrategy, MeanReversionStrategy,
        BreakoutStrategy, StochasticStrategy, SMCICTStrategy, ITS8OSStrategy,
    )

    brain = StrategyBrain(config={
        "min_strategies_required": 2,
        "consensus_threshold": consensus_threshold,
        "performance_weight": 0.4,
        "confidence_weight": 0.6,
    })

    strategies = [
        MovingAverageCrossover(StrategyConfig("MA_Crossover",  symbol, timeframe)),
        EMAcrossoverStrategy( StrategyConfig("EMA_Crossover", symbol, timeframe)),
        RSIStrategy(          StrategyConfig("RSI",           symbol, timeframe)),
        MACDStrategy(         StrategyConfig("MACD",          symbol, timeframe)),
        BollingerBandsStrategy(StrategyConfig("Bollinger",    symbol, timeframe)),
        MeanReversionStrategy(StrategyConfig("MeanReversion", symbol, timeframe)),
        BreakoutStrategy(     StrategyConfig("Breakout",      symbol, timeframe)),
        StochasticStrategy(   StrategyConfig("Stochastic",    symbol, timeframe)),
        SMCICTStrategy(       StrategyConfig("SMC_ICT",       symbol, timeframe)),
        ITS8OSStrategy(       StrategyConfig("ITS8OS",        symbol, timeframe)),
    ]
    for s in strategies:
        s.start()
        brain.register_strategy(s)

    return brain


def _extract_strategy_signals(brain, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract individual strategy signals from StrategyBrain's last signal_history
    entry and build the dict needed for model.predict(strategy_signals=...).
    """
    from strategies import SignalType

    signals: Dict[str, Any] = {}

    if brain.signal_history:
        raw_signals = brain.signal_history[-1].get("strategy_signals", {})
        buy_count = 0
        sell_count = 0
        for config_name, feat_suffix in _STRATEGY_NAME_MAP.items():
            sig = raw_signals.get(config_name)
            col = f"strategy_{feat_suffix}"
            if sig is not None:
                if sig.signal_type == SignalType.BUY:
                    signals[col] = 1
                    buy_count += 1
                elif sig.signal_type == SignalType.SELL:
                    signals[col] = -1
                    sell_count += 1
                else:
                    signals[col] = 0
            else:
                signals[col] = 0

        signals["strategy_buy_count"]  = buy_count
        signals["strategy_sell_count"] = sell_count

    # Consensus
    if result.get("consensus_reached") and result.get("consensus_signal"):
        from strategies import SignalType
        sig = result["consensus_signal"]
        signals["strategy_consensus"] = (
            1 if sig.signal_type == SignalType.BUY else
            -1 if sig.signal_type == SignalType.SELL else 0
        )
        signals["strategy_consensus_confidence"] = float(sig.confidence)
    else:
        signals["strategy_consensus"] = 0
        signals["strategy_consensus_confidence"] = 0.0

    return signals


# ---------------------------------------------------------------------------
# Equity simulation helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    trade_returns: List[float],
    equity_curve: List[float],
    initial_capital: float,
    backtest_df: pd.DataFrame,
    min_history: int,
) -> Dict[str, Any]:
    """Compute backtest performance metrics from trade returns and equity curve."""
    equity_arr = np.array(equity_curve)
    trade_arr = np.array(trade_returns)

    trades = [t for t in trade_returns if t != 0.0]
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_dd = float(drawdown.min()) * 100

    total_return = (equity_arr[-1] - initial_capital) / initial_capital * 100
    sharpe = (
        float(np.mean(trade_arr) / np.std(trade_arr) * np.sqrt(252 * 96))
        if np.std(trade_arr) > 0
        else 0.0
    )

    return {
        "period_start": str(backtest_df.index[min_history].date()),
        "period_end":   str(backtest_df.index[-1].date()),
        "total_candles": len(backtest_df),
        "simulated_candles": len(trade_returns),
        "total_trades":  len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "total_return_pct": float(total_return),
        "initial_capital":  initial_capital,
        "final_equity":     float(equity_arr[-1]),
        "sharpe_ratio":     sharpe,
    }


# ---------------------------------------------------------------------------
# Mode: strategy-only
# ---------------------------------------------------------------------------

def run_strategy_only(
    backtest_df: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    consensus_threshold: float,
) -> Dict[str, Any]:
    """Run all 10 strategies via StrategyBrain — no AI involved."""
    from strategies import SignalType

    logger.info("=" * 60)
    logger.info("STRATEGY-ONLY BACKTEST")
    logger.info("=" * 60)

    brain = _build_brain(symbol, "M15", consensus_threshold)

    close = backtest_df["close"].values
    equity = initial_capital
    equity_curve = [equity]
    trade_returns: List[float] = []

    min_history = 50

    cols = ["open", "high", "low", "close"]
    sub = backtest_df[cols].copy()
    sub["volume"] = backtest_df["volume"].values if "volume" in backtest_df.columns else 0.0
    all_bars = sub[["open", "high", "low", "close", "volume"]].astype(float).to_dict("records")

    for i in range(min_history, len(backtest_df) - 1):
        bar_dict = {
            **all_bars[i],
            "symbol": symbol,
            "prices": all_bars[: i + 1],
        }
        try:
            result = brain.analyze_joint(bar_dict)
        except Exception:
            result = None

        direction = 0
        if result and result.get("consensus_reached") and result.get("consensus_signal"):
            sig = result["consensus_signal"]
            if sig.signal_type == SignalType.BUY:
                direction = 1
            elif sig.signal_type == SignalType.SELL:
                direction = -1

        actual_return = (close[i + 1] - close[i]) / close[i]
        trade_ret = direction * actual_return
        trade_returns.append(trade_ret)
        equity *= 1 + trade_ret
        equity_curve.append(equity)

    report = _compute_metrics(trade_returns, equity_curve, initial_capital, backtest_df, min_history)
    report["symbol"] = symbol
    report["mode"] = "strategy-only"
    return report


# ---------------------------------------------------------------------------
# Mode: ai-only
# ---------------------------------------------------------------------------

def run_ai_only(
    backtest_df: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    predictor,
    min_confidence: float,
) -> Dict[str, Any]:
    """Run the AI ensemble only (uses training/backtester.py internally)."""
    from training.backtester import Backtester

    logger.info("=" * 60)
    logger.info("AI-ONLY BACKTEST")
    logger.info("=" * 60)

    backtester = Backtester(predictor=predictor, initial_capital=initial_capital)
    try:
        bt_report = backtester.run(backtest_df, symbol=symbol)
        logger.info(backtester.format_report(bt_report))
    except Exception as e:
        logger.error(f"AI-only backtest failed: {e}")
        return {"mode": "ai-only", "error": str(e)}

    # Map backtester report keys to our standard format
    report: Dict[str, Any] = {
        "symbol":            symbol,
        "mode":              "ai-only",
        "period_start":      str(backtest_df.index[100].date()),
        "period_end":        str(backtest_df.index[-1].date()),
        "total_candles":     bt_report.get("total_candles", 0),
        "simulated_candles": bt_report.get("backtested_candles", 0),
        "total_trades":      bt_report["signal_distribution"]["buy"] + bt_report["signal_distribution"]["sell"],
        "wins":              0,   # Backtester doesn't compute this separately
        "losses":            0,
        "win_rate":          bt_report.get("accuracy", 0.0),
        "profit_factor":     float("inf"),
        "max_drawdown_pct":  bt_report.get("max_drawdown_pct", 0.0),
        "total_return_pct":  bt_report.get("total_return_pct", 0.0),
        "initial_capital":   initial_capital,
        "final_equity":      bt_report.get("final_equity", initial_capital),
        "sharpe_ratio":      bt_report.get("sharpe_ratio", 0.0),
    }
    return report


# ---------------------------------------------------------------------------
# Mode: combined (AI + strategies)
# ---------------------------------------------------------------------------

def run_combined(
    backtest_df: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    predictor,
    consensus_threshold: float,
    min_confidence: float,
) -> Dict[str, Any]:
    """Blend AI predictions with strategy consensus — mirrors live_bot.py behaviour."""
    from strategies import SignalType

    logger.info("=" * 60)
    logger.info("COMBINED (AI + STRATEGIES) BACKTEST")
    logger.info("=" * 60)

    brain = _build_brain(symbol, "M15", consensus_threshold)

    close = backtest_df["close"].values
    equity = initial_capital
    equity_curve = [equity]
    trade_returns: List[float] = []

    min_history = 50

    cols = ["open", "high", "low", "close"]
    sub = backtest_df[cols].copy()
    sub["volume"] = backtest_df["volume"].values if "volume" in backtest_df.columns else 0.0
    all_bars = sub[["open", "high", "low", "close", "volume"]].astype(float).to_dict("records")

    for i in range(min_history, len(backtest_df) - 1):
        bar_dict = {
            **all_bars[i],
            "symbol": symbol,
            "prices": all_bars[: i + 1],
        }

        # Run strategies first
        try:
            strat_result = brain.analyze_joint(bar_dict)
        except Exception:
            strat_result = None

        # Extract strategy signals for AI features
        strategy_signals = {}
        if strat_result is not None:
            strategy_signals = _extract_strategy_signals(brain, strat_result)

        # Get AI prediction (passing current strategy signals as extra features)
        ai_result = None
        if predictor is not None:
            try:
                window = backtest_df.iloc[: i + 1]
                ai_result = predictor.predict(
                    window,
                    symbol=symbol,
                    strategy_signals=strategy_signals if strategy_signals else None,
                )
            except Exception as exc:
                logger.debug("AI prediction failed at bar %d: %s", i, exc)

        # Blend: combined = AI + strategies via analyze_with_ai
        try:
            if ai_result is not None:
                result = brain.analyze_with_ai(bar_dict, ai_result)
            else:
                result = strat_result
        except Exception:
            result = strat_result

        direction = 0
        if result and result.get("consensus_reached") and result.get("consensus_signal"):
            sig = result["consensus_signal"]
            if sig.signal_type == SignalType.BUY:
                direction = 1
            elif sig.signal_type == SignalType.SELL:
                direction = -1

        actual_return = (close[i + 1] - close[i]) / close[i]
        trade_ret = direction * actual_return
        trade_returns.append(trade_ret)
        equity *= 1 + trade_ret
        equity_curve.append(equity)

    report = _compute_metrics(trade_returns, equity_curve, initial_capital, backtest_df, min_history)
    report["symbol"] = symbol
    report["mode"] = "combined"
    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(report: Dict[str, Any]) -> str:
    symbol = report.get("symbol", "N/A")
    mode   = report.get("mode", "N/A")
    pf     = report.get("profit_factor", 0.0)
    pf_str = f"{pf:.3f}" if pf != float("inf") else "∞"
    lines = [
        "",
        "=" * 60,
        f"BACKTEST REPORT — {symbol}  [{mode.upper()}]",
        "=" * 60,
        f"Period          : {report.get('period_start', '?')} → {report.get('period_end', '?')}",
        f"Candles tested  : {report.get('simulated_candles', 0):,} / {report.get('total_candles', 0):,}",
        "",
        "Trade statistics:",
        f"  Total trades  : {report.get('total_trades', 0):,}",
        f"  Wins / Losses : {report.get('wins', 0)} / {report.get('losses', 0)}",
        f"  Win rate      : {report.get('win_rate', 0):.2%}",
        f"  Profit factor : {pf_str}",
        "",
        "Portfolio performance:",
        f"  Initial capital : ${report.get('initial_capital', 0):,.2f}",
        f"  Final equity    : ${report.get('final_equity', 0):,.2f}",
        f"  Total return    : {report.get('total_return_pct', 0):+.2f}%",
        f"  Sharpe ratio    : {report.get('sharpe_ratio', 0):.3f}",
        f"  Max drawdown    : {report.get('max_drawdown_pct', 0):.2f}%",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


def format_comparison_table(reports: Dict[str, Dict[str, Any]]) -> str:
    """Print a side-by-side comparison of multiple backtest modes."""
    modes = list(reports.keys())
    col_w = 18

    def row(label: str, *values) -> str:
        return f"  {label:<24}" + "".join(f"{str(v):>{col_w}}" for v in values)

    header = "  " + " " * 24 + "".join(f"{m.upper():>{col_w}}" for m in modes)
    sep = "  " + "-" * (24 + col_w * len(modes))

    def pf_str(r):
        pf = r.get("profit_factor", 0)
        return "∞" if pf == float("inf") else f"{pf:.3f}"

    lines = [
        "",
        "=" * (26 + col_w * len(modes)),
        "BACKTEST COMPARISON",
        "=" * (26 + col_w * len(modes)),
        header,
        sep,
        row("Total Trades",    *[r.get("total_trades", 0) for r in reports.values()]),
        row("Win Rate",        *[f"{r.get('win_rate', 0):.2%}" for r in reports.values()]),
        row("Profit Factor",   *[pf_str(r) for r in reports.values()]),
        row("Total Return",    *[f"{r.get('total_return_pct', 0):+.2f}%" for r in reports.values()]),
        row("Final Equity",    *[f"${r.get('final_equity', 0):,.2f}" for r in reports.values()]),
        row("Sharpe Ratio",    *[f"{r.get('sharpe_ratio', 0):.3f}" for r in reports.values()]),
        row("Max Drawdown",    *[f"{r.get('max_drawdown_pct', 0):.2f}%" for r in reports.values()]),
        "=" * (26 + col_w * len(modes)),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load predictor helper
# ---------------------------------------------------------------------------

def load_predictor(model_dir: str):
    """Load saved AI models and return a MarketPredictor."""
    from training.trainer import Trainer

    trainer = Trainer(model_dir=model_dir)
    trainer.load_all_models()
    return trainer.get_market_predictor()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from training.data_utils import load_csv, download_from_mt5, split_data

    # Step 1: Load data
    if args.from_mt5:
        df = download_from_mt5(args.symbol, args.timeframe, args.start)
    else:
        df = load_csv(args.csv)

    # Step 2: Split — use only the out-of-sample backtest period
    _train_df, _valid_df, backtest_df = split_data(df)

    if len(backtest_df) < 52:
        logger.error("Backtest set too small (< 52 candles). Check your data / date split.")
        sys.exit(1)

    # Step 3: Load AI models (needed for ai-only and combined modes)
    predictor = None
    if args.mode in ("ai-only", "combined", "compare"):
        try:
            predictor = load_predictor(args.model_dir)
        except Exception as e:
            logger.warning(f"Could not load AI models: {e}")
            if args.mode in ("ai-only",):
                logger.error("AI models required for ai-only mode. Run training/train.py first.")
                sys.exit(1)
            logger.warning("Falling back to strategy-only for combined/compare modes.")

    # Step 4: Run requested mode(s)
    if args.mode == "strategy-only":
        report = run_strategy_only(
            backtest_df, args.symbol, args.initial_capital, args.consensus_threshold
        )
        logger.info(format_report(report))

    elif args.mode == "ai-only":
        report = run_ai_only(
            backtest_df, args.symbol, args.initial_capital, predictor, args.min_confidence
        )
        # ai-only formatted by Backtester inside run_ai_only

    elif args.mode == "combined":
        report = run_combined(
            backtest_df, args.symbol, args.initial_capital,
            predictor, args.consensus_threshold, args.min_confidence,
        )
        logger.info(format_report(report))

    elif args.mode == "compare":
        reports: Dict[str, Dict[str, Any]] = {}

        logger.info("Running strategy-only …")
        reports["strategy-only"] = run_strategy_only(
            backtest_df, args.symbol, args.initial_capital, args.consensus_threshold
        )

        if predictor is not None:
            logger.info("Running ai-only …")
            ai_rep = run_ai_only(
                backtest_df, args.symbol, args.initial_capital, predictor, args.min_confidence
            )
            if "error" not in ai_rep:
                reports["ai-only"] = ai_rep

            logger.info("Running combined …")
            reports["combined"] = run_combined(
                backtest_df, args.symbol, args.initial_capital,
                predictor, args.consensus_threshold, args.min_confidence,
            )
        else:
            logger.warning("Skipping ai-only and combined modes — no AI models loaded.")

        logger.info(format_comparison_table(reports))


if __name__ == "__main__":
    main()
