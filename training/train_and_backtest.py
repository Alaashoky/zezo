#!/usr/bin/env python3
"""
training/train_and_backtest.py — Full Training + Backtest Pipeline
====================================================================
This is now a thin wrapper around the two separate modules:

  * training/train.py    — trains and saves AI models
  * training/backtest.py — backtests strategies and/or AI models

It keeps the original ``--csv``, ``--from-mt5``, ``--skip-lstm``,
``--initial-capital``, and ``--backtest-only`` flags so that existing
scripts and documentation continue to work unchanged.

New dedicated commands (preferred):
------------------------------------
    # Train only
    python -m training.train --csv data/XAUUSD.m_M15_historical.csv

    # Backtest only
    python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \
        --mode compare --initial-capital 400

Legacy combined command:
------------------------
    python -m training.train_and_backtest \
        --csv data/XAUUSD.m_M15_historical.csv --initial-capital 400

    python -m training.train_and_backtest \
        --csv data/XAUUSD.m_M15_historical.csv --backtest-only --initial-capital 400
"""

import argparse
import logging
import os
import sys

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

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
# Re-export shared helpers for backward compatibility
# ---------------------------------------------------------------------------
from training.data_utils import (       # noqa: F401  (public API)
    TRAIN_END, VALID_END,
    load_csv, download_from_mt5, split_data,
)
from training.backtest import (         # noqa: F401
    run_strategy_only as run_strategy_backtest,
    format_report as format_strategy_report,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI models and run strategy backtest on historical data "
                    "(legacy combined script — see training/train.py and training/backtest.py)"
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
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Skip training; load saved models from --model-dir and run backtest only",
    )
    parser.add_argument(
        "--no-strategy-features",
        action="store_true",
        help="Do NOT include strategy signals as training features "
             "(faster but lower quality; default: strategy features ON)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point — thin wrapper delegating to train.py / backtest.py logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Step 1: Load data
    if args.from_mt5:
        df = download_from_mt5(args.symbol, args.timeframe, args.start)
    else:
        df = load_csv(args.csv)

    # Step 2: Split data
    train_df, valid_df, backtest_df = split_data(df)

    if args.backtest_only:
        from training.trainer import Trainer

        logger.info("=" * 60)
        logger.info("BACKTEST-ONLY MODE — loading saved models")
        logger.info("=" * 60)
        trainer = Trainer(model_dir=args.model_dir)
        trainer.load_all_models()
    else:
        if len(train_df) < 200:
            logger.error("Training set too small (< 200 candles). Check your data.")
            sys.exit(1)

        from training.trainer import Trainer

        logger.info("=" * 60)
        logger.info("PHASE 1 — TRAINING")
        logger.info("=" * 60)

        add_strategy_features = not args.no_strategy_features
        trainer = Trainer(model_dir=args.model_dir)
        trainer.train_all_models(
            train_df,
            skip_lstm=args.skip_lstm,
            add_strategy_features=add_strategy_features,
        )
        trainer.save_all_models()
        trainer.print_summary()

    # Step 3: Validate on validation period
    if len(valid_df) >= 102:
        logger.info("=" * 60)
        logger.info("PHASE 2 — VALIDATION (AI models)")
        logger.info("=" * 60)
        from training.backtester import Backtester

        predictor = trainer.get_market_predictor()
        backtester = Backtester(predictor=predictor, initial_capital=args.initial_capital)
        try:
            val_report = backtester.run(valid_df, symbol=args.symbol)
            logger.info(backtester.format_report(val_report))
        except Exception as e:
            logger.warning(f"Validation backtest failed: {e}")
            val_report = {}
    else:
        logger.warning("Validation set too small — skipping validation phase.")
        val_report = {}

    # Step 4: Strategy backtest (out-of-sample)
    if len(backtest_df) >= 52:
        from training.backtest import run_combined, format_report

        predictor = trainer.get_market_predictor()
        strategy_report = run_combined(
            backtest_df,
            symbol=args.symbol,
            initial_capital=args.initial_capital,
            predictor=predictor,
            consensus_threshold=0.6,
            min_confidence=0.5,
        )
        logger.info(format_report(strategy_report))
    else:
        logger.warning("Backtest set too small — skipping strategy backtest phase.")
        strategy_report = {}

    # Step 5: Summary
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
    if not args.backtest_only:
        logger.info(f"Models saved -> {args.model_dir}/")


if __name__ == "__main__":
    main()
