#!/usr/bin/env python3
"""
training/train.py — AI Model Training Pipeline
===============================================
Loads historical OHLCV data, splits it into train/validation/backtest
periods, trains all AI models (RF + XGB + LSTM) on the training period,
and saves them to ``saved_models/``.

Run this once, then use ``training/backtest.py`` to experiment with
different backtest settings without re-training.

Usage
-----
    # Train using a pre-downloaded CSV (with strategy features)
    python -m training.train --csv data/XAUUSD.m_M15_historical.csv

    # Skip LSTM (faster, no GPU required)
    python -m training.train --csv data/XAUUSD.m_M15_historical.csv --skip-lstm

    # Train without strategy signals as features (legacy mode)
    python -m training.train --csv data/XAUUSD.m_M15_historical.csv --no-strategy-features

    # Download directly from MT5
    python -m training.train --from-mt5
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("training.train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI models (RF + XGB + LSTM) on historical OHLCV data"
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
        "--no-strategy-features",
        action="store_true",
        help="Do NOT include strategy signals as training features "
             "(faster but lower quality; default: strategy features ON)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(train_df, model_dir: str, skip_lstm: bool, add_strategy_features: bool):
    """Train all models on *train_df* and save them. Returns (trainer, results)."""
    from training.trainer import Trainer

    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    trainer = Trainer(model_dir=model_dir)
    results = trainer.train_all_models(
        train_df,
        skip_lstm=skip_lstm,
        add_strategy_features=add_strategy_features,
    )
    trainer.save_all_models()
    trainer.print_summary()
    return trainer, results


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

    # Step 2: Split data
    train_df, valid_df, backtest_df = split_data(df)

    if len(train_df) < 200:
        logger.error("Training set too small (< 200 candles). Check your data.")
        sys.exit(1)

    add_strategy_features = not args.no_strategy_features

    logger.info(
        f"Strategy features: {'ON' if add_strategy_features else 'OFF'} "
        f"(use --no-strategy-features to disable)"
    )

    # Step 3: Train
    trainer, results = run_training(
        train_df,
        model_dir=args.model_dir,
        skip_lstm=args.skip_lstm,
        add_strategy_features=add_strategy_features,
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Models saved → {args.model_dir}/")
    logger.info("Run backtest with:")
    logger.info(
        f"  python -m training.backtest --csv {args.csv or '<csv_path>'} "
        f"--mode compare --initial-capital 400"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
