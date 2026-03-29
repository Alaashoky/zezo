"""
training/data_utils.py — Shared data loading and splitting utilities
=====================================================================
Provides helpers for loading OHLCV data from a CSV or MetaTrader 5
and splitting it into training / validation / backtest periods.

Imported by both training/train.py and training/backtest.py.
"""
import logging
import sys
from datetime import datetime
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data split boundaries
# ---------------------------------------------------------------------------
TRAIN_END = "2024-06-30"
VALID_END = "2025-06-30"


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
