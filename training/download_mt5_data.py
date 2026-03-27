#!/usr/bin/env python3
"""
training/download_mt5_data.py — Download historical OHLCV data from MetaTrader 5
==================================================================================
Connects to MT5 using credentials from the .env file, downloads historical
candle data for the requested symbol/timeframe, and saves it as CSV to the
data/ directory.

Usage
-----
    python training/download_mt5_data.py
    python training/download_mt5_data.py --symbol XAUUSD --timeframe M15
    python training/download_mt5_data.py --symbol XAUUSD --timeframe M15 --start 2020-01-01
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_mt5_data")

# MetaTrader 5 timeframe map
TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from MetaTrader 5"
    )
    parser.add_argument(
        "--symbol", default="XAUUSD", help="Trading symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--timeframe",
        default="M15",
        choices=list(TIMEFRAME_MAP.keys()),
        help="Candle timeframe (default: M15)",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date in YYYY-MM-DD format (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save CSV file (default: data/)",
    )
    return parser.parse_args()


def connect_mt5() -> bool:
    """Initialise MT5 connection using environment variables."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        return False

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not all([login, password, server]):
        logger.error(
            "MT5 credentials not set. Copy .env.example to .env and fill in your credentials."
        )
        return False

    try:
        login_int = int(login)
    except ValueError:
        logger.error("MT5_LOGIN must be a numeric account number.")
        return False

    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False

    if not mt5.login(login_int, password=password, server=server):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    account_info = mt5.account_info()
    if account_info:
        logger.info(
            f"Connected to MT5 — account: {account_info.login}, "
            f"server: {account_info.server}, "
            f"balance: {account_info.balance:.2f} {account_info.currency}"
        )
    return True


def download_data(
    symbol: str,
    timeframe_str: str,
    start_date: datetime,
    end_date: datetime,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Download candles from MT5 and return as a DataFrame."""
    import MetaTrader5 as mt5
    import pandas as pd

    tf = TIMEFRAME_MAP[timeframe_str]

    logger.info(
        f"Downloading {symbol} {timeframe_str} from {start_date.date()} to {end_date.date()} …"
    )

    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        logger.error(f"No data returned for {symbol}. MT5 error: {error}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(
        columns={
            "time": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        },
        inplace=True,
    )
    # Drop MT5-specific columns we don't need
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("datetime", inplace=True)

    logger.info(f"Downloaded {len(df):,} candles — {df.index[0]} → {df.index[-1]}")
    return df


def save_csv(df: "pd.DataFrame", output_dir: str, symbol: str, timeframe: str) -> str:  # type: ignore[name-defined]
    """Save DataFrame to CSV and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol}_{timeframe}_historical.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    logger.info(f"Saved {len(df):,} rows → {filepath}")
    return filepath


def main() -> None:
    args = parse_args()

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid start date format: {args.start}. Use YYYY-MM-DD.")
        sys.exit(1)

    end_date = datetime.now()
    if args.end:
        try:
            end_date = datetime.strptime(args.end, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid end date format: {args.end}. Use YYYY-MM-DD.")
            sys.exit(1)

    if not connect_mt5():
        sys.exit(1)

    try:
        import MetaTrader5 as mt5
        import pandas as pd  # noqa: F401

        df = download_data(args.symbol, args.timeframe, start_date, end_date)

        if df.empty:
            logger.error("No data downloaded. Exiting.")
            sys.exit(1)

        save_csv(df, args.output_dir, args.symbol, args.timeframe)

    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed.")


if __name__ == "__main__":
    main()
