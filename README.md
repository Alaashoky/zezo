# zezo

Multi-Strategy Trading Bot with StrategyBrain

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your MetaTrader 5 account details:

```bash
cp .env.example .env
```

Then edit `.env`:

```
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_server_name
```

> **Note:** `.env` is listed in `.gitignore` and will never be committed to the repository.

You can find your MT5 credentials in MetaTrader 5 → File → Login to Trade Account.

### 3. Run the live bot

```bash
# Dry-run mode (observe signals, no real trades)
python live_bot.py --dry-run

# Live trading mode
python live_bot.py

# Specify symbol and timeframe
python live_bot.py --symbol XAUUSD --timeframe M15
```

---

## Download Historical Data

Use `training/download_mt5_data.py` to download OHLCV data from MetaTrader 5 and save it as a CSV file in `data/`.

```bash
# Download XAUUSD M15 data from 2020-01-01 to today
python training/download_mt5_data.py

# Custom symbol / timeframe / date range
python training/download_mt5_data.py --symbol XAUUSD --timeframe M15 --start 2020-01-01
```

The CSV will be saved as `data/XAUUSD_M15_historical.csv`.

---

## Training + Backtest Pipeline

Use `training/train_and_backtest.py` to train the AI models and run a full strategy backtest.

### Data split

| Period | Dates | Duration |
|---|---|---|
| **Training** | 2020-01-01 → 2024-06-30 | 4.5 years |
| **Validation** | 2024-07-01 → 2025-06-30 | 1 year |
| **Backtest** | 2025-07-01 → present | ~9 months (out-of-sample) |

### Run the pipeline

```bash
# Using a pre-downloaded CSV
python training/train_and_backtest.py --csv data/XAUUSD_M15_historical.csv

# Download directly from MT5
python training/train_and_backtest.py --from-mt5

# Skip LSTM (faster, no GPU required)
python training/train_and_backtest.py --csv data/XAUUSD_M15_historical.csv --skip-lstm
```

The pipeline:
1. Loads and splits data into train / validation / backtest periods
2. Trains LSTM, RandomForest, and XGBoost on the training period
3. Validates AI model performance on the validation period
4. Runs all 10 strategies via StrategyBrain on the out-of-sample backtest period
5. Generates a performance report (win rate, profit factor, max drawdown, total return)
6. Saves trained models to `saved_models/`

---

## File Structure

```
zezo/
├── .env.example          ← Template for credentials (copy to .env)
├── .gitignore            ← .env is excluded from version control
├── README.md
├── requirements.txt
├── bot.py
├── live_bot.py           ← Live trading bot
├── data/                 ← Historical CSV data (git-ignored except .gitkeep)
├── training/
│   ├── trainer.py        ← AI model training orchestrator
│   ├── backtester.py     ← Walk-forward backtester
│   ├── download_mt5_data.py   ← MT5 historical data downloader
│   └── train_and_backtest.py  ← Full training + backtest pipeline
├── strategies/           ← 10 trading strategies + StrategyBrain
├── models/               ← LSTM, RandomForest, XGBoost, MarketPredictor
├── config/               ← ModelConfig, TradingConfig
└── mt5/                  ← MT5 connector, executor, risk manager
```
