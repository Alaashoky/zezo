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

The pipeline is split into two independent scripts:

| Script | Purpose |
|--------|---------|
| `training/train.py` | Train AI models — run **once** |
| `training/backtest.py` | Backtest with many settings — run **repeatedly** |

### Data split

| Period | Dates | Duration |
|---|---|---|
| **Training** | 2020-01-01 → 2024-06-30 | 4.5 years |
| **Validation** | 2024-07-01 → 2025-06-30 | 1 year |
| **Backtest** | 2025-07-01 → present | ~9 months (out-of-sample) |

---

### Step 1 — Train

```bash
# Train RF + XGB + LSTM with strategy signals as extra features (default)
python -m training.train --csv data/XAUUSD.m_M15_historical.csv

# Skip LSTM (faster, no GPU required)
python -m training.train --csv data/XAUUSD.m_M15_historical.csv --skip-lstm

# Train without strategy features (legacy mode, faster)
python -m training.train --csv data/XAUUSD.m_M15_historical.csv --no-strategy-features

# Download directly from MT5
python -m training.train --from-mt5
```

What `train.py` does:
1. Loads and splits data into train / validation / backtest periods
2. Walk-forward computes all 10 strategy signals as training features (unless `--no-strategy-features`)
3. Trains LSTM, RandomForest, and XGBoost on the training period
4. Saves trained models to `saved_models/`
5. Prints a training summary

---

### Step 2 — Backtest

```bash
# Strategy-only mode (10 strategies via StrategyBrain, no AI)
python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \
    --mode strategy-only --initial-capital 400

# AI-only mode (LSTM + RF + XGB ensemble)
python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \
    --mode ai-only --initial-capital 400

# Combined mode — AI + strategies (default)
python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \
    --mode combined --initial-capital 400 --ai-weight 0.5

# Compare all three modes side by side
python -m training.backtest --csv data/XAUUSD.m_M15_historical.csv \
    --mode compare --initial-capital 400
```

#### Backtest arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv FILE` / `--from-mt5` | — | Data source |
| `--mode` | `combined` | `strategy-only`, `ai-only`, `combined`, `compare` |
| `--initial-capital` | `10000` | Starting equity |
| `--model-dir` | `saved_models` | Directory with saved models |
| `--ai-weight` | `0.3` | AI signal weight in combined mode |
| `--consensus-threshold` | `0.6` | Strategy consensus required (0–1) |
| `--min-confidence` | `0.5` | Minimum AI confidence to trade |

#### Mode explanations

| Mode | Description |
|------|-------------|
| **strategy-only** | Uses only the 10 strategies via `brain.analyze_joint()` — no AI |
| **ai-only** | Uses only the AI ensemble (LSTM + RF + XGB) |
| **combined** | Blends AI + strategies via `brain.analyze_with_ai()` — mirrors `live_bot.py` |
| **compare** | Runs all three modes and prints a comparison table |

---

### Legacy combined script (backward compatible)

```bash
# Full pipeline (train + validate + backtest)
python -m training.train_and_backtest --csv data/XAUUSD.m_M15_historical.csv

# Backtest only (load saved models, skip retraining)
python -m training.train_and_backtest \
    --csv data/XAUUSD.m_M15_historical.csv --backtest-only --initial-capital 400
```

---

## Strategy Signals as AI Training Features

The AI models (RF, XGB, LSTM) are now trained with the 10 strategy signals as
additional features.  For each training candle the following columns are added
to the feature matrix:

| Feature | Description |
|---------|-------------|
| `strategy_ma_crossover` | MA Crossover signal (−1/0/1) |
| `strategy_ema_crossover` | EMA Crossover signal |
| `strategy_rsi` | RSI signal |
| `strategy_macd` | MACD signal |
| `strategy_bollinger` | Bollinger Bands signal |
| `strategy_mean_reversion` | Mean Reversion signal |
| `strategy_breakout` | Breakout signal |
| `strategy_stochastic` | Stochastic signal |
| `strategy_smc_ict` | SMC ICT signal |
| `strategy_its8os` | ITS8OS signal |
| `strategy_consensus` | Majority consensus (−1/0/1) |
| `strategy_consensus_confidence` | Consensus confidence (0–1) |
| `strategy_buy_count` | Number of strategies saying BUY (0–10) |
| `strategy_sell_count` | Number of strategies saying SELL (0–10) |

This teaches the AI **when the strategies are right vs wrong**, significantly
improving signal quality.

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
│   ├── data_utils.py          ← Shared data loading & splitting
│   ├── train.py               ← Training-only pipeline (NEW)
│   ├── backtest.py            ← Backtest-only pipeline (NEW)
│   ├── train_and_backtest.py  ← Legacy combined wrapper
│   ├── trainer.py             ← AI model training orchestrator
│   ├── backtester.py          ← Walk-forward AI backtester
│   └── download_mt5_data.py   ← MT5 historical data downloader
├── strategies/           ← 10 trading strategies + StrategyBrain
├── models/               ← LSTM, RandomForest, XGBoost, MarketPredictor
├── config/               ← ModelConfig, TradingConfig
└── mt5/                  ← MT5 connector, executor, risk manager
```
