"""
ZEZO Multi-Strategy Trading Bot
Intelligent multi-strategy bot with Strategy Brain consensus system
and optional AI/ML model integration.
"""
import logging
import os
from typing import Optional

import pandas as pd

from strategies import (
    StrategyConfig, StrategyBrain,
    MovingAverageCrossover, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, SMCICTStrategy, ITS8OSStrategy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bot(symbol="XAUUSD", timeframe="5m"):
    """Create and configure the multi-strategy bot."""
    brain = StrategyBrain(config={
        "min_strategies_required": 2,
        "consensus_threshold": 0.6,
        "performance_weight": 0.4,
        "confidence_weight": 0.6,
    })

    strategies = [
        MovingAverageCrossover(StrategyConfig(name="MA_Crossover", symbol=symbol, timeframe=timeframe)),
        RSIStrategy(StrategyConfig(name="RSI", symbol=symbol, timeframe=timeframe)),
        MACDStrategy(StrategyConfig(name="MACD", symbol=symbol, timeframe=timeframe)),
        BollingerBandsStrategy(StrategyConfig(name="Bollinger", symbol=symbol, timeframe=timeframe)),
        SMCICTStrategy(StrategyConfig(name="SMC_ICT", symbol=symbol, timeframe=timeframe)),
        ITS8OSStrategy(StrategyConfig(name="ITS8OS", symbol=symbol, timeframe=timeframe)),
    ]

    for strategy in strategies:
        strategy.start()
        brain.register_strategy(strategy)

    return brain


def load_ai_predictor(model_dir: str = "saved_models") -> Optional[object]:
    """
    Attempt to load the MarketPredictor from *model_dir*.

    Returns the predictor if successful, or None if models are not available.
    This allows the bot to degrade gracefully when AI models haven't been trained yet.
    """
    try:
        from models.market_predictor import MarketPredictor
        from config.model_config import ModelConfig

        predictor = MarketPredictor(config=ModelConfig(), model_dir=model_dir)
        predictor.load_models(model_dir)
        logger.info("AI MarketPredictor loaded successfully")
        return predictor
    except Exception as e:
        logger.warning(f"AI models not available ({e}) — running in strategy-only mode")
        return None


def process_market_data(brain, data, ai_predictor=None, symbol="XAUUSD"):
    """
    Process market data through the Strategy Brain.

    If an AI predictor is supplied the Brain combines strategy consensus
    with the AI ensemble prediction.  If AI is not available (or fails)
    the bot continues with strategy-only consensus — no error is raised.
    """
    ai_prediction = None
    if ai_predictor is not None:
        try:
            import pandas as pd
            # data may be a bar dict or a DataFrame
            if isinstance(data, pd.DataFrame):
                ai_result = ai_predictor.predict(data, symbol=symbol)
            else:
                # wrap single bar dict into a minimal DataFrame for feature engineering
                ai_result = None
            if ai_result:
                ai_prediction = ai_result
        except Exception as e:
            logger.warning(f"AI prediction failed, falling back to strategy-only: {e}")

    if ai_prediction is not None:
        result = brain.analyze_with_ai(data, ai_prediction)
    else:
        result = brain.analyze_joint(data)

    if result.get("consensus_reached"):
        signal = result["consensus_signal"]
        logger.info(
            f"CONSENSUS: {signal.signal_type.value} at {signal.price} "
            f"(confidence: {signal.confidence:.2f})"
        )
        logger.info(f"Agreeing strategies: {signal.metadata.get('agreeing_strategies', [])}")
        if result.get("ai_contribution"):
            ai_info = result["ai_contribution"]
            logger.info(
                f"AI contribution: {ai_info.get('blend','N/A')} "
                f"(ai_signal={ai_info.get('ai_signal_type','N/A')}, "
                f"ai_confidence={ai_info.get('ai_confidence', 0):.2f})"
            )
        return signal
    else:
        logger.info(f"No consensus: {result.get('reason', 'N/A')}")
        return None


def run_training_mode(data_path: str, model_dir: str = "saved_models"):
    """
    Training mode: train all AI/ML models on historical data.

    Parameters
    ----------
    data_path : path to a CSV file with OHLCV columns
    model_dir : directory to save trained models
    """
    import pandas as pd
    from training.trainer import Trainer

    logger.info(f"Training mode — loading data from {data_path}")
    data = pd.read_csv(data_path, parse_dates=True, index_col=0)
    logger.info(f"Loaded {len(data)} rows")

    trainer = Trainer(model_dir=model_dir)
    results = trainer.train_all_models(data)
    trainer.save_all_models()
    trainer.print_summary()
    return results


if __name__ == "__main__":
    import sys

    mode = os.environ.get("BOT_MODE", "live")

    if mode == "train":
        data_path = sys.argv[1] if len(sys.argv) > 1 else "data/historical.csv"
        model_dir = sys.argv[2] if len(sys.argv) > 2 else "saved_models"
        run_training_mode(data_path, model_dir)
    else:
        model_dir = os.environ.get("MODEL_DIR", "saved_models")
        ai_predictor = load_ai_predictor(model_dir)

        bot = create_bot()
        logger.info("ZEZO Multi-Strategy Bot initialized!")
        logger.info(f"Registered {len(bot.strategies)} strategies")
        if ai_predictor:
            logger.info("AI MarketPredictor: ACTIVE")
        else:
            logger.info("AI MarketPredictor: NOT LOADED (strategy-only mode)")
        logger.info("Ready to process market data …")
