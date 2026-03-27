"""
ZEZO Multi-Strategy Trading Bot
Intelligent multi-strategy bot with Strategy Brain consensus system.
"""
import logging
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

def process_market_data(brain, data):
    """Process market data through the brain."""
    result = brain.analyze_joint(data)
    if result.get("consensus_reached"):
        signal = result["consensus_signal"]
        logger.info(f"CONSENSUS: {signal.signal_type.value} at {signal.price} (confidence: {signal.confidence:.2f})")
        logger.info(f"Agreeing strategies: {signal.metadata.get('agreeing_strategies', [])}")
        return signal
    else:
        logger.info(f"No consensus: {result.get('reason', 'N/A')}")
        return None

if __name__ == "__main__":
    bot = create_bot()
    logger.info("ZEZO Multi-Strategy Bot initialized!")
    logger.info(f"Registered {len(bot.strategies)} strategies")
    logger.info("Ready to process market data...")
