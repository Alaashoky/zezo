"""
Central training orchestrator.

Trains, evaluates, saves, and loads all AI/ML models.

Usage (standalone)
------------------
    import pandas as pd
    from training.trainer import Trainer

    data = pd.read_csv("historical_data.csv")
    trainer = Trainer(model_dir="saved_models/")
    results = trainer.train_all_models(data)
    trainer.save_all_models()
"""
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates training of all AI/ML models.

    Parameters
    ----------
    model_dir  : directory where trained models are saved / loaded
    config     : ModelConfig instance (uses defaults if None)
    """

    def __init__(self, model_dir: str = "saved_models", config=None):
        self.model_dir = model_dir
        if config is None:
            from config.model_config import ModelConfig
            config = ModelConfig()
        self.config = config

        self._lstm = None
        self._rf = None
        self._xgb = None
        self._results: Dict[str, Any] = {}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_lstm(self):
        from models.lstm_model import LSTMModel
        if self._lstm is None:
            self._lstm = LSTMModel(self.config.lstm)
        return self._lstm

    def _get_rf(self):
        from models.random_forest_model import RandomForestModel
        if self._rf is None:
            self._rf = RandomForestModel(self.config.random_forest)
        return self._rf

    def _get_xgb(self):
        from models.xgboost_model import XGBoostModel
        if self._xgb is None:
            self._xgb = XGBoostModel(self.config.xgboost)
        return self._xgb

    # ── public API ────────────────────────────────────────────────────────────

    def train_all_models(
        self,
        data: pd.DataFrame,
        skip_lstm: bool = False,
        add_strategy_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Train LSTM, Random Forest, and XGBoost on *data*.

        Parameters
        ----------
        data      : OHLCV DataFrame
        skip_lstm : set True to skip LSTM if torch is not available
        add_strategy_features : when True, walk-forward strategy signals are
            added as extra features during training (improves quality)

        Returns
        -------
        dict mapping model name → training metrics
        """
        results = {}

        # Random Forest
        try:
            logger.info("Training Random Forest …")
            rf_results = self._get_rf().train(data, add_strategy_features=add_strategy_features)
            results["random_forest"] = rf_results
            rf_acc = rf_results.get("accuracy")
            logger.info(f"Random Forest — accuracy: {rf_acc:.3f}" if isinstance(rf_acc, float) else f"Random Forest — accuracy: {rf_acc}")
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            results["random_forest"] = {"error": str(e)}

        # XGBoost
        try:
            logger.info("Training XGBoost …")
            xgb_results = self._get_xgb().train(data, add_strategy_features=add_strategy_features)
            results["xgboost"] = xgb_results
            xgb_acc = xgb_results.get("accuracy")
            logger.info(f"XGBoost — accuracy: {xgb_acc:.3f}" if isinstance(xgb_acc, float) else f"XGBoost — accuracy: {xgb_acc}")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results["xgboost"] = {"error": str(e)}

        # LSTM
        if not skip_lstm:
            try:
                logger.info("Training LSTM (may take a while) …")
                lstm_results = self._get_lstm().train(data, add_strategy_features=add_strategy_features)
                results["lstm"] = lstm_results
                val_loss = lstm_results.get("val_loss")
                logger.info(f"LSTM — val_loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"LSTM — val_loss: {val_loss}")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
                results["lstm"] = {"error": str(e)}

        self._results = results
        return results

    def train_single_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a single model by name ('lstm', 'random_forest', 'xgboost').

        Returns
        -------
        dict with training metrics for the model
        """
        model_name = model_name.lower().replace(" ", "_")
        dispatch = {
            "lstm": self._get_lstm,
            "random_forest": self._get_rf,
            "rf": self._get_rf,
            "xgboost": self._get_xgb,
            "xgb": self._get_xgb,
        }
        if model_name not in dispatch:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(dispatch)}")

        logger.info(f"Training {model_name} …")
        result = dispatch[model_name]().train(data)
        self._results[model_name] = result
        return result

    def evaluate_all(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate all trained models.

        If *data* is supplied, each model is evaluated on it.
        Otherwise returns cached metrics from training.
        """
        report = {}

        for name, getter in [("random_forest", self._get_rf), ("xgboost", self._get_xgb)]:
            try:
                model = getter()
                if model._is_trained:
                    report[name] = model.evaluate(data)
                else:
                    report[name] = {"status": "not_trained"}
            except Exception as e:
                report[name] = {"error": str(e)}

        try:
            lstm = self._get_lstm()
            if lstm._is_trained:
                report["lstm"] = {
                    "train_loss": lstm.train_losses[-1] if lstm.train_losses else None,
                    "val_loss": lstm.val_losses[-1] if lstm.val_losses else None,
                }
            else:
                report["lstm"] = {"status": "not_trained"}
        except Exception as e:
            report["lstm"] = {"error": str(e)}

        return report

    def save_all_models(self, directory: Optional[str] = None):
        """Save all trained models to *directory* (defaults to self.model_dir)."""
        target = directory or self.model_dir
        os.makedirs(target, exist_ok=True)

        for name, getter in [
            ("LSTM", self._get_lstm),
            ("Random Forest", self._get_rf),
            ("XGBoost", self._get_xgb),
        ]:
            try:
                model = getter()
                if model._is_trained:
                    model.save_model(target)
                    logger.info(f"Saved {name}")
                else:
                    logger.warning(f"Skipping {name} — not trained")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

    def load_all_models(self, directory: Optional[str] = None):
        """Load all models from *directory* (defaults to self.model_dir)."""
        source = directory or self.model_dir

        for name, getter in [
            ("LSTM", self._get_lstm),
            ("Random Forest", self._get_rf),
            ("XGBoost", self._get_xgb),
        ]:
            try:
                getter().load_model(source)
                logger.info(f"Loaded {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")

    def get_market_predictor(self):
        """Return a MarketPredictor pre-loaded with the trained model objects."""
        from models.market_predictor import MarketPredictor

        predictor = MarketPredictor(config=self.config)
        predictor.attach_models(
            lstm=self._lstm if self._lstm and self._lstm._is_trained else None,
            rf=self._rf if self._rf and self._rf._is_trained else None,
            xgb=self._xgb if self._xgb and self._xgb._is_trained else None,
        )
        return predictor

    def get_price_forecaster(self):
        """Return a PriceForecaster pre-loaded with the trained LSTM."""
        from models.price_forecaster import PriceForecaster

        forecaster = PriceForecaster(config=self.config)
        if self._lstm and self._lstm._is_trained:
            forecaster.attach_lstm(self._lstm)
        return forecaster

    def print_summary(self):
        """Print a formatted training summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for model_name, metrics in self._results.items():
            print(f"\n{model_name.upper()}")
            if "error" in metrics:
                print(f"  ✗ Error: {metrics['error']}")
            else:
                for k, v in metrics.items():
                    if k != "report":
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("=" * 60 + "\n")
