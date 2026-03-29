"""
Market Predictor — ensemble of LSTM, Random Forest and XGBoost.

Combines predictions using weighted voting; weights are updated dynamically
based on each model's recent accuracy.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)

# Label mapping
_SIGNAL_MAP = {0: SignalType.HOLD, 1: SignalType.BUY, 2: SignalType.SELL}


class MarketPredictor:
    """
    Ensemble market predictor that aggregates signals from all trained models.

    Usage
    -----
    predictor = MarketPredictor(model_dir="saved_models/")
    predictor.load_models()
    signal = predictor.predict(ohlcv_df, symbol="XAUUSD")
    """

    def __init__(self, config=None, model_dir: Optional[str] = None):
        if config is None:
            from config.model_config import ModelConfig
            config = ModelConfig()
        self.config = config
        self.model_dir = model_dir

        self._ensemble_cfg = config.ensemble

        # lazy-loaded models
        self._lstm: Optional[Any] = None
        self._rf: Optional[Any] = None
        self._xgb: Optional[Any] = None

        # dynamic weights (updated after each prediction when feedback provided)
        self._weights = {
            "lstm": self._ensemble_cfg.lstm_weight,
            "rf":   self._ensemble_cfg.rf_weight,
            "xgb":  self._ensemble_cfg.xgb_weight,
        }

        # per-model accuracy trackers
        self._correct = {"lstm": 0, "rf": 0, "xgb": 0}
        self._total   = {"lstm": 0, "rf": 0, "xgb": 0}

    # ── model loading ─────────────────────────────────────────────────────────

    def load_models(self, model_dir: Optional[str] = None):
        """Load all saved models from *model_dir*."""
        directory = model_dir or self.model_dir
        if not directory:
            raise ValueError("model_dir must be provided")

        errors = []

        try:
            from models.lstm_model import LSTMModel
            m = LSTMModel(self.config.lstm)
            m.load_model(directory)
            self._lstm = m
        except Exception as e:
            logger.warning(f"Could not load LSTM: {e}")
            errors.append(f"LSTM: {e}")

        try:
            from models.random_forest_model import RandomForestModel
            m = RandomForestModel(self.config.random_forest)
            m.load_model(directory)
            self._rf = m
        except Exception as e:
            logger.warning(f"Could not load Random Forest: {e}")
            errors.append(f"RF: {e}")

        try:
            from models.xgboost_model import XGBoostModel
            m = XGBoostModel(self.config.xgboost)
            m.load_model(directory)
            self._xgb = m
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")
            errors.append(f"XGB: {e}")

        n_loaded = sum(m is not None for m in [self._lstm, self._rf, self._xgb])
        if n_loaded < self._ensemble_cfg.min_models_required:
            raise RuntimeError(
                f"Only {n_loaded} models loaded (need {self._ensemble_cfg.min_models_required}). "
                f"Errors: {errors}"
            )
        logger.info(f"MarketPredictor: loaded {n_loaded}/3 models")

    def attach_models(self, lstm=None, rf=None, xgb=None):
        """Attach already-trained model objects directly (for use during training)."""
        if lstm is not None:
            self._lstm = lstm
        if rf is not None:
            self._rf = rf
        if xgb is not None:
            self._xgb = xgb

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, data: pd.DataFrame, symbol: str = "UNKNOWN", strategy_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an ensemble prediction for the most recent candle.

        Parameters
        ----------
        data   : OHLCV DataFrame (at least `sequence_length` rows)
        symbol : trading symbol name
        strategy_signals : optional dict mapping strategy feature column names to
            their current signal values (e.g. ``{"strategy_rsi": 1, ...}``).
            Pass this when the models were trained with ``add_strategy_features=True``
            so that the same features are available during inference.

        Returns
        -------
        dict with keys:
          - 'signal': Signal object compatible with strategies/base.py
          - 'signal_type': SignalType enum value
          - 'confidence': float [0, 1]
          - 'model_predictions': per-model raw predictions
          - 'consensus_reached': bool
        """
        model_preds: Dict[str, Dict[str, Any]] = {}

        # collect individual predictions
        for name, model in [("lstm", self._lstm), ("rf", self._rf), ("xgb", self._xgb)]:
            if model is None:
                continue
            try:
                pred = model.predict(data, strategy_signals=strategy_signals)
                model_preds[name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        if not model_preds:
            return {
                "consensus_reached": False,
                "reason": "All models failed to predict",
                "signal": None,
                "signal_type": SignalType.HOLD,
                "confidence": 0.0,
                "model_predictions": {},
            }

        # weighted vote
        vote_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # HOLD, BUY, SELL
        total_weight = 0.0

        for name, pred in model_preds.items():
            w = self._weights.get(name, 1.0 / 3)
            vote_scores[pred["signal"]] += w * pred["confidence"]
            total_weight += w

        if total_weight > 0:
            vote_scores = {k: v / total_weight for k, v in vote_scores.items()}

        best_label = max(vote_scores, key=vote_scores.__getitem__)
        best_confidence = vote_scores[best_label]

        signal_type = _SIGNAL_MAP[best_label]
        consensus_reached = best_confidence >= self._ensemble_cfg.confidence_threshold

        last_close = float(data["close"].iloc[-1])
        ai_signal = Signal(
            signal_type=signal_type,
            symbol=symbol,
            price=last_close,
            timestamp=datetime.now(timezone.utc),
            confidence=min(best_confidence, 1.0),
            metadata={
                "source": "ai_ensemble",
                "vote_scores": vote_scores,
                "model_predictions": {k: v["signal"] for k, v in model_preds.items()},
                "model_confidences": {k: v["confidence"] for k, v in model_preds.items()},
                "weights": self._weights.copy(),
            },
        )

        return {
            "consensus_reached": consensus_reached,
            "signal": ai_signal,
            "signal_type": signal_type,
            "confidence": best_confidence,
            "model_predictions": model_preds,
            "reason": f"Ensemble vote: BUY={vote_scores[1]:.2f}, SELL={vote_scores[2]:.2f}, HOLD={vote_scores[0]:.2f}",
        }

    # ── weight updates ────────────────────────────────────────────────────────

    def update_accuracy(self, model_name: str, was_correct: bool):
        """
        Provide feedback on whether a model's prediction was correct.
        Weights are recalculated automatically.
        """
        if model_name not in self._total:
            return
        self._total[model_name] += 1
        if was_correct:
            self._correct[model_name] += 1

        # update weights proportional to accuracy
        accuracies = {
            k: (self._correct[k] / self._total[k]) if self._total[k] > 0 else 0.5
            for k in self._weights
        }
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            self._weights = {k: v / total_acc for k, v in accuracies.items()}

        logger.debug(f"Updated model weights: {self._weights}")

    def get_model_accuracies(self) -> Dict[str, float]:
        """Return current accuracy of each model."""
        return {
            k: (self._correct[k] / self._total[k]) if self._total[k] > 0 else None
            for k in self._weights
        }
