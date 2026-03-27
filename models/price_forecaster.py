"""
Price Forecaster — uses the LSTM model for multi-step price forecasting.

Returns predicted next-N candle prices with simple confidence intervals.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PriceForecaster:
    """
    Forecasts the next N candle closing prices using the trained LSTMModel.

    Usage
    -----
    forecaster = PriceForecaster(forecast_steps=5)
    forecaster.load_model("saved_models/")
    result = forecaster.forecast(ohlcv_df)
    # result = {"predicted_prices": [...], "upper_bound": [...], "lower_bound": [...], "confidence": 0.73}
    """

    def __init__(self, config=None, forecast_steps: Optional[int] = None):
        if config is None:
            from config.model_config import ModelConfig
            config = ModelConfig()
        self.config = config
        self.forecast_steps = forecast_steps or config.lstm.forecast_steps
        self._lstm: Optional[Any] = None

    # ── model helpers ─────────────────────────────────────────────────────────

    def attach_lstm(self, lstm_model):
        """Attach an already-trained LSTMModel instance."""
        self._lstm = lstm_model

    def load_model(self, model_dir: str):
        """Load the LSTM model from *model_dir*."""
        from models.lstm_model import LSTMModel
        m = LSTMModel(self.config.lstm)
        m.load_model(model_dir)
        self._lstm = m
        logger.info(f"PriceForecaster: LSTM loaded from {model_dir}")

    # ── forecasting ───────────────────────────────────────────────────────────

    def forecast(
        self,
        data: pd.DataFrame,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict next *steps* closing prices.

        Parameters
        ----------
        data  : OHLCV DataFrame
        steps : override the default forecast_steps

        Returns
        -------
        dict with:
          - 'predicted_prices' : list of floats
          - 'upper_bound'      : list of floats (+ 1 ATR confidence band)
          - 'lower_bound'      : list of floats (- 1 ATR confidence band)
          - 'confidence'       : overall confidence (0-1)
          - 'steps'            : number of steps forecasted
        """
        if self._lstm is None:
            raise RuntimeError("No LSTM model attached. Call attach_lstm() or load_model() first.")

        n_steps = steps or self.forecast_steps
        result = self._lstm.forecast_prices(data, steps=n_steps)

        predicted = result["predicted_prices"]
        confidence = result["confidence"]

        # ── confidence intervals using ATR ────────────────────────────────────
        from models.feature_engineering import compute_atr
        df_lower = data.copy()
        df_lower.columns = [str(c).lower() for c in df_lower.columns]
        atr = compute_atr(df_lower)
        last_atr = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0

        upper: List[float] = []
        lower: List[float] = []
        for i, p in enumerate(predicted):
            # widen the band slightly with each step to reflect uncertainty
            band = last_atr * (1 + i * 0.1)
            upper.append(p + band)
            lower.append(p - band)

        return {
            "predicted_prices": predicted,
            "upper_bound": upper,
            "lower_bound": lower,
            "confidence": confidence,
            "steps": n_steps,
        }

    def forecast_summary(self, data: pd.DataFrame, steps: Optional[int] = None) -> str:
        """Return a human-readable forecast summary string."""
        result = self.forecast(data, steps=steps)
        last_price = float(data["close"].iloc[-1])
        final_price = result["predicted_prices"][-1]
        direction = "↑" if final_price > last_price else "↓"
        change_pct = (final_price - last_price) / last_price * 100
        return (
            f"Price Forecast ({result['steps']} candles): "
            f"{last_price:.4f} → {final_price:.4f} {direction} "
            f"({change_pct:+.2f}%) | confidence: {result['confidence']:.2f}"
        )
