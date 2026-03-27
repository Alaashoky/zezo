"""
Model configuration — default hyperparameters for all AI/ML models.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class LSTMConfig:
    sequence_length: int = 60          # look-back window (candles)
    hidden_size: int = 128             # LSTM hidden units
    num_layers: int = 2                # stacked LSTM layers
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    forecast_steps: int = 5            # candles ahead for price forecaster


@dataclass
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str = "balanced"


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    use_label_encoder: bool = False
    eval_metric: str = "mlogloss"
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 20


@dataclass
class EnsembleConfig:
    # initial weights — updated dynamically based on accuracy
    lstm_weight: float = 0.4
    rf_weight: float = 0.3
    xgb_weight: float = 0.3
    confidence_threshold: float = 0.6   # min confidence to emit a signal
    min_models_required: int = 2        # min models needed for prediction


@dataclass
class DataProcessorConfig:
    test_size: float = 0.15
    validation_size: float = 0.15
    scaler: str = "minmax"             # "minmax" | "standard"
    handle_missing: str = "ffill"      # "ffill" | "drop"


@dataclass
class ModelConfig:
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    data_processor: DataProcessorConfig = field(default_factory=DataProcessorConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lstm": self.lstm.__dict__,
            "random_forest": self.random_forest.__dict__,
            "xgboost": self.xgboost.__dict__,
            "ensemble": self.ensemble.__dict__,
            "data_processor": self.data_processor.__dict__,
        }
