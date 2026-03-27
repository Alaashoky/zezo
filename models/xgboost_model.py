"""
XGBoost model for high-accuracy signal classification.
"""
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import joblib
    from sklearn.metrics import classification_report, f1_score
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed — XGBoostModel will not be functional")


class XGBoostModel:
    """
    XGBoost classifier that predicts next-candle direction.

    Labels:  0 = HOLD, 1 = BUY, 2 = SELL
    """

    MODEL_FILE = "xgboost_model.pkl"

    def __init__(self, config=None):
        if config is None:
            from config.model_config import XGBoostConfig
            config = XGBoostConfig()
        self.config = config
        self.model: Optional[Any] = None
        self._feature_cols: list = []
        self._is_trained = False
        self._eval_results: Dict[str, Any] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, data: pd.DataFrame, add_target: bool = True):
        from models.feature_engineering import build_features, get_feature_columns

        feat_df = build_features(data, add_target=add_target)
        if add_target:
            self._feature_cols = get_feature_columns(feat_df)
            X = feat_df[self._feature_cols]
            y = feat_df["target"]
            return X, y
        else:
            cols = self._feature_cols or get_feature_columns(feat_df)
            available = [c for c in cols if c in feat_df.columns]
            return feat_df[available], None

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train XGBoost on OHLCV data.

        Returns
        -------
        dict with accuracy and F1 score
        """
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required — pip install xgboost>=2.0.0")

        logger.info("XGBoost: preparing data …")
        X, y = self._prepare(data, add_target=True)

        n = len(X)
        n_test = max(1, int(n * 0.15))
        X_train, X_test = X.iloc[: n - n_test], X.iloc[n - n_test:]
        y_train, y_test = y.iloc[: n - n_test], y.iloc[n - n_test:]

        num_classes = int(y.nunique())
        xgb_kwargs = dict(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            eval_metric=self.config.eval_metric,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
        if num_classes > 2:
            xgb_kwargs["objective"] = "multi:softprob"
            xgb_kwargs["num_class"] = num_classes
        else:
            xgb_kwargs["objective"] = "binary:logistic"

        self.model = XGBClassifier(**xgb_kwargs)

        eval_set = [(X_test, y_test)]
        logger.info(f"XGBoost: fitting {len(X_train)} samples …")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self._is_trained = True

        preds = self.model.predict(X_test)
        acc = float(np.mean(preds == y_test.values))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        present_labels = sorted(y_test.unique().tolist())
        target_names = [label_map[l] for l in present_labels if l in label_map]
        report = classification_report(
            y_test, preds,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        )

        self._eval_results = {"accuracy": acc, "f1_weighted": f1, "report": report}
        logger.info(f"XGBoost trained — accuracy: {acc:.3f}, F1: {f1:.3f}")
        return self._eval_results

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict direction for the most recent candle.

        Returns
        -------
        dict with 'signal' (0/1/2), 'confidence', 'probabilities'
        """
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required")
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        X, _ = self._prepare(data, add_target=False)
        probs = self.model.predict_proba(X.iloc[[-1]])[0]
        pred_class = int(np.argmax(probs))
        return {
            "signal": pred_class,
            "confidence": float(probs[pred_class]),
            "probabilities": probs.tolist(),
        }

    def evaluate(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Return evaluation metrics (or re-evaluate if *data* is provided)."""
        if data is not None and _XGB_AVAILABLE and self._is_trained:
            X, y = self._prepare(data, add_target=True)
            preds = self.model.predict(X)
            acc = float(np.mean(preds == y.values))
            f1 = float(f1_score(y, preds, average="weighted", zero_division=0))
            return {"accuracy": acc, "f1_weighted": f1}
        return self._eval_results

    def save_model(self, path: str):
        """Save model and metadata to *path* directory."""
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost / joblib is required")
        if not self._is_trained or self.model is None:
            raise RuntimeError("Nothing to save — model is not trained")
        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_cols": self._feature_cols, "config": self.config},
            os.path.join(path, self.MODEL_FILE),
        )
        logger.info(f"XGBoost model saved to {path}")

    def load_model(self, path: str):
        """Load model and metadata from *path* directory."""
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost / joblib is required")
        data = joblib.load(os.path.join(path, self.MODEL_FILE))
        self.model = data["model"]
        self._feature_cols = data["feature_cols"]
        self.config = data["config"]
        self._is_trained = True
        logger.info(f"XGBoost model loaded from {path}")
