"""
Backtester — tests AI model predictions against historical data.

Calculates performance metrics and can compare AI predictions
vs pure strategy signals.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtest AI predictions against historical OHLCV data.

    Usage
    -----
    backtester = Backtester(predictor=market_predictor)
    report = backtester.run(historical_df, symbol="XAUUSD")
    print(backtester.format_report(report))
    """

    # mapping from model label (0/1/2) to direction multiplier (+1/-1/0)
    _DIRECTION = {0: 0, 1: 1, 2: -1}

    def __init__(self, predictor=None, initial_capital: float = 10_000.0):
        self.predictor = predictor
        self.initial_capital = initial_capital
        self._last_report: Optional[Dict[str, Any]] = None

    # ── metrics helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def _precision_recall_f1(
        y_true: np.ndarray, y_pred: np.ndarray, label: int
    ):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        if len(equity_curve) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())

    # ── core backtest ─────────────────────────────────────────────────────────

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        min_history: int = 100,
    ) -> Dict[str, Any]:
        """
        Walk-forward backtest: for each candle (after the warm-up period),
        get a prediction from the predictor, then compare with what actually happened.

        Parameters
        ----------
        data        : full OHLCV DataFrame (chronological order)
        symbol      : symbol name
        min_history : number of candles to use as warm-up before predicting

        Returns
        -------
        Performance report dict
        """
        if self.predictor is None:
            raise RuntimeError("No predictor attached")

        if len(data) < min_history + 2:
            raise ValueError(f"Need at least {min_history + 2} rows for backtesting")

        predictions: List[int] = []
        actuals: List[int] = []
        confidences: List[float] = []
        equity_curve = [self.initial_capital]
        trade_returns: List[float] = []

        close = data["close"].values

        logger.info(f"Starting backtest on {len(data)} candles …")

        for i in range(min_history, len(data) - 1):
            window = data.iloc[: i + 1]
            try:
                result = self.predictor.predict(window, symbol=symbol)
                pred_signal = result["signal_type"].value if result["signal"] else "HOLD"
                label = {"BUY": 1, "SELL": 2, "HOLD": 0}.get(pred_signal, 0)
                confidence = result.get("confidence", 0.0)
            except Exception as e:
                logger.debug(f"Prediction error at index {i}: {e}")
                label = 0
                confidence = 0.0

            # actual direction (next candle close vs current close)
            actual_return = (close[i + 1] - close[i]) / close[i]
            actual_label = 1 if actual_return > 0.0005 else (2 if actual_return < -0.0005 else 0)

            predictions.append(label)
            actuals.append(actual_label)
            confidences.append(confidence)

            # simple equity simulation (enter on signal, exit next candle)
            direction = self._DIRECTION.get(label, 0)
            trade_return = direction * actual_return
            trade_returns.append(trade_return)
            equity_curve.append(equity_curve[-1] * (1 + trade_return))

        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)
        trade_returns_arr = np.array(trade_returns)
        equity_arr = np.array(equity_curve)

        # metrics
        accuracy = self._accuracy(actuals_arr, predictions_arr)
        buy_p, buy_r, buy_f1 = self._precision_recall_f1(actuals_arr, predictions_arr, 1)
        sell_p, sell_r, sell_f1 = self._precision_recall_f1(actuals_arr, predictions_arr, 2)
        sharpe = self._sharpe_ratio(trade_returns_arr)
        max_dd = self._max_drawdown(equity_arr)
        total_return = float((equity_arr[-1] - self.initial_capital) / self.initial_capital * 100)
        avg_confidence = float(np.mean(confidences))

        report = {
            "symbol": symbol,
            "total_candles": len(data),
            "backtested_candles": len(predictions),
            "accuracy": accuracy,
            "buy_precision": buy_p,
            "buy_recall": buy_r,
            "buy_f1": buy_f1,
            "sell_precision": sell_p,
            "sell_recall": sell_r,
            "sell_f1": sell_f1,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "total_return_pct": total_return,
            "final_equity": float(equity_arr[-1]),
            "initial_capital": self.initial_capital,
            "avg_confidence": avg_confidence,
            "signal_distribution": {
                "buy": int(np.sum(predictions_arr == 1)),
                "sell": int(np.sum(predictions_arr == 2)),
                "hold": int(np.sum(predictions_arr == 0)),
            },
        }

        self._last_report = report
        return report

    def compare_with_strategies(
        self,
        backtest_report: Dict[str, Any],
        strategy_accuracy: float,
    ) -> Dict[str, Any]:
        """
        Simple comparison between AI backtest results and a strategy-only accuracy.
        """
        ai_acc = backtest_report.get("accuracy", 0.0)
        return {
            "ai_accuracy": ai_acc,
            "strategy_accuracy": strategy_accuracy,
            "improvement": ai_acc - strategy_accuracy,
            "ai_sharpe": backtest_report.get("sharpe_ratio", 0.0),
            "ai_return_pct": backtest_report.get("total_return_pct", 0.0),
        }

    def format_report(self, report: Optional[Dict[str, Any]] = None) -> str:
        """Return a human-readable backtest report."""
        r = report or self._last_report
        if not r:
            return "No backtest report available"

        lines = [
            "",
            "=" * 60,
            f"BACKTEST REPORT — {r.get('symbol', 'N/A')}",
            "=" * 60,
            f"Candles tested  : {r['backtested_candles']} / {r['total_candles']}",
            f"Accuracy        : {r['accuracy']:.2%}",
            f"Avg confidence  : {r['avg_confidence']:.2f}",
            "",
            "Signal quality:",
            f"  BUY  — precision: {r['buy_precision']:.2%}, recall: {r['buy_recall']:.2%}, F1: {r['buy_f1']:.2%}",
            f"  SELL — precision: {r['sell_precision']:.2%}, recall: {r['sell_recall']:.2%}, F1: {r['sell_f1']:.2%}",
            "",
            "Portfolio performance:",
            f"  Initial capital : ${r['initial_capital']:,.2f}",
            f"  Final equity    : ${r['final_equity']:,.2f}",
            f"  Total return    : {r['total_return_pct']:+.2f}%",
            f"  Sharpe ratio    : {r['sharpe_ratio']:.3f}",
            f"  Max drawdown    : {r['max_drawdown_pct']:.2f}%",
            "",
            "Signal distribution:",
            f"  BUY={r['signal_distribution']['buy']}  "
            f"SELL={r['signal_distribution']['sell']}  "
            f"HOLD={r['signal_distribution']['hold']}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)
