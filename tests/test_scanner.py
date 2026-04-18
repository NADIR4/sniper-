"""Tests pour ml.scanner bidirectionnel (Sprint 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.scanner import _score_one
from ml.features import FEATURE_NAMES


def _synthetic_ohlcv(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.2, 1.0, n))
    close = np.maximum(close, 5)
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class _StubClassifier:
    """Mock minimal avec predict_proba retournant une proba fixe."""

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def predict_proba(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.tile([1 - self.prob, self.prob], (n, 1))


class _StubIso:
    def decision_function(self, X) -> np.ndarray:
        return np.zeros(len(X))


class _StubScaler:
    def transform(self, X):
        return np.asarray(X)


def _models_stub(long_prob: float, short_prob: float) -> dict:
    return {
        "scaler": _StubScaler(),
        "iso": _StubIso(),
        "rf": _StubClassifier(long_prob),  # legacy alias
        "xgb": _StubClassifier(long_prob),
        "lstm": None,
        "rf_long": _StubClassifier(long_prob),
        "xgb_long": _StubClassifier(long_prob),
        "lstm_long": None,
        "rf_short": _StubClassifier(short_prob),
        "xgb_short": _StubClassifier(short_prob),
        "metrics": {
            "hyperparameters": {
                "consensus_weights": {"rf": 0.35, "xgb": 0.40, "lstm": 0.15, "iso": 0.10}
            }
        },
    }


def test_scanner_picks_long_when_long_wins() -> None:
    df = _synthetic_ohlcv(400)
    models = _models_stub(long_prob=0.90, short_prob=0.10)
    result = _score_one("TEST", df, models)
    assert result is not None
    assert result.direction == "LONG"
    assert result.target_type == "PEAK_100"
    assert result.consensus_long > result.consensus_short


def test_scanner_picks_short_when_short_wins() -> None:
    df = _synthetic_ohlcv(400)
    models = _models_stub(long_prob=0.10, short_prob=0.90)
    result = _score_one("TEST", df, models)
    assert result is not None
    assert result.direction == "SHORT"
    assert result.target_type == "CRASH_30"
    assert result.consensus_short > result.consensus_long


def test_scanner_returns_none_on_insufficient_data() -> None:
    df = _synthetic_ohlcv(30)  # trop court
    models = _models_stub(0.5, 0.5)
    assert _score_one("TEST", df, models) is None


def test_scanner_missing_short_models_defaults_to_zero() -> None:
    """Si rf_short/xgb_short absents, consensus_short=iso uniquement."""
    df = _synthetic_ohlcv(400)
    models = _models_stub(0.80, 0.0)
    models["rf_short"] = None
    models["xgb_short"] = None
    result = _score_one("TEST", df, models)
    assert result is not None
    assert result.direction == "LONG"
    # consensus_short ne doit contenir que le poids iso (0.10 * ~0.5)
    assert result.consensus_short < 0.15
