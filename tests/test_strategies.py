"""Tests Protocol DirectionStrategy (Sprint 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.strategies import (
    DIRECTION_TO_TARGET,
    DirectionPrediction,
    DirectionStrategy,
)


class _FakeStrategy:
    """Implémentation minimale pour valider le runtime_checkable."""

    direction = "LONG"
    target_type = "PEAK_100"
    threshold = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._mean = float(y.mean())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean, dtype=float)

    def score_one(self, features_row: pd.Series) -> DirectionPrediction:
        return DirectionPrediction(
            direction="LONG", target_type="PEAK_100",
            consensus=self._mean, rf_prob=self._mean, xgb_prob=self._mean,
            lstm_prob=0.0, iso_score=0.0, threshold=self.threshold,
        )


def test_direction_to_target_mapping() -> None:
    assert DIRECTION_TO_TARGET["LONG"] == "PEAK_100"
    assert DIRECTION_TO_TARGET["SHORT"] == "CRASH_30"


def test_fake_strategy_satisfies_protocol() -> None:
    strat = _FakeStrategy()
    assert isinstance(strat, DirectionStrategy)


def test_prediction_is_frozen() -> None:
    pred = DirectionPrediction(
        direction="LONG", target_type="PEAK_100",
        consensus=0.7, rf_prob=0.6, xgb_prob=0.8,
        lstm_prob=0.0, iso_score=0.1, threshold=0.5,
    )
    with pytest.raises(Exception):
        pred.consensus = 0.9  # type: ignore[misc]


def test_prediction_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        DirectionPrediction(
            direction="LONG", target_type="PEAK_100",
            consensus=1.5, rf_prob=0.5, xgb_prob=0.5,
            lstm_prob=0.0, iso_score=0.0, threshold=0.5,
        )
    with pytest.raises(ValueError):
        DirectionPrediction(
            direction="LONG", target_type="PEAK_100",
            consensus=0.5, rf_prob=0.5, xgb_prob=0.5,
            lstm_prob=0.0, iso_score=-2.0, threshold=0.5,
        )


def test_fake_strategy_predicts() -> None:
    strat = _FakeStrategy()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 1])
    strat.fit(X, y)
    probs = strat.predict_proba(X)
    assert probs.shape == (3,)
    assert np.allclose(probs, 2 / 3)
