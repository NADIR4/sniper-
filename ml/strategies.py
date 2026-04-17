"""Protocoles & types pour la détection bidirectionnelle (Sprint 1).

Définit l'interface `DirectionStrategy` que chaque classifieur (LONG / SHORT)
doit respecter. Permet un scoring uniforme dans ml/scanner.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

Direction = Literal["LONG", "SHORT"]
TargetType = Literal["PEAK_100", "CRASH_30"]

DIRECTION_TO_TARGET: dict[Direction, TargetType] = {
    "LONG": "PEAK_100",
    "SHORT": "CRASH_30",
}


@dataclass(frozen=True)
class DirectionPrediction:
    """Résultat du scoring pour une direction sur un ticker donné."""

    direction: Direction
    target_type: TargetType
    consensus: float          # score consensus dans [0, 1]
    rf_prob: float
    xgb_prob: float
    lstm_prob: float
    iso_score: float          # dans [-1, 1] (IsolationForest.decision_function)
    threshold: float          # seuil optimal F1 issu du training

    def __post_init__(self) -> None:
        for name in ("consensus", "rf_prob", "xgb_prob", "lstm_prob", "threshold"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} hors [0, 1]")
        if not (-1.0 <= self.iso_score <= 1.0):
            raise ValueError(f"iso_score={self.iso_score} hors [-1, 1]")


@runtime_checkable
class DirectionStrategy(Protocol):
    """Contrat pour un ensemble de modèles détectant une direction (LONG/SHORT).

    Implémentations attendues : RF + XGB + (LSTM) + IsolationForest,
    entraînés sur labels spécifiques à la direction.
    """

    direction: Direction
    target_type: TargetType
    threshold: float

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne P(y=1 | X) dans [0, 1], shape (n_samples,)."""
        ...

    def score_one(self, features_row: pd.Series) -> DirectionPrediction:
        """Scoring consensus pour un seul échantillon (utilisé par scanner)."""
        ...
