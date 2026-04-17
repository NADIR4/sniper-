"""Tests unitaires pour ml.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_NAMES,
    _compute_forward_labels,
    build_training_dataset,
    build_training_dataset_dual,
    compute_features,
)


def _synthetic_ohlcv(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.5, 1.0, n))
    close = np.maximum(close, 5)
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx)


def test_compute_features_shape() -> None:
    df = _synthetic_ohlcv(300)
    feats = compute_features(df)
    assert list(feats.columns) == FEATURE_NAMES
    assert len(feats) == len(df)


def test_compute_features_no_all_nan() -> None:
    df = _synthetic_ohlcv(300)
    feats = compute_features(df).dropna()
    assert not feats.empty
    assert feats.isna().sum().sum() == 0


def test_build_training_dataset() -> None:
    panels = {"FAKE": _synthetic_ohlcv(400)}
    X, y, meta = build_training_dataset(panels, lookback=60, horizon=60, gain_threshold=0.05)
    assert len(X) == len(y) == len(meta)
    assert set(X.columns) == set(FEATURE_NAMES)


def test_empty_input() -> None:
    feats = compute_features(pd.DataFrame())
    assert feats.empty


# --- Sprint 1 : anti-leakage & bidirectionnel ---------------------------------


def test_forward_labels_strict_no_leakage_current_day() -> None:
    """Contrôle strict : high[t] NE DOIT PAS influencer y_long[t].

    On place un pic artificiellement énorme uniquement à t=50 (high passe à 1000).
    Si la fonction regardait high[t] (leakage), y_long[50] serait 1.
    Avec une fonction correcte, y_long[50] ne doit dépendre QUE de high[51..110].
    """
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = pd.Series(np.full(n, 100.0), index=idx)  # prix plat
    high = close.copy()
    low = close * 0.99
    high.iloc[50] = 1000.0  # pic isolé exclusivement au jour 50

    labels = _compute_forward_labels(
        close, high, low, horizon=60, peak_threshold=1.0, crash_threshold=0.30,
    )
    # Tout est plat sauf high[50]. Si y_long[50] == 1, il y a leakage.
    assert labels["y_long"].iloc[50] == 0, "LEAKAGE : high[t] influence y[t]"
    # En revanche, pour t=49, high[50] EST dans la fenêtre future → y_long[49] = 1
    assert labels["y_long"].iloc[49] == 1


def test_forward_labels_anti_leakage_tail_is_na() -> None:
    """Les `horizon` derniers jours doivent être NaN (pas de futur complet)."""
    df = _synthetic_ohlcv(300)
    close, high, low = df["close"], df["high"], df["low"]
    horizon = 60
    labels = _compute_forward_labels(
        close, high, low, horizon=horizon, peak_threshold=1.0, crash_threshold=0.30,
    )
    tail = labels.iloc[-horizon:]
    assert tail["y_long"].isna().all(), "fuite : la queue devrait être NaN"
    assert tail["y_short"].isna().all(), "fuite : la queue devrait être NaN"


def test_forward_labels_detect_peak_and_crash() -> None:
    """Sur un mouvement synthétique contrôlé, les labels capturent pic/crash."""
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = np.concatenate([
        np.linspace(100, 110, 40),   # t=0..39 plat
        np.linspace(110, 250, 60),   # t=40..99 : pic (+100% depuis t=39)
        np.linspace(250, 150, 50),   # t=100..149
        np.linspace(150, 90, 50),    # crash -40% en fin
    ])
    close = pd.Series(prices, index=idx)
    high = close * 1.005
    low = close * 0.995
    labels = _compute_forward_labels(
        close, high, low, horizon=60, peak_threshold=1.0, crash_threshold=0.30,
    )
    # t=39 : dans les 60j suivants, on atteint >=+100% → y_long == 1
    assert labels["y_long"].iloc[39] == 1
    # y_long et y_short ne doivent pas TOUS être à 1 partout (signal sensé)
    assert labels["y_long"].dropna().sum() >= 1
    assert labels["y_short"].dropna().sum() >= 1


def test_build_training_dataset_dual_returns_both() -> None:
    panels = {"FAKE": _synthetic_ohlcv(500)}
    out = build_training_dataset_dual(
        panels, lookback=60, horizon=60,
        peak_threshold=0.05, crash_threshold=0.05,
    )
    assert set(out.keys()) == {"long", "short"}
    for direction, (X, y, meta) in out.items():
        assert len(X) == len(y) == len(meta), f"tailles alignées pour {direction}"
        if not X.empty:
            assert set(X.columns) == set(FEATURE_NAMES)


def test_training_dataset_excludes_future_tail() -> None:
    """Aucun échantillon ne doit être pris dans les `horizon` derniers jours."""
    panels = {"FAKE": _synthetic_ohlcv(400)}
    horizon = 60
    _, _, meta = build_training_dataset(
        panels, lookback=60, horizon=horizon, gain_threshold=0.05,
    )
    last_valid_date = panels["FAKE"].index[-horizon - 1]
    for _, ts in meta:
        assert ts <= last_valid_date, f"fuite : échantillon à {ts} dans la queue"
