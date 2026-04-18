"""Tests Sprint 3 : compute_news_features avec garde anti-leakage shift(1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import (
    FEATURE_NAMES,
    NEWS_FEATURE_NAMES,
    compute_features,
    compute_news_features,
    _neutral_news_features,
)


def _synthetic_news(n_days: int = 30, score: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Génère `n_days` news, 1 par jour, score fixe."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "sentiment_score": np.full(n_days, score) + rng.normal(0, 0.01, n_days),
            "title": [f"news {i}" for i in range(n_days)],
        },
        index=idx,
    )
    df.index.name = "published_at"
    return df


def test_neutral_when_no_news() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    out = compute_news_features(pd.DataFrame(), idx)
    assert list(out.columns) == NEWS_FEATURE_NAMES
    assert (out["news_sent_7d"] == 0.0).all()
    assert (out["news_vol_spike"] == 1.0).all()


def test_news_features_shift_one_day_no_leakage() -> None:
    """La news du jour t ne doit PAS apparaître dans news_sent_7d[t] — shift(1)."""
    idx = pd.date_range("2026-02-01", periods=15, freq="D")
    news = pd.DataFrame(
        {"sentiment_score": [0.0] * 5 + [0.8] + [0.0] * 9, "title": ["x"] * 15},
        index=idx,
    )
    news.index.name = "published_at"

    # La news positive est publiée le 2026-02-06 (index 5)
    feats = compute_news_features(news, idx)

    # Au jour même (2026-02-06), la feature ne doit PAS encore refléter la news
    # car shift(1) repousse l'info à partir du 2026-02-07
    assert feats.loc["2026-02-06", "news_sent_7d"] == 0.0, (
        "Leakage : la news du jour t ne doit pas impacter la feature à t"
    )
    # Au lendemain, elle apparaît
    assert feats.loc["2026-02-07", "news_sent_7d"] > 0.0, (
        "La news de J doit apparaître en feature à J+1"
    )


def test_news_count_7d_sums_correctly() -> None:
    """news_count_7d[t] = nombre d'articles dans [t-7, t-1]."""
    # Aligner les dates news sur l'index du price
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    news = _synthetic_news(10)  # news aux mêmes 10 jours
    feats = compute_news_features(news, idx)
    # Au jour 5 (2026-01-05), on a vu 4 news précédentes (J1..J4), shift(1)
    assert feats.loc["2026-01-05", "news_count_7d"] == 4.0


def test_news_vol_spike_clipped() -> None:
    """Quand count_30d_avg est nul/NaN, spike reste ≤ 10 et sans inf."""
    # 7 news en burst puis rien : count_7d élevé vs count_30d peu peuplé
    idx = pd.date_range("2026-04-01", periods=30, freq="D")
    news_idx = pd.date_range("2026-04-20", periods=7, freq="D")
    news = pd.DataFrame(
        {"sentiment_score": [0.5] * 7, "title": ["x"] * 7}, index=news_idx
    )
    news.index.name = "published_at"
    feats = compute_news_features(news, idx)
    assert feats["news_vol_spike"].max() <= 10.0
    assert np.isfinite(feats["news_vol_spike"]).all()


def test_compute_features_integrates_news() -> None:
    """compute_features expose les 4 colonnes news à la fin du FEATURE_NAMES."""
    rng = np.random.default_rng(42)
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.2, 1.0, n))
    close = np.maximum(close, 5)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=idx,
    )
    news = _synthetic_news(50, score=0.3)

    feats = compute_features(df, news_df=news)
    assert list(feats.columns) == FEATURE_NAMES
    assert len(feats.columns) == 37  # 25 tech + 8 advanced + 4 news (Sprint 4)
    # Pas de NaN sur les features news après warm-up
    tail = feats.dropna().tail(50)
    assert tail["news_sent_7d"].notna().all()
    assert tail["news_vol_spike"].notna().all()


def test_compute_features_backward_compat_no_news() -> None:
    """Appel sans news_df → neutral (0.0 / 1.0) — rétrocompatible."""
    rng = np.random.default_rng(1)
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.2, 1.0, n))
    close = np.maximum(close, 5)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=idx,
    )
    feats = compute_features(df)  # pas de news_df
    tail = feats.dropna().tail(10)
    assert (tail["news_sent_7d"] == 0.0).all()
    assert (tail["news_vol_spike"] == 1.0).all()


def test_neutral_news_features_shape() -> None:
    idx = pd.date_range("2026-05-01", periods=5, freq="B")
    out = _neutral_news_features(idx)
    assert list(out.columns) == NEWS_FEATURE_NAMES
    assert (out["news_vol_spike"] == 1.0).all()
