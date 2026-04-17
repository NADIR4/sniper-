"""MarketScanner bidirectionnel : score LONG + SHORT, retient la direction forte.

Sprint 2 : chaque ticker est scoré dans les deux directions ; on conserve la
direction dont le consensus dépasse le seuil. Si les deux dépassent, on prend
la plus élevée. Si aucune, pas de signal.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from data.collector import collect_history
from data.news_fetcher import load_news_df
from data.universe import get_universe
from ml.features import compute_features
from ml.trainer import load_models

Direction = Literal["LONG", "SHORT"]


@dataclass
class DirectionScore:
    direction: Direction
    target_type: str
    consensus: float
    rf_prob: float
    xgb_prob: float
    lstm_prob: float
    iso_score: float


@dataclass
class ScanResult:
    ticker: str
    timestamp: datetime
    price: float
    # Direction retenue (LONG ou SHORT — celle du consensus le plus haut)
    direction: Direction
    target_type: str
    consensus: float
    rf_prob: float
    xgb_prob: float
    lstm_prob: float
    iso_score: float
    # Scores des deux directions pour transparence
    consensus_long: float = 0.0
    consensus_short: float = 0.0
    top_features: list[tuple[str, float]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)


def _sigmoid(x: float) -> float:
    """Sigmoid standard 1/(1+exp(-x)). Inliers (iso>0) → score>0.5."""
    return float(1.0 / (1.0 + np.exp(-x)))


def _consensus_weights(models: dict) -> dict[str, float]:
    return models.get("metrics", {}).get("hyperparameters", {}).get(
        "consensus_weights", {"rf": 0.35, "xgb": 0.40, "lstm": 0.15, "iso": 0.10}
    )


def _score_direction(
    direction: Direction,
    latest_row: pd.DataFrame,
    seq: np.ndarray | None,
    models: dict,
    weights: dict[str, float],
    iso_score: float,
) -> DirectionScore:
    key = "long" if direction == "LONG" else "short"
    rf = models.get(f"rf_{key}")
    xgb = models.get(f"xgb_{key}")
    lstm = models.get(f"lstm_{key}")
    scaler = (
        models.get(f"scaler_{key}")
        or models.get("scaler_long")
        or models.get("scaler")
    )
    X = scaler.transform(latest_row) if scaler is not None else latest_row.values

    rf_prob = float(rf.predict_proba(X)[0, 1]) if rf is not None else 0.0
    xgb_prob = float(xgb.predict_proba(X)[0, 1]) if xgb is not None else 0.0
    lstm_prob = 0.0
    if lstm is not None and seq is not None:
        try:
            lstm_prob = float(lstm.predict(seq, verbose=0)[0, 0])
        except Exception as exc:
            logger.debug(f"[scanner/{direction}] LSTM: {exc}")

    consensus = float(
        weights["rf"] * rf_prob + weights["xgb"] * xgb_prob
        + weights["lstm"] * lstm_prob + weights["iso"] * iso_score
    )
    return DirectionScore(
        direction=direction,
        target_type="PEAK_100" if direction == "LONG" else "CRASH_30",
        consensus=consensus,
        rf_prob=rf_prob,
        xgb_prob=xgb_prob,
        lstm_prob=lstm_prob,
        iso_score=iso_score,
    )


def _score_one(ticker: str, df: pd.DataFrame, models: dict) -> ScanResult | None:
    if df is None or df.empty or len(df) < settings.lookback_days + 5:
        return None

    # News cachées (vide si jamais fetché → features neutres)
    try:
        news_df = load_news_df(ticker)
    except Exception as exc:
        logger.debug(f"[scanner] load_news {ticker}: {exc}")
        news_df = pd.DataFrame()

    feats = compute_features(df, news_df=news_df if not news_df.empty else None).dropna()
    if feats.empty:
        return None

    latest_row = feats.iloc[[-1]]
    latest_price = float(df["close"].iloc[-1])

    # IsoForest fit sur X_long_s (voir trainer._train_iso) → scaler_long en référence
    iso_scaler = models.get("scaler_long") or models.get("scaler")
    X_iso = iso_scaler.transform(latest_row) if iso_scaler is not None else latest_row.values

    iso = models.get("iso")
    iso_raw = float(iso.decision_function(X_iso)[0]) if iso is not None else 0.0
    iso_score = _sigmoid(iso_raw)

    seq = None
    if len(feats) >= settings.lookback_days:
        seq_arr = feats.iloc[-settings.lookback_days:].values
        seq = np.expand_dims(seq_arr, axis=0).astype(np.float32)

    weights = _consensus_weights(models)

    score_long = _score_direction("LONG", latest_row, seq, models, weights, iso_score)
    score_short = _score_direction("SHORT", latest_row, None, models, weights, iso_score)
    # LSTM SHORT non entraîné → seq=None

    # Retenir la direction avec le consensus le plus fort
    winner = score_long if score_long.consensus >= score_short.consensus else score_short

    importance = (
        models.get("metrics", {})
        .get("feature_importance", {})
        .get("random_forest", {})
    )
    feat_contrib: list[tuple[str, float]] = []
    if importance:
        vals = latest_row.iloc[0].to_dict()
        for name, _imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            feat_contrib.append((name, float(vals.get(name, 0))))

    return ScanResult(
        ticker=ticker,
        timestamp=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        price=latest_price,
        direction=winner.direction,
        target_type=winner.target_type,
        consensus=winner.consensus,
        rf_prob=winner.rf_prob,
        xgb_prob=winner.xgb_prob,
        lstm_prob=winner.lstm_prob,
        iso_score=winner.iso_score,
        consensus_long=score_long.consensus,
        consensus_short=score_short.consensus,
        top_features=feat_contrib,
        features=latest_row.iloc[0].to_dict(),
    )


def scan_market(
    tickers: list[str] | None = None,
    models: dict | None = None,
) -> list[ScanResult]:
    tickers = tickers or get_universe()
    models = models or load_models()
    if models.get("rf") is None:
        raise RuntimeError("Modèles non entraînés. Lance `python main.py train` d'abord.")

    panels = collect_history(tickers, years=1)
    results: list[ScanResult] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_score_one, t, df, models): t for t, df in panels.items()}
        for fut in futures:
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as exc:
                logger.warning(f"[scanner] {futures[fut]}: {exc}")
    results.sort(key=lambda r: -r.consensus)
    top = results[0].ticker if results else "N/A"
    logger.info(f"[scanner] {len(results)} tickers scannés, top={top}")
    return results
