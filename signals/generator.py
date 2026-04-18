"""Génération de signaux enrichis bidirectionnels (targets, stop-loss, R/R).

Sprint 2 :
- LONG : targets haussiers (+50/+100/+200%), stop sous cours par ATR/−10%
- SHORT : targets baissiers (−15/−25/−35%), stop au-dessus du cours par ATR/+10%
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from loguru import logger

from config import settings
from data.cache import FEATURE_VERSION, SessionLocal, Signal
from ml.scanner import ScanResult

# Fenêtre anti-doublon : ne pas recréer un signal identique (ticker+direction)
# si un signal existe déjà dans les 60 dernières minutes. Évite le bruit des
# scans rapprochés (scheduler toutes les 15 min).
DEDUP_WINDOW_MINUTES = 60


def _confidence_level(score: float) -> str:
    if score >= 0.90:
        return "ULTRA"
    if score >= 0.80:
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    return "LOW"


def _targets_long(price: float, atr: float) -> tuple[float, float, float, float]:
    stop = max(price - 2 * atr, price * 0.90)
    return price * 1.50, price * 2.00, price * 3.00, stop


def _targets_short(price: float, atr: float) -> tuple[float, float, float, float]:
    """Targets pour SHORT : on vise BAS. Stop au-dessus du prix."""
    stop = min(price + 2 * atr, price * 1.10)
    return price * 0.85, price * 0.75, price * 0.65, stop


def _risk_reward(price: float, target: float, stop: float, direction: str) -> float:
    if direction == "LONG":
        reward = max(target - price, 0.0)
        risk = max(price - stop, 1e-9)
    else:
        reward = max(price - target, 0.0)
        risk = max(stop - price, 1e-9)
    return reward / risk if risk > 0 else 0.0


def build_signal(scan: ScanResult) -> Signal | None:
    if scan.consensus < settings.consensus_threshold:
        return None

    atr = float(scan.features.get("atr_14", 0) or 0)
    price = scan.price

    if scan.direction == "LONG":
        t1, t2, t3, stop = _targets_long(price, atr)
    else:
        t1, t2, t3, stop = _targets_short(price, atr)

    rr = _risk_reward(price, t2, stop, scan.direction)
    top_feats = ", ".join(f"{name}={val:.2f}" for name, val in scan.top_features[:5])

    sig = Signal(
        ticker=scan.ticker,
        created_at=scan.timestamp,
        price=price,
        direction=scan.direction,
        target_type=scan.target_type,
        consensus_score=scan.consensus,
        consensus_long=scan.consensus_long,
        consensus_short=scan.consensus_short,
        rf_prob=scan.rf_prob,
        xgb_prob=scan.xgb_prob,
        lgb_prob=scan.lgb_prob,
        lstm_prob=scan.lstm_prob,
        iso_score=scan.iso_score,
        confidence=_confidence_level(scan.consensus),
        target_1=t1,
        target_2=t2,
        target_3=t3,
        stop_loss=stop,
        risk_reward=rr,
        feature_version=FEATURE_VERSION,
        top_features=top_feats,
        features_json=json.dumps(scan.features, default=str),
        notified=False,
    )
    return sig


def _is_duplicate(session, signal: Signal, window_minutes: int) -> bool:
    """Vrai si un signal identique (ticker+direction) existe déjà dans la fenêtre."""
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    existing = (
        session.query(Signal)
        .filter(
            Signal.ticker == signal.ticker,
            Signal.direction == signal.direction,
            Signal.created_at >= cutoff,
        )
        .first()
    )
    return existing is not None


def persist_signals(
    signals: list[Signal],
    dedup_window_minutes: int = DEDUP_WINDOW_MINUTES,
) -> list[Signal]:
    if not signals:
        return []
    persisted: list[Signal] = []
    skipped = 0
    with SessionLocal() as session:
        for s in signals:
            if _is_duplicate(session, s, dedup_window_minutes):
                skipped += 1
                continue
            session.add(s)
            persisted.append(s)
        session.commit()
        for s in persisted:
            session.refresh(s)
    if skipped:
        logger.info(
            f"[signals] {len(persisted)} sauvegardés, {skipped} doublons ignorés "
            f"(fenêtre {dedup_window_minutes} min)"
        )
    else:
        logger.info(f"[signals] {len(persisted)} signaux sauvegardés")
    return persisted


def fetch_signals(limit: int = 200) -> list[Signal]:
    with SessionLocal() as session:
        return (
            session.query(Signal)
            .order_by(Signal.created_at.desc())
            .limit(limit)
            .all()
        )


def process_scan(results: list[ScanResult]) -> list[Signal]:
    built = [s for s in (build_signal(r) for r in results) if s is not None]
    return persist_signals(built)
