"""Génération de signaux enrichis bidirectionnels (targets, stop-loss, R/R).

Sprint 2 :
- LONG : targets haussiers (+50/+100/+200%), stop sous cours par ATR/−10%
- SHORT : targets baissiers (−15/−25/−35%), stop au-dessus du cours par ATR/+10%
"""
from __future__ import annotations

import json

from loguru import logger

from config import settings
from data.cache import FEATURE_VERSION, SessionLocal, Signal
from ml.scanner import ScanResult


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


def persist_signals(signals: list[Signal]) -> list[Signal]:
    if not signals:
        return []
    with SessionLocal() as session:
        for s in signals:
            session.add(s)
        session.commit()
        for s in signals:
            session.refresh(s)
    logger.info(f"[signals] {len(signals)} signaux sauvegardés")
    return signals


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
