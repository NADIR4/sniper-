"""Tests pour signals.generator (Sprint 2 : bidirectionnel)."""
from __future__ import annotations

from datetime import datetime

from ml.scanner import ScanResult
from signals.generator import build_signal


def _mock_scan(
    consensus: float = 0.85,
    direction: str = "LONG",
    price: float = 100.0,
) -> ScanResult:
    target_type = "PEAK_100" if direction == "LONG" else "CRASH_30"
    return ScanResult(
        ticker="TEST",
        timestamp=datetime.utcnow(),
        price=price,
        direction=direction,
        target_type=target_type,
        consensus=consensus,
        rf_prob=consensus,
        xgb_prob=consensus,
        lstm_prob=consensus,
        iso_score=consensus,
        consensus_long=consensus if direction == "LONG" else 0.3,
        consensus_short=consensus if direction == "SHORT" else 0.3,
        top_features=[("rsi_14", 65.0)],
        features={"atr_14": 2.0, "rsi_14": 65.0},
    )


def test_build_signal_long_above_threshold() -> None:
    sig = build_signal(_mock_scan(0.85, "LONG"))
    assert sig is not None
    assert sig.ticker == "TEST"
    assert sig.direction == "LONG"
    assert sig.target_type == "PEAK_100"
    assert sig.confidence == "HIGH"
    assert sig.target_1 == 150.0
    assert sig.target_2 == 200.0
    assert sig.stop_loss <= 100.0
    assert sig.risk_reward > 0


def test_build_signal_ultra() -> None:
    sig = build_signal(_mock_scan(0.95))
    assert sig is not None
    assert sig.confidence == "ULTRA"


def test_build_signal_below_threshold() -> None:
    sig = build_signal(_mock_scan(0.50))
    assert sig is None


def test_build_signal_short_targets_below_price() -> None:
    """SHORT : targets doivent être SOUS le prix, stop AU-DESSUS."""
    sig = build_signal(_mock_scan(0.85, "SHORT", price=100.0))
    assert sig is not None
    assert sig.direction == "SHORT"
    assert sig.target_type == "CRASH_30"
    assert sig.target_1 < 100.0, "target_1 SHORT doit être < prix"
    assert sig.target_2 < sig.target_1, "target_2 plus profond que target_1"
    assert sig.target_3 < sig.target_2, "target_3 plus profond que target_2"
    assert sig.stop_loss > 100.0, "stop SHORT doit être AU-DESSUS du prix"
    assert sig.risk_reward > 0


def test_build_signal_persists_consensus_per_direction() -> None:
    """Vérifie que consensus_long/short sont correctement renseignés."""
    sig_long = build_signal(_mock_scan(0.85, "LONG"))
    assert sig_long is not None
    assert sig_long.consensus_long == 0.85
    assert sig_long.consensus_short == 0.3

    sig_short = build_signal(_mock_scan(0.85, "SHORT"))
    assert sig_short is not None
    assert sig_short.consensus_short == 0.85
    assert sig_short.consensus_long == 0.3
