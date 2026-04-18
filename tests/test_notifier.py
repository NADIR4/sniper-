"""Tests pour signals.notifier — filtrage par tier de confiance."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from data.cache import Signal
from signals.notifier import (
    TIER_ORDER,
    TIER_THEMES,
    _compose,
    _should_notify,
)


def _mock_signal(confidence: str = "HIGH", ticker: str = "TEST") -> Signal:
    return Signal(
        ticker=ticker,
        created_at=datetime.utcnow(),
        direction="LONG",
        target_type="PEAK_100",
        price=100.0,
        consensus_score=0.85,
        consensus_long=0.85,
        consensus_short=0.30,
        confidence=confidence,
        rf_prob=0.80,
        xgb_prob=0.82,
        lgb_prob=0.84,
        lstm_prob=0.0,
        iso_score=0.70,
        target_1=150.0,
        target_2=200.0,
        target_3=300.0,
        stop_loss=85.0,
        risk_reward=3.33,
        top_features=[("rsi_14", 65.0), ("atr_14", 2.0)],
        notified=False,
    )


@pytest.mark.unit
class TestTierFilter:
    def test_tier_order_is_monotonic(self) -> None:
        assert TIER_ORDER["ULTRA"] > TIER_ORDER["HIGH"]
        assert TIER_ORDER["HIGH"] > TIER_ORDER["MEDIUM"]
        assert TIER_ORDER["MEDIUM"] > TIER_ORDER["LOW"]

    def test_all_tiers_have_theme(self) -> None:
        for tier in ("ULTRA", "HIGH", "MEDIUM", "LOW"):
            assert tier in TIER_THEMES

    @patch("signals.notifier.settings")
    def test_medium_threshold_accepts_high_and_ultra(self, mock_settings) -> None:
        mock_settings.notify_min_confidence = "MEDIUM"
        assert _should_notify(_mock_signal("ULTRA"))
        assert _should_notify(_mock_signal("HIGH"))
        assert _should_notify(_mock_signal("MEDIUM"))

    @patch("signals.notifier.settings")
    def test_medium_threshold_rejects_low(self, mock_settings) -> None:
        mock_settings.notify_min_confidence = "MEDIUM"
        assert not _should_notify(_mock_signal("LOW"))

    @patch("signals.notifier.settings")
    def test_high_threshold_rejects_medium(self, mock_settings) -> None:
        mock_settings.notify_min_confidence = "HIGH"
        assert _should_notify(_mock_signal("ULTRA"))
        assert _should_notify(_mock_signal("HIGH"))
        assert not _should_notify(_mock_signal("MEDIUM"))
        assert not _should_notify(_mock_signal("LOW"))

    @patch("signals.notifier.settings")
    def test_ultra_threshold_only_ultra(self, mock_settings) -> None:
        mock_settings.notify_min_confidence = "ULTRA"
        assert _should_notify(_mock_signal("ULTRA"))
        assert not _should_notify(_mock_signal("HIGH"))

    @patch("signals.notifier.settings")
    def test_low_threshold_accepts_all(self, mock_settings) -> None:
        mock_settings.notify_min_confidence = "LOW"
        for tier in ("ULTRA", "HIGH", "MEDIUM", "LOW"):
            assert _should_notify(_mock_signal(tier))


@pytest.mark.unit
class TestEmailComposition:
    @patch("signals.notifier.settings")
    def test_subject_contains_tier_emoji_and_ticker(self, mock_settings) -> None:
        mock_settings.smtp_user = "from@test.com"
        mock_settings.alert_to_email = "to@test.com"
        msg = _compose(_mock_signal("ULTRA", "AAPL"))
        subject = msg["Subject"]
        assert "AAPL" in subject
        assert "🔥" in subject
        assert "ULTRA" in subject

    @patch("signals.notifier.settings")
    def test_high_tier_uses_diamond_emoji(self, mock_settings) -> None:
        mock_settings.smtp_user = "from@test.com"
        mock_settings.alert_to_email = "to@test.com"
        msg = _compose(_mock_signal("HIGH", "MSFT"))
        assert "💎" in msg["Subject"]
        assert "FORT" in msg["Subject"]

    @patch("signals.notifier.settings")
    def test_medium_tier_uses_target_emoji(self, mock_settings) -> None:
        mock_settings.smtp_user = "from@test.com"
        mock_settings.alert_to_email = "to@test.com"
        msg = _compose(_mock_signal("MEDIUM", "TSLA"))
        assert "🎯" in msg["Subject"]
        assert "MOYEN" in msg["Subject"]

    @patch("signals.notifier.settings")
    def test_html_body_contains_targets_and_stop(self, mock_settings) -> None:
        mock_settings.smtp_user = "from@test.com"
        mock_settings.alert_to_email = "to@test.com"
        msg = _compose(_mock_signal("HIGH"))
        # Extract decoded HTML payload (MIME may base64-encode it)
        html_part = msg.get_payload()[0]
        body = html_part.get_payload(decode=True).decode("utf-8")
        assert "150.00" in body  # target_1
        assert "200.00" in body  # target_2
        assert "85.00" in body   # stop_loss
        assert "Plan de trade" in body
        assert "Accord modèles ML" in body
