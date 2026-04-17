"""Envoi d'emails pour les signaux via SMTP."""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

from config import settings
from data.cache import Signal, SessionLocal


def _compose(signal: Signal) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[SNIPER] {signal.ticker} — {signal.confidence} — score {signal.consensus_score:.2%}"
    msg["From"] = settings.smtp_user
    msg["To"] = settings.alert_to_email

    html = f"""
    <h2>Signal détecté : {signal.ticker}</h2>
    <p><b>Date :</b> {signal.created_at}</p>
    <p><b>Prix :</b> {signal.price:.2f} | <b>Confiance :</b> {signal.confidence}</p>
    <p><b>Score consensus :</b> {signal.consensus_score:.2%}</p>
    <ul>
      <li>Random Forest : {signal.rf_prob:.2%}</li>
      <li>XGBoost : {signal.xgb_prob:.2%}</li>
      <li>LSTM : {signal.lstm_prob:.2%}</li>
      <li>Iso Forest : {signal.iso_score:.2%}</li>
    </ul>
    <p><b>Cibles :</b> +50% ({signal.target_1:.2f}) | +100% ({signal.target_2:.2f}) | +200% ({signal.target_3:.2f})</p>
    <p><b>Stop-loss :</b> {signal.stop_loss:.2f} | <b>R/R :</b> {signal.risk_reward:.2f}</p>
    <p><b>Top features :</b> {signal.top_features}</p>
    """
    msg.attach(MIMEText(html, "html"))
    return msg


def send_signal_email(signal: Signal) -> bool:
    if not settings.smtp_host or not settings.smtp_user or not settings.alert_to_email:
        logger.info(f"[notifier] SMTP non configuré, email skip pour {signal.ticker}")
        return False
    try:
        msg = _compose(signal)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, [settings.alert_to_email], msg.as_string())
        logger.info(f"[notifier] Email envoyé pour {signal.ticker}")
        return True
    except Exception as e:
        logger.error(f"[notifier] Échec envoi email {signal.ticker}: {e}")
        return False


def notify_pending() -> int:
    with SessionLocal() as session:
        pending = session.query(Signal).filter_by(notified=False).all()
        sent = 0
        for s in pending:
            if send_signal_email(s):
                s.notified = True
                sent += 1
        session.commit()
    return sent
