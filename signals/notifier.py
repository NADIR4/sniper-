"""Envoi d'emails pour les signaux via SMTP.

Filtrage par tier de confiance (NOTIFY_MIN_CONFIDENCE) et templates HTML
personnalisés par tier (ULTRA / HIGH / MEDIUM / LOW).
"""
from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

from config import settings
from data.cache import Signal, SessionLocal


# ─────────────────────────────────────────────────────────────────────────────
# Tiers & templates
# ─────────────────────────────────────────────────────────────────────────────
TIER_ORDER: dict[str, int] = {"ULTRA": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass(frozen=True)
class TierTheme:
    """Thème visuel et éditorial d'un tier."""
    emoji: str
    label: str
    urgency: str
    primary: str      # couleur principale
    secondary: str    # couleur secondaire pour accent
    bg: str           # couleur de fond du bandeau
    tagline: str      # accroche personnalisée


TIER_THEMES: dict[str, TierTheme] = {
    "ULTRA": TierTheme(
        emoji="🔥",
        label="SIGNAL ULTRA",
        urgency="URGENT · conviction maximale",
        primary="#F59E0B",
        secondary="#EF4444",
        bg="linear-gradient(135deg,#F59E0B 0%,#EF4444 100%)",
        tagline="Configuration rare — tous les modèles alignés avec score > 90%.",
    ),
    "HIGH": TierTheme(
        emoji="💎",
        label="SIGNAL FORT",
        urgency="Opportunité solide",
        primary="#10B981",
        secondary="#059669",
        bg="linear-gradient(135deg,#10B981 0%,#059669 100%)",
        tagline="Consensus ML élevé — setup à fort potentiel validé par 4+ modèles.",
    ),
    "MEDIUM": TierTheme(
        emoji="🎯",
        label="SIGNAL MOYEN",
        urgency="À surveiller",
        primary="#3B82F6",
        secondary="#1D4ED8",
        bg="linear-gradient(135deg,#3B82F6 0%,#1D4ED8 100%)",
        tagline="Setup intéressant — à valider avec son propre money management.",
    ),
    "LOW": TierTheme(
        emoji="📊",
        label="SIGNAL FAIBLE",
        urgency="Info uniquement",
        primary="#6B7280",
        secondary="#4B5563",
        bg="linear-gradient(135deg,#6B7280 0%,#4B5563 100%)",
        tagline="Signal de faible conviction — informatif seulement.",
    ),
}


def _should_notify(signal: Signal) -> bool:
    """Retourne True si le tier du signal est >= au seuil configuré."""
    conf = (signal.confidence or "LOW").upper()
    threshold = settings.notify_min_confidence.upper()
    tier_rank = TIER_ORDER.get(conf, 0)
    threshold_rank = TIER_ORDER.get(threshold, TIER_ORDER["MEDIUM"])
    return tier_rank >= threshold_rank


# ─────────────────────────────────────────────────────────────────────────────
# Composition HTML par tier
# ─────────────────────────────────────────────────────────────────────────────
def _render_html(signal: Signal, theme: TierTheme) -> str:
    """HTML email pro, thème dark, couleurs du tier."""
    direction = (signal.direction or "LONG").upper()
    dir_color = "#10B981" if direction == "LONG" else "#EF4444"
    dir_arrow = "📈" if direction == "LONG" else "📉"

    rf = signal.rf_prob or 0.0
    xgb = signal.xgb_prob or 0.0
    lgb = signal.lgb_prob or 0.0
    lstm = signal.lstm_prob or 0.0
    iso = signal.iso_score or 0.0

    feats_html = "".join(
        f'<li style="margin:2px 0;"><code style="color:{theme.primary};">{name}</code> = {val:.3f}</li>'
        for name, val in (signal.top_features or [])[:5]
    ) or "<li>—</li>"

    return f"""
    <!DOCTYPE html>
    <html><body style="margin:0;padding:0;background:#0A0E1A;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#F1F5F9;">
      <div style="max-width:640px;margin:20px auto;background:#0F1628;border-radius:16px;overflow:hidden;border:1px solid rgba(148,163,184,0.12);">

        <!-- Bandeau tier -->
        <div style="background:{theme.bg};padding:22px 28px;color:white;">
          <div style="font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;opacity:0.9;">
            {theme.urgency}
          </div>
          <div style="font-size:1.7rem;font-weight:800;margin-top:4px;">
            {theme.emoji} {theme.label} — {signal.ticker}
          </div>
          <div style="font-size:0.9rem;margin-top:6px;opacity:0.92;">
            {theme.tagline}
          </div>
        </div>

        <!-- KPIs principaux -->
        <div style="padding:24px 28px;">
          <table style="width:100%;border-collapse:separate;border-spacing:8px;">
            <tr>
              <td style="background:rgba(30,41,59,0.6);border-radius:10px;padding:12px;width:33%;">
                <div style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em;">Score consensus</div>
                <div style="font-size:1.6rem;font-weight:800;color:{theme.primary};">{signal.consensus_score:.1%}</div>
              </td>
              <td style="background:rgba(30,41,59,0.6);border-radius:10px;padding:12px;width:33%;">
                <div style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em;">Direction</div>
                <div style="font-size:1.6rem;font-weight:800;color:{dir_color};">{dir_arrow} {direction}</div>
              </td>
              <td style="background:rgba(30,41,59,0.6);border-radius:10px;padding:12px;width:33%;">
                <div style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em;">Prix</div>
                <div style="font-size:1.6rem;font-weight:800;color:#F1F5F9;">{signal.price:.2f}</div>
              </td>
            </tr>
          </table>

          <!-- Plan de trade -->
          <h3 style="color:#F1F5F9;margin:24px 0 10px 0;font-size:1rem;">📍 Plan de trade</h3>
          <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
            <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
              <td style="padding:8px 4px;color:#94A3B8;">🎯 Target 1 (+50%)</td>
              <td style="padding:8px 4px;text-align:right;color:{theme.primary};font-weight:600;">{signal.target_1:.2f}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
              <td style="padding:8px 4px;color:#94A3B8;">🎯 Target 2 (+100%)</td>
              <td style="padding:8px 4px;text-align:right;color:{theme.primary};font-weight:600;">{signal.target_2:.2f}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
              <td style="padding:8px 4px;color:#94A3B8;">🎯 Target 3 (+200%)</td>
              <td style="padding:8px 4px;text-align:right;color:{theme.primary};font-weight:600;">{signal.target_3:.2f}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
              <td style="padding:8px 4px;color:#94A3B8;">🛑 Stop-loss</td>
              <td style="padding:8px 4px;text-align:right;color:#EF4444;font-weight:600;">{signal.stop_loss:.2f}</td>
            </tr>
            <tr>
              <td style="padding:8px 4px;color:#94A3B8;">⚖️ Risk / Reward</td>
              <td style="padding:8px 4px;text-align:right;color:#F1F5F9;font-weight:600;">{signal.risk_reward:.2f}</td>
            </tr>
          </table>

          <!-- Accord modèles ML -->
          <h3 style="color:#F1F5F9;margin:24px 0 10px 0;font-size:1rem;">🧠 Accord modèles ML</h3>
          <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
            <tr><td style="padding:6px 4px;color:#94A3B8;">Random Forest</td><td style="text-align:right;color:#F1F5F9;">{rf:.1%}</td></tr>
            <tr><td style="padding:6px 4px;color:#94A3B8;">XGBoost</td><td style="text-align:right;color:#F1F5F9;">{xgb:.1%}</td></tr>
            <tr><td style="padding:6px 4px;color:#94A3B8;">LightGBM</td><td style="text-align:right;color:#F1F5F9;">{lgb:.1%}</td></tr>
            <tr><td style="padding:6px 4px;color:#94A3B8;">LSTM</td><td style="text-align:right;color:#F1F5F9;">{lstm:.1%}</td></tr>
            <tr><td style="padding:6px 4px;color:#94A3B8;">Isolation Forest</td><td style="text-align:right;color:#F1F5F9;">{iso:.1%}</td></tr>
          </table>

          <!-- Top features -->
          <h3 style="color:#F1F5F9;margin:24px 0 10px 0;font-size:1rem;">🔍 Top features contributives</h3>
          <ul style="color:#CBD5E1;font-size:0.88rem;padding-left:20px;margin:0;">
            {feats_html}
          </ul>

          <!-- Métadonnées -->
          <div style="margin-top:26px;padding-top:16px;border-top:1px solid rgba(148,163,184,0.12);font-size:0.78rem;color:#64748B;">
            <div>📅 Détecté le {signal.created_at.strftime('%d/%m/%Y à %H:%M UTC')}</div>
            <div>🎯 Investment Sniper Bot · Consensus multi-modèles</div>
          </div>
        </div>
      </div>
    </body></html>
    """


def _compose(signal: Signal) -> MIMEMultipart:
    """Construit le MIMEMultipart avec subject + body HTML personnalisés par tier."""
    conf = (signal.confidence or "LOW").upper()
    theme = TIER_THEMES.get(conf, TIER_THEMES["LOW"])
    direction = (signal.direction or "LONG").upper()

    subject = (
        f"{theme.emoji} [{theme.label}] {signal.ticker} {direction} · "
        f"score {signal.consensus_score:.0%}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_user
    msg["To"] = settings.alert_to_email
    msg.attach(MIMEText(_render_html(signal, theme), "html"))
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Envoi
# ─────────────────────────────────────────────────────────────────────────────
def send_signal_email(signal: Signal) -> bool:
    """Envoie un email pour un signal. Retourne True si envoyé avec succès."""
    if not settings.smtp_host or not settings.smtp_user or not settings.alert_to_email:
        logger.info(f"[notifier] SMTP non configuré — skip {signal.ticker}")
        return False

    if not _should_notify(signal):
        logger.debug(
            f"[notifier] {signal.ticker} tier={signal.confidence} "
            f"< seuil={settings.notify_min_confidence} → skip"
        )
        return False

    try:
        msg = _compose(signal)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, [settings.alert_to_email], msg.as_string())
        logger.info(
            f"[notifier] ✉️  {signal.confidence} envoyé · {signal.ticker} "
            f"({signal.direction}) score={signal.consensus_score:.1%}"
        )
        return True
    except Exception as e:
        logger.error(f"[notifier] ❌ Échec envoi {signal.ticker}: {e}")
        return False


def notify_pending() -> int:
    """Notifie tous les signaux non encore envoyés qui passent le filtre de tier.

    Les signaux filtrés (tier < seuil) sont marqués `notified=True` pour éviter
    qu'ils restent en file d'attente indéfiniment.
    """
    with SessionLocal() as session:
        pending = session.query(Signal).filter_by(notified=False).all()
        sent = 0
        skipped = 0
        for s in pending:
            if not _should_notify(s):
                s.notified = True  # marqué traité — évite la ré-évaluation permanente
                skipped += 1
                continue
            if send_signal_email(s):
                s.notified = True
                sent += 1
        session.commit()

    if pending:
        logger.info(
            f"[notifier] 📬 Résumé : {sent} envoyés · {skipped} filtrés "
            f"(seuil={settings.notify_min_confidence}) · {len(pending)} total"
        )
    return sent
