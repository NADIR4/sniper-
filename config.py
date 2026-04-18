"""Configuration centrale chargée depuis .env (local) ou st.secrets (Streamlit Cloud)."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def _streamlit_secrets() -> dict:
    """Récupère les secrets Streamlit Cloud si dispo (sinon dict vide)."""
    try:
        import streamlit as st  # noqa: WPS433 — import conditionnel
        # st.secrets accède au fichier .streamlit/secrets.toml OU aux secrets cloud
        return dict(st.secrets) if hasattr(st, "secrets") and st.secrets else {}
    except Exception:
        return {}


_SECRETS = _streamlit_secrets()


def _env(key: str, default: str) -> str:
    """Lit depuis (par ordre de priorité) : os.environ > st.secrets > default."""
    val = os.getenv(key)
    if val is not None:
        return val
    if key in _SECRETS:
        return str(_SECRETS[key])
    return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_env: str = _env("APP_ENV", "development")
    log_level: str = _env("LOG_LEVEL", "INFO")

    db_path: Path = ROOT / _env("DB_PATH", "cache/sniper.db")
    models_dir: Path = ROOT / _env("MODELS_DIR", "ml/models")
    exports_dir: Path = ROOT / _env("EXPORTS_DIR", "exports")
    logs_dir: Path = ROOT / "logs"

    scan_interval_minutes: int = _env_int("SCAN_INTERVAL_MINUTES", 15)
    consensus_threshold: float = _env_float("CONSENSUS_THRESHOLD", 0.75)
    lookback_days: int = _env_int("LOOKBACK_DAYS", 60)
    history_years: int = _env_int("HISTORY_YEARS", 3)
    min_gain_pct: float = _env_float("MIN_GAIN_PCT", 100.0)

    smtp_host: str = _env("SMTP_HOST", "")
    smtp_port: int = _env_int("SMTP_PORT", 587)
    smtp_user: str = _env("SMTP_USER", "")
    smtp_password: str = _env("SMTP_PASSWORD", "")
    alert_to_email: str = _env("ALERT_TO_EMAIL", "")
    # Seuil minimum de confiance pour envoyer un email
    # Valeurs: ULTRA | HIGH | MEDIUM | LOW (LOW = tout envoyer)
    notify_min_confidence: str = _env("NOTIFY_MIN_CONFIDENCE", "MEDIUM").upper()

    # Mot de passe pour accéder à l'app en ligne (vide = pas de gate, accès public)
    app_password: str = _env("APP_PASSWORD", "")


settings = Settings()

for d in (settings.db_path.parent, settings.models_dir, settings.exports_dir, settings.logs_dir):
    d.mkdir(parents=True, exist_ok=True)
