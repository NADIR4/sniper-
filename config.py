"""Configuration centrale chargée depuis .env."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


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


settings = Settings()

for d in (settings.db_path.parent, settings.models_dir, settings.exports_dir, settings.logs_dir):
    d.mkdir(parents=True, exist_ok=True)
