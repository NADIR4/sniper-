"""Bootstrap des modèles ML au premier démarrage.

Utilisé pour le déploiement Streamlit Cloud : les `.pkl` ne sont pas dans Git
(trop lourds). Au 1er boot, si les modèles manquent on les télécharge depuis
une GitHub Release.

Variables d'env :
  - MODELS_RELEASE_URL : URL directe d'un ZIP contenant les .pkl
    (ex: https://github.com/USER/REPO/releases/download/v1.0/models.zip)
  - MODELS_BOOTSTRAP_TIMEOUT : timeout download en secondes (défaut 300)

Si MODELS_RELEASE_URL n'est pas définie et que les modèles manquent,
`bootstrap_models()` ne fait rien (fallback silencieux — l'app continuera
avec le message "Modèles non entraînés").
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import requests
from loguru import logger

from config import settings


_REQUIRED = ("rf_long.pkl", "xgb_long.pkl", "scaler_long.pkl")


def models_are_present() -> bool:
    """Retourne True si tous les modèles obligatoires sont présents."""
    return all((settings.models_dir / f).exists() for f in _REQUIRED)


def bootstrap_models(*, force: bool = False) -> bool:
    """Télécharge + dézippe les modèles depuis MODELS_RELEASE_URL si nécessaire.

    Retourne True si les modèles sont prêts après l'appel (déjà là OU bootstrap
    réussi), False sinon.
    """
    if not force and models_are_present():
        return True

    url = os.getenv("MODELS_RELEASE_URL", "").strip()
    if not url:
        logger.warning(
            "[bootstrap] MODELS_RELEASE_URL non définie — "
            "lance `python main.py train` en local pour générer les modèles."
        )
        return False

    timeout = int(os.getenv("MODELS_BOOTSTRAP_TIMEOUT", "300"))
    logger.info(f"[bootstrap] 📥 Téléchargement des modèles depuis {url}…")

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        # Télécharge en mémoire (plus simple + compatible Streamlit Cloud tmpfs)
        buf = io.BytesIO()
        total = 0
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
            buf.write(chunk)
            total += len(chunk)
        buf.seek(0)
        logger.info(f"[bootstrap] ✅ Téléchargé {total / 1_000_000:.1f} MB, extraction…")

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(settings.models_dir)

        if models_are_present():
            logger.info(f"[bootstrap] ✅ Modèles extraits dans {settings.models_dir}")
            return True
        logger.error(
            f"[bootstrap] ❌ Extraction OK mais modèles obligatoires manquants : "
            f"{[f for f in _REQUIRED if not (settings.models_dir / f).exists()]}"
        )
        return False

    except Exception as exc:
        logger.error(f"[bootstrap] ❌ Échec download/extract : {exc}")
        return False


if __name__ == "__main__":
    ok = bootstrap_models(force=True)
    raise SystemExit(0 if ok else 1)
