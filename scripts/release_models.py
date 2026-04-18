"""Empaquette les modèles locaux dans un ZIP et les pousse en GitHub Release.

Usage :
    python scripts/release_models.py                 # tag auto = models-YYYYMMDD
    python scripts/release_models.py --tag v1.0.0    # tag custom

Prérequis :
    - `gh` CLI installée et authentifiée (`gh auth status`)
    - Le repo courant doit avoir une remote `origin` pointant sur GitHub

Après upload, l'URL directe du ZIP est affichée — copie-la dans la variable
d'env `MODELS_RELEASE_URL` côté Streamlit Cloud (Settings → Secrets).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Windows : force UTF-8 sur stdout pour émojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "ml" / "models"
ARTIFACT = ROOT / "dist" / "models.zip"

# Modèles à embarquer (on exclut les variantes legacy rf.pkl/xgb.pkl/scaler.pkl
# qui sont des doublons de rf_long/xgb_long/scaler_long)
INCLUDE = [
    "rf_long.pkl", "rf_short.pkl",
    "xgb_long.pkl", "xgb_short.pkl",
    "lgb_long.pkl", "lgb_short.pkl",
    "iso.pkl",
    "scaler_long.pkl", "scaler_short.pkl",
    "metrics.json",
    "best_params_lgb_LONG.json", "best_params_lgb_SHORT.json",
    "best_params_xgb_LONG.json", "best_params_xgb_SHORT.json",
]


def _zip_models() -> int:
    """Crée dist/models.zip. Retourne la taille en bytes."""
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with zipfile.ZipFile(ARTIFACT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in INCLUDE:
            src = MODELS_DIR / name
            if not src.exists():
                missing.append(name)
                continue
            zf.write(src, arcname=name)
            print(f"  + {name:35s} {src.stat().st_size / 1_000_000:7.2f} MB")
    if missing:
        print(f"\n⚠️  Absents (ignorés) : {missing}", file=sys.stderr)
    return ARTIFACT.stat().st_size


def _gh_release(tag: str, notes: str) -> None:
    """Crée (ou met à jour) la Release et y attache le ZIP."""
    # 1. Vérifie si la release existe déjà
    check = subprocess.run(
        ["gh", "release", "view", tag],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print(f"ℹ️  Release '{tag}' existe — upload en écrasement")
        subprocess.run(
            ["gh", "release", "upload", tag, str(ARTIFACT), "--clobber"],
            check=True,
        )
    else:
        print(f"✨ Création release '{tag}'")
        subprocess.run(
            ["gh", "release", "create", tag, str(ARTIFACT),
             "--title", f"ML models {tag}",
             "--notes", notes],
            check=True,
        )


def _download_url(tag: str) -> str:
    """Retourne l'URL directe du ZIP pour MODELS_RELEASE_URL."""
    repo = subprocess.run(
        ["gh", "repo", "view", "--json", "url", "--jq", ".url"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return f"{repo}/releases/download/{tag}/models.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=f"models-{datetime.now():%Y%m%d}")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"❌ Répertoire introuvable : {MODELS_DIR}", file=sys.stderr)
        return 1

    print(f"📦 Zip des modèles → {ARTIFACT.relative_to(ROOT)}")
    size = _zip_models()
    print(f"\n✅ Zip créé : {size / 1_000_000:.1f} MB\n")

    notes = f"Modèles ML entraînés le {datetime.now():%d/%m/%Y}\n\n" \
            f"À consommer via la variable d'env `MODELS_RELEASE_URL`."
    _gh_release(args.tag, notes)

    url = _download_url(args.tag)
    print("\n" + "=" * 70)
    print("✅ UPLOAD RÉUSSI")
    print("=" * 70)
    print(f"\nURL directe du ZIP :\n  {url}\n")
    print("Étapes suivantes :")
    print("  1. Streamlit Cloud → Settings → Secrets → ajoute :")
    print(f'       MODELS_RELEASE_URL = "{url}"')
    print("  2. Redémarre l'app → les modèles seront téléchargés au 1er boot")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
