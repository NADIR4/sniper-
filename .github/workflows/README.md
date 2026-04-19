# GitHub Actions — Sniper Scan Automatique

Ce workflow lance **automatiquement un scan toutes les 30 min** pendant les heures de marché,
envoie les signaux par email et archive les résultats en artifacts.

## Configuration des secrets (à faire UNE fois)

Va sur : **GitHub → ton repo → Settings → Secrets and variables → Actions → New repository secret**

Crée ces secrets :

| Secret | Valeur | Requis |
|--------|--------|--------|
| `MODELS_RELEASE_URL` | URL ZIP modèles (ex: `https://github.com/NADIR4/sniper-/releases/download/models-v1/models.zip`) | ✅ |
| `SMTP_HOST` | `smtp.gmail.com` | ✅ |
| `SMTP_PORT` | `587` | ✅ |
| `SMTP_USER` | `ton_email@gmail.com` | ✅ |
| `SMTP_PASSWORD` | App Password Google (16 chars, pas ton mot de passe normal) | ✅ |
| `ALERT_TO_EMAIL` | Destinataire des alertes | ✅ |
| `NOTIFY_MIN_CONFIDENCE` | `MEDIUM` (ou `HIGH` / `ULTRA` / `LOW`) | ⚙️ |
| `CONSENSUS_THRESHOLD` | `0.60` | ⚙️ |
| `MIN_GAIN_PCT` | `50` | ⚙️ |

## Fréquence du cron

Actuel : `*/30 8-20 * * 1-5`
→ Toutes les 30 min, entre 08h et 20h UTC, du lundi au vendredi.

Pour modifier, édite `.github/workflows/scan.yml` ligne `- cron:` :

- `"*/15 * * * *"` — toutes les 15 min 24/7
- `"0 */1 * * *"` — une fois par heure
- `"0 13,16,19 * * 1-5"` — 3 fois par jour en semaine

Tester : [crontab.guru](https://crontab.guru/)

## Déclenchement manuel

**GitHub → Actions → Sniper Scan → Run workflow** → choisis la branche → Run.

## Artifacts générés

Chaque run archive pour 14 jours :
- `cache/sniper.db` — base SQLite avec signaux
- `exports/*.xlsx` — rapport Excel complet
- `logs/*.log` — logs détaillés

Téléchargement : **Actions → le run → Artifacts**.

## Quota GitHub Actions (gratuit)

- Repo public : **illimité**
- Repo privé : 2 000 min/mois → ~160 runs de 12 min = OK

Pour réduire si tu atteins la limite : passe à `*/60` (toutes les heures).
