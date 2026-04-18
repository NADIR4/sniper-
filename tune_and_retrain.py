"""Script one-shot : Optuna tuning puis re-train complet avec best_params.

Étapes :
1. Collecte 3 ans d'historique (cache)
2. Filtre gainers à MIN_GAIN_PCT=50 (nouveau seuil)
3. Build dataset LONG + SHORT
4. Tune XGB (LONG + SHORT) avec 20 trials
5. Tune LGB (LONG + SHORT) avec 20 trials
6. Persist best_params dans ml/models/best_params_*.json
7. Re-train complet (trainer.train_pipeline) — lit automatiquement les params
"""
from __future__ import annotations
import time
from loguru import logger

from data.collector import collect_history, filter_gainers
from data.universe import get_universe
from ml.features import build_training_dataset_dual
from ml.tuning import tune_xgb, tune_lgb
from ml.trainer import train_all
from config import settings

t_global = time.time()
logger.info(f"=== MIN_GAIN_PCT actuel : {settings.min_gain_pct}% ===")

logger.info("📡 Collecte historique (3 ans)…")
tickers = get_universe()
panels = collect_history(tickers, years=settings.history_years)
logger.info(f"📡 {len(panels)} panels collectés")

logger.info(f"🔎 Filtre gainers (+{settings.min_gain_pct}%)…")
gainers = filter_gainers(panels)
logger.info(f"🔎 {len(gainers)} gainers gardés sur {len(panels)}")

logger.info("🧬 Build dataset dual (LONG + SHORT)…")
duals = build_training_dataset_dual(gainers)
X_long, y_long, _ = duals["long"]
X_short, y_short, _ = duals["short"]
logger.info(f"🧬 LONG: {len(X_long)} rows, {int(y_long.sum())} pos | SHORT: {len(X_short)} rows, {int(y_short.sum())} pos")

N_TRIALS = 20
logger.info(f"🎯 Optuna tuning (n_trials={N_TRIALS}/direction/modèle)…")

tune_xgb(X_long, y_long, "LONG", n_trials=N_TRIALS, timeout_sec=900)
tune_xgb(X_short, y_short, "SHORT", n_trials=N_TRIALS, timeout_sec=900)
tune_lgb(X_long, y_long, "LONG", n_trials=N_TRIALS, timeout_sec=900)
tune_lgb(X_short, y_short, "SHORT", n_trials=N_TRIALS, timeout_sec=900)

logger.info(f"🎯 Tuning terminé — {time.time() - t_global:.0f}s total")

logger.info("🚂 Re-train complet avec best_params…")
train_all(gainers)

logger.info(f"✅ Pipeline complet terminé en {time.time() - t_global:.0f}s")
