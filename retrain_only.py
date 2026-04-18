"""Re-train complet uniquement (skip tuning, best_params déjà persisté).

Utilisé après un tune_and_retrain.py qui a planté au re-train.
"""
from __future__ import annotations
import time
from loguru import logger

from data.collector import collect_history, filter_gainers
from data.universe import get_universe
from ml.trainer import train_all
from config import settings

t_global = time.time()
logger.info(f"=== MIN_GAIN_PCT actuel : {settings.min_gain_pct}% ===")

logger.info("📡 Collecte historique (cache)…")
tickers = get_universe()
panels = collect_history(tickers, years=settings.history_years)
logger.info(f"📡 {len(panels)} panels collectés")

logger.info(f"🔎 Filtre gainers (+{settings.min_gain_pct}%)…")
gainers = filter_gainers(panels)
logger.info(f"🔎 {len(gainers)} gainers gardés sur {len(panels)}")

logger.info("🚂 Re-train complet avec best_params Optuna…")
train_all(gainers)

logger.info(f"✅ Re-train terminé en {time.time() - t_global:.0f}s")
