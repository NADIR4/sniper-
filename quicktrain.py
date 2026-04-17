"""Entraînement rapide avec un petit univers pour démarrage immédiat."""
from __future__ import annotations

from loguru import logger

from data.collector import collect_history
from ml.trainer import train_all


SMALL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "NFLX", "CRM", "ORCL", "ADBE", "COST", "LLY",
    "ASML.AS", "SAP.DE", "OR.PA", "MC.PA", "LVMH.PA",
]


def main() -> None:
    logger.info("=== QUICK TRAIN (20 tickers) ===")
    panels = collect_history(SMALL_UNIVERSE, years=3)
    metrics = train_all(panels)
    logger.info(f"RF AUC = {metrics['random_forest']['roc_auc']:.3f}")
    logger.info(f"XGB AUC = {metrics['xgboost']['roc_auc']:.3f}")
    logger.info("✅ Entraînement terminé. Lance `streamlit run app.py`")


if __name__ == "__main__":
    main()
