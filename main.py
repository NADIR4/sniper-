"""Pipeline principal CLI : collect, train, scan, report, schedule."""
from __future__ import annotations

import sys
import time
from datetime import datetime

from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler

from config import settings
from data.collector import collect_history, filter_gainers
from data.news_fetcher import cache_news_for_tickers
from data.universe import get_universe
from ml.trainer import train_all
from ml.scanner import scan_market
from signals.generator import process_scan
from signals.notifier import notify_pending
from reports.exporter import export_report


logger.add(settings.logs_dir / "sniper_{time}.log", rotation="10 MB", level=settings.log_level)


def cmd_collect() -> None:
    logger.info("=== COLLECT ===")
    collect_history(get_universe())


def cmd_train() -> None:
    logger.info("=== TRAIN ===")
    panels = collect_history(get_universe())
    gainers = filter_gainers(panels)
    logger.info(f"[train] {len(gainers)}/{len(panels)} actions gardées (+{settings.min_gain_pct}%)")
    if len(gainers) < 5:
        logger.warning("[train] Peu de gainers, utilisation de tout le dataset pour enrichir les négatifs")
    metrics = train_all(panels)
    logger.info(f"[train] Métriques : RF={metrics['random_forest']['roc_auc']:.3f}, XGB={metrics['xgboost']['roc_auc']:.3f}")


def cmd_fetch_news() -> None:
    """Rafraîchit le cache news (yfinance + VADER)."""
    logger.info("=== FETCH-NEWS ===")
    stats = cache_news_for_tickers(get_universe())
    total = sum(stats.values())
    logger.info(f"[fetch-news] +{total} articles sur {len(stats)} tickers")


def cmd_scan() -> None:
    logger.info("=== SCAN ===")
    # Best-effort : rafraîchir les news avant le scan (features news à jour)
    try:
        cache_news_for_tickers(get_universe())
    except Exception as exc:
        logger.warning(f"[scan] fetch-news échoué (continue) : {exc}")
    results = scan_market()
    signals = process_scan(results)
    sent = notify_pending()
    logger.info(f"[scan] {len(signals)} signaux détectés, {sent} emails envoyés")


def cmd_report() -> None:
    logger.info("=== REPORT ===")
    path = export_report()
    logger.info(f"[report] → {path}")


def cmd_schedule() -> None:
    logger.info(f"=== SCHEDULE every {settings.scan_interval_minutes} min ===")
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(cmd_scan, "interval", minutes=settings.scan_interval_minutes,
                  next_run_time=datetime.utcnow())
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


COMMANDS = {
    "collect": cmd_collect,
    "train": cmd_train,
    "fetch-news": cmd_fetch_news,
    "scan": cmd_scan,
    "report": cmd_report,
    "schedule": cmd_schedule,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py [collect|train|fetch-news|scan|report|schedule]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
