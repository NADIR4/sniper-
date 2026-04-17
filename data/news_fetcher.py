"""Fetch & cache des news horodatées par ticker (Sprint 3).

Pipeline : yfinance → normalisation UTC → score VADER → upsert SQLite.
Sépare fetch (I/O) et scoring (CPU) en deux passes pour permettre le parallélisme.
Rate-limit friendly : 0.3s entre appels par défaut.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy.exc import IntegrityError

from data.cache import News, SessionLocal
from ml.sentiment import SentimentScorer, get_scorer


def _utc_now_naive() -> datetime:
    """datetime.utcnow() est déprécié en 3.12+ — on force explicitement UTC naïf."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    published_at: datetime  # UTC naïf
    title: str
    summary: str
    source: str
    link: str


def _normalize_utc(raw) -> datetime | None:
    """Normalise un timestamp yfinance en datetime UTC naïf.

    Accepte : int/float (epoch), str ISO, datetime. Retourne None si invalide.
    """
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).replace(tzinfo=None)
        if isinstance(raw, datetime):
            if raw.tzinfo is not None:
                return raw.astimezone(timezone.utc).replace(tzinfo=None)
            return raw
        if isinstance(raw, str):
            ts = pd.to_datetime(raw, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.to_pydatetime().replace(tzinfo=None)
    except Exception as exc:
        logger.debug(f"[news_fetcher] parse date {raw!r}: {exc}")
    return None


def _extract_yfinance(ticker: str, max_items: int = 50) -> list[NewsItem]:
    """Récupère les articles bruts depuis yfinance et normalise les champs."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.debug(f"[news_fetcher] yfinance {ticker}: {exc}")
        return []

    out: list[NewsItem] = []
    for item in raw[:max_items]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = (content.get("title") or item.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or "")[:500]
        published_raw = (
            content.get("pubDate")
            or item.get("providerPublishTime")
            or content.get("displayTime")
        )
        published_at = _normalize_utc(published_raw)
        if published_at is None:
            continue
        link = ""
        cu = content.get("canonicalUrl")
        if isinstance(cu, dict):
            link = cu.get("url", "") or ""
        out.append(
            NewsItem(
                ticker=ticker,
                published_at=published_at,
                title=title[:512],
                summary=summary,
                source="yfinance",
                link=link[:1024],
            )
        )
    return out


def _existing_keys(ticker: str, session) -> set[tuple[datetime, str]]:
    """Clés (published_at, title) déjà en base pour ce ticker → dédup à l'insert."""
    rows = session.query(News.published_at, News.title).filter(News.ticker == ticker).all()
    return {(r[0], r[1]) for r in rows}


def cache_news_for_ticker(
    ticker: str,
    scorer: SentimentScorer | None = None,
    max_items: int = 50,
) -> int:
    """Fetch + score + upsert pour un ticker. Retourne le nombre de nouveaux articles."""
    scorer = scorer or get_scorer()
    items = _extract_yfinance(ticker, max_items=max_items)
    if not items:
        return 0

    with SessionLocal() as session:
        existing = _existing_keys(ticker, session)
        to_insert = [i for i in items if (i.published_at, i.title) not in existing]
        if not to_insert:
            return 0

        texts = [f"{i.title} {i.summary}".strip() for i in to_insert]
        scores = scorer.score(texts)

        now = _utc_now_naive()
        for item, score in zip(to_insert, scores):
            session.add(
                News(
                    ticker=item.ticker,
                    published_at=item.published_at,
                    fetched_at=now,
                    title=item.title,
                    summary=item.summary,
                    source=item.source,
                    link=item.link,
                    sentiment_score=float(score),
                    scorer_version=scorer.version,
                )
            )
        # Race-condition-safe : si un autre writer a inséré en parallèle,
        # la UniqueConstraint lève IntegrityError → on rollback et renvoie 0.
        try:
            session.commit()
        except IntegrityError as exc:
            logger.debug(f"[news_fetcher] dédup concurrente {ticker}: {exc}")
            session.rollback()
            return 0
    return len(to_insert)


def cache_news_for_tickers(
    tickers: Iterable[str],
    scorer: SentimentScorer | None = None,
    throttle_sec: float = 0.3,
) -> dict[str, int]:
    """Cache news pour un lot de tickers, avec throttle anti rate-limit yfinance."""
    scorer = scorer or get_scorer()
    stats: dict[str, int] = {}
    for t in tickers:
        try:
            stats[t] = cache_news_for_ticker(t, scorer=scorer)
        except Exception as exc:
            logger.warning(f"[news_fetcher] {t}: {exc}")
            stats[t] = 0
        time.sleep(throttle_sec)
    total = sum(stats.values())
    logger.info(f"[news_fetcher] {total} nouveaux articles cachés sur {len(stats)} tickers")
    return stats


def load_news_df(ticker: str) -> pd.DataFrame:
    """Charge toutes les news d'un ticker depuis la DB, indexées par published_at UTC naïf."""
    with SessionLocal() as session:
        rows = (
            session.query(News.published_at, News.sentiment_score, News.title)
            .filter(News.ticker == ticker)
            .order_by(News.published_at)
            .all()
        )
    if not rows:
        return pd.DataFrame(columns=["sentiment_score", "title"]).set_index(
            pd.DatetimeIndex([], name="published_at")
        )
    df = pd.DataFrame(rows, columns=["published_at", "sentiment_score", "title"])
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df.set_index("published_at").sort_index()
