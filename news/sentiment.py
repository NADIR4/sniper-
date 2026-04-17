"""Analyse de sentiment des news via yfinance + VADER."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import yfinance as yf
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()

# Lexique enrichi pour la finance
_FINANCE_BOOST = {
    "beat": 2.5, "beats": 2.5, "surge": 3.0, "surged": 3.0, "surges": 3.0,
    "soar": 3.0, "soared": 3.0, "soars": 3.0, "skyrocket": 3.5,
    "rally": 2.0, "rallies": 2.0, "upgrade": 2.0, "upgraded": 2.0,
    "buy": 1.5, "bullish": 2.5, "breakout": 2.5, "outperform": 2.0,
    "record": 1.5, "profit": 1.5, "growth": 1.5,
    "miss": -2.5, "missed": -2.5, "plunge": -3.0, "plunged": -3.0,
    "crash": -3.5, "crashed": -3.5, "tumble": -2.5, "tumbled": -2.5,
    "downgrade": -2.0, "downgraded": -2.0, "sell": -1.5, "bearish": -2.5,
    "lawsuit": -2.0, "investigation": -2.0, "fraud": -3.0, "bankruptcy": -3.5,
    "layoff": -2.0, "layoffs": -2.0, "loss": -1.5, "decline": -1.5,
}
_ANALYZER.lexicon.update(_FINANCE_BOOST)


@dataclass
class NewsSentiment:
    ticker: str
    n_articles: int
    avg_compound: float
    max_compound: float
    min_compound: float
    positive_ratio: float
    negative_ratio: float
    latest_title: str = ""
    latest_date: str = ""
    articles: list[dict] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _score_text(text: str) -> float:
    if not text:
        return 0.0
    return float(_ANALYZER.polarity_scores(text)["compound"])


def _fetch_yfinance_news(ticker: str, max_items: int = 20) -> list[dict]:
    try:
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.news or []
    except Exception as e:
        logger.debug(f"[news] yfinance {ticker}: {e}")
        return []

    items = []
    for item in raw[:max_items]:
        content = item.get("content", item)
        title = content.get("title", "") or item.get("title", "")
        if not title:
            continue
        summary = content.get("summary", "") or ""
        published = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(published).isoformat()
        elif not isinstance(published, str):
            published = ""
        items.append({
            "title": title,
            "summary": summary[:300],
            "published": published,
            "link": content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else "",
        })
    return items


def _fetch_google_news_rss(ticker: str, max_items: int = 10) -> list[dict]:
    try:
        query = ticker.split(".")[0] + " stock"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
            })
        return items
    except Exception as e:
        logger.debug(f"[news] google rss {ticker}: {e}")
        return []


def analyze_ticker_news(ticker: str, max_items: int = 25) -> NewsSentiment:
    """Récupère les news du ticker et calcule un score de sentiment agrégé."""
    articles = _fetch_yfinance_news(ticker, max_items=max_items)
    if len(articles) < 3:
        articles += _fetch_google_news_rss(ticker, max_items=10)

    if not articles:
        return NewsSentiment(ticker, 0, 0.0, 0.0, 0.0, 0.0, 0.0, articles=[])

    scores = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('summary', '')}".strip()
        s = _score_text(text)
        a["sentiment"] = s
        scores.append(s)

    latest = articles[0] if articles else {}

    return NewsSentiment(
        ticker=ticker,
        n_articles=len(articles),
        avg_compound=sum(scores) / len(scores),
        max_compound=max(scores),
        min_compound=min(scores),
        positive_ratio=sum(1 for s in scores if s > 0.2) / len(scores),
        negative_ratio=sum(1 for s in scores if s < -0.2) / len(scores),
        latest_title=latest.get("title", "")[:200],
        latest_date=latest.get("published", ""),
        articles=articles,
    )


def classify_sentiment(score: float) -> str:
    if score >= 0.4:
        return "TRÈS POSITIF"
    if score >= 0.15:
        return "POSITIF"
    if score <= -0.4:
        return "TRÈS NÉGATIF"
    if score <= -0.15:
        return "NÉGATIF"
    return "NEUTRE"
