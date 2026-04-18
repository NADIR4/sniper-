"""Cache SQLite pour OHLCV et signaux via SQLAlchemy.

Schéma bidirectionnel (Sprint 1) : Signal supporte LONG/SHORT + news.
Migration idempotente : ALTER TABLE ADD COLUMN si absent.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone


def _utcnow() -> datetime:
    """UTC naïf sans utcnow() déprécié en 3.12+."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

# Bump à chaque changement de features/labels pour invalider caches + models
FEATURE_VERSION = "v4_lgb"

Base = declarative_base()
_engine = create_engine(f"sqlite:///{settings.db_path}", future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


class OHLCV(Base):
    __tablename__ = "ohlcv"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ohlcv_ticker_date"),)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), index=True, nullable=False)
    created_at = Column(DateTime, default=_utcnow, index=True)
    price = Column(Float)

    # Direction & scoring bidirectionnel
    direction = Column(String(8), default="LONG", index=True)  # LONG | SHORT
    target_type = Column(String(16), default="PEAK_100")       # PEAK_100 | CRASH_30
    consensus_score = Column(Float)       # score dans la direction émise
    consensus_long = Column(Float)        # proba LONG (pic haussier)
    consensus_short = Column(Float)       # proba SHORT (crash)

    # Probas individuelles (direction émise)
    rf_prob = Column(Float)
    xgb_prob = Column(Float)
    lgb_prob = Column(Float)
    lstm_prob = Column(Float)
    iso_score = Column(Float)

    confidence = Column(String(10))

    # Cibles & stops
    target_1 = Column(Float)
    target_2 = Column(Float)
    target_3 = Column(Float)
    stop_loss = Column(Float)
    risk_reward = Column(Float)

    # News sentiment
    sentiment_score = Column(Float, default=0.0)
    sentiment_label = Column(String(16), default="NEUTRE")
    n_news_articles = Column(Integer, default=0)

    # Traçabilité
    feature_version = Column(String(16), default=FEATURE_VERSION)
    top_features = Column(Text)
    features_json = Column(Text)
    notified = Column(Boolean, default=False)


class News(Base):
    """Article de news horodaté + score de sentiment (Sprint 3).

    Une ligne = un article unique par (ticker, published_at, title).
    `scorer_version` permet de recomputer les scores sans re-fetcher.
    """

    __tablename__ = "news"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    published_at = Column(DateTime, nullable=False)  # UTC naïf (normalisé)
    fetched_at = Column(DateTime, nullable=False, default=_utcnow)
    title = Column(String(512), nullable=False)
    summary = Column(Text, default="")
    source = Column(String(128), default="yfinance")
    link = Column(String(1024), default="")
    sentiment_score = Column(Float, nullable=False, default=0.0)  # [-1, 1]
    scorer_version = Column(String(32), default="vader_fin_v1")

    __table_args__ = (
        UniqueConstraint(
            "ticker", "published_at", "title", name="uq_news_ticker_published_title"
        ),
        Index("ix_news_ticker_published", "ticker", "published_at"),
    )


# Colonnes ajoutées au Sprint 1 — utilisées par _migrate_signals
_NEW_SIGNAL_COLUMNS: dict[str, str] = {
    "direction": "VARCHAR(8) DEFAULT 'LONG'",
    "target_type": "VARCHAR(16) DEFAULT 'PEAK_100'",
    "consensus_long": "FLOAT",
    "consensus_short": "FLOAT",
    "lgb_prob": "FLOAT",
    "sentiment_score": "FLOAT DEFAULT 0.0",
    "sentiment_label": "VARCHAR(16) DEFAULT 'NEUTRE'",
    "n_news_articles": "INTEGER DEFAULT 0",
    "feature_version": f"VARCHAR(16) DEFAULT '{FEATURE_VERSION}'",
}


_SAFE_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
_SAFE_DDL = re.compile(
    r"^(VARCHAR\(\d+\)|FLOAT|INTEGER|BOOLEAN)"
    r"(\s+DEFAULT\s+('[A-Za-z0-9_]{0,32}'|-?\d+(\.\d+)?))?$"
)


def migrate_signals(engine=None) -> None:
    """Migration idempotente : ajoute les colonnes bidirectionnelles si absentes.

    SQLite ne supporte pas ADD COLUMN IF NOT EXISTS avant v3.35 ; on introspecte.
    Paramètre `engine` optionnel pour les tests ; par défaut utilise _engine global.

    Sécurité : noms de colonnes et DDL validés par regex (pas d'injection via
    FEATURE_VERSION même si celle-ci devenait dynamique plus tard).
    """
    eng = engine if engine is not None else _engine
    from sqlalchemy.exc import OperationalError

    inspector = inspect(eng)
    if "signals" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("signals")}
    missing = {k: v for k, v in _NEW_SIGNAL_COLUMNS.items() if k not in existing}
    if not missing:
        return
    with eng.begin() as conn:
        for name, ddl in missing.items():
            if not _SAFE_IDENT.match(name):
                raise ValueError(f"Nom de colonne invalide : {name!r}")
            if not _SAFE_DDL.match(ddl):
                raise ValueError(f"DDL invalide pour {name!r} : {ddl!r}")
            try:
                conn.execute(text(f'ALTER TABLE signals ADD COLUMN "{name}" {ddl}'))
                logger.info(f"[cache] migration : colonne signals.{name} ajoutée")
            except OperationalError as exc:
                logger.warning(f"[cache] migration colonne {name} ignorée : {exc}")


Base.metadata.create_all(_engine)
migrate_signals()
