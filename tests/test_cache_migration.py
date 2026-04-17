"""Tests pour la migration idempotente du schéma Signal (Sprint 1)."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
)
from sqlalchemy.orm import declarative_base

from data.cache import Base, _NEW_SIGNAL_COLUMNS, migrate_signals


def _columns(engine, table: str) -> set[str]:
    return {col["name"] for col in inspect(engine).get_columns(table)}


def test_fresh_db_has_bidirectional_columns(tmp_path: Path) -> None:
    """DB fraîche créée via Base.metadata contient déjà les nouvelles colonnes."""
    db = tmp_path / "fresh.db"
    eng = create_engine(f"sqlite:///{db}", future=True)
    Base.metadata.create_all(eng)
    migrate_signals(eng)  # no-op attendu

    cols = _columns(eng, "signals")
    expected = set(_NEW_SIGNAL_COLUMNS.keys())
    assert expected.issubset(cols), f"colonnes manquantes : {expected - cols}"


def test_migration_idempotent_on_fresh_db(tmp_path: Path) -> None:
    db = tmp_path / "idem.db"
    eng = create_engine(f"sqlite:///{db}", future=True)
    Base.metadata.create_all(eng)

    cols_before = _columns(eng, "signals")
    migrate_signals(eng)
    migrate_signals(eng)  # 2e appel : ne doit rien casser
    cols_after = _columns(eng, "signals")
    assert cols_before == cols_after


def test_migration_adds_missing_columns_on_legacy_table(tmp_path: Path) -> None:
    """Simule une DB legacy sans les nouvelles colonnes, vérifie l'ajout."""
    db = tmp_path / "legacy.db"
    eng = create_engine(f"sqlite:///{db}", future=True)
    LegacyBase = declarative_base()

    class LegacySignal(LegacyBase):  # type: ignore[misc, valid-type]
        __tablename__ = "signals"
        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20))
        price = Column(Float)

    LegacyBase.metadata.create_all(eng)
    cols_before = _columns(eng, "signals")
    assert "direction" not in cols_before

    migrate_signals(eng)

    cols_after = _columns(eng, "signals")
    assert "direction" in cols_after
    assert "consensus_long" in cols_after
    assert "consensus_short" in cols_after
    assert "sentiment_score" in cols_after
    assert "feature_version" in cols_after


def test_migration_no_signals_table_noop(tmp_path: Path) -> None:
    """Si la table n'existe pas, ne doit rien faire et ne pas lever."""
    db = tmp_path / "empty.db"
    eng = create_engine(f"sqlite:///{db}", future=True)
    migrate_signals(eng)  # ne doit pas lever
    assert "signals" not in inspect(eng).get_table_names()
