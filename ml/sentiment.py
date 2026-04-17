"""Scorer de sentiment : Protocol + VADER finance-boosted + FinBERT optionnel.

Sprint 3. Réutilise le lexique enrichi de `news/sentiment.py` via import léger.
Isole la logique de scoring pour pouvoir brancher FinBERT en production
sans toucher au pipeline de fetch/features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class SentimentScorer(Protocol):
    """Contrat : renvoie des scores compound dans [-1, 1]."""

    version: str

    def score(self, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class VaderScorer:
    """Wrapper autour de l'analyseur VADER enrichi finance (news/sentiment.py)."""

    version: str = "vader_fin_v1"

    def score(self, texts: list[str]) -> list[float]:
        # Import paresseux : charge le lexique enrichi une seule fois
        from news.sentiment import _ANALYZER  # type: ignore[attr-defined]

        out: list[float] = []
        for t in texts:
            if not t:
                out.append(0.0)
                continue
            try:
                out.append(float(_ANALYZER.polarity_scores(t)["compound"]))
            except Exception as exc:
                logger.debug(f"[sentiment/vader] erreur sur texte : {exc}")
                out.append(0.0)
        return out


_FINBERT_PIPELINE = None  # chargé paresseusement une seule fois


def _get_finbert_pipeline():
    """Charge et cache le pipeline FinBERT. Coût : 3-5s + ~440MB RAM la 1ʳᵉ fois."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        from transformers import pipeline  # type: ignore

        _FINBERT_PIPELINE = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _FINBERT_PIPELINE


@dataclass(frozen=True)
class FinBERTScorer:
    """FinBERT (ProsusAI/finbert) via transformers — optionnel, GPU recommandé.

    Le pipeline est chargé au premier appel via `_get_finbert_pipeline` et
    réutilisé ensuite (évite O(n_tickers) rechargements).
    """

    version: str = "finbert_prosus_v1"

    def score(self, texts: list[str]) -> list[float]:
        try:
            clf = _get_finbert_pipeline()
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers non installé — FinBERT indisponible") from exc

        out: list[float] = []
        for res in clf(texts or [""]):
            label = (res.get("label") or "").lower()
            p = float(res.get("score") or 0.0)
            # positive → +p, negative → −p, neutral → 0
            if label.startswith("pos"):
                out.append(p)
            elif label.startswith("neg"):
                out.append(-p)
            else:
                out.append(0.0)
        return out


def get_scorer(prefer_finbert: bool = False) -> SentimentScorer:
    """Retourne le scorer par défaut (VADER), ou FinBERT si dispo et demandé."""
    if prefer_finbert:
        try:
            import transformers  # noqa: F401
            return FinBERTScorer()
        except ImportError:
            logger.info("[sentiment] FinBERT indisponible — fallback VADER")
    return VaderScorer()
