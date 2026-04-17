"""Tests Sprint 3 : scorers sentiment (VADER + Protocol)."""
from __future__ import annotations

from ml.sentiment import SentimentScorer, VaderScorer, get_scorer


def test_vader_scorer_satisfies_protocol() -> None:
    scorer = VaderScorer()
    assert isinstance(scorer, SentimentScorer)
    assert scorer.version == "vader_fin_v1"


def test_vader_detects_positive_finance_terms() -> None:
    scorer = VaderScorer()
    scores = scorer.score(
        ["Apple beats earnings, stock surges 10%", "Bankruptcy filed, shares plunge"]
    )
    assert scores[0] > 0.5, "phrase positive doit avoir score > 0.5"
    assert scores[1] < -0.5, "phrase négative doit avoir score < -0.5"


def test_vader_empty_and_none() -> None:
    scorer = VaderScorer()
    assert scorer.score([]) == []
    assert scorer.score([""]) == [0.0]


def test_get_scorer_defaults_to_vader() -> None:
    scorer = get_scorer(prefer_finbert=False)
    assert scorer.version == "vader_fin_v1"


def test_scores_in_range() -> None:
    scorer = VaderScorer()
    scores = scorer.score(["neutral news", "good news", "terrible collapse"])
    assert all(-1.0 <= s <= 1.0 for s in scores)
