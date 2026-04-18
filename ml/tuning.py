"""Optimisation d'hyperparamètres via Optuna pour XGB et LightGBM.

Sprint 4 : recherche bayesienne TimeSeriesSplit-based, métrique cible PR-AUC
(plus robuste que ROC-AUC sur données déséquilibrées). Résultats persistés
dans `ml/models/best_params_{model}_{direction}.json`.

Usage :
    from ml.tuning import tune_xgb, tune_lgb
    best = tune_xgb(X, y, n_trials=50)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from config import settings

Direction = Literal["long", "short"]


def _require_optuna():
    try:
        import optuna  # noqa: F401

        return optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "optuna non installé — ajoute `optuna` à requirements.txt"
        ) from exc


def _cv_pr_auc(estimator_cls, params: dict, X: pd.DataFrame, y: pd.Series,
               n_splits: int = 5) -> float:
    """Moyenne de PR-AUC sur TimeSeriesSplit (strict, no-shuffle)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for tr, te in tscv.split(X):
        if len(np.unique(y.iloc[tr])) < 2 or len(np.unique(y.iloc[te])) < 2:
            continue
        clf = estimator_cls(**params)
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        scores.append(float(average_precision_score(y.iloc[te], p)))
    return float(np.mean(scores)) if scores else 0.0


def tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    direction: Direction = "long",
    n_trials: int = 40,
    timeout_sec: int | None = 1200,
) -> dict:
    """Recherche bayesienne XGBoost. Retourne best_params (dict)."""
    optuna = _require_optuna()
    from xgboost import XGBClassifier

    n_pos = max(int(y.sum()), 1)
    scale_pos = float((len(y) - y.sum()) / n_pos)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        return _cv_pr_auc(XGBClassifier, params, X, y)

    study = optuna.create_study(direction="maximize", study_name=f"xgb_{direction}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({
        "scale_pos_weight": scale_pos, "eval_metric": "aucpr",
        "tree_method": "hist", "random_state": 42, "n_jobs": -1,
    })
    logger.info(
        f"[tuning/xgb/{direction}] best PR-AUC={study.best_value:.4f} — "
        f"{study.best_trial.number + 1}/{n_trials} trials"
    )
    _persist(best, "xgb", direction, best_value=study.best_value)
    return best


def tune_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    direction: Direction = "long",
    n_trials: int = 40,
    timeout_sec: int | None = 1200,
) -> dict:
    """Recherche bayesienne LightGBM. Retourne best_params (dict)."""
    optuna = _require_optuna()
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lightgbm non installé") from exc

    n_pos = max(int(y.sum()), 1)
    scale_pos = float((len(y) - y.sum()) / n_pos)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": scale_pos,
            "objective": "binary",
            "metric": "average_precision",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        return _cv_pr_auc(LGBMClassifier, params, X, y)

    study = optuna.create_study(direction="maximize", study_name=f"lgb_{direction}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({
        "scale_pos_weight": scale_pos, "objective": "binary",
        "metric": "average_precision", "random_state": 42,
        "n_jobs": -1, "verbose": -1,
    })
    logger.info(
        f"[tuning/lgb/{direction}] best PR-AUC={study.best_value:.4f} — "
        f"{study.best_trial.number + 1}/{n_trials} trials"
    )
    _persist(best, "lgb", direction, best_value=study.best_value)
    return best


def _persist(params: dict, model: str, direction: str, best_value: float) -> None:
    path = Path(settings.models_dir) / f"best_params_{model}_{direction}.json"
    payload = {"best_value_pr_auc": float(best_value), "params": params}
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"[tuning] → {path}")


def load_best_params(model: str, direction: str) -> dict | None:
    """Charge les best params si déjà tunés, sinon None."""
    path = Path(settings.models_dir) / f"best_params_{model}_{direction}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("params")
