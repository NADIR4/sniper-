"""Entraînement bidirectionnel : RF + XGB + LSTM + IsoForest pour LONG et SHORT.

Sprint 2 : deux jeux de modèles indépendants (direction LONG pic +100%,
direction SHORT crash -30%) partagent scaler + IsolationForest. Chaque
direction a ses propres RF/XGB persistés avec suffixe `_long` / `_short`.

Anti-déséquilibre : scale_pos_weight = n_neg/n_pos, eval_metric="aucpr".
TimeSeriesSplit strict (jamais de shuffle).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from config import settings
from ml.features import (
    FEATURE_NAMES,
    build_sequences,
    build_training_dataset_dual,
)

Direction = Literal["long", "short"]

# === Hyperparamètres ultra-avancés ===
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 14,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
    "bootstrap": True,
    "oob_score": True,
}

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",  # PR-AUC recommandé sur données déséquilibrées
    "n_jobs": -1,
    "random_state": 42,
    "tree_method": "hist",
}

LSTM_PARAMS = {
    "units_1": 96,
    "units_2": 48,
    "dropout_1": 0.25,
    "dropout_2": 0.25,
    "dense_units": 24,
    "batch_size": 64,
    "epochs": 25,
    "optimizer": "adam",
    "learning_rate": 0.001,
}

LGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": -1,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "average_precision",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

ISO_PARAMS = {
    "n_estimators": 300,
    "contamination": 0.08,
    "max_samples": "auto",
    "n_jobs": -1,
    "random_state": 42,
}

# Sprint 4 : ajout LGB dans le consensus. LSTM réduit puisque souvent skippé.
CONSENSUS_WEIGHTS = {"rf": 0.25, "xgb": 0.30, "lgb": 0.25, "lstm": 0.10, "iso": 0.10}


@dataclass
class ModelMetrics:
    name: str
    direction: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    n_samples: int
    n_positives: int
    train_time_sec: float = 0.0
    oob_score: float = 0.0
    best_threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(fn, *a, **kw) -> float:
    try:
        return float(fn(*a, **kw))
    except Exception:
        return 0.0


def _find_best_threshold(y_true, y_prob) -> float:
    try:
        prec, rec, th = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        return float(th[np.argmax(f1s[:-1])]) if len(th) else 0.5
    except Exception:
        return 0.5


def _roc_data(y_true, y_prob) -> dict:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {"fpr": fpr.tolist()[:200], "tpr": tpr.tolist()[:200]}
    except Exception:
        return {"fpr": [], "tpr": []}


def _metrics(
    name: str,
    direction: Direction,
    y_true,
    y_prob,
    train_time: float = 0.0,
    oob: float = 0.0,
) -> ModelMetrics:
    has_two_classes = len(set(y_true)) > 1
    best_th = _find_best_threshold(y_true, y_prob) if has_two_classes else 0.5
    y_pred = (np.array(y_prob) >= best_th).astype(int)
    return ModelMetrics(
        name=name,
        direction=direction,
        accuracy=_safe(accuracy_score, y_true, y_pred),
        precision=_safe(precision_score, y_true, y_pred, zero_division=0),
        recall=_safe(recall_score, y_true, y_pred, zero_division=0),
        f1=_safe(f1_score, y_true, y_pred, zero_division=0),
        roc_auc=_safe(roc_auc_score, y_true, y_prob) if has_two_classes else 0.0,
        pr_auc=_safe(average_precision_score, y_true, y_prob) if has_two_classes else 0.0,
        n_samples=len(y_true),
        n_positives=int(np.sum(y_true)),
        train_time_sec=round(train_time, 2),
        oob_score=round(oob, 4),
        best_threshold=round(best_th, 3),
    )


def _calibrate(estimator, X: pd.DataFrame, y: pd.Series, method: str = "isotonic"):
    """Isotonic calibration prefit sur un hold-out TimeSeriesSplit (dernier fold)."""
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    tr, te = splits[-1]
    estimator.fit(X.iloc[tr], y.iloc[tr])
    calib = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calib.fit(X.iloc[te], y.iloc[te])
    return calib


def _train_rf(X: pd.DataFrame, y: pd.Series, direction: Direction):
    t0 = time.time()
    tscv = TimeSeriesSplit(n_splits=5)
    probs, trues = [], []
    for tr, te in tscv.split(X):
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        probs.extend(p)
        trues.extend(y.iloc[te].values)

    # Calibration isotonic pour des probabilités mieux calibrées (important pour le consensus pondéré)
    final_uncalibrated = RandomForestClassifier(**RF_PARAMS)
    final = _calibrate(final_uncalibrated, X, y)
    # OOB + importance sur le modèle sous-jacent non calibré (entraîné sur la partition train du calibrage)
    base = final.estimator
    oob = getattr(base, "oob_score_", 0.0)
    importance = dict(zip(FEATURE_NAMES, base.feature_importances_.tolist()))
    roc = _roc_data(trues, probs)
    m = _metrics("RandomForest", direction, trues, probs, time.time() - t0, oob)
    return final, m, importance, roc


def _train_xgb(X: pd.DataFrame, y: pd.Series, direction: Direction):
    t0 = time.time()
    n_pos = max(int(y.sum()), 1)
    scale_pos = float((len(y) - y.sum()) / n_pos)
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}

    tscv = TimeSeriesSplit(n_splits=5)
    probs, trues = [], []
    for tr, te in tscv.split(X):
        clf = XGBClassifier(**params)
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        probs.extend(p)
        trues.extend(y.iloc[te].values)

    # Calibration isotonic (probas mieux alignées pour le consensus pondéré)
    final = _calibrate(XGBClassifier(**params), X, y)
    base = final.estimator
    importance = dict(zip(FEATURE_NAMES, base.feature_importances_.tolist()))
    roc = _roc_data(trues, probs)
    m = _metrics("XGBoost", direction, trues, probs, time.time() - t0)
    return final, m, importance, roc


def _train_lgb(X: pd.DataFrame, y: pd.Series, direction: Direction):
    """LightGBM calibré isotonic — consensus plus robuste + importance stable.

    scale_pos_weight calculé comme XGB. Retourne (None, stub_metrics, {}, roc)
    si LightGBM absent (gracieux).
    """
    t0 = time.time()
    stub_metrics = ModelMetrics(
        "LightGBM", direction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        int(len(y)), int(y.sum()),
    )
    empty_roc = {"fpr": [], "tpr": []}
    if not _LGBM_AVAILABLE:
        logger.warning(f"[lgb/{direction}] LightGBM non installé — sauté")
        return None, stub_metrics, {}, empty_roc

    n_pos = max(int(y.sum()), 1)
    scale_pos = float((len(y) - y.sum()) / n_pos)
    params = {**LGB_PARAMS, "scale_pos_weight": scale_pos}

    tscv = TimeSeriesSplit(n_splits=5)
    probs, trues = [], []
    for tr, te in tscv.split(X):
        clf = LGBMClassifier(**params)
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        probs.extend(p)
        trues.extend(y.iloc[te].values)

    final = _calibrate(LGBMClassifier(**params), X, y)
    base = final.estimator
    importance = dict(zip(FEATURE_NAMES, base.feature_importances_.tolist()))
    roc = _roc_data(trues, probs)
    m = _metrics("LightGBM", direction, trues, probs, time.time() - t0)
    return final, m, importance, roc


def _train_lstm(X_seq: np.ndarray, y_seq: np.ndarray, direction: Direction):
    t0 = time.time()
    if len(X_seq) < 50 or len(np.unique(y_seq)) < 2:
        logger.warning(f"[lstm/{direction}] dataset insuffisant — sauté")
        stub = ModelMetrics("LSTM", direction, 0, 0, 0, 0, 0, 0,
                            int(len(y_seq)), int(np.sum(y_seq)))
        return None, stub, {"fpr": [], "tpr": []}
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        logger.warning(f"[lstm/{direction}] TensorFlow absent — sauté")
        stub = ModelMetrics("LSTM", direction, 0, 0, 0, 0, 0, 0,
                            int(len(y_seq)), int(np.sum(y_seq)))
        return None, stub, {"fpr": [], "tpr": []}

    split = int(len(X_seq) * 0.8)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    model = Sequential([
        LSTM(LSTM_PARAMS["units_1"],
             input_shape=(X_seq.shape[1], X_seq.shape[2]),
             return_sequences=True),
        Dropout(LSTM_PARAMS["dropout_1"]),
        LSTM(LSTM_PARAMS["units_2"]),
        Dropout(LSTM_PARAMS["dropout_2"]),
        Dense(LSTM_PARAMS["dense_units"], activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(LSTM_PARAMS["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    model.fit(X_tr, y_tr, epochs=LSTM_PARAMS["epochs"],
              batch_size=LSTM_PARAMS["batch_size"],
              validation_split=0.15, callbacks=[es], verbose=0)

    probs = model.predict(X_te, verbose=0).flatten()
    roc = _roc_data(y_te, probs)
    m = _metrics("LSTM", direction, y_te, probs, time.time() - t0)
    return model, m, roc


def _train_iso(X: pd.DataFrame) -> IsolationForest:
    clf = IsolationForest(**ISO_PARAMS)
    clf.fit(X)
    return clf


def _train_direction(
    direction: Direction,
    X_scaled: pd.DataFrame,
    y: pd.Series,
    panels: dict[str, pd.DataFrame],
    seq_horizon: int,
    seq_gain_threshold: float,
) -> dict:
    """Pipeline complet pour une direction : RF + XGB + LSTM."""
    logger.info(f"[trainer/{direction}] 🌲 Random Forest…")
    rf, rf_m, rf_imp, rf_roc = _train_rf(X_scaled, y, direction)
    logger.info(
        f"   → AUC={rf_m.roc_auc:.3f}, PR-AUC={rf_m.pr_auc:.3f}, "
        f"F1={rf_m.f1:.3f}, OOB={rf_m.oob_score:.3f}, {rf_m.train_time_sec}s"
    )

    logger.info(f"[trainer/{direction}] ⚡ XGBoost…")
    xgb, xgb_m, xgb_imp, xgb_roc = _train_xgb(X_scaled, y, direction)
    logger.info(
        f"   → AUC={xgb_m.roc_auc:.3f}, PR-AUC={xgb_m.pr_auc:.3f}, "
        f"F1={xgb_m.f1:.3f}, {xgb_m.train_time_sec}s"
    )

    logger.info(f"[trainer/{direction}] 💡 LightGBM…")
    lgb, lgb_m, lgb_imp, lgb_roc = _train_lgb(X_scaled, y, direction)
    logger.info(
        f"   → AUC={lgb_m.roc_auc:.3f}, PR-AUC={lgb_m.pr_auc:.3f}, "
        f"F1={lgb_m.f1:.3f}, {lgb_m.train_time_sec}s"
    )

    # LSTM uniquement pour LONG (compat descendante) — SHORT skippé
    lstm, lstm_m, lstm_roc = None, None, {"fpr": [], "tpr": []}
    if direction == "long":
        logger.info(f"[trainer/{direction}] 🧠 LSTM…")
        X_seq, y_seq = build_sequences(
            panels, lookback=settings.lookback_days, horizon=seq_horizon,
            gain_threshold=seq_gain_threshold,
        )
        lstm, lstm_m, lstm_roc = _train_lstm(X_seq, y_seq, direction)
    else:
        lstm_m = ModelMetrics("LSTM", direction, 0, 0, 0, 0, 0, 0, 0, 0)

    # Persist
    models_dir = settings.models_dir
    joblib.dump(rf, models_dir / f"rf_{direction}.pkl")
    joblib.dump(xgb, models_dir / f"xgb_{direction}.pkl")
    if lgb is not None:
        joblib.dump(lgb, models_dir / f"lgb_{direction}.pkl")
    if lstm is not None:
        lstm.save(models_dir / f"lstm_{direction}.keras")

    return {
        "rf": rf_m.to_dict(),
        "xgboost": xgb_m.to_dict(),
        "lightgbm": lgb_m.to_dict(),
        "lstm": lstm_m.to_dict(),
        "feature_importance": {
            "random_forest": rf_imp,
            "xgboost": xgb_imp,
            "lightgbm": lgb_imp,
        },
        "roc_curves": {
            "random_forest": rf_roc,
            "xgboost": xgb_roc,
            "lightgbm": lgb_roc,
            "lstm": lstm_roc,
        },
    }


def train_all(panels: dict[str, pd.DataFrame]) -> dict:
    """Entraîne LONG (pic +100%) et SHORT (crash -30%) avec dataset dual."""
    t0 = time.time()
    logger.info(f"[trainer] Construction dataset DUAL → {len(panels)} tickers")

    peak_threshold = settings.min_gain_pct / 100.0
    crash_threshold = 0.30

    duals = build_training_dataset_dual(
        panels,
        lookback=settings.lookback_days,
        horizon=120,
        peak_threshold=peak_threshold,
        crash_threshold=crash_threshold,
    )
    X_long, y_long, _ = duals["long"]
    X_short, y_short, _ = duals["short"]

    if X_long.empty or y_long.sum() < 5:
        raise ValueError(
            f"Dataset LONG trop petit : {len(X_long)} lignes, {int(y_long.sum())} positifs."
        )
    if X_short.empty:
        logger.warning(f"[trainer] Dataset SHORT vide — SHORT non entraîné")

    logger.info(
        f"[trainer] LONG  : {len(X_long)} lignes, {int(y_long.sum())} pos ({y_long.mean():.1%})"
    )
    if not X_short.empty:
        logger.info(
            f"[trainer] SHORT : {len(X_short)} lignes, {int(y_short.sum())} pos ({y_short.mean():.1%})"
        )

    # Scalers indépendants par direction : évite la contamination des distributions
    # (pics haussiers et crashes ont des statistiques de features radicalement différentes).
    scaler_long = StandardScaler().fit(X_long)
    scaler_short = StandardScaler().fit(X_short) if not X_short.empty else None

    def _scale(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    X_long_s = _scale(X_long, scaler_long)
    X_short_s = _scale(X_short, scaler_short) if scaler_short is not None else X_short

    long_metrics = _train_direction(
        "long", X_long_s, y_long, panels,
        seq_horizon=120, seq_gain_threshold=peak_threshold,
    )

    short_metrics: dict = {}
    n_short_pos = int(y_short.sum()) if not y_short.empty else 0
    if not X_short.empty and n_short_pos >= 5:
        short_metrics = _train_direction(
            "short", X_short_s, y_short, panels,
            seq_horizon=120, seq_gain_threshold=crash_threshold,
        )
    else:
        logger.warning(
            f"[trainer] SHORT sauté : {n_short_pos} positifs (< 5 requis). "
            "Réduire crash_threshold ou étendre l'horizon pour activer le SHORT."
        )

    logger.info("[trainer] 🔍 Isolation Forest (anomalies, fit sur LONG scalé)…")
    iso = _train_iso(X_long_s)

    # Persist scalers + iso + legacy aliases (compat ascendante)
    models_dir = settings.models_dir
    joblib.dump(scaler_long, models_dir / "scaler_long.pkl")
    if scaler_short is not None:
        joblib.dump(scaler_short, models_dir / "scaler_short.pkl")
    # Legacy: scaler.pkl = scaler_long (compat ancien scanner)
    joblib.dump(scaler_long, models_dir / "scaler.pkl")
    joblib.dump(iso, models_dir / "iso.pkl")
    # Alias legacy : copie binaire directe (évite re-sérialisation coûteuse)
    import shutil
    shutil.copy2(models_dir / "rf_long.pkl", models_dir / "rf.pkl")
    shutil.copy2(models_dir / "xgb_long.pkl", models_dir / "xgb.pkl")

    metrics = {
        "long": long_metrics,
        "short": short_metrics,
        # Alias rétro-compat pour l'UI existante (tab "Comparaison")
        "random_forest": long_metrics["rf"],
        "xgboost": long_metrics["xgboost"],
        "lstm": long_metrics["lstm"],
        "feature_importance": long_metrics["feature_importance"],
        "roc_curves": long_metrics["roc_curves"],
        "feature_names": FEATURE_NAMES,
        "hyperparameters": {
            "random_forest": RF_PARAMS,
            "xgboost": XGB_PARAMS,
            "lightgbm": LGB_PARAMS,
            "lstm": LSTM_PARAMS,
            "isolation_forest": ISO_PARAMS,
            "consensus_weights": CONSENSUS_WEIGHTS,
        },
        "dataset": {
            "n_rows": int(len(X_long) + len(X_short)),
            "n_long_samples": int(len(X_long)),
            "n_long_positives": int(y_long.sum()),
            "n_short_samples": int(len(X_short)),
            "n_short_positives": int(y_short.sum()) if not y_short.empty else 0,
            "n_positives": int(y_long.sum()),  # legacy
            "n_negatives": int(len(y_long) - y_long.sum()),
            "n_features": len(FEATURE_NAMES),
            "n_tickers": len(panels),
            "class_balance": round(float(y_long.mean()), 4),
            "lookback_days": settings.lookback_days,
            "horizon_days": 120,
            "gain_threshold_pct": settings.min_gain_pct,
            "crash_threshold_pct": crash_threshold * 100,
        },
        "total_train_time_sec": round(time.time() - t0, 2),
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    logger.info(f"[trainer] ✅ Entraînement terminé en {metrics['total_train_time_sec']}s")
    return metrics


def _load_optional(path):
    return joblib.load(path) if path.exists() else None


def load_models() -> dict:
    """Charge scaler + RF/XGB pour LONG et SHORT + IsolationForest + LSTM long.

    Conserve les clés legacy rf/xgb/lstm (= LONG) pour compat scanner+UI.
    """
    m = settings.models_dir
    scaler_legacy = _load_optional(m / "scaler.pkl")
    scaler_long = _load_optional(m / "scaler_long.pkl") or scaler_legacy
    scaler_short = _load_optional(m / "scaler_short.pkl") or scaler_legacy
    iso = _load_optional(m / "iso.pkl")

    rf_long = _load_optional(m / "rf_long.pkl") or _load_optional(m / "rf.pkl")
    xgb_long = _load_optional(m / "xgb_long.pkl") or _load_optional(m / "xgb.pkl")
    lgb_long = _load_optional(m / "lgb_long.pkl")
    rf_short = _load_optional(m / "rf_short.pkl")
    xgb_short = _load_optional(m / "xgb_short.pkl")
    lgb_short = _load_optional(m / "lgb_short.pkl")

    lstm_long = None
    for p in (m / "lstm_long.keras", m / "lstm.keras"):
        if p.exists():
            try:
                from tensorflow.keras.models import load_model
                lstm_long = load_model(p)
                break
            except ImportError:
                break

    metrics_path = m / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    return {
        # Legacy (= scaler_long)
        "scaler": scaler_long,
        "scaler_long": scaler_long,
        "scaler_short": scaler_short,
        "iso": iso,
        # Legacy keys (= LONG direction)
        "rf": rf_long,
        "xgb": xgb_long,
        "lgb": lgb_long,
        "lstm": lstm_long,
        # Explicit per-direction
        "rf_long": rf_long,
        "xgb_long": xgb_long,
        "lgb_long": lgb_long,
        "lstm_long": lstm_long,
        "rf_short": rf_short,
        "xgb_short": xgb_short,
        "lgb_short": lgb_short,
        "metrics": metrics,
    }
