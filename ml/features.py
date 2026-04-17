"""Feature engineering : indicateurs techniques + structure de prix."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, ADXIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


TECH_FEATURE_NAMES = [
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_high", "bb_low", "bb_pct", "atr_14",
    "obv", "obv_delta", "williams_r", "stoch_k", "stoch_d", "adx_14",
    "vol_rel_20", "sma_50_gt_200", "golden_cross",
    "hv_20", "hv_60", "roc_10", "roc_20",
    "hh_count_20", "hl_count_20",
    "close_ratio_sma50", "close_ratio_sma200",
]

# Sprint 4 — features multi-timeframe et régime (toujours basées sur info passée)
ADVANCED_FEATURE_NAMES = [
    "rsi_weekly",          # RSI 14 sur agrégation hebdo
    "roc_weekly",          # ROC 4 semaines
    "vol_regime_pct",      # percentile rang HV20 dans fenêtre 252j
    "trend_strength",      # pente SMA200 normalisée par prix
    "rsi_divergence",      # RSI ∆ vs ROC ∆ (détecte divergences baissières/haussières)
    "drawdown_60d",        # drawdown max sur 60j (négatif ou 0)
    "distance_to_52w_high",  # (52w_high - close)/close → 0 proche sommet
    "price_accel_20",      # ROC(5) - ROC(20) : accélération relative
]

# Sprint 3 — features news, toutes laggées (shift(1)) pour exclure les news du jour t
NEWS_FEATURE_NAMES = [
    "news_sent_7d",        # moyenne sentiment sur 7j, reflète ambiance récente
    "news_sent_momentum",  # sent_mean_3d − sent_mean_14d : accélération
    "news_count_7d",       # volume d'articles sur 7j
    "news_vol_spike",      # count_7d / count_30d_avg, clipé [0, 10]
]

FEATURE_NAMES = TECH_FEATURE_NAMES + ADVANCED_FEATURE_NAMES + NEWS_FEATURE_NAMES


def _pct_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(
        lambda x: (x[-1] > x[:-1]).sum() / max(len(x) - 1, 1),
        raw=True,
    )


def _neutral_news_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features news neutres quand aucune news dispo. `news_vol_spike`=1 (ratio neutre)."""
    return pd.DataFrame(
        {
            "news_sent_7d": 0.0,
            "news_sent_momentum": 0.0,
            "news_count_7d": 0.0,
            "news_vol_spike": 1.0,
        },
        index=index,
    )


def compute_news_features(
    news_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Agrège les news en features quotidiennes laggées d'un jour.

    Étapes :
    1. Resample par jour calendaire UTC → sent_mean + count (0 si aucune news).
    2. Rolling windows [t-N, t] (t inclus) pour sent_7d, momentum, count_7d, spike.
    3. `shift(1)` : la valeur visible au jour t ne contient que [t-N, t-1] → anti-leakage.
    4. Reindex sur l'index OHLCV (jours de bourse) avec ffill pour propager sur weekends.

    Clipping `news_vol_spike` ∈ [0, 10] évite les divisions explosives.
    Si `news_df` est vide → features neutres.
    """
    if price_index.empty:
        return pd.DataFrame(columns=NEWS_FEATURE_NAMES)
    if news_df.empty:
        return _neutral_news_features(price_index)

    # Normalisation tz pour que le reindex ne retourne pas tout en NaN
    idx = news_df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        news_df = news_df.tz_convert("UTC").tz_localize(None)

    price_idx = price_index
    if isinstance(price_idx, pd.DatetimeIndex) and price_idx.tz is not None:
        price_idx = price_idx.tz_convert("UTC").tz_localize(None)

    # Resample par jour calendaire couvrant toute la plage demandée
    start = min(news_df.index.min(), price_idx.min())
    end = max(news_df.index.max(), price_idx.max())
    daily_range = pd.date_range(start.normalize(), end.normalize(), freq="D")

    by_day = news_df["sentiment_score"].resample("D").agg(["mean", "count"])
    by_day = by_day.reindex(daily_range)
    by_day["mean"] = by_day["mean"].fillna(0.0)
    by_day["count"] = by_day["count"].fillna(0.0)

    sent = by_day["mean"]
    count = by_day["count"]

    out = pd.DataFrame(index=daily_range)
    out["news_sent_7d"] = sent.rolling(7, min_periods=1).mean()
    out["news_sent_momentum"] = (
        sent.rolling(3, min_periods=1).mean() - sent.rolling(14, min_periods=1).mean()
    )
    out["news_count_7d"] = count.rolling(7, min_periods=1).sum()
    count_30d = count.rolling(30, min_periods=7).mean().replace(0, np.nan)
    out["news_vol_spike"] = (
        (out["news_count_7d"] / count_30d).clip(lower=0.0, upper=10.0).fillna(1.0)
    )

    # Anti-leakage : la valeur au jour t ne reflète que les news jusqu'à t-1
    out = out.shift(1)

    # Reindex sur les jours de bourse, ffill limité à 3j (weekends standards) :
    # sur un gap plus long (fermeture exceptionnelle, halt) → NaN → fallback neutre.
    # Évite la dérive "double-shift" sur les lundis post-férié.
    out = out.reindex(price_idx, method="ffill", limit=3).fillna(
        {"news_sent_7d": 0.0, "news_sent_momentum": 0.0,
         "news_count_7d": 0.0, "news_vol_spike": 1.0}
    )
    return out[NEWS_FEATURE_NAMES]


def compute_features(
    df: pd.DataFrame,
    news_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcule les features techniques + news à partir d'un DataFrame OHLCV.

    Paramètre `news_df` optionnel : DataFrame indexé par `published_at` (UTC naïf)
    avec colonne `sentiment_score`. Si absent ou vide, les features news sont
    neutres (0.0 / 1.0 pour spike).

    Retourne un DataFrame aligné avec l'index source, NaN pour les périodes
    de warm-up techniques. Les features news ne propagent pas de NaN.
    """
    if df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = RSIIndicator(close, 14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close, window=20)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pct"] = bb.bollinger_pband()

    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    obv_ind = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out["obv"] = obv_ind
    out["obv_delta"] = obv_ind.diff()

    out["williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["adx_14"] = ADXIndicator(high, low, close, window=14).adx()

    vol_ma20 = vol.rolling(20).mean()
    out["vol_rel_20"] = vol / vol_ma20.replace(0, np.nan)

    sma50 = SMAIndicator(close, 50).sma_indicator()
    sma200 = SMAIndicator(close, 200).sma_indicator()
    out["sma_50_gt_200"] = (sma50 > sma200).astype(float)
    out["golden_cross"] = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).astype(float)

    log_ret = np.log(close / close.shift(1))
    out["hv_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    out["hv_60"] = log_ret.rolling(60).std() * np.sqrt(252)

    out["roc_10"] = ROCIndicator(close, 10).roc()
    out["roc_20"] = ROCIndicator(close, 20).roc()

    out["hh_count_20"] = _pct_rank(high, 20)
    out["hl_count_20"] = _pct_rank(low, 20)

    out["close_ratio_sma50"] = close / sma50
    out["close_ratio_sma200"] = close / sma200

    # --- Features avancées (Sprint 4) ---
    # Multi-timeframe : agrégation hebdo puis reindex journalier par ffill
    # anti-leakage : la valeur hebdo au vendredi est dispo dès lundi suivant (shift 1 bar hebdo)
    weekly = close.resample("W-FRI").last()
    weekly_rsi = RSIIndicator(weekly, 14).rsi()
    weekly_roc = ROCIndicator(weekly, 4).roc()
    # shift(1) sur série hebdo → pas de fuite intra-semaine, puis reindex forward-fill
    out["rsi_weekly"] = weekly_rsi.shift(1).reindex(close.index, method="ffill")
    out["roc_weekly"] = weekly_roc.shift(1).reindex(close.index, method="ffill")

    # Régime volatilité : percentile rank HV20 sur 252j glissants
    out["vol_regime_pct"] = out["hv_20"].rolling(252, min_periods=60).rank(pct=True)

    # Force de tendance : pente SMA200 (delta 20j) normalisée par prix
    out["trend_strength"] = (sma200 - sma200.shift(20)) / close.replace(0, np.nan)

    # Divergence RSI/ROC : différence standardisée (alerte fin de cycle)
    rsi_z = (out["rsi_14"] - 50) / 30  # ~[-1, 1]
    roc_z = out["roc_20"] / out["roc_20"].rolling(60, min_periods=20).std().replace(0, np.nan)
    out["rsi_divergence"] = rsi_z - roc_z

    # Drawdown 60j : (close - max_60d) / max_60d
    rolling_max_60 = close.rolling(60, min_periods=20).max()
    out["drawdown_60d"] = (close - rolling_max_60) / rolling_max_60.replace(0, np.nan)

    # Distance au plus haut 52 semaines (252 jours boursiers)
    high_52w = close.rolling(252, min_periods=60).max()
    out["distance_to_52w_high"] = (high_52w - close) / close.replace(0, np.nan)

    # Accélération prix : ROC court − ROC long
    out["price_accel_20"] = out["roc_10"] - out["roc_20"]

    # Features news (alignées sur l'index OHLCV, anti-leakage garantie)
    news_feats = (
        compute_news_features(news_df, df.index)
        if news_df is not None
        else _neutral_news_features(df.index)
    )
    for col in NEWS_FEATURE_NAMES:
        out[col] = news_feats[col].reindex(out.index).fillna(
            0.0 if col != "news_vol_spike" else 1.0
        )

    return out[FEATURE_NAMES]


def _compute_forward_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int,
    peak_threshold: float,
    crash_threshold: float,
) -> pd.DataFrame:
    """Calcule labels forward (fenêtre strictement future [t+1, t+horizon]).

    Formule : on décale high/low d'un jour (shift(-1)) AVANT le rolling.
    Ainsi la fenêtre rolling commence à t+1 et se termine à t+horizon
    → aucune fuite de l'information du jour t.

    Les `horizon` derniers jours ont un futur incomplet → labels = NA.
    """
    # shift(-1) : la valeur au jour t devient celle du jour t+1
    # rolling(horizon).max() sur cette série décalée agrège [t+1 .. t+horizon]
    # shift(-(horizon-1)) : aligne le résultat sur t (et non sur t+horizon-1)
    future_high = high.shift(-1).rolling(horizon, min_periods=horizon).max()
    future_low = low.shift(-1).rolling(horizon, min_periods=horizon).min()
    future_max = future_high.shift(-(horizon - 1))
    future_min = future_low.shift(-(horizon - 1))

    up_ret = (future_max / close) - 1
    down_ret = (future_min / close) - 1

    labels = pd.DataFrame(index=close.index)
    # NaN propagés (min_periods=horizon) → y reste NA si futur incomplet
    labels["y_long"] = (up_ret >= peak_threshold).where(up_ret.notna()).astype("Int64")
    labels["y_short"] = (
        (down_ret <= -abs(crash_threshold)).where(down_ret.notna()).astype("Int64")
    )
    return labels


def _sample_indices(
    labels: pd.Series,
    neg_samples_per_pos: int,
    rng: np.random.Generator,
) -> list:
    """Échantillonne pos + sous-échantillon neg pour équilibrer."""
    pos_idx = labels.index[labels == 1]
    neg_idx_all = labels.index[labels == 0]
    if len(neg_idx_all) == 0:
        return list(pos_idx)
    n_neg = min(len(neg_idx_all), max(1, len(pos_idx) * neg_samples_per_pos) or 200)
    neg_sample = (
        rng.choice(neg_idx_all, size=n_neg, replace=False)
        if len(neg_idx_all) >= n_neg else neg_idx_all
    )
    return list(pos_idx) + list(neg_sample)


def build_training_dataset(
    panels: dict[str, pd.DataFrame],
    lookback: int,
    horizon: int = 120,
    gain_threshold: float = 1.0,
    neg_samples_per_pos: int = 3,
    seed: int = 42,
    news_panels: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[tuple[str, pd.Timestamp]]]:
    """Dataset LONG (compat ascendante) : y=1 si pic +gain_threshold dans horizon j.

    Garde anti-leakage : exclut les `horizon` derniers jours.
    `news_panels` optionnel : {ticker: DataFrame indexé par published_at}.
    """
    rng = np.random.default_rng(seed)
    news_panels = news_panels or {}
    X_rows, y_rows, meta = [], [], []

    for ticker, df in panels.items():
        if df.empty or len(df) < lookback + horizon + 20:
            continue
        feats = compute_features(df, news_df=news_panels.get(ticker)).dropna()
        close = df["close"].astype(float).reindex(feats.index)
        high = df["high"].astype(float).reindex(feats.index)
        low = df["low"].astype(float).reindex(feats.index)

        labels_df = _compute_forward_labels(
            close, high, low, horizon,
            peak_threshold=gain_threshold, crash_threshold=0.30,
        )
        labels = labels_df["y_long"].dropna().astype(int)

        for ts in _sample_indices(labels, neg_samples_per_pos, rng):
            X_rows.append(feats.loc[ts].values)
            y_rows.append(int(labels.loc[ts]))
            meta.append((ticker, ts))

    if not X_rows:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int), []

    X = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
    y = pd.Series(y_rows, name="label")
    return X, y, meta


def build_training_dataset_dual(
    panels: dict[str, pd.DataFrame],
    lookback: int,
    horizon: int = 120,
    peak_threshold: float = 1.0,
    crash_threshold: float = 0.30,
    neg_samples_per_pos: int = 3,
    seed: int = 42,
    news_panels: dict[str, pd.DataFrame] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series, list[tuple[str, pd.Timestamp]]]]:
    """Construit deux datasets : LONG (pic +peak_threshold) et SHORT (crash -crash_threshold).

    Retourne {"long": (X, y, meta), "short": (X, y, meta)}.
    Anti-leakage : exclut les `horizon` derniers jours.
    `news_panels` optionnel : {ticker: DataFrame indexé par published_at}.
    """
    rng = np.random.default_rng(seed)
    news_panels = news_panels or {}
    out: dict[str, tuple[pd.DataFrame, pd.Series, list[tuple[str, pd.Timestamp]]]] = {}

    for direction, label_col in (("long", "y_long"), ("short", "y_short")):
        X_rows, y_rows, meta = [], [], []
        for ticker, df in panels.items():
            if df.empty or len(df) < lookback + horizon + 20:
                continue
            feats = compute_features(df, news_df=news_panels.get(ticker)).dropna()
            close = df["close"].astype(float).reindex(feats.index)
            high = df["high"].astype(float).reindex(feats.index)
            low = df["low"].astype(float).reindex(feats.index)

            labels_df = _compute_forward_labels(
                close, high, low, horizon,
                peak_threshold=peak_threshold, crash_threshold=crash_threshold,
            )
            labels = labels_df[label_col].dropna().astype(int)

            for ts in _sample_indices(labels, neg_samples_per_pos, rng):
                X_rows.append(feats.loc[ts].values)
                y_rows.append(int(labels.loc[ts]))
                meta.append((ticker, ts))

        if X_rows:
            X = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
            y = pd.Series(y_rows, name=f"label_{direction}")
            out[direction] = (X, y, meta)
        else:
            out[direction] = (pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int), [])

    return out


def build_sequences(
    panels: dict[str, pd.DataFrame],
    lookback: int,
    horizon: int = 120,
    gain_threshold: float = 1.0,
    news_panels: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit séquences 3D (n, lookback, n_features) pour LSTM.

    Utilise `_compute_forward_labels` → même garde anti-leakage que le reste
    du pipeline (queue `horizon` jours exclue via NA propagation).
    """
    news_panels = news_panels or {}
    X_seq, y_seq = [], []
    for ticker, df in panels.items():
        if df.empty or len(df) < lookback + horizon + 20:
            continue
        feats = compute_features(df, news_df=news_panels.get(ticker)).dropna()
        close = df["close"].astype(float).reindex(feats.index)
        high = df["high"].astype(float).reindex(feats.index)
        low = df["low"].astype(float).reindex(feats.index)

        labels_df = _compute_forward_labels(
            close, high, low, horizon,
            peak_threshold=gain_threshold, crash_threshold=0.30,
        )
        labels = labels_df["y_long"]  # Int64 avec NA

        arr = feats.values
        for i in range(lookback, len(feats)):
            ts = feats.index[i]
            if ts not in labels.index or pd.isna(labels.loc[ts]):
                continue
            X_seq.append(arr[i - lookback:i])
            y_seq.append(int(labels.loc[ts]))
    if not X_seq:
        return np.empty((0, lookback, len(FEATURE_NAMES))), np.empty((0,))
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)
