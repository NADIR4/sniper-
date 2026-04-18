"""Backtest walk-forward avec sorties ATR + targets partiels.

Sprint 4 — simule la stratégie réelle :
- Entrée : proba consensus > seuil sur X[t]
- Sortie : stop-loss ATR OU target atteinte OU horizon expiré
- Métriques : Sharpe, Max Drawdown, Win Rate, Profit Factor, Expectancy

Strict TimeSeriesSplit : modèle entraîné sur passé, testé sur futur.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.features import FEATURE_NAMES, build_training_dataset_dual, compute_features

Direction = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class Trade:
    ticker: str
    direction: Direction
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str  # "target" | "stop" | "timeout"
    pnl_pct: float
    hold_days: int


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float
    profit_factor: float
    sharpe: float
    max_drawdown_pct: float
    total_return_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_hold_days: float

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_atr(panel: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR en valeur absolue (pas en %)."""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _simulate_trade(
    panel: pd.DataFrame,
    entry_idx: int,
    direction: Direction,
    atr_mult_stop: float = 2.0,
    target_pct: float = 0.50,
    max_hold_days: int = 120,
) -> Trade | None:
    """Simule une entrée à `entry_idx` avec stop ATR + target + timeout."""
    if entry_idx + 1 >= len(panel):
        return None

    atr = _compute_atr(panel).iloc[entry_idx]
    if pd.isna(atr) or atr <= 0:
        return None

    entry_price = float(panel["close"].iloc[entry_idx])
    entry_date = panel.index[entry_idx]

    if direction == "LONG":
        stop = entry_price - atr_mult_stop * atr
        target = entry_price * (1.0 + target_pct)
    else:
        stop = entry_price + atr_mult_stop * atr
        target = entry_price * (1.0 - target_pct)

    end_idx = min(entry_idx + max_hold_days + 1, len(panel))
    future = panel.iloc[entry_idx + 1 : end_idx]

    exit_price = float(future["close"].iloc[-1])
    exit_date = future.index[-1]
    exit_reason = "timeout"

    for ts, row in future.iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        if direction == "LONG":
            if lo <= stop:
                exit_price, exit_date, exit_reason = stop, ts, "stop"
                break
            if hi >= target:
                exit_price, exit_date, exit_reason = target, ts, "target"
                break
        else:
            if hi >= stop:
                exit_price, exit_date, exit_reason = stop, ts, "stop"
                break
            if lo <= target:
                exit_price, exit_date, exit_reason = target, ts, "target"
                break

    if direction == "LONG":
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price

    hold_days = int((exit_date - entry_date).days)
    return Trade(
        ticker=str(panel.attrs.get("ticker", "N/A")),
        direction=direction,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=float(exit_price),
        exit_reason=exit_reason,
        pnl_pct=float(pnl),
        hold_days=hold_days,
    )


def _compute_metrics(trades: list[Trade]) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(pnls))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = float(win_rate * avg_win + (1 - win_rate) * avg_loss)

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 1e-9
    profit_factor = gross_win / max(gross_loss, 1e-9)

    # Sharpe approximé (per-trade basis, pas quotidien)
    sharpe = float(pnls.mean() / (pnls.std() + 1e-9) * np.sqrt(max(len(pnls), 1)))

    # Equity curve → max drawdown
    equity = (1.0 + pnls).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    return BacktestMetrics(
        n_trades=len(trades),
        win_rate=round(win_rate, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        expectancy_pct=round(expectancy, 4),
        profit_factor=round(profit_factor, 3),
        sharpe=round(sharpe, 3),
        max_drawdown_pct=round(max_dd, 4),
        total_return_pct=round(float(equity[-1] - 1.0), 4) if len(equity) else 0.0,
        best_trade_pct=round(float(pnls.max()), 4),
        worst_trade_pct=round(float(pnls.min()), 4),
        avg_hold_days=round(float(np.mean([t.hold_days for t in trades])), 1),
    )


@dataclass
class WalkForwardBacktest:
    panels: dict[str, pd.DataFrame]
    direction: Direction = "LONG"
    threshold: float = 0.65
    target_pct: float = 0.50
    atr_mult_stop: float = 2.0
    max_hold_days: int = 120
    lookback_days: int = 60
    horizon: int = 120
    peak_threshold: float = 1.0
    crash_threshold: float = 0.30
    n_splits: int = 3
    trades: list[Trade] = field(default_factory=list)

    def run(self) -> BacktestMetrics:
        """Exécute le walk-forward. Retourne métriques globales."""
        key = "long" if self.direction == "LONG" else "short"
        duals = build_training_dataset_dual(
            self.panels,
            lookback=self.lookback_days,
            horizon=self.horizon,
            peak_threshold=self.peak_threshold,
            crash_threshold=self.crash_threshold,
        )
        X, y, meta = duals[key]
        if X.empty or y.sum() < 10:
            logger.warning(f"[backtest/{key}] dataset insuffisant")
            return _compute_metrics([])

        # Mapping positional (index X) → (ticker, date) pour retrouver le panel.
        # meta est une list[tuple[str, pd.Timestamp]] → on la garde telle quelle
        # et on accède par index positionnel.
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        all_trades: list[Trade] = []
        n_pos = max(int(y.sum()), 1)
        scale_pos = float((len(y) - y.sum()) / n_pos)

        for fold, (tr, te) in enumerate(tscv.split(X)):
            scaler = StandardScaler().fit(X.iloc[tr])
            X_tr_s = scaler.transform(X.iloc[tr])
            X_te_s = scaler.transform(X.iloc[te])

            clf = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=scale_pos, eval_metric="aucpr",
                tree_method="hist", random_state=42, n_jobs=-1,
            )
            clf.fit(X_tr_s, y.iloc[tr])
            probs = clf.predict_proba(X_te_s)[:, 1]

            # Déclenche un trade sur chaque signal au-dessus du seuil
            for j, p in enumerate(probs):
                if p < self.threshold:
                    continue
                ticker, date = meta[te[j]]
                panel = self.panels.get(ticker)
                if panel is None or date not in panel.index:
                    continue
                panel.attrs["ticker"] = ticker
                entry_idx = panel.index.get_loc(date)
                if isinstance(entry_idx, slice):
                    entry_idx = entry_idx.start
                trade = _simulate_trade(
                    panel, entry_idx, self.direction,
                    atr_mult_stop=self.atr_mult_stop,
                    target_pct=self.target_pct,
                    max_hold_days=self.max_hold_days,
                )
                if trade:
                    all_trades.append(trade)
            logger.info(
                f"[backtest/{key}] fold {fold + 1}/{self.n_splits}: "
                f"{sum(1 for p in probs if p >= self.threshold)} signaux"
            )

        self.trades = all_trades
        metrics = _compute_metrics(all_trades)
        logger.info(
            f"[backtest/{key}] {metrics.n_trades} trades | "
            f"WR={metrics.win_rate:.1%} | Sharpe={metrics.sharpe:.2f} | "
            f"PF={metrics.profit_factor:.2f} | MDD={metrics.max_drawdown_pct:.1%}"
        )
        return metrics


def run_walk_forward(
    panels: dict[str, pd.DataFrame],
    direction: Direction = "LONG",
    threshold: float = 0.65,
    **kwargs,
) -> tuple[BacktestMetrics, list[Trade]]:
    """Helper one-shot : lance le backtest et retourne (métriques, trades)."""
    bt = WalkForwardBacktest(
        panels=panels, direction=direction, threshold=threshold, **kwargs
    )
    metrics = bt.run()
    return metrics, bt.trades
