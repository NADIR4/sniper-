"""DataCollector : récupère OHLCV via yfinance, cache SQLite, filtre +100%."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from data.cache import OHLCV, SessionLocal
from data.universe import get_universe


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _download(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=False,
        progress=False, threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"Pas de données pour {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    return df


def _save_ohlcv(ticker: str, df: pd.DataFrame) -> int:
    rows = 0
    with SessionLocal() as session:
        for ts, row in df.iterrows():
            exists = session.query(OHLCV).filter_by(ticker=ticker, date=ts).first()
            if exists:
                continue
            session.add(OHLCV(
                ticker=ticker, date=ts,
                open=float(row.get("open", 0) or 0),
                high=float(row.get("high", 0) or 0),
                low=float(row.get("low", 0) or 0),
                close=float(row.get("close", 0) or 0),
                adj_close=float(row.get("adj close", row.get("close", 0)) or 0),
                volume=float(row.get("volume", 0) or 0),
            ))
            rows += 1
        session.commit()
    return rows


def load_from_cache(ticker: str) -> pd.DataFrame:
    with SessionLocal() as session:
        rows = session.query(OHLCV).filter_by(ticker=ticker).order_by(OHLCV.date.asc()).all()
    if not rows:
        return pd.DataFrame()
    data = [{
        "date": r.date, "open": r.open, "high": r.high, "low": r.low,
        "close": r.close, "adj_close": r.adj_close, "volume": r.volume,
    } for r in rows]
    df = pd.DataFrame(data).set_index("date")
    return df


def collect_history(tickers: Iterable[str] | None = None, years: int | None = None) -> dict[str, pd.DataFrame]:
    tickers = list(tickers) if tickers else get_universe()
    years = years or settings.history_years
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years + 30)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = _download(t, start, end)
            saved = _save_ohlcv(t, df)
            out[t] = df
            logger.info(f"[collector] {t}: {len(df)} lignes, {saved} nouvelles")
        except Exception as e:
            logger.warning(f"[collector] {t} erreur: {e}")
    return out


def filter_gainers(data: dict[str, pd.DataFrame], min_gain_pct: float | None = None) -> dict[str, pd.DataFrame]:
    """Garde seulement actions avec drawdown->peak >= min_gain_pct."""
    min_gain = min_gain_pct or settings.min_gain_pct
    kept: dict[str, pd.DataFrame] = {}
    for t, df in data.items():
        if df.empty or "close" not in df.columns:
            continue
        close = df["close"].dropna()
        if len(close) < 50:
            continue
        trough_idx = close.idxmin()
        peak_after = close.loc[trough_idx:].max()
        trough = close.loc[trough_idx]
        if trough <= 0:
            continue
        gain_pct = (peak_after / trough - 1) * 100
        if gain_pct >= min_gain:
            kept[t] = df
            logger.info(f"[filter] {t} +{gain_pct:.0f}% → conservé")
    return kept


def build_multiindex(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for t, df in data.items():
        d = df.copy()
        d["ticker"] = t
        frames.append(d.reset_index().set_index(["ticker", "date"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()
