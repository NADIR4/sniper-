"""Page Tableau de bord : KPIs et dernier scan."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from data.cache import Signal, OHLCV, SessionLocal
from signals.generator import fetch_signals


def _kpis(signals: list[Signal]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📡 Signaux totaux", len(signals))
    c2.metric("🔥 ULTRA", sum(1 for s in signals if s.confidence == "ULTRA"))
    c3.metric("💎 HIGH", sum(1 for s in signals if s.confidence == "HIGH"))
    avg = sum(s.consensus_score for s in signals) / len(signals) if signals else 0
    c4.metric("📊 Score moyen", f"{avg:.1%}")
    last = max((s.created_at for s in signals), default=None)
    delta = (datetime.utcnow() - last).total_seconds() / 60 if last else 0
    c5.metric("⏰ Dernier scan", f"{delta:.0f} min" if last else "—")


def _dedupe_latest(signals: list[Signal]) -> list[Signal]:
    """Garde 1 signal par (ticker, direction) — le plus récent."""
    seen: dict[tuple[str, str], Signal] = {}
    for s in sorted(signals, key=lambda x: x.created_at, reverse=True):
        key = (s.ticker, s.direction or "LONG")
        if key not in seen:
            seen[key] = s
    return list(seen.values())


def _top_signals_table(signals: list[Signal], n: int = 10) -> None:
    unique = _dedupe_latest(signals)
    top = sorted(unique, key=lambda s: -s.consensus_score)[:n]
    if not top:
        st.info("Aucun signal pour le moment. Lance un scan pour commencer.")
        return
    df = pd.DataFrame([{
        "Ticker": s.ticker,
        "Score": s.consensus_score,
        "Confiance": s.confidence,
        "Prix": s.price,
        "Cible +100%": s.target_2,
        "Stop-loss": s.stop_loss,
        "R/R": s.risk_reward,
        "Date": s.created_at.strftime("%Y-%m-%d %H:%M"),
    } for s in top])
    st.dataframe(
        df.style.format({
            "Score": "{:.1%}", "Prix": "{:.2f}",
            "Cible +100%": "{:.2f}", "Stop-loss": "{:.2f}", "R/R": "{:.2f}",
        }).background_gradient(subset=["Score"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )


def _price_chart(ticker: str) -> None:
    with SessionLocal() as session:
        rows = (session.query(OHLCV).filter_by(ticker=ticker)
                .order_by(OHLCV.date.desc()).limit(500).all())
    if not rows:
        st.warning(f"Pas de données pour {ticker}")
        return
    df = pd.DataFrame([{
        "date": r.date, "open": r.open, "high": r.high,
        "low": r.low, "close": r.close, "volume": r.volume,
    } for r in reversed(rows)])

    fig = go.Figure(data=[go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=ticker,
    )])
    fig.update_layout(
        title=f"{ticker} — 500 dernières séances",
        template="plotly_dark", xaxis_rangeslider_visible=False, height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard() -> None:
    st.title("🎯 Investment Sniper — Tableau de bord")
    st.caption("Détection ML d'opportunités d'investissement en temps réel")

    signals = fetch_signals(limit=500)
    _kpis(signals)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔥 Top 10 signaux")
        _top_signals_table(signals, 10)
    with col2:
        st.subheader("📈 Graphique")
        tickers = sorted({s.ticker for s in signals}) or ["AAPL"]
        selected = st.selectbox("Ticker", tickers)
        _price_chart(selected)
