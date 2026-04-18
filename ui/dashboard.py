"""Page Tableau de bord PRO : Hero, KPIs, charts, timeline, top signaux."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import settings
from data.cache import Signal, OHLCV, SessionLocal
from signals.generator import fetch_signals


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dedupe_latest(signals: list[Signal]) -> list[Signal]:
    """Garde 1 signal par (ticker, direction) — le plus récent."""
    seen: dict[tuple[str, str], Signal] = {}
    for s in sorted(signals, key=lambda x: x.created_at, reverse=True):
        key = (s.ticker, s.direction or "LONG")
        if key not in seen:
            seen[key] = s
    return list(seen.values())


def _confidence_color(conf: str) -> str:
    return {
        "ULTRA": "#F59E0B",
        "HIGH": "#10B981",
        "MEDIUM": "#3B82F6",
        "LOW": "#6B7280",
    }.get(conf, "#94A3B8")


def _confidence_badge(conf: str) -> str:
    cls = {"ULTRA": "badge-ultra", "HIGH": "badge-high",
           "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(conf, "badge-low")
    return f'<span class="sniper-badge {cls}">{conf}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
def _hero(signals: list[Signal]) -> None:
    unique = _dedupe_latest(signals)
    n_ultra = sum(1 for s in unique if s.confidence == "ULTRA")
    n_high = sum(1 for s in unique if s.confidence == "HIGH")
    last = max((s.created_at for s in signals), default=None)

    if last:
        delta_min = (datetime.utcnow() - last).total_seconds() / 60
        last_txt = f"il y a {delta_min:.0f} min" if delta_min < 120 else last.strftime("%d/%m %H:%M")
        status_color = "#10B981" if delta_min < 30 else "#F59E0B" if delta_min < 180 else "#EF4444"
    else:
        last_txt = "aucun scan"
        status_color = "#6B7280"

    st.markdown(
        f"""
        <div class="sniper-hero">
            <div>
                <div class="sniper-hero-title"><span class="sniper-pulse"></span>État système</div>
                <div class="sniper-hero-value" style="color:{status_color};">●  {last_txt}</div>
            </div>
            <div>
                <div class="sniper-hero-title">Tickers suivis</div>
                <div class="sniper-hero-value">260 <span style="font-size:0.9rem;color:#94A3B8;">USA + EU</span></div>
            </div>
            <div>
                <div class="sniper-hero-title">Signaux uniques</div>
                <div class="sniper-hero-value">{len(unique)}</div>
            </div>
            <div>
                <div class="sniper-hero-title">🔥 ULTRA / 💎 HIGH</div>
                <div class="sniper-hero-value" style="color:#F59E0B;">{n_ultra}<span style="color:#10B981;"> / {n_high}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
def _kpis(signals: list[Signal]) -> None:
    unique = _dedupe_latest(signals)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📡 Signaux totaux", len(signals), delta=f"{len(unique)} uniques")

    n_long = sum(1 for s in unique if (s.direction or "LONG") == "LONG")
    n_short = sum(1 for s in unique if s.direction == "SHORT")
    c2.metric("📈 LONG / 📉 SHORT", f"{n_long} / {n_short}")

    avg = sum(s.consensus_score for s in unique) / len(unique) if unique else 0
    c3.metric("📊 Score moyen", f"{avg:.1%}")

    avg_rr = sum(s.risk_reward for s in unique if s.risk_reward) / max(
        1, sum(1 for s in unique if s.risk_reward)
    )
    c4.metric("⚖️ R/R moyen", f"{avg_rr:.2f}")

    recent_24h = sum(1 for s in signals if (datetime.utcnow() - s.created_at).total_seconds() < 86400)
    c5.metric("⏰ 24h", recent_24h, delta="signaux récents")


# ─────────────────────────────────────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────────────────────────────────────
def _confidence_donut(signals: list[Signal]) -> None:
    unique = _dedupe_latest(signals)
    if not unique:
        st.info("Aucun signal pour le donut.")
        return
    counts: dict[str, int] = {}
    for s in unique:
        counts[s.confidence] = counts.get(s.confidence, 0) + 1
    order = ["ULTRA", "HIGH", "MEDIUM", "LOW"]
    labels = [k for k in order if k in counts]
    values = [counts[k] for k in labels]
    colors = [_confidence_color(k) for k in labels]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0A0E1A", width=2)),
        textinfo="label+percent", textposition="outside",
    )])
    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Répartition confiance", x=0.5, font=dict(size=14, color="#F1F5F9")),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _timeline(signals: list[Signal]) -> None:
    if not signals:
        st.info("Aucune activité récente.")
        return
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent = [s for s in signals if s.created_at >= cutoff]
    if not recent:
        st.info("Pas de signaux sur les 7 derniers jours.")
        return
    df = pd.DataFrame([{"date": s.created_at.date(), "confidence": s.confidence} for s in recent])
    grouped = df.groupby(["date", "confidence"]).size().reset_index(name="count")
    fig = px.bar(
        grouped, x="date", y="count", color="confidence",
        color_discrete_map={
            "ULTRA": "#F59E0B", "HIGH": "#10B981",
            "MEDIUM": "#3B82F6", "LOW": "#6B7280",
        },
        category_orders={"confidence": ["ULTRA", "HIGH", "MEDIUM", "LOW"]},
    )
    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Activité 7 derniers jours", x=0.5, font=dict(size=14, color="#F1F5F9")),
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(title="Signaux", gridcolor="rgba(148,163,184,0.08)"),
        barmode="stack",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def _top_signals_table(signals: list[Signal], n: int = 10) -> None:
    unique = _dedupe_latest(signals)
    top = sorted(unique, key=lambda s: -s.consensus_score)[:n]
    if not top:
        st.info("🎯 Aucun signal pour le moment. Lance un scan pour commencer.")
        return
    df = pd.DataFrame([{
        "Ticker": s.ticker,
        "Dir": s.direction or "LONG",
        "Score": s.consensus_score,
        "Conf.": s.confidence,
        "Prix": s.price,
        "T1 (+50%)": s.target_1,
        "T2 (+100%)": s.target_2,
        "Stop": s.stop_loss,
        "R/R": s.risk_reward,
        "Date": s.created_at.strftime("%d/%m %H:%M"),
    } for s in top])
    st.dataframe(
        df.style.format({
            "Score": "{:.1%}", "Prix": "{:.2f}",
            "T1 (+50%)": "{:.2f}", "T2 (+100%)": "{:.2f}",
            "Stop": "{:.2f}", "R/R": "{:.2f}",
        }).background_gradient(subset=["Score"], cmap="RdYlGn")
          .background_gradient(subset=["R/R"], cmap="viridis"),
        use_container_width=True, hide_index=True, height=380,
    )


def _price_chart(ticker: str) -> None:
    with SessionLocal() as session:
        rows = (session.query(OHLCV).filter_by(ticker=ticker)
                .order_by(OHLCV.date.desc()).limit(260).all())
    if not rows:
        st.warning(f"Pas de données pour {ticker}")
        return
    df = pd.DataFrame([{
        "date": r.date, "open": r.open, "high": r.high,
        "low": r.low, "close": r.close, "volume": r.volume,
    } for r in reversed(rows)])

    # Moving averages
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=ticker,
        increasing_line_color="#10B981", decreasing_line_color="#EF4444",
    ))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20",
                             line=dict(color="#F59E0B", width=1.5)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], name="MA50",
                             line=dict(color="#8B5CF6", width=1.5)))

    fig.update_layout(
        title=dict(text=f"{ticker} — 260 séances", x=0.5, font=dict(size=14, color="#F1F5F9")),
        template="plotly_dark", xaxis_rangeslider_visible=False, height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard() -> None:
    st.title("🎯 Investment Sniper")
    st.caption("Détection ML d'opportunités d'investissement — temps réel · consensus multi-modèles")

    signals = fetch_signals(limit=500)

    _hero(signals)
    _kpis(signals)
    st.markdown("---")

    # Analytics row
    col_a, col_b = st.columns([1, 2])
    with col_a:
        _confidence_donut(signals)
    with col_b:
        _timeline(signals)

    st.markdown("---")

    # Main content
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("🔥 Top 10 signaux")
        _top_signals_table(signals, 10)
    with col2:
        st.subheader("📈 Analyse graphique")
        tickers = sorted({s.ticker for s in signals}) or ["AAPL"]
        selected = st.selectbox("Ticker à analyser", tickers, label_visibility="collapsed")
        _price_chart(selected)
