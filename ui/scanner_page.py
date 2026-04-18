"""Page Scanner PRO : KPI tiles + progression live + résultats enrichis."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from data.universe import get_universe
from ml.scanner import scan_market
from signals.generator import process_scan
from signals.notifier import notify_pending


def _render_results(results) -> None:
    """Affiche les résultats du scan avec panels multiples."""
    if not results:
        st.info("Aucun résultat. Vérifie que des modèles sont entraînés.")
        return

    df = pd.DataFrame([{
        "Ticker": r.ticker,
        "Score": r.consensus,
        "Prix": r.price,
        "RF": r.rf_prob,
        "XGB": r.xgb_prob,
        "LGB": r.lgb_prob,
        "LSTM": r.lstm_prob,
        "IsoForest": r.iso_score,
    } for r in results])

    # KPIs de scan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Tickers scannés", len(df))
    c2.metric("🎯 Score ≥ 70%", (df["Score"] >= 0.70).sum())
    c3.metric("🔥 Score ≥ 90%", (df["Score"] >= 0.90).sum())
    c4.metric("⭐ Meilleur score", f"{df['Score'].max():.1%}")

    st.markdown("---")

    # Top 20 + distribution
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("🎯 Top 20 résultats")
        top = df.nlargest(20, "Score")
        st.dataframe(
            top.style.format({
                "Score": "{:.1%}", "Prix": "{:.2f}",
                "RF": "{:.1%}", "XGB": "{:.1%}", "LGB": "{:.1%}",
                "LSTM": "{:.1%}", "IsoForest": "{:.1%}",
            }).background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True, height=560,
        )
    with col2:
        st.subheader("📊 Distribution scores")
        fig = px.histogram(
            df, x="Score", nbins=20, template="plotly_dark",
            color_discrete_sequence=["#10B981"],
        )
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(148,163,184,0.08)", tickformat=".0%"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.08)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧬 Accord modèles")
        # Un scatter RF vs XGB coloré par score
        fig2 = px.scatter(
            df, x="RF", y="XGB", color="Score",
            hover_data=["Ticker"], template="plotly_dark",
            color_continuous_scale="viridis",
        )
        fig2.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(148,163,184,0.08)", tickformat=".0%"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.08)", tickformat=".0%"),
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_scanner() -> None:
    st.title("🔍 Scanner live")
    st.caption("Analyse l'ensemble de l'univers · consensus multi-modèles en temps réel")

    # KPIs en haut
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌐 Univers", len(get_universe()), delta="USA + Europe")
    if "last_scan" in st.session_state:
        delta = (datetime.utcnow() - st.session_state.last_scan).total_seconds() / 60
        c2.metric("⏱️ Dernier scan", f"il y a {delta:.0f} min")
    else:
        c2.metric("⏱️ Dernier scan", "jamais")
    c3.metric("🟢 État", st.session_state.get("scan_status", "Prêt"))
    c4.metric("📧 Emails envoyés", st.session_state.get("last_emails_sent", 0))

    st.markdown("---")

    # Bouton scan
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        launch = st.button("🚀 Lancer le scan", type="primary", use_container_width=True)
    with col_info:
        st.info(
            "Le scan analyse chaque ticker avec RF + XGBoost + LightGBM + LSTM + IsolationForest, "
            "calcule le consensus et génère des signaux si score ≥ seuil."
        )

    st.markdown("---")

    if launch:
        with st.status("Scan en cours…", expanded=True) as status:
            st.write("📥 Récupération des données de marché…")
            progress = st.progress(0.0, text="Initialisation")
            try:
                progress.progress(0.15, text="Téléchargement yfinance")
                results = scan_market()
                progress.progress(0.55, text="Calcul consensus multi-modèles")
                signals = process_scan(results)
                progress.progress(0.85, text="Envoi des notifications")
                sent = notify_pending()
                progress.progress(1.0, text="✅ Terminé")

                st.session_state.last_scan = datetime.utcnow()
                st.session_state.scan_status = "OK"
                st.session_state.last_emails_sent = sent

                status.update(
                    label=f"✅ Scan terminé · {len(results)} tickers · {len(signals)} signaux · {sent} emails",
                    state="complete",
                )
                _render_results(results)

            except Exception as e:
                st.session_state.scan_status = "ERREUR"
                status.update(label=f"❌ Erreur : {e}", state="error")
                st.exception(e)
    else:
        st.info("Clique sur **🚀 Lancer le scan** pour analyser tous les tickers.")
