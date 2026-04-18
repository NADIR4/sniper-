"""Page Signaux : table filtrable + export Excel."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from reports.exporter import export_report
from signals.generator import fetch_signals


def render_signals() -> None:
    st.title("📡 Signaux")
    st.caption("Historique complet des signaux avec filtres")

    signals = fetch_signals(limit=1000)
    if not signals:
        st.info("Aucun signal encore. Lance un scan d'abord.")
        return

    df = pd.DataFrame([{
        "Ticker": s.ticker,
        "Date": s.created_at,
        "Prix": s.price,
        "Score": s.consensus_score,
        "Confiance": s.confidence,
        "RF": s.rf_prob,
        "XGB": s.xgb_prob,
        "LGB": s.lgb_prob,
        "LSTM": s.lstm_prob,
        "IsoForest": s.iso_score,
        "Cible +50%": s.target_1,
        "Cible +100%": s.target_2,
        "Cible +200%": s.target_3,
        "Stop-Loss": s.stop_loss,
        "R/R": s.risk_reward,
        "Top Features": s.top_features,
    } for s in signals])

    with st.sidebar:
        st.markdown("### Filtres")
        unique_only = st.checkbox(
            "🔸 1 ligne par ticker (dernier signal)", value=True,
            help="Masque les doublons — garde le signal le plus récent par ticker.",
        )
        conf_filter = st.multiselect(
            "Confiance", ["ULTRA", "HIGH", "MEDIUM", "LOW"],
            default=["ULTRA", "HIGH", "MEDIUM"],
        )
        min_score = st.slider("Score minimum", 0.0, 1.0, 0.5, 0.05)
        tickers = st.multiselect("Tickers", sorted(df["Ticker"].unique()), default=[])

    mask = df["Confiance"].isin(conf_filter) & (df["Score"] >= min_score)
    if tickers:
        mask &= df["Ticker"].isin(tickers)
    filtered = df[mask]
    if unique_only and not filtered.empty:
        filtered = (
            filtered.sort_values("Date", ascending=False)
            .drop_duplicates(subset=["Ticker"], keep="first")
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total filtrés", len(filtered))
    c2.metric("Score moyen", f"{filtered['Score'].mean():.1%}" if len(filtered) else "—")
    c3.metric("R/R moyen", f"{filtered['R/R'].mean():.2f}" if len(filtered) else "—")

    st.markdown("---")
    st.dataframe(
        filtered.style.format({
            "Score": "{:.1%}", "Prix": "{:.2f}", "RF": "{:.1%}", "XGB": "{:.1%}",
            "LSTM": "{:.1%}", "IsoForest": "{:.1%}", "Cible +50%": "{:.2f}",
            "Cible +100%": "{:.2f}", "Cible +200%": "{:.2f}",
            "Stop-Loss": "{:.2f}", "R/R": "{:.2f}",
        }).background_gradient(subset=["Score"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True, height=500,
    )

    st.markdown("---")
    if st.button("📥 Exporter rapport Excel complet", type="primary"):
        with st.spinner("Génération du rapport…"):
            path: Path = export_report()
        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇️ Télécharger {path.name}",
                data=f, file_name=path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.success(f"Rapport prêt : {path}")
