"""Page Scanner : lance un scan en temps réel avec barre de progression."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from data.universe import get_universe
from ml.scanner import scan_market
from signals.generator import process_scan
from signals.notifier import notify_pending


def render_scanner() -> None:
    st.title("🔍 Scanner en temps réel")
    st.caption("Analyse toute la base d'actions et détecte les opportunités")

    col1, col2, col3 = st.columns(3)
    col1.metric("Actions suivies", len(get_universe()))
    if "last_scan" in st.session_state:
        delta = (datetime.utcnow() - st.session_state.last_scan).total_seconds() / 60
        col2.metric("Dernier scan", f"il y a {delta:.0f} min")
    col3.metric("État", st.session_state.get("scan_status", "Prêt"))

    st.markdown("---")

    if st.button("🚀 Lancer le scan maintenant", type="primary", use_container_width=True):
        with st.status("Scan en cours…", expanded=True) as status:
            st.write("📥 Récupération des données de marché…")
            progress = st.progress(0.0, text="Initialisation")
            progress.progress(0.2, text="Téléchargement yfinance")
            try:
                results = scan_market()
                progress.progress(0.6, text="Calcul consensus")
                signals = process_scan(results)
                progress.progress(0.85, text="Notifications")
                sent = notify_pending()
                progress.progress(1.0, text="Terminé")

                st.session_state.last_scan = datetime.utcnow()
                st.session_state.scan_status = "OK"
                status.update(label=f"Scan terminé : {len(results)} tickers, {len(signals)} signaux, {sent} emails",
                              state="complete")

                if results:
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
                    st.subheader("🎯 Résultats du scan (triés par score)")
                    st.dataframe(
                        df.style.format({
                            "Score": "{:.1%}", "Prix": "{:.2f}",
                            "RF": "{:.1%}", "XGB": "{:.1%}",
                            "LGB": "{:.1%}",
                            "LSTM": "{:.1%}", "IsoForest": "{:.1%}",
                        }).background_gradient(subset=["Score"], cmap="RdYlGn"),
                        use_container_width=True, hide_index=True,
                    )
            except Exception as e:
                status.update(label=f"Erreur : {e}", state="error")
                st.exception(e)
    else:
        st.info("Clique sur **Lancer le scan maintenant** pour démarrer.")
