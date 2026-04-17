"""Page Paramètres : affiche la configuration et permet d'exporter le rapport."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from data.universe import get_universe


def render_settings() -> None:
    st.title("⚙️ Paramètres")
    st.caption("Configuration système et univers d'actions (éditable via .env)")

    cfg = {
        "App env": settings.app_env,
        "Log level": settings.log_level,
        "DB path": str(settings.db_path),
        "Models dir": str(settings.models_dir),
        "Exports dir": str(settings.exports_dir),
        "Scan interval (min)": settings.scan_interval_minutes,
        "Consensus threshold": settings.consensus_threshold,
        "Lookback days": settings.lookback_days,
        "History years": settings.history_years,
        "Min gain %": settings.min_gain_pct,
        "SMTP host": settings.smtp_host or "—",
        "SMTP user": settings.smtp_user or "—",
        "Alert email": settings.alert_to_email or "—",
    }
    df = pd.DataFrame([{"Paramètre": k, "Valeur": v} for k, v in cfg.items()])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📋 Univers d'actions")
    universe = get_universe()
    st.info(f"**{len(universe)} actions suivies** (S&P500 + EuroStoxx)")
    st.code("\n".join(", ".join(universe[i:i + 10]) for i in range(0, len(universe), 10)))

    st.markdown("---")
    st.warning(
        "Pour modifier ces paramètres, édite le fichier `.env` à la racine du projet "
        "puis redémarre l'application."
    )
