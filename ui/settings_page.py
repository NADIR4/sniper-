"""Page Paramètres PRO : configuration système + univers + infos SMTP."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from data.universe import get_universe


def render_settings() -> None:
    st.title("⚙️ Paramètres")
    st.caption("Configuration système · univers d'actions · infos SMTP (éditable via `.env`)")

    # ─── KPIs rapides ───
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Seuil consensus", f"{settings.consensus_threshold:.0%}")
    c2.metric("📈 MIN gain", f"+{settings.min_gain_pct}%")
    c3.metric("⏱️ Scan", f"{settings.scan_interval_minutes} min")
    c4.metric("📊 Lookback", f"{settings.lookback_days} j")

    st.markdown("---")

    # ─── Config tabs ───
    tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "📋 Univers", "📧 Notifications"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Système")
            sys_cfg = {
                "Environnement": settings.app_env,
                "Log level": settings.log_level,
                "DB path": str(settings.db_path),
                "Models dir": str(settings.models_dir),
                "Exports dir": str(settings.exports_dir),
            }
            df = pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in sys_cfg.items()])
            st.dataframe(df, width="stretch", hide_index=True)

        with col2:
            st.markdown("#### Pipeline ML")
            ml_cfg = {
                "Scan interval (min)": settings.scan_interval_minutes,
                "Consensus threshold": f"{settings.consensus_threshold:.0%}",
                "Lookback days": settings.lookback_days,
                "History years": settings.history_years,
                "Min gain %": f"+{settings.min_gain_pct}%",
            }
            df = pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in ml_cfg.items()])
            st.dataframe(df, width="stretch", hide_index=True)

        st.warning(
            "💡 Pour modifier ces paramètres, édite le fichier `.env` à la racine du projet "
            "puis redémarre l'application (`Ctrl+C` puis `streamlit run app.py`)."
        )

    with tab2:
        universe = get_universe()
        st.info(f"📊 **{len(universe)} actions suivies** — S&P 500 + EuroStoxx + valeurs sélectionnées")

        # Groupage USA / Europe grossier
        usa = [t for t in universe if "." not in t]
        eu = [t for t in universe if "." in t]

        c1, c2 = st.columns(2)
        c1.metric("🇺🇸 USA", len(usa))
        c2.metric("🇪🇺 Europe", len(eu))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🇺🇸 Tickers USA")
            with st.container(height=400):
                st.code("\n".join(", ".join(usa[i:i + 8]) for i in range(0, len(usa), 8)))
        with col2:
            st.markdown("#### 🇪🇺 Tickers Europe")
            with st.container(height=400):
                st.code("\n".join(", ".join(eu[i:i + 6]) for i in range(0, len(eu), 6)))

    with tab3:
        st.markdown("#### 📧 Configuration SMTP")
        smtp_cfg = {
            "SMTP host": settings.smtp_host or "—",
            "SMTP port": settings.smtp_port or "—",
            "SMTP user": settings.smtp_user or "—",
            "Password": "••••••••" if settings.smtp_password else "—",
            "Alert recipient": settings.alert_to_email or "—",
        }
        df = pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in smtp_cfg.items()])
        st.dataframe(df, width="stretch", hide_index=True)

        if settings.alert_to_email:
            st.success(f"✅ Les alertes seront envoyées à : **{settings.alert_to_email}**")
        else:
            st.error("❌ Aucune adresse email configurée — aucune notification ne sera envoyée.")
