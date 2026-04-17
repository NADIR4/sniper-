"""Streamlit entry point — Investment Sniper Bot."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config import settings
from ui.dashboard import render_dashboard
from ui.scanner_page import render_scanner
from ui.signals_page import render_signals
from ui.ml_page import render_ml
from ui.settings_page import render_settings


st.set_page_config(
    page_title="Investment Sniper Bot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background: linear-gradient(180deg, #0B1220 0%, #0F1A2E 100%);
        }
        h1, h2, h3 { color: #F3F6FB; }
        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px; padding: 10px;
        }
        .stDataFrame { border-radius: 8px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


PAGES = {
    "🎯 Tableau de bord": render_dashboard,
    "🔍 Scanner": render_scanner,
    "📡 Signaux": render_signals,
    "🧠 Modèles PRO": render_ml,
    "⚙️ Paramètres": render_settings,
}


def main() -> None:
    _inject_css()
    st.sidebar.title("🎯 Sniper Bot")
    st.sidebar.caption("Investment Opportunity Detection")
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info(f"Seuil consensus : **{settings.consensus_threshold:.0%}**")
    st.sidebar.info(f"Intervalle scan : **{settings.scan_interval_minutes} min**")
    PAGES[choice]()


if __name__ == "__main__":
    main()
