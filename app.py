"""Streamlit entry point — Investment Sniper Bot (UI PRO MAX)."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from config import settings
from ui.auth import render_logout_button, require_auth
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
    menu_items={
        "About": "Investment Sniper Bot — Détection ML d'opportunités d'investissement",
    },
)


def _inject_css() -> None:
    """Injecte un thème PRO sombre inspiré de Bloomberg Terminal / Notion Dark."""
    st.markdown(
        """
        <style>
        /* ==== GLOBAL ==== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            background:
                radial-gradient(ellipse at top left, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                linear-gradient(180deg, #0A0E1A 0%, #0F1628 50%, #0A0E1A 100%);
            background-attachment: fixed;
        }

        [data-testid="stHeader"] {
            background: rgba(10, 14, 26, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* ==== TYPOGRAPHY ==== */
        h1 {
            background: linear-gradient(90deg, #10B981 0%, #8B5CF6 50%, #F59E0B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        h2, h3 {
            color: #F3F6FB !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #94A3B8 !important;
        }

        /* ==== METRIC CARDS ==== */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: rgba(16, 185, 129, 0.35);
            box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
        }
        [data-testid="metric-container"] [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #F1F5F9 !important;
            font-weight: 800 !important;
            font-size: 1.7rem !important;
        }

        /* ==== BUTTONS ==== */
        .stButton > button {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 10px 20px;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
        }
        .stButton > button[kind="secondary"] {
            background: rgba(148, 163, 184, 0.1);
            color: #F1F5F9;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: none;
        }
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.3);
        }

        /* ==== SIDEBAR ==== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F1628 0%, #0A0E1A 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.08);
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.4rem !important;
        }
        [data-testid="stSidebar"] .stRadio > label {
            padding: 10px 14px;
            border-radius: 10px;
            transition: background 0.2s;
            cursor: pointer;
        }
        [data-testid="stSidebar"] .stRadio > label:hover {
            background: rgba(16, 185, 129, 0.08);
        }

        /* ==== DATAFRAMES ==== */
        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.12);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        /* ==== TABS ==== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(15, 23, 42, 0.4);
            padding: 4px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            color: #94A3B8;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            color: white !important;
        }

        /* ==== INFO/WARNING/SUCCESS BOXES ==== */
        .stAlert {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.15);
            backdrop-filter: blur(8px);
        }

        /* ==== PROGRESS BAR ==== */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #10B981, #8B5CF6, #F59E0B);
        }

        /* ==== EXPANDER ==== */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 10px;
            font-weight: 600;
        }

        /* ==== SELECTBOX ==== */
        [data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 10px;
        }

        /* ==== CUSTOM BADGES ==== */
        .sniper-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-right: 6px;
        }
        .badge-ultra { background: linear-gradient(135deg, #F59E0B, #EF4444); color: white; }
        .badge-high { background: linear-gradient(135deg, #10B981, #059669); color: white; }
        .badge-medium { background: linear-gradient(135deg, #3B82F6, #1D4ED8); color: white; }
        .badge-low { background: rgba(148,163,184,0.2); color: #94A3B8; }

        /* ==== HERO HEADER ==== */
        .sniper-hero {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 14px;
        }
        .sniper-hero-title { font-size: 0.85rem; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; }
        .sniper-hero-value { font-size: 1.6rem; color: #F1F5F9; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .sniper-pulse { display: inline-block; width: 10px; height: 10px; background: #10B981; border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); animation: pulse 2s infinite; margin-right: 8px; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* ==== FOOTER ==== */
        .sniper-footer {
            text-align: center;
            padding: 20px;
            color: #64748B;
            font-size: 0.8rem;
            border-top: 1px solid rgba(148, 163, 184, 0.08);
            margin-top: 40px;
        }

        /* ==== SCROLLBAR ==== */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: #0A0E1A; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #10B981, #8B5CF6);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #059669, #7C3AED); }

        /* Hide Streamlit footer branding */
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_header() -> None:
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0; border-bottom: 1px solid rgba(148,163,184,0.1); margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 2rem;">🎯</div>
                <div>
                    <div style="font-weight: 800; font-size: 1.25rem; background: linear-gradient(90deg,#10B981,#8B5CF6);
                                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        Sniper Bot
                    </div>
                    <div style="font-size: 0.7rem; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase;">
                        Investment AI
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #64748B;">
            <div><span class="sniper-pulse"></span>Live · v2.0 PRO</div>
            <div style="margin-top: 8px;">🎯 Consensus : <b style="color:#10B981;">{settings.consensus_threshold:.0%}</b></div>
            <div>⏱️ Scan : <b style="color:#10B981;">{settings.scan_interval_minutes} min</b></div>
            <div>📈 Seuil gain : <b style="color:#10B981;">+{settings.min_gain_pct}%</b></div>
            <div>📊 Lookback : <b style="color:#10B981;">{settings.lookback_days}j</b></div>
            <div style="margin-top: 10px; opacity: 0.6;">© {datetime.utcnow().year} Sniper Bot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PAGES = {
    "🎯 Dashboard": render_dashboard,
    "🔍 Scanner live": render_scanner,
    "📡 Signaux": render_signals,
    "🧠 Modèles PRO": render_ml,
    "⚙️ Paramètres": render_settings,
}


def main() -> None:
    _inject_css()
    require_auth()  # 🔒 bloque l'app tant que le mot de passe n'est pas entré (si APP_PASSWORD défini)
    _sidebar_header()
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    render_logout_button()
    _sidebar_footer()
    PAGES[choice]()
    st.markdown(
        '<div class="sniper-footer">🎯 Investment Sniper Bot — Détection ML ultra-avancée · '
        'RF + XGBoost + LightGBM + LSTM + IsoForest · Optuna-tuned</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
