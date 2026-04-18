"""Gate d'authentification par mot de passe pour l'app Streamlit.

Logique :
  - Si APP_PASSWORD est vide ou non défini → aucun gate (app publique)
  - Sinon → écran de login bloquant, comparaison constant-time, retry
  - Session verrouillée via st.session_state (ne reste pas logué après reload
    complet du navigateur)
"""
from __future__ import annotations

import hmac
import time

import streamlit as st

from config import settings


_SESSION_KEY = "sniper_authenticated"
_ATTEMPTS_KEY = "sniper_auth_attempts"
_LOCKED_UNTIL_KEY = "sniper_locked_until"
_MAX_ATTEMPTS = 5
_LOCK_SECONDS = 60


def _password_required() -> bool:
    """Retourne True si un mot de passe est configuré."""
    return bool((settings.app_password or "").strip())


def _check(pwd: str) -> bool:
    """Comparaison constant-time pour éviter les timing attacks."""
    return hmac.compare_digest(pwd.encode("utf-8"), settings.app_password.encode("utf-8"))


def _render_login() -> None:
    """Affiche l'écran de login — hero centré, branding Sniper."""
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none; }
          [data-testid="collapsedControl"] { display: none; }
          .block-container { padding-top: 4rem !important; max-width: 460px !important; }
          .sniper-login-hero {
            text-align: center;
            padding: 32px 24px;
            background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(139,92,246,0.1));
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 18px;
            margin-bottom: 28px;
          }
          .sniper-login-icon { font-size: 3.2rem; margin-bottom: 8px; }
          .sniper-login-title {
            font-size: 1.8rem; font-weight: 800;
            background: linear-gradient(90deg,#10B981 0%,#8B5CF6 50%,#F59E0B 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
          }
          .sniper-login-sub { color: #94A3B8; font-size: 0.9rem; }
        </style>
        <div class="sniper-login-hero">
          <div class="sniper-login-icon">🔒</div>
          <div class="sniper-login-title">Investment Sniper</div>
          <div class="sniper-login-sub">Accès sécurisé — entrez le mot de passe</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Lockout après trop de tentatives
    locked_until = st.session_state.get(_LOCKED_UNTIL_KEY, 0)
    now = time.time()
    if locked_until > now:
        remaining = int(locked_until - now)
        st.error(f"🚫 Trop de tentatives. Réessaie dans {remaining} s.")
        st.stop()

    with st.form("sniper_login", clear_on_submit=True):
        pwd = st.text_input("Mot de passe", type="password", placeholder="••••••••")
        submit = st.form_submit_button("🔓 Déverrouiller", use_container_width=True)

    if submit:
        if _check(pwd):
            st.session_state[_SESSION_KEY] = True
            st.session_state[_ATTEMPTS_KEY] = 0
            st.rerun()
        else:
            attempts = st.session_state.get(_ATTEMPTS_KEY, 0) + 1
            st.session_state[_ATTEMPTS_KEY] = attempts
            left = _MAX_ATTEMPTS - attempts
            if left <= 0:
                st.session_state[_LOCKED_UNTIL_KEY] = now + _LOCK_SECONDS
                st.session_state[_ATTEMPTS_KEY] = 0
                st.error(f"🚫 Trop d'échecs — verrouillé pour {_LOCK_SECONDS} s.")
            else:
                st.error(f"❌ Mot de passe incorrect — {left} tentative(s) restante(s).")

    st.caption("🎯 Investment Sniper Bot · Accès privé")
    st.stop()


def require_auth() -> None:
    """Bloque le rendu de l'app tant que l'utilisateur n'est pas authentifié.

    À appeler en tête de `main()` — AVANT tout rendu de page.
    No-op si APP_PASSWORD n'est pas configuré.
    """
    if not _password_required():
        return
    if st.session_state.get(_SESSION_KEY):
        return
    _render_login()


def render_logout_button() -> None:
    """Ajoute un bouton logout dans la sidebar (si auth activée)."""
    if not _password_required():
        return
    if not st.session_state.get(_SESSION_KEY):
        return
    if st.sidebar.button("🔒 Se déconnecter", use_container_width=True, type="secondary"):
        st.session_state[_SESSION_KEY] = False
        st.rerun()
