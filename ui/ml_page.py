"""Page Analyse ML PRO : métriques, ROC, PR, feature importance, hyperparams."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.trainer import load_models


def _metric_header(metrics: dict) -> None:
    ds = metrics.get("dataset", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📊 Échantillons", f"{ds.get('n_rows', 0):,}")
    c2.metric("✅ Positifs", f"{ds.get('n_positives', 0):,}")
    c3.metric("❌ Négatifs", f"{ds.get('n_negatives', 0):,}")
    c4.metric("🎯 Tickers", ds.get("n_tickers", 0))
    c5.metric("🧬 Features", ds.get("n_features", 0))
    c6.metric("⏱️ Train", f"{metrics.get('total_train_time_sec', 0):.0f}s")


def _models_comparison(metrics: dict) -> None:
    rows = []
    for key in ["random_forest", "xgboost", "lstm"]:
        m = metrics.get(key, {})
        rows.append({
            "Modèle": m.get("name", key),
            "Accuracy": m.get("accuracy", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
            "F1": m.get("f1", 0),
            "ROC-AUC": m.get("roc_auc", 0),
            "Best Threshold": m.get("best_threshold", 0.5),
            "OOB Score": m.get("oob_score", 0),
            "Train (s)": m.get("train_time_sec", 0),
            "N pos": m.get("n_positives", 0),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                                  "Best Threshold", "OOB Score"]
        }).background_gradient(cmap="RdYlGn", subset=["Accuracy", "F1", "ROC-AUC"]),
        use_container_width=True, hide_index=True,
    )

    melted = df.melt(
        id_vars="Modèle", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        var_name="Métrique", value_name="Score",
    )
    fig = px.bar(melted, x="Modèle", y="Score", color="Métrique", barmode="group",
                 template="plotly_dark", title="Comparaison des métriques par modèle",
                 height=400)
    st.plotly_chart(fig, use_container_width=True)


def _roc_curves(metrics: dict) -> None:
    rocs = metrics.get("roc_curves", {})
    fig = go.Figure()
    for key, color in [("random_forest", "#10B981"), ("xgboost", "#F59E0B"), ("lstm", "#EF4444")]:
        d = rocs.get(key, {})
        if d.get("fpr"):
            auc = metrics.get(key, {}).get("roc_auc", 0)
            fig.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"],
                name=f"{metrics.get(key, {}).get('name', key)} (AUC={auc:.3f})",
                line=dict(color=color, width=2.5),
            ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                             name="Random (AUC=0.5)", showlegend=True))
    fig.update_layout(
        title="Courbes ROC", template="plotly_dark",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def _feature_importance(metrics: dict) -> None:
    imp = metrics.get("feature_importance", {})
    tab_rf, tab_xgb, tab_combined = st.tabs(["🌲 Random Forest", "⚡ XGBoost", "🔀 Combinée"])

    for tab, key in [(tab_rf, "random_forest"), (tab_xgb, "xgboost")]:
        with tab:
            imp_dict = imp.get(key, {})
            if not imp_dict:
                st.info("Pas de données")
                continue
            df_imp = (pd.DataFrame([{"Feature": k, "Importance": v} for k, v in imp_dict.items()])
                        .sort_values("Importance", ascending=True))
            fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                         template="plotly_dark", height=650,
                         color="Importance", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

    with tab_combined:
        rf_imp = imp.get("random_forest", {})
        xgb_imp = imp.get("xgboost", {})
        combined = []
        for name in rf_imp:
            combined.append({
                "Feature": name,
                "RF": rf_imp.get(name, 0),
                "XGB": xgb_imp.get(name, 0),
                "Moyenne": (rf_imp.get(name, 0) + xgb_imp.get(name, 0)) / 2,
            })
        df_c = pd.DataFrame(combined).sort_values("Moyenne", ascending=False)
        st.dataframe(
            df_c.style.format({"RF": "{:.4f}", "XGB": "{:.4f}", "Moyenne": "{:.4f}"})
                .background_gradient(cmap="viridis", subset=["Moyenne"]),
            use_container_width=True, hide_index=True, height=500,
        )


def _hyperparameters(metrics: dict) -> None:
    hp = metrics.get("hyperparameters", {})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌲 Random Forest — Hyperparamètres")
        rf = hp.get("random_forest", {})
        st.dataframe(
            pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in rf.items()]),
            use_container_width=True, hide_index=True, height=380,
        )
        st.markdown("#### 🧠 LSTM — Architecture")
        lstm = hp.get("lstm", {})
        st.dataframe(
            pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in lstm.items()]),
            use_container_width=True, hide_index=True, height=380,
        )

    with col2:
        st.markdown("#### ⚡ XGBoost — Hyperparamètres")
        xgb = hp.get("xgboost", {})
        st.dataframe(
            pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in xgb.items()]),
            use_container_width=True, hide_index=True, height=380,
        )
        st.markdown("#### 🔍 Isolation Forest")
        iso = hp.get("isolation_forest", {})
        st.dataframe(
            pd.DataFrame([{"Paramètre": k, "Valeur": str(v)} for k, v in iso.items()]),
            use_container_width=True, hide_index=True, height=200,
        )
        st.markdown("#### ⚖️ Poids du consensus")
        weights = hp.get("consensus_weights", {})
        df_w = pd.DataFrame([{"Modèle": k.upper(), "Poids": v} for k, v in weights.items()])
        fig = px.pie(df_w, values="Poids", names="Modèle", template="plotly_dark",
                     hole=0.4, color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444", "#8B5CF6"])
        st.plotly_chart(fig, use_container_width=True)


def _dataset_stats(metrics: dict) -> None:
    ds = metrics.get("dataset", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Statistiques du dataset")
        df = pd.DataFrame([
            {"Métrique": "Nombre total d'échantillons", "Valeur": f"{ds.get('n_rows', 0):,}"},
            {"Métrique": "Exemples positifs (décollage)", "Valeur": f"{ds.get('n_positives', 0):,}"},
            {"Métrique": "Exemples négatifs", "Valeur": f"{ds.get('n_negatives', 0):,}"},
            {"Métrique": "Équilibre de classes", "Valeur": f"{ds.get('class_balance', 0):.2%}"},
            {"Métrique": "Nombre de features", "Valeur": ds.get("n_features", 0)},
            {"Métrique": "Nombre de tickers", "Valeur": ds.get("n_tickers", 0)},
            {"Métrique": "Fenêtre lookback", "Valeur": f"{ds.get('lookback_days', 0)} jours"},
            {"Métrique": "Horizon prédiction", "Valeur": f"{ds.get('horizon_days', 0)} jours"},
            {"Métrique": "Seuil de gain", "Valeur": f"+{ds.get('gain_threshold_pct', 0):.0f}%"},
        ])
        st.dataframe(df, use_container_width=True, hide_index=True, height=360)
    with col2:
        st.markdown("#### 🎯 Répartition des classes")
        fig = go.Figure(data=[go.Pie(
            labels=["Positifs (décollage)", "Négatifs"],
            values=[ds.get("n_positives", 0), ds.get("n_negatives", 0)],
            hole=0.5,
            marker=dict(colors=["#10B981", "#6B7280"]),
        )])
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, use_container_width=True)


def render_ml() -> None:
    st.title("🧠 Analyse des modèles — PRO")
    st.caption("Métriques d'entraînement, hyperparamètres ultra-avancés, ROC, feature importance")

    models = load_models()
    metrics = models.get("metrics", {})
    if not metrics:
        st.warning("⚠️ Aucun modèle entraîné. Lance `python main.py train` dans un terminal.")
        st.code("cd investment-sniper\npython main.py train", language="bash")
        return

    _metric_header(metrics)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Comparaison", "📉 Courbes ROC", "🧬 Feature Importance",
        "⚙️ Hyperparamètres", "📊 Dataset",
    ])
    with tab1:
        _models_comparison(metrics)
    with tab2:
        _roc_curves(metrics)
    with tab3:
        _feature_importance(metrics)
    with tab4:
        _hyperparameters(metrics)
    with tab5:
        _dataset_stats(metrics)

    st.markdown("---")
    with st.expander("🔬 Métriques brutes (JSON)"):
        st.json(metrics)
