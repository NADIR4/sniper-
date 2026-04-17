# 🎯 Investment Sniper Bot

Système Python complet de détection d'opportunités d'investissement basé sur machine learning. Interface Streamlit déployable en ligne (pas localhost).

## 🧠 Comment ça marche

1. **Collect** — Télécharge 3 ans de données OHLCV pour ~100 actions (S&P500 + EuroStoxx) via yfinance
2. **Filter** — Garde les actions ayant fait +100% ou plus sur la période
3. **Train** — Entraîne 4 modèles ML (RandomForest, XGBoost, LSTM, IsolationForest) à reconnaître les patterns des 60 jours précédant un décollage
4. **Scan** — Toutes les 15 minutes, scanne le marché actuel et calcule un score de consensus pondéré
5. **Signal** — Si score ≥ 0.75, génère un signal avec prix cibles (+50/+100/+200%), stop-loss (ATR×2) et R/R
6. **Notify** — Envoie email + sauvegarde SQLite
7. **Report** — Export Excel pro avec 6 feuilles, graphiques, formatage conditionnel
8. **Dashboard** — Interface Streamlit 5 pages

## 🚀 Installation

```bash
cd investment-sniper
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# édite .env avec tes clés SMTP
```

## 🏃 Utilisation

### Entraînement initial (une fois)
```bash
python main.py collect    # télécharge les données
python main.py train      # entraîne les 4 modèles
```

### Lancer le scanner manuel
```bash
python main.py scan       # scan unique
python main.py schedule   # scan auto toutes les 15 min
python main.py report     # génère rapport Excel
```

### Lancer l'interface web
```bash
streamlit run app.py
# → http://localhost:8501
```

## 🌐 Déploiement en ligne (pas localhost)

### Option A — Streamlit Community Cloud (gratuit)
1. Pousse le projet sur GitHub
2. Va sur https://streamlit.io/cloud
3. Clique "New app", choisis ton repo, branche, `app.py`
4. Ajoute tes secrets (SMTP, etc.) dans **Settings → Secrets** au format TOML

### Option B — Railway.app
```bash
railway login
railway init
railway up
```
Ajoute un service cron pour `python main.py scan` toutes les 15 min.

### Option C — VPS (Docker)
```bash
docker build -t sniper .
docker run -p 8501:8501 --env-file .env sniper
```

### Dockerfile recommandé
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📂 Structure

```
investment-sniper/
├── app.py                 # Streamlit entry
├── main.py                # CLI pipeline
├── config.py              # Settings depuis .env
├── data/
│   ├── collector.py       # yfinance + SQLite
│   ├── cache.py           # SQLAlchemy models
│   └── universe.py        # Top 100 tickers
├── ml/
│   ├── features.py        # 25 indicateurs techniques
│   ├── trainer.py         # RF + XGB + LSTM + IsoForest
│   └── scanner.py         # Consensus temps réel
├── signals/
│   ├── generator.py       # Signal + targets + R/R
│   └── notifier.py        # Email SMTP
├── reports/
│   └── exporter.py        # Excel 6 feuilles
├── ui/
│   ├── dashboard.py       # KPIs + graphique
│   ├── scanner_page.py    # Scan temps réel
│   ├── signals_page.py    # Table + export
│   ├── ml_page.py         # Métriques ML
│   └── settings_page.py   # Config
└── tests/                 # pytest unitaires
```

## ⚙️ Configuration (`.env`)

| Variable | Défaut | Description |
|----------|--------|-------------|
| `CONSENSUS_THRESHOLD` | 0.75 | Score minimum pour générer un signal |
| `SCAN_INTERVAL_MINUTES` | 15 | Période du scanner auto |
| `LOOKBACK_DAYS` | 60 | Fenêtre pré-décollage (features) |
| `HISTORY_YEARS` | 3 | Historique téléchargé |
| `MIN_GAIN_PCT` | 100 | Seuil de gain pour "gainer" historique |
| `SMTP_*` | — | Envoi d'alertes email |

## 🧪 Tests

```bash
pytest tests/ -v --cov=.
```

## ⚠️ Disclaimer

Cet outil est **éducatif**. Les signaux ne constituent **pas** un conseil financier. Les performances passées ne préjugent pas des performances futures. Utilisez à vos propres risques.
