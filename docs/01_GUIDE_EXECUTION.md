# Guide d'exécution

## Installation

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dans l'environnement actuel du projet, les commandes peuvent être lancées avec :

```bash
.venv/bin/python
```

## Relancer le pipeline Yahoo US

```bash
.venv/bin/python -m src.main
```

Cette commande génère notamment :

- `data/processed/prices.csv`
- `data/processed/news_scored_finbert.csv`
- `data/processed/daily_sentiment_finbert.csv`
- `data/processed/model_dataset.csv`
- `data/processed/results.csv`
- les fichiers `backtest_*.csv`

## Relancer le pipeline Euronext

```bash
.venv/bin/python -m src.euronext_prediction
```

Cette commande génère notamment :

- `data/processed/euronext_microstructure.csv`
- `data/processed/euronext_prediction_dataset.csv`
- `data/processed/euronext_prediction_results.csv`
- `data/processed/euronext_predictions.csv`
- `data/processed/euronext_backtest.csv`
- `data/processed/euronext_feature_importance.csv`
- `data/processed/euronext_daily_sentiment.csv`

## Collecter les news Euronext

Collecte Yahoo Finance sur les instruments Euronext les plus liquides :

```bash
.venv/bin/python -m src.euronext_news_collector --limit 30
```

Collecte avec Boursorama en plus :

```bash
.venv/bin/python -m src.euronext_news_collector --limit 30 --include-boursorama
```

Attention : les news collectées automatiquement sont principalement récentes. Elles ne sont donc pas forcément exploitables pour expliquer les journées Euronext de janvier 2024.

## Lancer le dashboard

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501
```

Puis ouvrir :

```text
http://127.0.0.1:8501
```

## Ordre recommandé d'exécution

```bash
.venv/bin/python -m src.main
.venv/bin/python -m src.euronext_prediction
.venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501
```

Optionnel :

```bash
.venv/bin/python -m src.euronext_news_collector --limit 30
.venv/bin/python -m src.euronext_prediction
```

## Problèmes fréquents

### Streamlit ne se met pas à jour

Recharger la page ou relancer :

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501
```

### `Lignes ML sentiment = 0`

Cela signifie que les news collectées ne correspondent pas aux dates de marché Euronext disponibles.

Le projet ne force pas l'utilisation de news mal datées pour éviter une fuite temporelle.

### FinBERT ne fonctionne pas

Sur Python 3.9, certaines versions récentes de `transformers` exigent `torch>=2.6`, qui n'est pas toujours disponible.

Le projet utilise donc un fallback VADER lorsque FinBERT n'est pas disponible.
