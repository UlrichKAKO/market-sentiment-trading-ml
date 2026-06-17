# Documentation complète du projet

## Titre du projet

Prédiction des rendements boursiers basé sur les informations

## Objectif général

Ce projet vise à prédire la direction des rendements boursiers à partir de deux familles d'informations :

- les informations de marché : prix, volumes, volatilité, momentum, microstructure ;
- les informations textuelles : actualités financières et sentiment associé.

Le projet contient deux pipelines complémentaires :

- un pipeline Yahoo Finance sur des actions américaines ;
- un pipeline Euronext sur des actions européennes identifiées par ISIN.

## Problématique

Dans quelle mesure les informations de marché et le sentiment extrait des actualités peuvent-ils améliorer la prédiction des rendements boursiers ?

## Architecture générale

```text
market-sentiment-trading-ml/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   ├── news.csv
│   │   └── euronext_news.csv
│   └── processed/
│       ├── results.csv
│       ├── model_dataset.csv
│       ├── euronext_microstructure.csv
│       ├── euronext_prediction_dataset.csv
│       ├── euronext_prediction_results.csv
│       ├── euronext_predictions.csv
│       ├── euronext_backtest.csv
│       └── euronext_feature_importance.csv
├── docs/
├── notebooks/
├── reports/
├── src/
│   ├── backtest.py
│   ├── config.py
│   ├── data_loader.py
│   ├── euronext_loader.py
│   ├── euronext_news_collector.py
│   ├── euronext_prediction.py
│   ├── features.py
│   ├── main.py
│   ├── models.py
│   └── sentiment.py
├── README.md
└── requirements.txt
```

## Résumé des pipelines

### Pipeline Yahoo US

Le pipeline principal utilise les prix Yahoo Finance et les news financières pour tester deux expériences :

- `market_only` : variables de marché uniquement ;
- `market_plus_sentiment` : variables de marché plus variables de sentiment.

Actifs par défaut :

- AAPL
- MSFT
- GOOGL
- AMZN
- META
- TSLA

Modèles :

- Logistic Regression
- Random Forest
- Gradient Boosting

Sorties :

- `data/processed/model_dataset.csv`
- `data/processed/results.csv`
- `data/processed/backtest_<ticker>_<experience>_<model>.csv`

### Pipeline Euronext

Le pipeline Euronext utilise des fichiers de marché bruts sur les journées disponibles :

- `20240102`
- `20240103`
- `20240104`
- `20240105`

Il agrège les transactions par ISIN et par date, puis prédit la direction du rendement du jour suivant.

Expériences :

- `microstructure_only`
- `microstructure_plus_sentiment`

Sorties :

- `data/processed/euronext_microstructure.csv`
- `data/processed/euronext_prediction_dataset.csv`
- `data/processed/euronext_prediction_results.csv`
- `data/processed/euronext_predictions.csv`
- `data/processed/euronext_backtest.csv`
- `data/processed/euronext_feature_importance.csv`
- `data/processed/euronext_instrument_mapping.csv`
- `data/processed/euronext_daily_sentiment.csv`

## Dashboard Streamlit

Le dashboard présente :

- le classement des modèles Yahoo US ;
- les courbes de backtest Yahoo US ;
- les statistiques Euronext ;
- les résultats du pipeline prédictif Euronext ;
- les graphes de performance ;
- les probabilités de hausse par instrument ;
- le mini-backtest Euronext ;
- l'importance des variables ;
- une conclusion synthétique.

Commande :

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501
```

URL locale :

```text
http://127.0.0.1:8501
```

## Résultats actuellement observés

### Yahoo US

Le dashboard affiche le meilleur modèle selon la métrique choisie, ainsi que :

- accuracy ;
- balanced accuracy ;
- precision ;
- recall ;
- F1-score ;
- AUC ;
- rendement total ;
- Sharpe ;
- drawdown maximal.

### Euronext

Le pipeline Euronext utilise une vraie cible temporelle :

```text
next_day_direction
```

Cela signifie qu'il cherche à prédire si le rendement du prochain jour sera positif.

Les résultats doivent être interprétés avec prudence, car la fenêtre disponible ne couvre que quatre jours de marché.

## Limites principales

- Les données Euronext ne couvrent que quatre journées.
- Les news Yahoo/Boursorama récupérées automatiquement sont récentes et ne couvrent pas janvier 2024.
- Le signal de sentiment Euronext est techniquement prêt, mais non exploitable tant que les news ne sont pas alignées temporellement avec les données de marché.
- Les backtests Euronext sont courts et illustratifs.
- Les modèles utilisés sont volontairement classiques et interprétables.

## Apports du projet

- Mise en place d'un pipeline complet de prédiction financière.
- Comparaison de modèles ML.
- Comparaison avec et sans sentiment.
- Intégration de données de microstructure Euronext.
- Création d'un mini-backtest.
- Visualisation interactive avec Streamlit.
- Documentation des limites méthodologiques.
