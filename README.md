# Market Sentiment Trading — ML / DL / LLM

## Objectif

Ce projet étudie si le sentiment extrait de news financières permet d'améliorer la prédiction directionnelle des rendements boursiers sur un univers de grandes valeurs technologiques.

## Documentation complète

La documentation détaillée est disponible dans le dossier `docs/` :

- `docs/00_DOCUMENTATION_PROJET.md` : vue d'ensemble du projet
- `docs/01_GUIDE_EXECUTION.md` : commandes d'installation et d'exécution
- `docs/02_DONNEES.md` : documentation des données
- `docs/03_METHODOLOGIE.md` : méthodologie des pipelines
- `docs/04_DASHBOARD.md` : guide du dashboard Streamlit
- `docs/05_PLAN_MEMOIRE.md` : plan de mémoire proposé
- `docs/06_DICTIONNAIRE_VARIABLES.md` : dictionnaire des variables

## Données

- Prix de marché : Yahoo Finance
- News financières : `data/raw/news.csv`
- Actifs étudiés par défaut : AAPL, MSFT, GOOGL, AMZN, META, TSLA
- Données de marché Euronext : dossier `20240104/` avec messages `FullTradeInformation`, `StandingData`, `PriceUpdate`, etc.

## Méthodologie

1. Collecte des prix
2. Nettoyage des news
3. Extraction du sentiment avec FinBERT
4. Feature engineering marché + sentiment
5. Comparaison d'expériences : marché seul vs marché + sentiment
6. Modélisation ML avec séparation temporelle par actif
7. Backtest long/short avec coûts de transaction
8. Évaluation financière et dashboard Streamlit

## Modèles testés

- Logistic Regression
- Random Forest
- Gradient Boosting

> XGBoost n'est pas inclus par défaut pour éviter les problèmes fréquents de `libomp` sur Mac.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Lancement

Depuis la racine du projet :

```bash
python -m src.main
```

Puis pour le dashboard :

```bash
streamlit run app/streamlit_app.py
```

Le pipeline écrit les principaux artefacts dans `data/processed/` :

- `prices.csv` : prix téléchargés ou réutilisés depuis le cache
- `news_scored_finbert.csv` : news scorées par FinBERT
- `daily_sentiment_finbert.csv` : sentiment agrégé par jour
- `model_dataset.csv` : dataset final de modélisation
- `results.csv` : métriques de classification et de performance financière
- `backtest_<ticker>_<experience>_<modele>.csv` : courbes de backtest
- `euronext_microstructure.csv` : agrégation quotidienne des trades Euronext par ISIN
- `euronext_prediction_dataset.csv` : dataset du pipeline prédictif Euronext
- `euronext_prediction_results.csv` : métriques des modèles Euronext
- `euronext_predictions.csv` : prédictions instrument par instrument
- `euronext_backtest.csv` : mini-backtest Euronext long-only et long/short
- `euronext_feature_importance.csv` : importance des variables par modèle
- `euronext_instrument_mapping.csv` : table ISIN, symbole, nom et marché
- `euronext_daily_sentiment.csv` : sentiment quotidien agrégé par ISIN

## Données Euronext 20240104

Le projet sait agréger les fichiers bruts Euronext du dossier `20240104/`.
L'agrégation exploite `StandingData_20240104.csv` pour les métadonnées instruments et les fichiers `FullTradeInformation_20240104_<ISIN>.csv` pour les transactions.

Colonnes produites :

- nombre de trades
- volume échangé
- valeur échangée
- VWAP
- prix open/high/low/close intrajournalier
- rendement intrajournalier
- amplitude high/low

## Pipeline Prédiction Euronext

Le pipeline Euronext est séparé du pipeline Yahoo US :

```bash
python -m src.euronext_prediction
```

Objectif cible :

- avec plusieurs dates Euronext : prédire la direction du rendement du prochain jour par ISIN ;
- avec une seule date disponible : produire une classification exploratoire cross-section du rendement intrajournalier.

Cette séparation évite de mélanger les tickers Yahoo US avec les ISIN Euronext et garde une méthodologie claire.

### Format des news Euronext

Pour activer réellement l'expérience `microstructure_plus_sentiment`, ajouter un fichier :

```text
data/raw/euronext_news.csv
```

Format recommandé :

```csv
date,isin,text
2024-01-03,FR0000120271,"TotalEnergies announces..."
```

Colonnes minimales :

- `date`
- `text`

Colonne recommandée :

- `isin`

Si `isin` est absent, le pipeline tente une correspondance approximative à partir du symbole ou du nom de société, mais la correspondance par ISIN est préférable.

Un collecteur expérimental peut aussi créer ce fichier depuis Yahoo Finance et Boursorama :

```bash
python -m src.euronext_news_collector --limit 80
```

Pour ajouter aussi une collecte générale Boursorama :

```bash
python -m src.euronext_news_collector --limit 80 --include-boursorama
```

Limite importante : les accès simples à Yahoo/Boursorama donnent surtout des news récentes. Pour expliquer les journées Euronext de janvier 2024, il faut idéalement des news datées du 2024-01-02 au 2024-01-05.

## Format attendu du fichier news.csv

```csv
date,text
2020-01-03,"Apple shares rise after strong iPhone demand..."
```

Si un fichier de sentiment traité existe déjà mais ne recouvre pas la période d'étude, le pipeline repart du fichier brut `data/raw/news.csv`.

## Structure

```text
market-sentiment-trading-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/news.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── sentiment.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   └── main.py
├── reports/figures/
└── app/streamlit_app.py
```
