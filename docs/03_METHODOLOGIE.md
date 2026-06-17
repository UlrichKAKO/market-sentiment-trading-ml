# Méthodologie

## Objectif prédictif

Le projet cherche à prédire la direction des rendements boursiers.

La cible est binaire :

```text
1 si le rendement futur est positif
0 sinon
```

## Pipeline Yahoo US

### 1. Collecte des prix

Les prix sont téléchargés via Yahoo Finance.

Chaque actif est identifié par son ticker.

### 2. Construction des variables de marché

Variables utilisées :

- `return_lag_1`
- `return_lag_2`
- `volatility_5d`
- `volatility_20d`
- `momentum_5d`
- `volume_change`
- `rsi_14`

### 3. Analyse de sentiment

Le projet tente d'utiliser FinBERT.

Si FinBERT n'est pas disponible, le projet utilise VADER comme fallback.

Variables de sentiment :

- `sentiment_mean`
- `sentiment_std`
- `sentiment_count`
- `sentiment_lag_1`
- `sentiment_rolling_3d`
- `sentiment_rolling_7d`

### 4. Expériences

Deux expériences sont comparées :

- `market_only`
- `market_plus_sentiment`

### 5. Modélisation

Modèles :

- Logistic Regression
- Random Forest
- Gradient Boosting

Le split est temporel, afin d'éviter d'entraîner le modèle sur des observations futures.

### 6. Backtest

Le backtest transforme les prédictions en positions :

- prédiction positive : position longue ;
- prédiction négative : position courte.

Des coûts de transaction sont appliqués.

## Pipeline Euronext

### 1. Agrégation microstructure

Les transactions Euronext sont agrégées par date et par ISIN.

Variables produites :

- nombre de trades ;
- volume échangé ;
- valeur échangée ;
- VWAP ;
- prix d'ouverture ;
- prix haut ;
- prix bas ;
- prix de clôture ;
- rendement intrajournalier ;
- amplitude high/low.

### 2. Construction des variables prédictives

Variables microstructure :

- `log_n_trades`
- `log_traded_volume`
- `log_traded_value`
- `vwap_to_open`
- `close_to_vwap`
- `high_low_spread`

Variables sentiment :

- `sentiment_mean`
- `sentiment_std`
- `sentiment_count`
- `sentiment_positive_ratio`
- `sentiment_negative_ratio`

### 3. Cible

Avec plusieurs dates disponibles, la cible est :

```text
direction du rendement du prochain jour
```

Formule :

```text
target_return = close_price(t+1) / close_price(t) - 1
target = 1 si target_return > 0, sinon 0
```

### 4. Expériences

Deux expériences sont comparées :

- `microstructure_only`
- `microstructure_plus_sentiment`

### 5. Mini-backtest

Le mini-backtest utilise les probabilités prédites.

Stratégies :

- long-only : acheter les instruments avec les plus fortes probabilités de hausse ;
- long/short : acheter les plus fortes probabilités et vendre les plus faibles ;
- benchmark : rendement moyen égal-pondéré.

### 6. Importance des variables

Le projet calcule l'importance des variables pour :

- Random Forest ;
- Gradient Boosting ;
- Logistic Regression, via la valeur absolue des coefficients.

## Métriques

Métriques de classification :

- accuracy ;
- balanced accuracy ;
- precision ;
- recall ;
- F1-score ;
- AUC.

Métriques financières :

- rendement total ;
- Sharpe ratio ;
- drawdown maximal ;
- hit ratio ;
- turnover ;
- nombre de trades.

## Justification méthodologique

Le projet sépare les deux univers :

- Yahoo US : séries longues quotidiennes ;
- Euronext : données haute fréquence agrégées sur quelques jours.

Cette séparation évite de mélanger des actifs, des identifiants et des fréquences de données différentes.
