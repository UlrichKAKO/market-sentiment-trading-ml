# Documentation des données

## Données Yahoo US

Source :

- Yahoo Finance via `yfinance`.

Actifs :

- AAPL
- MSFT
- GOOGL
- AMZN
- META
- TSLA

Période :

- de `START_DATE` à `END_DATE`, définis dans `src/config.py`.

Colonnes principales :

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `ticker`

## News Yahoo US

Fichier :

```text
data/raw/news.csv
```

Format :

```csv
date,text
2020-01-03,"Apple shares rise after strong iPhone demand..."
```

Colonnes :

- `date` : date de publication ;
- `text` : texte ou titre de news.

## Données Euronext brutes

Dossiers utilisés :

- `20240102`
- `20240103`
- `20240104`
- `20240105`

Chaque dossier contient des sous-dossiers par ISIN.

Types de fichiers observés :

- `FullTradeInformation`
- `StandingData`
- `PriceUpdate`
- `OrderUpdate`
- `MarketUpdate`
- `MarketStatusChange`
- `Statistics`

Le pipeline utilise principalement :

- `StandingData` pour les métadonnées instruments ;
- `FullTradeInformation` pour les transactions exécutées.

## Agrégation Euronext

Fichier généré :

```text
data/processed/euronext_microstructure.csv
```

Colonnes principales :

- `date`
- `isin`
- `symbol`
- `name`
- `market`
- `country`
- `currency`
- `instrument_id`
- `n_trades`
- `traded_volume`
- `traded_value`
- `vwap`
- `open_price`
- `high_price`
- `low_price`
- `close_price`
- `intraday_return`
- `high_low_spread`
- `first_trade_time`
- `last_trade_time`

## Mapping instruments

Fichier généré :

```text
data/processed/euronext_instrument_mapping.csv
```

Objectif :

- relier ISIN, symbole, nom de société, marché et devise ;
- permettre la collecte et l'association des news.

## News Euronext

Fichier attendu :

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

Colonnes optionnelles :

- `symbol`
- `company`
- `source`
- `url`

## Sentiment Euronext

Fichier généré :

```text
data/processed/euronext_daily_sentiment.csv
```

Colonnes :

- `date`
- `isin`
- `sentiment_mean`
- `sentiment_std`
- `sentiment_count`
- `sentiment_positive_ratio`
- `sentiment_negative_ratio`

## Remarque importante sur l'alignement temporel

Les données Euronext couvrent janvier 2024, alors que les news collectées automatiquement via Yahoo/Boursorama sont principalement récentes.

Le pipeline n'utilise le sentiment dans le modèle que si les dates des news correspondent aux dates de marché disponibles.

Cela évite une fuite temporelle.
