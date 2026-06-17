# Dictionnaire des variables

## Variables Yahoo US

| Variable | Description |
|---|---|
| `return` | Rendement logarithmique quotidien |
| `return_lag_1` | Rendement retardé d'un jour |
| `return_lag_2` | Rendement retardé de deux jours |
| `volatility_5d` | Volatilité glissante sur 5 jours |
| `volatility_20d` | Volatilité glissante sur 20 jours |
| `momentum_5d` | Momentum sur 5 jours |
| `volume_change` | Variation relative du volume |
| `rsi_14` | Relative Strength Index sur 14 jours |
| `target` | Direction du rendement futur |

## Variables de sentiment Yahoo US

| Variable | Description |
|---|---|
| `sentiment_mean` | Sentiment moyen quotidien |
| `sentiment_std` | Ecart-type du sentiment |
| `sentiment_count` | Nombre de news |
| `sentiment_lag_1` | Sentiment retardé d'un jour |
| `sentiment_rolling_3d` | Moyenne glissante du sentiment sur 3 jours |
| `sentiment_rolling_7d` | Moyenne glissante du sentiment sur 7 jours |

## Variables Euronext microstructure

| Variable | Description |
|---|---|
| `n_trades` | Nombre de transactions |
| `traded_volume` | Quantité totale échangée |
| `traded_value` | Valeur totale échangée |
| `vwap` | Prix moyen pondéré par les volumes |
| `open_price` | Premier prix traité de la journée |
| `high_price` | Plus haut prix traité |
| `low_price` | Plus bas prix traité |
| `close_price` | Dernier prix traité de la journée |
| `intraday_return` | Rendement intrajournalier |
| `high_low_spread` | Amplitude relative high/low |

## Variables Euronext utilisées par le modèle

| Variable | Description |
|---|---|
| `log_n_trades` | Logarithme du nombre de trades |
| `log_traded_volume` | Logarithme du volume échangé |
| `log_traded_value` | Logarithme de la valeur échangée |
| `vwap_to_open` | Ecart relatif entre VWAP et prix d'ouverture |
| `close_to_vwap` | Ecart relatif entre clôture et VWAP |
| `high_low_spread` | Amplitude intrajournalière |

## Variables de sentiment Euronext

| Variable | Description |
|---|---|
| `sentiment_mean` | Sentiment moyen par date et ISIN |
| `sentiment_std` | Dispersion du sentiment |
| `sentiment_count` | Nombre de news associées |
| `sentiment_positive_ratio` | Part des news positives |
| `sentiment_negative_ratio` | Part des news négatives |

## Variables de sortie Euronext

| Variable | Description |
|---|---|
| `target_return` | Rendement du prochain jour |
| `target` | 1 si le rendement futur est positif, 0 sinon |
| `prediction` | Classe prédite |
| `probability_positive` | Probabilité prédite d'une hausse |
