# Documentation du dashboard

## Lancement

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501
```

URL :

```text
http://127.0.0.1:8501
```

## Titre

Le dashboard affiche :

```text
Prédiction des rendements boursiers basé sur les informations
```

## Section Yahoo US

Cette section affiche :

- le classement des modèles ;
- le choix du ticker ;
- la métrique de classement ;
- le meilleur modèle ;
- le Sharpe ;
- le rendement total ;
- l'equity curve.

## Section Données Euronext

Cette section affiche :

- nombre d'instruments ;
- nombre de dates ;
- nombre total de trades ;
- valeur échangée ;
- détail par instrument ;
- top 20 des instruments par valeur échangée.

## Section Pipeline Prédiction Euronext

Cette section affiche :

- mode de prédiction ;
- nombre de dates ;
- nombre d'instruments ;
- taux de rendements positifs ;
- nombre de news collectées ;
- nombre de news reliées à un ISIN ;
- nombre de lignes ML avec sentiment.

## Graphes Euronext

Graphes disponibles :

- performance des modèles ;
- F1-score par modèle ;
- mini-backtest Euronext ;
- importance des variables ;
- distribution des probabilités de hausse ;
- top probabilités de hausse.

## Section Conclusion

La section Conclusion résume :

- le rôle du pipeline Yahoo US ;
- le rôle du pipeline Euronext ;
- les limites ;
- les prochaines étapes.

## Interprétation des compteurs de sentiment

### News Euronext

Nombre total de news collectées.

### News reliées

Nombre de news associées à un ISIN.

### ISIN avec sentiment

Nombre d'instruments pour lesquels un sentiment a été calculé.

### Lignes ML sentiment

Nombre d'observations du dataset prédictif qui ont réellement un sentiment utilisable aux bonnes dates.

Si cette valeur est `0`, les news collectées ne couvrent pas la période de marché étudiée.
