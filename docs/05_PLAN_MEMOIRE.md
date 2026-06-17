# Plan de mémoire proposé

## Titre

Prédiction des rendements boursiers basée sur les informations de marché et l'analyse de sentiment

## Introduction

- Contexte des marchés financiers.
- Rôle de l'information dans la formation des prix.
- Apport potentiel du machine learning.
- Apport potentiel de l'analyse de sentiment.
- Problématique.
- Objectifs du projet.
- Présentation des deux pipelines.

## Revue de littérature

- Hypothèse d'efficience des marchés.
- Prédiction des rendements boursiers.
- Machine learning appliqué à la finance.
- Analyse de sentiment financière.
- FinBERT et VADER.
- Microstructure de marché.
- Backtesting et évaluation financière.

## Données

### Données Yahoo US

- Source.
- Tickers.
- Période.
- Variables utilisées.
- News financières.

### Données Euronext

- Période disponible.
- Structure des fichiers.
- Rôle des ISIN.
- Agrégation des transactions.
- Variables de microstructure.

### Données textuelles

- News Yahoo/Boursorama.
- Problème d'alignement temporel.
- Limites du sentiment Euronext.

## Méthodologie

### Pipeline Yahoo US

- Collecte.
- Feature engineering.
- Sentiment.
- Modèles.
- Backtest.

### Pipeline Euronext

- Lecture des données brutes.
- Agrégation microstructure.
- Construction de la cible.
- Expériences avec et sans sentiment.
- Mini-backtest.
- Importance des variables.

## Résultats

### Résultats Yahoo US

- Tableau des modèles.
- Equity curve.
- Discussion de la performance.

### Résultats Euronext

- Statistiques descriptives.
- Résultats de classification.
- Mini-backtest.
- Importance des variables.
- Discussion du rôle du sentiment.

## Discussion

- Difficulté de prédire les rendements.
- Intérêt des variables de marché.
- Intérêt potentiel du sentiment.
- Importance de l'alignement temporel.
- Interprétation prudente des résultats Euronext.

## Limites

- Peu de jours Euronext.
- News non alignées avec janvier 2024.
- Modèles classiques.
- Coûts de transaction simplifiés.
- Absence de validation longue période pour Euronext.

## Conclusion

- Réponse à la problématique.
- Synthèse des résultats.
- Apports du projet.
- Ouverture vers des extensions futures.

## Perspectives

- Obtenir des news historiques par ISIN.
- Ajouter plus de jours Euronext si disponibles.
- Tester XGBoost ou LightGBM.
- Validation walk-forward.
- Backtest plus robuste.
- Optimisation de portefeuille.

## Annexes

- Structure du code.
- Dictionnaire des variables.
- Commandes d'exécution.
- Captures du dashboard.
- Exemples de fichiers CSV.
