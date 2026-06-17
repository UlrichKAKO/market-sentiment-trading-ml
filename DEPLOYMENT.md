# Déploiement du dashboard Streamlit

Ce projet est prêt pour un déploiement sur Streamlit Community Cloud.

## 1. Vérifier l'application en local

```bash
streamlit run app/streamlit_app.py
```

L'application doit afficher le dashboard à l'adresse locale indiquée par Streamlit.

## 2. Créer un dépôt Git local

Si le projet n'est pas encore versionné :

```bash
git init
git add .
git commit -m "Prepare Streamlit deployment"
```

## 3. Créer un dépôt GitHub

Créer un nouveau dépôt sur GitHub, puis relier le dossier local :

```bash
git branch -M main
git remote add origin https://github.com/<utilisateur>/<nom-du-depot>.git
git push -u origin main
```

Remplacer `<utilisateur>` et `<nom-du-depot>` par les valeurs de ton compte GitHub.

## 4. Déployer sur Streamlit Community Cloud

1. Aller sur Streamlit Community Cloud.
2. Cliquer sur **New app**.
3. Sélectionner le dépôt GitHub.
4. Renseigner le fichier principal :

```text
app/streamlit_app.py
```

5. Cliquer sur **Deploy**.

## 5. Dépendances

Le fichier `requirements.txt` est volontairement léger pour le déploiement du dashboard.
Il contient seulement les bibliothèques nécessaires à l'affichage de l'application.

Le fichier `requirements-full.txt` conserve les dépendances complètes pour reproduire
les pipelines de collecte, NLP, FinBERT, Yahoo et Euronext en local.

## 6. Données embarquées

Le dashboard lit des fichiers déjà générés dans `data/processed/`.
Les principaux fichiers nécessaires au déploiement sont autorisés dans `.gitignore`,
notamment :

- `results.csv`
- `backtest_*.csv`
- `euronext_microstructure.csv`
- `euronext_prediction_results.csv`
- `euronext_predictions.csv`
- `euronext_backtest.csv`
- `euronext_feature_importance.csv`

Sans ces fichiers, l'application affichera un message demandant de relancer les pipelines.
