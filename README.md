# Agritech Answers

Prototype MLOps pour la prediction de rendement agricole et la recommandation de culture, base sur les jeux de donnees fournis dans `DATA/`.

## Objectifs

- consolider les donnees historiques de rendement avec les variables climatiques et pesticides ;
- produire une analyse ACP sur le dataset agronomique `crop_yield.csv` ;
- entrainer un modele de regression suivi dans MLflow ;
- exposer le modele via une API FastAPI ;
- fournir une interface Streamlit pour la prediction et la recommandation ;
- industrialiser le flux avec tests, Docker et CI GitHub Actions.

## Structure

- `agritech/data/preprocess.py` : consolidation du dataset source de verite.
- `agritech/analysis/explore.py` : ACP et resume analytique sur `crop_yield.csv`.
- `agritech/models/train.py` : entrainement, comparaison de modeles et sauvegarde d'artefacts.
- `agritech/models/predictor.py` : chargement des artefacts et logique d'inference.
- `api/main.py` : API FastAPI (`/predict`, `/recommend`, `/metadata`).
- `frontend/app.py` : interface Streamlit.
- `tests/` : tests unitaires sur la preparation et l'API.
- `.github/workflows/ci.yml` : pipeline de test et build Docker.

## Demarrage rapide

```powershell
./scripts/bootstrap.ps1
./scripts/run_api.ps1
./scripts/run_frontend.ps1
```

## Pipeline data et modele

```powershell
py -m agritech.data.preprocess
py -m agritech.analysis.explore
py -m agritech.models.train
```

Les artefacts produits sont ecrits dans `artifacts/` :

- `artifacts/data/consolidated_yield.csv`
- `artifacts/reports/crop_factor_pca_summary.json`
- `artifacts/models/model.joblib`
- `artifacts/models/metadata.json`

## MLflow

Le script d'entrainement configure un tracking local dans `mlruns/`.

```powershell
py -m mlflow ui --backend-store-uri ./mlruns
```

## Livrables a completer ensuite

- exporter le rapport metier final en PDF a partir de `reports/business_report_outline.md` ;
- ajouter des captures MLflow apres les premiers runs ;
- configurer un deploiement automatique sur vos cibles finales.
