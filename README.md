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

## Demarrage rapide (Docker)

Prerequis : Docker Desktop (avec `docker compose`) actif.

Configuration unique des ports (fichier local non versionne) :

1. Creer un fichier `.env` a la racine du projet.
2. Y definir tous les ports Docker utilises.
3. Modifier uniquement ce fichier pour simuler un autre environnement sur la meme machine.

Exemple de `.env` :

```env
COMPOSE_PROJECT_NAME=agritech_dev
API_HOST_PORT=8000
API_CONTAINER_PORT=8000
FRONTEND_HOST_PORT=8501
FRONTEND_CONTAINER_PORT=8501
MLFLOW_HOST_PORT=5000
MLFLOW_CONTAINER_PORT=5000
```

Exemple simulation prod locale (ports differents) :

```env
COMPOSE_PROJECT_NAME=agritech_prod_sim
API_HOST_PORT=18000
API_CONTAINER_PORT=8000
FRONTEND_HOST_PORT=18501
FRONTEND_CONTAINER_PORT=8501
MLFLOW_HOST_PORT=15000
MLFLOW_CONTAINER_PORT=5000
```

```powershell
./scripts/docker_up.ps1
```

Services exposes :

- API FastAPI : `http://localhost:${API_HOST_PORT}`
- Frontend Streamlit : `http://localhost:${FRONTEND_HOST_PORT}`
- MLflow UI : `http://localhost:${MLFLOW_HOST_PORT}`

Execution batch (data + training) :

```powershell
./scripts/docker_train.ps1
```

Execution analyse (ACP) :

```powershell
docker compose run --rm api python -m agritech.analysis.explore
```

Execution des tests + couverture :

```powershell
./scripts/docker_tests.ps1
```

Compatibilite scripts historiques (desormais Docker) :

```powershell
./scripts/bootstrap.ps1
./scripts/run_api.ps1
./scripts/run_frontend.ps1
./scripts/run_tests.ps1
```

Arret de la stack :

```powershell
./scripts/docker_down.ps1
```

Les artefacts produits sont ecrits dans `artifacts/` :

- `artifacts/data/consolidated_yield.csv`
- `artifacts/reports/crop_factor_pca_summary.json`
- `artifacts/models/model.joblib`
- `artifacts/models/metadata.json`

## MLflow

Le tracking est stocke dans `mlruns/` et expose via Docker sur `http://localhost:${MLFLOW_HOST_PORT}`.

## Livrables a completer ensuite

- exporter le rapport metier final en PDF a partir de `reports/business_report_outline.md` ;
- ajouter des captures MLflow apres les premiers runs ;
- configurer un deploiement automatique sur vos cibles finales.
