# Rapport metier final - Agritech Answers

Date : 2026-06-12

## 1. Resume executif

Le prototype Agritech Answers permet de :

- predire le rendement agricole attendu pour une culture donnee ;
- recommander la culture la plus pertinente selon un contexte de parcelle.

Le modele retenu est un Random Forest optimise en deux etapes (RandomizedSearchCV puis GridSearchCV). Les performances obtenues sont elevees :

- RMSE : 7560.18
- MAE : 2979.00
- R2 : 0.99398

Conclusion metier : le prototype est suffisamment robuste pour aider a la decision agronomique sur des contextes comparables aux donnees historiques utilisees.

## 2. Contexte et objectif metier

Le besoin est d'orienter les decisions culturales en reduisant l'incertitude sur le rendement attendu.

Objectifs business :

- anticiper les rendements selon climat et pratiques agricoles ;
- comparer des options de culture sur une meme parcelle ;
- outiller les equipes operationnelles avec une API et une interface simple.

Public cible :

- exploitants et responsables d'exploitation ;
- responsables operationnels et chefs de projet data/metier.

## 3. Donnees exploitees et perimetre

Sources consolidees pour la prediction :

- `yield.csv` (cible rendement) ;
- `rainfall.csv` (pluie) ;
- `pesticides.csv` (usage pesticides) ;
- `temp.csv` (temperature).

Variables finales du modele :

- categorielles : `Area`, `Item` ;
- numeriques : `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`.

Le dataset consolide de reference est :

- `artifacts/data/consolidated_yield.csv`

## 4. Demarche d'analyse et de modelisation

La demarche retenue est la suivante :

1. nettoyage et normalisation des donnees (dont harmonisation des noms de pays) ;
2. fusion des sources sur `pays normalise + annee` ;
3. verification qualite et production d'un dataset consolide ;
4. comparaison de modeles ;
5. optimisation des hyperparametres ;
6. suivi des runs dans MLflow ;
7. deploiement du modele via API et interface front.

## 5. Resultats de performance

Modele final retenu : Random Forest optimise.

Hyperparametres finaux :

- `n_estimators` : 800
- `min_samples_split` : 3
- `min_samples_leaf` : 1
- `max_features` : `sqrt`
- `max_depth` : `None`
- `bootstrap` : `False`

Metriques finales (jeu de test) :

- RMSE = 7560.1764
- MAE = 2978.9975
- R2 = 0.9939773

Interpretation metier :

- le modele explique une tres grande part de la variabilite du rendement ;
- l'erreur absolue moyenne reste contenue a l'echelle du probleme ;
- la performance est compatible avec un usage d'aide a la decision (et non de certitude absolue).

## 6. Justification du choix Random Forest

Le choix Random Forest est justifie par :

- une meilleure robustesse sur relations non lineaires ;
- une bonne gestion des interactions entre variables climatiques et agricoles ;
- de meilleures metriques apres optimisation par rapport aux alternatives testees.

La selection a ete conduite de maniere traçable via MLflow, avec historique des runs, parametres et metriques.

## 7. Traduction metier des enseignements

Les leviers les plus actionnables dans le cadre du prototype :

- adequation culture/contexte local (`Area`, `Item`) ;
- gestion des pratiques liees aux pesticides ;
- prise en compte du signal climatique (pluie, temperature) ;
- choix d'une culture en fonction du rendement predit dans un contexte donne.

Exemple d'usage : pour une parcelle et une annee donnees, comparer plusieurs cultures candidates et retenir celle au meilleur rendement predit, puis valider la decision avec contrainte terrain (cout, disponibilite, risque).

## 8. Preuves et traçabilite

### Captures MLflow

- `docs/capture_mlflow_experience_recherche_hyperparametres.png`
- `docs/capture_mlflow_experience_finale.png`
- `docs/capture_mlflow_detail_run_parametres.png`
- `docs/capture_mlflow_detail_run_metriques.png`

### Artefacts techniques

- dataset consolide : `artifacts/data/consolidated_yield.csv`
- rapport fusion : `artifacts/data/merge_report.json`
- modele final : `artifacts/models/model.joblib`
- metadonnees modele : `artifacts/models/metadata.json`

## 9. Limites et conditions d'usage

Limites actuelles :

- pas de variable economique directe (prix, marge, couts logistiques) dans le modele ;
- pas de mesure explicite de profitabilite dans les metriques ;
- couverture conditionnee aux distributions historiques disponibles.

Bonnes pratiques d'usage :

- utiliser la prediction comme support de decision, pas comme verite absolue ;
- confronter les recommandations a l'expertise terrain ;
- reentrainer periodiquement avec de nouvelles donnees.

## 10. Plan d'action recommande

Priorites court terme :

1. ajouter un indicateur de profitabilite (metrique metier) ;
2. finaliser la partie deploiement automatise en CI/CD ;
3. exporter ce rapport en PDF pour diffusion metier.

Priorites moyen terme :

1. enrichir les features (sol, irrigation, intrants, prix) ;
2. ajouter des garde-fous de monitoring data/model drift ;
3. industrialiser la boucle d'amelioration continue MLOps.

## 11. Conclusion

Le projet atteint son objectif principal : fournir une chaine de valeur complete, de la donnee brute a une recommandation exploitable via API et interface.

Le modele final est performant, traçable et justifiable. Le prototype est pret pour une phase de consolidation metier (profitabilite, deploiement automatise, formalisation du reporting final PDF).
