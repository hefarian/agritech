🌾 Analyse des facteurs qui influencent les rendements agricoles
🎯 Pourquoi cette analyse ?
Nous avons étudié un grand volume de données agricoles pour répondre à une question simple :

Qu’est-ce qui explique les différences de rendement entre les cultures ?

Pour cela, nous avons utilisé une méthode qui permet de résumer beaucoup d’informations en quelques grandes tendances faciles à comprendre.

🔍 Ce que l’on a trouvé
✅ 1. Le facteur le plus important : les ressources disponibles
Le premier élément qui explique les différences de rendement est :

la quantité de pluie 🌧️
l’utilisation d’engrais 🌱
l’irrigation 🚿

👉 En résumé :

Plus une culture bénéficie d’eau et de ressources, plus son rendement est élevé.

C’est la tendance principale observée dans les données.

✅ 2. Un deuxième facteur : climat vs pratiques agricoles
Le second élément met en évidence une opposition entre :

les conditions naturelles, comme la température 🌡️
les interventions humaines, comme l’utilisation d’engrais

👉 Cela signifie qu’il existe différents types d’agriculture :

certaines reposent surtout sur le climat
d’autres sont plus intensives (avec davantage d’intrants)


📊 Comment lire ces résultats ?
Pour simplifier, on peut imaginer un graphique avec deux axes :

👉 horizontal : niveau de rendement (faible → élevé)
👉 vertical : type d’agriculture (climat vs intrants)

Chaque point représente une exploitation ou une culture.

🌍 Ce que ça nous apprend

Les rendements ne sont pas dus à un seul facteur
Ils dépendent d’un équilibre entre environnement et pratiques agricoles
Il existe une grande diversité de situations (pas de profils totalement séparés)
# Rapport synthetique metier - Rendement et profit agricole

Date : 2026-06-14

## 1. Message cle pour la direction

Le prototype Agritech Answers est operationnel pour soutenir les decisions de culture.
Il combine :

- une base consolidee multi-sources propre et exploitable ;
- un modele de prediction de rendement tres performant ;
- une lecture explicative des facteurs agronomiques via ACP (notebook 1).

En synthese business :

- le rendement est fortement lie au couple eau + pratiques agronomiques ;
- les profils de production varient sur un continuum climat/intrants ;
- l outil peut deja optimiser les choix culturaux en rendement attendu ;
- l optimisation du profit est possible des maintenant via une couche de decision economique au-dessus des predictions.

## 2. Ce que disent les donnees (base notebook 1)

### 2.1 Qualite et perimetre de la base consolidee

Le notebook 1 construit un dataset consolide a partir de :

- yield.csv
- rainfall.csv
- pesticides.csv
- temp.csv

Regles de consolidation :

- jointure sur zone geographique normalisee + annee ;
- retrait des lignes avant 1990 ;
- exclusion des lignes avec variables critiques manquantes.

Resultats de consolidation :

- 38 382 lignes exploitables ;
- periode 1990 a 2013 ;
- 110 zones geographiques ;
- 10 cultures ;
- doublons restants : 4 200 (a traiter prioritairement dans le prochain cycle data quality).

Signaux de couverture avant exclusion des manquants (part de manquants) :

- pluie : 12.45 %
- pesticides : 21.20 %
- temperature : 18.60 %

Lecture metier : la couverture est suffisante pour produire un modele utile, mais l amont data doit etre renforce pour reduire le risque de biais sur certains pays/periodes.

### 2.2 Variables utilisees pour la prediction

Variables categorielles :

- Area
- Item

Variables numeriques :

- Year
- average_rain_fall_mm_per_year
- pesticides_tonnes
- avg_temp

Ces variables sont coherentes avec la logique metier : contexte local, choix de culture, dynamique temporelle et exposition climatique/technique.

## 3. Performance modele et implication business

Modele final retenu : Random Forest optimise.

Metriques test :

- RMSE : 7 560.18
- MAE : 2 979.00
- R2 : 0.99398

Parametres finaux :

- n_estimators : 800
- min_samples_split : 3
- min_samples_leaf : 1
- max_features : sqrt
- max_depth : None
- bootstrap : False

Interpretation metier :

- le modele explique une part tres elevee de la variabilite observee ;
- il est assez fiable pour classer des scenarios et prioriser des options de culture ;
- il doit etre utilise comme aide a la decision, pas comme garantie contractuelle de rendement.

## 4. Variables cles et enseignements explicatifs (ACP du notebook 1)

Le notebook 1 ajoute une ACP sur un echantillon de 50 000 observations du dataset agronomique complementaire.

Variance expliquee (ACP pipeline) :

- PC1 : 21.60 %
- PC2 : 11.08 %
- total PC1+PC2 : 32.68 %

Principales contributions observees :

- PC1 est principalement porte par Yield_tons_per_hectare, Rainfall_mm, Fertilizer_Used, Irrigation_Used.
- PC2 oppose surtout Temperature_Celsius et Fertilizer_Used (axes de strategie differente selon contexte).

Interpretation metier :

- axe 1 : intensite productive liee a la disponibilite en eau et aux intrants ;
- axe 2 : arbitrage climat vs intensification ;
- les projections d individus montrent plutot un continuum de profils qu une segmentation totalement nette.

Conclusion sur les variables cles :

- eau (pluie + irrigation) ;
- fertilisation/intrants ;
- temperature ;
- choix culture x territoire.

## 5. Recommandations pour optimiser rendement et profit

### 5.1 Recommandations immediates (0 a 3 mois)

1. Operationaliser une grille de decision scenario par scenario.
Comparer pour chaque parcelle 3 a 5 cultures candidates avec prediction de rendement, puis classer selon score economique.

2. Ajouter une couche profit simple dans la decision.
Calculer une marge previsionnelle par scenario :
Profit estime = prix attendu x rendement predit - couts variables (intrants, eau, interventions).

3. Prioriser les leviers a plus fort effet.
Piloter finement irrigation et fertilisation, car ce sont les dimensions les plus associees aux gains de rendement dans l ACP.

4. Encadrer le risque climatique.
Segmenter les recommandations par regimes de temperature/pluie pour limiter les choix trop agressifs en annees contraintes.

### 5.2 Recommandations moyen terme (3 a 9 mois)

1. Integrer de vraies variables economiques.
Ajouter prix de vente, cout de l eau, cout intrants, cout energie, cout logistique, pour passer d une optimisation rendement a une optimisation marge.

2. Nettoyer les doublons residuels et renforcer la couverture.
Traiter les 4 200 doublons et ameliorer la disponibilite pesticides/temperature/pluie pour augmenter la stabilite des recommandations.

3. Construire un tableau de bord pilotage rendement-profit.
Suivre prediction vs realise, marge theorique vs marge reelle, ecarts par culture/zone/periode.

4. Mettre en place des seuils de decision.
Ne recommander un scenario que si le gain de profit attendu depasse un seuil minimal et reste robuste sous hypotheses climatiques degradees.

### 5.3 Recommandations gouvernance

1. Formaliser un comite metier-data mensuel.
Objectif : valider les hypotheses de cout/prix et ajuster les regles de recommandation.

2. Institutionnaliser le reentrainement periodique.
Frequence proposee : trimestrielle, ou plus frequente en cas de rupture climat/prix.

3. Documenter les decisions et leurs resultats.
Conserver l historique des scenarios recommandes et des choix reels pour progresser vers une boucle d amelioration continue.

## 6. Plan KPI pour suivre l impact

KPI rendement :

- ecart rendement predit vs realise (%)
- taux de scenarios dont rendement reel >= rendement cible

KPI profit :

- marge brute/ha (reelle et previsionnelle)
- gain de marge vs reference historique
- ratio cout intrants / tonne produite

KPI risque et adoption :

- taux de recommandation appliquee
- dispersion des resultats par zone
- part de scenarios abandonnes pour risque trop eleve

## 7. Limites actuelles

- Le modele predit le rendement, pas directement le profit.
- Les metriques economiques ne sont pas encore nativement integrees au training.
- Les conclusions ACP restent descriptives et non causales.

Ces limites ne bloquent pas l usage metier, mais elles imposent un cadre de decision prudent et trace.

## 8. Conclusion operative

Le projet est deja a un niveau utile pour la decision agricole.
La combinaison prediction + lecture explicative permet de :

- mieux choisir les cultures selon contexte ;
- agir sur les bons leviers agronomiques ;
- preparer une optimisation du profit a court terme via une couche economique.

Prochaine etape prioritaire : passer d un pilotage rendement a un pilotage rendement + marge, avec KPI financiers explicites et boucle d amelioration continue.