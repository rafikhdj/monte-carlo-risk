# Réduction du coût de la VaR et de l’ES à 99 %

Le projet compare 15 méthodes sur cinq lois de perte synthétiques. Les portefeuilles d’options sont réévalués en Black–Scholes sous des chocs gaussiens de marché. L’extension à quatre facteurs utilise une corrélation de type AR(1), avec `Corr(i,j) = 0,55^|i−j|`. Les positions sont courtes en puts et en calls, ce qui produit deux queues de pertes. À quatre facteurs, chaque sous-portefeuille dépend de son propre sous-jacent. La perte totale est additive, malgré la corrélation des chocs.

## Reproduire

Depuis ce dossier, avec Python 3.14 et un environnement virtuel dédié ou celui des TD Monte Carlo.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python benchmark.py --output results
.venv/bin/python analyze.py --output results
```

Pour un contrôle rapide, lancer `benchmark.py --smoke --output smoke`. Pour affiner uniquement les références dans un jeu de résultats existant, lancer `benchmark.py --references-only --ref-power 21`, puis `analyze.py`. Les graines, tailles et réglages retenus sont traçables dans les scripts et dans `results/protocol.json`. Les temps et la configuration pilotée par les temps peuvent varier selon la machine.

## Protocole

- Cinq modèles et 40 réplications d’évaluation par point. Les graines de calibration, de référence, d’appariement des temps et d’évaluation appartiennent à des plages disjointes. Les méthodes partagent les indices de réplication et passent dans un ordre randomisé. Une partie des tirages peut être commune selon l’algorithme.
- La calibration compare trois largeurs de domaine sur des pertes exactes tenues à part des nœuds. Elle compare aussi trois amplitudes de déplacement IS et trois multiplicateurs de scénarios en minimisant le produit du temps médian et de la somme des variances relatives des deux estimateurs, sur 16 graines pilotes. Le nombre de nœuds est fixé à 31. Les références VaR et ES n’interviennent ni dans la construction des splines ni dans cette sélection.
- À coût exact égal, les budgets sont 512, 2 048 et 8 192 scénarios. Chaque spline corrigée consomme 31 scénarios de construction, 32 points de contrôle indépendants sous une proposition favorisant les queues et le reste en valorisations exactes pour la correction. Tous les appels adaptatifs sont comptés et réutilisés. La spline non corrigée est aussi étudiée à 63 valorisations, avec trois références exactes à 63, et dans les deux régimes de temps.
- À temps visé égal, un pilote ajuste les nombres de scénarios à la médiane du MC classique, aux ancres de 2 048 et 8 192 valorisations. La comparaison principale utilise les temps effectivement observés. Un ratio de temps entre 0,8 et 1,2 est considéré comme apparié. Le plancher de 128 et le coût fixe des splines peuvent empêcher l’appariement. Les nombres d’évaluations exactes sont indiqués séparément.
- Le chronomètre comprend la génération, la construction de la spline, les contrôles, la prédiction, la valorisation et les deux mesures de risque. La calibration et l’appariement des temps sont des coûts hors ligne mesurés séparément. Les constantes communes du modèle, la Cholesky et le prix initial sont mis en cache au premier appel et partagés par toutes les méthodes. Le benchmark est séquentiel, avec un plafond d’un fil par bibliothèque numérique.
- Le pic mémoire est un échantillon mesuré par tracemalloc, sur un appel distinct par configuration. Il mesure les allocations suivies, y compris les buffers NumPy traçables, et peut manquer des allocations natives. Les p10/p90 de durée et le p90 de l’erreur absolue décrivent la dispersion des runs.
- Budget d’arrêt fixé à cette grille, avec 40 répétitions. Critère de poursuite générale par modèle testé sur les deux ancres de temps. Une même variante spline doit réduire d’au moins 20 % la RMSE de chaque mesure face à chacune des quatre références exactes (MC, stratifié, IS et IS + stratification). Elle doit prendre au plus 120 % du temps correspondant et chacun des intervalles exploratoires du ratio de RMSE doit rester sous 1. Les intervalles bootstrap sont appariés par indice de graine et évalués aussi aux extrémités de la marge d’incertitude de la référence. Ils ne sont pas corrigés pour les comparaisons multiples.

## Estimateurs et diagnostics

Pour des pertes (L_i) et des poids positifs, la VaR est l’inverse généralisé de la CDF empirique après normalisation des poids. L’ES est évalué par (q + (1-\alpha)^{-1} \, \mathbb E[(L-q)_+]) avec (q) égal au quantile estimé et l’espérance empirique pondérée. Cette définition alloue la fraction appropriée d’un atome à la VaR. Le MC non pondéré utilise une partition, qui évite le tri intégral. Student et mélange utilisent des générateurs directs dans le MC classique.

L’IS porte sur les facteurs latents. La proposition (g) est un mélange de normales de même covariance, avec 20 % de la loi cible (f). Les rapports (f/g) sont calculés en log, sont bornés par 5 et ont une espérance de 1 sous (g). Les tests sur les poids vérifient aussi une probabilité de dépassement et les moments des facteurs. La stratification IS répartit des strates d’égale masse sur le sélecteur du mélange puis sur le premier facteur conditionnel. Une strate qui chevauche deux composantes garde la règle d’échantillonnage du mélange. L’ESS est un diagnostic de concentration des poids, à lire avec la RMSE car la dépendance entre observations stratifiées change leur variance. La normalisation empirique des poids et le passage au quantile rendent les estimateurs généralement biaisés à taille finie.

La scission gaussienne génère deux vecteurs indépendants et calcule ((Z_1+Z_2)/\sqrt{2}) avant de corréler les facteurs. Elle conserve la loi, ce qui est testé, et ajoute une génération et une opération de recombinaison. Le test supplémentaire conditionne la paire sur une somme stratifiée. À partir de la somme normalisée stratifiée Z et d’un auxiliaire gaussien indépendant W, les deux composantes sont (Z+W)/√2 et (Z−W)/√2. Leur somme normalisée retrouve Z. Cette variante reprend les paramètres gelés de spline + stratification et chronomètre les deux variantes en alternance à taille identique. Les temps appariés de ce contrôle sont dans `decomposition_bridge` du protocole. La scission à elle seule conserve la loi finale.

Les splines cubiques utilisent les grilles uniforme, quantile de facteur, densifiée près des extrémités et adaptative par raffinement des intervalles à forte variation. Aux extrémités, on prolonge linéairement selon la tangente. La construction multidimensionnelle exploite des splines marginales des composantes additives. Pour le mélange gaussien, la branche discrète est conservée exactement et seule la composante gaussienne est interpolée.

L’estimateur corrigé vaut `R(grosses simulations spline) + R(sous-échantillon exact) − R(sous-échantillon spline)`. Les deux derniers termes utilisent les mêmes chocs et les mêmes poids. La correction est une différence de mesures de risque, pas une moyenne des erreurs de prix. Elle n’impose pas l’absence de biais à budget fini. Avec deux tailles croissantes, des CDF convergentes, une VaR régulière et des queues uniformément intégrables, la différence des limites annule la contribution de la spline fixe. Pour le quantile, une faible densité au seuil ou des discontinuités peuvent ralentir la convergence alors que l’ES reste sensible aux erreurs extrêmes par intégration.

Sur un même espace probabilisé, une erreur de perte uniformément bornée par (\epsilon) donne `|ΔVaR| ≤ ε` et `|ΔES| ≤ ε`. Pour l’ES, la formule par partie positive donne en outre `|ΔES| ≤ E|ΔL| / (1−α)`. Une faible RMSE des prix sur 32 contrôles ne certifie aucune de ces bornes. La Student transformée croît rapidement dans les queues latentes. Une extrapolation jamais observée sur l’échantillon reste possible dans la distribution.

Les références sont analytiques pour normale et Student, et calculées par recherche de racine de la survie du mélange, avec moment tronqué analytique. Les options à un facteur ont une référence par racines de la probabilité de dépassement et quadrature des deux queues. Pour les options corrélées, huit plans Sobol brouillés indépendants de (2^{21}) points utilisent uniquement les valorisations exactes. Le protocole conserve le demi-intervalle de Student entre brouillages et l’écart au calcul à demi-taille. La transformation des points est protégée aux bornes de la CDF pour garder des scénarios finis.

La comparaison de chacun des trois budgets montre la convergence empirique. Les colonnes `bias_*`, `bias_se_*`, `variance_*`, `rmse_*` et `p90_error_*` de `summary.csv` permettent d’isoler le biais de la dispersion. `fit_rmse` et `fit_max` concernent l’approximation des pertes sur les contrôles, tandis que `rmse_var` et `rmse_es` concernent les estimateurs complets. La masse observée en extrapolation et les corrections moyennes de risque sont enregistrées séparément.

## Résultats et bibliographie

Voir [rapport.md](rapport.md) et les figures dans [results](results). Les réplications individuelles, les résumés et les ratios bootstrap sont dans `runs.csv`, `summary.csv` et `comparisons.csv`.

- L’usage conjoint d’IS et de la stratification pour la VaR précède cette étude. [Glasserman, Heidelberger et Shahabuddin, 2000](https://pubsonline.informs.org/doi/10.1287/mnsc.46.10.1349.12274)
- La formulation de l’ES par partie positive et le traitement des atomes suivent la définition du CVaR pour les lois générales. [Rockafellar et Uryasev, 2002](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)
- La correction couplée s’inspire d’une différence de niveaux. Les résultats de complexité MLMC pour une espérance ne s’appliquent pas directement à la différence de quantiles étudiée ici. [Giles, 2008](https://pubsonline.informs.org/doi/10.1287/opre.1070.0496)
