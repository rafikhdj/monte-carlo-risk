# VaR et Expected Shortfall à 99 %

**Le benchmark ne confirme pas un gain général du montage combiné.** L’interpolation réduit sensiblement le nombre de valorisations exactes sur les deux portefeuilles synthétiques. À temps comparable, son avantage supplémentaire face à l’IS stratifié dépend du portefeuille, du budget et de la mesure de risque. Aucune variante à spline ne franchit le critère conjoint prédéfini sur les deux budgets de temps.

## Options à quatre facteurs corrélés

Au budget de temps élevé, le MC exact prend 4,07 ms, avec des RMSE de **1,109 pour la VaR** et **1,486 pour l’ES**. L’IS stratifié exact prend également 4,07 ms et réduit ces erreurs à **0,648 et 0,444**.

- **Spline + IS + stratification** prend 4,05 ms pour 1 275 valorisations exactes, contre 6 120 pour l’IS stratifié. Les RMSE sont **0,518 et 0,400**, soit environ **20 % et 10 % de moins**. Les intervalles exploratoires des ratios sont [0,557 ; 1,124] et [0,697 ; 1,117]. Au budget de temps inférieur, les deux erreurs sont environ 16 % et 17 % plus élevées que celles de l’IS stratifié.
- **Spline + IS** prend 4,08 ms pour 1 211 valorisations exactes. Les RMSE sont **0,406 et 0,436**. Face à l’IS stratifié, le ratio pour la VaR est **0,626** [0,452 ; 0,862]. Pour l’ES il est **0,981** [0,780 ; 1,194]. Le pic d’allocations suivi est de **2,08 Mio**, contre **5,65 Mio** pour l’IS stratifié. Cette indication mémoire provient d’un appel par configuration.
- **Spline sans correction** utilise 63 valorisations exactes. À 4,15 ms, elle donne des RMSE de **0,615 et 0,714**. La RMSE de l’ES est ici 61 % plus élevée que pour l’IS stratifié. Elle reste intéressante quand l’appel au pricer lui-même est la ressource rare.
- À **8 192 valorisations exactes** pour chaque méthode, spline + IS + stratification abaisse les RMSE de **0,474 et 0,354** à **0,171 et 0,152** par rapport à l’IS stratifié. Le temps passe de **5,41 ms à 24,0 ms** et le pic suivi de **7,57 à 14,66 Mio**. L’augmentation du nombre de scénarios approchés explique une part du gain de précision et une part du coût.

La référence est **54,4135 pour la VaR** et **64,3153 pour l’ES**. Les demi-intervalles entre brouillages Sobol sont respectivement **0,0149 et 0,00271**. L’écart au calcul à demi-taille est de **0,00889 et 0,00145**.

## Résultats par modèle

- **Normale.** La stratification directe du facteur dominant réduit les RMSE du MC d’environ **96 % et 93 %** au budget de temps élevé. La spline reproduit une perte affine à la précision machine, mais son coût fixe dépasse largement le temps du MC.
- **Student à 5 degrés de liberté.** Au budget de temps élevé, l’IS stratifié produit des RMSE de **0,0151 et 0,00965**, contre **0,0877 et 0,192** pour le générateur Student direct. Au budget inférieur, le coût fixe de l’IS le place hors appariement. La RMSE de prix sous la proposition de contrôle monte à **0,596** avec la grille en quantiles, contre **0,000272** pour la grille uniforme. Une grille dense au centre laisse des intervalles trop larges vers les extrémités du domaine latent.
- **Mélange gaussien.** Au budget de temps élevé, l’IS stratifié réduit la RMSE de l’ES de 32 % face au MC et augmente celle de la VaR de 16 %. La stratification du seul premier facteur ne contrôle pas directement l’appartenance à la composante rare. Cette composante explique une partie de la variance résiduelle. Le coût fixe des méthodes à spline exclut leur appariement avec le MC dans la plage testée.
- **Options à un facteur.** Au budget de temps élevé, l’IS stratifié donne des RMSE de **0,00587 et 0,00144**, contre **0,00502 et 0,00146** avec la spline combinée. Les deux intervalles du ratio de RMSE contiennent 1. La spline combinée abaisse l’erreur de l’ES au budget inférieur, mais dépasse le temps de l’IS stratifié d’environ 40 %. La référence par quadrature vaut **16,79730 et 19,95541**.
- **Grilles et scission.** Le contrôle de prix à quatre facteurs donne une RMSE d’environ **0,00059** pour l’uniforme, **0,00214** pour l’adaptative, **0,00287** pour la grille de queue et **0,180** pour celle en quantiles. Avec correction, ces écarts d’ajustement peuvent déboucher sur des RMSE de risque proches. La scission gaussienne conserve la loi et ne montre aucun avantage de précision reproductible contre le MC. Elle coûte particulièrement cher sur Student, où elle requiert le passage par le quantile. La combinaison scission + stratification + spline reproduit le facteur stratifié par construction. À taille identique et 8 192 valorisations, son contrôle chronométré en alternance montre un surcoût de 7 % à 36 % selon le modèle, dont 9,5 % pour les options corrélées.

## Convergence et limites

Entre 512 et 8 192 valorisations, les pentes empiriques de la RMSE du MC en échelle logarithmique sont proches de −0,5. Sur le portefeuille corrélé, les exposants observés sont environ **0,47 et 0,46** pour le MC, puis **0,54 et 0,54** pour spline + IS + stratification. Sur le cas unidimensionnel, l’IS stratifié atteint environ **0,99 et 1,15**. Trois tailles et 40 graines décrivent la plage testée, avec une incertitude sur les pentes asymptotiques.

Le biais estimé de la spline combinée à temps élevé et quatre facteurs est de **+0,145** pour la VaR (erreur-type **0,080**) et **+0,096** pour l’ES (erreur-type **0,062**). Le signe du biais reste incertain. Les RMSE de cet estimateur tiennent compte des deux contributions, biais et dispersion.

Les temps ont été mesurés sur un Apple M3 et dépendent de la machine et de l’implémentation des pricers. Aux budgets faibles, certains coûts fixes empêchent de réaliser des comparaisons appariées. Les cas synthétiques sont lisses. Les quatre facteurs permettent des splines marginales par additivité du portefeuille. Une option panier ou une barrière introduirait des interactions ou des irrégularités absentes de cette étude. Les contrôles en queue et la mesure de la masse extrapolée n’établissent pas une borne uniforme. Une analyse locale de la sensibilité du quantile et des queues reste nécessaire.

## Décision et questions suivantes

Abandonner la scission gaussienne comme optimisation autonome et écarter le montage combiné comme solution générale. **Poursuivre l’étude de spline + IS sur des pricers coûteux**, avec l’IS stratifié exact comme référence principale et un seuil de réussite pour chacune des deux mesures. Les résultats à quatre facteurs justifient ce test ciblé grâce à la baisse du nombre de valorisations exactes et à l’amélioration observée pour la VaR.

1. À quel coût par revalorisation l’interpolation devient-elle avantageuse face à l’IS stratifié, construction et contrôles compris ?
2. Une correction sur la fonction de répartition au voisinage du quantile et sur la partie positive pour l’ES stabilise-t-elle mieux les deux estimateurs qu’une différence de quantiles empiriques ?
3. Comment l’avantage évolue-t-il lorsqu’on remplace le portefeuille additif par une option panier ou à barrière, avec une référence numérique de précision contrôlée ?
4. Une stratification du sélecteur de composante, ou une direction de queue estimée par gradient sur des graines pilotes, améliore-t-elle la variance sans accroître excessivement le coût des poids ?

[Erreur selon le temps](results/erreur_temps.png) · [Erreur selon le nombre de valorisations](results/erreur_exact.png) · [Tableau complet](results/summary.csv) · [Comparaisons et intervalles](results/comparisons.csv) · [Protocole et bibliographie](README.md)
