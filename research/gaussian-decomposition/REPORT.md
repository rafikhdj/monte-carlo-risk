# Décomposition gaussienne et stratification Monte Carlo

## Question et théorie

L’expérience estime l’espérance de fonctions d’un vecteur normal standard. Elle compare le tirage direct de ce vecteur à la somme normalisée de composantes normales indépendantes, stratifiées ou non. La cinquième question du mandat se termine sur « ou ». Nous la lisons comme une question d’attribution du gain éventuel entre la décomposition et le plan de stratification.

La scission i.i.d. conserve exactement la loi de la somme et sa variance Monte Carlo. Les plans stratifiés, eux, changent la dépendance **entre** lignes de simulation. Dans un LHS correctement permuté, la loi de chaque ligne reste le produit des lois normales des composantes. Dans la grille conjointe, chaque ligne a la loi conditionnelle de sa cellule et la moyenne de cellule porte la masse probabiliste de celle-ci. Les appariements de rang identique ou opposé entre deux composantes modifieraient la loi jointe. L’antithèse testée utilise le signe opposé d’une ligne complète.

L’[audit mathématique](AUDIT.md) établit trois résultats utiles à l’interprétation :

- Pour des strates indépendantes, la variance est la somme des variances de cellule multipliées par le carré de leur masse et divisées par l’effectif de la cellule. À même nombre de strates équiprobables et allocation proportionnelle, les strates ordonnées par la valeur de la fonction minimisent cette variance. Une fonction monotone bénéficie ainsi directement des quantiles de son facteur. Les LHS dépendants et les fonctions non monotones sortent de ce résultat.
- Sur une fonction scalaire, la scission en `k` composantes multiplie par `k^(1−m)` la part de variance d’ordre Hermite `m` que la stratification marginale de premier ordre peut éliminer asymptotiquement. L’effet préservé est celui d’ordre 1. Pour la fonction carré, la variance résiduelle asymptotique du LHS vaut asymptotiquement `2(1−1/k)/N`. L’effet de l’additivité sur le LHS est documenté par Stein. [Stein, 1987](https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488205)
- Intégrer conditionnellement une composante peut réduire la variance par Rao–Blackwell. Cette voie exploite une opération mathématique supplémentaire et les exemples analytiques de l’audit n’entrent pas dans le classement expérimental de boîte noire.

## Protocole et reproductibilité

- Dix-neuf intégrandes à référence analytique. En dimension 1, les tests couvrent les fonctions linéaire, carrée, exponentielle, sinus, call, indicatrice rare et deux pics. Aux dimensions 2, 5, 10 et 20, on compare une somme de carrés, une exponentielle de projection et une moyenne de fonctions d’interactions voisines.
- Budgets exacts de 128, 512, 2 048 et 8 192 appels. Cent réplications indépendantes par point. Le CPU comprend génération, projection, préparation de grille, sondes et évaluations. Une campagne distincte rapproche les coûts CPU de ceux de la stratification directe en dimension 1 et projetée en dimension supérieure aux ancres 512 et 2 048. Les tailles sont sélectionnées sur cinq graines pilotes de chronométrage indépendantes. Le classement apparié retient les ratios CPU dans [0,8 ; 1,25]. Sobol garde une puissance de deux et son coût fixe peut empêcher un appariement.
- Les références comprennent MC, quantiles directs du premier facteur, projection par rotation orthogonale, LHS, LHS projeté, antithétique et Sobol brouillé avec ou sans la même rotation. La direction est connue par la définition du problème. Elle est donnée à tous les plans projetés. La grille et les LHS sur composantes utilisent cette même direction. Pour les interactions, la direction projette sur le premier couple du cycle. Elle ne résume qu’une partie des interactions.
- La grille conjointe coupe les deux composantes de la projection en quantiles. Aux quatre budgets exacts, elle tire une fois dans chaque cellule. Pour les tailles appariées non divisibles par le nombre de cellules, elle répartit les appels restants et pondère toujours chaque **moyenne de cellule** par sa masse. Les autres facteurs sont tirés directement sous la loi conditionnelle orthogonale. Stratifier conjointement toutes les composantes de tous les facteurs créerait une grille exponentielle.
- L’allocation de Neyman utilise 16 strates et **48 appels pilotes** dans le budget de chaque réplication. Les appels finaux sont indépendants du pilote et chaque cellule en reçoit au moins un. Une régularisation de 10 % de l’écart-type moyen atténue les zéros de variance estimés sur trois sondes. Le même dispositif compare une grille de composantes 4 × 4 et 16 tranches sur la projection. [Étoré et Jourdain](https://arxiv.org/abs/0711.4514)
- Le contrôle d’allocation séparé tient la partition fixe. Il compare Neyman à deux répartitions uniformes, l’une au même coût total et l’autre au même nombre de tirages finaux. Les 45 600 mesures utilisent une autre plage de graines.
- Les ratios de variance et de RMSE ont des intervalles bootstrap appariés à 95 %, avec 1 500 rééchantillonnages des indices de réplication. Ce sont des intervalles exploratoires par paire. Les tests de biais bilatéraux de Student sont accompagnés de leur correction FDR. Pour les cas antithétiques impairs, les erreurs d’arrondi sont traitées comme des zéros. Le test de loi prélève une seule ligne par plan indépendant pour éviter une fausse indépendance entre lignes d’un même LHS.
- Arrêt fixé à ces quatre budgets exacts et deux ancres CPU. Un gain reproductible en dimension élevée doit rester présent à deux budgets exacts successifs face à la stratification directe ou projetée **et** face à la meilleure référence LHS ou Sobol pertinente. Au minimum un appariement CPU doit le confirmer. La comparaison CPU est limitée aux temps effectivement appariés.

Depuis ce dossier, dans le venv du dépôt :

```sh
python -m unittest -v
python experiment.py
python analyze.py
python allocation.py
```

Le protocole JSON consigne les versions, les graines et les tailles choisies. Le mode `python experiment.py --smoke --output /tmp/gaussian-smoke` exécute le premier budget avec quatre réplications.

## Résultats et décision

### 1. La scission stratifiée réduit-elle la variance face au MC ?

**Oui, dans plusieurs cas.** Pour l’exponentielle à 8 192 appels, la grille conjointe ramène la variance de **4,89 × 10⁻⁴ à 3,33 × 10⁻⁵**. Le ratio est **0,068**, avec un intervalle à 95 % de **[0,046 ; 0,101]**. Le CPU augmente d’un facteur **4,62**. La scission i.i.d. a la même variance théorique que le MC et ses fluctuations empiriques sont compatibles avec cette identité à l’échelle de la campagne.

### 2. Fait-elle mieux que la stratification directe ?

**Les gains au-delà de la stratification directe sont ponctuels.** Sur la même exponentielle, la stratification directe atteint une RMSE de **0,00139**, contre **0,00577** pour la grille, soit un ratio **4,14** [3,17 ; 5,57]. Pour l’indicatrice rare à 8 192 appels, les RMSE sont **1,71 × 10⁻⁵** en direct et **3,81 × 10⁻⁴** avec la grille. Le seuil coupe au plus une strate en direct. Il traverse de nombreuses cellules de la grille latente.

À temps comparable face à la référence informée, les seuls intervalles de gain qui excluent 1 concernent le LHS scindé antithétique sur les fonctions impaires, linéaire et sinus. L’antithétique direct annule aussi ces deux erreurs et demande moins de CPU. L’[audit](AUDIT.md) donne en plus un contre-exemple mathématique à deux appels où stratifier une composante bat les quantiles directs. L’antithétique direct y gagne également.

### 3. Fait-elle mieux que Sobol ou LHS ?

**Elle peut battre un plan non projeté. La comparaison change lorsque la référence dispose de la même direction.** Pour l’exponentielle projetée en dimension 20 à 8 192 appels, la grille a un tiers de la RMSE du LHS non projeté, ratio **0,335** [0,269 ; 0,411]. Comparée au LHS projeté, sa RMSE est **4,06 fois plus grande** [3,10 ; 5,35]. Sobol projeté atteint **0,00278**, contre **0,00577** pour la grille. Sur les pics multiples, Sobol direct atteint **1,04 × 10⁻⁶** et la grille **4,64 × 10⁻⁴** au même budget.

### 4. Le gain tient-il en dimension élevée ?

**Aucune méthode scindée n’atteint le critère face aux références informées.** Sur l’exponentielle projetée en dimension 20, au régime CPU ciblant 2 048 appels de la référence, la grille utilise **1 562 appels** contre **2 072** pour la projection directe. Le CPU est effectivement apparié, avec un ratio de **1,01**. La RMSE de la grille est **2,35 fois plus élevée** [1,67 ; 3,70]. Sur les interactions en dimension 20 à 8 192 appels, **Sobol direct donne 1,26 × 10⁻⁴** de RMSE, contre **1,41 × 10⁻³** pour le meilleur LHS scindé, qui coûte davantage. Sur la somme de carrés dans la même dimension, le LHS direct atteint **7,09 × 10⁻⁵**, contre **2,19 × 10⁻³** pour le LHS sur deux composantes. La scission transforme une partie de la structure additive en interactions entre coordonnées latentes.

### 5. D’où vient le gain ?

**Des strates, de la projection ou de la symétrie.** Pour la fonction carré à 8 192 appels, la variance du LHS scindé est **1,16 × 10⁻⁴** à deux composantes et **1,89 × 10⁻⁴** à seize. La théorie prévoit une hausse du terme dominant avec k, sans garantir une progression monotone dans chaque échantillon fini. Sur la fonction linéaire, l’appariement indépendant avec remise perd l’essentiel de la réduction obtenue par la permutation bijective. Pour les fonctions impaires, le gain exact vient de la symétrie antithétique.

À partition fixe sur l’exponentielle, l’allocation pilote abaisse la RMSE de la grille 4 × 4 de **0,0147 à 0,0103**, avec 48 sondes incluses. Le même principe appliqué directement à 16 tranches de la projection abaisse la RMSE de **0,0118 à 0,00527**. Sur l’indicatrice rare, l’allocation pilote de la grille est moins précise que l’allocation uniforme dans ce contrôle. Trois sondes ne caractérisent pas toujours les cellules de faible probabilité.

## Limites et suite

Le temps dépend de l’architecture et du coût de la fonction. Les fonctions analytiques testées sont rapides, si bien que le surcoût de génération peut peser lourd. Les comparaisons à appels égaux restent pertinentes quand la fonction coûte beaucoup plus cher. Les tests de biais ne détectent aucun écart au seuil FDR de 5 %. Leur puissance dépend de la dispersion et des événements rares. Les intervalles comparatifs sont exploratoires et les quelques victoires individuelles ne prouvent pas une dominance après sélection multiple.

**Retenir la stratification directe ou projetée, le LHS pour les fonctions additives et Sobol brouillé pour les interactions testées.** La suite mathématique la plus utile consiste à comparer projection, allocation de Neyman et intégration conditionnelle quand la fonction fournit une condition exploitable. La scission i.i.d. ne constitue pas une optimisation et l’ajout systématique de composantes augmente le coût de génération.

[Résultats par graine](results/runs.csv) · [Tableau comparatif](results/summary.csv) · [Ratios et intervalles](results/comparisons.csv) · [Contrôle d’allocation](results/allocation_summary.csv) · [Protocole](results/protocol.json)

[RMSE selon les appels exacts](results/rmse_exact.png) · [RMSE selon le temps CPU](results/rmse_cpu_ms.png)
