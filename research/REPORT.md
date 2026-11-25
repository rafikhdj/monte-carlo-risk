# Estimer une espérance avec un budget de calcul exact

## Question scientifique

Peut-on réduire le nombre d’appels à une fonction coûteuse (f) pour estimer \(I=\int f(x)p(x)\,dx\), tout en améliorant la précision par rapport à une référence adaptée au problème ? Cette campagne sépare le nombre d’appels, la précision et le temps CPU. Elle comprend douze problèmes en dimensions 1, 2, 5, 10 et 20.

## Méthodes et hypothèses

Les références opérationnelles sont le MC i.i.d., l’hypercube latin (stratification de chaque marginale), la méthode antithétique, les poids d’importance, le contrôle polynomial à moyenne connue, Sobol brouillé, Gauss–Legendre, Gauss–Hermite et Gauss–Kronrod adaptatif en dimension 1. Gauss–Legendre scinde l’intervalle aux ruptures connues de la fonction lorsqu’elles existent. Les quadratures à deux facteurs utilisent la plus grande grille tensorielle carrée permise par le budget. La scission génère \(Z=(Z_1+Z_2)/\sqrt 2\) avec deux gaussiennes indépendantes. Par stabilité gaussienne, elle produit la même loi qu’un tirage direct. Sans conditionnement supplémentaire, sa variance théorique est identique.

Les approximations unidimensionnelles utilisent des splines cubiques sur des grilles uniformes ou de quantiles, PCHIP, un polynôme de Chebyshev de degré au plus 24 et un raffinement adaptatif par courbure pondérée. Les approximations sont constantes au-delà de ±4,5 écarts-types. Le surrogate multidimensionnel combine des fonctions radiales gaussiennes et des termes polynomiaux centrés de moyenne connue. En dimension 2, le polynôme inclut des produits croisés jusqu’au degré total 5. En dimensions supérieures, le modèle utilise des termes linéaires et quadratiques par facteur, plus les fonctions radiales conjointes. La moyenne des splines est intégrée par morceaux sous la densité normale. Celle du modèle de régression est son intercept.

Pour un modèle \(g\) ajusté sur des appels indépendants de la correction, l’estimateur est

$$
\widehat I=\mu_g+\frac{1}{m}\sum_{i=1}^m w(X_i)\,[f(X_i)-g(X_i)],
\qquad w=p/q,\quad \mu_g=\mathbb E_p[g].
$$

Sous couverture de \(p\) par \(q\), intégrabilité et moyenne du surrogate exacte, l’espérance conditionnelle de cet estimateur vaut \(I\). Si l’intégration du surrogate comporte une erreur \(\delta\), le biais conditionnel vaut \(\delta\). Avec des corrections i.i.d., la variance conditionnelle est \(\operatorname{Var}_q[w(f-g)]/m\), sous existence du second moment. Avec des points stratifiés ou Sobol brouillés, les points sont dépendants et les covariances changent cette formule. L’incertitude de ces méthodes est mesurée entre brouillages ou réplications. Les poids IS sont bruts, calculés en log et bornés par 5 car \(q\) contient 20 % de \(p\). Le modèle sans correction peut rester biaisé lorsque la fonction possède une queue mal approchée.

La réduction de variance dépend du résidu dans la loi utilisée pour la correction. Une amélioration de l’erreur locale près des nœuds peut augmenter la variance ailleurs. Le choix de la fraction d’entraînement doit aussi tenir compte du nombre d’appels qu’il reste pour la correction. Les expériences de contrôle fonctionnel et de Monte Carlo multifidélité étudient cette décomposition et les allocations entre modèles. [Oates, Girolami et Chopin](https://onlinelibrary.wiley.com/doi/10.1111/rssb.12185/abstract), [Peherstorfer, Willcox et Gunzburger](https://doi.org/10.1137/15M1046472)

## Protocole

- Trois budgets plafonds de 128, 512 et 2 048 appels exacts, et 32 graines d’évaluation. Toute évaluation pendant la construction, le raffinement, le contrôle local ou la correction passe par le même compteur. La calibration hors ligne est facturée séparément dans le protocole. Les résultats en ligne s’interprètent comme ceux d’une méthode déjà calibrée sur le problème.
- La calibration compare des fractions de construction de 1/8, 1/4 et 1/2, sur 12 graines indépendantes. Elle minimise la variance de l’estimateur complet et n’accède pas à l’intégrale de référence. La version « allocation » réserve en plus 1/8 du budget à un contrôle du résidu. Une corrélation absolue supérieure à 0,45 avec le premier facteur déclenche la correction stratifiée. La correction est tirée après les sondes et ne les réutilise pas. Sobol emploie la moitié du budget pour sa correction afin de garder une puissance de deux.
- Les fonctions couvrent un polynôme, une exponentielle, une oscillation rapide, un pic local étroit, une rupture, un événement rare, deux options et quatre dimensions d’un modèle à interactions. L’option asiatique utilise quatre dates, deux chocs extérieurs et une quadrature conditionnelle de 64 nœuds avec intégration analytique du dernier choc. Cette charge provient du calcul du payoff conditionnel. Les autres fonctions ont des références analytiques.
- Les temps comprennent les tirages, les appels exacts, les ajustements et les estimations. La mémoire est le pic d’allocations suivi sur une exécution séparée par configuration. Les intervalles bootstrap exploratoires utilisent 3 000 rééchantillonnages appariés par indice de graine. Le meilleur concurrent est recherché dans chaque rééchantillonnage. Les comparaisons CPU acceptent les références prenant au plus 120 % du temps du candidat. La campagne principale s’arrête aux trois budgets fixés. L’expérience sur les pics est un contrôle de transfert distinct.
- Pour retenir un avantage avec moins d’appels, on exige une RMSE inférieure à celle de la meilleure référence admissible jusqu’au budget suivant et une borne haute du ratio inférieure à 1. Les comparaisons couvrent une grille finie de budgets et ne corrigent pas pour la multiplicité des hypothèses. Les valeurs d’arrondi sous \(10^{-15}\) sont traitées comme un plancher dans les ratios et le tracé.

## Résultats

**Aucune variante n’améliore de façon reproductible la RMSE avec moins d’appels que la meilleure référence pertinente sur la grille testée.** Plusieurs variantes battent le MC classique, et le pic local fournit un gain à budget égal. La supériorité dépend cependant du budget et du choix de la référence.

- **Pic local étroit.** À 128 appels, la spline seule obtient une erreur de **3,08 × 10⁻⁹** en **0,144 ms**, contre **1,05 × 10⁻³** en **0,311 ms** pour Gauss–Legendre et **8,89 × 10⁻³** pour le MC. L’adaptatif corrigé obtient **8,98 × 10⁻⁷** à ce budget, en **0,532 ms**. À 399 appels, Gauss–Kronrod rejoint la référence à la précision d’arrondi.
- **Vérification du transfert.** Après cette observation, un lot exploratoire de neuf pics croise trois centres inédits et trois largeurs fixées avant son exécution. La spline à 64 appels produit des erreurs allant de **6,67 × 10⁻⁹ à 5,34 × 10⁻³**. Pour chacun des neuf pics, une quadrature candidate plafonnée à 256 appels obtient une erreur plus faible. La sensibilité de la spline à la position du pic exclut une promesse générale.
- **Asiatique avec intégration interne.** À 128 appels, la régression seule réduit la RMSE du MC de **1,36 × 10⁻²** à **3,88 × 10⁻⁴**. Sobol seul donne **1,71 × 10⁻³**. Gauss–Hermite obtient **7,44 × 10⁻⁷** avec **121 appels** en **0,218 ms**, contre **0,537 ms** pour le surrogate. La référence est **0,0936166881104**. Les changements d’ordre et de troncature extérieure la modifient d’environ **1,7 × 10⁻¹⁴** dans ce test. Doubler l’ordre de l’intégration interne la modifie de moins de **10⁻¹⁶**. Ces écarts de raffinement sont des diagnostics numériques.
- **Événement rare.** À 2 048 appels, l’IS seul réduit la RMSE du MC de **3,45 × 10⁻⁴** à **1,29 × 10⁻⁵**. Le surrogate corrigé par IS remonte à **3,93 × 10⁻⁵**. L’IS reçoit une direction de queue tirée de la géométrie de l’événement. Le surrogate générique la représente mal.
- **Interactions à 20 dimensions.** À 2 048 appels, les RMSE sont **3,10 × 10⁻³** pour le MC, **4,38 × 10⁻⁴** pour Sobol et **5,28 × 10⁻⁴** pour le surrogate seul. Sobol est aussi moins coûteux que ce surrogate, avec **2,25 ms et 1,25 Mio** contre **2,96 ms et 2,86 Mio**. La correction Sobol n’aide pas ici. Elle divise le budget entre ajustement et échantillon résiduel, dont les interactions subsistent.
- **Oscillation et rupture.** En oscillation, la spline seule atteint un plateau proche de **4,78 × 10⁻⁶**. Il est compatible avec l’erreur de prolongement constant et les résidus en queue, rarement visités. La correction est sans biais en théorie pour une moyenne intégrée exactement, mais peut sembler biaisée sur 32 graines si les corrections de queue sont rares. Sur la rupture connue, scinder la quadrature autour du saut évite le lissage et domine. Sur le polynôme, l’exponentielle et l’option européenne, les quadratures bien adaptées restent les comparateurs à battre.

## Limites et suite

Le régime coûteux testé repose sur un intégrand conditionnel lisse à deux facteurs. Une option asiatique avec plus de facteurs extérieurs ou une simulation physique imbriquée coûterait davantage et réduirait l’applicabilité de la quadrature tensorielle. La loi normale facilite l’intégration du surrogate et la construction des propositions. Les résultats ne valident pas une extension directe à une densité inconnue, à des queues lourdes ou à des interactions d’ordre élevé. Les coûts CPU sont des mesures locales. Le suivi mémoire ne garantit pas la capture de toutes les allocations natives. Les répétitions d’une règle déterministe mesurent la stabilité du chronométrage et répètent la même erreur de quadrature.

**Conserver le contrôle par surrogate comme piste ciblée.** Les bons candidats réunissent une fonction coûteuse, une dimension effective faible à modérée, un surrogate dont la moyenne se calcule à faible coût et un résidu peu variable dans les régions importantes. La priorité est une expérience avec suffisamment d’interactions pour exclure la quadrature tensorielle, une référence numérique indépendante et un budget déterminé par la précision visée. Comparer l’allocation par estimation de la variance du résidu et du coût marginal à un Sobol direct correctement brouillé. Écarter la scission gaussienne simple, les grilles qui ignorent les ruptures et le surrogate global sans correction lorsque le biais dépasse la tolérance.

## Reproduire

Depuis la racine du dépôt, dans l’environnement Python 3.14 créé avec `requirements.txt` :

```bash
python -m unittest discover -s research/tests -v
python -m research.benchmarks.run
python -m research.benchmarks.analyze
python -m research.benchmarks.peaks
```

`python -m research.benchmarks.run --smoke --output /tmp/research-smoke` exécute un contrôle court dans un répertoire séparé.

[Audit](AUDIT.md) · [Protocole et versions](results/protocol.json) · [Résultats par graine](results/runs.csv) · [Tableau complet](results/summary.csv) · [Ratios et intervalles](results/comparisons.csv) · [Transfert des pics](results/peak_transfer.csv)

[RMSE selon les appels exacts](results/rmse_exact.png) · [RMSE selon le temps CPU](results/rmse_cpu_ms.png)
