# Audit des expériences VaR et ES

La branche `main` conserve le code et les résultats du projet VaR et ES. La campagne d’intégration est isolée dans `research/`.

- `README.md` et `rapport.md` définissent les budgets exacts, documentent les cas négatifs et distinguent la RMSE des prix de celle des estimateurs de risque. Leur principale limite vient du choix d’un problème de quantile et d’un portefeuille dont les contributions sont additives.
- `models.py` montre comment séparer la génération des facteurs latents, la transformation du modèle et le calcul de la perte. Les nouvelles fonctions tests sont définies directement en coordonnées normales pour éviter d’imposer le coût d’une transformation au MC classique.
- `methods.py` fournit des modèles pour la stratification, la proposition défensive et le compte des appels exacts. Les poids auto-normalisés et la correction par différence de quantiles sont propres à l’ancienne expérience. Pour l’espérance, le nouveau code utilise les poids bruts et corrige directement la moyenne du résidu. Le surrogate multidimensionnel travaille sur les facteurs conjoints, ce qui permet des interactions.
- `benchmark.py` fournit la structure de calibration, le tirage aléatoire de l’ordre des méthodes et le profilage mémoire à part. Le nouveau chronomètre mesure aussi le temps CPU et utilise le compteur à l’entrée de la fonction coûteuse.
- `analyze.py` fournit les agrégats de biais, variance, RMSE et intervalles de ratios entre graines. La nouvelle comparaison prend le minimum des erreurs des références à chaque tirage bootstrap, ce qui tient compte du choix du meilleur concurrent.
- `test_methods.py` vérifie les atomes des distributions, les poids, la covariance, la référence de quadrature et les budgets. Les tests d’intégration vérifient en plus la moyenne d’un surrogate par une règle indépendante.
- `results/` contient le protocole VaR et ES, les mesures par graine, les tableaux et les deux graphiques. Ces résultats restent consultables. La nouvelle campagne dispose de ses propres références et graines.
