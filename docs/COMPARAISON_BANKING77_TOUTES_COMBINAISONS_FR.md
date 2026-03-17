# Comparaison complete des runs BANKING77

Ce document compare toutes les combinaisons executees sur le meme dataset :

- benchmark : `BANKING77 train`
- taille : `9993` phrases
- clusterer : `MiniBatchKMeans`
- `K = 77`
- `sample-size = 10`
- `sampler = farthest`
- `epsilon = 0.05`

Les dimensions comparees sont :

- evaluateur / nommage : `dummy` ou `Mistral local`
- embeddings : `TF-IDF` ou `sentence-transformers`
- profondeur iterative : `tmax = 1` ou `tmax = 2`

Cela donne `8` runs au total.

## 1. Tableau comparatif global

| Run | Evaluateur | Embeddings | tmax | Clusters finaux | Couverture | NMI assigne | ARI assigne | V assigne | NMI global | ARI global | V global |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | dummy | TF-IDF | 1 | 54 | 60.65 % | 0.560 | 0.202 | 0.560 | 0.396 | 0.025 | 0.396 |
| B | dummy | TF-IDF | 2 | 99 | 76.26 % | 0.559 | 0.190 | 0.559 | 0.458 | 0.053 | 0.458 |
| C | dummy | sentence-transformers | 1 | 62 | 84.48 % | 0.753 | 0.456 | 0.753 | 0.708 | 0.260 | 0.708 |
| D | dummy | sentence-transformers | 2 | 125 | 97.81 % | 0.757 | 0.438 | 0.757 | 0.751 | 0.429 | 0.751 |
| E | Mistral local | TF-IDF | 1 | 45 | 57.84 % | 0.416 | 0.036 | 0.416 | 0.298 | 0.017 | 0.298 |
| F | Mistral local | TF-IDF | 2 | 83 | 72.23 % | 0.465 | 0.046 | 0.465 | 0.375 | 0.027 | 0.375 |
| G | Mistral local | sentence-transformers | 1 | 44 | 51.23 % | 0.791 | 0.521 | 0.791 | 0.543 | 0.040 | 0.543 |
| H | Mistral local | sentence-transformers | 2 | 83 | 72.13 % | 0.803 | 0.525 | 0.803 | 0.675 | 0.116 | 0.675 |

Calcul :

- `assignees = 9993 - num_remaining`
- `couverture = assignees / 9993`
- `assigne` = metriques calculees seulement sur les phrases placees dans un cluster
- `global` = metriques calculees sur tout le dataset, en traitant `__unassigned__` comme categorie speciale

## 2. Conclusions principales

### Meilleure couverture globale

Le meilleur run en couverture brute est :

- `dummy + sentence-transformers + tmax = 2`

Avec :

- `125` clusters finaux
- `133` clusters `Good`
- `219` phrases restantes
- `97.81 %` de couverture

En plus, c'est aussi le meilleur run en metriques globales :

- `NMI global = 0.751`
- `ARI global = 0.429`
- `V global = 0.751`

### Meilleur run avec un vrai LLM local

Le meilleur compromis cote LLM local est :

- `Mistral local + sentence-transformers + tmax = 2`

Avec :

- `83` clusters finaux
- `85` clusters `Good`
- `2785` phrases restantes
- `72.13 %` de couverture

Et surtout :

- `NMI assigne = 0.803`
- `ARI assigne = 0.525`
- `V assigne = 0.803`

Ce sont les meilleures metriques de qualite de clustering sur les phrases effectivement assignees.

### Plus mauvais run

Le plus faible en couverture est :

- `Mistral local + sentence-transformers + tmax = 1`

Avec :

- `44` clusters finaux
- `4874` phrases restantes
- `51.23 %` de couverture

Ce point est instructif : cette configuration est faible en couverture, mais deja tres forte en qualite des clusters acceptes :

- `NMI assigne = 0.791`
- `ARI assigne = 0.521`

## 3. Ce que montre la comparaison

### Effet de `tmax`

Le passage de `tmax = 1` a `tmax = 2` ameliore tous les couples compares.

Exemples :

- `dummy + TF-IDF` : `60.65 %` -> `76.26 %`
- `dummy + sentence-transformers` : `84.48 %` -> `97.81 %`
- `Mistral + TF-IDF` : `57.84 %` -> `72.23 %`
- `Mistral + sentence-transformers` : `51.23 %` -> `72.13 %`

Conclusion :

- `tmax = 2` est clairement preferable a `tmax = 1` sur `BANKING77`.

### Effet des embeddings

Avec l'evaluateur `dummy`, `sentence-transformers` domine nettement `TF-IDF`.

Exemples :

- `dummy`, `tmax = 1` : `60.65 %` -> `84.48 %`
- `dummy`, `tmax = 2` : `76.26 %` -> `97.81 %`

Avec `Mistral local`, le gain se voit moins sur la couverture brute, mais il apparait tres clairement sur les metriques externes.

Exemple :

- `Mistral + TF-IDF + tmax = 2` : `NMI assigne = 0.465`, `ARI assigne = 0.046`
- `Mistral + sentence-transformers + tmax = 2` : `NMI assigne = 0.803`, `ARI assigne = 0.525`

Donc :

- pour `Mistral`, `sentence-transformers` ameliore massivement la qualite semantique des clusters.

### Effet du LLM local

A configuration equivalente, le `dummy` reste plus couvrant que `Mistral local` dans cette implementation.

Cela suggere surtout que :

- le `dummy` est plus permissif ;
- `Mistral` est plus selectif ;
- la couverture brute seule ne suffit donc pas a dire quelle methode est semantiquement la meilleure.

Les metriques externes confirment justement cette nuance :

- `dummy + sentence-transformers + tmax = 2` est meilleur sur le dataset complet
- `Mistral + sentence-transformers + tmax = 2` est meilleur sur la qualite des clusters acceptes

## 4. Classement synthétique

### Classement par couverture

1. `dummy + sentence-transformers + tmax = 2` -> `97.81 %`
2. `dummy + sentence-transformers + tmax = 1` -> `84.48 %`
3. `dummy + TF-IDF + tmax = 2` -> `76.26 %`
4. `Mistral local + TF-IDF + tmax = 2` -> `72.23 %`
5. `Mistral local + sentence-transformers + tmax = 2` -> `72.13 %`
6. `dummy + TF-IDF + tmax = 1` -> `60.65 %`
7. `Mistral local + TF-IDF + tmax = 1` -> `57.84 %`
8. `Mistral local + sentence-transformers + tmax = 1` -> `51.23 %`

### Classement par qualite de clustering sur les phrases assignees

1. `Mistral local + sentence-transformers + tmax = 2` -> `NMI 0.803`, `ARI 0.525`
2. `Mistral local + sentence-transformers + tmax = 1` -> `NMI 0.791`, `ARI 0.521`
3. `dummy + sentence-transformers + tmax = 2` -> `NMI 0.757`, `ARI 0.438`
4. `dummy + sentence-transformers + tmax = 1` -> `NMI 0.753`, `ARI 0.456`
5. `dummy + TF-IDF + tmax = 1` -> `NMI 0.560`, `ARI 0.202`
6. `dummy + TF-IDF + tmax = 2` -> `NMI 0.559`, `ARI 0.190`
7. `Mistral local + TF-IDF + tmax = 2` -> `NMI 0.465`, `ARI 0.046`
8. `Mistral local + TF-IDF + tmax = 1` -> `NMI 0.416`, `ARI 0.036`

### Classement pour une version plus fidele au papier

1. `Mistral local + sentence-transformers + tmax = 2`
2. `Mistral local + TF-IDF + tmax = 2`
3. `Mistral local + TF-IDF + tmax = 1`
4. `Mistral local + sentence-transformers + tmax = 1`

## 5. Conclusion

A ce stade, la situation est claire :

- pour la performance brute sur `BANKING77`, le `dummy` reste la meilleure option dans cette implementation ;
- pour une version plus proche de l'idee du papier, le meilleur compromis est `Mistral local + sentence-transformers + tmax = 2` ;
- pour la qualite de clustering mesuree contre les labels reels, `Mistral local + sentence-transformers + tmax = 2` est maintenant la meilleure configuration ;
- `TF-IDF` devient rapidement limitant ;
- `tmax = 2` devrait devenir la valeur minimale recommandee pour les runs de comparaison sur ce benchmark.
