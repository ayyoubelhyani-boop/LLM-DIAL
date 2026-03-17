# Synthese Resultats BANKING77

Ce document presente de facon concise les resultats les plus importants du projet sur le benchmark `BANKING77`.

## 1. Objectif

Le but etait de comparer plusieurs variantes du pipeline `Dial-In LLM` sur **le meme dataset** afin d'identifier la methode la plus interessante.

Dataset utilise :

- `BANKING77 train`
- `9993` phrases

Parametres communs :

- `K = 77`
- `MiniBatchKMeans`
- `sample-size = 10`
- `sampler = farthest`

## 2. Variantes comparees

Nous avons compare 8 configurations en faisant varier :

- l'evaluateur : `dummy` ou `Mistral local`
- les embeddings : `TF-IDF` ou `sentence-transformers`
- la profondeur iterative : `tmax = 1` ou `tmax = 2`

## 3. Resultats principaux

| Evaluateur | Embeddings | tmax | Clusters finaux | Restant | Couverture | NMI assigne | ARI assigne | NMI global | ARI global |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dummy | TF-IDF | 1 | 54 | 3932 | 60.65 % | 0.560 | 0.202 | 0.396 | 0.025 |
| dummy | TF-IDF | 2 | 99 | 2372 | 76.26 % | 0.559 | 0.190 | 0.458 | 0.053 |
| dummy | sentence-transformers | 1 | 62 | 1551 | 84.48 % | 0.753 | 0.456 | 0.708 | 0.260 |
| dummy | sentence-transformers | 2 | 125 | 219 | 97.81 % | 0.757 | 0.438 | 0.751 | 0.429 |
| Mistral local | TF-IDF | 1 | 45 | 4213 | 57.84 % | 0.416 | 0.036 | 0.298 | 0.017 |
| Mistral local | TF-IDF | 2 | 83 | 2775 | 72.23 % | 0.465 | 0.046 | 0.375 | 0.027 |
| Mistral local | sentence-transformers | 1 | 44 | 4874 | 51.23 % | 0.791 | 0.521 | 0.543 | 0.040 |
| Mistral local | sentence-transformers | 2 | 83 | 2785 | 72.13 % | 0.803 | 0.525 | 0.675 | 0.116 |

Lecture :

- `restant` = nombre de phrases non assignees
- `couverture` = proportion de phrases regroupees dans des clusters conserves
- `NMI assigne` et `ARI assigne` = qualite des clusters sur les phrases effectivement assignees
- `NMI global` et `ARI global` = qualite sur tout le dataset, en comptant `__unassigned__` comme categorie speciale

## 4. Ce qu'il faut retenir

### 1. La meilleure couverture brute

La meilleure configuration est :

- `dummy + sentence-transformers + tmax = 2`

Resultat :

- `125` clusters finaux
- `219` phrases restantes
- `97.81 %` de couverture

Autrement dit, c'est la configuration qui couvre le mieux le dataset.

### 2. La meilleure qualite de clustering

Si l'on compare les clusters produits aux labels reels de `BANKING77`, la meilleure qualite de clustering sur les phrases assignees est obtenue avec :

- `Mistral local + sentence-transformers + tmax = 2`

Resultat :

- `NMI assigne = 0.803`
- `ARI assigne = 0.525`

Autrement dit, c'est la configuration qui produit les groupes les plus proches des intents reels parmi les phrases qu'elle accepte.

### 3. Le meilleur compromis avec un vrai LLM

Si l'on veut rester plus proche de l'esprit du papier, le meilleur compromis est :

- `Mistral local + sentence-transformers + tmax = 2`

Resultat :

- `83` clusters finaux
- `2785` phrases restantes
- `72.13 %` de couverture

Cette configuration est moins couvrante que le `dummy`, mais elle est plus fidele a une logique `LLM-in-the-loop`.

### 4. Le facteur le plus important

Le changement le plus utile est :

- passer de `tmax = 1` a `tmax = 2`

Dans tous les cas, cela ameliore les resultats.

Le second levier majeur est :

- remplacer `TF-IDF` par `sentence-transformers`

## 5. Interpretation scientifique

Les experiments montrent que :

- `TF-IDF` est trop limite pour un benchmark comme `BANKING77`,
- `sentence-transformers` produit des regroupements plus solides,
- le mode `dummy` est tres performant en couverture brute,
- le mode `Mistral local` est plus selectif, donc plus proche d'une validation semantique explicite,
- l'evaluation externe montre que la meilleure methode depend du critere retenu.

Il faut donc distinguer deux objectifs :

- **objectif performance brute** : choisir `dummy + sentence-transformers + tmax = 2`
- **objectif fidelite au papier** : choisir `Mistral local + sentence-transformers + tmax = 2`
- **objectif qualite de clustering mesuree par les labels reels** : choisir `Mistral local + sentence-transformers + tmax = 2`

## 6. Conclusion

La conclusion generale est simple :

1. La methode la plus efficace sur `BANKING77` dans notre implementation est `dummy + sentence-transformers + tmax = 2`.
2. La methode la plus interessante pour une version plus proche du papier est `Mistral local + sentence-transformers + tmax = 2`.
3. La meilleure qualite de clustering mesuree par `NMI` et `ARI` est aussi obtenue par `Mistral local + sentence-transformers + tmax = 2`.
4. `tmax = 2` doit devenir la configuration minimale recommandee pour comparer les variantes du pipeline.

## 7. Limite actuelle

Cette comparaison repose maintenant sur deux niveaux :

- des metriques internes : clusters conserves, couverture, phrases restantes
- des metriques externes : `NMI`, `ARI`, `V-measure`

La principale limite restante est donc plutot :

- l'absence d'une exploration plus large de `K`,
- l'absence d'un test a `tmax = 3`,
- l'absence d'une comparaison equivalente sur d'autres benchmarks comme `MASSIVE`.

## 8. Documents techniques associes

Pour plus de detail :

- `docs/COMPARAISON_BANKING77_TOUTES_COMBINAISONS_FR.md`
- `docs/RUN_BANKING77_FR.md`
- `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_FR.md`
- `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_ST_FR.md`
