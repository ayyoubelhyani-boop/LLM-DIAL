# Synthese Resultats BANKING77 - Mistral local + sentence-transformers, sweep `tmax = 1..6`

## 1. Objectif

Ce document presente une synthese courte et presentable de la campagne `BANKING77` executee avec un LLM local `Mistral-7B-Instruct`, des embeddings `sentence-transformers`, et un sweep de `tmax` allant de `1` a `6`.

Le but etait de repondre a une question simple : jusqu ou est-il utile d augmenter `tmax` dans cette configuration locale ?

## 2. Configuration comparee

- dataset : `BANKING77 train` (`9993` phrases)
- evaluateur : `Mistral local`
- embeddings : `sentence-transformers`
- clusterer : `MiniBatchKMeans`
- `K = 77` fixe
- `sample-size = 10`
- `sampler = farthest`
- `epsilon = 0.0`
- seul parametre variable : `tmax`

## 3. Resultats principaux

| tmax | Clusters finaux | Restant | Couverture | ARI assigne | ARI global |
|---|---:|---:|---:|---:|---:|
| 1 | 44 | 4874 | 0.5123 | 0.5213 | 0.0396 |
| 2 | 83 | 2785 | 0.7213 | 0.5250 | 0.1160 |
| 3 | 121 | 1436 | 0.8563 | 0.5040 | 0.2670 |
| 4 | 154 | 483 | 0.9517 | 0.4758 | 0.4219 |
| 5 | 189 | 215 | 0.9785 | 0.4636 | 0.4465 |
| 6 | 207 | 143 | 0.9857 | 0.4609 | 0.4505 |

## 4. Ce qu il faut retenir

### 1. La meilleure couverture

La meilleure couverture est obtenue avec `tmax = 6` :

- couverture : `0.9857`
- phrases restantes : `143`
- ARI global : `0.4505`

### 2. La meilleure qualite de clustering sur les phrases assignees

Le meilleur `ARI assigne` est obtenu avec `tmax = 2` :

- ARI assigne : `0.5250`
- NMI assigne : `0.8033`
- couverture : `0.7213`

### 3. Le meilleur compromis pratique

Le meilleur compromis me parait etre `tmax = 4` :

- couverture deja tres haute : `0.9517`
- phrases restantes : `483`
- ARI global : `0.4219`
- le gain de couverture existe encore ensuite, mais les metriques assignees baissent progressivement

## 5. Interpretation

Le sweep fait apparaitre trois phases assez nettes :

- `tmax = 1 -> 2` : gros gain de couverture avec une qualite assignee encore excellente ;
- `tmax = 2 -> 4` : la couverture continue a monter fortement et le score global s ameliore beaucoup ;
- `tmax = 4 -> 6` : on gagne encore en couverture, mais surtout sur des phrases plus difficiles, avec une erosion graduelle de l `ARI assigne`.

Autrement dit :

- si l objectif est la purete des clusters acceptes, `tmax = 2` reste le meilleur point ;
- si l objectif est de couvrir quasiment tout le dataset, `tmax = 6` est le meilleur ;
- si l objectif est un bon compromis benchmark / qualite / couverture, `tmax = 4` est la recommandation la plus raisonnable.

## 6. Recommandation finale

1. Pour les futurs runs comparatifs, garder `tmax = 4` comme default de travail.
2. Utiliser `tmax = 2` quand on veut privilegier la qualite des clusters assignees.
3. Utiliser `tmax = 6` quand la priorite est de minimiser le pool non assigne.

## 7. Documents associes

- note de run complete : `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_ST_TMAX_1_TO_6_FR.md`
- tableau agrege : `out/banking77_mistral_st_tmax1_6_results.json`
- tableau CSV : `out/banking77_mistral_st_tmax1_6_results.csv`
- rapport d environnement : `out/banking77_mistral_st_tmax1_6_env.txt`

## 8. Limite de lecture

Les temps de run du sweep ne doivent pas etre lus comme un benchmark temps pur, car un cache LLM partage a ete utilise pour eviter de recalculer les prompts identiques entre runs.

