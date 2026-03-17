# Comparaison des runs BANKING77

Ce document compare les principales executions realisees jusqu'ici sur **le meme dataset** :

- `BANKING77 train`
- `9993` phrases
- `K = 77`

L'objectif est de comparer proprement :

- le mode `dummy`
- le run avec LLM local `Mistral-7B-Instruct` + `TF-IDF`
- le run avec LLM local `Mistral-7B-Instruct` + `sentence-transformers`

## 1. Runs compares

### Run A : Dummy + sentence-transformers + tmax = 1

Source : `docs/RUN_BANKING77_FR.md`

Configuration principale :

- `--embed sentence-transformers`
- `--use-llm false`
- `--tmax 1`
- `--candidate-ks 77`

### Run B : Mistral local + TF-IDF + tmax = 1

Source : `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_FR.md`

Configuration principale :

- `--embed tfidf`
- `--use-llm true`
- `--llm-provider local`
- `--local-llm-model mistralai/Mistral-7B-Instruct-v0.3`
- `--local-llm-device-map cuda:1`
- `--tmax 1`
- `--candidate-ks 77`

### Run C : Mistral local + sentence-transformers + tmax = 2

Source : run distant sur `~/nfs/LLM-DIAL`

Configuration principale :

- `--embed sentence-transformers`
- `--use-llm true`
- `--llm-provider local`
- `--local-llm-model mistralai/Mistral-7B-Instruct-v0.3`
- processus isole sur le GPU physique 1 via `CUDA_VISIBLE_DEVICES=1`
- `--local-llm-device-map cuda:0` dans le processus isole
- `--tmax 2`
- `--candidate-ks 77`

## 2. Tableau comparatif

| Run | Embeddings | Evaluateur / Nommage | Iterations | Clusters finaux | Clusters Good | Restant | Phrases assignees | Couverture |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | sentence-transformers | dummy | 1 | 62 | 66 | 1551 | 8442 | 84.48 % |
| B | TF-IDF | Mistral local | 1 | 45 | 49 | 4213 | 5780 | 57.84 % |
| C | sentence-transformers | Mistral local | 2 | 83 | 85 | 2785 | 7208 | 72.13 % |

Calcul :

- `phrases assignees = 9993 - num_remaining`
- `couverture = phrases assignees / 9993`

## 3. Scores observes

### Run A

- iteration 1 : `66 Good`, `11 Bad`, score `5.5`

### Run B

- iteration 1 : `49 Good`, `28 Bad`, score `1.6896551724137931`

### Run C

- iteration 1 : `45 Good`, `32 Bad`, score `1.3636363636363635`
- iteration 2 : `40 Good`, `37 Bad`, score `1.0526315789473684`

Remarque importante :

- les scores ne se comparent pas parfaitement entre runs lorsque `tmax` change ;
- le run C ajoute une seconde iteration, donc il faut surtout regarder l'evolution du nombre de clusters utiles et du nombre de phrases restantes.

## 4. Lecture des resultats

### Meilleure couverture brute

Le meilleur run en couverture brute est le **Run A** :

- `1551` phrases restantes seulement
- `84.48 %` de couverture

Cela montre qu'avec `sentence-transformers`, meme le mode `dummy` est deja tres efficace pour absorber une grande partie de `BANKING77`.

### Plus mauvais resultat

Le plus faible resultat est clairement le **Run B** :

- `TF-IDF`
- LLM local Mistral
- `tmax = 1`

Ce run laisse `4213` phrases non assignees et produit seulement `45` clusters finaux. Le principal facteur limitant n'est pas le LLM lui-meme, mais la representation `TF-IDF`, trop faible semantiquement pour ce benchmark.

### Meilleur run avec un vrai LLM local

Si l'on se limite aux runs utilisant un vrai LLM local, le meilleur compromis est le **Run C** :

- `83` clusters finaux
- `85` clusters `Good`
- `2785` phrases restantes
- `72.13 %` de couverture

Ce run reste moins couvrant que le mode `dummy`, mais il est nettement meilleur que le run Mistral + `TF-IDF`.

## 5. Ce que montre vraiment la comparaison

### Effet des embeddings

La comparaison entre B et C montre clairement que le changement le plus utile est le passage de `TF-IDF` a `sentence-transformers`.

Entre B et C :

- clusters finaux : `45` -> `83`
- clusters `Good` : `49` -> `85`
- phrases restantes : `4213` -> `2785`
- couverture : `57.84 %` -> `72.13 %`

Conclusion :

- la qualite des embeddings a un impact plus fort que le simple fait d'utiliser un LLM local.

### Effet du LLM local

La comparaison entre A et C est plus subtile.

Le mode `dummy` obtient une meilleure couverture brute que le LLM local :

- Run A : `84.48 %`
- Run C : `72.13 %`

Mais cela ne signifie pas automatiquement que le `dummy` est meilleur au sens qualitatif.

Le run C produit en pratique :

- plus de clusters finaux (`83` contre `62`)
- plus de labels interpretabes et plus specifiques
- une boucle plus proche de l'esprit du papier, car la coherence et le naming passent par un vrai modele generatif

Autrement dit :

- **Run A** est le meilleur en rendement brut sur cette configuration precise ;
- **Run C** est le meilleur si l'on veut une version plus fidele a l'idee "LLM-in-the-loop" du papier.

### Effet de `tmax`

Le run C montre aussi qu'autoriser une seconde iteration reste utile :

- iteration 1 : pool restant `4874`
- iteration 2 : pool restant `2785`

Donc la deuxieme iteration extrait encore un volume important d'intentions exploitables.

## 6. Qualite qualitative des labels

D'apres les runs documentes :

- le run `dummy` retrouve plusieurs themes bancaires plausibles, mais avec des labels parfois plus frustes (`card-new`, `refund-want`, `exchange-know`) ;
- le run Mistral + `TF-IDF` produit des labels interpretabes, mais souffre de gros clusters trop larges (`card-status-inquiry`) ;
- le run Mistral + `sentence-transformers` produit des labels plus fins comme `expedite-card`, `track-card-delivery`, `pending-transaction-resolve`, `auto-top-up-limit-inquiry`.

Sur ce point, le run C est qualitativement le plus convaincant.

## 7. Conclusion generale

Si l'on resume toute la comparaison :

- **meilleure couverture brute** : Run A, `dummy + sentence-transformers + tmax=1`
- **meilleur run avec LLM local** : Run C, `Mistral local + sentence-transformers + tmax=2`
- **configuration la moins performante** : Run B, `Mistral local + TF-IDF + tmax=1`

Conclusion pratique :

1. Le facteur le plus important jusqu'ici est la qualite des embeddings.
2. `TF-IDF` penalise fortement le pipeline sur `BANKING77`.
3. Le backend local Mistral devient interessant surtout lorsqu'il est combine a `sentence-transformers`.
4. Une seconde iteration ameliore nettement le resultat sans changer la methode globale.

## 8. Suite logique

Les prochaines etapes les plus pertinentes sont :

- rerun `BANKING77` avec `sentence-transformers` et `tmax = 3`
- tester plusieurs valeurs de `K` autour de `77`
- ajouter une evaluation quantitative avec une metrique comme `NMI`
- eventuellement comparer `dummy` et `Mistral` a configuration strictement identique (`sentence-transformers`, meme `tmax`, meme `K`) pour isoler uniquement l'effet de l'evaluateur
