# Execution sur MASSIVE

Ce document decrit l'execution du pipeline sur `MASSIVE` en configuration `en-US`, egalement cite dans l'article.

## 1. Objectif

Executer le pipeline sur un second benchmark public du papier afin d'obtenir une comparaison avec un jeu plus large et plus heterogene que `BANKING77`.

## 2. Changements effectues

Les changements utiles pour ce run ont ete les suivants :

- ajout d'un fallback pour `MASSIVE` via les fichiers parquet publics de Hugging Face, car le chargement par script de dataset echouait dans l'environnement local,
- export du split `train` de `MASSIVE en-US` vers `data/massive_en_us_train.csv`,
- utilisation du projet deja present sur le serveur dans `~/nfs/LLM-DIAL`,
- chargement du fichier `data/massive_en_us_train.csv` dans ce projet serveur,
- execution avec `sentence-transformers` et `MiniBatchKMeans`,
- run borne avec `tmax = 1` et `K = 60`, en coherence avec le nombre d'intentions du benchmark.

## 3. Environnement distant

Le run a ete execute sur le meme serveur JupyterHub distant :

- `http://maje-gpu01.biomedicale.univ-paris5.fr`

Configuration observee :

- `Python 3.12.6`
- environ `1.0 TiB` de RAM
- `2 x NVIDIA L40`
- `torch 2.10.0+cu128`
- `sentence-transformers 5.3.0`

## 4. Commande executee

Depuis `~/nfs/LLM-DIAL`, la commande suivante a ete lancee :

```bash
python3 -m dialin_llm.cli run \
  --input data/massive_en_us_train.csv \
  --text-col text \
  --id-col sentence_id \
  --embed sentence-transformers \
  --clusterer minibatch \
  --candidate-ks 60 \
  --sample-size 10 \
  --sampler farthest \
  --epsilon 0.05 \
  --tmax 1 \
  --use-llm false \
  --summary-out out/massive_en_us_summary.json \
  --out out/massive_en_us_clusters.json
```

## 5. Resultats obtenus

Resume principal :

- `num_clusters` : `29`
- `num_good_clusters` : `30`
- `num_remaining` : `6024`
- `iterations_used` : `1`
- `selected_k` : `60`
- `good_clusters` : `30`
- `bad_clusters` : `30`
- `score` : `0.967741935483871`

Quelques labels produits :

- `meeting-today` : taille `267`
- `turn-plug` : taille `97`
- `calendar-event` : taille `301`
- `news-tell` : taille `317`
- `tweet-send` : taille `181`
- `alarm-set` : taille `288`
- `raining-rain` : taille `181`
- `lights-dim` : taille `329`
- `play-want` : taille `156`
- `delete-list` : taille `145`

Une copie locale du resume distant a ete conservee dans :

- `out/massive_en_us_summary.remote.json`

## 6. Interpretation

`MASSIVE` est plus difficile pour cette implementation que `BANKING77`.

Points positifs :

- le pipeline tourne correctement sur l'ensemble `train`,
- plusieurs labels produits sont plausibles et correspondent a des intents assistant vocal,
- l'utilisation des embeddings `sentence-transformers` reste stable a cette echelle.

Limites observees :

- seulement `29` clusters finaux sont conserves alors que `K = 60`,
- la moitie des clusters candidats est jugee `Bad`,
- `6024` utterances restent non assignees apres la premiere iteration,
- le benchmark est plus varie semantiquement, ce qui rend le mode `dummy` beaucoup moins fiable pour juger la coherence.

## 7. Conclusion

Cette execution montre que `MASSIVE` constitue un test plus exigeant et met mieux en evidence les limites de l'implementation actuelle.

Le pipeline reste runnable, mais les resultats suggerent clairement que pour ce benchmark il faudrait :

- un evaluateur de coherence plus robuste qu'un `dummy`,
- plusieurs iterations au lieu d'une seule,
- une exploration de plusieurs valeurs de `K`,
- une evaluation quantitative par rapport aux intents connus.
