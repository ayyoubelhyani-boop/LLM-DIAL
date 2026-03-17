# Run BANKING77 avec `sentence-transformers` et LLM local sur GPU 1

Ce document resume une nouvelle execution du pipeline `Dial-In LLM` sur `BANKING77`, avec deux ajustements par rapport au run local precedent :

- remplacement des embeddings `TF-IDF` par `sentence-transformers`
- passage de `tmax = 1` a `tmax = 2`

L'objectif etait d'ameliorer la qualite des clusters sans changer la logique generale de la methode.

## 1. Rappel de l'objectif

Le run precedent avec `TF-IDF` et `tmax = 1` validait bien l'infrastructure GPU locale, mais la qualite de clustering restait limitee :

- `45` clusters finaux
- `49` clusters `Good`
- `4213` phrases restantes

La suite logique etait donc de garder le meme benchmark et le meme backend local, mais avec une representation semantique plus forte et une boucle iterative un peu plus profonde.

## 2. Materiel et contrainte GPU

Serveur utilise :

- machine : `maje-gpu01`
- GPU 0 : `NVIDIA L40`, `46068 MB`
- GPU 1 : `NVIDIA L40`, `46068 MB`

Pour garantir que le run utilise bien le deuxieme GPU physique pour les embeddings et pour le LLM local, le processus complet a ete isole avec :

- `CUDA_VISIBLE_DEVICES=1`

Point important :

- une fois cette variable definie, le GPU physique `1` devient le seul GPU visible dans le processus ;
- a l'interieur du processus, il apparait donc comme `cuda:0`.

C'est pourquoi la commande combine :

- `CUDA_VISIBLE_DEVICES=1`
- `--local-llm-device-map cuda:0`

Ce choix permet de garder les embeddings `sentence-transformers` et le LLM local sur le meme GPU physique, sans modifier le code du projet.

## 3. Jeu de donnees utilise

Fichier d'entree :

- `data/banking77_train.csv`

Ce fichier correspond au split `train` de `BANKING77`, deja prepare dans le format attendu par le pipeline.

## 4. Commande executee

```bash
cd ~/nfs/LLM-DIAL && mkdir -p .hf-home .hf-cache out && \
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=$PWD/.hf-home \
HUGGINGFACE_HUB_CACHE=$PWD/.hf-home/hub \
TRANSFORMERS_CACHE=$PWD/.hf-cache \
HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
python3 -m dialin_llm.cli run \
  --input data/banking77_train.csv \
  --text-col text \
  --id-col sentence_id \
  --embed sentence-transformers \
  --clusterer minibatch \
  --candidate-ks 77 \
  --sample-size 10 \
  --sampler farthest \
  --epsilon 0.05 \
  --tmax 2 \
  --use-llm true \
  --llm-provider local \
  --local-llm-model mistralai/Mistral-7B-Instruct-v0.3 \
  --local-llm-device-map cuda:0 \
  --local-llm-max-new-tokens 12 \
  --local-llm-cache-dir .hf-cache \
  --cache-path out/banking77_mistral_gpu1_st_tmax2_cache.json \
  --summary-out out/banking77_mistral_gpu1_st_tmax2_summary.json \
  --out out/banking77_mistral_gpu1_st_tmax2_clusters.json \
  > out/banking77_mistral_gpu1_st_tmax2_run.log 2>&1
```

## 5. Resultats obtenus

Resume global :

- nombre de clusters finaux apres merge : `83`
- nombre de clusters juges `Good` avant merge : `85`
- nombre de phrases restantes non assignees : `2785`
- nombre d'iterations utilisees : `2`
- valeur de `K` retenue a chaque iteration : `77`

Detail par iteration :

- iteration 1 : `45 Good`, `32 Bad`, score `1.3636363636363635`, pool restant `4874`
- iteration 2 : `40 Good`, `37 Bad`, score `1.0526315789473684`, pool restant `2785`

## 6. Comparaison avec le run precedent

Comparaison directe avec le run local precedent (`TF-IDF`, `tmax = 1`, meme benchmark, meme LLM local) :

- clusters finaux : `45` -> `83`
- clusters `Good` : `49` -> `85`
- phrases restantes : `4213` -> `2785`

Interpretation :

- le nombre de clusters utiles augmente nettement ;
- le pool restant diminue fortement ;
- les groupes obtenus paraissent plus fins et plus semantiques ;
- l'amelioration vient a la fois d'une meilleure representation (`sentence-transformers`) et du fait de laisser la boucle iterative extraire plus d'intentions.

## 7. Conclusion

Ce rerun confirme que l'amelioration la plus rentable etait bien celle pressentie :

- passer a `sentence-transformers`
- autoriser au moins `2` iterations

Sans changer la methode globale, ce simple ajustement produit un resultat nettement meilleur que le run precedent avec `TF-IDF`.
