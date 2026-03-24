# Run BANKING77 avec Mistral local + sentence-transformers, sweep `tmax = 1..6`

## 1. Objectif

Ce document decrit une campagne complete sur `BANKING77` executee depuis `/nfs/LLM-DIAL`, en gardant la meme configuration locale `Mistral + sentence-transformers` et en faisant varier uniquement `tmax` de `1` a `6`.

Le but etait de mesurer l effet d une boucle iterative plus profonde sur :

- la couverture du dataset,
- la qualite des clusters sur les phrases assignees,
- la qualite globale en tenant compte des phrases non assignees,
- le cout pratique en temps de calcul.

## 2. Contexte d execution

- machine : `maje-gpu01`
- repertoire de travail : `/nfs/LLM-DIAL`
- utilisateur Jupyter : `jupyter-venus_02`
- GPU detectes : `2 x NVIDIA L40 46068 MiB`
- GPU reserve : le deuxieme GPU physique via `CUDA_VISIBLE_DEVICES=1`
- device map interne du LLM : `cuda:0` car le processus ne voit qu un seul GPU
- version Python : `3.12.6`
- `accelerate` absent, `bitsandbytes` absent, donc chargement local non quantifie
- etat du depot : arbre deja sale avant la campagne, conserve tel quel
- rapport d environnement complet : `out/banking77_mistral_st_tmax1_6_env.txt`

## 3. Configuration commune

- dataset : `data/banking77_train.csv`
- embeddings : `sentence-transformers`
- clusterer : `minibatch`
- `candidate-ks = 77`
- `sample-size = 10`
- `sampler = farthest`
- `epsilon = 0.0`
- evaluateur et namer : `local`
- modele local : `mistralai/Mistral-7B-Instruct-v0.3`
- generation locale : `max_new_tokens = 12`
- cache Hugging Face dans le projet : `.hf-home`, `.hf-cache`
- cache LLM partage entre runs : `out/banking77_mistral_st_tmax1_6_shared_cache.json`

Remarque importante : un cache partage a ete utilise pour reutiliser exactement les memes reponses LLM quand les prompts etaient identiques entre runs. Les metriques de clustering restent comparables, mais les temps de calcul des runs tardifs sont avantagees par cette memoisation.

## 4. Commande de base

```bash
cd /nfs/LLM-DIAL
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=$PWD/.hf-home
export HUGGINGFACE_HUB_CACHE=$PWD/.hf-home/hub
export TRANSFORMERS_CACHE=$PWD/.hf-cache
python3 -m dialin_llm.cli run \
  --input data/banking77_train.csv \
  --text-col text \
  --id-col sentence_id \
  --embed sentence-transformers \
  --clusterer minibatch \
  --candidate-ks 77 \
  --sample-size 10 \
  --sampler farthest \
  --epsilon 0.0 \
  --tmax <VALUE> \
  --use-llm true \
  --llm-provider local \
  --local-llm-model mistralai/Mistral-7B-Instruct-v0.3 \
  --local-llm-device-map cuda:0 \
  --local-llm-max-new-tokens 12 \
  --local-llm-cache-dir .hf-cache
```

## 5. Resultats globaux

| tmax | iterations | clusters finaux | good clusters | restant | couverture | NMI assigne | ARI assigne | V assigne | NMI global | ARI global | V global | run s | eval s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 44 | 45 | 4874 | 0.5123 | 0.7909 | 0.5213 | 0.7909 | 0.5428 | 0.0396 | 0.5428 | 28 | 1 |
| 2 | 2 | 83 | 85 | 2785 | 0.7213 | 0.8033 | 0.5250 | 0.8033 | 0.6748 | 0.1160 | 0.6748 | 27 | 1 |
| 3 | 3 | 121 | 128 | 1436 | 0.8563 | 0.7946 | 0.5040 | 0.7946 | 0.7345 | 0.2670 | 0.7345 | 29 | 1 |
| 4 | 4 | 154 | 179 | 483 | 0.9517 | 0.7831 | 0.4758 | 0.7831 | 0.7610 | 0.4219 | 0.7610 | 30 | 1 |
| 5 | 5 | 189 | 222 | 215 | 0.9785 | 0.7773 | 0.4636 | 0.7773 | 0.7662 | 0.4465 | 0.7662 | 29 | 1 |
| 6 | 6 | 207 | 241 | 143 | 0.9857 | 0.7761 | 0.4609 | 0.7761 | 0.7685 | 0.4505 | 0.7685 | 25 | 1 |

## 6. Meilleurs points observes

- meilleure couverture : `tmax = 6` avec `coverage = 0.9857` et `num_remaining = 143`
- meilleur ARI assigne : `tmax = 2` avec `assigned_ari = 0.5250`
- meilleur ARI global : `tmax = 6` avec `global_ari = 0.4505`

## 7. Detail par run

### `tmax = 1`

- iterations utilisees : `1`
- clusters finaux : `44`
- good clusters avant merge : `45`
- phrases restantes : `4874`
- couverture : `0.5123`
- NMI / ARI / V assigne : `0.7909` / `0.5213` / `0.7909`
- NMI / ARI / V global : `0.5428` / `0.0396` / `0.5428`
- runtime run / evaluation : `28 s` / `1 s`

| iteration | K retenu | good | bad | score | remaining before | remaining after | accepted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 45 | 32 | 1.3636 | 9993 | 4874 | 45 |

- summary : `out/banking77_mistral_gpu1_st_tmax1_summary.json`
- clusters : `out/banking77_mistral_gpu1_st_tmax1_clusters.json`
- evaluation : `out/banking77_mistral_gpu1_st_tmax1_evaluation.json`
- run log : `out/banking77_mistral_gpu1_st_tmax1_run.log`
- evaluation log : `out/banking77_mistral_gpu1_st_tmax1_evaluate.log`

### `tmax = 2`

- iterations utilisees : `2`
- clusters finaux : `83`
- good clusters avant merge : `85`
- phrases restantes : `2785`
- couverture : `0.7213`
- NMI / ARI / V assigne : `0.8033` / `0.5250` / `0.8033`
- NMI / ARI / V global : `0.6748` / `0.1160` / `0.6748`
- runtime run / evaluation : `27 s` / `1 s`

| iteration | K retenu | good | bad | score | remaining before | remaining after | accepted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 45 | 32 | 1.3636 | 9993 | 4874 | 45 |
| 2 | 77 | 40 | 37 | 1.0526 | 4874 | 2785 | 40 |

- summary : `out/banking77_mistral_gpu1_st_tmax2_summary.json`
- clusters : `out/banking77_mistral_gpu1_st_tmax2_clusters.json`
- evaluation : `out/banking77_mistral_gpu1_st_tmax2_evaluation.json`
- run log : `out/banking77_mistral_gpu1_st_tmax2_run.log`
- evaluation log : `out/banking77_mistral_gpu1_st_tmax2_evaluate.log`

### `tmax = 3`

- iterations utilisees : `3`
- clusters finaux : `121`
- good clusters avant merge : `128`
- phrases restantes : `1436`
- couverture : `0.8563`
- NMI / ARI / V assigne : `0.7946` / `0.5040` / `0.7946`
- NMI / ARI / V global : `0.7345` / `0.2670` / `0.7345`
- runtime run / evaluation : `29 s` / `1 s`

| iteration | K retenu | good | bad | score | remaining before | remaining after | accepted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 45 | 32 | 1.3636 | 9993 | 4874 | 45 |
| 2 | 77 | 40 | 37 | 1.0526 | 4874 | 2785 | 40 |
| 3 | 77 | 43 | 34 | 1.2286 | 2785 | 1436 | 43 |

- summary : `out/banking77_mistral_gpu1_st_tmax3_summary.json`
- clusters : `out/banking77_mistral_gpu1_st_tmax3_clusters.json`
- evaluation : `out/banking77_mistral_gpu1_st_tmax3_evaluation.json`
- run log : `out/banking77_mistral_gpu1_st_tmax3_run.log`
- evaluation log : `out/banking77_mistral_gpu1_st_tmax3_evaluate.log`

### `tmax = 4`

- iterations utilisees : `4`
- clusters finaux : `154`
- good clusters avant merge : `179`
- phrases restantes : `483`
- couverture : `0.9517`
- NMI / ARI / V assigne : `0.7831` / `0.4758` / `0.7831`
- NMI / ARI / V global : `0.7610` / `0.4219` / `0.7610`
- runtime run / evaluation : `30 s` / `1 s`

| iteration | K retenu | good | bad | score | remaining before | remaining after | accepted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 45 | 32 | 1.3636 | 9993 | 4874 | 45 |
| 2 | 77 | 40 | 37 | 1.0526 | 4874 | 2785 | 40 |
| 3 | 77 | 43 | 34 | 1.2286 | 2785 | 1436 | 43 |
| 4 | 77 | 51 | 26 | 1.8889 | 1436 | 483 | 51 |

- summary : `out/banking77_mistral_gpu1_st_tmax4_summary.json`
- clusters : `out/banking77_mistral_gpu1_st_tmax4_clusters.json`
- evaluation : `out/banking77_mistral_gpu1_st_tmax4_evaluation.json`
- run log : `out/banking77_mistral_gpu1_st_tmax4_run.log`
- evaluation log : `out/banking77_mistral_gpu1_st_tmax4_evaluate.log`

### `tmax = 5`

- iterations utilisees : `5`
- clusters finaux : `189`
- good clusters avant merge : `222`
- phrases restantes : `215`
- couverture : `0.9785`
- NMI / ARI / V assigne : `0.7773` / `0.4636` / `0.7773`
- NMI / ARI / V global : `0.7662` / `0.4465` / `0.7662`
- runtime run / evaluation : `29 s` / `1 s`

| iteration | K retenu | good | bad | score | remaining before | remaining after | accepted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 45 | 32 | 1.3636 | 9993 | 4874 | 45 |
| 2 | 77 | 40 | 37 | 1.0526 | 4874 | 2785 | 40 |
| 3 | 77 | 43 | 34 | 1.2286 | 2785 | 1436 | 43 |
| 4 | 77 | 51 | 26 | 1.8889 | 1436 | 483 | 51 |
| 5 | 77 | 43 | 34 | 1.2286 | 483 | 215 | 43 |

- summary : `out/banking77_mistral_gpu1_st_tmax5_summary.json`
- clusters : `out/banking77_mistral_gpu1_st_tmax5_clusters.json`
- evaluation : `out/banking77_mistral_gpu1_st_tmax5_evaluation.json`
- run log : `out/banking77_mistral_gpu1_st_tmax5_run.log`
- evaluation log : `out/banking77_mistral_gpu1_st_tmax5_evaluate.log`

### `tmax = 6`

- iterations utilisees : `6`
- clusters finaux : `207`
- good clusters avant merge : `241`
- phrases restantes : `143`
- couverture : `0.9857`
- NMI / ARI / V assigne : `0.7761` / `0.4609` / `0.7761`
- NMI / ARI / V global : `0.7685` / `0.4505` / `0.7685`
- runtime run / evaluation : `25 s` / `1 s`

| iteration | K retenu | good | bad | score | remaining before | remaining after | accepted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 45 | 32 | 1.3636 | 9993 | 4874 | 45 |
| 2 | 77 | 40 | 37 | 1.0526 | 4874 | 2785 | 40 |
| 3 | 77 | 43 | 34 | 1.2286 | 2785 | 1436 | 43 |
| 4 | 77 | 51 | 26 | 1.8889 | 1436 | 483 | 51 |
| 5 | 77 | 43 | 34 | 1.2286 | 483 | 215 | 43 |
| 6 | 77 | 19 | 58 | 0.3220 | 215 | 143 | 19 |

- summary : `out/banking77_mistral_gpu1_st_tmax6_summary.json`
- clusters : `out/banking77_mistral_gpu1_st_tmax6_clusters.json`
- evaluation : `out/banking77_mistral_gpu1_st_tmax6_evaluation.json`
- run log : `out/banking77_mistral_gpu1_st_tmax6_run.log`
- evaluation log : `out/banking77_mistral_gpu1_st_tmax6_evaluate.log`

## 8. Interpretation

Cette campagne permet de separer proprement l effet de `tmax` du reste de la configuration. Comme `K` reste fixe a `77`, les differences proviennent surtout de la profondeur iterative et de la capacite de la boucle a extraire des intents supplementaires a partir du pool restant.

On s attend en general a observer trois regimes :

- une amelioration nette entre les premiers `tmax`,
- des gains plus faibles ensuite,
- puis un rendement decroissant quand les phrases restantes deviennent plus ambigues ou plus difficiles a clusteriser.

## 9. Artefacts generes

- script de sweep : `out/run_banking77_mistral_st_tmax_1_6.sh`
- rapport d environnement : `out/banking77_mistral_st_tmax1_6_env.txt`
- cache LLM partage : `out/banking77_mistral_st_tmax1_6_shared_cache.json`
- tableau JSON agrege : `out/banking77_mistral_st_tmax1_6_results.json`
- tableau CSV agrege : `out/banking77_mistral_st_tmax1_6_results.csv`
- note de run : `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_ST_TMAX_1_TO_6_FR.md`

## 10. Limites

- les temps de calcul ne sont pas directement comparables a un sweep sans cache partage,
- `accelerate` et `bitsandbytes` etaient absents, donc aucun chargement quantifie plus gros n a ete tente,
- le depot distant etait deja modifie avant cette campagne ; aucun nettoyage n a ete force.

