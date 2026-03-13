# Run BANKING77 avec LLM local sur GPU 1

Ce document resume une execution complete du pipeline `Dial-In LLM` sur le benchmark public `BANKING77`, en utilisant un LLM local `Mistral-7B-Instruct` charge sur le deuxieme GPU du serveur.

## 1. Objectif du run

L'objectif de ce run etait de valider trois points :

- que le backend LLM local fonctionne sans API externe ;
- que le modele peut etre charge proprement sur `cuda:1` ;
- que le pipeline complet peut etre execute sur un benchmark du papier avec ce backend local.

## 2. Materiel et environnement

Serveur utilise :

- machine : `maje-gpu01`
- GPU 0 : `NVIDIA L40`, `46068 MB`
- GPU 1 : `NVIDIA L40`, `46068 MB`

Constat avant execution :

- `torch.cuda.device_count()` retourne `2`
- le GPU `1` etait libre
- `accelerate` n'etait pas installe
- `bitsandbytes` n'etait pas installe

Consequence pratique :

- le run a ete fait en mode non quantifie (`--local-llm-quantization none`)
- le chargement explicite sur le deuxieme GPU a ete force avec `--local-llm-device-map cuda:1`

## 3. Changements effectues pour rendre ce run possible

Les changements suivants ont ete faits dans `~/nfs/LLM-DIAL` avant ce run :

- ajout d'un backend LLM local `transformers` dans `dialin_llm/llm_utils.py` ;
- ajout des options CLI locales dans `dialin_llm/cli.py` ;
- ajout de l'extra `local-llm` dans `pyproject.toml` ;
- ajout d'une documentation d'usage dans `README.md` ;
- ajout de tests dans `tests/test_local_llm_utils.py` ;
- ajout d'un fallback de chargement mono-GPU quand `accelerate` est absent ;
- ajout de la prise en charge explicite de `cuda:1` pour cibler le deuxieme GPU.

Verification associee :

- `python3 -m pytest -q tests/test_local_llm_utils.py` -> **4 tests reussis**
- un smoke test de chargement local a confirme que `device_map=cuda:1` place bien le modele sur `GPU 1`

## 4. Jeu de donnees utilise

Fichier d'entree :

- `data/banking77_train.csv`

Ce fichier correspond au benchmark public `BANKING77`, deja prepare dans le format attendu par le pipeline.

## 5. Commande executee

```bash
cd ~/nfs/LLM-DIAL

HF_HOME=$PWD/.hf-home HUGGINGFACE_HUB_CACHE=$PWD/.hf-home/hub TRANSFORMERS_CACHE=$PWD/.hf-cache HF_HUB_DISABLE_SYMLINKS_WARNING=1 python3 -m dialin_llm.cli run   --input data/banking77_train.csv   --text-col text   --id-col sentence_id   --embed tfidf   --clusterer minibatch   --candidate-ks 77   --sample-size 10   --sampler farthest   --epsilon 0.05   --tmax 1   --use-llm true   --llm-provider local   --local-llm-model mistralai/Mistral-7B-Instruct-v0.3   --local-llm-device-map cuda:1   --local-llm-max-new-tokens 12   --local-llm-cache-dir .hf-cache   --cache-path out/banking77_mistral_gpu1_cache.json   --summary-out out/banking77_mistral_gpu1_summary.json   --out out/banking77_mistral_gpu1_clusters.json   > out/banking77_mistral_gpu1_run.log 2>&1
```

Fichiers generes :

- `out/banking77_mistral_gpu1_summary.json`
- `out/banking77_mistral_gpu1_clusters.json`
- `out/banking77_mistral_gpu1_cache.json`
- `out/banking77_mistral_gpu1_run.log`

Remarque : les caches Hugging Face ont ete gardes dans le projet (`.hf-home`, `.hf-cache`) afin de ne rien ecrire hors de `~/nfs/LLM-DIAL`.

## 6. Resultats obtenus

Resume global :

- nombre de clusters finaux apres merge : `45`
- nombre de clusters juges `Good` avant merge : `49`
- nombre de phrases restantes non assignees : `4213`
- nombre d'iterations utilisees : `1`
- valeur de `K` testee et retenue : `77`
- score obtenu : `49 / (28 + 1) = 1.6896551724137931`

Interpretation quantitative immediate :

- le pipeline a bien execute toute la boucle iterative avec LLM local ;
- une partie importante du dataset est restee non assignee apres une seule iteration ;
- la fusion finale a reduit `49` clusters acceptes a `45` clusters finaux.

## 7. Exemples de clusters produits

Quelques clusters volumineux observes :

- `card-status-inquiry` : `2203` phrases
- `link-card-app` : `263` phrases
- `exchange-rate-inquiry` : `199` phrases
- `delivery-status-tracking` : `194` phrases
- `track-card-status` : `190` phrases
- `activate-card` : `158` phrases
- `virtual-card-request` : `156` phrases
- `unblock-pin` : `153` phrases
- `cash-withdrawal-status-inquiry` : `114` phrases
- `card-delivery-status-inquiry` : `111` phrases

Exemples de labels produits par le modele local :

- `activate-card`
- `virtual-card-request`
- `investigate-direct-debit-dispute`
- `cash-withdrawal-status-inquiry`
- `refund-request`
- `exchange-rate-inquiry`

Ces labels montrent que le nommage local fonctionne globalement bien sur plusieurs groupes interpretabes.

## 8. Analyse qualitative

Le point positif principal est que le backend local fonctionne vraiment de bout en bout :

- chargement du modele local ;
- utilisation du GPU `1` ;
- evaluation `Good` / `Bad` sans API externe ;
- nommage des clusters avec le meme modele ;
- production des sorties JSON attendues.

En revanche, le resultat n'est pas encore satisfaisant comme clustering final de benchmark.

Les limites visibles sont les suivantes :

- plusieurs intentions proches autour de la carte et de la livraison sont absorbees dans de tres gros clusters ;
- `card-status-inquiry` devient un cluster massif, signe d'une granularite trop large ;
- une seule iteration (`tmax = 1`) laisse beaucoup de phrases dans le pool restant ;
- l'usage de `TF-IDF` favorise les rapprochements lexicaux plutot que semantiques ;
- le LLM local valide certains regroupements utiles, mais ne compense pas entierement la faiblesse de la representation vectorielle initiale.

Autrement dit :

- le run valide bien l'infrastructure GPU locale ;
- il ne valide pas encore une qualite de clustering comparable a une evaluation benchmark aboutie.

## 9. Cout pratique observe

Ce run a eu un cout raisonnable sur le serveur :

- telechargement initial du modele au premier lancement ;
- chargement du modele en FP16 sur `GPU 1` ;
- execution complete en environ deux minutes sur cette configuration borne.

Le mode choisi est donc realiste pour des essais iteratifs sur le serveur.

## 10. Conclusion

Ce run confirme que le projet peut maintenant executer un vrai benchmark du papier avec un LLM local sur GPU, sans dependre d'OpenAI.

Ce que ce run valide :

- support d'un backend `transformers` local ;
- ciblage du deuxieme GPU avec `cuda:1` ;
- execution complete du pipeline sur `BANKING77` ;
- generation de labels interpretabes et d'un cache local reutilisable.

Ce que ce run ne valide pas encore :

- une qualite de clustering suffisante pour servir d'evaluation benchmark finale ;
- une couverture satisfaisante du dataset en une seule iteration ;
- l'interet maximal du LLM local tant que les embeddings restent en `TF-IDF`.

La suite logique est donc :

- relancer `BANKING77` avec `sentence-transformers` ;
- tester plusieurs valeurs de `K` au lieu d'un seul `K = 77` ;
- augmenter `tmax` pour laisser la boucle iterative extraire plus d'intentions ;
- comparer ensuite ce backend local aux runs precedents documentes.
