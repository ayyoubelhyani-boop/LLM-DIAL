# Execution sur BANKING77

Ce document decrit l'execution du pipeline sur `BANKING77`, un benchmark public utilise dans l'article.

## 1. Objectif

Remplacer le petit jeu de demonstration par un benchmark reel proche du papier, puis executer le pipeline complet dans un environnement distant plus adapte aux gros volumes.

## 2. Changements effectues

Les changements utiles pour ce run ont ete les suivants :

- export du split `train` de `BANKING77` vers `data/banking77_train.csv`,
- utilisation d'un serveur JupyterHub distant avec beaucoup de RAM et des GPU,
- execution a distance du pipeline a partir du projet deja present sur le serveur dans `~/nfs/LLM-DIAL`,
- chargement du fichier `data/banking77_train.csv` dans ce projet serveur,
- utilisation de `sentence-transformers` pour eviter les problemes memoire de la version `TF-IDF` dense sur plusieurs milliers d'exemples,
- choix d'un run borne avec `tmax = 1` et `K = 77` pour obtenir une premiere reference stable.

## 3. Environnement distant

Le run a ete execute sur un serveur JupyterHub distant accessible via :

- `http://maje-gpu01.biomedicale.univ-paris5.fr`

Configuration observee pendant l'execution :

- `Python 3.12.6`
- environ `1.0 TiB` de RAM
- `2 x NVIDIA L40`
- `torch 2.10.0+cu128`
- `sentence-transformers 5.3.0`

## 4. Commande executee

La commande suivante a ete lancee depuis `~/nfs/LLM-DIAL` :

```bash
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
  --tmax 1 \
  --use-llm false \
  --summary-out out/banking77_summary.json \
  --out out/banking77_clusters.json
```

## 5. Resultats obtenus

Resume principal :

- `num_clusters` : `62`
- `num_good_clusters` : `66`
- `num_remaining` : `1551`
- `iterations_used` : `1`
- `selected_k` : `77`
- `good_clusters` : `66`
- `bad_clusters` : `11`
- `score` : `5.5`

Quelques labels produits :

- `account-identity` : taille `92`
- `card-new` : taille `300`
- `pending-cash` : taille `87`
- `transfer-stop` : taille `285`
- `refund-want` : taille `147`
- `exchange-know` : taille `124`
- `pin-attempts` : taille `258`
- `card-credit` : taille `104`
- `transfer-sepa` : taille `190`
- `card-didn` : taille `109`

Une copie locale du resume distant a ete conservee dans :

- `out/banking77_summary.remote.json`

## 6. Interpretation

Ce run montre que le pipeline fonctionne bien sur un benchmark reel de taille significative.

Points positifs :

- le pipeline termine correctement sur l'ensemble `train`,
- la plupart des clusters candidats sont juges `Good`,
- les labels retrouvent plusieurs themes bancaires credibles,
- l'usage de `sentence-transformers` permet de traiter le benchmark sans surcharge memoire.

Limites observees :

- `62` clusters finaux restent inferieurs aux `77` intents de reference,
- `1551` utterances restent non assignees apres une seule iteration,
- certains labels generes restent approximatifs car le nommage repose encore sur le mode `dummy`,
- ce run est volontairement borne a `tmax = 1`, donc il ne pousse pas encore la boucle iterative jusqu'au bout.

## 7. Conclusion

Cette execution constitue une premiere reference serieuse sur `BANKING77`, bien plus proche du papier que le jeu de demonstration initial.

Les prochaines ameliorations les plus utiles seraient :

- tester plusieurs valeurs de `K` au lieu d'une seule,
- lancer plus d'une iteration,
- comparer les clusters obtenus aux labels de reference via une metrique comme le `NMI`,
- activer un vrai LLM pour la coherence et le naming.

## 8. Voir aussi

Pour la suite des experiments sur le meme dataset :

- `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_FR.md` : run local Mistral avec `TF-IDF`
- `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_ST_FR.md` : run local Mistral avec `sentence-transformers`
- `docs/COMPARAISON_BANKING77_FR.md` : premiere comparaison ciblee
- `docs/COMPARAISON_BANKING77_TOUTES_COMBINAISONS_FR.md` : comparaison complete des 8 combinaisons
