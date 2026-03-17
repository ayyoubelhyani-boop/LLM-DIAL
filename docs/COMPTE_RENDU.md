# Compte Rendu Projet

## Sujet

Ce projet propose une version runnable du coeur de l'algorithme de l'article *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues*.

L'objectif etait de reproduire le mecanisme principal de clustering iteratif assiste par LLM dans une base simple, modulaire, testable et executable sur CPU.

## Principe retenu

L'idee est de regrouper automatiquement des phrases de dialogues clients en intentions coherentes en combinant :

- une representation vectorielle des phrases,
- un clustering non supervise,
- une evaluation de coherence de chaque cluster,
- une boucle iterative qui ne conserve que les clusters juges pertinents.

Cette implementation ne reprend pas tout le papier a l'identique, mais elle respecte ses etapes centrales.

## Architecture du projet

Le code a ete organise dans le package Python `dialin_llm`, avec une separation claire des responsabilites :

- `io.py` : chargement des donnees `CSV` ou `JSONL` et suppression des doublons.
- `embeddings.py` : embeddings en `TF-IDF`, avec extension optionnelle vers `sentence-transformers`.
- `sampling.py` : echantillonnage aleatoire ou `farthest-first` pour representer chaque cluster.
- `clustering.py` : utilisation de `KMeans` et `MiniBatchKMeans`.
- `llm_utils.py` : evaluation et nommage des clusters en mode `dummy`, via OpenAI, ou via un backend local `transformers` teste avec Mistral.
- `iterative.py` : boucle iterative principale.
- `merge.py` : fusion des clusters proches a partir de la similarite semantique de leurs labels.
- `metrics.py` : metriques simples d'evaluation.
- `cli.py` : execution du pipeline complet en ligne de commande.

## Algorithme implemente

La boucle principale suit ce schema :

1. Charger puis vectoriser les phrases.
2. Tester plusieurs valeurs candidates de `K`.
3. Clusteriser les phrases restantes pour chaque `K`.
4. Echantillonner des phrases representatives dans chaque cluster.
5. Evaluer chaque cluster en `Good` ou `Bad`.
6. Calculer le score `Good / (Bad + 1)` pour chaque `K`.
7. Conserver le meilleur `K`.
8. Garder uniquement les clusters `Good` et retirer leurs phrases du pool.
9. Recommencer jusqu'au seuil `epsilon` ou au maximum `tmax`.

En sortie, chaque cluster est nomme sous la forme `action-objectif`, puis les labels semantiquement proches peuvent etre fusionnes.

## Fidelite et simplifications

Les points fideles a l'article sont :

- la boucle iterative de clustering,
- le test de plusieurs valeurs de `K`,
- l'evaluation `Good` / `Bad`,
- le choix du meilleur `K` par `Good / (Bad + 1)`,
- la conservation des bons clusters,
- la fusion finale par proximite semantique.

Les principales simplifications sont :

- le sampling "convex" est approxime par `farthest-first`,
- un mode `dummy` permet une execution hors ligne,
- la fusion probabiliste basee sur `vMF` reste optionnelle car certains parametres du papier sont insuffisamment precises.

Le projet a egalement ete etendu avec un backend local `transformers` pour tester une version GPU sans dependance a une API distante.

## Validation

Des tests automatises verifient notamment :

- le retrait effectif des phrases des clusters `Good`,
- le bon choix de `K`,
- la fusion de labels proches,
- le caractere borne et deterministe du sampling.

Resultats obtenus :

- `python -m pytest -q` : **6 tests reussis**
- un test d'execution complet via la CLI a egalement fonctionne.

Des experiments complementaires ont ensuite ete menes sur le benchmark `BANKING77` pour comparer plusieurs regimes :

- evaluateur `dummy` contre evaluateur local `Mistral`,
- embeddings `TF-IDF` contre `sentence-transformers`,
- `tmax = 1` contre `tmax = 2`.

Ces comparaisons montrent notamment que :

- `sentence-transformers` ameliore nettement les resultats par rapport a `TF-IDF`,
- `tmax = 2` est plus efficace que `tmax = 1`,
- le mode `dummy` reste le plus performant en couverture brute,
- le meilleur compromis dans une logique plus proche du papier est `Mistral local + sentence-transformers + tmax = 2`.

## Conclusion

Le projet fournit une premiere implementation propre et exploitable du coeur de l'approche Dial-In LLM.

Cette base permet d'executer l'algorithme de bout en bout hors ligne, d'integrer ensuite un vrai LLM sans changer l'architecture generale, puis d'ameliorer progressivement les embeddings, le sampling et le merge probabiliste.

La documentation des runs et des comparaisons a ete ajoutee dans `docs/` afin de garder une trace claire des choix de configuration et des resultats observés sur `BANKING77`.

Pour une lecture rapide et presentable, le document de synthese recommande est :

- `docs/SYNTHESE_BANKING77_FR.md`
