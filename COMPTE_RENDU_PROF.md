# Compte Rendu Projet

## Sujet

Ce travail consiste a implementer une version runnable du coeur de l'algorithme presente dans l'article *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues*.

L'objectif est de reproduire le mecanisme principal de clustering iteratif assiste par LLM, tout en gardant une base simple, modulaire et executable sur CPU.

## Objectif scientifique et technique

L'idee generale de l'article est de regrouper automatiquement des phrases de dialogues clients en intentions coherentes, en combinant :

- une etape de representation vectorielle des phrases,
- un clustering non supervise,
- une evaluation de coherence de chaque cluster,
- une boucle iterative qui conserve uniquement les clusters juges pertinents.

Dans cette implementation, l'objectif n'etait pas de reproduire tout l'article a l'identique, mais de construire une base logicielle propre et testable qui respecte les etapes centrales de la methode.

## Travail realise

Le projet a ete structure sous forme de package Python `dialin_llm`, avec une separation claire des responsabilites :

- `io.py` : chargement des phrases depuis des fichiers `CSV` ou `JSONL`, avec suppression des doublons.
- `embeddings.py` : generation d'embeddings avec une version simple en `TF-IDF`, et une extension optionnelle vers `sentence-transformers`.
- `sampling.py` : echantillonnage de phrases representatives, soit aleatoire, soit par strategie `farthest-first` pour favoriser la diversite.
- `clustering.py` : encapsulation des algorithmes `KMeans` et `MiniBatchKMeans`.
- `llm_utils.py` : definition de l'interface d'evaluation de coherence et de nommage des clusters, avec :
  - une version `dummy` hors ligne,
  - une version optionnelle OpenAI avec cache, validation stricte et gestion des retries.
- `iterative.py` : implementation de la boucle iterative principale.
- `merge.py` : fusion des clusters proches a partir de la similarite semantique de leurs labels.
- `metrics.py` : metriques simples d'evaluation.
- `cli.py` : interface en ligne de commande pour executer tout le pipeline.

## Algorithme implemente

La boucle principale suit la logique generale de l'article :

1. Les phrases sont chargees puis vectorisees.
2. A chaque iteration, plusieurs valeurs candidates de `K` sont testees.
3. Pour chaque `K`, les phrases restantes sont clusterisees.
4. Pour chaque cluster, un echantillon de phrases representatives est selectionne.
5. Cet echantillon est evalue par une fonction de coherence qui retourne `Good` ou `Bad`.
6. Un score est calcule pour chaque `K` selon la formule :

   `score = Good / (Bad + 1)`

7. Le meilleur `K` est retenu.
8. Les clusters juges `Good` sont conserves.
9. Les phrases appartenant a ces clusters sont retirees du pool restant.
10. Le processus recommence jusqu'a atteindre un seuil `epsilon` ou un nombre maximal d'iterations `tmax`.

Apres cette phase iterative :

- chaque cluster est nomme sous forme d'un label de type `action-objectif`,
- les clusters aux labels semantiquement proches sont fusionnes a l'aide d'une distance geodesique calculee sur des embeddings normalises.

## Fidelite par rapport a l'article

Les elements suivants sont fideles a la logique du papier :

- boucle iterative de clustering,
- test de plusieurs valeurs de `K`,
- evaluation `Good` / `Bad`,
- choix du meilleur `K` par la formule `Good / (Bad + 1)`,
- conservation uniquement des bons clusters,
- fusion finale par proximite semantique des labels.

Certaines parties ont cependant ete simplifiees ou approximees :

- Le sampling "convex" de l'article a ete approxime par une methode pratique `farthest-first`.
- Une version `dummy` a ete introduite pour permettre une execution sans dependance reseau.
- Le mecanisme probabiliste de fusion base sur une loi `vMF` n'est pas active par defaut, car certains parametres, notamment `kappa`, ne sont pas suffisamment precises dans le papier. Une extension optionnelle a ete ajoutee avec hypothese explicite.

## Qualites de l'implementation

L'implementation a ete pensee pour etre :

- modulaire : chaque composant peut etre remplace independamment ;
- testable : des tests unitaires couvrent les comportements essentiels ;
- reproductible : les procedures aleatoires utilisent un `seed` ;
- exploitable : une CLI permet de lancer facilement le pipeline sur un petit jeu de donnees.

## Validation

Des tests automatises ont ete ajoutes pour verifier notamment :

- que la boucle iterative retire bien les phrases des clusters juges `Good`,
- que le choix de `K` suit bien la meilleure valeur du score,
- que la fusion par labels fonctionne lorsque deux labels sont proches,
- que l'echantillonnage reste borne et deterministe avec une graine fixee.

Le resultat de la validation est le suivant :

- `python -m pytest -q` : **6 tests reussis**
- un test d'execution complet via la CLI a egalement fonctionne.

## Conclusion

Le projet aboutit a une premiere implementation propre et exploitable du coeur de l'approche Dial-In LLM.

Cette base permet :

- d'executer l'algorithme de bout en bout avec une version hors ligne,
- d'activer ensuite un vrai LLM sans modifier l'architecture generale,
- d'etendre ulterieurement la qualite des embeddings, du sampling et du merge probabiliste.

