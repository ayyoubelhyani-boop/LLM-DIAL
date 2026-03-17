# Documentation des composantes

Ce document explique le role de chaque composante du projet `dialin_llm`, la maniere dont elle fonctionne, et ce qu'elle ajoute au pipeline global.

## 1. Vue d'ensemble

Le pipeline suit la chaine suivante :

1. charger les phrases d'entree,
2. produire des embeddings,
3. tester plusieurs clusterings candidats,
4. echantillonner des phrases representatives par cluster,
5. evaluer la coherence de chaque cluster,
6. retenir le meilleur `K`,
7. conserver uniquement les clusters juges `Good`,
8. iterer sur les phrases restantes,
9. nommer les clusters retenus,
10. fusionner les labels proches,
11. exporter le resultat.

Autrement dit, chaque module n'est pas isole : il intervient dans une chaine logique ou chaque etape prepare la suivante.

## 2. `io.py`

### Ce que fait le module

`io.py` charge les donnees d'entree depuis un fichier `CSV` ou `JSONL`.

### Comment il marche

- il verifie que le fichier existe,
- il detecte automatiquement le format via l'extension,
- il lit les lignes,
- il extrait la colonne de texte demandee,
- il ignore les lignes vides,
- il peut dedupliquer les phrases identiques,
- il construit des objets `SentenceRecord`.

### Ce qu'il ajoute

Ce module normalise l'entree. Sans lui, le reste du pipeline devrait gerer lui-meme le parsing des fichiers, les erreurs de colonnes et la suppression des doublons.

## 3. `embeddings.py`

### Ce que fait le module

`embeddings.py` transforme les phrases en vecteurs numeriques.

### Comment il marche

- `TfidfEmbeddingBackend` produit une representation TF-IDF classique,
- `SentenceTransformerEmbeddingBackend` permet d'utiliser un encodeur semantique moderne,
- `build_embedding_backend()` choisit le backend a partir du nom passe en CLI,
- `l2_normalize()` sert a normaliser certains vecteurs, notamment pour la phase de merge.

### Ce qu'il ajoute

Ce module fournit l'espace vectoriel necessaire au clustering. Sans representation numerique, il n'est pas possible d'appliquer `KMeans`, de comparer les labels ou de mesurer des distances.

## 4. `clustering.py`

### Ce que fait le module

`clustering.py` applique l'algorithme de clustering sur les embeddings.

### Comment il marche

- `cluster_embeddings()` choisit entre `KMeans` et `MiniBatchKMeans`,
- le parametre `k` est borne pour ne jamais depasser le nombre de points,
- `fit_predict()` attribue un label de cluster a chaque phrase,
- `group_cluster_members()` reconstruit ensuite les groupes a partir des labels predits.

### Ce qu'il ajoute

Ce module realise le decoupage brut des phrases en groupes candidats. C'est la premiere vraie hypothese structurelle du pipeline : quelles phrases semblent aller ensemble.

## 5. `sampling.py`

### Ce que fait le module

`sampling.py` choisit un sous-ensemble representatif de phrases dans chaque cluster.

### Comment il marche

- `random_sample()` prend un echantillon aleatoire,
- `farthest_first_sample()` commence pres du centre puis ajoute les points les plus eloignes parmi ceux qui restent,
- `sample_indices()` choisit la bonne strategie selon le parametre `sampler`.

### Ce qu'il ajoute

Ce module evite d'evaluer ou de nommer un cluster sur toutes ses phrases. Il reduit donc le cout et permet de presenter au LLM, ou au composant dummy, un echantillon plus compact mais informatif.

## 6. `llm_utils.py`

### Ce que fait le module

`llm_utils.py` contient les composants lies a l'evaluation de coherence et au nommage des clusters.

### Comment il marche

- `CoherenceEvaluator` definit l'interface attendue pour juger un cluster,
- `ClusterNamer` definit l'interface attendue pour nommer un cluster,
- `DummyCoherenceEvaluator` valide un cluster si certains tokens dominants reviennent assez souvent,
- `DummyClusterNamer` construit un label simple a partir des mots les plus frequents,
- `LocalTransformersTextGenerator` charge un modele Hugging Face local et genere les reponses de coherence et de nommage,
- `LocalTransformersCoherenceEvaluator` utilise ce generateur pour retourner `Good` ou `Bad`,
- `LocalTransformersClusterNamer` utilise le meme backend local pour produire un label `action-objectif`,
- `OpenAICoherenceEvaluator` interroge un modele OpenAI pour retourner `Good` ou `Bad`,
- `OpenAIClusterNamer` demande au modele un label `action-objectif`,
- `JsonCache` memorise les reponses afin d'eviter des appels repetes,
- des verifications strictes garantissent que les sorties du LLM respectent le format attendu.

En pratique, la logique `Good` / `Bad` signifie :

- `Good` : les phrases du cluster semblent exprimer une meme intention,
- `Bad` : le cluster semble melanger plusieurs intentions ou rester trop ambigu.

Dans la version `dummy`, cette decision repose sur la frequence des mots dominants dans l'echantillon du cluster. Si un mot important revient suffisamment souvent, le cluster est juge coherent donc `Good`. Sinon, il est classe `Bad`.

### Ce qu'il ajoute

Ce module ajoute l'intelligence de validation et de denomination. Le clustering seul produit des groupes mathematiques ; ici, on ajoute une couche de jugement semantique inspiree de l'article.

Le projet supporte donc maintenant trois regimes principaux :

- un mode `dummy` entierement hors ligne,
- un mode OpenAI,
- un mode local `transformers`, teste avec `Mistral-7B-Instruct` sur GPU.

## 7. `iterative.py`

### Ce que fait le module

`iterative.py` implemente le coeur de l'algorithme Dial-In LLM.

### Comment il marche

- il recoit les phrases et leurs embeddings,
- il maintient la liste des phrases encore non assignees,
- a chaque iteration, il teste plusieurs valeurs de `K`,
- pour chaque `K`, il clusterise uniquement les phrases restantes,
- il echantillonne les clusters obtenus,
- il appelle l'evaluateur de coherence,
- il calcule le score `Good / (Bad + 1)`,
- il choisit le meilleur `K`,
- il accepte uniquement les clusters juges `Good`,
- il retire leurs phrases du pool restant,
- il recommence jusqu'a `epsilon`, epuisement ou `tmax`.

Les dataclasses du module servent aussi a garder une trace exploitable des iterations, des clusters et des scores.

### Ce qu'il ajoute

Ce module ajoute la dynamique iterative. C'est lui qui fait la specificite de l'approche : on ne garde pas tout, on filtre progressivement les clusters juges coherents.

## 8. `merge.py`

### Ce que fait le module

`merge.py` nomme les clusters retenus puis fusionne les clusters dont les labels sont tres proches.

### Comment il marche

- `name_clusters()` applique le namer sur un echantillon de phrases de chaque cluster,
- `merge_clusters_by_label()` encode les labels avec un TF-IDF caractere,
- les labels sont normalises,
- une distance geodesique est calculee entre chaque paire de labels,
- si la distance est inferieure a `theta`, les clusters sont fusionnes,
- `UnionFind` sert a gerer les regroupements de maniere efficace,
- une porte probabiliste optionnelle `vMF` existe mais n'est pas active par defaut.

### Ce qu'il ajoute

Ce module corrige un effet courant du clustering : deux clusters presque identiques peuvent etre conserves separement. Le merge final reduit cette redondance.

## 9. `metrics.py`

### Ce que fait le module

`metrics.py` fournit quelques outils d'evaluation.

### Comment il marche

- `normalized_mutual_info()` compare des labels predits a des labels de reference,
- `goodness_stats()` resume rapidement un run,
- `cluster_sizes()` donne la taille de chaque cluster.

### Ce qu'il ajoute

Ce module ne participe pas directement a la generation des clusters, mais il ajoute la capacite de mesurer et de comparer les resultats.

## 10. `cli.py`

### Ce que fait le module

`cli.py` assemble tout le pipeline et l'expose en ligne de commande.

### Comment il marche

- il parse les arguments utilisateur,
- il charge les phrases,
- il construit le backend d'embedding,
- il choisit l'evaluateur et le namer (`dummy` ou OpenAI),
- il lance la boucle iterative,
- il appelle le nommage et le merge,
- il construit un resume final,
- il ecrit les sorties JSON.

### Ce qu'il ajoute

Ce module rend le projet executable de bout en bout. C'est le point d'entree qui transforme les briques techniques en un outil concret.

## 11. Comment les composantes collaborent

Le fonctionnement global peut se resumer ainsi :

- `cli.py` orchestre,
- `io.py` alimente,
- `embeddings.py` represente,
- `clustering.py` partitionne,
- `sampling.py` resume localement chaque cluster,
- `llm_utils.py` juge et nomme,
- `iterative.py` decide et itere,
- `merge.py` nettoie et consolide,
- `metrics.py` mesure.

Cette separation des responsabilites apporte trois avantages majeurs :

- lisibilite : chaque module a une responsabilite nette,
- remplacabilite : on peut changer un embedder ou un evaluateur sans casser le reste,
- testabilite : chaque brique peut etre verifiee independamment.

## 12. Conclusion

L'architecture du projet est volontairement modulaire : chaque composante ajoute une fonction precise a la chaine globale, et l'ensemble reproduit une version executable et pedagogiquement claire de la logique Dial-In LLM.

Le point central a retenir est que la qualite finale ne depend pas d'une seule brique. Elle depend de l'interaction entre :

- la qualite des embeddings,
- la pertinence du clustering,
- la representativite du sampling,
- la fiabilite de l'evaluation de coherence,
- la qualite du nommage et du merge final.

Les experiments menes sur `BANKING77` confirment concretement ce point :

- `sentence-transformers` ameliore fortement la couverture par rapport a `TF-IDF`,
- `tmax = 2` est plus robuste que `tmax = 1`,
- le mode `dummy` maximise la couverture brute,
- le mode `Mistral local` est plus fidele a une logique `LLM-in-the-loop`, mais plus selectif.
