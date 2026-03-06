# Execution de demonstration et interpretation

Ce document resume une execution complete du pipeline `Dial-In LLM` sur un petit jeu de donnees de demonstration.

## 1. Jeu de donnees utilise

Fichier d'entree :

- `data/demo_support_utterances.csv`

Ce jeu contient 24 utterances de support client en anglais, reparties autour de 6 intentions simples :

- reinitialisation de mot de passe,
- annulation d'abonnement,
- demande de remboursement,
- probleme de livraison,
- probleme de paiement / facturation,
- changement d'adresse.

Remarque : le jeu est en anglais car l'evaluateur factice actuel (`DummyCoherenceEvaluator`) repose sur une tokenisation anglaise.

## 2. Commande executee

```bash
python -m dialin_llm.cli run ^
  --input data/demo_support_utterances.csv ^
  --text-col text ^
  --id-col id ^
  --embed tfidf ^
  --clusterer kmeans ^
  --candidate-ks 5,6,7 ^
  --sample-size 5 ^
  --sampler farthest ^
  --epsilon 0.0 ^
  --tmax 3 ^
  --use-llm false ^
  --summary-out out/demo_summary.json ^
  --out out/demo_clusters.json
```

Sorties generees :

- `out/demo_summary.json`
- `out/demo_clusters.json`

## 3. Resultat obtenu

Resume global :

- nombre de clusters finaux : 6
- nombre de clusters juges `Good` : 6
- nombre de phrases restantes non assignees : 0
- nombre d'iterations utilisees : 1
- valeur de `K` retenue : 6

Scores observes pour les valeurs candidates :

- `K = 5` -> `5 Good`, `0 Bad`, score `5.0`
- `K = 6` -> `6 Good`, `0 Bad`, score `6.0`
- `K = 7` -> `6 Good`, `1 Bad`, score `3.0`

Le systeme a donc choisi `K = 6`, car c'est la valeur qui maximise la metrique `Good / (Bad + 1)`.

## 4. Clusters produits

### Cluster 1

- label : `password-reset`
- ids : `1, 2, 3, 4`
- interpretation : demandes de reinitialisation de mot de passe

### Cluster 2

- label : `cancel-subscription`
- ids : `13, 14, 15, 16`
- interpretation : demandes d'annulation d'abonnement ou de service

### Cluster 3

- label : `refund-request`
- ids : `5, 7, 8`
- interpretation : demandes de remboursement

### Cluster 4

- label : `address-change`
- ids : `12, 21, 22, 23, 24`
- interpretation : majoritairement des demandes de changement d'adresse

Observation : la phrase `12` (`shipment delay tracking`) est plutot liee a la livraison qu'a l'adresse.

### Cluster 5

- label : `payment-billing`
- ids : `6, 17, 18, 19, 20`
- interpretation : majoritairement des problemes de paiement et de facturation

Observation : la phrase `6` (`refund payment amount`) est semantiquement proche du remboursement, mais son vocabulaire la rapproche ici du cluster paiement.

### Cluster 6

- label : `shipping-parcel`
- ids : `9, 10, 11`
- interpretation : demandes de suivi ou retard de livraison

## 5. Interpretation

Cette execution montre que le pipeline fonctionne de bout en bout :

- chargement du fichier,
- vectorisation TF-IDF,
- test de plusieurs valeurs de `K`,
- evaluation de coherence,
- selection du meilleur `K`,
- nomination des clusters,
- production d'un resultat JSON exploitable.

Sur ce jeu de demonstration, le comportement est globalement coherent :

- 3 clusters sont tres propres : `password-reset`, `cancel-subscription`, `shipping-parcel`,
- 2 clusters sont globalement corrects mais contiennent une phrase frontiere : `address-change` et `payment-billing`,
- 1 cluster de remboursement reste incomplet car une phrase de remboursement a ete attiree vers le champ lexical du paiement.

L'interpretation importante est la suivante : avec des embeddings TF-IDF simples et un evaluateur factice base sur les mots dominants, le systeme retrouve bien les grandes intentions, mais il reste sensible aux proximités lexicales locales.

Autrement dit :

- `refund payment amount` ressemble lexicalement a un probleme de paiement,
- `shipment delay tracking` ressemble lexicalement a une demande d'adresse a cause de la petite taille du jeu et des limites du clustering vectoriel simple.

## 6. Conclusion

Cette execution valide bien la faisabilite technique du projet, mais pas encore une qualite "production".

Pour ameliorer les resultats, les prochaines etapes les plus utiles sont :

- utiliser un embedder plus semantique, par exemple `sentence-transformers`,
- activer un vrai evaluateur LLM avec `--use-llm true`,
- tester sur un jeu de donnees reel plus volumineux,
- comparer les clusters obtenus a des labels connus pour mesurer la qualite.

## 7. Verification complementaire

Les tests du projet ont ete relances apres l'execution :

```bash
python -m pytest -q
```

Resultat :

- `6 passed`
