# Execution de demonstration et interpretation

Ce document resume une execution complete du pipeline `Dial-In LLM` sur un petit jeu de donnees de demonstration plus proche du papier que le jeu artificiel initial.

## 1. Jeu de donnees utilise

Fichier d'entree :

- `data/demo_support_utterances.csv`

Ce fichier contient maintenant **24 utterances reelles issues de `BANKING77`**, un benchmark public de classification d'intentions utilise dans le papier.

Le sous-ensemble retenu couvre 6 intentions bancaires :

- `card_arrival`
- `beneficiary_not_allowed`
- `cash_withdrawal_charge`
- `declined_cash_withdrawal`
- `transfer_not_received_by_recipient`
- `cash_withdrawal_not_recognised`

Le fichier conserve une taille reduite pour rester pratique en demonstration, mais il est semantiquement beaucoup plus proche du domaine du papier : service client, demandes courtes, intents metier et formulations naturelles.

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

Scores observes :

- `K = 5` -> `4 Good`, `1 Bad`, score `2.0`
- `K = 6` -> `6 Good`, `0 Bad`, score `6.0`
- `K = 7` -> `6 Good`, `1 Bad`, score `3.0`

Le systeme retient donc `K = 6`, ce qui correspond ici au nombre d'intentions presentes dans le sous-ensemble.

## 4. Interpretation des clusters

Le comportement observe est globalement coherent :

- un cluster recouvre correctement les demandes de `card_arrival`,
- un cluster recouvre correctement les demandes de `transfer_not_received_by_recipient`,
- un petit cluster isole bien une partie des demandes `beneficiary_not_allowed`,
- plusieurs utterances autour du cash withdrawal sont rapprochees entre elles, ce qui est plausible car elles partagent un vocabulaire tres proche.

On observe aussi des melanges semantiques attendus avec un couple `TF-IDF` + evaluateur `dummy` :

- certaines demandes de frais de retrait et de retrait refuse sont rapprochees,
- une partie des utterances sur les transferts et les beneficiaires se regroupent par champ lexical,
- les labels generes restent approximatifs car ils sont produits sans vrai LLM.

## 5. Conclusion

Cette demonstration est plus representative du papier que la version initiale, car elle repose sur un **benchmark public reel du meme type de tache**.

Elle ne remplace pas encore les gros jeux du papier, mais elle fournit une base plus credible pour :

- tester le pipeline de bout en bout,
- comparer plusieurs embeddings,
- evaluer l'effet d'un vrai LLM sur la coherence et le naming,
- preparer ensuite des experiences sur `BANKING77`, `CLINC150`, `MTOP` ou `MASSIVE`.

## 6. Verification complementaire

La commande de demonstration a ete relancee avec succes sur ce nouveau fichier, et les sorties `out/demo_summary.json` et `out/demo_clusters.json` ont bien ete regenerees.
