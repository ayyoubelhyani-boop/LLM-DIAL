# Chapitre - Reproduction du papier *Dial-In LLM*

## 1. Objectif de la reproduction

L objectif de cette partie du projet etait de reproduire aussi fidelement que possible les resultats presentes dans l article *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues*, en se concentrant sur la partie la plus reproductible avec les ressources publiques : l evaluation sur les benchmarks anglais.

Plutot que de viser immediatement les resultats principaux sur le corpus chinois des auteurs, nous avons choisi de cibler le tableau de comparaison sur quatre benchmarks anglais, car :

- les jeux de donnees sont publics ;
- les tailles de benchmark sont connues ;
- les scores cibles du papier sont explicitement rapportes ;
- cette partie permet deja d evaluer la qualite du pipeline LLM-in-the-loop dans un cadre comparable.

Les scores de reference du papier sont les suivants :

| Benchmark | NMI papier (%) |
|---|---:|
| `Bank77` | `82.32` |
| `CLINC(I)` | `94.12` |
| `MTOP(I)` | `72.45` |
| `Massive(I)` | `78.12` |

## 2. Alignement du protocole

Avant de lancer les runs, un travail d alignement a ete realise pour garantir que la comparaison se fasse sur les bons jeux de donnees et selon un protocole coherent avec l article.

Les benchmarks exportes et verifies sont :

| Benchmark | Taille observee | Taille papier |
|---|---:|---:|
| `Bank77` | `3080` | `3080` |
| `CLINC(I)` | `4500` | `4500` |
| `MTOP(I)` | `4386` | `4386` |
| `Massive(I)` | `2974` | `2974` |

Ce travail a necessite plusieurs ajustements :

- export explicite des splits `test` dans le format du pipeline ;
- filtrage `intent-only` pour `CLINC(I)` ;
- ajout d un flag `--dedupe false` pour eviter qu un benchmark soit modifie artificiellement par la deduplication interne ;
- generation d un fichier de configuration de run afin de conserver un protocole completement tracable.

## 3. Environnement experimental

Les experiences de reproduction ont ete executees sur le serveur Jupyter du projet, dans le repertoire :

- `/nfs/LLM-DIAL`

Contraintes retenues :

- utilisation systematique du second GPU `cuda:1` ;
- ecriture uniquement dans le depot du projet ;
- caches modeles conserves dans `.hf-cache/` ;
- sorties de runs conservees dans `out/` ;
- aucune modification de fichier ou de depot hors du projet.

Environnement observe :

- Python `3.12.6`
- `torch` disponible
- `transformers` disponible
- `sentence-transformers` disponible
- `2 x NVIDIA L40`
- `accelerate` non disponible
- `bitsandbytes` non disponible

Ces contraintes ont directement influence le niveau de fidelite accessible par rapport au papier.

## 4. Choix d implementation

L article indique que les benchmarks anglais sont traites avec un `Llama-7B` fine-tune sur `800` clusters d intents annotes. En pratique, la reproduction exacte de cette configuration n etait pas possible dans notre environnement, car les checkpoints officiels equivalents `Meta-Llama` sont proteges par un acces `gated`, et les modeles fine-tunes des auteurs ne sont pas publics.

Nous avons donc retenu la pile suivante comme approximation la plus fidele reellement executable :

- embeddings : `BAAI/bge-large-en-v1.5`
- evaluateur et namer locaux : `mistralai/Mistral-7B-Instruct-v0.3`
- clustering : `MiniBatchKMeans`
- `sample-size = 10`
- `tmax = 5`
- recherche par candidats de `K`
- `dedupe = false`

Cette pile est une approximation raisonnable du protocole du papier, mais pas une reproduction exacte. Les principaux ecarts sont :

- absence du `Llama-7B` fine-tune des auteurs ;
- absence du sampling `convex` exact ;
- merge final encore simplifie ;
- prompting local necessairement different du protocole original.

## 5. Resultats de reproduction

La campagne principale a donne les resultats suivants :

| Benchmark | NMI papier (%) | NMI reproduit (%) | Delta | Couverture | Clusters finaux | Restant |
|---|---:|---:|---:|---:|---:|---:|
| `Bank77` | `82.32` | `83.78` | `+1.46` | `98.77 %` | `142` | `38` |
| `CLINC(I)` | `94.12` | `90.90` | `-3.22` | `95.96 %` | `233` | `182` |
| `MTOP(I)` | `72.45` | `71.12` | `-1.33` | `96.26 %` | `185` | `164` |
| `Massive(I)` | `78.12` | `74.02` | `-4.10` | `86.79 %` | `101` | `389` |

Plusieurs observations ressortent :

1. La reproduction est deja tres competitive sur `Bank77`, ou notre score depasse legerement celui du papier.
2. Le resultat sur `MTOP(I)` est tres proche du score cible.
3. Les ecarts les plus nets concernent `CLINC(I)` et `Massive(I)`.
4. La couverture reste tres forte sur trois benchmarks, mais le nombre de clusters finaux produits reste souvent superieur au nombre d intents de reference.

## 6. Essai d amelioration : approximation du sampling `convex`

Comme l article souligne l importance du `convex sampling`, une approximation repo-locale de cette idee a ete implemente. Cette version selectionne d abord des points extremes via une projection basse dimension et une coque convexe, puis complete l echantillon avec une logique de diversification.

Nous avons teste cette variante sur les benchmarks les plus en retard :

| Benchmark | NMI random (%) | NMI convex-proxy (%) | Delta |
|---|---:|---:|---:|
| `CLINC(I)` | `90.90` | `90.11` | `-0.79` |
| `Massive(I)` | `74.02` | `72.53` | `-1.49` |

Cet essai montre que l approximation actuelle du `convex sampling` n apporte pas d amelioration mesurable sur ces benchmarks, et degrade meme legerement les scores. Cela suggere que :

- soit notre approximation est encore trop eloignee du protocole original ;
- soit le sampling n est pas le facteur principal limitant la reproduction dans notre implementation.

## 7. Analyse des ecarts restants

Les ecarts entre nos resultats et ceux du papier peuvent raisonnablement s expliquer par plusieurs facteurs cumulatifs.

### 7.1 Difference de modele LLM

Le papier utilise un `Llama-7B` fine-tune pour les benchmarks anglais, alors que notre meilleure configuration executable repose sur `Mistral-7B-Instruct`. Ce changement affecte probablement :

- la qualite des jugements `Good/Bad` ;
- la stabilite du namer ;
- la sensibilite aux formulations de prompts.

### 7.2 Sampling encore approximatif

Le papier signale explicitement l interet du `convex sampling`. Notre pipeline en propose maintenant une approximation, mais pas l algorithme original exact.

### 7.3 Merge final simplifie

Le merge utilise encore une logique plus simple que le merge probabiliste discute dans le papier. Cela peut contribuer a la sur-fragmentation observee sur certains benchmarks.

### 7.4 Recherche de `K`

Nos listes de `K` candidates restent heuristiques. Or le papier insiste sur la selection dynamique de l espace de recherche et sur l importance de ce point pour l efficacite globale du pipeline.

## 8. Bilan

La reproduction obtenue est suffisamment solide pour soutenir les conclusions suivantes :

- le coeur de l approche du papier a bien ete reconstruit et execute de bout en bout ;
- la comparaison sur les benchmarks anglais est pertinente et techniquement defensible ;
- la methode reproduite reste proche des resultats du papier sur plusieurs jeux de donnees ;
- les ecarts restants sont localises, explicables, et peuvent faire l objet d une amelioration ciblee.

En particulier, la reproduction actuelle est deja forte sur `Bank77` et `MTOP(I)`, et raisonnablement proche sur `CLINC(I)` et `Massive(I)` compte tenu des contraintes d acces aux modeles et des simplifications d implementation.

## 9. Suite du travail

Au vu des resultats obtenus, la suite la plus logique du projet n est plus de relancer des benchmarks bruts, mais de cibler les leviers susceptibles de reduire les ecarts restants :

1. ameliorer encore le comportement du couple coherence / naming ;
2. reduire la sur-fragmentation des clusters finaux ;
3. revisiter la recherche de `K` ;
4. tester, si possible, une alternative plus proche du `Llama-7B` du papier.

Ainsi, cette phase de reproduction constitue a la fois :

- une validation experimentale de la methode ;
- une base propre pour les ameliorations algorithmiques proposees dans la suite du projet.
