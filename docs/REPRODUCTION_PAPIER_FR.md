# Reproduction des resultats du papier

Ce document fixe un protocole clair pour reproduire les resultats rapportes dans l'article *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues*.

## 1. Ce que le papier rapporte

Sur les benchmarks anglais, le papier donne les scores `NMI (%)` suivants pour la methode `ours` :

| Benchmark | Score papier |
|---|---:|
| `Bank77` | `82.32` |
| `CLINC(I)` | `94.12` |
| `MTOP(I)` | `72.45` |
| `Massive(I)` | `78.12` |

Source primaire :

- article ACL EMNLP 2025, Table 5
- `https://aclanthology.org/2025.emnlp-main.300.pdf`

Le papier indique aussi les tailles de benchmark suivantes :

| Benchmark | Taille | Nb intents |
|---|---:|---:|
| `Bank77` | `3080` | `77` |
| `CLINC(I)` | `4500` | `150` |
| `MTOP(I)` | `4386` | `102` |
| `Massive(I)` | `2974` | `59` |

Source primaire :

- article ACL EMNLP 2025, Table 1

## 2. Ecart avec l'etat actuel du projet

Aujourd'hui, le repo a deja execute plusieurs campagnes utiles, mais pas encore une reproduction stricte du protocole du papier :

- les runs documentes portent surtout sur `BANKING77 train` (`9993` phrases), pas sur le benchmark `Bank77` de taille `3080` ;
- le pipeline courant utilise des approximations explicites :
  - sampling `farthest-first` a la place du sampling `convex`,
  - merge probabiliste `vMF` non reproduit a l'identique,
  - backend local ou `dummy` a la place des LLMs fine-tunes des auteurs ;
- nous avons des metriques `NMI/ARI`, mais pas encore une campagne unique et alignee avec les quatre benchmarks de Table 5.

Conclusion honnête :

- nous avons une reproduction fonctionnelle du coeur de la methode ;
- nous n'avons pas encore une reproduction experimentale fidele des resultats du papier.

## 3. Priorite immediate

La premiere etape utile est de realigner les donnees et les scripts sur le protocole du papier.

Le projet supporte maintenant l'export explicite des benchmarks via la CLI :

```bash
python -m dialin_llm.cli export-benchmark --benchmark banking77 --split test --out data/banking77_test.csv
python -m dialin_llm.cli export-benchmark --benchmark clinc150 --split test --intent-only true --out data/clinc150_test.csv
python -m dialin_llm.cli export-benchmark --benchmark mtop --split test --config en --out data/mtop_en_test.csv
python -m dialin_llm.cli export-benchmark --benchmark massive --split test --config en-US --out data/massive_en_us_test.csv
```

## 4. Protocole de reproduction vise

Ordre recommande :

1. Exporter les quatre benchmarks avec les bons splits.
2. Verifier que les tailles exportees correspondent aux tailles rapportees par le papier.
3. Lancer une configuration "paper-like" commune sur les quatre jeux.
4. Evaluer chaque sortie avec `NMI`.
5. Comparer nos scores aux cibles de Table 5.

## 5. Criteres de succes

Nous considererons que la reproduction avance correctement si :

- les jeux exportes ont les bonnes tailles ;
- un pipeline unique tourne sur `Bank77`, `CLINC(I)`, `MTOP(I)` et `Massive(I)` ;
- un tableau de comparaison `score papier vs score reproduit` est produit ;
- les differences restantes sont expliquees par les ecarts d'implementation et de ressources.

## 6. Prochaine etape

La prochaine action concrete est :

- exporter les jeux `test`,
- preparer un script de campagne de reproduction,
- lancer un premier benchmark de reference sur `Bank77`.

## 7. Execution serveur effectuee

Une premiere campagne de reproduction a ete executee sur le serveur Jupyter dans :

- `/nfs/LLM-DIAL`

Contraintes respectees :

- tous les runs ont ete executes sur `cuda:1` ;
- tous les caches sont restes dans `.hf-cache/` sous le repo ;
- tous les artefacts sont restes dans `out/` sous le repo ;
- aucune modification n a ete faite hors du projet.

Environnement observe :

- Python `3.12.6`
- Linux `x86_64`
- `torch` disponible
- `transformers` disponible
- `sentence-transformers` disponible
- `datasets` non disponible sur le serveur
- `accelerate` non disponible
- `bitsandbytes` non disponible
- `2 x NVIDIA L40`
- GPU utilise pour les runs : `cuda:1`

## 8. Stack retenue pour la reproduction

La stack la plus fidele reellement executable dans cet environnement a ete :

- embeddings : `BAAI/bge-large-en-v1.5`
- LLM local : `mistralai/Mistral-7B-Instruct-v0.3`
- clustering : `MiniBatchKMeans`
- `sample-size = 10`
- `sampler = random`
- `tmax = 5`
- recherche de `K` par liste de candidats
- `--dedupe false`

Note de fidelite :

- les checkpoints `Meta-Llama` officiels sont bloques par acces `gated` sur ce serveur ;
- un probe `NousResearch/Llama-2-7b-chat-hf` etait possible, mais la sortie etait trop instable avec le prompting actuel pour obtenir une campagne robuste ;
- `Mistral-7B-Instruct` a donc ete retenu comme **approximation reproductible la plus praticable** dans cet environnement.

## 9. Resultats obtenus sur les benchmarks anglais

Comparaison avec les cibles du papier :

| Benchmark | NMI papier (%) | NMI reproduit (%) | Delta | Couverture | Clusters finaux | Restant |
|---|---:|---:|---:|---:|---:|---:|
| `Bank77` | `82.32` | `83.78` | `+1.46` | `98.77 %` | `142` | `38` |
| `CLINC(I)` | `94.12` | `90.90` | `-3.22` | `95.96 %` | `233` | `182` |
| `MTOP(I)` | `72.45` | `71.12` | `-1.33` | `96.26 %` | `185` | `164` |
| `Massive(I)` | `78.12` | `74.02` | `-4.10` | `86.79 %` | `101` | `389` |

Lecture :

- les scores reproduits ci-dessus correspondent au `NMI` **with_unassigned** ;
- les runs ont tous termine en `5` iterations ;
- les fichiers de sortie associes sont dans `out/*_paper_repro_*`.

## 10. Interpretation

Cette campagne montre que notre reproduction approchée est deja tres proche du papier sur plusieurs benchmarks :

- `Bank77` depasse meme legerement la cible papier ;
- `MTOP(I)` est proche de la cible ;
- `CLINC(I)` et `Massive(I)` restent en retrait, mais dans un ordre de grandeur credible.

Les principales causes probables d ecart restant sont :

- usage de `Mistral-7B-Instruct` au lieu du `Llama-7B` fine-tune des auteurs ;
- absence d une reproduction exacte du sampling `convex` ;
- merge final encore approximatif par rapport au protocole complet du papier ;
- choix des listes de `K` encore heuristiques ;
- prompting local different de celui des auteurs.

## 11. Priorite suivante

La prochaine amelioration la plus prometteuse est :

1. rapprocher davantage le comportement du couple coherence / naming des auteurs ;
2. reduire l explosion du nombre de clusters finaux sur `CLINC(I)` et `MTOP(I)` ;
3. tester une version plus proche de `Llama-7B` si un checkpoint ouvert et stable devient disponible ;
4. rapprocher la strategie de sampling de la version `convex` du papier.

## 12. Ablation rapide sur le sampling `convex`

Une approximation repo-locale du sampling `convex` a ensuite ete implemente et testee sur les deux benchmarks ou l ecart au papier etait le plus net :

| Benchmark | NMI random (%) | NMI convex-proxy (%) | Delta |
|---|---:|---:|---:|
| `CLINC(I)` | `90.90` | `90.11` | `-0.79` |
| `Massive(I)` | `74.02` | `72.53` | `-1.49` |

Conclusion pratique :

- dans l implementation actuelle, cette approximation de `convex sampling` n ameliore pas la reproduction ;
- l ecart ne semble donc pas venir uniquement du mode de sampling ;
- les prochains leviers les plus probables restent le couple coherence / naming, la selection de `K`, et le merge final.
