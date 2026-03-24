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
