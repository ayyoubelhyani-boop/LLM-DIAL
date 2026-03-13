# Dial-In LLM

This repository contains a runnable Python implementation of the core iterative intent clustering loop from the paper *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues*.

## What the algorithm does

1. Load and deduplicate dialogue utterances.
2. Embed all sentences once.
3. At each iteration, try several candidate cluster counts `K`.
4. Cluster only the sentences that are still unassigned.
5. Sample up to 20 representative sentences per cluster.
6. Ask a coherence evaluator to mark each cluster as `Good` or `Bad`.
7. Score each candidate `K` as `Good / (Bad + 1)`.
8. Keep only the clusters judged `Good`.
9. Remove their sentences and repeat until the remaining fraction is small enough or `tmax` is reached.
10. Name clusters, then merge semantically similar labels using geodesic distance on normalized label embeddings.

## Install

```bash
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[dev,llm,sentence-transformers,benchmarks]
```

## Run with the dummy evaluator

```bash
python -m dialin_llm.cli run \
  --input data/sentences.csv \
  --text-col text \
  --embed tfidf \
  --clusterer kmeans \
  --candidate-ks 50,100,150,200 \
  --sample-size 20 \
  --sampler farthest \
  --epsilon 0.05 \
  --tmax 5 \
  --use-llm false \
  --out out/clusters.json
```

The command writes `clusters.json` and prints a summary JSON with cluster counts, remaining sentences, and iterations used.

## Enable a real LLM

1. Install the extra: `pip install -e .[llm]`
2. Set `OPENAI_API_KEY`
3. Run with `--use-llm true`
4. Optionally pass `--llm-model` and `--cache-path`

Example:

```bash
set OPENAI_API_KEY=your_key_here
python -m dialin_llm.cli run --input data/sentences.csv --text-col text --use-llm true --out out/clusters.json
```

## Use the paper datasets

The paper combines two kinds of data:

- an official `Dial-in-LLM` repository sample dataset,
- public intent benchmarks such as `BANKING77`, `CLINC150`, `MTOP`, and `MASSIVE`.

Important:

- the full 55,085-sentence Chinese customer-service dataset is not publicly exposed in the official repo,
- the official repo provides sample annotation files, not the private full corpus.

Install the benchmark extra before exporting Hugging Face datasets:

```bash
pip install -e .[benchmarks]
```

Export a public benchmark to the CSV format expected by the pipeline:

```bash
python -m dialin_llm.cli prepare-data \
  --source banking77 \
  --split train \
  --format csv \
  --out data/banking77_train.csv
```

Other benchmark examples:

```bash
python -m dialin_llm.cli prepare-data --source clinc150 --split train --format csv --out data/clinc150_train.csv
python -m dialin_llm.cli prepare-data --source mtop --config en --split train --format csv --out data/mtop_en_train.csv
python -m dialin_llm.cli prepare-data --source massive --config en-US --split train --format csv --out data/massive_en_us_train.csv
```

Export the official sample annotation files:

```bash
python -m dialin_llm.cli prepare-data --source dialin-labels --format csv --out data/data_labels.csv
python -m dialin_llm.cli prepare-data --source dialin-goodness --format jsonl --layout clusters --out data/sample_goodness.jsonl
python -m dialin_llm.cli prepare-data --source dialin-label --format jsonl --layout clusters --out data/sample_labels.jsonl
```

If you want sentence-level rows from the official sample files instead of cluster-level JSONL:

```bash
python -m dialin_llm.cli prepare-data \
  --source dialin-goodness \
  --format csv \
  --layout utterances \
  --out data/sample_goodness_utterances.csv
```

Then run the existing pipeline on an exported benchmark:

```bash
python -m dialin_llm.cli run \
  --input data/banking77_train.csv \
  --text-col text \
  --id-col sentence_id \
  --embed tfidf \
  --clusterer kmeans \
  --candidate-ks 50,100,150 \
  --out out/banking77_clusters.json
```

## Paper-faithful vs approximated

- Faithful: iterative sentence embedding, candidate-`K` evaluation, `Good / (Bad + 1)` selection, acceptance of only `Good` clusters, repeated removal of accepted sentences, and deterministic geodesic merge with threshold `theta`.
- Approximated: representative sampling uses random or farthest-first diverse sampling instead of an exact convex sampling routine.
- Approximated: dummy coherence and naming components are offline heuristics so the pipeline runs without network access.
- Optional assumption: the paper's probabilistic vMF merge gate is underspecified because `kappa` is not fixed. This implementation exposes it only as an optional extension and documents the assumed acceptance probability.

