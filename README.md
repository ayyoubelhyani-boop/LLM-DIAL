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
pip install -e .[dev,llm,local-llm,sentence-transformers]
```

For cluster visualization:

```bash
pip install -e .[viz]
```

## Run with the dummy evaluator

```bash
python -m dialin_llm.cli run   --input data/sentences.csv   --text-col text   --embed tfidf   --clusterer kmeans   --candidate-ks 50,100,150,200   --sample-size 20   --sampler farthest   --epsilon 0.05   --tmax 5   --use-llm false   --out out/clusters.json
```

The command writes `clusters.json` and prints a summary JSON with cluster counts, remaining sentences, and iterations used.

## Enable a real LLM

### OpenAI backend

1. Install the extra: `pip install -e .[llm]`
2. Set `OPENAI_API_KEY`
3. Run with `--use-llm true --llm-provider openai`
4. Optionally pass `--llm-model` and `--cache-path`

Example:

```bash
export OPENAI_API_KEY=your_key_here
python -m dialin_llm.cli run --input data/sentences.csv --text-col text --use-llm true --llm-provider openai --out out/clusters.json
```

### Local GPU backend

This project also supports a local Hugging Face LLM backend through `transformers`.

Recommended install:

```bash
pip install -e .[local-llm,sentence-transformers]
```

Example with a model that is realistic on a single ~20 GB GPU:

```bash
python -m dialin_llm.cli run   --input data/sentences.csv   --text-col text   --use-llm true   --llm-provider local   --local-llm-model mistralai/Mistral-7B-Instruct-v0.3   --local-llm-cache-dir .hf-cache   --out out/clusters.json
```

Example for a larger server-side model using quantization:

```bash
python -m dialin_llm.cli run   --input data/sentences.csv   --text-col text   --use-llm true   --llm-provider local   --local-llm-model mistralai/Mistral-Small-3.1-24B-Instruct-2503   --local-llm-quantization 4bit   --local-llm-cache-dir .hf-cache   --out out/clusters.json
```

Notes:

- `--local-llm-cache-dir` defaults to `.hf-cache`, so model files stay inside the project directory.
- To pin the run to the second GPU on a multi-GPU server, use `--local-llm-device-map cuda:1`.
- Plain `--local-llm-quantization none` now works on a single GPU even if `accelerate` is not installed.
- `--local-llm-quantization 4bit` or `8bit` still requires `accelerate` and `bitsandbytes`.
- For a strict ~20 GB VRAM target, a 7B instruct model is the honest default; Mistral Small 3.1 is a larger model and should be used quantized on the server.

## Run Notes

- `docs/SYNTHESE_BANKING77_FR.md`
- `docs/RUN_DEMO_FR.md`
- `docs/RUN_BANKING77_FR.md`
- `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_FR.md`
- `docs/RUN_BANKING77_LOCAL_MISTRAL_GPU1_ST_FR.md`
- `docs/COMPARAISON_BANKING77_FR.md`
- `docs/COMPARAISON_BANKING77_TOUTES_COMBINAISONS_FR.md`

## Plot clusters

You can render a 2D view of predicted clusters as a PNG with PCA:

```bash
python -m dialin_llm.cli plot-clusters --input data/banking77_train.csv --clusters out/banking77_clusters.json --id-col sentence_id --text-col text --embed sentence-transformers --out out/banking77_clusters.png
```

This keeps unassigned points in grey and annotates the largest clusters directly on the figure.

## Paper-faithful vs approximated

- Faithful: iterative sentence embedding, candidate-`K` evaluation, `Good / (Bad + 1)` selection, acceptance of only `Good` clusters, repeated removal of accepted sentences, and deterministic geodesic merge with threshold `theta`.
- Approximated: representative sampling uses random or farthest-first diverse sampling instead of an exact convex sampling routine.
- Approximated: dummy coherence and naming components are offline heuristics so the pipeline runs without network access.
- Optional assumption: the paper's probabilistic vMF merge gate is underspecified because `kappa` is not fixed. This implementation exposes it only as an optional extension and documents the assumed acceptance probability.
