# Prompt Agent Pour Executer Sur Le Serveur

Ce document fournit un prompt pret a copier-coller pour demander a un agent de lancer les experimentations sur le serveur distant, sans toucher d'autres repertoires.

Important :

- ne pas stocker les identifiants du serveur dans le depot ;
- fournir l'URL et les identifiants a l'agent au moment de l'execution ;
- limiter tout le travail distant a `~/nfs/LLM-DIAL`.

## Prompt principal

```text
You are working on the remote server that hosts the project in ~/nfs/LLM-DIAL.

Execution constraints:
- Never touch any directory outside ~/nfs/LLM-DIAL.
- Run every command from ~/nfs/LLM-DIAL unless a command explicitly needs another working directory.
- If the machine has two GPUs, use the second one. Prefer CUDA_VISIBLE_DEVICES=1 so the process only sees that GPU.
- Keep Hugging Face caches inside the project, not in the global home directory.
- Before editing files, inspect the repo state and existing docs.
- After each substantial step, summarize what changed and what was produced.
- Do not use GitHub as a source of truth for the project state. The source of truth is the remote working tree in ~/nfs/LLM-DIAL.

Your job:
1. Verify the hardware and environment.
2. Confirm whether there are two GPUs and record the GPU model and VRAM.
3. Check that the project CLI works.
4. Run the requested clustering or evaluation pipeline.
5. Save outputs under ~/nfs/LLM-DIAL/out.
6. Add or update documentation under ~/nfs/LLM-DIAL/docs with:
   - exact command used,
   - configuration,
   - runtime context,
   - results,
   - interpretation.
7. Do not remove existing outputs or docs unless explicitly asked.
8. At the end, report:
   - files created or modified,
   - key metrics,
   - any blocker or limitation.

Use these checks first:
- pwd
- cd ~/nfs/LLM-DIAL && git status --short
- cd ~/nfs/LLM-DIAL && python3 -m dialin_llm.cli run --help
- nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

Use this environment pattern for local Hugging Face runs:
- cd ~/nfs/LLM-DIAL
- export HF_HOME=$PWD/.hf-home
- export HUGGINGFACE_HUB_CACHE=$PWD/.hf-home/hub
- export TRANSFORMERS_CACHE=$PWD/.hf-cache

If two GPUs are present, run with:
- export CUDA_VISIBLE_DEVICES=1

Because CUDA_VISIBLE_DEVICES=1 remaps the visible GPU, use:
- --local-llm-device-map cuda:0

If sentence-transformers is required, prefer:
- --embed sentence-transformers

If the goal is maximum coverage on BANKING77, start from:
- evaluator: dummy
- embed: sentence-transformers
- clusterer: minibatch
- candidate-ks: 60,77,90
- sample-size: 10
- sampler: farthest
- epsilon: 0.0
- tmax: 4

If the goal is a more paper-faithful LLM-in-the-loop run, start from:
- evaluator: local
- local model: mistralai/Mistral-7B-Instruct-v0.3
- embed: sentence-transformers
- clusterer: minibatch
- candidate-ks: 77
- sample-size: 10
- sampler: farthest
- epsilon: 0.0
- tmax: 2
- --local-llm-cache-dir .hf-cache

After any clustering run, also execute the evaluation command against the benchmark labels and report:
- coverage
- assigned_only.nmi
- assigned_only.ari
- assigned_only.v_measure
- with_unassigned.nmi
- with_unassigned.ari
- with_unassigned.v_measure

Use the existing CLI commands already present in the repo:
- python3 -m dialin_llm.cli run ...
- python3 -m dialin_llm.cli evaluate ...
- python3 -m dialin_llm.cli plot-clusters ...

Keep all generated artifacts under:
- ~/nfs/LLM-DIAL/out

Keep all run notes under:
- ~/nfs/LLM-DIAL/docs
```

## Prompt specialise pour un run de couverture maximale

```text
Run a heavy BANKING77 benchmark from ~/nfs/LLM-DIAL with the goal of maximizing coverage.

Constraints:
- Never touch directories outside ~/nfs/LLM-DIAL.
- If two GPUs are available, reserve the second one with CUDA_VISIBLE_DEVICES=1.
- Keep Hugging Face caches inside the project.

Steps:
1. Check git status, CLI availability, and GPU state.
2. Run:
   python3 -m dialin_llm.cli run \
     --input data/banking77_train.csv \
     --text-col text \
     --id-col sentence_id \
     --embed sentence-transformers \
     --clusterer minibatch \
     --candidate-ks 60,77,90 \
     --sample-size 10 \
     --sampler farthest \
     --epsilon 0.0 \
     --tmax 4 \
     --use-llm false \
     --summary-out out/banking77_dummy_st_tmax4_summary.json \
     --out out/banking77_dummy_st_tmax4_clusters.json
3. Evaluate:
   python3 -m dialin_llm.cli evaluate \
     --input data/banking77_train.csv \
     --clusters out/banking77_dummy_st_tmax4_clusters.json \
     --id-col sentence_id \
     --text-col text \
     --label-col label \
     --out out/banking77_dummy_st_tmax4_evaluation.json
4. Optionally plot:
   python3 -m dialin_llm.cli plot-clusters \
     --input data/banking77_train.csv \
     --clusters out/banking77_dummy_st_tmax4_clusters.json \
     --id-col sentence_id \
     --text-col text \
     --embed sentence-transformers \
     --max-points 3000 \
     --title "BANKING77 - dummy + sentence-transformers + tmax=4" \
     --out out/banking77_dummy_st_tmax4_plot.png
5. Create or update a run note in docs with the command, outputs, metrics, and interpretation.
6. Return a concise summary of the produced files and the final metrics.
```

## Prompt specialise pour un run avec LLM local

```text
Run a BANKING77 benchmark from ~/nfs/LLM-DIAL with a local Mistral model on the second GPU if available.

Constraints:
- Never touch directories outside ~/nfs/LLM-DIAL.
- Use CUDA_VISIBLE_DEVICES=1 if two GPUs are present.
- Keep Hugging Face caches inside the project.
- If CUDA_VISIBLE_DEVICES=1 is set, use --local-llm-device-map cuda:0.

Steps:
1. Check git status, CLI availability, and GPU state.
2. Export:
   export HF_HOME=$PWD/.hf-home
   export HUGGINGFACE_HUB_CACHE=$PWD/.hf-home/hub
   export TRANSFORMERS_CACHE=$PWD/.hf-cache
   export CUDA_VISIBLE_DEVICES=1
3. Run:
   python3 -m dialin_llm.cli run \
     --input data/banking77_train.csv \
     --text-col text \
     --id-col sentence_id \
     --embed sentence-transformers \
     --clusterer minibatch \
     --candidate-ks 77 \
     --sample-size 10 \
     --sampler farthest \
     --epsilon 0.0 \
     --tmax 2 \
     --use-llm true \
     --llm-provider local \
     --local-llm-model mistralai/Mistral-7B-Instruct-v0.3 \
     --local-llm-device-map cuda:0 \
     --local-llm-cache-dir .hf-cache \
     --summary-out out/banking77_mistral_st_tmax2_summary.json \
     --out out/banking77_mistral_st_tmax2_clusters.json
4. Evaluate the resulting clusters with the CLI evaluate command.
5. Optionally create a cluster plot.
6. Update docs with the exact hardware context and final metrics.
```

## Ce qu'un bon compte rendu agent doit contenir

- le repertoire de travail utilise ;
- la configuration exacte ;
- le materiel detecte ;
- la commande executee ;
- les chemins de sortie ;
- les metriques principales ;
- les limites rencontrees ;
- les fichiers modifies dans `docs/`, `out/`, et eventuellement `README.md`.
