from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .embeddings import build_embedding_backend
from .io import load_sentences
from .iterative import ClusterRunResult, IntentCluster, run_iterative_clustering
from .llm_utils import (
    DummyClusterNamer,
    DummyCoherenceEvaluator,
    JsonCache,
    OpenAIClusterNamer,
    OpenAICoherenceEvaluator,
)
from .merge import merge_clusters_by_label, name_clusters
from .paper_data import HF_BENCHMARKS, OFFICIAL_SAMPLE_URLS, export_hf_benchmark, export_official_sample


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_command(args)
        return
    if args.command == "prepare-data":
        prepare_data_command(args)
        return
    parser.error("A subcommand is required")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dial-In LLM iterative intent clustering")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run iterative intent clustering")
    run_parser.add_argument("--input", required=True, help="Path to CSV or JSONL input")
    run_parser.add_argument("--text-col", default="text", help="Input text column")
    run_parser.add_argument("--id-col", default=None, help="Optional sentence ID column")
    run_parser.add_argument("--embed", default="tfidf", help="Embedding backend")
    run_parser.add_argument("--clusterer", default="kmeans", help="Clusterer: kmeans or minibatch")
    run_parser.add_argument("--candidate-ks", required=True, help="Comma-separated candidate K values")
    run_parser.add_argument("--sample-size", type=int, default=20, help="Representative sample size")
    run_parser.add_argument("--sampler", default="farthest", help="Sampler: random or farthest")
    run_parser.add_argument("--epsilon", type=float, default=0.05, help="Stop when remaining fraction <= epsilon")
    run_parser.add_argument("--tmax", type=int, default=5, help="Maximum iterations")
    run_parser.add_argument("--theta", type=float, default=0.8, help="Geodesic merge threshold")
    run_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    run_parser.add_argument("--use-llm", default="false", help="Whether to use OpenAI-backed evaluator/namer")
    run_parser.add_argument("--llm-model", default="gpt-4.1-mini", help="OpenAI model name")
    run_parser.add_argument("--cache-path", default=None, help="Optional cache JSON path for LLM calls")
    run_parser.add_argument("--summary-out", default=None, help="Optional summary JSON output path")
    run_parser.add_argument("--out", required=True, help="Output JSON path for clusters")
    run_parser.add_argument(
        "--include-sentences",
        default="true",
        help="Whether to include raw sentences in cluster output",
    )

    prepare_parser = subparsers.add_parser("prepare-data", help="Export paper datasets into CSV or JSONL")
    prepare_parser.add_argument(
        "--source",
        required=True,
        choices=sorted([*HF_BENCHMARKS.keys(), *OFFICIAL_SAMPLE_URLS.keys()]),
        help="Dataset source to export",
    )
    prepare_parser.add_argument("--out", required=True, help="Output CSV or JSONL path")
    prepare_parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "jsonl"],
        help="Output format",
    )
    prepare_parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face split to export, for example train, validation, or test",
    )
    prepare_parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config or language, for example en for MTOP or en-US for MASSIVE",
    )
    prepare_parser.add_argument(
        "--layout",
        default="clusters",
        choices=["clusters", "utterances"],
        help="Official sample export shape. HF benchmarks always export utterances.",
    )
    return parser


def run_command(args: argparse.Namespace) -> None:
    candidate_ks = _parse_candidate_ks(args.candidate_ks)
    use_llm = _parse_bool(args.use_llm)
    include_sentences = _parse_bool(args.include_sentences)

    records = load_sentences(args.input, text_col=args.text_col, id_col=args.id_col, dedupe=True)
    if not records:
        raise ValueError("No sentences were loaded from the input file")

    embedder = build_embedding_backend(args.embed)
    embeddings = embedder.fit_transform([record.text for record in records])

    if use_llm:
        cache = JsonCache(args.cache_path)
        evaluator = OpenAICoherenceEvaluator(model=args.llm_model, cache=cache)
        namer = OpenAIClusterNamer(model=args.llm_model, cache=cache)
    else:
        evaluator = DummyCoherenceEvaluator()
        namer = DummyClusterNamer()

    run_result = run_iterative_clustering(
        records,
        embeddings,
        candidate_ks=candidate_ks,
        evaluator=evaluator,
        clusterer=args.clusterer,
        sample_size=args.sample_size,
        sampler=args.sampler,
        epsilon=args.epsilon,
        tmax=args.tmax,
        seed=args.seed,
    )
    named_clusters = name_clusters(run_result.clusters, namer, sample_size=args.sample_size)
    merged_clusters = merge_clusters_by_label(named_clusters, theta=args.theta, seed=args.seed)

    summary = _build_summary(run_result, merged_clusters)
    cluster_payload = [_cluster_to_dict(cluster, include_sentences=include_sentences) for cluster in merged_clusters]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(cluster_payload, handle, indent=2)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def prepare_data_command(args: argparse.Namespace) -> None:
    source = args.source.strip().lower()
    out_path = Path(args.out)
    if source in HF_BENCHMARKS:
        exported = export_hf_benchmark(
            source,
            out_path,
            split=args.split,
            config=args.config,
            output_format=args.format,
        )
    else:
        exported = export_official_sample(
            source,
            out_path,
            output_format=args.format,
            layout=args.layout,
        )

    print(json.dumps({"source": source, "output_path": str(exported)}, indent=2))


def _build_summary(result: ClusterRunResult, merged_clusters: list[IntentCluster]) -> dict[str, object]:
    num_good_clusters = sum(
        candidate.good_clusters
        for item in result.iterations
        for candidate in item.candidate_results
        if candidate.k == item.selected_k
    )
    return {
        "num_clusters": len(merged_clusters),
        "num_good_clusters": num_good_clusters,
        "num_remaining": len(result.remaining_sentence_ids),
        "iterations_used": result.iterations_used,
        "remaining_sentence_ids": result.remaining_sentence_ids,
        "iteration_summaries": [asdict(summary) for summary in result.iterations],
    }


def _cluster_to_dict(cluster: IntentCluster, *, include_sentences: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "cluster_id": cluster.cluster_id,
        "member_sentence_ids": cluster.member_sentence_ids,
        "label": cluster.label,
        "iteration_found": cluster.iteration_found,
        "source_cluster_ids": cluster.source_cluster_ids,
    }
    if include_sentences:
        payload["sentences"] = cluster.sentences
    return payload


def _parse_candidate_ks(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --candidate-ks value: {raw}") from exc
    if not values:
        raise ValueError("--candidate-ks must contain at least one integer")
    return values


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


if __name__ == "__main__":
    main()
