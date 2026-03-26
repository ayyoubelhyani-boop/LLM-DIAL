from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialin_llm.final_campaign import (
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    resolve_benchmark_specs,
    resolve_campaign_specs,
    run_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final paper-vs-reproduction-vs-improved benchmark campaign.")
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated benchmark keys (bank77,clinc150,mtop,massive) or 'all'",
    )
    parser.add_argument(
        "--configs",
        default="both",
        help="Comma-separated config keys (reproduction,improved) or 'both'",
    )
    parser.add_argument(
        "--output-root",
        default="out/final_campaign",
        help="Output directory relative to the repo root unless absolute",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to invoke the CLI runs",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES mask. Use '1' to pin the whole campaign to the second physical GPU.",
    )
    parser.add_argument(
        "--local-llm-device-map",
        default=None,
        help="Optional explicit device_map passed to the local LLM backend. Defaults to cuda:0 when CUDA_VISIBLE_DEVICES is set, otherwise cuda:1.",
    )
    parser.add_argument(
        "--sentence-transformer-model",
        default=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        help="SentenceTransformer model used by the campaign",
    )
    parser.add_argument(
        "--local-llm-model",
        default=DEFAULT_LOCAL_LLM_MODEL,
        help="Local Hugging Face LLM used by the campaign",
    )
    parser.add_argument(
        "--skip-existing",
        default="true",
        help="Reuse existing run artifacts when summary/evaluation/config files are already present",
    )
    parser.add_argument(
        "--include-sentences",
        default="true",
        help="Whether to keep raw cluster sentences in cluster output payloads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the commands that would be executed without running the campaign",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    payload = run_campaign(
        repo_root=repo_root,
        output_root=Path(args.output_root),
        benchmarks=resolve_benchmark_specs(args.benchmarks),
        campaigns=resolve_campaign_specs(args.configs),
        python_executable=args.python_executable,
        cuda_visible_devices=_parse_optional_text(args.cuda_visible_devices),
        local_llm_device_map=_parse_optional_text(args.local_llm_device_map),
        sentence_transformer_model=args.sentence_transformer_model,
        local_llm_model=args.local_llm_model,
        skip_existing=_parse_bool(args.skip_existing),
        dry_run=args.dry_run,
        include_sentences=_parse_bool(args.include_sentences),
    )
    print(json.dumps(payload, indent=2))


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


def _parse_optional_text(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


if __name__ == "__main__":
    main()
