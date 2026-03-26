from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_LOCAL_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    input_path: str
    candidate_ks: tuple[int, ...]
    paper_nmi_percent: float

    @property
    def candidate_ks_arg(self) -> str:
        return ",".join(str(value) for value in self.candidate_ks)


@dataclass(frozen=True)
class CampaignSpec:
    key: str
    display_name: str
    llm_prompt_style: str
    naming_sampler: str
    candidate_k_policy: str
    candidate_k_min: int


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="bank77",
        display_name="Bank77",
        input_path="data/banking77_test.csv",
        candidate_ks=(50, 77, 100),
        paper_nmi_percent=82.32,
    ),
    BenchmarkSpec(
        key="clinc150",
        display_name="CLINC(I)",
        input_path="data/clinc150_test.csv",
        candidate_ks=(100, 150, 200),
        paper_nmi_percent=94.12,
    ),
    BenchmarkSpec(
        key="mtop",
        display_name="MTOP(I)",
        input_path="data/mtop_en_test.csv",
        candidate_ks=(80, 102, 120),
        paper_nmi_percent=72.45,
    ),
    BenchmarkSpec(
        key="massive",
        display_name="Massive(I)",
        input_path="data/massive_en_us_test.csv",
        candidate_ks=(40, 59, 80),
        paper_nmi_percent=78.12,
    ),
)

CAMPAIGN_SPECS: tuple[CampaignSpec, ...] = (
    CampaignSpec(
        key="reproduction",
        display_name="Reproduction",
        llm_prompt_style="simple",
        naming_sampler="head",
        candidate_k_policy="fixed",
        candidate_k_min=2,
    ),
    CampaignSpec(
        key="improved",
        display_name="Improved",
        llm_prompt_style="benchmark",
        naming_sampler="centroid",
        candidate_k_policy="sqrt",
        candidate_k_min=2,
    ),
)

BENCHMARKS_BY_KEY = {spec.key: spec for spec in BENCHMARK_SPECS}
CAMPAIGNS_BY_KEY = {spec.key: spec for spec in CAMPAIGN_SPECS}


def resolve_benchmark_specs(selection: str) -> list[BenchmarkSpec]:
    normalized = selection.strip().lower()
    if normalized == "all":
        return list(BENCHMARK_SPECS)
    resolved: list[BenchmarkSpec] = []
    for raw in normalized.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in BENCHMARKS_BY_KEY:
            raise ValueError(f"Unknown benchmark: {key}")
        resolved.append(BENCHMARKS_BY_KEY[key])
    if not resolved:
        raise ValueError("At least one benchmark must be selected")
    return resolved


def resolve_campaign_specs(selection: str) -> list[CampaignSpec]:
    normalized = selection.strip().lower()
    if normalized == "both":
        return list(CAMPAIGN_SPECS)
    resolved: list[CampaignSpec] = []
    for raw in normalized.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in CAMPAIGNS_BY_KEY:
            raise ValueError(f"Unknown campaign config: {key}")
        resolved.append(CAMPAIGNS_BY_KEY[key])
    if not resolved:
        raise ValueError("At least one campaign config must be selected")
    return resolved


def resolve_local_llm_device_map(
    *,
    local_llm_device_map: str | None,
    cuda_visible_devices: str | None,
) -> str:
    if local_llm_device_map:
        return local_llm_device_map
    if cuda_visible_devices:
        # Once a single physical GPU is masked via CUDA_VISIBLE_DEVICES,
        # the selected device becomes logical cuda:0 inside the process.
        return "cuda:0"
    return "cuda:1"


def build_repo_local_env(
    repo_root: Path,
    *,
    cuda_visible_devices: str | None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.update(
        {
            "HF_HOME": str(repo_root / ".hf-home"),
            "HF_HUB_CACHE": str(repo_root / ".hf-cache" / "hub"),
            "TRANSFORMERS_CACHE": str(repo_root / ".hf-cache" / "transformers"),
            "SENTENCE_TRANSFORMERS_HOME": str(repo_root / ".hf-cache" / "sentence-transformers"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def ensure_repo_local_cache_dirs(repo_root: Path) -> None:
    for path in (
        repo_root / ".hf-home",
        repo_root / ".hf-cache",
        repo_root / ".hf-cache" / "hub",
        repo_root / ".hf-cache" / "transformers",
        repo_root / ".hf-cache" / "sentence-transformers",
        repo_root / ".hf-cache" / "local-llm",
    ):
        path.mkdir(parents=True, exist_ok=True)


def build_artifact_paths(
    output_root: Path,
    *,
    benchmark: BenchmarkSpec,
    campaign: CampaignSpec,
) -> dict[str, Path]:
    root = output_root / campaign.key / benchmark.key
    return {
        "root": root,
        "clusters": root / "clusters.json",
        "summary": root / "summary.json",
        "config": root / "config.json",
        "evaluation": root / "evaluation.json",
        "cache": root / "cache.json",
        "run_log": root / "run.log",
        "eval_log": root / "evaluate.log",
    }


def build_run_command(
    *,
    python_executable: str,
    repo_root: Path,
    output_root: Path,
    benchmark: BenchmarkSpec,
    campaign: CampaignSpec,
    local_llm_device_map: str,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    local_llm_model: str = DEFAULT_LOCAL_LLM_MODEL,
    include_sentences: bool = True,
) -> list[str]:
    artifacts = build_artifact_paths(output_root, benchmark=benchmark, campaign=campaign)
    return [
        python_executable,
        "-m",
        "dialin_llm.cli",
        "run",
        "--input",
        benchmark.input_path,
        "--text-col",
        "text",
        "--id-col",
        "sentence_id",
        "--dedupe",
        "false",
        "--embed",
        "sentence-transformers",
        "--sentence-transformer-model",
        sentence_transformer_model,
        "--sentence-transformer-cache-dir",
        ".hf-cache/sentence-transformers",
        "--clusterer",
        "minibatch",
        "--candidate-ks",
        benchmark.candidate_ks_arg,
        "--candidate-k-policy",
        campaign.candidate_k_policy,
        "--candidate-k-min",
        str(campaign.candidate_k_min),
        "--sample-size",
        "10",
        "--sampler",
        "random",
        "--epsilon",
        "0.0",
        "--tmax",
        "5",
        "--theta",
        "0.8",
        "--merge-strategy",
        "label",
        "--merge-label-weight",
        "0.5",
        "--seed",
        "0",
        "--use-llm",
        "true",
        "--llm-provider",
        "local",
        "--local-llm-model",
        local_llm_model,
        "--local-llm-device-map",
        local_llm_device_map,
        "--local-llm-max-new-tokens",
        "32",
        "--local-llm-temperature",
        "0.0",
        "--local-llm-cache-dir",
        ".hf-cache/local-llm",
        "--local-llm-trust-remote-code",
        "false",
        "--cache-path",
        _relative_to_root(artifacts["cache"], repo_root),
        "--llm-prompt-style",
        campaign.llm_prompt_style,
        "--naming-sample-size",
        "10",
        "--naming-sampler",
        campaign.naming_sampler,
        "--summary-out",
        _relative_to_root(artifacts["summary"], repo_root),
        "--config-out",
        _relative_to_root(artifacts["config"], repo_root),
        "--include-sentences",
        "true" if include_sentences else "false",
        "--out",
        _relative_to_root(artifacts["clusters"], repo_root),
    ]


def build_evaluate_command(
    *,
    python_executable: str,
    repo_root: Path,
    output_root: Path,
    benchmark: BenchmarkSpec,
    campaign: CampaignSpec,
) -> list[str]:
    artifacts = build_artifact_paths(output_root, benchmark=benchmark, campaign=campaign)
    return [
        python_executable,
        "-m",
        "dialin_llm.cli",
        "evaluate",
        "--input",
        benchmark.input_path,
        "--clusters",
        _relative_to_root(artifacts["clusters"], repo_root),
        "--text-col",
        "text",
        "--id-col",
        "sentence_id",
        "--label-col",
        "label",
        "--out",
        _relative_to_root(artifacts["evaluation"], repo_root),
    ]


def collect_run_record(
    *,
    repo_root: Path,
    output_root: Path,
    benchmark: BenchmarkSpec,
    campaign: CampaignSpec,
) -> dict[str, Any]:
    artifacts = build_artifact_paths(output_root, benchmark=benchmark, campaign=campaign)
    summary = _load_json(artifacts["summary"])
    evaluation = _load_json(artifacts["evaluation"])
    config = _load_json(artifacts["config"])

    return {
        "benchmark_key": benchmark.key,
        "benchmark": benchmark.display_name,
        "campaign_key": campaign.key,
        "campaign": campaign.display_name,
        "paper_nmi_percent": benchmark.paper_nmi_percent,
        "nmi_percent": _round_percent(evaluation["with_unassigned"]["nmi"]),
        "ari_percent": _round_percent(evaluation["with_unassigned"]["ari"]),
        "coverage_percent": _round_percent(evaluation["coverage"]),
        "num_clusters": int(summary["num_clusters"]),
        "num_remaining": int(summary["num_remaining"]),
        "iterations_used": int(summary["iterations_used"]),
        "config": config,
        "artifacts": {
            "clusters": _relative_to_root(artifacts["clusters"], repo_root),
            "summary": _relative_to_root(artifacts["summary"], repo_root),
            "evaluation": _relative_to_root(artifacts["evaluation"], repo_root),
            "config": _relative_to_root(artifacts["config"], repo_root),
            "cache": _relative_to_root(artifacts["cache"], repo_root),
            "run_log": _relative_to_root(artifacts["run_log"], repo_root),
            "eval_log": _relative_to_root(artifacts["eval_log"], repo_root),
        },
    }


def build_comparison_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        by_key[(str(record["benchmark_key"]), str(record["campaign_key"]))] = record

    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_SPECS:
        reproduction = by_key.get((benchmark.key, "reproduction"))
        improved = by_key.get((benchmark.key, "improved"))

        reproduction_nmi = _maybe_float(reproduction, "nmi_percent")
        improved_nmi = _maybe_float(improved, "nmi_percent")
        best_campaign = _best_campaign_key(reproduction_nmi, improved_nmi)

        rows.append(
            {
                "benchmark": benchmark.display_name,
                "paper_nmi_percent": benchmark.paper_nmi_percent,
                "reproduction_nmi_percent": reproduction_nmi,
                "improved_nmi_percent": improved_nmi,
                "gain_improved_vs_reproduction": _delta(improved_nmi, reproduction_nmi),
                "delta_reproduction_vs_paper": _delta(reproduction_nmi, benchmark.paper_nmi_percent),
                "delta_improved_vs_paper": _delta(improved_nmi, benchmark.paper_nmi_percent),
                "best_config": best_campaign,
                "reproduction_coverage_percent": _maybe_float(reproduction, "coverage_percent"),
                "improved_coverage_percent": _maybe_float(improved, "coverage_percent"),
                "reproduction_num_clusters": _maybe_int(reproduction, "num_clusters"),
                "improved_num_clusters": _maybe_int(improved, "num_clusters"),
                "reproduction_num_remaining": _maybe_int(reproduction, "num_remaining"),
                "improved_num_remaining": _maybe_int(improved, "num_remaining"),
            }
        )
    return rows


def run_campaign(
    *,
    repo_root: Path,
    output_root: Path,
    benchmarks: Sequence[BenchmarkSpec],
    campaigns: Sequence[CampaignSpec],
    python_executable: str = sys.executable,
    cuda_visible_devices: str | None = None,
    local_llm_device_map: str | None = None,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    local_llm_model: str = DEFAULT_LOCAL_LLM_MODEL,
    skip_existing: bool = True,
    dry_run: bool = False,
    include_sentences: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = (repo_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    ensure_repo_local_cache_dirs(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    effective_device_map = resolve_local_llm_device_map(
        local_llm_device_map=local_llm_device_map,
        cuda_visible_devices=cuda_visible_devices,
    )
    env = build_repo_local_env(repo_root, cuda_visible_devices=cuda_visible_devices)

    manifest = {
        "repo_root": str(repo_root),
        "output_root": _relative_or_absolute(output_root, repo_root),
        "python_executable": python_executable,
        "cuda_visible_devices": cuda_visible_devices,
        "local_llm_device_map": effective_device_map,
        "sentence_transformer_model": sentence_transformer_model,
        "local_llm_model": local_llm_model,
        "benchmarks": [benchmark.key for benchmark in benchmarks],
        "campaigns": [campaign.key for campaign in campaigns],
        "skip_existing": skip_existing,
        "dry_run": dry_run,
    }
    _write_json(output_root / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    dry_run_commands: list[dict[str, Any]] = []

    for campaign in campaigns:
        for benchmark in benchmarks:
            artifacts = build_artifact_paths(output_root, benchmark=benchmark, campaign=campaign)
            artifacts["root"].mkdir(parents=True, exist_ok=True)
            run_command = build_run_command(
                python_executable=python_executable,
                repo_root=repo_root,
                output_root=output_root,
                benchmark=benchmark,
                campaign=campaign,
                local_llm_device_map=effective_device_map,
                sentence_transformer_model=sentence_transformer_model,
                local_llm_model=local_llm_model,
                include_sentences=include_sentences,
            )
            eval_command = build_evaluate_command(
                python_executable=python_executable,
                repo_root=repo_root,
                output_root=output_root,
                benchmark=benchmark,
                campaign=campaign,
            )

            if dry_run:
                dry_run_commands.append(
                    {
                        "benchmark": benchmark.key,
                        "campaign": campaign.key,
                        "run_command": run_command,
                        "evaluate_command": eval_command,
                    }
                )
                continue

            if skip_existing and artifacts["summary"].exists() and artifacts["evaluation"].exists() and artifacts["config"].exists():
                records.append(
                    collect_run_record(
                        repo_root=repo_root,
                        output_root=output_root,
                        benchmark=benchmark,
                        campaign=campaign,
                    )
                )
                continue

            _run_logged_command(run_command, cwd=repo_root, env=env, log_path=artifacts["run_log"])
            _run_logged_command(eval_command, cwd=repo_root, env=env, log_path=artifacts["eval_log"])
            records.append(
                collect_run_record(
                    repo_root=repo_root,
                    output_root=output_root,
                    benchmark=benchmark,
                    campaign=campaign,
                )
            )

    if dry_run:
        payload = {"manifest": manifest, "dry_run_commands": dry_run_commands}
        _write_json(output_root / "dry_run_commands.json", payload)
        return payload

    records = sorted(records, key=lambda item: (_benchmark_sort_key(str(item["benchmark_key"])), str(item["campaign_key"])))
    comparison_rows = build_comparison_rows(records)
    payload = {
        "manifest": manifest,
        "records": records,
        "comparison": comparison_rows,
    }
    _write_json(output_root / "records.json", records)
    _write_json(output_root / "comparison.json", comparison_rows)
    _write_json(output_root / "summary_bundle.json", payload)
    _write_csv(output_root / "comparison.csv", comparison_rows)
    return payload


def _run_logged_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {_format_shell_command(command)}\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        handle.write("\n")


def _format_shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _relative_to_root(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    return _relative_to_root(path, repo_root)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "paper_nmi_percent",
        "reproduction_nmi_percent",
        "improved_nmi_percent",
        "gain_improved_vs_reproduction",
        "delta_reproduction_vs_paper",
        "delta_improved_vs_paper",
        "best_config",
        "reproduction_coverage_percent",
        "improved_coverage_percent",
        "reproduction_num_clusters",
        "improved_num_clusters",
        "reproduction_num_remaining",
        "improved_num_remaining",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _round_percent(value: float) -> float:
    return round(float(value) * 100.0, 2)


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return round(lhs - rhs, 2)


def _benchmark_sort_key(key: str) -> int:
    for index, benchmark in enumerate(BENCHMARK_SPECS):
        if benchmark.key == key:
            return index
    return len(BENCHMARK_SPECS)


def _best_campaign_key(reproduction_nmi: float | None, improved_nmi: float | None) -> str | None:
    if reproduction_nmi is None and improved_nmi is None:
        return None
    if improved_nmi is None:
        return "reproduction"
    if reproduction_nmi is None:
        return "improved"
    return "improved" if improved_nmi > reproduction_nmi else "reproduction"


def _maybe_float(record: Mapping[str, Any] | None, key: str) -> float | None:
    if record is None:
        return None
    value = record.get(key)
    if value is None:
        return None
    return float(value)


def _maybe_int(record: Mapping[str, Any] | None, key: str) -> int | None:
    if record is None:
        return None
    value = record.get(key)
    if value is None:
        return None
    return int(value)
