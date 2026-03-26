from __future__ import annotations

from pathlib import Path

from dialin_llm.final_campaign import (
    BENCHMARKS_BY_KEY,
    CAMPAIGNS_BY_KEY,
    build_comparison_rows,
    build_run_command,
    build_repo_local_env,
    resolve_local_llm_device_map,
)


def test_build_run_command_for_improved_campaign_exposes_expected_flags() -> None:
    repo_root = Path("/repo")
    output_root = repo_root / "out" / "final_campaign"
    command = build_run_command(
        python_executable="python",
        repo_root=repo_root,
        output_root=output_root,
        benchmark=BENCHMARKS_BY_KEY["bank77"],
        campaign=CAMPAIGNS_BY_KEY["improved"],
        local_llm_device_map="cuda:0",
        include_sentences=False,
    )

    assert "--candidate-ks" in command
    assert command[command.index("--candidate-ks") + 1] == "50,77,100"
    assert command[command.index("--candidate-k-policy") + 1] == "sqrt"
    assert command[command.index("--llm-prompt-style") + 1] == "benchmark"
    assert command[command.index("--naming-sampler") + 1] == "centroid"
    assert command[command.index("--local-llm-device-map") + 1] == "cuda:0"
    assert Path(command[command.index("--out") + 1]).parts[-5:] == (
        "out",
        "final_campaign",
        "improved",
        "bank77",
        "clusters.json",
    )
    assert command[command.index("--include-sentences") + 1] == "false"


def test_build_comparison_rows_prefers_best_available_config() -> None:
    rows = build_comparison_rows(
        [
            {
                "benchmark_key": "bank77",
                "campaign_key": "reproduction",
                "nmi_percent": 83.78,
                "coverage_percent": 98.77,
                "num_clusters": 142,
                "num_remaining": 38,
            },
            {
                "benchmark_key": "bank77",
                "campaign_key": "improved",
                "nmi_percent": 82.28,
                "coverage_percent": 100.0,
                "num_clusters": 49,
                "num_remaining": 0,
            },
            {
                "benchmark_key": "mtop",
                "campaign_key": "reproduction",
                "nmi_percent": 71.12,
                "coverage_percent": 96.26,
                "num_clusters": 185,
                "num_remaining": 164,
            },
            {
                "benchmark_key": "mtop",
                "campaign_key": "improved",
                "nmi_percent": 73.08,
                "coverage_percent": 94.53,
                "num_clusters": 119,
                "num_remaining": 240,
            },
        ]
    )

    bank77_row = next(row for row in rows if row["benchmark"] == "Bank77")
    assert bank77_row["best_config"] == "reproduction"
    assert bank77_row["gain_improved_vs_reproduction"] == -1.5
    assert bank77_row["delta_improved_vs_paper"] == -0.04

    mtop_row = next(row for row in rows if row["benchmark"] == "MTOP(I)")
    assert mtop_row["best_config"] == "improved"
    assert mtop_row["gain_improved_vs_reproduction"] == 1.96
    assert mtop_row["delta_improved_vs_paper"] == 0.63


def test_resolve_local_llm_device_map_tracks_cuda_visible_devices() -> None:
    assert resolve_local_llm_device_map(local_llm_device_map=None, cuda_visible_devices="1") == "cuda:0"
    assert resolve_local_llm_device_map(local_llm_device_map=None, cuda_visible_devices=None) == "cuda:1"
    assert resolve_local_llm_device_map(local_llm_device_map="cuda:3", cuda_visible_devices="1") == "cuda:3"


def test_build_repo_local_env_keeps_hf_state_inside_repo() -> None:
    repo_root = Path("/repo")
    env = build_repo_local_env(repo_root, cuda_visible_devices="1", base_env={"PATH": "x"})

    assert Path(env["HF_HOME"]).parts[-2:] == ("repo", ".hf-home")
    assert Path(env["HF_HUB_CACHE"]).parts[-3:] == ("repo", ".hf-cache", "hub")
    assert Path(env["TRANSFORMERS_CACHE"]).parts[-3:] == ("repo", ".hf-cache", "transformers")
    assert Path(env["SENTENCE_TRANSFORMERS_HOME"]).parts[-3:] == ("repo", ".hf-cache", "sentence-transformers")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PATH"] == "x"
