import csv
import json
import subprocess
import sys
from pathlib import Path

from dialin_llm.io import SentenceRecord
from dialin_llm.metrics import evaluate_clustering, load_cluster_memberships


def test_evaluate_clustering_reports_assigned_and_unassigned_metrics() -> None:
    records = [
        SentenceRecord("s1", "reset password", {"label": "password"}),
        SentenceRecord("s2", "forgot password", {"label": "password"}),
        SentenceRecord("s3", "change password", {"label": "password"}),
    ]
    memberships = {"s1": "cluster-0", "s2": "cluster-0"}

    report = evaluate_clustering(records, memberships)

    assert report["num_records"] == 3
    assert report["num_assigned"] == 2
    assert report["num_unassigned"] == 1
    assert report["coverage"] == 2 / 3
    assert report["assigned_only"]["nmi"] == 1.0
    assert report["assigned_only"]["ari"] == 1.0
    assert report["assigned_only"]["v_measure"] == 1.0
    assert report["with_unassigned"]["nmi"] < 1.0


def test_load_cluster_memberships_rejects_duplicate_sentence_ids(tmp_path: Path) -> None:
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps(
            [
                {"cluster_id": "cluster-0", "member_sentence_ids": ["s1"]},
                {"cluster_id": "cluster-1", "member_sentence_ids": ["s1"]},
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_cluster_memberships(cluster_path)
    except ValueError as exc:
        assert "appears in more than one predicted cluster" in str(exc)
    else:
        raise AssertionError("Expected duplicate cluster membership to raise ValueError")


def test_cli_evaluate_outputs_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sentence_id", "text", "label"])
        writer.writeheader()
        writer.writerow({"sentence_id": "s1", "text": "reset password", "label": "password"})
        writer.writerow({"sentence_id": "s2", "text": "forgot password", "label": "password"})
        writer.writerow({"sentence_id": "s3", "text": "cancel plan", "label": "cancel"})

    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps(
            [
                {"cluster_id": "cluster-0", "member_sentence_ids": ["s1", "s2"]},
                {"cluster_id": "cluster-1", "member_sentence_ids": ["s3"]},
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dialin_llm.cli",
            "evaluate",
            "--input",
            str(dataset_path),
            "--clusters",
            str(cluster_path),
            "--id-col",
            "sentence_id",
            "--text-col",
            "text",
            "--label-col",
            "label",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["coverage"] == 1.0
    assert report["assigned_only"]["nmi"] == 1.0
    assert report["with_unassigned"]["ari"] == 1.0
