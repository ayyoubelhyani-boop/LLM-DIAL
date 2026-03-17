import json
from pathlib import Path

import numpy as np

from dialin_llm.io import SentenceRecord
from dialin_llm.plotting import (
    UNASSIGNED_CLUSTER_ID,
    load_cluster_payload,
    prepare_cluster_plot_data,
)


def test_load_cluster_payload_requires_list(tmp_path: Path) -> None:
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(json.dumps({"cluster_id": "cluster-0"}), encoding="utf-8")

    try:
        load_cluster_payload(cluster_path)
    except ValueError as exc:
        assert "JSON list" in str(exc)
    else:
        raise AssertionError("Expected invalid cluster payload to raise ValueError")


def test_prepare_cluster_plot_data_tracks_clusters_and_unassigned() -> None:
    records = [
        SentenceRecord("s1", "reset card", {}),
        SentenceRecord("s2", "check card", {}),
        SentenceRecord("s3", "bank transfer", {}),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    cluster_payload = [
        {"cluster_id": "cluster-0", "label": "card-help", "member_sentence_ids": ["s1", "s2"]},
    ]

    plot_data = prepare_cluster_plot_data(records, embeddings, cluster_payload)

    assert plot_data.coordinates.shape == (3, 2)
    assert plot_data.cluster_ids.count("cluster-0") == 2
    assert plot_data.cluster_ids.count(UNASSIGNED_CLUSTER_ID) == 1
    assert plot_data.cluster_display_names["cluster-0"] == "card-help"
    assert plot_data.cluster_display_names[UNASSIGNED_CLUSTER_ID] == "Unassigned"


def test_prepare_cluster_plot_data_applies_max_points_sampling() -> None:
    records = [
        SentenceRecord(f"s{index}", f"text {index}", {})
        for index in range(6)
    ]
    embeddings = np.eye(6, dtype=float)
    cluster_payload = [
        {"cluster_id": "cluster-0", "label": "all", "member_sentence_ids": [record.sentence_id for record in records]},
    ]

    plot_data = prepare_cluster_plot_data(records, embeddings, cluster_payload, max_points=4, seed=7)

    assert plot_data.coordinates.shape == (4, 2)
    assert sum(plot_data.cluster_sizes.values()) == 4
