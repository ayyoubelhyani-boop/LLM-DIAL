from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

from .io import SentenceRecord
from .iterative import ClusterRunResult, IntentCluster


def normalized_mutual_info(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    return float(normalized_mutual_info_score(true_labels, predicted_labels))


def adjusted_rand_index(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    return float(adjusted_rand_score(true_labels, predicted_labels))


def v_measure(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    return float(v_measure_score(true_labels, predicted_labels))


def goodness_stats(result: ClusterRunResult) -> dict[str, int]:
    return {
        "num_clusters": len(result.clusters),
        "num_remaining": len(result.remaining_sentence_ids),
        "iterations_used": result.iterations_used,
    }


def cluster_sizes(clusters: Sequence[IntentCluster]) -> list[int]:
    return [len(cluster.member_sentence_ids) for cluster in clusters]


def load_cluster_memberships(path: str | Path) -> dict[str, str]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    memberships: dict[str, str] = {}
    for idx, cluster in enumerate(payload):
        cluster_id = str(cluster.get("cluster_id") or f"cluster-{idx}")
        sentence_ids = cluster.get("member_sentence_ids")
        if not isinstance(sentence_ids, list):
            raise ValueError(f"Cluster payload is missing member_sentence_ids for cluster {cluster_id}")
        for sentence_id in sentence_ids:
            normalized_id = str(sentence_id)
            if normalized_id in memberships:
                raise ValueError(f"Sentence {normalized_id!r} appears in more than one predicted cluster")
            memberships[normalized_id] = cluster_id
    return memberships


def evaluate_clustering(
    records: Sequence[SentenceRecord],
    memberships: dict[str, str],
    *,
    label_key: str = "label",
    unassigned_label: str = "__unassigned__",
) -> dict[str, object]:
    true_all: list[str] = []
    predicted_all: list[str] = []
    true_assigned: list[str] = []
    predicted_assigned: list[str] = []

    for record in records:
        if label_key not in record.metadata:
            raise ValueError(f"Missing reference label {label_key!r} for sentence {record.sentence_id!r}")
        true_label = str(record.metadata[label_key])
        predicted_label = memberships.get(record.sentence_id, unassigned_label)

        true_all.append(true_label)
        predicted_all.append(predicted_label)
        if predicted_label != unassigned_label:
            true_assigned.append(true_label)
            predicted_assigned.append(predicted_label)

    assigned_count = len(true_assigned)
    total_count = len(records)
    unassigned_count = total_count - assigned_count
    if total_count == 0:
        raise ValueError("Cannot evaluate clustering on an empty record set")

    assigned_metrics: dict[str, float | None]
    if assigned_count == 0:
        assigned_metrics = {"nmi": None, "ari": None, "v_measure": None}
    else:
        assigned_metrics = {
            "nmi": normalized_mutual_info(true_assigned, predicted_assigned),
            "ari": adjusted_rand_index(true_assigned, predicted_assigned),
            "v_measure": v_measure(true_assigned, predicted_assigned),
        }

    return {
        "num_records": total_count,
        "num_assigned": assigned_count,
        "num_unassigned": unassigned_count,
        "coverage": assigned_count / total_count,
        "num_predicted_clusters": len(set(memberships.values())),
        "num_reference_labels": len(set(true_all)),
        "assigned_only": assigned_metrics,
        "with_unassigned": {
            "nmi": normalized_mutual_info(true_all, predicted_all),
            "ari": adjusted_rand_index(true_all, predicted_all),
            "v_measure": v_measure(true_all, predicted_all),
        },
    }
