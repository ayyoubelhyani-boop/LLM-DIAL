from __future__ import annotations

from typing import Sequence

from sklearn.metrics import normalized_mutual_info_score

from .iterative import ClusterRunResult, IntentCluster


def normalized_mutual_info(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    return float(normalized_mutual_info_score(true_labels, predicted_labels))


def goodness_stats(result: ClusterRunResult) -> dict[str, int]:
    return {
        "num_clusters": len(result.clusters),
        "num_remaining": len(result.remaining_sentence_ids),
        "iterations_used": result.iterations_used,
    }


def cluster_sizes(clusters: Sequence[IntentCluster]) -> list[int]:
    return [len(cluster.member_sentence_ids) for cluster in clusters]

