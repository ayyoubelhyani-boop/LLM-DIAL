from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .embeddings import TfidfEmbeddingBackend, l2_normalize
from .iterative import IntentCluster
from .llm_utils import ClusterNamer


def name_clusters(
    clusters: Sequence[IntentCluster],
    namer: ClusterNamer,
    *,
    sample_size: int = 20,
) -> list[IntentCluster]:
    named_clusters: list[IntentCluster] = []
    for cluster in clusters:
        sample = cluster.sentences[:sample_size]
        label = namer.name_cluster(sample)
        named_clusters.append(replace(cluster, label=label))
    return named_clusters


def merge_clusters_by_label(
    clusters: Sequence[IntentCluster],
    *,
    theta: float = 0.8,
    use_vmf_gate: bool = False,
    kappa: float = 5.0,
    seed: int = 0,
) -> list[IntentCluster]:
    if len(clusters) <= 1:
        return list(clusters)

    labels = [cluster.label or "general-request" for cluster in clusters]
    embedder = TfidfEmbeddingBackend(analyzer="char_wb", ngram_range=(3, 5))
    normalized_embeddings = l2_normalize(embedder.fit_transform(labels))

    union_find = UnionFind(len(clusters))
    rng = np.random.default_rng(seed)
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            dot = float(np.clip(np.dot(normalized_embeddings[left], normalized_embeddings[right]), -1.0, 1.0))
            distance = float(np.arccos(dot))
            if distance >= theta:
                continue
            if use_vmf_gate:
                # Assumption: use exp(kappa * (cos(theta) - 1)) as the merge acceptance probability.
                probability = float(np.exp(kappa * (dot - 1.0)))
                if rng.random() > probability:
                    continue
            union_find.union(left, right)

    grouped: dict[int, list[int]] = {}
    for idx in range(len(clusters)):
        root = union_find.find(idx)
        grouped.setdefault(root, []).append(idx)

    merged_clusters: list[IntentCluster] = []
    for _, member_positions in sorted(grouped.items(), key=lambda item: min(item[1])):
        member_clusters = [clusters[pos] for pos in member_positions]
        primary = member_clusters[0]
        merged_ids: list[str] = []
        merged_sentences: list[str] = []
        merged_sources: list[str] = []
        for cluster in member_clusters:
            merged_ids.extend(cluster.member_sentence_ids)
            merged_sentences.extend(cluster.sentences)
            merged_sources.extend(cluster.source_cluster_ids or [cluster.cluster_id])
        merged_clusters.append(
            IntentCluster(
                cluster_id=primary.cluster_id,
                member_sentence_ids=merged_ids,
                sentences=merged_sentences,
                iteration_found=min(cluster.iteration_found for cluster in member_clusters),
                label=primary.label,
                source_cluster_ids=merged_sources,
            )
        )
    return merged_clusters


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

