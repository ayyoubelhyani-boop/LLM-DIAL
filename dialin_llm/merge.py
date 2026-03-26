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
    sampler: str = "head",
) -> list[IntentCluster]:
    named_clusters: list[IntentCluster] = []
    for cluster in clusters:
        sample = _sample_cluster_for_naming(cluster.sentences, sample_size=sample_size, sampler=sampler)
        label = namer.name_cluster(sample)
        named_clusters.append(replace(cluster, label=label))
    return named_clusters


def merge_clusters_by_label(
    clusters: Sequence[IntentCluster],
    *,
    theta: float = 0.8,
    strategy: str = "label",
    label_weight: float = 0.5,
    use_vmf_gate: bool = False,
    kappa: float = 5.0,
    seed: int = 0,
) -> list[IntentCluster]:
    if len(clusters) <= 1:
        return list(clusters)
    if not 0.0 <= label_weight <= 1.0:
        raise ValueError("label_weight must be between 0.0 and 1.0")

    labels = [cluster.label or "general-request" for cluster in clusters]
    label_embedder = TfidfEmbeddingBackend(analyzer="char_wb", ngram_range=(3, 5))
    label_embeddings = l2_normalize(label_embedder.fit_transform(labels))
    content_embeddings = _build_cluster_content_embeddings(clusters) if strategy.strip().lower() == "hybrid" else None

    union_find = UnionFind(len(clusters))
    rng = np.random.default_rng(seed)
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            dot = _pairwise_merge_similarity(
                left,
                right,
                label_embeddings=label_embeddings,
                content_embeddings=content_embeddings,
                strategy=strategy,
                label_weight=label_weight,
            )
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


def _sample_cluster_for_naming(
    sentences: Sequence[str],
    *,
    sample_size: int,
    sampler: str,
) -> list[str]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive for cluster naming")
    if len(sentences) <= sample_size:
        return list(sentences)

    normalized = sampler.strip().lower()
    if normalized == "head":
        return list(sentences[:sample_size])
    if normalized != "centroid":
        raise ValueError(f"Unknown naming sampler: {sampler}")

    embedder = TfidfEmbeddingBackend(analyzer="word", ngram_range=(1, 2))
    sentence_embeddings = l2_normalize(embedder.fit_transform(sentences))
    centroid = l2_normalize(np.mean(sentence_embeddings, axis=0, keepdims=True))[0]
    scores = sentence_embeddings @ centroid
    ranked_indices = np.argsort(-scores)
    return [sentences[int(idx)] for idx in ranked_indices[:sample_size]]


def _build_cluster_content_embeddings(clusters: Sequence[IntentCluster]) -> np.ndarray:
    documents = [_cluster_content_document(cluster) for cluster in clusters]
    embedder = TfidfEmbeddingBackend(analyzer="word", ngram_range=(1, 2), min_df=1)
    return l2_normalize(embedder.fit_transform(documents))


def _cluster_content_document(cluster: IntentCluster) -> str:
    if cluster.sentences:
        return " ".join(sentence.strip() for sentence in cluster.sentences if sentence.strip())
    return cluster.label or "general request"


def _pairwise_merge_similarity(
    left: int,
    right: int,
    *,
    label_embeddings: np.ndarray,
    content_embeddings: np.ndarray | None,
    strategy: str,
    label_weight: float,
) -> float:
    label_dot = float(np.dot(label_embeddings[left], label_embeddings[right]))
    normalized = strategy.strip().lower()
    if normalized == "label":
        return float(np.clip(label_dot, -1.0, 1.0))
    if normalized != "hybrid":
        raise ValueError(f"Unknown merge strategy: {strategy}")

    if content_embeddings is None:
        raise ValueError("content_embeddings must be provided for hybrid merge")
    content_dot = float(np.dot(content_embeddings[left], content_embeddings[right]))
    combined = label_weight * label_dot + (1.0 - label_weight) * content_dot
    return float(np.clip(combined, -1.0, 1.0))
