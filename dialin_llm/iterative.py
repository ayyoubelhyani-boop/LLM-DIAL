from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .clustering import cluster_embeddings, group_cluster_members
from .io import SentenceRecord
from .llm_utils import CoherenceEvaluator
from .sampling import sample_indices


@dataclass
class IntentCluster:
    cluster_id: str
    member_sentence_ids: list[str]
    sentences: list[str]
    iteration_found: int
    label: str | None = None
    source_cluster_ids: list[str] = field(default_factory=list)


@dataclass
class ClusterCandidateResult:
    k: int
    good_clusters: int
    bad_clusters: int
    score: float


@dataclass
class IterationSummary:
    iteration: int
    remaining_before: int
    remaining_after: int
    selected_k: int | None
    candidate_results: list[ClusterCandidateResult]
    accepted_cluster_ids: list[str]


@dataclass
class ClusterRunResult:
    clusters: list[IntentCluster]
    remaining_sentence_ids: list[str]
    iterations: list[IterationSummary]

    @property
    def iterations_used(self) -> int:
        return len(self.iterations)


@dataclass
class _EvaluatedCluster:
    member_indices: list[int]
    sampled_indices: list[int]
    is_good: bool


def run_iterative_clustering(
    records: Sequence[SentenceRecord],
    embeddings: np.ndarray,
    *,
    candidate_ks: Sequence[int],
    evaluator: CoherenceEvaluator,
    clusterer: str = "kmeans",
    sample_size: int = 20,
    sampler: str = "farthest",
    epsilon: float = 0.05,
    tmax: int = 5,
    seed: int = 0,
) -> ClusterRunResult:
    if len(records) != len(embeddings):
        raise ValueError("records and embeddings must have the same length")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    if not candidate_ks:
        raise ValueError("candidate_ks must not be empty")

    total_count = len(records)
    remaining_indices = list(range(total_count))
    accepted_clusters: list[IntentCluster] = []
    iteration_summaries: list[IterationSummary] = []
    next_cluster_number = 0

    for iteration in range(1, tmax + 1):
        if not remaining_indices:
            break
        if len(remaining_indices) / max(total_count, 1) <= epsilon:
            break

        candidate_results: list[tuple[ClusterCandidateResult, list[_EvaluatedCluster]]] = []
        subset_embeddings = embeddings[np.asarray(remaining_indices)]

        for offset, candidate_k in enumerate(candidate_ks):
            effective_k = min(int(candidate_k), len(remaining_indices))
            if effective_k <= 0:
                continue
            labels = cluster_embeddings(
                subset_embeddings,
                k=effective_k,
                method=clusterer,
                seed=seed + iteration * 997 + offset,
            )
            grouped = group_cluster_members(labels, remaining_indices)
            evaluated_clusters: list[_EvaluatedCluster] = []
            good_clusters = 0
            bad_clusters = 0

            for cluster_idx, member_indices in sorted(grouped.items()):
                sampled_indices = sample_indices(
                    member_indices,
                    sample_size=sample_size,
                    sampler=sampler,
                    embeddings=embeddings,
                    seed=seed + iteration * 1009 + effective_k * 31 + cluster_idx,
                )
                sampled_sentences = [records[idx].text for idx in sampled_indices]
                is_good = evaluator.coherence_eval(sampled_sentences)
                evaluated_clusters.append(
                    _EvaluatedCluster(
                        member_indices=list(member_indices),
                        sampled_indices=list(sampled_indices),
                        is_good=is_good,
                    )
                )
                if is_good:
                    good_clusters += 1
                else:
                    bad_clusters += 1

            score = good_clusters / (bad_clusters + 1)
            candidate_results.append(
                (
                    ClusterCandidateResult(
                        k=effective_k,
                        good_clusters=good_clusters,
                        bad_clusters=bad_clusters,
                        score=score,
                    ),
                    evaluated_clusters,
                )
            )

        if not candidate_results:
            break

        best_candidate, best_clusters = max(
            candidate_results,
            key=lambda item: (item[0].score, item[0].good_clusters, -item[0].bad_clusters, -item[0].k),
        )

        accepted_this_round: list[str] = []
        accepted_indices: set[int] = set()
        remaining_before = len(remaining_indices)
        for evaluated in best_clusters:
            if not evaluated.is_good:
                continue
            cluster_id = f"cluster-{next_cluster_number}"
            next_cluster_number += 1
            member_sentence_ids = [records[idx].sentence_id for idx in evaluated.member_indices]
            cluster_sentences = [records[idx].text for idx in evaluated.member_indices]
            accepted_clusters.append(
                IntentCluster(
                    cluster_id=cluster_id,
                    member_sentence_ids=member_sentence_ids,
                    sentences=cluster_sentences,
                    iteration_found=iteration,
                    source_cluster_ids=[cluster_id],
                )
            )
            accepted_this_round.append(cluster_id)
            accepted_indices.update(evaluated.member_indices)

        if not accepted_indices:
            iteration_summaries.append(
                IterationSummary(
                    iteration=iteration,
                    remaining_before=remaining_before,
                    remaining_after=remaining_before,
                    selected_k=best_candidate.k,
                    candidate_results=[result for result, _ in candidate_results],
                    accepted_cluster_ids=[],
                )
            )
            break

        remaining_indices = [idx for idx in remaining_indices if idx not in accepted_indices]
        iteration_summaries.append(
            IterationSummary(
                iteration=iteration,
                remaining_before=remaining_before,
                remaining_after=len(remaining_indices),
                selected_k=best_candidate.k,
                candidate_results=[result for result, _ in candidate_results],
                accepted_cluster_ids=accepted_this_round,
            )
        )

    remaining_sentence_ids = [records[idx].sentence_id for idx in remaining_indices]
    return ClusterRunResult(
        clusters=accepted_clusters,
        remaining_sentence_ids=remaining_sentence_ids,
        iterations=iteration_summaries,
    )
