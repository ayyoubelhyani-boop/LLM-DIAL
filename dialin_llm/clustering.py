from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    k: int,
    method: str = "kmeans",
    seed: int = 0,
) -> np.ndarray:
    if len(embeddings) == 0:
        raise ValueError("Cannot cluster an empty embedding matrix")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(embeddings))

    normalized = method.strip().lower()
    if normalized == "kmeans":
        estimator = KMeans(n_clusters=k, random_state=seed, n_init=10)
    elif normalized in {"minibatch", "minibatchkmeans"}:
        estimator = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            n_init=5,
            batch_size=min(1024, len(embeddings)),
        )
    else:
        raise ValueError(f"Unknown clusterer: {method}")

    return estimator.fit_predict(embeddings)


def group_cluster_members(labels: Sequence[int], original_indices: Sequence[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for label, original_idx in zip(labels, original_indices):
        grouped[int(label)].append(int(original_idx))
    return dict(grouped)
