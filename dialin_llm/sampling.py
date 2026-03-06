from __future__ import annotations

from typing import Sequence

import numpy as np


def sample_indices(
    indices: Sequence[int],
    *,
    sample_size: int,
    sampler: str,
    embeddings: np.ndarray | None = None,
    seed: int = 0,
) -> list[int]:
    if sample_size <= 0:
        return []
    if len(indices) <= sample_size:
        return list(indices)

    normalized = sampler.strip().lower()
    if normalized == "random":
        return random_sample(indices, sample_size=sample_size, seed=seed)
    if normalized == "farthest":
        if embeddings is None:
            raise ValueError("embeddings are required for farthest sampling")
        return farthest_first_sample(indices, embeddings=embeddings, sample_size=sample_size, seed=seed)
    raise ValueError(f"Unknown sampler: {sampler}")


def random_sample(indices: Sequence[int], *, sample_size: int, seed: int = 0) -> list[int]:
    if sample_size <= 0:
        return []
    if len(indices) <= sample_size:
        return list(indices)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(indices), size=sample_size, replace=False)
    return chosen.tolist()


def farthest_first_sample(
    indices: Sequence[int],
    *,
    embeddings: np.ndarray,
    sample_size: int,
    seed: int = 0,
) -> list[int]:
    if sample_size <= 0:
        return []
    if len(indices) <= sample_size:
        return list(indices)

    rng = np.random.default_rng(seed)
    ordered_positions = rng.permutation(len(indices))
    ordered_indices = [indices[pos] for pos in ordered_positions]
    vectors = embeddings[np.asarray(ordered_indices)]
    centroid = vectors.mean(axis=0, keepdims=True)
    distances_to_centroid = np.linalg.norm(vectors - centroid, axis=1)
    first_idx = int(np.argmin(distances_to_centroid))

    selected_positions = [first_idx]
    selected_mask = np.zeros(len(ordered_indices), dtype=bool)
    selected_mask[first_idx] = True

    while len(selected_positions) < sample_size:
        selected_vectors = vectors[np.asarray(selected_positions)]
        remaining_positions = np.flatnonzero(~selected_mask)
        remaining_vectors = vectors[remaining_positions]
        pairwise = np.linalg.norm(
            remaining_vectors[:, None, :] - selected_vectors[None, :, :],
            axis=2,
        )
        min_distances = pairwise.min(axis=1)
        next_pos = int(remaining_positions[int(np.argmax(min_distances))])
        selected_positions.append(next_pos)
        selected_mask[next_pos] = True

    return [ordered_indices[pos] for pos in selected_positions]

