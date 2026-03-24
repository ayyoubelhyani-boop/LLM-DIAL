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
    if normalized == "convex":
        if embeddings is None:
            raise ValueError("embeddings are required for convex sampling")
        return convex_hull_sample(indices, embeddings=embeddings, sample_size=sample_size, seed=seed)
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


def convex_hull_sample(
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

    projected = _project_to_2d(vectors)
    hull_positions = _convex_hull_positions(projected)
    if len(hull_positions) < 2:
        return farthest_first_sample(indices, embeddings=embeddings, sample_size=sample_size, seed=seed)

    chosen_positions = _select_hull_subset(hull_positions, sample_size=min(sample_size, len(hull_positions)))
    selected_mask = np.zeros(len(ordered_indices), dtype=bool)
    selected_mask[np.asarray(chosen_positions)] = True

    while len(chosen_positions) < sample_size:
        selected_vectors = vectors[np.asarray(chosen_positions)]
        remaining_positions = np.flatnonzero(~selected_mask)
        remaining_vectors = vectors[remaining_positions]
        pairwise = np.linalg.norm(
            remaining_vectors[:, None, :] - selected_vectors[None, :, :],
            axis=2,
        )
        min_distances = pairwise.min(axis=1)
        next_pos = int(remaining_positions[int(np.argmax(min_distances))])
        chosen_positions.append(next_pos)
        selected_mask[next_pos] = True

    return [ordered_indices[pos] for pos in chosen_positions]


def _project_to_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if centered.ndim != 2 or centered.shape[0] == 0:
        return np.zeros((len(vectors), 2), dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((len(vectors), 2), dtype=np.float64)

    if vh.size == 0:
        return np.zeros((len(vectors), 2), dtype=np.float64)

    components = vh[:2]
    if components.shape[0] == 1:
        projected = centered @ components.T
        return np.hstack([projected, np.zeros((len(vectors), 1), dtype=np.float64)])
    return centered @ components.T


def _convex_hull_positions(points: np.ndarray) -> list[int]:
    if len(points) <= 1:
        return list(range(len(points)))

    indexed = sorted(
        ((float(point[0]), float(point[1]), idx) for idx, point in enumerate(points)),
        key=lambda item: (item[0], item[1], item[2]),
    )

    def cross(origin: tuple[float, float, int], left: tuple[float, float, int], right: tuple[float, float, int]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float, int]] = []
    for point in indexed:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float, int]] = []
    for point in reversed(indexed):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    unique_positions: list[int] = []
    seen: set[int] = set()
    for _, _, idx in hull:
        if idx not in seen:
            seen.add(idx)
            unique_positions.append(idx)
    return unique_positions


def _select_hull_subset(hull_positions: Sequence[int], *, sample_size: int) -> list[int]:
    if sample_size >= len(hull_positions):
        return list(hull_positions)
    selected: list[int] = []
    for step in range(sample_size):
        pos = int(np.floor(step * len(hull_positions) / sample_size))
        selected.append(hull_positions[pos])
    return selected
