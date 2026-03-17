from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

from .io import SentenceRecord


UNASSIGNED_CLUSTER_ID = "__unassigned__"


@dataclass(frozen=True)
class ClusterPlotData:
    coordinates: np.ndarray
    cluster_ids: list[str]
    cluster_sizes: dict[str, int]
    cluster_display_names: dict[str, str]


def load_cluster_payload(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Cluster plot input must be a JSON list")
    return payload


def prepare_cluster_plot_data(
    records: Sequence[SentenceRecord],
    embeddings: np.ndarray,
    cluster_payload: Sequence[dict[str, object]],
    *,
    max_points: int | None = None,
    seed: int = 0,
    unassigned_label: str = UNASSIGNED_CLUSTER_ID,
) -> ClusterPlotData:
    if len(records) == 0:
        raise ValueError("Cannot plot an empty dataset")
    if embeddings.shape[0] != len(records):
        raise ValueError("Embeddings row count must match the number of records")

    cluster_display_names, memberships = _build_cluster_maps(cluster_payload)

    sample_indices = _sample_indices(len(records), max_points=max_points, seed=seed)
    sampled_embeddings = embeddings[sample_indices]
    sampled_records = [records[index] for index in sample_indices]
    reduced = PCA(n_components=2).fit_transform(sampled_embeddings)

    cluster_ids: list[str] = []
    cluster_sizes: dict[str, int] = {}
    for record in sampled_records:
        cluster_id = memberships.get(record.sentence_id, unassigned_label)
        cluster_ids.append(cluster_id)
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    cluster_display_names.setdefault(unassigned_label, "Unassigned")

    return ClusterPlotData(
        coordinates=reduced,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        cluster_display_names=cluster_display_names,
    )


def save_cluster_plot(
    plot_data: ClusterPlotData,
    *,
    output_path: str | Path,
    title: str,
    annotate_top_n: int = 10,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install the 'viz' extra to use plot-clusters."
        ) from exc

    coordinates = plot_data.coordinates
    cluster_ids = plot_data.cluster_ids
    unique_cluster_ids = list(dict.fromkeys(cluster_ids))
    colors = _build_color_map(unique_cluster_ids)

    figure, axis = plt.subplots(figsize=(12, 8))
    for cluster_id in unique_cluster_ids:
        mask = np.array([value == cluster_id for value in cluster_ids], dtype=bool)
        display_name = plot_data.cluster_display_names.get(cluster_id, cluster_id)
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=24 if cluster_id != UNASSIGNED_CLUSTER_ID else 18,
            alpha=0.72 if cluster_id != UNASSIGNED_CLUSTER_ID else 0.4,
            c=[colors[cluster_id]],
            label=f"{display_name} ({plot_data.cluster_sizes[cluster_id]})",
            linewidths=0,
        )

    for cluster_id in _largest_clusters(plot_data.cluster_sizes, top_n=annotate_top_n):
        if cluster_id == UNASSIGNED_CLUSTER_ID:
            continue
        mask = np.array([value == cluster_id for value in cluster_ids], dtype=bool)
        centroid = coordinates[mask].mean(axis=0)
        axis.text(
            float(centroid[0]),
            float(centroid[1]),
            plot_data.cluster_display_names.get(cluster_id, cluster_id),
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        )

    handles, labels = axis.get_legend_handles_labels()
    if len(handles) > 12:
        handles = handles[:12]
        labels = labels[:12]

    axis.legend(handles, labels, loc="best", fontsize=8, frameon=True)
    axis.set_title(title)
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.grid(alpha=0.15)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _build_cluster_maps(cluster_payload: Sequence[dict[str, object]]) -> tuple[dict[str, str], dict[str, str]]:
    display_names: dict[str, str] = {}
    memberships: dict[str, str] = {}
    for index, cluster in enumerate(cluster_payload):
        cluster_id = str(cluster.get("cluster_id") or f"cluster-{index}")
        label = str(cluster.get("label") or cluster_id)
        display_names[cluster_id] = label
        sentence_ids = cluster.get("member_sentence_ids")
        if not isinstance(sentence_ids, list):
            raise ValueError(f"Cluster payload is missing member_sentence_ids for cluster {cluster_id}")
        for sentence_id in sentence_ids:
            normalized_id = str(sentence_id)
            if normalized_id in memberships:
                raise ValueError(f"Sentence {normalized_id!r} appears in more than one predicted cluster")
            memberships[normalized_id] = cluster_id
    return display_names, memberships


def _sample_indices(total_count: int, *, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or max_points >= total_count:
        return np.arange(total_count)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total_count, size=max_points, replace=False)
    return np.sort(indices)


def _largest_clusters(cluster_sizes: dict[str, int], *, top_n: int) -> list[str]:
    return [
        cluster_id
        for cluster_id, _ in sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    ]


def _build_color_map(cluster_ids: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install the 'viz' extra to use plot-clusters."
        ) from exc

    palette = plt.cm.get_cmap("tab20", max(len(cluster_ids), 1))
    color_map: dict[str, tuple[float, float, float, float]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        if cluster_id == UNASSIGNED_CLUSTER_ID:
            color_map[cluster_id] = (0.6, 0.6, 0.6, 1.0)
        else:
            color_map[cluster_id] = palette(index % palette.N)
    return color_map
