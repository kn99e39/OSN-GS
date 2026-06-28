from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.utils.geometry import pairwise_distances


@dataclass
class ColorClusters:
    centers: np.ndarray
    assignments: np.ndarray

    def assign_colors(self, colors: np.ndarray) -> np.ndarray:
        if len(self.centers) == 0:
            return np.zeros(len(colors), dtype=np.int32)
        distances = pairwise_distances(np.asarray(colors, dtype=np.float32), self.centers)
        return distances.argmin(axis=1).astype(np.int32)


def fit_color_clusters(colors: np.ndarray, k: int = 4, iterations: int = 8) -> ColorClusters:
    colors = np.asarray(colors, dtype=np.float32)
    if len(colors) == 0:
        return ColorClusters(np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32))
    k = max(1, min(k, len(colors)))
    centers = colors[np.linspace(0, len(colors) - 1, k).astype(np.int32)].copy()
    assignments = np.zeros(len(colors), dtype=np.int32)
    for _ in range(iterations):
        distances = pairwise_distances(colors, centers)
        assignments = distances.argmin(axis=1).astype(np.int32)
        for idx in range(k):
            member_mask = assignments == idx
            if member_mask.any():
                centers[idx] = colors[member_mask].mean(axis=0)
    return ColorClusters(centers=centers, assignments=assignments)


def colors_for_cluster_ids(clusters: ColorClusters, cluster_ids: np.ndarray) -> np.ndarray:
    if len(clusters.centers) == 0:
        return np.full((len(cluster_ids), 3), 0.5, dtype=np.float32)
    cluster_ids = np.clip(np.asarray(cluster_ids, dtype=np.int32), 0, len(clusters.centers) - 1)
    return clusters.centers[cluster_ids]
