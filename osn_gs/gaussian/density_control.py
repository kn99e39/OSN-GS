from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from osn_gs.gaussian.certain_gaussians import CertainGaussianSet
from osn_gs.gaussian.uncertain_gaussians import UncertainGaussianSet


@dataclass
class DensityControlStats:
    split_counts: dict[int, int] = field(default_factory=dict)
    clone_counts: dict[int, int] = field(default_factory=dict)
    prune_counts: dict[int, int] = field(default_factory=dict)


def collect_cluster_density_stats(
    gaussians: CertainGaussianSet,
    cluster_ids: np.ndarray,
    large_scale_threshold: float = 0.02,
    low_opacity_threshold: float = 0.05,
) -> DensityControlStats:
    stats = DensityControlStats()
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    mean_scale = gaussians.scales.mean(axis=1)
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        cid = int(cluster_id)
        stats.split_counts[cid] = int((mean_scale[mask] > large_scale_threshold).sum())
        stats.clone_counts[cid] = int(mask.sum())
        stats.prune_counts[cid] = int((gaussians.opacities[mask] < low_opacity_threshold).sum())
    return stats


def update_uncertain_confidence_from_residual(
    gaussians: UncertainGaussianSet,
    residual: float,
    step: float = 0.05,
) -> None:
    if gaussians.confidence is None or len(gaussians) == 0:
        return
    direction = -1.0 if residual > 0.1 else 1.0
    gaussians.confidence[:] = np.clip(gaussians.confidence + direction * step, 0.0, 1.0)
