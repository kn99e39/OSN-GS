"""Unadopted exact-KNN candidates for the WL119 Performance Track.

The immutable ``torch.cdist`` reference lives in
``torch_coverage_first_subset_partition._knn``.  Nothing in this module is a
runtime fallback or production default; adoption requires full-scene identity.
"""

from __future__ import annotations

from typing import Any, Callable

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CandidateGraph,
    CoverageFirstPartitionConfig,
    _EPS,
)
from osn_gs.surface.torch_gaussian_surface_orientation import unsigned_normal_alignment
from osn_gs.utils.torch_ops import require_torch


def scipy_ckdtree_exact_knn(
    positions: Any,
    k: int,
    *,
    workers: int = -1,
    progress: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    """Exact Euclidean cKDTree candidate with explicit row-index self removal."""

    import numpy as np
    from scipy.spatial import cKDTree

    torch = require_torch()
    count = int(positions.shape[0])
    if count == 0 or int(k) <= 0:
        return (
            torch.zeros((count, 0), dtype=torch.int64, device=positions.device),
            torch.zeros((count, 0), dtype=positions.dtype, device=positions.device),
        )
    host = positions.detach().cpu().numpy().astype(np.float64, copy=False)
    if progress:
        progress(f"scipy cKDTree build count={count}")
    tree = cKDTree(host, compact_nodes=True, balanced_tree=True)
    if progress:
        progress(f"scipy cKDTree exact query k={int(k) + 1} workers={workers}")
    _, raw_index = tree.query(host, k=int(k) + 1, eps=0.0, workers=int(workers))
    raw_index = np.asarray(raw_index, dtype=np.int64).reshape(count, int(k) + 1)
    rows = np.arange(count, dtype=np.int64)[:, None]
    self_match = raw_index == rows
    has_self = self_match.any(axis=1)
    self_position = self_match.argmax(axis=1)
    output_columns = np.arange(int(k), dtype=np.int64)[None, :]
    source_columns = output_columns + (
        has_self[:, None] & (output_columns >= self_position[:, None])
    ).astype(np.int64)
    neighbor_index_host = np.take_along_axis(raw_index, source_columns, axis=1)
    neighbor_index = torch.from_numpy(neighbor_index_host).to(positions.device)
    neighbor_distance = (
        positions[:, None, :] - positions[neighbor_index]
    ).norm(dim=-1)
    return neighbor_index, neighbor_distance


def candidate_graph_from_neighbors(
    orientation: Any,
    config: CoverageFirstPartitionConfig,
    neighbor_index: Any,
    neighbor_distance: Any,
    *,
    progress: Callable[[str], None] | None = None,
) -> CandidateGraph:
    """Reference graph post-processing applied verbatim to candidate neighbors."""

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    count = int(positions.shape[0])
    k = int(neighbor_index.shape[1])
    local_spacing = neighbor_distance.median(dim=1).values
    rows = torch.arange(count, dtype=torch.int64, device=positions.device).unsqueeze(1).expand(-1, k)
    left = torch.minimum(rows.reshape(-1), neighbor_index.reshape(-1))
    right = torch.maximum(rows.reshape(-1), neighbor_index.reshape(-1))
    unique_key = torch.unique(left * count + right)
    candidate_left = torch.div(unique_key, count, rounding_mode="floor")
    candidate_right = unique_key - candidate_left * count
    candidate_edges = torch.stack((candidate_left, candidate_right), dim=1)
    edge_distance = (positions[candidate_left] - positions[candidate_right]).norm(dim=-1)
    connect_scale = torch.minimum(local_spacing[candidate_left], local_spacing[candidate_right])
    spatial = edge_distance <= config.spatial_connect_spacing_multiplier * connect_scale.clamp_min(_EPS)
    alignment = unsigned_normal_alignment(normals[candidate_left], normals[candidate_right])
    normal = alignment >= config.normal_compatibility_min_alignment
    if progress:
        progress(
            f"candidate edges={int(candidate_edges.shape[0])} "
            f"spatial={int(spatial.sum())} accepted={int((spatial & normal).sum())}"
        )
    return CandidateGraph(
        count=count, local_spacing=local_spacing, candidate_edges=candidate_edges,
        spatial_edge_mask=spatial, normal_compatible_mask=normal,
        normal_alignment=alignment, neighbor_index=neighbor_index,
    )
