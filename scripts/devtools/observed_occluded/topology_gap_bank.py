from __future__ import annotations

"""Worklog 121 -- SUPPLEMENTAL query bank built on the ACTUAL frozen WL107/109
visible topology (directive section 11).

Worklog 120's R5 probes were same-region nearest-anchor midpoints -- an
approximation that never replayed component fragmentation, and which worklog 120
disclosed as such. This module replaces the approximation for the supplemental
bank while leaving the original R5 queries untouched for historical replay.

Construction, which never invents a pair:

  1. Replay WL107/109 topology with the existing read-only functions, exactly as
     worklogs 112-119 do: `build_candidate_graph` ->
     `accumulate_image_space_pairs` -> `filter_by_3d_locality` ->
     `apply_secondary_geometric_gate` -> `_connected_component_roots`. Nothing is
     modified, merged, split, or re-thresholded.
  2. In each view, scan the renderer's own per-pixel representative map at the
     SAME minimal 4-connectivity (right/down) worklog 107 uses, and keep raster
     neighbours whose two representatives are DIFFERENT surfels lying in
     DIFFERENT FINAL visible components. These are exactly the observed
     raster-local adjacency contexts that sit across a current component
     separation -- not nearest 3D pairs, not region labels, not proximity.
  3. Record for each retained context: view id, both pixel coordinates, both
     representative ids, both final component ids, both G2 world positions, and
     the gating provenance (was the pair rejected by the 3D-locality filter, by
     the geometric gate and with which reason, or kept as a positive edge yet
     still ending in different components through the connectivity result).
  4. Emit three diagnostic queries per context: endpoint A, endpoint B, and the
     world midpoint. No interpolation parameters are tuned; the midpoint is the
     single a-priori interior position.

These are diagnostic probes of ACTUAL CURRENT fragmentation only. Physical
continuity is never inferred from proximity, and topology is never modified.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .shared import QueryBank

# Provenance codes for why an observed cross-component raster adjacency is not
# an edge of the frozen topology. Read off the existing replay's own outputs.
GATING_LOCALITY_REJECTED = 0
GATING_GEOMETRIC_REJECTED = 1
GATING_POSITIVE_EDGE_BUT_SPLIT = 2
GATING_UNKNOWN = 3
GATING_NAMES = {
    GATING_LOCALITY_REJECTED: "REJECTED_BY_3D_LOCALITY_FILTER",
    GATING_GEOMETRIC_REJECTED: "REJECTED_BY_SECONDARY_GEOMETRIC_GATE",
    GATING_POSITIVE_EDGE_BUT_SPLIT: "POSITIVE_EDGE_YET_DIFFERENT_COMPONENTS",
    GATING_UNKNOWN: "NOT_ATTRIBUTED",
}

KIND_GAP_ENDPOINT_A = "T1_TOPOLOGY_GAP_ENDPOINT_A"
KIND_GAP_ENDPOINT_B = "T1_TOPOLOGY_GAP_ENDPOINT_B"
KIND_GAP_MIDPOINT = "T1_TOPOLOGY_GAP_MIDPOINT"
KIND_VERIFIED_OUT_OF_FRUSTUM = "T2_VERIFIED_OUT_OF_FRUSTUM_CONTROL"

# Fixed a priori. Selection strides and counts only -- no decision depends on them.
PER_VIEW_CONTEXT_CAP = 2000     # deterministic raster-order stride cap per view
CONTEXTS_PER_REGION = 60
VERIFIED_CONTROL_TARGET = 16
VERIFIED_CONTROL_EXTENT_MULTIPLES = (12.0, 16.0, 20.0, 24.0)


@dataclass
class TopologyReplay:
    subset_ids: torch.Tensor         # (N,) final visible component id per surfel
    subset_count: int
    subset_sizes: torch.Tensor
    positive_edge_keys: torch.Tensor  # sorted int64 keys of kept edges
    locality_edge_keys: torch.Tensor  # sorted int64 keys of locality-passed pairs
    geometric_rejected_keys: torch.Tensor
    geometric_rejected_reason: torch.Tensor
    stats: dict[str, Any]


def _pair_key(pairs: torch.Tensor, count: int) -> torch.Tensor:
    low = torch.minimum(pairs[:, 0], pairs[:, 1])
    high = torch.maximum(pairs[:, 0], pairs[:, 1])
    return low * count + high


def replay_frozen_topology(
    orientation: Any,
    per_view_representative_ids: list[torch.Tensor],
    *,
    progress: Callable[[str], None] | None = None,
) -> TopologyReplay:
    """WL107/109 replay through the existing read-only functions, unmodified."""

    from osn_gs.surface.torch_camera_induced_visible_adjacency import (
        REJECTION_REASONS, REASON_GEOMETRIC_DISCONTINUITY,
        CameraInducedAdjacencyConfig, accumulate_image_space_pairs,
        apply_secondary_geometric_gate, filter_by_3d_locality,
    )
    from osn_gs.surface.torch_coverage_first_subset_partition import (
        CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph,
    )

    count = int(orientation.positions.shape[0])
    device = orientation.positions.device
    config = CameraInducedAdjacencyConfig(local=CoverageFirstPartitionConfig())
    with torch.no_grad():
        if progress:
            progress("[WL107/109 replay, unchanged] build_candidate_graph")
        graph = build_candidate_graph(orientation, config.local, progress=progress)
        if progress:
            progress("[WL107/109 replay, unchanged] accumulate_image_space_pairs")
        raw_pairs, _ = accumulate_image_space_pairs(count, per_view_representative_ids, progress=None)
        local_pairs, _ = filter_by_3d_locality(raw_pairs, count, graph)
        if progress:
            progress("[WL107/109 replay, unchanged] apply_secondary_geometric_gate")
        geometry = apply_secondary_geometric_gate(local_pairs, orientation, config, progress=None)
        kept_mask = geometry["kept_mask"]
        positive_edges = local_pairs[kept_mask]
        rejected_pairs = local_pairs[~kept_mask]
        rejected_reason = torch.where(
            geometry["fails_residual"][~kept_mask],
            torch.full((int((~kept_mask).sum()),), REJECTION_REASONS.index(REASON_GEOMETRIC_DISCONTINUITY), dtype=torch.int8, device=device),
            torch.full((int((~kept_mask).sum()),), 1, dtype=torch.int8, device=device),
        )
        roots = _connected_component_roots(count, positive_edges, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
        subset_ids = subset_id_of_position[inverse]
        subset_sizes = counts[order]

    stats = {
        "visible_component_count": int(order.shape[0]),
        "largest_component_surfel_fraction": float(subset_sizes[0]) / count,
        "singleton_surfel_count": int((subset_sizes == 1).sum()),
        "raw_image_space_pairs": int(raw_pairs.shape[0]),
        "locality_passed_pairs": int(local_pairs.shape[0]),
        "positive_edges": int(positive_edges.shape[0]),
        "geometric_rejected_pairs": int(rejected_pairs.shape[0]),
    }
    if progress:
        progress(f"[replay consistency check] {stats}")
    return TopologyReplay(
        subset_ids=subset_ids, subset_count=int(order.shape[0]), subset_sizes=subset_sizes,
        positive_edge_keys=torch.sort(_pair_key(positive_edges, count)).values,
        locality_edge_keys=torch.sort(_pair_key(local_pairs, count)).values,
        geometric_rejected_keys=_pair_key(rejected_pairs, count),
        geometric_rejected_reason=rejected_reason,
        stats=stats,
    )


def collect_cross_component_contexts(
    view_index: int,
    representative_map: torch.Tensor,
    subset_ids: torch.Tensor,
    event_world: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Observed raster-local adjacency contexts spanning a component separation
    in ONE view. Same right/down 4-connectivity worklog 107 itself uses."""

    device = representative_map.device
    height, width = representative_map.shape
    valid = representative_map >= 0
    component = torch.where(valid, subset_ids[representative_map.clamp(min=0)], torch.full_like(representative_map, -1))

    rows_list, cols_a, cols_b = [], [], []
    grid_rows = torch.arange(height, device=device).reshape(-1, 1)
    grid_cols = torch.arange(width, device=device).reshape(1, -1)

    both_h = valid[:, :-1] & valid[:, 1:]
    split_h = both_h & (representative_map[:, :-1] != representative_map[:, 1:]) & (component[:, :-1] != component[:, 1:])
    rows_h = grid_rows.expand(height, width - 1)[split_h]
    cols_h = grid_cols[:, :-1].expand(height, width - 1)[split_h]

    both_v = valid[:-1, :] & valid[1:, :]
    split_v = both_v & (representative_map[:-1, :] != representative_map[1:, :]) & (component[:-1, :] != component[1:, :])
    rows_v = grid_rows[:-1, :].expand(height - 1, width)[split_v]
    cols_v = grid_cols.expand(height - 1, width)[split_v]

    row_a = torch.cat([rows_h, rows_v])
    col_a = torch.cat([cols_h, cols_v])
    row_b = torch.cat([rows_h, rows_v + 1])
    col_b = torch.cat([cols_h + 1, cols_v])

    flat_a = row_a * width + col_a
    flat_b = row_b * width + col_b
    return {
        "view_index": torch.full_like(flat_a, view_index),
        "row_a": row_a, "col_a": col_a, "row_b": row_b, "col_b": col_b,
        "representative_a": representative_map.reshape(-1)[flat_a],
        "representative_b": representative_map.reshape(-1)[flat_b],
        "component_a": component.reshape(-1)[flat_a],
        "component_b": component.reshape(-1)[flat_b],
        "world_a": event_world.reshape(-1, 3)[flat_a],
        "world_b": event_world.reshape(-1, 3)[flat_b],
    }


def deterministic_stride(total: int, take: int, device) -> torch.Tensor:
    if total <= take:
        return torch.arange(total, device=device)
    picks = torch.linspace(0, total - 1, steps=take, device=device).round().to(torch.int64)
    return torch.unique(picks, sorted=True)


def attribute_gating(
    representative_a: torch.Tensor, representative_b: torch.Tensor, count: int, replay: TopologyReplay,
) -> torch.Tensor:
    """Why is this observed adjacency not a topology edge? Read off the replay."""

    keys = _pair_key(torch.stack([representative_a, representative_b], dim=1), count)
    in_locality = torch.isin(keys, replay.locality_edge_keys)
    in_positive = torch.isin(keys, replay.positive_edge_keys)
    reason = torch.full_like(keys, GATING_UNKNOWN, dtype=torch.int64)
    reason[~in_locality] = GATING_LOCALITY_REJECTED
    reason[in_locality & ~in_positive] = GATING_GEOMETRIC_REJECTED
    reason[in_positive] = GATING_POSITIVE_EDGE_BUT_SPLIT
    return reason


def build_verified_out_of_frustum_controls(
    cameras: list[Any], scene_centre: torch.Tensor, scene_extent: float, device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Directive section 12: a deterministic control set whose relevant-view
    count is CONFIRMED to be 0 before any candidate is evaluated. Worklog 120's
    R6 used 4x/8x scene extent and 8 of its 12 points turned out to still be
    inside some camera's support -- this walks outward deterministically and
    keeps only points verified to have zero relevant views."""

    from .shared import RELEVANCE_OK, project_queries

    accepted: list[torch.Tensor] = []
    inspected = 0
    for multiple in VERIFIED_CONTROL_EXTENT_MULTIPLES:
        for axis in range(3):
            for sign in (-1.0, 1.0):
                offset = torch.zeros(3, device=device)
                offset[axis] = sign * multiple * scene_extent
                point = (scene_centre + offset).reshape(1, 3)
                inspected += 1
                relevant = 0
                for camera in cameras:
                    geometry = project_queries(camera, point)
                    relevant += int((geometry.relevance_code == RELEVANCE_OK).sum().item())
                    if relevant:
                        break
                if relevant == 0:
                    accepted.append(point.reshape(3))
                if len(accepted) >= VERIFIED_CONTROL_TARGET:
                    break
            if len(accepted) >= VERIFIED_CONTROL_TARGET:
                break
        if len(accepted) >= VERIFIED_CONTROL_TARGET:
            break
    meta = {
        "candidates_inspected": inspected,
        "accepted": len(accepted),
        "extent_multiples": list(VERIFIED_CONTROL_EXTENT_MULTIPLES),
        "verification": "relevant-view count confirmed == 0 across all training cameras BEFORE candidate evaluation",
    }
    if not accepted:
        return torch.zeros((0, 3), dtype=torch.float32, device=device), meta
    return torch.stack(accepted).to(torch.float32), meta


def build_supplemental_bank(
    contexts: dict[str, np.ndarray],
    controls: torch.Tensor,
    region_of_surfel: torch.Tensor,
    device,
) -> tuple[QueryBank, dict[str, Any]]:
    """Three queries per retained context (endpoint A, endpoint B, midpoint),
    then the verified out-of-frustum controls."""

    world_a = torch.as_tensor(contexts["world_a"], dtype=torch.float32, device=device)
    world_b = torch.as_tensor(contexts["world_b"], dtype=torch.float32, device=device)
    midpoint = (world_a + world_b) * 0.5
    context_count = int(world_a.shape[0])

    positions = torch.cat([world_a, world_b, midpoint, controls.to(device)], dim=0)
    kinds = (
        [KIND_GAP_ENDPOINT_A] * context_count
        + [KIND_GAP_ENDPOINT_B] * context_count
        + [KIND_GAP_MIDPOINT] * context_count
        + [KIND_VERIFIED_OUT_OF_FRUSTUM] * int(controls.shape[0])
    )
    view = np.concatenate([
        contexts["view_index"], contexts["view_index"], contexts["view_index"],
        np.full(int(controls.shape[0]), -1, dtype=np.int64),
    ])
    surfel = np.concatenate([
        contexts["representative_a"], contexts["representative_b"], contexts["representative_a"],
        np.full(int(controls.shape[0]), -1, dtype=np.int64),
    ])
    region = np.full(surfel.shape[0], -1, dtype=np.int64)
    known = surfel >= 0
    if known.any():
        region[known] = region_of_surfel[torch.as_tensor(surfel[known], dtype=torch.int64, device=device)].cpu().numpy()

    bank = QueryBank(
        positions=positions.contiguous(),
        kind=kinds,
        source_view=view,
        source_surfel=surfel,
        region=region,
        ladder_step=np.full(surfel.shape[0], np.nan, dtype=np.float32),
        support_radius=np.full(surfel.shape[0], np.nan, dtype=np.float32),
    )
    sidecar = {
        "context_count": context_count,
        "endpoint_a_rows": [0, context_count],
        "endpoint_b_rows": [context_count, 2 * context_count],
        "midpoint_rows": [2 * context_count, 3 * context_count],
        "control_rows": [3 * context_count, 3 * context_count + int(controls.shape[0])],
        "per_context": {
            key: contexts[key].tolist() for key in
            ("view_index", "row_a", "col_a", "row_b", "col_b", "representative_a",
             "representative_b", "component_a", "component_b", "gating_reason", "region")
        },
    }
    return bank, sidecar
