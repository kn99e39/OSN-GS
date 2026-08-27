from __future__ import annotations

"""Worklog 107 -- Camera-Induced Visible Adjacency.

Worklog 106 falsified the architecture "3D candidate edge -> ask each camera
whether the pair is continuous -> aggregate votes/contradictions": of WL103's
720,052 singleton-but-actually-contributing surfels, 96.9% of those that
stayed singleton in WL106 did so because of multi-view OBSERVATION CONFLICT/
CONTRADICTION, not because no co-contributing 3D neighbor existed. The
pairwise-corridor relation itself -- built from an ASSUMED surface passing
through the two candidate surfels' own centers/contribution status -- is the
bottleneck, not evidence sparsity.

This module inverts the construction direction (directive's central intent):

    OLD (WL103/106): 3D candidate edge -> camera approves/rejects the pair.
    NEW (this batch): camera's own rendered surface -> GENERATES local
                       surfel adjacency directly from image-space structure.

Two semantic corrections, both mandatory (directive's Central Intent A/B):

  A. No center-depth RANGE corridor. WL105 already proved a surfel can
     genuinely contribute without its center ever matching the rendered
     depth -- comparing a candidate edge against an assumed surface through
     two surfel CENTERS is not renderer-grounded. This module never computes
     or compares against `min/max(center_depth_i, center_depth_j)` at all.

  B. View-dependent occlusion is not a global veto. If view A positively
     observes surfels i and j as image-adjacent surface, and view B simply
     cannot see that relation (occluded, out of frame, or the pixels are not
     both valid there), view B contributes NO EDGE -- it does not invalidate
     view A's positive observation. Global topology is the UNION of every
     view's own positive relations, never an intersection or a conflict
     check across views. There is no `UNRESOLVED_OBSERVATION_CONFLICT` state
     in this module at all -- it is structurally impossible to construct
     one, since a view that disagrees simply never contributes that edge in
     the first place (it only ever contributes ITS OWN relations).

Per-view pipeline:

    1. IMAGE-SPACE ADJACENCY (this module, new): the renderer's own per-pixel
       surface-REPRESENTATIVE identity (Worklog 105's diagnostic-only
       `render_with_pixel_representative`, reusing the official renderCUDA's
       own `median_contributor`/T=0.5-crossing semantics -- see that module's
       docstring for exactly what is and is not exposed) is compared at
       minimal 4-connectivity (right/down) raster-grid neighbors. Two
       DIFFERENT representative surfels at adjacent, both-valid pixels are a
       positive image-space relation IN THAT VIEW. No pixel-radius parameter,
       no depth comparison, no assumed surface -- purely "the renderer itself
       resolved a continuous surface across this raster edge, and two
       different primitives were each other's neighboring representative."
    2. 3D LOCALITY FILTER (reused unchanged): the image-space relation must
       ALSO already be a spatial candidate edge in Worklog 96-106's own
       `build_candidate_graph` output. This is a SANITY/RESTRICTION role
       only -- it never independently generates topology (directive section
       6) -- it exists so that two primitives that happen to render as
       image-adjacent representatives (rare coincidence at large 3D
       distances, e.g. through a thin gap) don't create an obviously
       non-local edge.
    3. SECONDARY GEOMETRIC GATE (reused unchanged from Worklog 98/102/103/
       106): the same shape-operator residual + corrected signed positional-
       offset formula, applied ONCE to the union of all locality-passed
       pairs across all views (a purely 3D/orientation test, independent of
       which view generated the candidate). Normal-direction difference
       alone is never a cut; smooth curved surfaces remain eligible for one
       component.
    4. UNION ACROSS VIEWS: a pair that survives steps 2-3 in >=1 view is a
       final positive camera-induced visible edge. No percentage-of-views
       threshold, no majority vote -- directive section 9.

Worklog 103 (`torch_positive_visible_adjacency.py`), Worklog 104, Worklog 105,
and Worklog 106 (`torch_renderer_grounded_visible_adjacency.py`) are all left
completely unmodified and remain separately replayable as
`PAIRWISE_CAMERA_APPROVAL_BASELINE`.
"""

from dataclasses import dataclass
from typing import Any, Callable

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CandidateGraph,
    CoverageFirstPartitionConfig,
    SurfaceOrientationEvidence,
    VERY_SMALL_SUBSET_SIZE,
    _connected_component_roots,
    build_candidate_graph,
)
from osn_gs.surface.torch_discontinuity_first_surfel_partition import (
    _auto_chunk_size,
    _fit_shape_operators,
    _knn,
    _predicted_delta_n_t,
    _tangent_plane_components,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

REASON_GEOMETRIC_DISCONTINUITY = "GEOMETRIC_DISCONTINUITY"
REASON_POSITIONAL_SHEET_SEPARATION = "POSITIONAL_SHEET_SEPARATION"
REJECTION_REASONS = (REASON_GEOMETRIC_DISCONTINUITY, REASON_POSITIONAL_SHEET_SEPARATION)


@dataclass(frozen=True)
class CameraInducedAdjacencyConfig:
    """3D-locality candidate graph and secondary geometric-gate parameters
    reused verbatim from Worklog 96-106 -- no new corridor/threshold
    parameter exists in this module. `min_pixel_neighbor_connectivity` is
    fixed at 4 (right + down raster neighbors, the minimal connectivity that
    still covers every adjacent-pixel pair exactly once per image) and is
    not exposed as a tunable field, per directive section 5's explicit
    instruction not to introduce a tunable pixel-radius neighborhood."""

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    shape_operator_neighbor_count: int = 0
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0
    parallel_sheet_normal_over_tangent_ratio: float = 1.0

    def resolved_shape_operator_neighbor_count(self) -> int:
        return int(self.shape_operator_neighbor_count) or int(self.local.neighbor_count)

    def payload(self) -> dict[str, Any]:
        return {
            "local": self.local.payload(),
            "shape_operator_neighbor_count": self.resolved_shape_operator_neighbor_count(),
            "shape_operator_ridge": self.shape_operator_ridge,
            "residual_mad_multiplier": self.residual_mad_multiplier,
            "parallel_sheet_normal_over_tangent_ratio": self.parallel_sheet_normal_over_tangent_ratio,
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class CameraInducedAdjacencyResult:
    subset_ids: Any
    subset_count: int
    subset_sizes: Any

    graph: CandidateGraph
    gaussian_ids: Any

    positive_visible_edges: Any  # (E, 2) int64 -- final unique camera-induced edges, (min, max) surfel-id pairs
    raw_image_space_pair_count: int  # total (view, pair) observations before dedup, before any filtering
    locality_rejected_pair_count: int  # distinct pairs seen in image space but never a 3D candidate edge
    geometric_rejected_pairs: Any  # (E', 2) -- locality-passed pairs rejected by the geometric gate
    geometric_rejection_reason: Any  # (E',) int8 index into REJECTION_REASONS
    view_support_count: Any  # (E,) int32 -- number of views that generated each final positive edge
    normal_gradient_magnitude: Any
    residual_threshold: float
    config: CameraInducedAdjacencyConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])


def accumulate_image_space_pairs(
    count: int,
    per_view_representative_ids: list[Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    """Step 1 + UNION ACROSS VIEWS (directive section 9), combined: for every
    view, compares minimal 4-connectivity raster neighbors of the renderer's
    own per-pixel surface-representative map (already remapped to the
    caller's surfel-index space, -1 = no representative at that pixel) and
    accumulates every DISTINCT-surfel adjacent pair, together with how many
    views generated it. Returns `(unique_pairs (E,2) int64, view_support_count
    (E,) int32)`. A view with no valid representatives anywhere contributes
    nothing (absence of evidence, never a contradiction) -- see module
    docstring correction B.
    """

    torch = require_torch()
    if count == 0 or not per_view_representative_ids:
        empty = torch.zeros((0, 2), dtype=torch.int64)
        return empty, torch.zeros((0,), dtype=torch.int32)

    device = per_view_representative_ids[0].device
    all_pairs: list[Any] = []
    all_view_index: list[Any] = []

    for view_index, rep in enumerate(per_view_representative_ids):
        rep = rep.to(device=device, dtype=torch.int64)
        valid = rep >= 0

        left_ids, right_ids = rep[:, :-1], rep[:, 1:]
        both_valid_h = valid[:, :-1] & valid[:, 1:]
        differ_h = both_valid_h & (left_ids != right_ids)
        pairs_h = torch.stack([left_ids[differ_h], right_ids[differ_h]], dim=1)

        top_ids, bottom_ids = rep[:-1, :], rep[1:, :]
        both_valid_v = valid[:-1, :] & valid[1:, :]
        differ_v = both_valid_v & (top_ids != bottom_ids)
        pairs_v = torch.stack([top_ids[differ_v], bottom_ids[differ_v]], dim=1)

        pairs = torch.cat([pairs_h, pairs_v], dim=0)
        if int(pairs.shape[0]) == 0:
            if progress is not None:
                progress(f"[view {view_index}] no image-space adjacent-distinct-representative pairs")
            continue
        normalized = torch.stack([torch.minimum(pairs[:, 0], pairs[:, 1]), torch.maximum(pairs[:, 0], pairs[:, 1])], dim=1)
        normalized = torch.unique(normalized, dim=0)
        all_pairs.append(normalized)
        all_view_index.append(torch.full((int(normalized.shape[0]),), view_index, dtype=torch.int32))
        if progress is not None:
            progress(f"[view {view_index}] {int(normalized.shape[0])} distinct image-space pairs")

    if not all_pairs:
        empty = torch.zeros((0, 2), dtype=torch.int64, device=device)
        return empty, torch.zeros((0,), dtype=torch.int32, device=device)

    combined = torch.cat(all_pairs, dim=0)
    keys = combined[:, 0] * count + combined[:, 1]
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    unique_pairs = torch.stack([unique_keys // count, unique_keys % count], dim=1)
    view_support_count = torch.zeros((int(unique_keys.shape[0]),), dtype=torch.int32, device=device)
    view_support_count.index_add_(0, inverse, torch.ones((int(combined.shape[0]),), dtype=torch.int32, device=device))
    return unique_pairs, view_support_count


def filter_by_3d_locality(pairs: Any, count: int, graph: CandidateGraph) -> tuple[Any, Any]:
    """Step 2 (directive section 6): the 3D candidate graph is a RESTRICTION
    only -- an image-space pair survives iff it is ALSO an existing spatial
    candidate edge. Returns `(kept_pairs, kept_mask)`; `kept_mask` indexes
    into the input `pairs`."""

    torch = require_torch()
    if int(pairs.shape[0]) == 0:
        return pairs, torch.zeros((0,), dtype=torch.bool, device=pairs.device)

    spatial_edges = graph.candidate_edges[graph.spatial_edge_mask]
    if int(spatial_edges.shape[0]) == 0:
        return pairs[:0], torch.zeros((int(pairs.shape[0]),), dtype=torch.bool, device=pairs.device)

    normalized_graph = torch.stack(
        [torch.minimum(spatial_edges[:, 0], spatial_edges[:, 1]), torch.maximum(spatial_edges[:, 0], spatial_edges[:, 1])], dim=1
    )
    graph_keys = torch.unique(normalized_graph[:, 0] * count + normalized_graph[:, 1])
    query_keys = pairs[:, 0] * count + pairs[:, 1]
    position = torch.searchsorted(graph_keys, query_keys)
    position = position.clamp(max=int(graph_keys.shape[0]) - 1 if int(graph_keys.shape[0]) > 0 else 0)
    kept_mask = (int(graph_keys.shape[0]) > 0) & (graph_keys[position] == query_keys)
    return pairs[kept_mask], kept_mask


def apply_secondary_geometric_gate(
    pairs: Any,
    orientation: SurfaceOrientationEvidence,
    config: CameraInducedAdjacencyConfig,
    *,
    neighbor_index: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Step 3 (directive section 8): reused verbatim from Worklog 98/102/103/
    106 -- shape-operator residual + corrected signed positional-offset
    formula. Applied ONCE to the union of all locality-passed pairs (a pure
    3D/orientation test, independent of which view generated the candidate).
    Normal-direction difference alone is never sufficient for a cut."""

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    count = int(positions.shape[0])
    device = positions.device
    edge_count = int(pairs.shape[0])

    fails_residual = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    fails_positional = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    normal_gradient_magnitude = torch.zeros((count,), dtype=torch.float32, device=device)
    residual_threshold = 0.0

    if count > 0 and edge_count > 0:
        tangent_u = orientation.tangent_axis_u
        tangent_v = orientation.tangent_axis_v
        k_shape = min(config.resolved_shape_operator_neighbor_count(), max(count - 1, 1))
        chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
        can_reuse_neighbors = (
            neighbor_index is not None
            and tuple(neighbor_index.shape) == (count, k_shape)
            and neighbor_index.device == device
        )
        if progress is not None:
            source = "reused candidate-graph kNN" if can_reuse_neighbors else "fresh kNN"
            progress(f"fitting local shape operators: k={k_shape} ({source})")
        if not can_reuse_neighbors:
            neighbor_index, _ = _knn(positions, k_shape, chunk_size, progress)
        shape_operator = _fit_shape_operators(positions, normals, tangent_u, tangent_v, neighbor_index, float(config.shape_operator_ridge))
        normal_gradient_magnitude = torch.linalg.matrix_norm(shape_operator)

        left, right = pairs[:, 0], pairs[:, 1]
        delta_x = positions[right] - positions[left]

        delta_x_t_left = _tangent_plane_components(delta_x, tangent_u[left], tangent_v[left])
        delta_x_t_right = _tangent_plane_components(-delta_x, tangent_u[right], tangent_v[right])
        sign_lr = torch.where((normals[left] * normals[right]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
        aligned_right_normal = normals[right] * sign_lr
        delta_n_left = aligned_right_normal - normals[left]
        delta_n_t_left = _tangent_plane_components(delta_n_left, tangent_u[left], tangent_v[left])
        aligned_left_normal = normals[left] * sign_lr
        delta_n_right = aligned_left_normal - normals[right]
        delta_n_t_right = _tangent_plane_components(delta_n_right, tangent_u[right], tangent_v[right])

        predicted_left = _predicted_delta_n_t(shape_operator[left], delta_x_t_left)
        predicted_right = _predicted_delta_n_t(shape_operator[right], delta_x_t_right)
        residual_left = (delta_n_t_left - predicted_left).norm(dim=-1)
        residual_right = (delta_n_t_right - predicted_right).norm(dim=-1)
        edge_residual = torch.minimum(residual_left, residual_right)

        average_normal = torch.nn.functional.normalize(normals[left] + aligned_right_normal, dim=-1, eps=_EPS)
        signed_normal_offset = (delta_x * average_normal).sum(dim=-1)
        normal_offset = signed_normal_offset.abs()
        tangential_offset = (delta_x - signed_normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
        normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)

        median_residual = torch.median(edge_residual)
        mad = torch.median((edge_residual - median_residual).abs())
        residual_threshold = float(median_residual + config.residual_mad_multiplier * 1.4826 * mad)

        fails_residual = edge_residual > residual_threshold
        fails_positional = normal_offset_ratio > config.parallel_sheet_normal_over_tangent_ratio

    kept_mask = ~(fails_residual | fails_positional)
    return {
        "kept_mask": kept_mask,
        "fails_residual": fails_residual,
        "fails_positional": fails_positional,
        "normal_gradient_magnitude": normal_gradient_magnitude,
        "residual_threshold": residual_threshold,
    }


def partition_camera_induced_visible_adjacency(
    orientation: SurfaceOrientationEvidence,
    per_view_representative_ids: list[Any] | None,
    config: CameraInducedAdjacencyConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> CameraInducedAdjacencyResult:
    """Full pipeline: image-space pair generation + union -> 3D-locality
    filter -> secondary geometric gate -> connected components. See module
    docstring for the complete contract."""

    torch = require_torch()
    config = config or CameraInducedAdjacencyConfig()
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(
        orientation, config.local, retain_neighbor_index=True, progress=progress
    )

    if count == 0 or not per_view_representative_ids:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_edges = torch.zeros((0, 2), dtype=torch.int64, device=device)
        empty_reason = torch.zeros((0,), dtype=torch.int8, device=device)
        return CameraInducedAdjacencyResult(
            subset_ids=torch.arange(count, dtype=torch.int64, device=device), subset_count=count,
            subset_sizes=torch.ones((count,), dtype=torch.int64, device=device),
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            positive_visible_edges=empty_edges, raw_image_space_pair_count=0, locality_rejected_pair_count=0,
            geometric_rejected_pairs=empty_edges, geometric_rejection_reason=empty_reason,
            view_support_count=torch.zeros((0,), dtype=torch.int32, device=device),
            normal_gradient_magnitude=torch.zeros((count,), dtype=torch.float32, device=device),
            residual_threshold=0.0, config=config,
        )

    raw_pairs, raw_view_support = accumulate_image_space_pairs(count, per_view_representative_ids, progress=progress)
    raw_pair_count = int(raw_pairs.shape[0])

    local_pairs, local_mask = filter_by_3d_locality(raw_pairs, count, graph)
    local_view_support = raw_view_support[local_mask]
    locality_rejected = raw_pair_count - int(local_pairs.shape[0])

    geometry = apply_secondary_geometric_gate(
        local_pairs,
        orientation,
        config,
        neighbor_index=graph.neighbor_index,
        progress=progress,
    )
    kept_mask = geometry["kept_mask"]
    positive_edges = local_pairs[kept_mask]
    positive_view_support = local_view_support[kept_mask]
    rejected_mask = ~kept_mask
    rejected_pairs = local_pairs[rejected_mask]
    rejected_reason = torch.where(
        geometry["fails_residual"][rejected_mask],
        torch.full((int(rejected_mask.sum()),), REJECTION_REASONS.index(REASON_GEOMETRIC_DISCONTINUITY), dtype=torch.int8, device=device),
        torch.full((int(rejected_mask.sum()),), REJECTION_REASONS.index(REASON_POSITIONAL_SHEET_SEPARATION), dtype=torch.int8, device=device),
    )

    roots = _connected_component_roots(count, positive_edges, config.local)
    unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order)
    subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order]

    return CameraInducedAdjacencyResult(
        subset_ids=subset_ids, subset_count=int(order.shape[0]), subset_sizes=subset_sizes,
        graph=graph, gaussian_ids=orientation.gaussian_ids,
        positive_visible_edges=positive_edges, raw_image_space_pair_count=raw_pair_count,
        locality_rejected_pair_count=locality_rejected,
        geometric_rejected_pairs=rejected_pairs, geometric_rejection_reason=rejected_reason,
        view_support_count=positive_view_support,
        normal_gradient_magnitude=geometry["normal_gradient_magnitude"], residual_threshold=geometry["residual_threshold"],
        config=config,
    )


def camera_induced_visible_adjacency_accounting(result: CameraInducedAdjacencyResult) -> dict[str, Any]:
    torch = require_torch()
    count = len(result)
    sizes = result.subset_sizes

    owner_histogram = torch.bincount(result.subset_ids.reshape(-1), minlength=result.subset_count)
    assigned = int((result.subset_ids >= 0).sum())
    sizes_match = bool(int(sizes.shape[0]) == int(owner_histogram.shape[0]) and torch.equal(owner_histogram.to(sizes.dtype), sizes))

    size_stats: dict[str, Any] = {}
    if int(sizes.shape[0]) > 0:
        sorted_sizes = torch.sort(sizes).values.to(torch.float64)

        def _percentile(fraction: float) -> int:
            position = min(int(sorted_sizes.shape[0]) - 1, max(0, int(round(fraction * (int(sorted_sizes.shape[0]) - 1)))))
            return int(sorted_sizes[position].item())

        size_stats = {
            "min": int(sorted_sizes[0].item()), "median": _percentile(0.5),
            "mean": float(sorted_sizes.mean().item()), "p95": _percentile(0.95), "max": int(sorted_sizes[-1].item()),
        }

    singleton = int((sizes == 1).sum()) if int(sizes.shape[0]) > 0 else 0
    largest_fraction = (float(sizes[0]) / count) if (int(sizes.shape[0]) > 0 and count) else 0.0

    return {
        "input_surfel_count": count,
        "assigned_surfel_count": assigned,
        "unassigned_surfel_count": count - assigned,
        "coverage_identity_holds": bool(
            assigned == count and sizes_match and (int(sizes.sum()) if int(sizes.shape[0]) > 0 else 0) == count
        ),
        "visible_component_count": result.subset_count,
        "component_size": size_stats,
        "largest_component_size": int(sizes[0]) if int(sizes.shape[0]) > 0 else 0,
        "largest_component_surfel_fraction": largest_fraction,
        "singleton_surfel_count": singleton,
        "singleton_surfel_fraction": (singleton / count) if count > 0 else 0.0,
        "raw_image_space_pair_count": result.raw_image_space_pair_count,
        "locality_rejected_pair_count": result.locality_rejected_pair_count,
        "geometric_rejected_pair_count": int(result.geometric_rejected_pairs.shape[0]),
        "geometric_rejection_reason_counts": {
            name: int((result.geometric_rejection_reason == index).sum()) for index, name in enumerate(REJECTION_REASONS)
        },
        "final_positive_edge_count": int(result.positive_visible_edges.shape[0]),
        "view_support_count_distribution": (
            {
                "min": int(result.view_support_count.min()), "median": int(result.view_support_count.median()),
                "mean": float(result.view_support_count.float().mean()), "max": int(result.view_support_count.max()),
            }
            if int(result.view_support_count.shape[0]) > 0 else {"min": 0, "median": 0, "mean": 0.0, "max": 0}
        ),
        "residual_threshold": result.residual_threshold,
        "partition_parameters": result.config.payload(),
    }
