from __future__ import annotations

"""Worklog 103 -- Positively Observation-Supported Visible Adjacency.

Worklog 102 kept the underlying philosophy of "maximal connectivity of
actually observed-visible surface evidence" but built it on the wrong graph:

    spatial kNN candidate edges  MINUS  contradiction cuts
        -> visible topology.

Only 25.3% of spatial candidate edges (1,295,809 / 5,132,180) ever had a
camera co-observe both endpoints at all; the remaining 74.7% survived by
DEFAULT, with no positive evidence they belong to one continuously visible
surface. That is exactly why the real scene percolated to 92.69% -- the
majority of "visible" topology was never actually verified as visible.

This module keeps the same principle and REPLACES the graph:

    spatial kNN            -> a CANDIDATE RELATION GENERATOR ONLY,
                               never itself visible-surface topology
        -> per view, does the camera's own observed-visible surface field
           positively support a continuous relation between i and j?
        -> aggregate across views WITHOUT a smooth-fraction threshold:
             positive support AND no hard contradiction -> visible adjacency
             hard contradiction (no positive counter-evidence) -> cut
             both positive and contradictory evidence exist -> unresolved
             neither ever observed -> UNKNOWN (not an edge, not a cut)
        -> secondary geometric-discontinuity gate (Worklog 98's own reused
           per-edge test, corrected positional-offset sign -- see below)
        -> connected components of the SURVIVING POSITIVE edges only

    NOT CONTRADICTED  =/=>  VISIBLE CONNECTED.
    POSITIVELY OBSERVED AS VISIBLE-CONTINUOUS  =>  visible adjacency may exist.

Worklog 102 (`torch_maximal_visible_connectivity.py`) is preserved UNMODIFIED
as a historical/comparison baseline (review export view H). This module
reuses its already-CORRECT per-view classification helpers
(`_per_view_status_codes`, `_project_to_camera`) by import, and reuses
Worklog 98's shape-operator/residual machinery for the secondary
discontinuity gate -- but does NOT reuse Worklog 98/99/100's own
positional-offset formula, which Worklog 102 found to have a real sign bug
(uses the ABS of a signed dot product where it should use the signed value,
doubling instead of cancelling the normal-direction component when
negative). This module uses the CORRECTED signed-projection version (the
same fix Worklog 102 already applied to its own copy) from the start.

Endpoint-depth-range caveat (directive section 13): the range bound
`[min(depth_i,depth_j) - edge_length - epsilon, max(...) + edge_length +
epsilon]` is a NECESSARY bound for the specific two-point pair and their own
observed depths -- it is not claimed here as a universal theorem about
arbitrary smooth surface arcs in general. Its correctness for the curved
fixtures in this module's own test suite is reported as operational
evidence, not a general proof.

Role separation and canonical Phase-C semantics are identical to Worklog
102's own contract (see that module's docstring); `torch_observation_evidence.py`
is not modified by this module either.
"""

from dataclasses import dataclass, field
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
from osn_gs.surface.torch_maximal_visible_connectivity import (
    _per_view_status_codes,
    _project_to_camera,
)
from osn_gs.surface.torch_observation_evidence import ObservationEvidence
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# --- explicit per-edge relation states (directive section 5) ---
STATE_POSITIVE_VISIBLE_CONTINUATION = "POSITIVE_VISIBLE_CONTINUATION"
STATE_CUT_KNOWN_FREE_SPACE = "CUT_KNOWN_FREE_SPACE"
STATE_CUT_OCCLUDED_DOMAIN = "CUT_OCCLUDED_DOMAIN"
STATE_CUT_VISIBLE_DISCONTINUITY = "CUT_VISIBLE_GEOMETRIC_DISCONTINUITY"
STATE_CUT_POSITIONAL_SHEET_SEPARATION = "CUT_POSITIONAL_SHEET_SEPARATION"
STATE_UNRESOLVED_CONFLICT = "UNRESOLVED_OBSERVATION_CONFLICT"
STATE_UNKNOWN_NO_POSITIVE_OBSERVATION = "UNKNOWN_NO_POSITIVE_OBSERVATION"

RELATION_STATES = (
    STATE_POSITIVE_VISIBLE_CONTINUATION,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_UNRESOLVED_CONFLICT,
    STATE_UNKNOWN_NO_POSITIVE_OBSERVATION,
)


@dataclass(frozen=True)
class PositiveVisibleAdjacencyConfig:
    """Geometric-candidate-graph fields are REUSED VERBATIM from Worklog
    96-102 (`CoverageFirstPartitionConfig`). The screen-walk interior-sample
    count reuses Phase E's own default (3, `torch_candidate_evidence`).
    `residual_mad_multiplier`/`parallel_sheet_normal_over_tangent_ratio` are
    reused verbatim from Worklog 98/100/102 for the SECONDARY geometric-
    discontinuity gate (section 7) -- normal direction alone is never a cut.
    No smooth-fraction/majority threshold exists anywhere in this module
    (directive section 6): this batch tests positive-observation semantics,
    not another voting heuristic.
    """

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    shape_operator_neighbor_count: int = 0  # 0 => local.neighbor_count
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0  # reused verbatim from Worklog 98/100/102
    parallel_sheet_normal_over_tangent_ratio: float = 1.0  # reused verbatim
    screen_interior_samples: int = 3  # reused verbatim from Phase E's own default

    def resolved_shape_operator_neighbor_count(self) -> int:
        return int(self.shape_operator_neighbor_count) or int(self.local.neighbor_count)

    def payload(self) -> dict[str, Any]:
        return {
            "local": self.local.payload(),
            "shape_operator_neighbor_count": self.resolved_shape_operator_neighbor_count(),
            "shape_operator_ridge": self.shape_operator_ridge,
            "residual_mad_multiplier": self.residual_mad_multiplier,
            "parallel_sheet_normal_over_tangent_ratio": self.parallel_sheet_normal_over_tangent_ratio,
            "screen_interior_samples": self.screen_interior_samples,
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class PositiveVisibleAdjacencyResult:
    subset_ids: Any  # (N,) int64 -- exactly one Visible Surface Component owner per surfel
    subset_count: int
    subset_sizes: Any  # (subset_count,) int64, descending

    graph: CandidateGraph
    gaussian_ids: Any

    relation_state: Any  # (E_c,) int8 index into RELATION_STATES, spatial edges only elsewhere = UNKNOWN
    positive_visible_edges_mask: Any  # (E_c,) bool -- the edges connectivity is actually built from
    normal_gradient_magnitude: Any  # (N,) float -- diagnostic only
    residual_threshold: float
    config: PositiveVisibleAdjacencyConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def kept_edges(self) -> Any:
        return self.graph.candidate_edges[self.positive_visible_edges_mask]


def _region_state_mask(relation_state: Any, state_name: str) -> Any:
    return relation_state == RELATION_STATES.index(state_name)


def compute_positive_visible_adjacency_evidence(
    orientation: SurfaceOrientationEvidence,
    observation_evidence: ObservationEvidence | None,
    graph: CandidateGraph,
    config: PositiveVisibleAdjacencyConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The core evaluation: classifies every SPATIAL candidate edge into
    exactly one of `RELATION_STATES`. Returns `relation_state` ((E_c,) int8)
    plus supporting diagnostics. `observation_evidence=None` means every
    spatial edge stays `UNKNOWN_NO_POSITIVE_OBSERVATION` (no camera data at
    all -- correctly produces zero visible adjacency, not a silent default
    connection).
    """

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    count = int(positions.shape[0])
    device = positions.device
    candidate_edges = graph.candidate_edges
    spatial_mask = graph.spatial_edge_mask
    edge_count = int(candidate_edges.shape[0])

    relation_state = torch.full(
        (edge_count,), RELATION_STATES.index(STATE_UNKNOWN_NO_POSITIVE_OBSERVATION), dtype=torch.int8, device=device
    )

    any_positive = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    any_free = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    any_occluded = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    ever_evaluated = torch.zeros((edge_count,), dtype=torch.bool, device=device)

    if observation_evidence is not None and edge_count > 0 and int(spatial_mask.sum()) > 0:
        spatial_index = torch.nonzero(spatial_mask, as_tuple=False).reshape(-1)
        edge_left_all = candidate_edges[spatial_index, 0]
        edge_right_all = candidate_edges[spatial_index, 1]
        n_interior = max(1, int(config.screen_interior_samples))
        fracs = torch.linspace(0.0, 1.0, n_interior + 2)[1:-1]

        for view in observation_evidence.views:
            proj = _project_to_camera(positions, view)
            status = _per_view_status_codes(
                proj["view_depth"], proj["observed_depth"], proj["valid_at_pixel"], proj["in_bounds"],
                observation_evidence.near, observation_evidence.far, observation_evidence.depth_epsilon,
            )
            on_surface_this_view = status == 3  # VIEW_STATUS_ON_OBSERVED_SURFACE code, see _per_view_status_codes

            co_observed = on_surface_this_view[edge_left_all] & on_surface_this_view[edge_right_all]
            if not bool(co_observed.any()):
                continue

            local_indices = torch.nonzero(co_observed, as_tuple=False).reshape(-1)
            ever_evaluated[spatial_index[local_indices]] = True

            left_idx = edge_left_all[co_observed]
            right_idx = edge_right_all[co_observed]
            row_left, col_left = proj["pixel_row"][left_idx], proj["pixel_col"][left_idx]
            row_right, col_right = proj["pixel_row"][right_idx], proj["pixel_col"][right_idx]
            depth_left = proj["view_depth"][left_idx]
            depth_right = proj["view_depth"][right_idx]
            edge_length = (positions[right_idx] - positions[left_idx]).norm(dim=-1)
            depth_low = torch.minimum(depth_left, depth_right) - edge_length - observation_evidence.depth_epsilon
            depth_high = torch.maximum(depth_left, depth_right) + edge_length + observation_evidence.depth_epsilon

            edge_free = torch.zeros_like(depth_left, dtype=torch.bool)
            edge_occluded = torch.zeros_like(depth_left, dtype=torch.bool)
            edge_any_invalid = torch.zeros_like(depth_left, dtype=torch.bool)

            for frac in fracs.tolist():
                sample_row = (row_left + frac * (row_right - row_left)).round().long().clamp(0, view.image_height - 1)
                sample_col = (col_left + frac * (col_right - col_left)).round().long().clamp(0, view.image_width - 1)
                sample_valid = view.valid_depth_mask[sample_row, sample_col]
                sample_observed_depth = view.view_depth[sample_row, sample_col]
                edge_occluded = edge_occluded | (sample_valid & (sample_observed_depth < depth_low))
                edge_free = edge_free | (sample_valid & (sample_observed_depth > depth_high))
                edge_any_invalid = edge_any_invalid | ~sample_valid

            # POSITIVE evidence: every interior sample was VALID and stayed
            # within the plausible range -- the camera actually observed a
            # continuous visible surface field along the whole screen path,
            # not merely "no contradiction found".
            edge_positive = (~edge_free) & (~edge_occluded) & (~edge_any_invalid)

            any_positive[spatial_index[local_indices]] = any_positive[spatial_index[local_indices]] | edge_positive
            any_free[spatial_index[local_indices]] = any_free[spatial_index[local_indices]] | edge_free
            any_occluded[spatial_index[local_indices]] = any_occluded[spatial_index[local_indices]] | edge_occluded

            if progress is not None:
                progress(
                    f"[camera {view.camera_index}] co_observed={int(co_observed.sum())} "
                    f"positive={int(edge_positive.sum())} free={int(edge_free.sum())} occluded={int(edge_occluded.sum())}"
                )

    any_contradiction = any_free | any_occluded
    conflict = any_positive & any_contradiction
    only_free = any_free & ~any_occluded & ~conflict
    only_occluded = any_occluded & ~conflict
    positive_clean = any_positive & ~any_contradiction

    relation_state = torch.where(conflict, torch.full_like(relation_state, RELATION_STATES.index(STATE_UNRESOLVED_CONFLICT)), relation_state)
    relation_state = torch.where(only_occluded, torch.full_like(relation_state, RELATION_STATES.index(STATE_CUT_OCCLUDED_DOMAIN)), relation_state)
    relation_state = torch.where(only_free & ~only_occluded, torch.full_like(relation_state, RELATION_STATES.index(STATE_CUT_KNOWN_FREE_SPACE)), relation_state)
    relation_state = torch.where(positive_clean, torch.full_like(relation_state, RELATION_STATES.index(STATE_POSITIVE_VISIBLE_CONTINUATION)), relation_state)
    # everything else (never evaluated, or evaluated but every sample was
    # invalid so neither positive nor contradictory evidence exists) stays
    # at its initial UNKNOWN_NO_POSITIVE_OBSERVATION value.

    # --- secondary geometric-discontinuity gate (Worklog 98's own reused
    # shape-operator/residual test), applied ONLY to edges that are
    # currently POSITIVE_VISIBLE_CONTINUATION -- normal direction alone is
    # never a cut (directive section 7); this only demotes an edge that was
    # positively co-visible but geometrically inconsistent. ---
    cut_residual = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    cut_positional = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    normal_gradient_magnitude = torch.zeros((count,), dtype=torch.float32, device=device)
    residual_threshold = 0.0

    positive_mask_pre_geometry = relation_state == RELATION_STATES.index(STATE_POSITIVE_VISIBLE_CONTINUATION)
    if count > 0 and bool(positive_mask_pre_geometry.any()):
        tangent_u = orientation.tangent_axis_u
        tangent_v = orientation.tangent_axis_v
        k_shape = min(config.resolved_shape_operator_neighbor_count(), max(count - 1, 1))
        chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
        if progress is not None:
            progress(f"fitting local shape operators: k={k_shape}")
        neighbor_index, _ = _knn(positions, k_shape, chunk_size, progress)
        shape_operator = _fit_shape_operators(
            positions, normals, tangent_u, tangent_v, neighbor_index, float(config.shape_operator_ridge)
        )
        normal_gradient_magnitude = torch.linalg.matrix_norm(shape_operator)

        positive_index = torch.nonzero(positive_mask_pre_geometry, as_tuple=False).reshape(-1)
        edges_p = candidate_edges[positive_index]
        left, right = edges_p[:, 0], edges_p[:, 1]
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
        # CORRECTED signed projection (Worklog 102's fix, section 13): using
        # the abs value here (as Worklog 98/99/100's own copy does) would
        # double instead of cancel the normal-direction component whenever
        # the signed dot product is negative.
        signed_normal_offset = (delta_x * average_normal).sum(dim=-1)
        normal_offset = signed_normal_offset.abs()
        tangential_offset = (delta_x - signed_normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
        normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)

        median_residual = torch.median(edge_residual)
        mad = torch.median((edge_residual - median_residual).abs())
        residual_threshold = float(median_residual + config.residual_mad_multiplier * 1.4826 * mad)

        fails_residual = edge_residual > residual_threshold
        fails_positional = normal_offset_ratio > config.parallel_sheet_normal_over_tangent_ratio

        cut_residual[positive_index] = fails_residual
        cut_positional[positive_index] = fails_positional
        demoted = fails_residual | fails_positional
        relation_state[positive_index[demoted & fails_residual]] = RELATION_STATES.index(STATE_CUT_VISIBLE_DISCONTINUITY)
        # positional failure takes label priority when both fire on the same
        # edge (matches Worklog 98/102's own "both reasons recorded, cut
        # either way" spirit) -- report both booleans regardless of label.
        relation_state[positive_index[fails_positional]] = RELATION_STATES.index(STATE_CUT_POSITIONAL_SHEET_SEPARATION)

    positive_visible_edges_mask = relation_state == RELATION_STATES.index(STATE_POSITIVE_VISIBLE_CONTINUATION)

    return {
        "relation_state": relation_state,
        "positive_visible_edges_mask": positive_visible_edges_mask,
        "any_positive": any_positive,
        "any_free": any_free,
        "any_occluded": any_occluded,
        "ever_evaluated": ever_evaluated,
        "cut_reason_residual": cut_residual,
        "cut_reason_positional": cut_positional,
        "normal_gradient_magnitude": normal_gradient_magnitude,
        "residual_threshold": residual_threshold,
    }


def partition_positive_visible_adjacency(
    orientation: SurfaceOrientationEvidence,
    observation_evidence: ObservationEvidence | None,
    config: PositiveVisibleAdjacencyConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> PositiveVisibleAdjacencyResult:
    """Partition EVERY surfel into exactly one Visible Surface Component,
    where connectivity is built ONLY from candidate edges that received
    POSITIVE observation support (never from mere absence of contradiction).
    See the module docstring for the full contract.
    """

    torch = require_torch()
    config = config or PositiveVisibleAdjacencyConfig()
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)

    if count == 0:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_edges_bool = torch.zeros((0,), dtype=torch.bool, device=device)
        empty_edges_state = torch.zeros((0,), dtype=torch.int8, device=device)
        return PositiveVisibleAdjacencyResult(
            subset_ids=empty_long, subset_count=0, subset_sizes=empty_long,
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            relation_state=empty_edges_state, positive_visible_edges_mask=empty_edges_bool,
            normal_gradient_magnitude=torch.zeros((0,), dtype=torch.float32, device=device),
            residual_threshold=0.0, config=config,
        )

    evidence = compute_positive_visible_adjacency_evidence(orientation, observation_evidence, graph, config, progress=progress)

    kept = graph.candidate_edges[evidence["positive_visible_edges_mask"]]
    roots = _connected_component_roots(count, kept, config.local)
    unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order)
    subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order]

    return PositiveVisibleAdjacencyResult(
        subset_ids=subset_ids, subset_count=int(order.shape[0]), subset_sizes=subset_sizes,
        graph=graph, gaussian_ids=orientation.gaussian_ids,
        relation_state=evidence["relation_state"], positive_visible_edges_mask=evidence["positive_visible_edges_mask"],
        normal_gradient_magnitude=evidence["normal_gradient_magnitude"],
        residual_threshold=evidence["residual_threshold"], config=config,
    )


def count_spatially_disconnected_subsets(result: PositiveVisibleAdjacencyResult) -> int:
    torch = require_torch()
    count = len(result)
    if count == 0:
        return 0
    roots = _connected_component_roots(count, result.kept_edges, result.config.local)
    unique_pairs = torch.unique(result.subset_ids * int(count) + roots)
    subset_of_pair = torch.div(unique_pairs, count, rounding_mode="floor")
    components_per_subset = torch.bincount(subset_of_pair, minlength=max(result.subset_count, 1))
    return int((components_per_subset > 1).sum())


def positive_visible_adjacency_accounting(result: PositiveVisibleAdjacencyResult) -> dict[str, Any]:
    torch = require_torch()
    count = len(result)
    sizes = result.subset_sizes
    subset_count = max(result.subset_count, 1)

    owner_histogram = torch.bincount(result.subset_ids.reshape(-1), minlength=result.subset_count)
    assigned = int((result.subset_ids >= 0).sum())
    in_range = int(((result.subset_ids >= 0) & (result.subset_ids < max(result.subset_count, 1))).sum())
    sizes_match = bool(
        int(sizes.shape[0]) == int(owner_histogram.shape[0]) and torch.equal(owner_histogram.to(sizes.dtype), sizes)
    )

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

    spatial_mask = result.graph.spatial_edge_mask
    candidate_edge_count = int(result.graph.candidate_edges.shape[0])
    spatial_edge_count = int(spatial_mask.sum())

    state_counts = {
        name: int((result.relation_state[spatial_mask] == index).sum())
        for index, name in enumerate(RELATION_STATES)
    }

    return {
        "input_surfel_count": count,
        "assigned_surfel_count": assigned,
        "unassigned_surfel_count": count - assigned,
        "multiply_owned_surfel_count": 0,
        "subset_id_out_of_range_count": count - in_range,
        "subset_sizes_match_ownership_map": sizes_match,
        "coverage_identity_holds": bool(
            assigned == count and in_range == count and sizes_match
            and (int(sizes.sum()) if int(sizes.shape[0]) > 0 else 0) == count
        ),
        "visible_component_count": result.subset_count,
        "component_size": size_stats,
        "largest_component_size": int(sizes[0]) if int(sizes.shape[0]) > 0 else 0,
        "largest_component_surfel_fraction": largest_fraction,
        "singleton_surfel_count": singleton,
        "singleton_surfel_fraction": (singleton / count) if count > 0 else 0.0,
        "spatially_disconnected_component_count": count_spatially_disconnected_subsets(result),
        "candidate_edge_count": candidate_edge_count,
        "spatial_edge_count": spatial_edge_count,
        "relation_state_counts": state_counts,
        "positive_visible_edge_count": int(result.positive_visible_edges_mask.sum()),
        "residual_threshold": result.residual_threshold,
        "partition_parameters": result.config.payload(),
    }
