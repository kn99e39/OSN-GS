from __future__ import annotations

"""Worklog 106 -- Renderer-Grounded Visible Adjacency.

Worklog 105 proved that Phase-C's point-sample CENTER query is an inadequate
primitive-level visibility definition for trained 2DGS surfels: 95.4% of the
surfels Worklog 103/104 treated as "never visible" actually contribute to
the official rasterizer's accepted alpha-compositing across many training
views. This module keeps Worklog 103's central principle -- visible topology
must be positively grounded in observed-visible evidence, not merely absence
of contradiction -- but replaces the PRIMITIVE-LEVEL prerequisite:

    Phase-C center == on_observed_surface           (Worklog 103)
        ->
    official-renderer accepted contribution, per view   (this module)

Three concepts stay strictly separate (directive's own framing):

    1. RENDERER CONTRIBUTION  -- does this surfel actually participate in
       visible image formation in a given view? (Worklog 105's diagnostic
       signal, reused verbatim via `compute_renderer_contribution_for_view`,
       never re-derived here.)
    2. LOCAL CANDIDACY        -- is this pair spatially local enough to test
       at all? (Worklog 96-103's own candidate graph, reused unmodified via
       `build_candidate_graph` -- still only a candidate-relation generator,
       never itself topology.)
    3. VISIBLE SURFACE ADJACENCY -- does an actual training observation
       support these two CONTRIBUTING surfels as a locally continuous
       visible-surface relation? (this module's own per-view corridor test.)

Worklog 103 is NOT modified (`torch_positive_visible_adjacency.py` stays
untouched and remains separately replayable as baseline A). This module is a
structural sibling, not a subclass or wrapper: `compute_positive_visible_
adjacency_evidence`'s corridor/geometric-gate CODE is duplicated here rather
than imported, because the one thing that changes -- which pairs are even
eligible to be evaluated by the corridor test -- is threaded through the
exact same loop that also needs the screen-projection/sampling machinery, and
Worklog 103 does not expose a hook for substituting the eligibility test. The
corridor RANGE test itself, the multi-view aggregation semantics (no
percentage threshold, absence of observation is not a contradiction), and the
secondary geometric-discontinuity/positional-sheet gate (reused from Worklog
98/102/103, corrected signed-offset formula) are IDENTICAL to Worklog 103's
own -- see the module docstring comments inline at each point marking exactly
what changed and what did not.

What changed vs Worklog 103, precisely:
  - Endpoint eligibility for corridor evaluation: `on_observed_surface`
    (Phase-C center classification) -> `renderer_contributing` (Worklog 105
    diagnostic, official rasterizer accepted-alpha-compositing semantics).
  - Relation-state naming: `POSITIVE_VISIBLE_CONTINUATION` ->
    `POSITIVE_RENDERER_VISIBLE_CONTINUATION`; `UNKNOWN_NO_POSITIVE_
    OBSERVATION` -> `UNKNOWN_NO_RENDERER_SUPPORTED_RELATION` (directive
    section 7's own naming).
What did NOT change:
  - The range-based screen-walk corridor test (same formula, same interior-
    sample count, same depth_epsilon source).
  - `CUT_KNOWN_FREE_SPACE` / `CUT_OCCLUDED_DOMAIN` / `UNRESOLVED_OBSERVATION_
    CONFLICT` semantics (a co-contributing pair can still be cut by an actual
    occluder/free-space reading along the corridor -- directive section 9:
    positive contribution does not bridge an occluded gap).
  - The secondary geometric-discontinuity/positional-sheet gate (WL98/102/
    103's shape-operator residual + corrected signed positional-offset
    formula), applied only to pairs that already cleared the corridor test.
  - Multi-view aggregation: no percentage-of-cameras threshold anywhere;
    absence of co-contribution in a view is absence of evidence, not a
    contradiction.
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
from osn_gs.surface.torch_maximal_visible_connectivity import _project_to_camera
from osn_gs.surface.torch_observation_evidence import ObservationEvidence
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION = "POSITIVE_RENDERER_VISIBLE_CONTINUATION"
STATE_CUT_KNOWN_FREE_SPACE = "CUT_KNOWN_FREE_SPACE"
STATE_CUT_OCCLUDED_DOMAIN = "CUT_OCCLUDED_DOMAIN"
STATE_CUT_VISIBLE_DISCONTINUITY = "CUT_VISIBLE_GEOMETRIC_DISCONTINUITY"
STATE_CUT_POSITIONAL_SHEET_SEPARATION = "CUT_POSITIONAL_SHEET_SEPARATION"
STATE_UNRESOLVED_CONFLICT = "UNRESOLVED_OBSERVATION_CONFLICT"
STATE_UNKNOWN_NO_RENDERER_SUPPORTED_RELATION = "UNKNOWN_NO_RENDERER_SUPPORTED_RELATION"

RELATION_STATES = (
    STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_UNRESOLVED_CONFLICT,
    STATE_UNKNOWN_NO_RENDERER_SUPPORTED_RELATION,
)


@dataclass(frozen=True)
class RendererGroundedVisibleAdjacencyConfig:
    """Identical fields, identical values, to Worklog 103's own
    `PositiveVisibleAdjacencyConfig` -- reused unchanged wherever semantically
    applicable, per directive section 6 (controlled replay: change ONLY the
    endpoint primitive evidence, not the corridor/geometry parameters)."""

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    shape_operator_neighbor_count: int = 0
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0
    parallel_sheet_normal_over_tangent_ratio: float = 1.0
    screen_interior_samples: int = 3

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
class RendererGroundedVisibleAdjacencyResult:
    subset_ids: Any
    subset_count: int
    subset_sizes: Any

    graph: CandidateGraph
    gaussian_ids: Any

    relation_state: Any
    positive_visible_edges_mask: Any
    normal_gradient_magnitude: Any
    residual_threshold: float
    config: RendererGroundedVisibleAdjacencyConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def kept_edges(self) -> Any:
        return self.graph.candidate_edges[self.positive_visible_edges_mask]


def compute_renderer_grounded_visible_adjacency_evidence(
    orientation: SurfaceOrientationEvidence,
    observation_evidence: ObservationEvidence | None,
    contributing_masks: list[Any] | None,
    graph: CandidateGraph,
    config: RendererGroundedVisibleAdjacencyConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Core evaluation, structurally parallel to Worklog 103's own
    `compute_positive_visible_adjacency_evidence` (see module docstring for
    exactly what changed). `contributing_masks[k]` must be a `(N,)` bool
    tensor aligned to `observation_evidence.views[k]` -- surfel `i` is
    eligible as an endpoint in view `k` iff `contributing_masks[k][i]` is
    True (Worklog 105's renderer-contribution semantics; NOT recomputed
    here, callers supply it, e.g. via `compute_renderer_contribution_for_view`
    per camera). `observation_evidence=None` or `contributing_masks=None`
    means every spatial edge stays `UNKNOWN_NO_RENDERER_SUPPORTED_RELATION`.
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
        (edge_count,), RELATION_STATES.index(STATE_UNKNOWN_NO_RENDERER_SUPPORTED_RELATION), dtype=torch.int8, device=device
    )
    any_positive = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    any_free = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    any_occluded = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    ever_evaluated = torch.zeros((edge_count,), dtype=torch.bool, device=device)  # same-view co-contribution existed

    have_evidence = observation_evidence is not None and contributing_masks is not None and edge_count > 0 and int(spatial_mask.sum()) > 0
    if have_evidence:
        spatial_index = torch.nonzero(spatial_mask, as_tuple=False).reshape(-1)
        edge_left_all = candidate_edges[spatial_index, 0]
        edge_right_all = candidate_edges[spatial_index, 1]
        n_interior = max(1, int(config.screen_interior_samples))
        fracs = torch.linspace(0.0, 1.0, n_interior + 2)[1:-1]

        for view, contributing in zip(observation_evidence.views, contributing_masks):
            contributing = contributing.to(device=device, dtype=torch.bool)
            # --- CHANGED from Worklog 103: eligibility source is renderer
            # contribution (Worklog 105), not Phase-C center classification.
            co_observed = contributing[edge_left_all] & contributing[edge_right_all]
            if not bool(co_observed.any()):
                continue

            local_indices = torch.nonzero(co_observed, as_tuple=False).reshape(-1)
            ever_evaluated[spatial_index[local_indices]] = True

            # --- UNCHANGED from Worklog 103: the range-based screen-walk
            # corridor test itself (projection math, RANGE bound, interior
            # sampling) is identical -- it reasons about the camera's own
            # rendered depth field along the screen path, independent of
            # which endpoint-eligibility rule selected this pair.
            proj = _project_to_camera(positions, view)
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

            edge_positive = (~edge_free) & (~edge_occluded) & (~edge_any_invalid)

            any_positive[spatial_index[local_indices]] = any_positive[spatial_index[local_indices]] | edge_positive
            any_free[spatial_index[local_indices]] = any_free[spatial_index[local_indices]] | edge_free
            any_occluded[spatial_index[local_indices]] = any_occluded[spatial_index[local_indices]] | edge_occluded

            if progress is not None:
                progress(
                    f"[camera {view.camera_index}] co_contributing={int(co_observed.sum())} "
                    f"positive={int(edge_positive.sum())} free={int(edge_free.sum())} occluded={int(edge_occluded.sum())}"
                )

    # --- UNCHANGED from Worklog 103: multi-view aggregation, no percentage
    # threshold anywhere. ---
    any_contradiction = any_free | any_occluded
    conflict = any_positive & any_contradiction
    only_free = any_free & ~any_occluded & ~conflict
    only_occluded = any_occluded & ~conflict
    positive_clean = any_positive & ~any_contradiction

    relation_state = torch.where(conflict, torch.full_like(relation_state, RELATION_STATES.index(STATE_UNRESOLVED_CONFLICT)), relation_state)
    relation_state = torch.where(only_occluded, torch.full_like(relation_state, RELATION_STATES.index(STATE_CUT_OCCLUDED_DOMAIN)), relation_state)
    relation_state = torch.where(only_free & ~only_occluded, torch.full_like(relation_state, RELATION_STATES.index(STATE_CUT_KNOWN_FREE_SPACE)), relation_state)
    relation_state = torch.where(positive_clean, torch.full_like(relation_state, RELATION_STATES.index(STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION)), relation_state)

    # --- UNCHANGED from Worklog 103/98/102: secondary geometric-
    # discontinuity/positional-sheet gate, applied only to edges that already
    # cleared the corridor test. Corrected signed positional-offset formula
    # (Worklog 102's fix), same as Worklog 103's own copy. ---
    cut_residual = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    cut_positional = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    normal_gradient_magnitude = torch.zeros((count,), dtype=torch.float32, device=device)
    residual_threshold = 0.0

    positive_mask_pre_geometry = relation_state == RELATION_STATES.index(STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION)
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
        relation_state[positive_index[fails_positional]] = RELATION_STATES.index(STATE_CUT_POSITIONAL_SHEET_SEPARATION)

    positive_visible_edges_mask = relation_state == RELATION_STATES.index(STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION)

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


def partition_renderer_grounded_visible_adjacency(
    orientation: SurfaceOrientationEvidence,
    observation_evidence: ObservationEvidence | None,
    contributing_masks: list[Any] | None,
    config: RendererGroundedVisibleAdjacencyConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> RendererGroundedVisibleAdjacencyResult:
    """Partition every surfel into exactly one Visible Surface Component,
    where connectivity is built ONLY from candidate edges positively
    supported by same-view renderer co-contribution + observed visible
    continuity. See module docstring for the full contract."""

    torch = require_torch()
    config = config or RendererGroundedVisibleAdjacencyConfig()
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)

    if count == 0:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_edges_bool = torch.zeros((0,), dtype=torch.bool, device=device)
        empty_edges_state = torch.zeros((0,), dtype=torch.int8, device=device)
        return RendererGroundedVisibleAdjacencyResult(
            subset_ids=empty_long, subset_count=0, subset_sizes=empty_long,
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            relation_state=empty_edges_state, positive_visible_edges_mask=empty_edges_bool,
            normal_gradient_magnitude=torch.zeros((0,), dtype=torch.float32, device=device),
            residual_threshold=0.0, config=config,
        )

    evidence = compute_renderer_grounded_visible_adjacency_evidence(
        orientation, observation_evidence, contributing_masks, graph, config, progress=progress
    )

    kept = graph.candidate_edges[evidence["positive_visible_edges_mask"]]
    roots = _connected_component_roots(count, kept, config.local)
    unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order)
    subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order]

    return RendererGroundedVisibleAdjacencyResult(
        subset_ids=subset_ids, subset_count=int(order.shape[0]), subset_sizes=subset_sizes,
        graph=graph, gaussian_ids=orientation.gaussian_ids,
        relation_state=evidence["relation_state"], positive_visible_edges_mask=evidence["positive_visible_edges_mask"],
        normal_gradient_magnitude=evidence["normal_gradient_magnitude"],
        residual_threshold=evidence["residual_threshold"], config=config,
    )


def count_spatially_disconnected_subsets(result: RendererGroundedVisibleAdjacencyResult) -> int:
    torch = require_torch()
    count = len(result)
    if count == 0:
        return 0
    roots = _connected_component_roots(count, result.kept_edges, result.config.local)
    unique_pairs = torch.unique(result.subset_ids * int(count) + roots)
    subset_of_pair = torch.div(unique_pairs, count, rounding_mode="floor")
    components_per_subset = torch.bincount(subset_of_pair, minlength=max(result.subset_count, 1))
    return int((components_per_subset > 1).sum())


def renderer_grounded_visible_adjacency_accounting(result: RendererGroundedVisibleAdjacencyResult) -> dict[str, Any]:
    torch = require_torch()
    count = len(result)
    sizes = result.subset_sizes

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
