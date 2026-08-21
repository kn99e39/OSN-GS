from __future__ import annotations

"""Worklog 102 -- Maximal Visible Surface Components.

Worklogs 97-101 all shared one philosophy: start from a CONSERVATIVE
partition and require a candidate merge to PROVE bilateral smoothness before
two regions are allowed to become one subset. Worklog 101 showed this
architecture's own granularity -- not support starvation, not a tunable
threshold -- is why it recovers so little of the scene (112,768 final
regions). This module replaces that philosophy entirely:

    full observed-visible surfel evidence
        -> local candidate connectivity (Worklog 96-101's own candidate graph,
           REUSED unchanged)
        -> CUT connectivity only when EXPLICIT evidence requires it:
             known free space in the way,
             a positive occluded domain in the way,
             a supported visible-surface discontinuity
        -> maximal Visible Surface Components

Connectivity is the DEFAULT; a cut must be justified by evidence, not the
other way around. "Maximal" means maximal WITHIN the observed-visible
domain -- this module never bridges through occluded or free space, and
never claims two visible surfaces are the same component merely because they
might belong to the same underlying physical surface.

    Physical Surface Continuity != Visible Surface Connectivity.

Canonical Phase-C reuse
-----------------------
This module reuses `osn_gs.surface.torch_observation_evidence`'s STATUS
constants, dataclasses (`ObservationEvidence`, `CameraViewEvidence`), and
per-view classification RULE (unchanged three-way depth-epsilon comparison)
verbatim -- it does not redefine what `known_free_space` /
`on_observed_surface` / `behind_first_observed_surface` / `unobserved` /
`outside_valid_view` mean. `torch_observation_evidence.py` itself is NOT
modified. What IS new is a VECTORIZED per-surfel/per-edge evaluation (the
canonical `classify_world_samples` is a per-sample Python loop, written for
the low-hundreds-of-candidates scale Phase E actually runs at -- not the
millions-of-candidate-edges scale this module needs); `test_maximal_visible_connectivity.py::
test_vectorized_per_view_classification_matches_canonical_classify_world_samples`
proves the vectorized path produces IDENTICAL per-view statuses to
`classify_world_samples` on the same inputs.

Why not a straight 3D chord, and not a linear depth guess either
------------------------------------------------------------------
Two neighbouring surfels on a CURVED visible surface can have a Euclidean
chord whose interior sits measurably behind the true (curved) surface, even
over a very short local distance -- sampling that chord in 3D and asking "is
any point behind the observed surface" would misclassify ordinary curvature
as occlusion. A LINEAR INTERPOLATION of the two endpoints' own camera-space
depths has exactly the same failure mode (a straight 3D chord's depth is
itself approximately linear in the interpolation fraction, so comparing
against it is not meaningfully different -- discovered while writing this
module's own docstring, before any fixture was run, and fixed before use).

This module instead walks the camera's own 2D SCREEN-SPACE path between the
two pixel projections and reads the camera's OWN rendered depth there, but
classifies each sample against a RANGE, not a point estimate:

    [min(depth_i, depth_j) - edge_length - depth_epsilon,
     max(depth_i, depth_j) + edge_length + depth_epsilon]

where `edge_length` is the edge's own 3D Euclidean length (already computed
for the candidate graph, not a new quantity). This bound is geometrically
NECESSARY, not tuned: even if the true surface between the two endpoints ran
maximally along the camera's view ray, its depth could not exceed this
range. A sample nearer than the range means a genuine foreground occluder
resolves there; a sample farther than the range means the camera sees
THROUGH to something beyond (no surface bridges the gap at any plausible
depth); a sample inside the range is consistent with ANY surface shape
(however curved) connecting the two endpoints, and is never penalized.

Role separation (directive section 2)
--------------------------------------
`on_observed_surface` evidence supports visible connectivity.
`occluded_candidate` (i.e. `behind_first_observed_surface`) domains and
`known_free_space` domains are HARD CUTS for visible connectivity -- this
module never materializes a visible surface through either. `unobserved` and
`outside_valid_view` carry no positive evidence either way (directive
section 4) and are never treated as proof of a bridge.
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
from osn_gs.surface.torch_observation_evidence import (
    ObservationEvidence,
    _project_points,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# --- explicit, disjoint-where-possible cut-reason provenance (directive section 9) ---
CUT_KNOWN_FREE_SPACE = "CUT_KNOWN_FREE_SPACE"
CUT_OCCLUDED_DOMAIN = "CUT_OCCLUDED_DOMAIN"
CUT_VISIBLE_GEOMETRIC_DISCONTINUITY = "CUT_VISIBLE_GEOMETRIC_DISCONTINUITY"
CUT_POSITIONAL_SHEET_SEPARATION = "CUT_POSITIONAL_SHEET_SEPARATION"
UNRESOLVED_OBSERVATION_CONFLICT = "UNRESOLVED_OBSERVATION_CONFLICT"

CUT_REASONS = (
    CUT_KNOWN_FREE_SPACE,
    CUT_OCCLUDED_DOMAIN,
    CUT_VISIBLE_GEOMETRIC_DISCONTINUITY,
    CUT_POSITIONAL_SHEET_SEPARATION,
    UNRESOLVED_OBSERVATION_CONFLICT,
)

# A 2x2 linear model needs at least 2 same-region... here, simply 2 (any)
# neighbours to be non-degenerate -- reused convention from Worklog
# 98/100 (structural fact, not swept).
_MIN_SHAPE_OPERATOR_SUPPORT = 2


@dataclass(frozen=True)
class MaximalVisibleConnectivityConfig:
    """Every geometric-candidate-graph field is REUSED VERBATIM from Worklog
    96-101 (`CoverageFirstPartitionConfig`). The observation-evidence fields
    reuse `ObservationEvidence.depth_epsilon` (never redefined here) plus
    exactly the interior-sample count Phase E's own
    `validate_candidate_observation_evidence` already uses by default (3) --
    not an independently chosen new constant.

    `residual_mad_multiplier` / `parallel_sheet_normal_over_tangent_ratio` are
    REUSED VERBATIM from Worklog 98/100 for the geometric-discontinuity gate
    (directive section 8: differential geometry is evidence of a discontinuity
    here, never a smoothness requirement, so there is no majority-fraction
    parameter in this module at all -- `interface_smooth_majority_fraction`
    has no equivalent here by design).
    """

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    shape_operator_neighbor_count: int = 0  # 0 => local.neighbor_count
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0  # reused verbatim from Worklog 98/100
    parallel_sheet_normal_over_tangent_ratio: float = 1.0  # reused verbatim from Worklog 98/100
    screen_interior_samples: int = 3  # reused verbatim from torch_candidate_evidence's own default

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
            "screen_interior_samples_derivation": "torch_candidate_evidence.validate_candidate_observation_evidence's own default",
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class MaximalVisibleConnectivityResult:
    subset_ids: Any  # (N,) int64 -- exactly one Visible Surface Component owner per surfel
    subset_count: int
    subset_sizes: Any  # (subset_count,) int64, descending

    graph: CandidateGraph
    gaussian_ids: Any

    cut_known_free_space: Any  # (E_c,) bool over graph.candidate_edges
    cut_occluded_domain: Any  # (E_c,) bool
    cut_visible_geometric_discontinuity: Any  # (E_c,) bool
    cut_positional_sheet_separation: Any  # (E_c,) bool
    cut_unresolved_observation_conflict: Any  # (E_c,) bool
    cut_mask: Any  # (E_c,) bool -- OR of all five reasons above, spatial edges only elsewhere False
    observation_evaluated: Any  # (E_c,) bool -- at least one camera co-observed both endpoints
    normal_gradient_magnitude: Any  # (N,) float -- diagnostic only (Worklog 98 lineage)
    residual_threshold: float
    config: MaximalVisibleConnectivityConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def kept_edges(self) -> Any:
        return self.graph.candidate_edges[self.graph.spatial_edge_mask & ~self.cut_mask]

    @property
    def boundary_edges(self) -> Any:
        return self.graph.candidate_edges[self.graph.spatial_edge_mask & self.cut_mask]


def _per_view_status_codes(
    view_depth: Any, obs_depth: Any, valid_at_pixel: Any, in_bounds: Any, near: float, far: float, depth_epsilon: float
) -> Any:
    """Vectorized equivalent of `classify_world_samples`'s per-view rule for a
    batch of samples already projected into ONE camera. Returns an int8 code:
    0=outside_valid_view, 1=unobserved, 2=known_free_space,
    3=on_observed_surface, 4=behind_first_observed_surface -- the SAME five
    states and SAME priority order (bounds check, then validity/near-far,
    then the three-way depth-epsilon compare) as the canonical per-sample
    function, just computed for every sample in one batch instead of a
    Python loop.
    """

    torch = require_torch()
    code = torch.zeros_like(view_depth, dtype=torch.int8)
    is_valid = valid_at_pixel & (view_depth >= near) & (view_depth <= far)
    unobserved = in_bounds & ~is_valid
    diff = view_depth - obs_depth
    free = in_bounds & is_valid & (diff < -depth_epsilon)
    behind = in_bounds & is_valid & (diff > depth_epsilon)
    on_surface = in_bounds & is_valid & ~free & ~behind
    code = torch.where(~in_bounds, torch.zeros_like(code), code)
    code = torch.where(unobserved, torch.ones_like(code), code)
    code = torch.where(free, torch.full_like(code, 2), code)
    code = torch.where(on_surface, torch.full_like(code, 3), code)
    code = torch.where(behind, torch.full_like(code, 4), code)
    return code


def _project_to_camera(world_points: Any, view) -> dict[str, Any]:
    """Project `world_points` into one `CameraViewEvidence` view, returning
    everything `_per_view_status_codes` and the screen-space walk need.
    Reuses `torch_observation_evidence._project_points` unchanged."""

    torch = require_torch()
    points = world_points.to(device=view.view_depth.device, dtype=view.view_depth.dtype)
    view_depth, ndc_x, ndc_y, behind_camera = _project_points(points, view.world_view_transform, view.full_proj_transform)
    pixel_col = (ndc_x * 0.5 + 0.5) * view.image_width
    pixel_row = (ndc_y * 0.5 + 0.5) * view.image_height
    row_idx = pixel_row.round().long()
    col_idx = pixel_col.round().long()
    in_bounds = (
        (row_idx >= 0) & (row_idx < view.image_height) & (col_idx >= 0) & (col_idx < view.image_width) & (~behind_camera)
    )
    clamped_row = row_idx.clamp(0, view.image_height - 1)
    clamped_col = col_idx.clamp(0, view.image_width - 1)
    observed_depth = view.view_depth[clamped_row, clamped_col]
    valid_at_pixel = view.valid_depth_mask[clamped_row, clamped_col]
    return {
        "view_depth": view_depth, "pixel_row": pixel_row, "pixel_col": pixel_col,
        "in_bounds": in_bounds, "observed_depth": observed_depth, "valid_at_pixel": valid_at_pixel,
    }


def compute_visible_connectivity_evidence(
    orientation: SurfaceOrientationEvidence,
    observation_evidence: ObservationEvidence | None,
    graph: CandidateGraph,
    config: MaximalVisibleConnectivityConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The core, vectorized, per-camera-streamed evidence computation.
    Returns per-CANDIDATE-EDGE boolean tensors (full `graph.candidate_edges`
    length; entries for non-spatial edges are always False) for each cut
    reason, plus `observation_evaluated`.

    `observation_evidence=None` skips the observation-based gates entirely
    (both cut_known_free_space and cut_occluded_domain stay all-False,
    observation_evaluated stays all-False) -- geometric-discontinuity cuts
    still apply. This lets the geometric-only path be tested/run
    independently of any camera data.
    """

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    count = int(positions.shape[0])
    device = positions.device
    candidate_edges = graph.candidate_edges
    spatial_mask = graph.spatial_edge_mask
    edge_count = int(candidate_edges.shape[0])

    cut_free = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    cut_occluded = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    cut_conflict = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    observation_evaluated = torch.zeros((edge_count,), dtype=torch.bool, device=device)

    if observation_evidence is not None and edge_count > 0 and int(spatial_mask.sum()) > 0:
        spatial_index = torch.nonzero(spatial_mask, as_tuple=False).reshape(-1)
        edge_left_all = candidate_edges[spatial_index, 0]
        edge_right_all = candidate_edges[spatial_index, 1]
        n_interior = max(1, int(config.screen_interior_samples))
        fracs = torch.linspace(0.0, 1.0, n_interior + 2)[1:-1]  # strictly interior, matches Phase E convention

        any_free = torch.zeros((edge_left_all.shape[0],), dtype=torch.bool, device=device)
        any_occluded = torch.zeros((edge_left_all.shape[0],), dtype=torch.bool, device=device)
        any_clean = torch.zeros((edge_left_all.shape[0],), dtype=torch.bool, device=device)

        for view in observation_evidence.views:
            proj = _project_to_camera(positions, view)
            status = _per_view_status_codes(
                proj["view_depth"], proj["observed_depth"], proj["valid_at_pixel"], proj["in_bounds"],
                observation_evidence.near, observation_evidence.far, observation_evidence.depth_epsilon,
            )
            on_surface_this_view = status == 3

            co_observed = on_surface_this_view[edge_left_all] & on_surface_this_view[edge_right_all]
            if not bool(co_observed.any()):
                if progress is not None:
                    progress(f"[camera {view.camera_index}] no co-observed spatial edges")
                continue
            observation_evaluated[spatial_index[co_observed]] = True

            left_idx = edge_left_all[co_observed]
            right_idx = edge_right_all[co_observed]
            row_left, col_left = proj["pixel_row"][left_idx], proj["pixel_col"][left_idx]
            row_right, col_right = proj["pixel_row"][right_idx], proj["pixel_col"][right_idx]
            depth_left = proj["view_depth"][left_idx]
            depth_right = proj["view_depth"][right_idx]
            # The geometrically NECESSARY depth slack for two points at 3D
            # distance `edge_length` apart: even if the true (however curved)
            # surface between them ran maximally along this camera's view
            # ray, its depth could not exceed the endpoints' own depth range
            # by more than `edge_length` -- this is a hard bound, not a
            # tuned tolerance. Using it (never a naive linear-interpolated
            # "expected depth") is exactly how this module avoids the
            # straight-chord false-occlusion failure directive section 5
            # warns about: a strongly curved LOCAL surface can freely deviate
            # from a linear depth guess within this bound and is never
            # penalized for it.
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
                # diff-free/occluded classification against the RANGE, not a
                # point guess: a sample nearer than the whole plausible range
                # means something else (a real foreground occluder) resolves
                # there; a sample farther than the whole range means the
                # camera sees THROUGH to something beyond -- i.e. nothing
                # bridges the gap at the expected depth, a free-space read.
                edge_occluded = edge_occluded | (sample_valid & (sample_observed_depth < depth_low))
                edge_free = edge_free | (sample_valid & (sample_observed_depth > depth_high))
                edge_any_invalid = edge_any_invalid | ~sample_valid

            edge_clean = ~edge_free & ~edge_occluded & ~edge_any_invalid
            any_free_local = torch.zeros_like(any_free)
            any_free_local[co_observed.nonzero(as_tuple=False).reshape(-1)] = edge_free
            any_occluded_local = torch.zeros_like(any_occluded)
            any_occluded_local[co_observed.nonzero(as_tuple=False).reshape(-1)] = edge_occluded
            any_clean_local = torch.zeros_like(any_clean)
            any_clean_local[co_observed.nonzero(as_tuple=False).reshape(-1)] = edge_clean
            any_free = any_free | any_free_local
            any_occluded = any_occluded | any_occluded_local
            any_clean = any_clean | any_clean_local
            if progress is not None:
                progress(
                    f"[camera {view.camera_index}] co_observed={int(co_observed.sum())} "
                    f"free={int(edge_free.sum())} occluded={int(edge_occluded.sum())}"
                )

        has_gap_evidence = any_free | any_occluded
        # genuine multi-camera disagreement: at least one camera shows a gap
        # AND at least one (co-observing) camera shows fully clean continuity
        # for the SAME edge -> fail-safe cut, distinct from a consistent verdict.
        conflict = has_gap_evidence & any_clean
        cut_free[spatial_index] = any_free & ~conflict
        cut_occluded[spatial_index] = any_occluded & ~conflict
        cut_conflict[spatial_index] = conflict

    # --- geometric-discontinuity gate: WL98's own per-edge cut logic, reused
    # verbatim (imported, not reimplemented) -- a CUT test, exactly the
    # semantics this module also needs (unlike WL100's bilateral-MERGE proof
    # logic, which this module deliberately does not use as topology law,
    # directive section 8). ---
    cut_residual = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    cut_positional = torch.zeros((edge_count,), dtype=torch.bool, device=device)
    normal_gradient_magnitude = torch.zeros((count,), dtype=torch.float32, device=device)
    residual_threshold = 0.0

    if count > 0 and int(spatial_mask.sum()) > 0:
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

        spatial_edges = candidate_edges[spatial_mask]
        spatial_index2 = torch.nonzero(spatial_mask, as_tuple=False).reshape(-1)
        left, right = spatial_edges[:, 0], spatial_edges[:, 1]
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
        # Signed projection, not abs, before subtracting -- using the abs
        # magnitude here (as Worklog 98/99/100's own copy of this formula
        # does) silently DOUBLES the normal-direction component instead of
        # cancelling it whenever the signed dot product is negative (masked
        # in those modules' own fixtures by their normals/offsets always
        # happening to align sign-positive; exposed here by a fixture with
        # an oppositely-signed normal convention). Fixed in THIS module's own
        # copy only -- Worklog 100 is preserved unmodified as directed.
        signed_normal_offset = (delta_x * average_normal).sum(dim=-1)
        normal_offset = signed_normal_offset.abs()
        tangential_offset = (delta_x - signed_normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
        normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)

        median_residual = torch.median(edge_residual)
        mad = torch.median((edge_residual - median_residual).abs())
        residual_threshold = float(median_residual + config.residual_mad_multiplier * 1.4826 * mad)

        cut_residual[spatial_index2] = edge_residual > residual_threshold
        cut_positional[spatial_index2] = normal_offset_ratio > config.parallel_sheet_normal_over_tangent_ratio

    cut_mask = cut_free | cut_occluded | cut_conflict | cut_residual | cut_positional
    return {
        "cut_known_free_space": cut_free,
        "cut_occluded_domain": cut_occluded,
        "cut_unresolved_observation_conflict": cut_conflict,
        "cut_visible_geometric_discontinuity": cut_residual,
        "cut_positional_sheet_separation": cut_positional,
        "cut_mask": cut_mask,
        "observation_evaluated": observation_evaluated,
        "normal_gradient_magnitude": normal_gradient_magnitude,
        "residual_threshold": residual_threshold,
    }


def partition_maximal_visible_components(
    orientation: SurfaceOrientationEvidence,
    observation_evidence: ObservationEvidence | None,
    config: MaximalVisibleConnectivityConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> MaximalVisibleConnectivityResult:
    """Partition EVERY surfel into exactly one Visible Surface Component:
    start from the full candidate graph (connectivity is the default), CUT an
    edge only when evidence requires it, then take connected components of
    what survives. See the module docstring for the full contract.
    """

    torch = require_torch()
    config = config or MaximalVisibleConnectivityConfig()
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)

    if count == 0:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_edges = torch.zeros((0,), dtype=torch.bool, device=device)
        return MaximalVisibleConnectivityResult(
            subset_ids=empty_long, subset_count=0, subset_sizes=empty_long,
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            cut_known_free_space=empty_edges, cut_occluded_domain=empty_edges,
            cut_visible_geometric_discontinuity=empty_edges, cut_positional_sheet_separation=empty_edges,
            cut_unresolved_observation_conflict=empty_edges, cut_mask=empty_edges, observation_evaluated=empty_edges,
            normal_gradient_magnitude=torch.zeros((0,), dtype=torch.float32, device=device),
            residual_threshold=0.0, config=config,
        )

    evidence = compute_visible_connectivity_evidence(orientation, observation_evidence, graph, config, progress=progress)

    kept = graph.candidate_edges[graph.spatial_edge_mask & ~evidence["cut_mask"]]
    roots = _connected_component_roots(count, kept, config.local)
    unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order)
    subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order]

    return MaximalVisibleConnectivityResult(
        subset_ids=subset_ids, subset_count=int(order.shape[0]), subset_sizes=subset_sizes,
        graph=graph, gaussian_ids=orientation.gaussian_ids,
        cut_known_free_space=evidence["cut_known_free_space"],
        cut_occluded_domain=evidence["cut_occluded_domain"],
        cut_visible_geometric_discontinuity=evidence["cut_visible_geometric_discontinuity"],
        cut_positional_sheet_separation=evidence["cut_positional_sheet_separation"],
        cut_unresolved_observation_conflict=evidence["cut_unresolved_observation_conflict"],
        cut_mask=evidence["cut_mask"], observation_evaluated=evidence["observation_evaluated"],
        normal_gradient_magnitude=evidence["normal_gradient_magnitude"],
        residual_threshold=evidence["residual_threshold"], config=config,
    )


def count_spatially_disconnected_subsets(result: MaximalVisibleConnectivityResult) -> int:
    torch = require_torch()
    count = len(result)
    if count == 0:
        return 0
    roots = _connected_component_roots(count, result.kept_edges, result.config.local)
    unique_pairs = torch.unique(result.subset_ids * int(count) + roots)
    subset_of_pair = torch.div(unique_pairs, count, rounding_mode="floor")
    components_per_subset = torch.bincount(subset_of_pair, minlength=max(result.subset_count, 1))
    return int((components_per_subset > 1).sum())


def maximal_visible_connectivity_accounting(result: MaximalVisibleConnectivityResult) -> dict[str, Any]:
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
    cut_edge_count = int(result.cut_mask.sum())

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
        "boundary_cut_edge_count": cut_edge_count,
        "boundary_cut_reason_counts": {
            CUT_KNOWN_FREE_SPACE: int(result.cut_known_free_space.sum()),
            CUT_OCCLUDED_DOMAIN: int(result.cut_occluded_domain.sum()),
            CUT_VISIBLE_GEOMETRIC_DISCONTINUITY: int(result.cut_visible_geometric_discontinuity.sum()),
            CUT_POSITIONAL_SHEET_SEPARATION: int(result.cut_positional_sheet_separation.sum()),
            UNRESOLVED_OBSERVATION_CONFLICT: int(result.cut_unresolved_observation_conflict.sum()),
        },
        "observation_evaluated_edge_count": int(result.observation_evaluated.sum()),
        "kept_edge_count": spatial_edge_count - cut_edge_count,
        "residual_threshold": result.residual_threshold,
        "partition_parameters": result.config.payload(),
    }
