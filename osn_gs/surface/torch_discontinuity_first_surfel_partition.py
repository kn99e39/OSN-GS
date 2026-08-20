from __future__ import annotations

"""Worklog 98 -- discontinuity-first Surfel Subset partition.

Worklog 97 fixed single-linkage chaining by gating region growth on a
GLOBAL orientation-concentration floor. Real-scene review then showed the
opposite failure mode: a smoothly curved surface (the rounded side of a
table) fragments into several subsets simply because its normal rotates --
concentration decays with accumulated normal ROTATION even when that
rotation is exactly what a valid curved NURBS patch is expected to have.
Worklog 97 is kept as a diagnostic baseline; it is not the final partition
model.

This module replaces the union rule again, this time with a genuinely
different signal: not "how much do the normals in this region vary" but
"is the observed normal change between two neighbouring surfels consistent
with ONE locally smooth surface, or does it fail that model". Concretely:

    full local candidate graph (same kNN + local-spacing gate as Worklog 96/97)
        -> per-surfel local SHAPE OPERATOR estimate (how the normal is
           expected to vary across the tangent plane, from the surfel's own
           neighbourhood -- captures smooth curvature, not just orientation)
        -> per-edge SMOOTH-SURFACE RESIDUAL (how far the observed normal
           change deviates from that local prediction)
        -> per-edge POSITIONAL/PARALLEL-SHEET separation (normal-direction
           offset a smooth tangent-plane displacement cannot explain)
        -> CUT edges whose residual or separation fails the smooth model
        -> connected components of the surviving graph -> final subsets

A smooth, strongly curved surface has a LARGE normal gradient but a SMALL
residual (the curvature is exactly what its own local shape operator
predicts) -- it is kept as one subset. A sharp crease has a large residual
(no single local linear model explains both sides) -- it is cut. Two nearby
parallel sheets can share almost identical normals yet still separate,
because their positional offset along the shared normal direction is not
explained by tangent-plane displacement.

No closed boundary loop is ever constructed or required. A "boundary" here
is graph-cut evidence -- a set of rejected edges -- not an ordered polyline;
open, incomplete, or disconnected cut evidence is expected and normal.

Coverage is unconditional and structurally simple: cutting an edge can never
remove a node, so plain connected components of the surviving graph already
give every surfel exactly one final subset (an isolated surfel with no
surviving edge is trivially its own singleton component -- no separate
propagation mechanism is needed here, unlike Worklog 97's region/ownership
split).
"""

from dataclasses import dataclass
from typing import Any, Callable

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CandidateGraph,
    CoverageFirstPartitionConfig,
    SurfaceOrientationEvidence,
    VERY_SMALL_SUBSET_SIZE,
    _connected_component_roots,
    _knn,
    _auto_chunk_size,
    build_candidate_graph,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# --- ownership kinds: every surfel carries exactly one, mirroring
# Worklog 96's own reporting vocabulary so the two are directly comparable. ---
OWNERSHIP_SMOOTH_CONTINUATION = "smooth_continuation_component"
OWNERSHIP_FALLBACK_ALL_EDGES_CUT = "fallback_all_local_edges_cut"
OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR = "fallback_no_spatial_neighbor"

OWNERSHIP_KINDS: tuple[str, ...] = (
    OWNERSHIP_SMOOTH_CONTINUATION,
    OWNERSHIP_FALLBACK_ALL_EDGES_CUT,
    OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR,
)

# --- cut reasons: an edge can be cut for either or both. Reported
# separately so the review export can distinguish "not one smooth surface"
# from "not the same sheet at all". ---
CUT_REASON_RESIDUAL = "smooth_surface_residual"
CUT_REASON_PARALLEL_SHEET = "positional_normal_offset"


@dataclass(frozen=True)
class DiscontinuityFirstConfig:
    """Every heuristic this module needs, centralized.

    The candidate graph itself (`local`) is REUSED VERBATIM from Worklog
    96/97 -- same kNN neighbour count, same local-spacing spatial gate. Its
    `normal_compatibility_min_alignment` field is carried along for
    provenance/comparability ONLY: this module never gates candidate
    connectivity or the final cut decision on raw normal alignment (that is
    precisely the Worklog 97 behaviour this batch replaces).

    Exactly TWO new heuristic constants are introduced, both dimensionless
    ratios chosen for what they mean geometrically, not swept against this
    scene's visualization:

    * `residual_mad_multiplier` -- a standard robust-statistics convention
      (median + k*MAD outlier fence); the fence itself is computed from THIS
      replay's own residual distribution, only the multiplier k is fixed in
      advance.
    * `parallel_sheet_normal_over_tangent_ratio` -- the natural "parity"
      value 1.0 (a displacement that leaves the tangent plane by more than
      it moves along it), not an independently swept angle.

    An earlier version of this criterion tried to reuse
    `local.spatial_connect_spacing_multiplier` directly -- discovered by
    a fixture test to be DEGENERATE: since the candidate graph already
    bounds total displacement at `multiplier * spacing`, and the
    normal-direction component can never exceed the total displacement, that
    ratio could structurally never exceed the SAME multiplier used to admit
    the edge as a candidate in the first place. The tangential-vs-normal
    comparison below has no such degeneracy.
    """

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()

    # Neighbourhood size used to FIT each surfel's own local shape operator.
    # Reuses `local.neighbor_count` by default so "local neighbourhood" keeps
    # one meaning across this module and the candidate graph -- overridable
    # only for isolated testing of the fit itself, never for the primary replay.
    shape_operator_neighbor_count: int = 0  # 0 => local.neighbor_count

    # median + k * MAD(residual) outlier fence, computed over this replay's
    # OWN edge population. k=3 is the standard "3 MAD" robust-outlier
    # convention (the MAD analogue of a 3-sigma rule), not scene-tuned.
    residual_mad_multiplier: float = 3.0

    # Cut a candidate edge as a parallel-sheet/positional discontinuity when
    # its displacement's component ALONG the shared normal direction exceeds
    # this multiple of its component WITHIN the tangent plane -- 1.0 is the
    # natural parity value ("leaves the tangent plane by more than it moves
    # along it"), not a value tuned against this scene. Self-normalizing: no
    # spacing/scale reference is needed because both quantities are the same
    # displacement vector's own orthogonal components.
    parallel_sheet_normal_over_tangent_ratio: float = 1.0

    # Regularization added to the 2x2 normal-equations system before solving
    # for each surfel's shape operator, purely to keep a near-degenerate
    # (near-collinear or too-small) neighbourhood's fit numerically finite --
    # never large enough to bias a well-conditioned fit.
    shape_operator_ridge: float = 1e-8

    def resolved_shape_operator_neighbor_count(self) -> int:
        return int(self.shape_operator_neighbor_count) or int(self.local.neighbor_count)

    def payload(self) -> dict[str, Any]:
        return {
            "local": self.local.payload(),
            "shape_operator_neighbor_count": self.resolved_shape_operator_neighbor_count(),
            "residual_mad_multiplier": self.residual_mad_multiplier,
            "shape_operator_ridge": self.shape_operator_ridge,
            "parallel_sheet_normal_over_tangent_ratio": self.parallel_sheet_normal_over_tangent_ratio,
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class DiscontinuityFirstPartition:
    """Result of one discontinuity-first partition run."""

    subset_ids: Any  # (N,) int64 -- exactly one owner per surfel
    subset_count: int
    subset_sizes: Any  # (subset_count,) int64, descending by size
    ownership_kind: Any  # (N,) int8 index into OWNERSHIP_KINDS
    normal_gradient_magnitude: Any  # (N,) float -- diagnostic only, per surfel (see docstring section 2)
    shape_operator: Any  # (N, 2, 2) float -- fitted S_i, tangent-plane basis (tangent_axis_u, tangent_axis_v)
    edge_residual: Any  # (E_c,) float over graph.candidate_edges -- smooth-surface residual, spatial edges only elsewhere 0
    edge_normal_offset_ratio: Any  # (E_c,) float -- positional/parallel-sheet ratio, spatial edges only elsewhere 0
    cut_mask: Any  # (E_c,) bool over graph.candidate_edges -- spatially adjacent edge CUT as a discontinuity
    cut_reason_residual: Any  # (E_c,) bool -- this edge was cut because of the residual test
    cut_reason_parallel_sheet: Any  # (E_c,) bool -- this edge was cut because of the positional test
    residual_threshold: float
    graph: CandidateGraph
    gaussian_ids: Any
    config: DiscontinuityFirstConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def kept_edges(self) -> Any:
        """Spatially adjacent edges that survive as smooth continuation --
        exactly the graph connected components are computed over."""

        return self.graph.candidate_edges[self.graph.spatial_edge_mask & ~self.cut_mask]

    @property
    def boundary_edges(self) -> Any:
        """Spatially adjacent edges CUT as discontinuity evidence -- the
        review export's boundary view. Never required to form a closed loop."""

        return self.graph.candidate_edges[self.graph.spatial_edge_mask & self.cut_mask]


def _tangent_plane_components(vectors: Any, tangent_u: Any, tangent_v: Any) -> Any:
    """Project (..., 3) vectors onto a per-row (tangent_u, tangent_v) basis,
    returning (..., 2) local coordinates. `tangent_u`/`tangent_v` broadcast
    against `vectors`' leading batch dimensions."""

    torch = require_torch()
    return torch.stack([(vectors * tangent_u).sum(dim=-1), (vectors * tangent_v).sum(dim=-1)], dim=-1)


def _fit_shape_operators(
    positions: Any, normals: Any, tangent_u: Any, tangent_v: Any,
    neighbor_index: Any, ridge: float,
) -> Any:
    """Batched local shape-operator fit, one 2x2 matrix S_i per surfel.

    Uses the DIFFERENTIAL relation ``Delta n ~= -S Delta x_T`` (architecture
    directive section 3): for surfel i's own k nearest neighbours, regress
    the neighbour's sign-aligned tangential normal change against the
    neighbour's tangential displacement, in i's own (tangent_u, tangent_v)
    basis. A vectorized weighted least squares over ALL N surfels
    simultaneously (weight = 1, i.e. ordinary least squares -- see module
    docstring: no trust weighting is introduced here).

    Normal sign ambiguity is resolved per-edge by aligning the neighbour's
    normal to the QUERY surfel's own normal sign before differencing (never a
    global outward-orientation choice) -- exactly the local, non-global
    alignment the architecture directive requires.
    """

    torch = require_torch()
    count = int(positions.shape[0])
    k = int(neighbor_index.shape[1])

    neighbor_positions = positions[neighbor_index]  # (N, k, 3)
    neighbor_normals = normals[neighbor_index]  # (N, k, 3)

    delta_x = neighbor_positions - positions.unsqueeze(1)  # (N, k, 3)
    tangent_u_b = tangent_u.unsqueeze(1)
    tangent_v_b = tangent_v.unsqueeze(1)
    delta_x_t = _tangent_plane_components(delta_x, tangent_u_b, tangent_v_b)  # (N, k, 2)

    query_normal = normals.unsqueeze(1)
    sign = torch.where((neighbor_normals * query_normal).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
    aligned_neighbor_normal = neighbor_normals * sign
    delta_n = aligned_neighbor_normal - query_normal  # (N, k, 3)
    delta_n_t = _tangent_plane_components(delta_n, tangent_u_b, tangent_v_b)  # (N, k, 2)

    # Solve, per surfel, for (-S)^T in  delta_n_t ~= delta_x_t @ (-S)^T,
    # i.e. ordinary least squares Y = X @ B with X=(k,2), Y=(k,2), via the
    # regularized normal equations (X^T X + ridge*I) B = X^T Y -- a 2x2
    # linear solve per surfel, batched over all N at once.
    xtx = torch.einsum("nki,nkj->nij", delta_x_t, delta_x_t)  # (N, 2, 2)
    xty = torch.einsum("nki,nkj->nij", delta_x_t, delta_n_t)  # (N, 2, 2)
    identity = torch.eye(2, dtype=xtx.dtype, device=xtx.device).expand(count, 2, 2)
    neg_s_transpose = torch.linalg.solve(xtx + ridge * identity, xty)  # (N, 2, 2)
    shape_operator = -neg_s_transpose.transpose(-1, -2)  # S_i = -B^T
    return shape_operator


def _predicted_delta_n_t(shape_operator: Any, delta_x_t: Any) -> Any:
    """``-S_i @ Delta x_T`` for a batch of (N, 2) tangent displacements
    against a batch of (N, 2, 2) shape operators."""

    torch = require_torch()
    return torch.einsum("nij,nj->ni", -shape_operator, delta_x_t)


def partition_surfels_discontinuity_first(
    orientation: SurfaceOrientationEvidence,
    config: DiscontinuityFirstConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> DiscontinuityFirstPartition:
    """Partition EVERY surfel by cutting graph edges that fail a local
    smooth-surface model, then taking connected components of what remains.
    See the module docstring for the full contract.
    """

    torch = require_torch()
    config = config or DiscontinuityFirstConfig()
    positions = orientation.positions
    normals = orientation.surface_normal
    tangent_u = orientation.tangent_axis_u
    tangent_v = orientation.tangent_axis_v
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)

    if count == 0:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_edges = torch.zeros((0,), dtype=torch.bool, device=device)
        return DiscontinuityFirstPartition(
            subset_ids=empty_long, subset_count=0, subset_sizes=empty_long,
            ownership_kind=torch.zeros((0,), dtype=torch.int8, device=device),
            normal_gradient_magnitude=torch.zeros((0,), dtype=torch.float32, device=device),
            shape_operator=torch.zeros((0, 2, 2), dtype=torch.float32, device=device),
            edge_residual=torch.zeros((0,), dtype=torch.float32, device=device),
            edge_normal_offset_ratio=torch.zeros((0,), dtype=torch.float32, device=device),
            cut_mask=empty_edges, cut_reason_residual=empty_edges, cut_reason_parallel_sheet=empty_edges,
            residual_threshold=0.0, graph=graph, gaussian_ids=orientation.gaussian_ids, config=config,
        )

    spatial_edges = graph.candidate_edges[graph.spatial_edge_mask]
    spatial_index = torch.nonzero(graph.spatial_edge_mask, as_tuple=False).reshape(-1)
    left, right = spatial_edges[:, 0], spatial_edges[:, 1]

    k_shape = min(config.resolved_shape_operator_neighbor_count(), max(count - 1, 1))
    if progress is not None:
        progress(f"fitting local shape operators: k={k_shape}")
    chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
    neighbor_index, _ = _knn(positions, k_shape, chunk_size, progress)
    shape_operator = _fit_shape_operators(
        positions, normals, tangent_u, tangent_v, neighbor_index, float(config.shape_operator_ridge)
    )
    normal_gradient_magnitude = torch.linalg.matrix_norm(shape_operator)

    if int(spatial_edges.shape[0]) == 0:
        edge_residual_full = torch.zeros((int(graph.candidate_edges.shape[0]),), dtype=torch.float32, device=device)
        edge_offset_full = torch.zeros_like(edge_residual_full)
        cut_mask = torch.zeros((int(graph.candidate_edges.shape[0]),), dtype=torch.bool, device=device)
        residual_threshold = 0.0
        cut_residual = cut_mask.clone()
        cut_parallel = cut_mask.clone()
    else:
        delta_x = positions[right] - positions[left]  # (E, 3)

        # --- smooth-surface residual, symmetric MIN of both directions: an
        # edge is cut only when BOTH i's own local model AND j's own local
        # model independently fail to explain the transition. Taking the max
        # (either side alone can flag it) was tried first and found, on the
        # sharp-crease fixture, to spuriously flag same-side edges purely
        # adjacent to a real discontinuity (a node whose kNN neighbourhood
        # straddles the crease gets a contaminated, poorly-conditioned fit,
        # which then makes ALL of its edges look suspect, not just the one
        # crossing edge). Requiring both directions to agree keeps a true
        # discontinuity (large residual on both sides, since Delta n really is
        # large there) while substantially reducing that false-positive band. ---
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

        # --- positional / parallel-sheet separation ---
        average_normal = torch.nn.functional.normalize(normals[left] + aligned_right_normal, dim=-1, eps=_EPS)
        normal_offset = (delta_x * average_normal).sum(dim=-1).abs()
        # Tangential component: the SAME displacement vector's own residual
        # after removing the normal-direction part -- self-normalizing, no
        # spacing/scale reference needed (see class docstring for why
        # comparing against the candidate-graph spacing budget is degenerate).
        tangential_offset = (delta_x - normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
        normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)

        # --- robust, data-derived residual fence (the one disclosed constant) ---
        median_residual = torch.median(edge_residual)
        mad = torch.median((edge_residual - median_residual).abs())
        # 1.4826 is the standard MAD -> Gaussian-sigma-equivalent scale factor
        # (textbook robust-statistics constant, not scene-tuned).
        robust_sigma = 1.4826 * mad
        residual_threshold = float(median_residual + config.residual_mad_multiplier * robust_sigma)

        cut_residual_spatial = edge_residual > residual_threshold
        cut_parallel_spatial = normal_offset_ratio > config.parallel_sheet_normal_over_tangent_ratio
        cut_spatial = cut_residual_spatial | cut_parallel_spatial

        if progress is not None:
            progress(
                f"discontinuity test: spatial_edges={int(spatial_edges.shape[0])} "
                f"residual_threshold={residual_threshold:.6f} "
                f"cut_residual={int(cut_residual_spatial.sum())} cut_parallel_sheet={int(cut_parallel_spatial.sum())} "
                f"cut_total={int(cut_spatial.sum())}"
            )

        edge_residual_full = torch.zeros((int(graph.candidate_edges.shape[0]),), dtype=torch.float32, device=device)
        edge_offset_full = torch.zeros_like(edge_residual_full)
        cut_mask = torch.zeros((int(graph.candidate_edges.shape[0]),), dtype=torch.bool, device=device)
        cut_residual = torch.zeros_like(cut_mask)
        cut_parallel = torch.zeros_like(cut_mask)
        edge_residual_full[spatial_index] = edge_residual.to(torch.float32)
        edge_offset_full[spatial_index] = normal_offset_ratio.to(torch.float32)
        cut_mask[spatial_index] = cut_spatial
        cut_residual[spatial_index] = cut_residual_spatial
        cut_parallel[spatial_index] = cut_parallel_spatial

    kept = graph.candidate_edges[graph.spatial_edge_mask & ~cut_mask]
    roots = _connected_component_roots(count, kept, config.local)
    unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order)
    subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order]

    spatial_degree = torch.zeros((count,), dtype=torch.int64, device=device)
    if int(spatial_edges.shape[0]) > 0:
        ones = torch.ones((int(spatial_edges.shape[0]),), dtype=torch.int64, device=device)
        spatial_degree.index_add_(0, spatial_edges[:, 0], ones)
        spatial_degree.index_add_(0, spatial_edges[:, 1], ones)
    kept_degree = torch.zeros((count,), dtype=torch.int64, device=device)
    if int(kept.shape[0]) > 0:
        ones = torch.ones((int(kept.shape[0]),), dtype=torch.int64, device=device)
        kept_degree.index_add_(0, kept[:, 0], ones)
        kept_degree.index_add_(0, kept[:, 1], ones)

    ownership_kind = torch.full(
        (count,), OWNERSHIP_KINDS.index(OWNERSHIP_SMOOTH_CONTINUATION), dtype=torch.int8, device=device
    )
    no_kept = kept_degree == 0
    ownership_kind = torch.where(
        no_kept & (spatial_degree > 0),
        torch.tensor(OWNERSHIP_KINDS.index(OWNERSHIP_FALLBACK_ALL_EDGES_CUT), dtype=torch.int8, device=device),
        ownership_kind,
    )
    ownership_kind = torch.where(
        no_kept & (spatial_degree == 0),
        torch.tensor(OWNERSHIP_KINDS.index(OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR), dtype=torch.int8, device=device),
        ownership_kind,
    )

    return DiscontinuityFirstPartition(
        subset_ids=subset_ids,
        subset_count=int(order.shape[0]),
        subset_sizes=subset_sizes,
        ownership_kind=ownership_kind,
        normal_gradient_magnitude=normal_gradient_magnitude,
        shape_operator=shape_operator,
        edge_residual=edge_residual_full,
        edge_normal_offset_ratio=edge_offset_full,
        cut_mask=cut_mask,
        cut_reason_residual=cut_residual,
        cut_reason_parallel_sheet=cut_parallel,
        residual_threshold=residual_threshold,
        graph=graph,
        gaussian_ids=orientation.gaussian_ids,
        config=config,
    )


def count_spatially_disconnected_subsets(partition: DiscontinuityFirstPartition) -> int:
    """Independent re-derivation, same pattern as Worklog 96/97."""

    torch = require_torch()
    count = len(partition)
    if count == 0:
        return 0
    roots = _connected_component_roots(count, partition.kept_edges, partition.config.local)
    unique_pairs = torch.unique(partition.subset_ids * int(count) + roots)
    subset_of_pair = torch.div(unique_pairs, count, rounding_mode="floor")
    components_per_subset = torch.bincount(subset_of_pair, minlength=max(partition.subset_count, 1))
    return int((components_per_subset > 1).sum())


def discontinuity_first_accounting(partition: DiscontinuityFirstPartition) -> dict[str, Any]:
    """Full accounting block, matching Worklog 96/97's own field vocabulary
    wherever the same concept applies. Diagnostic only."""

    torch = require_torch()
    count = len(partition)
    sizes = partition.subset_sizes
    subset_count = max(partition.subset_count, 1)

    owner_histogram = torch.bincount(partition.subset_ids.reshape(-1), minlength=partition.subset_count)
    assigned = int((partition.subset_ids >= 0).sum())
    in_range = int(((partition.subset_ids >= 0) & (partition.subset_ids < max(partition.subset_count, 1))).sum())
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
    very_small = int((sizes <= VERY_SMALL_SUBSET_SIZE).sum()) if int(sizes.shape[0]) > 0 else 0
    largest_fraction = (float(sizes[0]) / count) if (int(sizes.shape[0]) > 0 and count) else 0.0

    ownership_counts = torch.bincount(partition.ownership_kind.reshape(-1).to(torch.int64), minlength=len(OWNERSHIP_KINDS))
    fallback_total = int(ownership_counts[1]) + int(ownership_counts[2])

    spatial_mask = partition.graph.spatial_edge_mask
    candidate_edge_count = int(partition.graph.candidate_edges.shape[0])
    spatial_edge_count = int(spatial_mask.sum())
    cut_edge_count = int(partition.cut_mask.sum())
    kept_edge_count = spatial_edge_count - cut_edge_count

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
        "subset_count": partition.subset_count,
        "subset_size": size_stats,
        "largest_subset_size": int(sizes[0]) if int(sizes.shape[0]) > 0 else 0,
        "largest_subset_surfel_fraction": largest_fraction,
        "singleton_subset_count": singleton,
        "singleton_subset_fraction": singleton / subset_count,
        "very_small_subset_size_threshold": VERY_SMALL_SUBSET_SIZE,
        "very_small_subset_count": very_small,
        "very_small_subset_fraction": very_small / subset_count,
        "spatially_disconnected_subset_count": count_spatially_disconnected_subsets(partition),
        "candidate_edge_count": candidate_edge_count,
        "spatial_edge_count": spatial_edge_count,
        "boundary_cut_edge_count": cut_edge_count,
        "boundary_cut_reason_counts": {
            CUT_REASON_RESIDUAL: int(partition.cut_reason_residual.sum()),
            CUT_REASON_PARALLEL_SHEET: int(partition.cut_reason_parallel_sheet.sum()),
            "both": int((partition.cut_reason_residual & partition.cut_reason_parallel_sheet).sum()),
        },
        "kept_edge_count": kept_edge_count,
        "residual_threshold": partition.residual_threshold,
        "ownership_kind_counts": {name: int(ownership_counts[index]) for index, name in enumerate(OWNERSHIP_KINDS)},
        "fallback_ownership_count": fallback_total,
        "fallback_ownership_fraction": fallback_total / max(count, 1),
        "partition_parameters": partition.config.payload(),
    }
