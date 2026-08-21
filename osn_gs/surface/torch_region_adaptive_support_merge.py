from __future__ import annotations

"""Worklog 101 -- rejected-interface attribution and region-adaptive support
for the Worklog 100 bilateral interface merge.

Worklog 100 is ACCEPTED as the merge certificate (region-conditioned +
bilateral). This module does not replace it and does not modify
`torch_bilateral_interface_region_merge.py` at all -- it is added
SEPARATELY so an A/B comparison against the exact Worklog 100 baseline stays
possible (`SUPPORT_MODE_FIXED_MASKED_KNN` reproduces Worklog 100's own
support acquisition byte-for-byte; every merge threshold is imported from
Worklog 100's own field values, never redefined).

The question this batch answers: does Worklog 100's very low merge count
(1,652 out of 1,763,096 evaluated interfaces) reflect genuine surface
discontinuity, or a MECHANICAL support-starvation deadlock --

    conservative WL97-with-gate initialization over-segments a smooth surface
        -> a small fragment's FIXED, GLOBAL k=8 neighbourhood is dominated by
           the OTHER region's surfels (masked out)
        -> fewer than 2 same-region neighbours survive the mask
        -> its region-conditioned shape operator is UNSUPPORTED
        -> bilateral smoothness can never be established
        -> the fragment can never merge, however smooth it really is

-- a circular dependency between the CONSERVATIVE initialization and the
FIXED-k masked support acquisition, not a claim about the true geometry.

Section 1: complete rejected-interface attribution (non-overlapping /
multi-label reasons), never collapsing UNSUPPORTED (unknown) into
DISCONTINUITY (supported and genuinely inconsistent).

Section 2: region-size-conditioned support statistics.

Section 3: `SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL` -- for a boundary
surfel in region R, search a BOUNDED, LOCAL pool of candidates (still capped,
never "the whole region") for same-region members, instead of masking a
fixed 8-nearest-neighbour set that may be dominated by the opposite region.
Locality is enforced by REUSING `local.spatial_connect_spacing_multiplier`
(the exact same local-scale contract the candidate graph itself already
uses) -- no new absolute radius, no swept parameter. The BILATERAL
certificate itself is untouched: an interface is still smooth only when BOTH
independently-supported directions pass the SAME (reused) residual
threshold, majority fraction, and support/extent floors as Worklog 100.
Adaptive support can only help a fragment ACQUIRE legitimate same-region
evidence -- it can never lower what counts as smooth.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CandidateGraph,
    CoverageFirstPartitionConfig,
    SurfaceOrientationEvidence,
    VERY_SMALL_SUBSET_SIZE,
    build_candidate_graph,
)
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    RegionCoherenceConfig,
    RegionCoherentPartition,
    partition_surfels_region_coherent,
)
from osn_gs.surface.torch_bilateral_interface_region_merge import (
    _MIN_SAME_REGION_SUPPORT,
    _fit_region_conditioned_shape_operators,
)
from osn_gs.surface.torch_discontinuity_first_surfel_partition import (
    _predicted_delta_n_t,
    _tangent_plane_components,
    _knn,
    _auto_chunk_size,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

SUPPORT_MODE_FIXED_MASKED_KNN = "fixed_masked_knn"  # exact Worklog 100 behaviour
SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL = "adaptive_same_region_local"  # new (section 7-9)
SUPPORT_MODES = (SUPPORT_MODE_FIXED_MASKED_KNN, SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)

# Region-size bins for the support-starvation measurement (directive section 3).
_SIZE_BIN_EDGES = (1, 2, 5, 9, 17, 33, 65)  # left edges; last bin is open-ended (>64)
_SIZE_BIN_LABELS = ("1", "2-4", "5-8", "9-16", "17-32", "33-64", ">64")

# Rejection-reason keys (directive section 2) -- every evaluated interface
# gets an explicit boolean for each, never a single collapsed label.
REASON_INSUFFICIENT_SUPPORT_A = "insufficient_region_support_A"
REASON_INSUFFICIENT_SUPPORT_B = "insufficient_region_support_B"
REASON_INSUFFICIENT_SUPPORT_BOTH = "insufficient_region_support_both"
REASON_UNIQUE_SUPPORT_FAILURE = "interface_unique_support_failure"
REASON_EXTENT_FAILURE = "interface_extent_failure"
REASON_RESIDUAL_FAILURE_A_TO_B = "directional_residual_failure_A_to_B"
REASON_RESIDUAL_FAILURE_B_TO_A = "directional_residual_failure_B_to_A"
REASON_POSITIONAL_FAILURE = "positional_continuity_failure"
REASON_BILATERAL_FRACTION_FAILURE = "bilateral_smooth_fraction_failure"
REASON_WOULD_PASS_IF_SUPPORTED = "would_pass_geometry_tests_but_for_support"

REJECTION_REASON_KEYS = (
    REASON_INSUFFICIENT_SUPPORT_A,
    REASON_INSUFFICIENT_SUPPORT_B,
    REASON_INSUFFICIENT_SUPPORT_BOTH,
    REASON_UNIQUE_SUPPORT_FAILURE,
    REASON_EXTENT_FAILURE,
    REASON_RESIDUAL_FAILURE_A_TO_B,
    REASON_RESIDUAL_FAILURE_B_TO_A,
    REASON_POSITIONAL_FAILURE,
    REASON_BILATERAL_FRACTION_FAILURE,
    REASON_WOULD_PASS_IF_SUPPORTED,
)


@dataclass(frozen=True)
class AdaptiveSupportConfig:
    """Every threshold that governs the MERGE CERTIFICATE (as opposed to
    support acquisition) is identical in NAME and DEFAULT VALUE to Worklog
    100's `BilateralInterfaceMergeConfig` -- none of them is touched by this
    batch (directive section 13). Only `support_mode` and the ONE derived
    (not swept) pool-size multiplier it needs are new."""

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    region: RegionCoherenceConfig = field(
        default_factory=lambda: RegionCoherenceConfig(require_positional_continuity=True)
    )

    shape_operator_neighbor_count: int = 0  # 0 => local.neighbor_count; target SAME-REGION support size
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0  # reused verbatim from Worklog 98/99/100
    parallel_sheet_normal_over_tangent_ratio: float = 1.0  # reused verbatim
    interface_smooth_majority_fraction: float = 0.5  # reused verbatim, NOT swept

    support_mode: str = SUPPORT_MODE_FIXED_MASKED_KNN

    # Only consulted when support_mode == ADAPTIVE_SAME_REGION_LOCAL. The
    # candidate POOL a same-region search draws from is `neighbor_count *
    # this multiplier` -- derived from the existing neighbor_count (not an
    # independent new magic number), and still hard-bounded (never "the
    # whole region", directive section 8). The LOCALITY bound applied to
    # that pool is `local.spatial_connect_spacing_multiplier` -- the SAME
    # distance contract the candidate graph itself already uses, not a new
    # absolute radius.
    adaptive_pool_size_multiplier: int = 4

    def min_unique_surfels_per_interface_side(self) -> int:
        return int(self.local.neighbor_count)

    def min_interface_extent_in_spacing_units(self) -> float:
        return float(self.local.spatial_connect_spacing_multiplier)

    def resolved_shape_operator_neighbor_count(self) -> int:
        return int(self.shape_operator_neighbor_count) or int(self.local.neighbor_count)

    def resolved_pool_size(self) -> int:
        return self.resolved_shape_operator_neighbor_count() * int(self.adaptive_pool_size_multiplier)

    def payload(self) -> dict[str, Any]:
        return {
            "local": self.local.payload(),
            "region": self.region.payload(),
            "shape_operator_neighbor_count": self.resolved_shape_operator_neighbor_count(),
            "shape_operator_ridge": self.shape_operator_ridge,
            "residual_mad_multiplier": self.residual_mad_multiplier,
            "parallel_sheet_normal_over_tangent_ratio": self.parallel_sheet_normal_over_tangent_ratio,
            "interface_smooth_majority_fraction": self.interface_smooth_majority_fraction,
            "min_unique_surfels_per_interface_side": self.min_unique_surfels_per_interface_side(),
            "min_interface_extent_in_spacing_units": self.min_interface_extent_in_spacing_units(),
            "min_same_region_support_for_shape_operator": _MIN_SAME_REGION_SUPPORT,
            "support_mode": self.support_mode,
            "adaptive_pool_size": self.resolved_pool_size() if self.support_mode == SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL else None,
            "adaptive_pool_size_multiplier": self.adaptive_pool_size_multiplier,
            "adaptive_locality_bound_derivation": "local.spatial_connect_spacing_multiplier * local_spacing (same contract as the candidate graph)",
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


def _size_bin_index(size: Any) -> Any:
    """Vectorized bucket index into `_SIZE_BIN_LABELS` for a tensor of
    positive integer sizes."""

    torch = require_torch()
    edges = torch.tensor(_SIZE_BIN_EDGES, dtype=size.dtype, device=size.device)
    # searchsorted(edges, size, right=True) - 1 gives the bin whose left edge
    # is <= size, capped at the last bin.
    index = torch.searchsorted(edges, size, right=True) - 1
    return index.clamp(0, len(_SIZE_BIN_LABELS) - 1)


def _support_neighbor_pool(
    support_mode: str,
    boundary_nodes: Any,
    node_root: Any,
    pool_neighbor_index: Any,
    pool_neighbor_distance: Any,
    local_spacing: Any,
    spacing_multiplier: float,
    target_k: int,
) -> tuple[Any, Any]:
    """Returns `(neighbor_index, valid_mask)` for the boundary nodes, under
    the requested support-acquisition mode. `pool_neighbor_index`/
    `pool_neighbor_distance` are the FULL-scene, STATIC (position-only,
    never recomputed per round) kNN pool -- width `target_k` in FIXED mode
    (identical to Worklog 100's own `full_neighbor_index`), width
    `target_k * adaptive_pool_size_multiplier` in ADAPTIVE mode.
    """

    torch = require_torch()
    neighbor_index = pool_neighbor_index[boundary_nodes]
    same_region_mask = node_root[neighbor_index] == node_root[boundary_nodes].unsqueeze(1)

    if support_mode == SUPPORT_MODE_FIXED_MASKED_KNN:
        return neighbor_index, same_region_mask

    if support_mode != SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL:
        raise ValueError(f"unknown support_mode: {support_mode!r}")

    neighbor_distance = pool_neighbor_distance[boundary_nodes]
    locality_bound = (spacing_multiplier * local_spacing[boundary_nodes]).unsqueeze(1)
    within_locality_mask = neighbor_distance <= locality_bound
    valid_mask = same_region_mask & within_locality_mask
    # keep only the TARGET_K closest valid candidates (the pool is already
    # distance-sorted ascending, since `_knn` returns nearest-first) -- a
    # bounded, deterministic "same-region kNN", never the unbounded full region.
    rank = valid_mask.to(torch.int64).cumsum(dim=1)
    keep_mask = valid_mask & (rank <= target_k)
    return neighbor_index, keep_mask


def _evaluate_interface_evidence(
    orientation: SurfaceOrientationEvidence,
    pool_neighbor_index: Any,
    pool_neighbor_distance: Any,
    local_spacing: Any,
    node_root: Any,
    edge_left: Any,
    edge_right: Any,
    config: AdaptiveSupportConfig,
) -> dict[str, Any]:
    """Region-conditioned, bilateral evidence for the given edge set, under
    `config.support_mode`. Mirrors
    `torch_bilateral_interface_region_merge._region_conditioned_bilateral_residuals`
    exactly except for HOW the same-region neighbour set is acquired (the
    one axis this module changes) -- reuses the SAME (unmodified, imported)
    `_fit_region_conditioned_shape_operators` for the actual fit.
    """

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    tangent_u = orientation.tangent_axis_u
    tangent_v = orientation.tangent_axis_v
    device = positions.device
    target_k = config.resolved_shape_operator_neighbor_count()

    boundary_nodes = torch.unique(torch.cat([edge_left, edge_right]))
    neighbor_index_boundary, valid_mask = _support_neighbor_pool(
        config.support_mode, boundary_nodes, node_root, pool_neighbor_index, pool_neighbor_distance,
        local_spacing, float(config.local.spatial_connect_spacing_multiplier), target_k,
    )
    shape_operator_boundary, support_count_boundary = _fit_region_conditioned_shape_operators(
        positions, normals, tangent_u, tangent_v,
        boundary_nodes, neighbor_index_boundary, valid_mask, float(config.shape_operator_ridge),
    )
    unsupported_boundary = support_count_boundary < _MIN_SAME_REGION_SUPPORT

    inverse_map = torch.full((int(positions.shape[0]),), -1, dtype=torch.int64, device=device)
    inverse_map[boundary_nodes] = torch.arange(int(boundary_nodes.shape[0]), dtype=torch.int64, device=device)
    idx_left = inverse_map[edge_left]
    idx_right = inverse_map[edge_right]

    shape_operator_left = shape_operator_boundary[idx_left]
    shape_operator_right = shape_operator_boundary[idx_right]
    unsupported_left = unsupported_boundary[idx_left]
    unsupported_right = unsupported_boundary[idx_right]
    support_count_left = support_count_boundary[idx_left]
    support_count_right = support_count_boundary[idx_right]

    delta_x = positions[edge_right] - positions[edge_left]
    delta_x_t_left = _tangent_plane_components(delta_x, tangent_u[edge_left], tangent_v[edge_left])
    delta_x_t_right = _tangent_plane_components(-delta_x, tangent_u[edge_right], tangent_v[edge_right])

    sign_lr = torch.where((normals[edge_left] * normals[edge_right]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
    aligned_right_normal = normals[edge_right] * sign_lr
    delta_n_left = aligned_right_normal - normals[edge_left]
    delta_n_t_left = _tangent_plane_components(delta_n_left, tangent_u[edge_left], tangent_v[edge_left])

    aligned_left_normal = normals[edge_left] * sign_lr
    delta_n_right = aligned_left_normal - normals[edge_right]
    delta_n_t_right = _tangent_plane_components(delta_n_right, tangent_u[edge_right], tangent_v[edge_right])

    predicted_left = _predicted_delta_n_t(shape_operator_left, delta_x_t_left)
    predicted_right = _predicted_delta_n_t(shape_operator_right, delta_x_t_right)
    r_left_own_model = (delta_n_t_left - predicted_left).norm(dim=-1)
    r_right_own_model = (delta_n_t_right - predicted_right).norm(dim=-1)

    average_normal = torch.nn.functional.normalize(normals[edge_left] + aligned_right_normal, dim=-1, eps=_EPS)
    normal_offset = (delta_x * average_normal).sum(dim=-1).abs()
    tangential_offset = (delta_x - normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
    normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)

    return {
        "r_left_own_model": r_left_own_model,
        "r_right_own_model": r_right_own_model,
        "unsupported_left": unsupported_left,
        "unsupported_right": unsupported_right,
        "support_count_left": support_count_left,
        "support_count_right": support_count_right,
        "normal_offset_ratio": normal_offset_ratio,
        "boundary_nodes": boundary_nodes,
        "support_count_boundary": support_count_boundary,
        "unsupported_boundary": unsupported_boundary,
    }


def _compute_global_residual_threshold(
    orientation: SurfaceOrientationEvidence,
    pool_neighbor_index: Any,
    pool_neighbor_distance: Any,
    local_spacing: Any,
    node_root: Any,
    graph: CandidateGraph,
    config: AdaptiveSupportConfig,
) -> float:
    """Same formula and same compute-once-over-all-spatial-edges timing as
    Worklog 100's `_compute_initial_residual_threshold` (see that function's
    docstring for why) -- reimplemented here only because it must route
    through THIS module's support-mode-aware evidence function."""

    torch = require_torch()
    spatial_edges = graph.candidate_edges[graph.spatial_edge_mask]
    if int(spatial_edges.shape[0]) == 0:
        return 0.0
    evidence = _evaluate_interface_evidence(
        orientation, pool_neighbor_index, pool_neighbor_distance, local_spacing, node_root,
        spatial_edges[:, 0], spatial_edges[:, 1], config,
    )
    supported_left = evidence["r_left_own_model"][~evidence["unsupported_left"]]
    supported_right = evidence["r_right_own_model"][~evidence["unsupported_right"]]
    pooled_residual = torch.cat([supported_left, supported_right])
    if int(pooled_residual.shape[0]) == 0:
        return 0.0
    median_residual = torch.median(pooled_residual)
    mad = torch.median((pooled_residual - median_residual).abs())
    return float(median_residual + config.residual_mad_multiplier * 1.4826 * mad)


class _RegionUnionFind:
    """Identical to Worklog 99/100's own DSU."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return root_a
        if root_a > root_b:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        return root_a


@dataclass(frozen=True)
class AttributedInterface:
    """One interface's full, non-overlapping-where-possible / multi-label
    attribution (directive section 2)."""

    region_a: int
    region_b: int
    round: int
    edge_count: int
    unique_surfel_count_a: int
    unique_surfel_count_b: int
    extent_in_spacing_units: float
    unsupported_a_to_b_count: int
    unsupported_b_to_a_count: int
    supported_unsmooth_a_to_b_count: int
    supported_unsmooth_b_to_a_count: int
    supported_smooth_a_to_b_count: int
    supported_smooth_b_to_a_count: int
    bilateral_smooth_fraction: float
    mean_normal_offset_ratio: float
    accepted: bool
    reasons: dict[str, bool]


@dataclass(frozen=True)
class RegionAdaptiveSupportPartition:
    subset_ids: Any
    subset_count: int
    subset_sizes: Any

    initial_region_ids: Any
    initial_region_count: int
    final_region_of_initial: Any
    round_count: int
    merge_provenance: tuple[dict[str, Any], ...]
    attributed_interfaces: tuple[AttributedInterface, ...]
    support_by_region_size: tuple[dict[str, Any], ...]  # one row per size bin

    graph: CandidateGraph
    gaussian_ids: Any
    initial_partition: RegionCoherentPartition
    config: AdaptiveSupportConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def cross_region_edges_mask(self) -> Any:
        left, right = self.graph.candidate_edges[:, 0], self.graph.candidate_edges[:, 1]
        return self.graph.spatial_edge_mask & (self.initial_region_ids[left] != self.initial_region_ids[right])

    @property
    def accepted_merge_edges_mask(self) -> Any:
        left, right = self.graph.candidate_edges[:, 0], self.graph.candidate_edges[:, 1]
        return self.cross_region_edges_mask & (self.subset_ids[left] == self.subset_ids[right])

    @property
    def rejected_interface_edges_mask(self) -> Any:
        return self.cross_region_edges_mask & ~self.accepted_merge_edges_mask


def partition_surfels_region_adaptive_support(
    orientation: SurfaceOrientationEvidence,
    config: AdaptiveSupportConfig | None = None,
    *,
    max_rounds: int = 64,
    progress: Callable[[str], None] | None = None,
) -> RegionAdaptiveSupportPartition:
    """Same overall algorithm as Worklog 100
    (`partition_surfels_bilateral_interface`): positional-gated WL97
    initialization, region adjacency over the local candidate graph, a
    region-conditioned + bilateral merge certificate, deterministic
    round-based sequential DSU merging. The ONLY axis this function varies
    is `config.support_mode` (see module docstring); every merge threshold
    is the same field, same default, same meaning as Worklog 100's. Also
    collects the full rejected-interface attribution and region-size-binned
    support statistics the directive requires.
    """

    torch = require_torch()
    config = config or AdaptiveSupportConfig()
    if config.support_mode not in SUPPORT_MODES:
        raise ValueError(f"unknown support_mode: {config.support_mode!r}")
    if config.region.local != config.local:
        raise ValueError(
            "AdaptiveSupportConfig.region.local must match config.local -- the candidate graph is shared "
            "verbatim between region initialization and interface evaluation."
        )
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)
    initial_partition = partition_surfels_region_coherent(orientation, config.region, progress=progress)
    initial_region_ids = initial_partition.subset_ids
    initial_region_count = initial_partition.subset_count

    if count == 0 or initial_region_count <= 1:
        return RegionAdaptiveSupportPartition(
            subset_ids=initial_region_ids, subset_count=initial_region_count,
            subset_sizes=initial_partition.subset_sizes,
            initial_region_ids=initial_region_ids, initial_region_count=initial_region_count,
            final_region_of_initial=torch.arange(max(initial_region_count, 0), dtype=torch.int64, device=device),
            round_count=0, merge_provenance=(), attributed_interfaces=(), support_by_region_size=(),
            graph=graph, gaussian_ids=orientation.gaussian_ids, initial_partition=initial_partition, config=config,
        )

    spatial_mask = graph.spatial_edge_mask
    candidate_left = graph.candidate_edges[:, 0]
    candidate_right = graph.candidate_edges[:, 1]
    local_spacing = graph.local_spacing
    edge_spacing = (local_spacing[candidate_left] + local_spacing[candidate_right]) / 2.0
    positions_left = positions[candidate_left]
    positions_right = positions[candidate_right]

    target_k = config.resolved_shape_operator_neighbor_count()
    pool_k = target_k if config.support_mode == SUPPORT_MODE_FIXED_MASKED_KNN else config.resolved_pool_size()
    pool_k = min(pool_k, max(count - 1, 1))
    chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
    if progress is not None:
        progress(f"precomputing static kNN pool for support acquisition: mode={config.support_mode} pool_k={pool_k}")
    pool_neighbor_index, pool_neighbor_distance = _knn(positions, pool_k, chunk_size, progress)

    residual_threshold = _compute_global_residual_threshold(
        orientation, pool_neighbor_index, pool_neighbor_distance, local_spacing, initial_region_ids, graph, config
    )
    if progress is not None:
        progress(f"global region-conditioned residual threshold (computed once): {residual_threshold:.6f}")

    dsu = _RegionUnionFind(initial_region_count)
    merge_provenance: list[dict[str, Any]] = []
    attributed_interfaces: list[AttributedInterface] = []

    # region-size-conditioned support accumulators (directive section 3),
    # collected across every round's boundary-node population.
    size_bin_boundary_count = [0] * len(_SIZE_BIN_LABELS)
    size_bin_supported_count = [0] * len(_SIZE_BIN_LABELS)
    size_bin_residual_sum = [0.0] * len(_SIZE_BIN_LABELS)
    size_bin_residual_sq_sum = [0.0] * len(_SIZE_BIN_LABELS)
    size_bin_residual_count = [0] * len(_SIZE_BIN_LABELS)

    min_side = config.min_unique_surfels_per_interface_side()
    min_extent = config.min_interface_extent_in_spacing_units()
    majority = config.interface_smooth_majority_fraction

    round_index = 0
    for round_index in range(1, max_rounds + 1):
        current_root = torch.tensor(
            [dsu.find(region) for region in range(initial_region_count)], dtype=torch.int64, device=device
        )
        node_root = current_root[initial_region_ids]
        root_left = node_root[candidate_left]
        root_right = node_root[candidate_right]
        cross_mask = spatial_mask & (root_left != root_right)
        if progress is not None:
            progress(f"[round {round_index}] evaluating cross-region interfaces over {int(cross_mask.sum())} edges")
        if not bool(cross_mask.any()):
            round_index -= 1
            break

        cross_left = candidate_left[cross_mask]
        cross_right = candidate_right[cross_mask]
        cross_root_left = root_left[cross_mask]
        cross_root_right = root_right[cross_mask]

        evidence = _evaluate_interface_evidence(
            orientation, pool_neighbor_index, pool_neighbor_distance, local_spacing, node_root,
            cross_left, cross_right, config,
        )
        if progress is not None:
            progress(f"[round {round_index}] region-conditioned fit over {int(evidence['boundary_nodes'].shape[0])} boundary nodes")

        # --- region-size-conditioned support bookkeeping (section 3) ------
        current_region_size = torch.bincount(node_root, minlength=initial_region_count)
        boundary_region_size = current_region_size[node_root[evidence["boundary_nodes"]]]
        boundary_bin = _size_bin_index(boundary_region_size)
        supported_boundary = ~evidence["unsupported_boundary"]
        for bin_index in range(len(_SIZE_BIN_LABELS)):
            in_bin = boundary_bin == bin_index
            size_bin_boundary_count[bin_index] += int(in_bin.sum().item())
            size_bin_supported_count[bin_index] += int((in_bin & supported_boundary).sum().item())
        # residual distribution (both directions, supported only), binned by
        # the QUERY side's own region size at evaluation time.
        left_bin = _size_bin_index(current_region_size[node_root[cross_left]])
        right_bin = _size_bin_index(current_region_size[node_root[cross_right]])
        residual_left_supported = evidence["r_left_own_model"][~evidence["unsupported_left"]]
        residual_left_bins = left_bin[~evidence["unsupported_left"]]
        residual_right_supported = evidence["r_right_own_model"][~evidence["unsupported_right"]]
        residual_right_bins = right_bin[~evidence["unsupported_right"]]
        for values, bins in ((residual_left_supported, residual_left_bins), (residual_right_supported, residual_right_bins)):
            for bin_index in range(len(_SIZE_BIN_LABELS)):
                selected = values[bins == bin_index]
                if int(selected.shape[0]) == 0:
                    continue
                size_bin_residual_count[bin_index] += int(selected.shape[0])
                size_bin_residual_sum[bin_index] += float(selected.sum().item())
                size_bin_residual_sq_sum[bin_index] += float((selected * selected).sum().item())

        # --- per-edge classification (identical formulas to Worklog 100) --
        smooth_left = (~evidence["unsupported_left"]) & (evidence["r_left_own_model"] <= residual_threshold)
        smooth_right = (~evidence["unsupported_right"]) & (evidence["r_right_own_model"] <= residual_threshold)
        positional_ok = evidence["normal_offset_ratio"] <= config.parallel_sheet_normal_over_tangent_ratio
        bilateral_smooth_edge = smooth_left & smooth_right & positional_ok

        left_is_a = cross_root_left < cross_root_right
        side_a_node = torch.where(left_is_a, cross_left, cross_right)
        side_b_node = torch.where(left_is_a, cross_right, cross_left)
        unsupported_a_edge = torch.where(left_is_a, evidence["unsupported_left"], evidence["unsupported_right"])
        unsupported_b_edge = torch.where(left_is_a, evidence["unsupported_right"], evidence["unsupported_left"])
        smooth_a_edge = torch.where(left_is_a, smooth_left, smooth_right)
        smooth_b_edge = torch.where(left_is_a, smooth_right, smooth_left)
        residual_a_edge = torch.where(left_is_a, evidence["r_left_own_model"], evidence["r_right_own_model"])
        residual_b_edge = torch.where(left_is_a, evidence["r_right_own_model"], evidence["r_left_own_model"])
        supported_unsmooth_a_edge = (~unsupported_a_edge) & (residual_a_edge > residual_threshold)
        supported_unsmooth_b_edge = (~unsupported_b_edge) & (residual_b_edge > residual_threshold)

        cross_spacing = edge_spacing[cross_mask].clamp_min(_EPS)
        cross_mid = (positions_left[cross_mask] + positions_right[cross_mask]) / 2.0
        cross_ratio = evidence["normal_offset_ratio"]

        pair_low = torch.minimum(root_left, root_right)[cross_mask]
        pair_high = torch.maximum(root_left, root_right)[cross_mask]
        pair_key = pair_low * int(initial_region_count) + pair_high
        unique_keys, group_id = torch.unique(pair_key, return_inverse=True)
        group_count = int(unique_keys.shape[0])

        def _group_sum(values):
            return torch.bincount(group_id, weights=values.to(torch.float32), minlength=group_count)

        edge_count_per_group = torch.bincount(group_id, minlength=group_count)
        unsupported_a_sum = _group_sum(unsupported_a_edge)
        unsupported_b_sum = _group_sum(unsupported_b_edge)
        supported_unsmooth_a_sum = _group_sum(supported_unsmooth_a_edge)
        supported_unsmooth_b_sum = _group_sum(supported_unsmooth_b_edge)
        smooth_a_sum = _group_sum(smooth_a_edge)
        smooth_b_sum = _group_sum(smooth_b_edge)
        bilateral_sum = _group_sum(bilateral_smooth_edge)
        positional_fail_sum = _group_sum(~positional_ok)
        ratio_sum_per_group = torch.bincount(group_id, weights=cross_ratio, minlength=group_count)

        side_a_key = group_id.to(torch.int64) * int(count) + side_a_node
        side_b_key = group_id.to(torch.int64) * int(count) + side_b_node
        unique_a_group = torch.div(torch.unique(side_a_key), count, rounding_mode="floor")
        unique_b_group = torch.div(torch.unique(side_b_key), count, rounding_mode="floor")
        unique_count_a_per_group = torch.bincount(unique_a_group, minlength=group_count)
        unique_count_b_per_group = torch.bincount(unique_b_group, minlength=group_count)

        min_xyz = torch.full((group_count, 3), float("inf"), dtype=torch.float32, device=device)
        max_xyz = torch.full((group_count, 3), float("-inf"), dtype=torch.float32, device=device)
        for axis in range(3):
            min_xyz[:, axis].scatter_reduce_(0, group_id, cross_mid[:, axis], reduce="amin", include_self=True)
            max_xyz[:, axis].scatter_reduce_(0, group_id, cross_mid[:, axis], reduce="amax", include_self=True)
        extent_diagonal = (max_xyz - min_xyz).norm(dim=-1)
        mean_spacing_per_group = torch.bincount(group_id, weights=cross_spacing, minlength=group_count) / edge_count_per_group.clamp_min(1)
        extent_in_spacing_units = extent_diagonal / mean_spacing_per_group.clamp_min(_EPS)

        bilateral_smooth_fraction = bilateral_sum / edge_count_per_group.clamp_min(1)
        mean_ratio = ratio_sum_per_group / edge_count_per_group.clamp_min(1)

        support_ok_a = unique_count_a_per_group >= min_side
        support_ok_b = unique_count_b_per_group >= min_side
        extent_ok = extent_in_spacing_units >= min_extent
        positional_ok_group = mean_ratio <= config.parallel_sheet_normal_over_tangent_ratio
        bilateral_ok = bilateral_smooth_fraction >= majority
        accept = support_ok_a & support_ok_b & extent_ok & bilateral_ok & positional_ok_group

        group_region_a = torch.div(unique_keys, int(initial_region_count), rounding_mode="floor")
        group_region_b = unique_keys - group_region_a * int(initial_region_count)

        # "would pass geometry tests but for support": restrict to SUPPORTED
        # edges only, recompute the bilateral fraction among those, and
        # check whether the aggregate + support/extent/positional tests
        # would have accepted -- directly measuring the circular-dependency
        # hypothesis at the interface level, not just per-edge.
        supported_edge = (~unsupported_a_edge) & (~unsupported_b_edge)
        bilateral_among_supported_sum = _group_sum(bilateral_smooth_edge & supported_edge)
        supported_edge_count_per_group = _group_sum(supported_edge)
        bilateral_fraction_if_supported = torch.where(
            supported_edge_count_per_group > 0,
            bilateral_among_supported_sum / supported_edge_count_per_group.clamp_min(1),
            torch.zeros_like(bilateral_smooth_fraction),
        )
        would_pass_if_supported = (
            (~accept)
            & (unsupported_a_sum + unsupported_b_sum > 0)
            & support_ok_a & support_ok_b & extent_ok & positional_ok_group
            & (bilateral_fraction_if_supported >= majority)
        )

        for group in range(group_count):
            unsupported_a = int(unsupported_a_sum[group].item())
            unsupported_b = int(unsupported_b_sum[group].item())
            reasons = {
                REASON_INSUFFICIENT_SUPPORT_A: unsupported_a > 0,
                REASON_INSUFFICIENT_SUPPORT_B: unsupported_b > 0,
                REASON_INSUFFICIENT_SUPPORT_BOTH: unsupported_a > 0 and unsupported_b > 0,
                REASON_UNIQUE_SUPPORT_FAILURE: not bool(support_ok_a[group].item() and support_ok_b[group].item()),
                REASON_EXTENT_FAILURE: not bool(extent_ok[group].item()),
                REASON_RESIDUAL_FAILURE_A_TO_B: bool(supported_unsmooth_a_sum[group].item() > 0),
                REASON_RESIDUAL_FAILURE_B_TO_A: bool(supported_unsmooth_b_sum[group].item() > 0),
                REASON_POSITIONAL_FAILURE: bool(positional_fail_sum[group].item() > 0),
                REASON_BILATERAL_FRACTION_FAILURE: not bool(bilateral_ok[group].item()),
                REASON_WOULD_PASS_IF_SUPPORTED: bool(would_pass_if_supported[group].item()),
            }
            attributed_interfaces.append(
                AttributedInterface(
                    region_a=int(group_region_a[group].item()), region_b=int(group_region_b[group].item()),
                    round=round_index, edge_count=int(edge_count_per_group[group].item()),
                    unique_surfel_count_a=int(unique_count_a_per_group[group].item()),
                    unique_surfel_count_b=int(unique_count_b_per_group[group].item()),
                    extent_in_spacing_units=float(extent_in_spacing_units[group].item()),
                    unsupported_a_to_b_count=unsupported_a, unsupported_b_to_a_count=unsupported_b,
                    supported_unsmooth_a_to_b_count=int(supported_unsmooth_a_sum[group].item()),
                    supported_unsmooth_b_to_a_count=int(supported_unsmooth_b_sum[group].item()),
                    supported_smooth_a_to_b_count=int(smooth_a_sum[group].item()),
                    supported_smooth_b_to_a_count=int(smooth_b_sum[group].item()),
                    bilateral_smooth_fraction=float(bilateral_smooth_fraction[group].item()),
                    mean_normal_offset_ratio=float(mean_ratio[group].item()),
                    accepted=bool(accept[group].item()),
                    reasons=reasons,
                )
            )

        accepted_index = torch.nonzero(accept, as_tuple=False).reshape(-1)
        if int(accepted_index.shape[0]) == 0:
            round_index -= 1
            break
        order = accepted_index[torch.argsort(group_region_b[accepted_index], stable=True)]
        order = order[torch.argsort(group_region_a[order], stable=True)]
        order = order[torch.argsort(bilateral_smooth_fraction[order], descending=True, stable=True)]

        applied_this_round = 0
        for group in order.tolist():
            region_a = int(group_region_a[group].item())
            region_b = int(group_region_b[group].item())
            if dsu.find(region_a) == dsu.find(region_b):
                continue
            dsu.union(region_a, region_b)
            applied_this_round += 1
            merge_provenance.append(
                {
                    "round": round_index, "region_a": region_a, "region_b": region_b,
                    "edge_count": int(edge_count_per_group[group].item()),
                    "unique_surfel_count_a": int(unique_count_a_per_group[group].item()),
                    "unique_surfel_count_b": int(unique_count_b_per_group[group].item()),
                    "extent_in_spacing_units": float(extent_in_spacing_units[group].item()),
                    "bilateral_smooth_fraction": float(bilateral_smooth_fraction[group].item()),
                    "mean_normal_offset_ratio": float(mean_ratio[group].item()),
                }
            )
        if progress is not None:
            progress(
                f"[round {round_index}] interfaces={group_count} accepted={int(accept.sum())} "
                f"merges_applied={applied_this_round}"
            )
        if applied_this_round == 0:
            break
    else:
        raise RuntimeError("Region-adaptive-support merge: did not converge within max_rounds.")

    final_region_of_initial = torch.tensor(
        [dsu.find(region) for region in range(initial_region_count)], dtype=torch.int64, device=device
    )
    final_root_of_surfel = final_region_of_initial[initial_region_ids]
    unique_roots, inverse, counts = torch.unique(final_root_of_surfel, return_inverse=True, return_counts=True)
    order_by_size = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order_by_size)
    subset_id_of_position[order_by_size] = torch.arange(int(order_by_size.shape[0]), dtype=order_by_size.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order_by_size]

    support_by_region_size = []
    for bin_index, label in enumerate(_SIZE_BIN_LABELS):
        boundary_count = size_bin_boundary_count[bin_index]
        residual_count = size_bin_residual_count[bin_index]
        residual_mean = (size_bin_residual_sum[bin_index] / residual_count) if residual_count > 0 else None
        residual_variance = (
            (size_bin_residual_sq_sum[bin_index] / residual_count - residual_mean * residual_mean)
            if residual_count > 0 else None
        )
        support_by_region_size.append(
            {
                "region_size_bin": label,
                "boundary_node_observations": boundary_count,
                "supported_boundary_node_observations": size_bin_supported_count[bin_index],
                "supported_fraction": (size_bin_supported_count[bin_index] / boundary_count) if boundary_count > 0 else None,
                "unsupported_fraction": (1.0 - size_bin_supported_count[bin_index] / boundary_count) if boundary_count > 0 else None,
                "residual_observation_count": residual_count,
                "residual_mean_when_supported": residual_mean,
                "residual_std_when_supported": (residual_variance ** 0.5) if residual_variance is not None and residual_variance >= 0 else None,
            }
        )

    return RegionAdaptiveSupportPartition(
        subset_ids=subset_ids, subset_count=int(order_by_size.shape[0]), subset_sizes=subset_sizes,
        initial_region_ids=initial_region_ids, initial_region_count=initial_region_count,
        final_region_of_initial=final_region_of_initial, round_count=round_index,
        merge_provenance=tuple(merge_provenance), attributed_interfaces=tuple(attributed_interfaces),
        support_by_region_size=tuple(support_by_region_size),
        graph=graph, gaussian_ids=orientation.gaussian_ids, initial_partition=initial_partition, config=config,
    )


def region_adaptive_support_accounting(partition: RegionAdaptiveSupportPartition) -> dict[str, Any]:
    """Full accounting block, matching prior worklogs' field vocabulary,
    plus the attribution/support-by-size diagnostics this batch adds."""

    torch = require_torch()
    count = len(partition)
    sizes = partition.subset_sizes
    subset_count = max(partition.subset_count, 1)

    owner_histogram = torch.bincount(partition.subset_ids.reshape(-1), minlength=partition.subset_count)
    assigned_surfel_count = int((partition.subset_ids >= 0).sum())
    in_range = int(((partition.subset_ids >= 0) & (partition.subset_ids < max(partition.subset_count, 1))).sum())
    sizes_match = bool(
        int(sizes.shape[0]) == int(owner_histogram.shape[0]) and torch.equal(owner_histogram.to(sizes.dtype), sizes)
    )
    unassigned_surfel_count = count - assigned_surfel_count
    multiply_owned_surfel_count = 0

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

    accepted = [i for i in partition.attributed_interfaces if i.accepted]
    rejected = [i for i in partition.attributed_interfaces if not i.accepted]
    reason_counts = {key: sum(1 for i in rejected if i.reasons.get(key)) for key in REJECTION_REASON_KEYS}

    final_of_initial = partition.final_region_of_initial
    if int(final_of_initial.shape[0]) > 0:
        _, _, fragment_counts = torch.unique(final_of_initial, return_inverse=True, return_counts=True)
        max_fragments_merged = int(fragment_counts.max().item())
    else:
        max_fragments_merged = 0

    return {
        "input_surfel_count": count,
        "assigned_surfel_count": assigned_surfel_count,
        "unassigned_surfel_count": unassigned_surfel_count,
        "multiply_owned_surfel_count": multiply_owned_surfel_count,
        "subset_id_out_of_range_count": count - in_range,
        "subset_sizes_match_ownership_map": sizes_match,
        "coverage_identity_holds": bool(
            assigned_surfel_count == count and unassigned_surfel_count == 0 and multiply_owned_surfel_count == 0
            and in_range == count and sizes_match
            and (int(sizes.sum()) if int(sizes.shape[0]) > 0 else 0) == count
        ),
        "subset_count": partition.subset_count,
        "subset_size": size_stats,
        "largest_subset_size": int(sizes[0]) if int(sizes.shape[0]) > 0 else 0,
        "largest_subset_surfel_fraction": largest_fraction,
        "singleton_subset_count": singleton,
        "singleton_subset_fraction": singleton / subset_count,
        "very_small_subset_fraction": very_small / subset_count,
        "initial_region_count": partition.initial_region_count,
        "final_region_count": partition.subset_count,
        "round_count": partition.round_count,
        "total_interfaces_evaluated": len(partition.attributed_interfaces),
        "interfaces_accepted": len(accepted),
        "interfaces_rejected": len(rejected),
        "rejection_reason_counts": reason_counts,
        "merges_applied": len(partition.merge_provenance),
        "max_initial_regions_merged_into_one_final_region": max_fragments_merged,
        "support_by_region_size": list(partition.support_by_region_size),
        "partition_parameters": partition.config.payload(),
    }
