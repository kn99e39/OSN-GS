from __future__ import annotations

"""Worklog 100 -- region-conditioned, bilateral interface Surfel Region merge.

Worklog 99 kept Worklog 97's over-segmented, percolation-safe regions as an
initialization and merged two regions only when their COMPLETE shared
interface, in aggregate, looked like one smooth surface. On the real scene it
was still too permissive: the initial giant region was ~20.62% but iterative
merging grew a patio+grass+hedge component to 53.86%. This module changes
EXACTLY ONE thing: the per-edge smooth-surface certificate the interface
aggregate is built from. Everything else -- initialization, candidate graph,
support/extent floors, positional-continuity threshold, majority fraction --
is REUSED VERBATIM from Worklog 99 (see `BilateralInterfaceMergeConfig`).

Two changes to the certificate itself, both semantic, not a new threshold:

1. REGION-CONDITIONED local models. Worklog 98's shape operator `S_i` was
   fit from surfel `i`'s full kNN neighbourhood, including neighbours across
   whatever interface is currently being tested -- exactly the kind of
   cross-boundary contamination Worklog 98's own docstring identifies as the
   reason a NAIVE max-of-both-directions residual over-fragments near a
   crease. For a REGION MERGE decision (a much stronger claim than "cut this
   one edge") that contamination is not acceptable: `S_i` must be fit using
   ONLY `i`'s neighbours that currently belong to `i`'s OWN region. Because
   region membership changes every round as merges are applied, these models
   are refit EVERY ROUND, and only for the surfels that actually sit on a
   current cross-region interface (the boundary node set) -- not the whole
   scene, and never carried over stale from an earlier round.

   A boundary surfel with fewer than 2 same-region neighbours cannot support
   ANY 2x2 linear model (a structural fact, not a tunable minimum) and is
   marked UNSUPPORTED. An unsupported direction is never treated as smooth --
   see (2).

2. BILATERAL agreement. Worklog 98 combined the two directional residuals
   with `min(r_i->j, r_j->i)` -- deliberately permissive, because it was
   deciding whether to CUT one edge and either side explaining the
   transition was enough reason not to. A region MERGE is the opposite kind
   of claim ("these two regions are one surface"), so one side alone must
   never be sufficient. An edge is `bilaterally_smooth` only when BOTH
   `r_A->B` (region A's own-region-conditioned model predicting the step
   toward B) AND `r_B->A` (region B's own model predicting the step toward
   A) independently pass the (reused) residual threshold, both directions
   are SUPPORTED (see (1)), and the (reused) positional-continuity test also
   passes. `bilateral_smooth_fraction` over the whole interface replaces
   Worklog 99's `fraction_smooth_continuation` in the majority-vote merge
   gate -- the majority THRESHOLD itself (0.5) is unchanged.

SUBSET OWNERSHIP != TRUSTABILITY still holds; nothing here filters, weights,
or scores trust. Coverage remains unconditional.
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
from osn_gs.surface.torch_discontinuity_first_surfel_partition import (
    _predicted_delta_n_t,
    _tangent_plane_components,
    _knn,
    _auto_chunk_size,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# A 2x2 linear model needs at least 2 (independent) data points -- with
# fewer same-region neighbours the regression is rank-deficient BY
# CONSTRUCTION regardless of any ridge regularization. This is a structural
# fact about fitting a 2x2 linear map, not a tunable/swept constant.
_MIN_SAME_REGION_SUPPORT = 2


@dataclass(frozen=True)
class BilateralInterfaceMergeConfig:
    """Every threshold here is REUSED VERBATIM from Worklog 99
    (`InterfaceCoherentMergeConfig`) -- see the module docstring. This batch
    introduces NO new free parameter: the only change is which per-edge
    certificate `interface_smooth_majority_fraction` is applied to."""

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    region: RegionCoherenceConfig = field(
        default_factory=lambda: RegionCoherenceConfig(require_positional_continuity=True)
    )

    shape_operator_neighbor_count: int = 0  # 0 => local.neighbor_count
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0  # reused verbatim from Worklog 98/99
    parallel_sheet_normal_over_tangent_ratio: float = 1.0  # reused verbatim from Worklog 98/99
    interface_smooth_majority_fraction: float = 0.5  # reused verbatim from Worklog 99, NOT swept here

    def min_unique_surfels_per_interface_side(self) -> int:
        return int(self.local.neighbor_count)

    def min_interface_extent_in_spacing_units(self) -> float:
        return float(self.local.spatial_connect_spacing_multiplier)

    def resolved_shape_operator_neighbor_count(self) -> int:
        return int(self.shape_operator_neighbor_count) or int(self.local.neighbor_count)

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
            "min_unique_surfels_per_interface_side_derivation": "local.neighbor_count",
            "min_interface_extent_in_spacing_units": self.min_interface_extent_in_spacing_units(),
            "min_interface_extent_in_spacing_units_derivation": "local.spatial_connect_spacing_multiplier",
            "min_same_region_support_for_shape_operator": _MIN_SAME_REGION_SUPPORT,
            "min_same_region_support_derivation": "structural minimum for any 2x2 linear fit, not tunable",
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class RegionInterface:
    """One shared interface between two INITIAL (Worklog-97) regions, fully
    aggregated. `region_a`/`region_b` are the ORIGINAL Worklog 97 subset ids
    (`region_a < region_b`)."""

    region_a: int
    region_b: int
    edge_count: int
    unique_surfel_count_a: int
    unique_surfel_count_b: int
    extent_in_spacing_units: float
    smooth_a_to_b_fraction: float
    smooth_b_to_a_fraction: float
    bilateral_smooth_fraction: float
    unsupported_a_to_b_count: int
    unsupported_b_to_a_count: int
    mean_normal_offset_ratio: float
    accepted: bool


@dataclass(frozen=True)
class BilateralInterfacePartition:
    """Result of one region-conditioned bilateral interface merge run."""

    subset_ids: Any
    subset_count: int
    subset_sizes: Any

    initial_region_ids: Any
    initial_region_count: int
    final_region_of_initial: Any
    round_count: int
    merge_provenance: tuple[dict[str, Any], ...]
    all_interfaces: tuple[RegionInterface, ...]

    graph: CandidateGraph
    gaussian_ids: Any
    initial_partition: RegionCoherentPartition
    config: BilateralInterfaceMergeConfig

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


def _fit_region_conditioned_shape_operators(
    positions: Any, normals: Any, tangent_u: Any, tangent_v: Any,
    query_index: Any, neighbor_index: Any, valid_mask: Any, ridge: float,
) -> tuple[Any, Any]:
    """Region-conditioned variant of
    `torch_discontinuity_first_surfel_partition._fit_shape_operators`: fits
    one 2x2 shape operator per QUERY node (a subset of the full scene,
    `query_index`), using only the neighbours flagged valid in `valid_mask`
    (same current region as the query node). Invalid neighbours are given
    ZERO WEIGHT (their displacement/normal-change rows are zeroed before the
    normal-equations reduction), not merely down-weighted -- so a
    cross-region neighbour contributes nothing to the fit at all, closing
    exactly the contamination gap the module docstring describes.

    Returns `(shape_operator (M,2,2), support_count (M,))`. `support_count`
    is the number of same-region neighbours actually used; the caller is
    responsible for treating `support_count < _MIN_SAME_REGION_SUPPORT` as
    UNSUPPORTED rather than trusting the (potentially degenerate-but-solved)
    operator.
    """

    torch = require_torch()
    query_positions = positions[query_index]
    query_normals = normals[query_index]
    query_tangent_u = tangent_u[query_index]
    query_tangent_v = tangent_v[query_index]

    neighbor_positions = positions[neighbor_index]
    neighbor_normals = normals[neighbor_index]

    delta_x = neighbor_positions - query_positions.unsqueeze(1)
    tu = query_tangent_u.unsqueeze(1)
    tv = query_tangent_v.unsqueeze(1)
    delta_x_t = _tangent_plane_components(delta_x, tu, tv)

    query_normal = query_normals.unsqueeze(1)
    sign = torch.where((neighbor_normals * query_normal).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
    aligned_neighbor_normal = neighbor_normals * sign
    delta_n = aligned_neighbor_normal - query_normal
    delta_n_t = _tangent_plane_components(delta_n, tu, tv)

    weight = valid_mask.to(delta_x_t.dtype).unsqueeze(-1)  # (M,k,1), exactly {0,1}
    masked_delta_x_t = delta_x_t * weight
    masked_delta_n_t = delta_n_t * weight

    count = int(query_index.shape[0])
    xtx = torch.einsum("mki,mkj->mij", masked_delta_x_t, masked_delta_x_t)
    xty = torch.einsum("mki,mkj->mij", masked_delta_x_t, masked_delta_n_t)
    identity = torch.eye(2, dtype=xtx.dtype, device=xtx.device).expand(count, 2, 2)
    neg_s_transpose = torch.linalg.solve(xtx + ridge * identity, xty)
    shape_operator = -neg_s_transpose.transpose(-1, -2)

    support_count = valid_mask.sum(dim=1)
    return shape_operator, support_count


def _region_conditioned_bilateral_residuals(
    orientation: SurfaceOrientationEvidence,
    full_neighbor_index: Any,
    node_root: Any,
    edge_left: Any,
    edge_right: Any,
    config: BilateralInterfaceMergeConfig,
) -> dict[str, Any]:
    """Region-conditioned shape operators (fit only from same-`node_root`
    neighbours) and the two DIRECTIONAL residuals for the given edge set,
    WITHOUT yet applying any threshold. Shared by both the one-time global
    threshold computation and the per-round edge evidence -- so the same
    fitting/residual code path is never duplicated.
    """

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    tangent_u = orientation.tangent_axis_u
    tangent_v = orientation.tangent_axis_v
    device = positions.device

    boundary_nodes = torch.unique(torch.cat([edge_left, edge_right]))
    neighbor_index_boundary = full_neighbor_index[boundary_nodes]
    same_region_mask = node_root[neighbor_index_boundary] == node_root[boundary_nodes].unsqueeze(1)
    shape_operator_boundary, support_count_boundary = _fit_region_conditioned_shape_operators(
        positions, normals, tangent_u, tangent_v,
        boundary_nodes, neighbor_index_boundary, same_region_mask, float(config.shape_operator_ridge),
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
    # r_left_own_model: how well LEFT's own-region-conditioned model predicts
    # the transition toward RIGHT (== "r_A->B" when left is on side A).
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
        "normal_offset_ratio": normal_offset_ratio,
        "boundary_node_count": int(boundary_nodes.shape[0]),
    }


def _compute_initial_residual_threshold(
    orientation: SurfaceOrientationEvidence,
    full_neighbor_index: Any,
    node_root: Any,
    graph: CandidateGraph,
    config: BilateralInterfaceMergeConfig,
) -> float:
    """The ONE global residual threshold, computed ONCE (never recomputed
    per round) -- same median + k*MAD formula as Worklog 98/99, but the
    population it is taken over is EVERY spatial edge in the graph (not just
    cross-region interface edges), each evaluated with its own endpoints'
    REGION-CONDITIONED model at the INITIAL (round-1) region membership.

    This is necessary, not merely a stylistic choice: an interface can be
    perfectly REGULAR (e.g. a flat crease meeting at a constant angle
    everywhere), giving a residual population with ZERO variance. A
    threshold computed ONLY from that population degenerates (median + k*MAD
    == the residual value itself, so nothing is ever classified as an
    outlier, no matter how large the residual is in absolute terms) --
    discovered via a focused test (a uniform 90-degree crease was classified
    as 100% bilaterally smooth). Pooling in the SAME-region internal edges
    (whose region-conditioned residual is near zero by construction, since
    that is exactly the population each shape operator was fit from) gives
    the median + MAD statistic genuine contrast to detect against, exactly
    as Worklog 98/99's own scene-wide population did.
    """

    torch = require_torch()
    spatial_edges = graph.candidate_edges[graph.spatial_edge_mask]
    if int(spatial_edges.shape[0]) == 0:
        return 0.0
    evidence = _region_conditioned_bilateral_residuals(
        orientation, full_neighbor_index, node_root, spatial_edges[:, 0], spatial_edges[:, 1], config
    )
    supported_left = evidence["r_left_own_model"][~evidence["unsupported_left"]]
    supported_right = evidence["r_right_own_model"][~evidence["unsupported_right"]]
    pooled_residual = torch.cat([supported_left, supported_right])
    if int(pooled_residual.shape[0]) == 0:
        return 0.0
    median_residual = torch.median(pooled_residual)
    mad = torch.median((pooled_residual - median_residual).abs())
    return float(median_residual + config.residual_mad_multiplier * 1.4826 * mad)


def _compute_bilateral_edge_evidence(
    orientation: SurfaceOrientationEvidence,
    full_neighbor_index: Any,
    node_root: Any,
    cross_left: Any,
    cross_right: Any,
    config: BilateralInterfaceMergeConfig,
    residual_threshold: float,
) -> dict[str, Any]:
    """Region-conditioned, bilateral per-edge evidence for exactly the given
    cross-region edge set, under the given (round-specific) `node_root`
    membership snapshot, classified against the ONE global `residual_threshold`
    (see `_compute_initial_residual_threshold`). Reused identically by the
    main round loop AND by the Worklog 99 lineage trace (§5).
    """

    residuals = _region_conditioned_bilateral_residuals(
        orientation, full_neighbor_index, node_root, cross_left, cross_right, config
    )
    smooth_left_to_right = (~residuals["unsupported_left"]) & (residuals["r_left_own_model"] <= residual_threshold)
    smooth_right_to_left = (~residuals["unsupported_right"]) & (residuals["r_right_own_model"] <= residual_threshold)
    positional_ok = residuals["normal_offset_ratio"] <= config.parallel_sheet_normal_over_tangent_ratio
    bilateral_smooth = smooth_left_to_right & smooth_right_to_left & positional_ok

    return {
        "smooth_left_to_right": smooth_left_to_right,
        "smooth_right_to_left": smooth_right_to_left,
        "bilateral_smooth": bilateral_smooth,
        "unsupported_left": residuals["unsupported_left"],
        "unsupported_right": residuals["unsupported_right"],
        "normal_offset_ratio": residuals["normal_offset_ratio"],
        "residual_threshold": residual_threshold,
        "boundary_node_count": residuals["boundary_node_count"],
    }


class _RegionUnionFind:
    """Identical to Worklog 99's `_RegionUnionFind` -- pure sequential DSU
    over INITIAL region ids, deterministic smaller-index-survives union."""

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


def partition_surfels_bilateral_interface(
    orientation: SurfaceOrientationEvidence,
    config: BilateralInterfaceMergeConfig | None = None,
    *,
    max_rounds: int = 64,
    progress: Callable[[str], None] | None = None,
) -> BilateralInterfacePartition:
    """Partition EVERY surfel via Worklog 97-with-positional-gate
    initialization, then iteratively merge two regions only when their
    complete shared interface is BROADLY SUPPORTED and BILATERALLY smooth
    (region-conditioned models on both sides independently agree). See the
    module docstring for the full contract.
    """

    torch = require_torch()
    config = config or BilateralInterfaceMergeConfig()
    if config.region.local != config.local:
        raise ValueError(
            "BilateralInterfaceMergeConfig.region.local must match config.local -- the candidate graph is "
            "shared verbatim between region initialization and interface evaluation."
        )
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)
    initial_partition = partition_surfels_region_coherent(orientation, config.region, progress=progress)
    initial_region_ids = initial_partition.subset_ids
    initial_region_count = initial_partition.subset_count

    if count == 0 or initial_region_count <= 1:
        return BilateralInterfacePartition(
            subset_ids=initial_region_ids,
            subset_count=initial_region_count,
            subset_sizes=initial_partition.subset_sizes,
            initial_region_ids=initial_region_ids,
            initial_region_count=initial_region_count,
            final_region_of_initial=torch.arange(max(initial_region_count, 0), dtype=torch.int64, device=device),
            round_count=0,
            merge_provenance=(),
            all_interfaces=(),
            graph=graph,
            gaussian_ids=orientation.gaussian_ids,
            initial_partition=initial_partition,
            config=config,
        )

    spatial_mask = graph.spatial_edge_mask
    candidate_left = graph.candidate_edges[:, 0]
    candidate_right = graph.candidate_edges[:, 1]
    local_spacing = graph.local_spacing
    edge_spacing = (local_spacing[candidate_left] + local_spacing[candidate_right]) / 2.0
    positions_left = positions[candidate_left]
    positions_right = positions[candidate_right]

    k_shape = min(config.resolved_shape_operator_neighbor_count(), max(count - 1, 1))
    chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
    if progress is not None:
        progress(f"precomputing static kNN for region-conditioned shape operators: k={k_shape}")
    full_neighbor_index, _ = _knn(positions, k_shape, chunk_size, progress)

    # ONE global threshold, computed once from the INITIAL (round-1) region
    # membership over EVERY spatial edge (not just cross-region interface
    # edges) -- see `_compute_initial_residual_threshold` for why this is
    # required rather than recomputing per round. Reused verbatim for every
    # round's classification, exactly mirroring Worklog 98/99's own
    # compute-once precedent.
    residual_threshold = _compute_initial_residual_threshold(
        orientation, full_neighbor_index, initial_region_ids, graph, config
    )
    if progress is not None:
        progress(f"global region-conditioned residual threshold (computed once): {residual_threshold:.6f}")

    dsu = _RegionUnionFind(initial_region_count)
    merge_provenance: list[dict[str, Any]] = []
    all_interfaces: list[RegionInterface] = []

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

        evidence = _compute_bilateral_edge_evidence(
            orientation, full_neighbor_index, node_root, cross_left, cross_right, config, residual_threshold
        )
        if progress is not None:
            progress(
                f"[round {round_index}] region-conditioned fit over {evidence['boundary_node_count']} boundary nodes"
            )

        left_is_a = cross_root_left < cross_root_right
        side_a_node = torch.where(left_is_a, cross_left, cross_right)
        side_b_node = torch.where(left_is_a, cross_right, cross_left)
        # bilateral evidence is symmetric in meaning but stored per (left,
        # right) as computed -- re-express as (a,b) directional fractions.
        smooth_a_to_b_edge = torch.where(left_is_a, evidence["smooth_left_to_right"], evidence["smooth_right_to_left"])
        smooth_b_to_a_edge = torch.where(left_is_a, evidence["smooth_right_to_left"], evidence["smooth_left_to_right"])
        unsupported_a_edge = torch.where(left_is_a, evidence["unsupported_left"], evidence["unsupported_right"])
        unsupported_b_edge = torch.where(left_is_a, evidence["unsupported_right"], evidence["unsupported_left"])
        bilateral_smooth_edge = evidence["bilateral_smooth"]
        cross_ratio = evidence["normal_offset_ratio"]
        cross_spacing = edge_spacing[cross_mask].clamp_min(_EPS)
        cross_mid = (positions_left[cross_mask] + positions_right[cross_mask]) / 2.0

        pair_low = torch.minimum(root_left, root_right)[cross_mask]
        pair_high = torch.maximum(root_left, root_right)[cross_mask]
        pair_key = pair_low * int(initial_region_count) + pair_high
        unique_keys, group_id = torch.unique(pair_key, return_inverse=True)
        group_count = int(unique_keys.shape[0])

        edge_count_per_group = torch.bincount(group_id, minlength=group_count)
        smooth_a_sum = torch.bincount(group_id, weights=smooth_a_to_b_edge.to(torch.float32), minlength=group_count)
        smooth_b_sum = torch.bincount(group_id, weights=smooth_b_to_a_edge.to(torch.float32), minlength=group_count)
        bilateral_sum = torch.bincount(group_id, weights=bilateral_smooth_edge.to(torch.float32), minlength=group_count)
        unsupported_a_sum = torch.bincount(group_id, weights=unsupported_a_edge.to(torch.float32), minlength=group_count)
        unsupported_b_sum = torch.bincount(group_id, weights=unsupported_b_edge.to(torch.float32), minlength=group_count)
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

        smooth_a_to_b_fraction = smooth_a_sum / edge_count_per_group.clamp_min(1)
        smooth_b_to_a_fraction = smooth_b_sum / edge_count_per_group.clamp_min(1)
        bilateral_smooth_fraction = bilateral_sum / edge_count_per_group.clamp_min(1)
        mean_ratio = ratio_sum_per_group / edge_count_per_group.clamp_min(1)

        accept = (
            (unique_count_a_per_group >= min_side)
            & (unique_count_b_per_group >= min_side)
            & (extent_in_spacing_units >= min_extent)
            & (bilateral_smooth_fraction >= majority)
            & (mean_ratio <= config.parallel_sheet_normal_over_tangent_ratio)
        )

        group_region_a = torch.div(unique_keys, int(initial_region_count), rounding_mode="floor")
        group_region_b = unique_keys - group_region_a * int(initial_region_count)

        for group in range(group_count):
            all_interfaces.append(
                RegionInterface(
                    region_a=int(group_region_a[group].item()),
                    region_b=int(group_region_b[group].item()),
                    edge_count=int(edge_count_per_group[group].item()),
                    unique_surfel_count_a=int(unique_count_a_per_group[group].item()),
                    unique_surfel_count_b=int(unique_count_b_per_group[group].item()),
                    extent_in_spacing_units=float(extent_in_spacing_units[group].item()),
                    smooth_a_to_b_fraction=float(smooth_a_to_b_fraction[group].item()),
                    smooth_b_to_a_fraction=float(smooth_b_to_a_fraction[group].item()),
                    bilateral_smooth_fraction=float(bilateral_smooth_fraction[group].item()),
                    unsupported_a_to_b_count=int(unsupported_a_sum[group].item()),
                    unsupported_b_to_a_count=int(unsupported_b_sum[group].item()),
                    mean_normal_offset_ratio=float(mean_ratio[group].item()),
                    accepted=bool(accept[group].item()),
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
                    "round": round_index,
                    "region_a": region_a,
                    "region_b": region_b,
                    "edge_count": int(edge_count_per_group[group].item()),
                    "unique_surfel_count_a": int(unique_count_a_per_group[group].item()),
                    "unique_surfel_count_b": int(unique_count_b_per_group[group].item()),
                    "extent_in_spacing_units": float(extent_in_spacing_units[group].item()),
                    "smooth_a_to_b_fraction": float(smooth_a_to_b_fraction[group].item()),
                    "smooth_b_to_a_fraction": float(smooth_b_to_a_fraction[group].item()),
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
        raise RuntimeError("Bilateral interface region merge: did not converge within max_rounds.")

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

    return BilateralInterfacePartition(
        subset_ids=subset_ids,
        subset_count=int(order_by_size.shape[0]),
        subset_sizes=subset_sizes,
        initial_region_ids=initial_region_ids,
        initial_region_count=initial_region_count,
        final_region_of_initial=final_region_of_initial,
        round_count=round_index,
        merge_provenance=tuple(merge_provenance),
        all_interfaces=tuple(all_interfaces),
        graph=graph,
        gaussian_ids=orientation.gaussian_ids,
        initial_partition=initial_partition,
        config=config,
    )


def bilateral_interface_accounting(partition: BilateralInterfacePartition) -> dict[str, Any]:
    """Full accounting block, matching the prior worklogs' own field
    vocabulary. Coverage identity fields are named explicitly (Worklog 99's
    doc had a copy-paste typo in the prose, not the code -- the code here and
    in Worklog 99 both correctly compute `assigned_surfel_count == count`,
    `unassigned_surfel_count == 0`, `multiply_owned_surfel_count == 0`)."""

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
    multiply_owned_surfel_count = 0  # structurally impossible: subset_ids is single-valued per surfel

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

    accepted_interfaces = [interface for interface in partition.all_interfaces if interface.accepted]
    rejected_interfaces = [interface for interface in partition.all_interfaces if not interface.accepted]

    final_of_initial = partition.final_region_of_initial
    if int(final_of_initial.shape[0]) > 0:
        _, _, fragment_counts = torch.unique(final_of_initial, return_inverse=True, return_counts=True)
        fragments_per_final_region = fragment_counts.tolist()
        max_fragments_merged = int(fragment_counts.max().item())
    else:
        fragments_per_final_region = []
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
        "very_small_subset_size_threshold": VERY_SMALL_SUBSET_SIZE,
        "very_small_subset_count": very_small,
        "very_small_subset_fraction": very_small / subset_count,
        "initial_region_count": partition.initial_region_count,
        "final_region_count": partition.subset_count,
        "round_count": partition.round_count,
        "total_interfaces_evaluated": len(partition.all_interfaces),
        "interfaces_accepted": len(accepted_interfaces),
        "interfaces_rejected": len(rejected_interfaces),
        "merges_applied": len(partition.merge_provenance),
        "max_initial_regions_merged_into_one_final_region": max_fragments_merged,
        "final_region_fragment_counts": fragments_per_final_region,
        "merge_provenance": list(partition.merge_provenance),
        "partition_parameters": partition.config.payload(),
    }
