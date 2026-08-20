from __future__ import annotations

"""Worklog 99 -- interface-coherent Surfel Region merge.

Worklog 97 (region-level orientation concentration) prevents dense-graph
percolation but over-segments genuinely smooth curved surfaces, because a
surface's own normal ROTATION erodes concentration even when nothing
discontinuous is happening. Worklog 98 (local shape-operator residual) models
smooth curvature correctly but, applied as a per-EDGE cut rule, does not
reliably disconnect a dense kNN graph: 18.15% of edges cut was nowhere near
enough, and the result was a 94.51% giant subset -- worse than either prior
worklog. Neither module alone is the final partition; this one replaces BOTH
final union rules with a two-stage pipeline that keeps what each got right:

    trained 2DGS surfels
        -> intrinsic tangent frame / t_w normal (UNCHANGED)
        -> Worklog 97 region-coherent partition, used ONLY as a
           deliberately OVER-SEGMENTED, percolation-safe initialization
           (its own final semantics are NOT reused)
        -> REGION ADJACENCY GRAPH: every pair of initial regions connected
           by at least one local candidate edge
        -> for every adjacent pair, the COMPLETE shared INTERFACE (every
           candidate edge crossing between them, not just the best one)
        -> Worklog 98's local shape-operator residual and positional
           continuity, aggregated ACROSS the whole interface (never from one
           edge in isolation)
        -> a region-level merge decision requiring BROAD, SUPPORTED, smooth
           continuation across the interface
        -> deterministic, iterative region merging
        -> final Coverage-first Surfel Subsets

The central intent (architecture directive): two regions merge only when
their SHARED INTERFACE, as a whole, provides sustained evidence that both
are part of one smooth curved surface. A single accidental smooth bridge
edge, or a long chain of narrow bridges, must never be sufficient -- that
would silently recreate Worklog 96/98-style single-linkage percolation. A
smoothly, broadly curved shared face must not be blocked just because the
normal direction changes substantially across it, or Worklog 97's own
over-segmentation problem returns.

Coverage remains unconditional and non-negotiable: merging changes region
ownership IDs only. Nothing here filters, weights, or scores trustability --
SUBSET OWNERSHIP != TRUSTABILITY still holds (see
`torch_coverage_first_subset_partition`'s module docstring).
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
    _fit_shape_operators,
    _predicted_delta_n_t,
    _tangent_plane_components,
    _knn,
    _auto_chunk_size,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


@dataclass(frozen=True)
class InterfaceCoherentMergeConfig:
    """Every heuristic this module adds, centralized.

    `local` (the candidate graph) and `region` (Worklog 97's own config, used
    only to build the initial over-segmented regions) are both REUSED
    verbatim. The shape-operator fit and its two thresholds
    (`residual_mad_multiplier`, `parallel_sheet_normal_over_tangent_ratio`)
    are also reused verbatim from Worklog 98 -- this module does not refit or
    re-tune the per-edge differential evidence, only how it is AGGREGATED and
    what decision is made from the aggregate.

    Exactly ONE genuinely new free parameter is introduced:
    `interface_smooth_majority_fraction`. Everything else that gates a merge
    is either a REUSED threshold or a FLOOR ALGEBRAICALLY DERIVED from an
    existing constant (see `min_unique_surfels_per_interface_side` and
    `min_interface_extent_in_spacing_units` below) -- neither is swept
    against this scene's visualization.
    """

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()
    # `require_positional_continuity=True`: Worklog 97's own candidate
    # acceptance has no positional test, so two nearby PARALLEL sheets with
    # identical normals would otherwise be fused into a single INITIAL
    # region before this module's region-merge step ever runs -- and a
    # merge-only algorithm can never split a region back apart. Turning this
    # on for region initialization (discovered necessary empirically, see
    # Worklog 99) closes that gap without changing Worklog 97's own default,
    # standalone behavior (still `False` there).
    region: RegionCoherenceConfig = field(
        default_factory=lambda: RegionCoherenceConfig(require_positional_continuity=True)
    )

    shape_operator_neighbor_count: int = 0  # 0 => local.neighbor_count
    shape_operator_ridge: float = 1e-8
    residual_mad_multiplier: float = 3.0  # reused verbatim from Worklog 98
    parallel_sheet_normal_over_tangent_ratio: float = 1.0  # reused verbatim from Worklog 98

    # The ONE new parameter: an interface is curvature-consistent only when a
    # MAJORITY of its own edges independently classify as smooth continuation
    # under Worklog 98's per-edge test (same global residual/ratio thresholds
    # as above, computed once over the whole scene). 0.5 is the plain
    # majority-vote value -- not swept against this scene's visualization.
    # This is deliberately an aggregate over the WHOLE interface, never the
    # single best (or single worst) edge in it (architecture directive
    # sections 5-6: "do NOT use the minimum residual over the interface").
    interface_smooth_majority_fraction: float = 0.5

    # Support/extent floors below are NOT new independent constants: they are
    # read directly from the SAME local-graph parameters that already define
    # "one local neighbourhood" and "one local candidate edge", so that a
    # genuine bridge -- reaching no farther than a single node's own kNN
    # neighbourhood, or touching no more surfels than that neighbourhood
    # contains -- can never pass regardless of how smooth it looks.
    def min_unique_surfels_per_interface_side(self) -> int:
        """An interface must touch MORE surfels, on its narrower side, than
        a single node's own local neighbourhood (`local.neighbor_count`) --
        otherwise it is at most one node's own local reach, not a genuinely
        shared face between two regions."""

        return int(self.local.neighbor_count)

    def min_interface_extent_in_spacing_units(self) -> float:
        """An interface's own spatial extent (bounding diagonal of its
        member surfels, normalized by their own local spacing) must exceed
        the SAME spacing multiplier (`local.spatial_connect_spacing_multiplier`)
        that already defines how far a single candidate edge is allowed to
        reach -- otherwise the interface is no broader than one edge's own
        connectivity radius, i.e. exactly the "isolated / sparse graph
        bridge" case the architecture directive requires rejecting."""

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
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class RegionInterface:
    """One shared interface between two INITIAL (Worklog-97) regions, fully
    aggregated -- never reduced to a single edge. `region_a`/`region_b` are
    the ORIGINAL Worklog 97 subset ids (`region_a < region_b`, fixed for the
    life of the run so provenance stays traceable back to the initial
    over-segmented partition, independent of later DSU roots)."""

    region_a: int
    region_b: int
    edge_count: int
    unique_surfel_count_a: int
    unique_surfel_count_b: int
    extent_in_spacing_units: float
    fraction_smooth_continuation: float
    mean_residual: float
    max_residual: float
    mean_normal_offset_ratio: float
    accepted: bool


@dataclass(frozen=True)
class InterfaceCoherentPartition:
    """Result of one interface-coherent region-merge run."""

    subset_ids: Any  # (N,) int64 -- exactly one owner per surfel
    subset_count: int
    subset_sizes: Any  # (subset_count,) int64, descending

    initial_region_ids: Any  # (N,) int64 -- Worklog 97 result, BEFORE any merge
    initial_region_count: int
    final_region_of_initial: Any  # (initial_region_count,) int64 -- initial region -> final DSU root
    round_count: int
    merge_provenance: tuple[dict[str, Any], ...]  # every ACCEPTED merge, in application order
    all_interfaces: tuple[RegionInterface, ...]  # every interface ever evaluated, across all rounds

    graph: CandidateGraph
    gaussian_ids: Any
    initial_partition: RegionCoherentPartition
    config: InterfaceCoherentMergeConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def cross_region_edges_mask(self) -> Any:
        """Spatial candidate edges whose endpoints started in DIFFERENT
        Worklog 97 regions -- the full population any interface is drawn
        from (accepted or rejected)."""

        torch = require_torch()
        left, right = self.graph.candidate_edges[:, 0], self.graph.candidate_edges[:, 1]
        return self.graph.spatial_edge_mask & (self.initial_region_ids[left] != self.initial_region_ids[right])

    @property
    def accepted_merge_edges_mask(self) -> Any:
        """Cross-region spatial edges whose two endpoints ended up in the
        SAME final subset -- i.e. edges belonging to an accepted merge."""

        torch = require_torch()
        left, right = self.graph.candidate_edges[:, 0], self.graph.candidate_edges[:, 1]
        return self.cross_region_edges_mask & (self.subset_ids[left] == self.subset_ids[right])

    @property
    def rejected_interface_edges_mask(self) -> Any:
        """Cross-region spatial edges whose two endpoints ended up in
        DIFFERENT final subsets -- the final boundary the review export
        visualizes."""

        torch = require_torch()
        return self.cross_region_edges_mask & ~self.accepted_merge_edges_mask


def _compute_discontinuity_evidence(
    orientation: SurfaceOrientationEvidence,
    graph: CandidateGraph,
    config: InterfaceCoherentMergeConfig,
    progress: Callable[[str], None] | None,
) -> tuple[Any, Any, float]:
    """Per SPATIAL candidate edge: Worklog 98's smooth-surface residual and
    positional normal-offset ratio. Deliberately mirrors
    ``torch_discontinuity_first_surfel_partition.partition_surfels_discontinuity_first``'s
    own per-edge computation exactly (same shape-operator fit, same MIN
    combination, same self-normalizing ratio) via the SAME imported helper
    functions, so the differential evidence itself is not redefined here --
    only how it is later aggregated across an interface differs. Returns
    ``(edge_residual, edge_normal_offset_ratio, residual_threshold)``, each
    tensor indexed over ALL of `graph.candidate_edges` (zero where not a
    spatial edge).
    """

    torch = require_torch()
    positions = orientation.positions
    normals = orientation.surface_normal
    tangent_u = orientation.tangent_axis_u
    tangent_v = orientation.tangent_axis_v
    count = int(positions.shape[0])
    device = positions.device

    edge_residual_full = torch.zeros((int(graph.candidate_edges.shape[0]),), dtype=torch.float32, device=device)
    edge_offset_full = torch.zeros_like(edge_residual_full)
    spatial_edges = graph.candidate_edges[graph.spatial_edge_mask]
    spatial_index = torch.nonzero(graph.spatial_edge_mask, as_tuple=False).reshape(-1)
    if int(spatial_edges.shape[0]) == 0 or count == 0:
        return edge_residual_full, edge_offset_full, 0.0

    k_shape = min(config.resolved_shape_operator_neighbor_count(), max(count - 1, 1))
    if progress is not None:
        progress(f"fitting local shape operators: k={k_shape}")
    chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
    neighbor_index, _ = _knn(positions, k_shape, chunk_size, progress)
    shape_operator = _fit_shape_operators(
        positions, normals, tangent_u, tangent_v, neighbor_index, float(config.shape_operator_ridge)
    )

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
    normal_offset = (delta_x * average_normal).sum(dim=-1).abs()
    tangential_offset = (delta_x - normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
    normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)

    median_residual = torch.median(edge_residual)
    mad = torch.median((edge_residual - median_residual).abs())
    robust_sigma = 1.4826 * mad
    residual_threshold = float(median_residual + config.residual_mad_multiplier * robust_sigma)

    edge_residual_full[spatial_index] = edge_residual.to(torch.float32)
    edge_offset_full[spatial_index] = normal_offset_ratio.to(torch.float32)
    return edge_residual_full, edge_offset_full, residual_threshold


class _RegionUnionFind:
    """Plain sequential DSU over INITIAL region ids. Pure Python, mirroring
    Worklog 97's own `_region_coherent_merge_cpu` precedent: a dynamically
    evolving per-root union-find is an inherently sequential fixpoint."""

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
        # Deterministic: smaller original index always survives as root,
        # matching the min-label convention used throughout this codebase.
        if root_a > root_b:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        return root_a


def partition_surfels_interface_coherent(
    orientation: SurfaceOrientationEvidence,
    config: InterfaceCoherentMergeConfig | None = None,
    *,
    max_rounds: int = 64,
    progress: Callable[[str], None] | None = None,
) -> InterfaceCoherentPartition:
    """Partition EVERY surfel by initializing from Worklog 97's
    over-segmented, percolation-safe regions, then iteratively merging any
    two regions whose complete shared interface provides broad, supported,
    curvature-consistent, positionally-continuous evidence of one smooth
    surface. See the module docstring for the full contract.
    """

    torch = require_torch()
    config = config or InterfaceCoherentMergeConfig()
    if config.region.local != config.local:
        raise ValueError(
            "InterfaceCoherentMergeConfig.region.local must match config.local -- the candidate graph is "
            "shared verbatim between region initialization and interface evaluation, never rebuilt with "
            "different parameters."
        )
    positions = orientation.positions
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)
    initial_partition = partition_surfels_region_coherent(orientation, config.region, progress=progress)
    initial_region_ids = initial_partition.subset_ids
    initial_region_count = initial_partition.subset_count

    if count == 0 or initial_region_count <= 1:
        return InterfaceCoherentPartition(
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

    edge_residual, edge_normal_offset_ratio, residual_threshold = _compute_discontinuity_evidence(
        orientation, graph, config, progress
    )
    edge_smooth = (edge_residual <= residual_threshold) & (
        edge_normal_offset_ratio <= config.parallel_sheet_normal_over_tangent_ratio
    )

    spatial_mask = graph.spatial_edge_mask
    candidate_left = graph.candidate_edges[:, 0]
    candidate_right = graph.candidate_edges[:, 1]
    local_spacing = graph.local_spacing
    edge_spacing = (local_spacing[candidate_left] + local_spacing[candidate_right]) / 2.0
    positions_left = positions[candidate_left]
    positions_right = positions[candidate_right]

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

        pair_low = torch.minimum(root_left, root_right)
        pair_high = torch.maximum(root_left, root_right)
        pair_key = pair_low * int(initial_region_count) + pair_high
        cross_key = pair_key[cross_mask]
        unique_keys, group_id = torch.unique(cross_key, return_inverse=True)
        group_count = int(unique_keys.shape[0])

        cross_left = candidate_left[cross_mask]
        cross_right = candidate_right[cross_mask]
        cross_root_left = root_left[cross_mask]
        cross_root_right = root_right[cross_mask]
        # side "a" is always the numerically smaller current root (matches
        # pair_low/pair_high above), so side-a/side-b membership is a pure
        # function of root identity, not of candidate-edge storage order.
        left_is_a = cross_root_left < cross_root_right
        side_a_node = torch.where(left_is_a, cross_left, cross_right)
        side_b_node = torch.where(left_is_a, cross_right, cross_left)

        cross_residual = edge_residual[cross_mask]
        cross_ratio = edge_normal_offset_ratio[cross_mask]
        cross_smooth = edge_smooth[cross_mask].to(torch.float32)
        cross_spacing = edge_spacing[cross_mask].clamp_min(_EPS)
        cross_mid = (positions_left[cross_mask] + positions_right[cross_mask]) / 2.0

        edge_count_per_group = torch.bincount(group_id, minlength=group_count)
        smooth_sum_per_group = torch.bincount(group_id, weights=cross_smooth, minlength=group_count)
        residual_sum_per_group = torch.bincount(group_id, weights=cross_residual, minlength=group_count)
        ratio_sum_per_group = torch.bincount(group_id, weights=cross_ratio, minlength=group_count)
        max_residual_per_group = torch.full((group_count,), -1.0, dtype=torch.float32, device=device)
        max_residual_per_group.scatter_reduce_(0, group_id, cross_residual, reduce="amax", include_self=True)

        # unique surfel counts per side, per group: dedup via a combined key.
        side_a_key = group_id.to(torch.int64) * int(count) + side_a_node
        side_b_key = group_id.to(torch.int64) * int(count) + side_b_node
        unique_a = torch.unique(side_a_key)
        unique_b = torch.unique(side_b_key)
        unique_a_group = torch.div(unique_a, count, rounding_mode="floor")
        unique_b_group = torch.div(unique_b, count, rounding_mode="floor")
        unique_count_a_per_group = torch.bincount(unique_a_group, minlength=group_count)
        unique_count_b_per_group = torch.bincount(unique_b_group, minlength=group_count)

        # spatial extent: bounding-box diagonal of interface edge midpoints,
        # normalized by the mean local spacing of the edges forming it.
        min_xyz = torch.full((group_count, 3), float("inf"), dtype=torch.float32, device=device)
        max_xyz = torch.full((group_count, 3), float("-inf"), dtype=torch.float32, device=device)
        for axis in range(3):
            min_xyz[:, axis].scatter_reduce_(0, group_id, cross_mid[:, axis], reduce="amin", include_self=True)
            max_xyz[:, axis].scatter_reduce_(0, group_id, cross_mid[:, axis], reduce="amax", include_self=True)
        extent_diagonal = (max_xyz - min_xyz).norm(dim=-1)
        mean_spacing_per_group = torch.bincount(group_id, weights=cross_spacing, minlength=group_count) / edge_count_per_group.clamp_min(1)
        extent_in_spacing_units = extent_diagonal / mean_spacing_per_group.clamp_min(_EPS)

        fraction_smooth = smooth_sum_per_group / edge_count_per_group.clamp_min(1)
        mean_residual = residual_sum_per_group / edge_count_per_group.clamp_min(1)
        mean_ratio = ratio_sum_per_group / edge_count_per_group.clamp_min(1)

        accept = (
            (unique_count_a_per_group >= min_side)
            & (unique_count_b_per_group >= min_side)
            & (extent_in_spacing_units >= min_extent)
            & (fraction_smooth >= majority)
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
                    fraction_smooth_continuation=float(fraction_smooth[group].item()),
                    mean_residual=float(mean_residual[group].item()),
                    max_residual=float(max_residual_per_group[group].item()),
                    mean_normal_offset_ratio=float(mean_ratio[group].item()),
                    accepted=bool(accept[group].item()),
                )
            )

        # Deterministic application order within the round: strongest
        # curvature-consistency evidence first, ties broken by ascending
        # (region_a, region_b) -- composed via stable sorts, last-first.
        accepted_index = torch.nonzero(accept, as_tuple=False).reshape(-1)
        if int(accepted_index.shape[0]) == 0:
            round_index -= 1
            break
        order = accepted_index[torch.argsort(group_region_b[accepted_index], stable=True)]
        order = order[torch.argsort(group_region_a[order], stable=True)]
        order = order[torch.argsort(fraction_smooth[order], descending=True, stable=True)]

        applied_this_round = 0
        for group in order.tolist():
            region_a = int(group_region_a[group].item())
            region_b = int(group_region_b[group].item())
            if dsu.find(region_a) == dsu.find(region_b):
                continue  # already merged earlier this round (transitivity)
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
                    "fraction_smooth_continuation": float(fraction_smooth[group].item()),
                    "mean_residual": float(mean_residual[group].item()),
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
        raise RuntimeError("Interface-coherent region merge: did not converge within max_rounds.")

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

    return InterfaceCoherentPartition(
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


def interface_coherent_accounting(partition: InterfaceCoherentPartition) -> dict[str, Any]:
    """Full accounting block, matching the prior worklogs' own field
    vocabulary wherever the same concept applies, plus the interface/merge
    provenance the architecture directive requires (sections 13-15)."""

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

    accepted_interfaces = [interface for interface in partition.all_interfaces if interface.accepted]
    rejected_interfaces = [interface for interface in partition.all_interfaces if not interface.accepted]

    # Worklog-97-fragment recovery: how many INITIAL regions ended up inside
    # each FINAL region.
    final_of_initial = partition.final_region_of_initial
    if int(final_of_initial.shape[0]) > 0:
        fragment_unique, fragment_inverse, fragment_counts = torch.unique(
            final_of_initial, return_inverse=True, return_counts=True
        )
        fragments_per_final_region = fragment_counts.tolist()
        max_fragments_merged = int(fragment_counts.max().item())
    else:
        fragments_per_final_region = []
        max_fragments_merged = 0

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
