from __future__ import annotations

"""Worklog 97 -- region-level anti-chaining Surfel Subset partition.

Worklog 96 showed that intrinsic 2DGS normals do not, by themselves, remove
the giant-component pathology: local pairwise normal compatibility plus plain
connected components allowed **single-linkage chaining** -- a flat patio floor
and a differently-oriented hedge/background wall absorbed into one 894,378-
surfel subset (74.70% of the scene), because every step along the chain was
LOCALLY plausible even though the endpoints were not.

This module keeps everything about Worklog 96 that worked and changes
EXACTLY one thing: the union rule.

    local pairwise normal-compatible edge (Worklog 96, UNCHANGED)
        + REGION-LEVEL surface-orientation coherence (NEW)
        -> final Coverage-first Surfel Subsets

A locally accepted edge is necessary but no longer sufficient to merge two
components: two regions merge only if the region their union would form is
ITSELF still orientation-coherent. This is evaluated on a sign-invariant
scatter-matrix summary of a region's own accumulated normals
(:class:`RegionOrientationState`), never on any single edge in isolation --
"one coherent chain of pairwise-compatible surfels" is explicitly NOT treated
as evidence of "one coherent surface".

Everything else is unchanged from Worklog 96:

* the primitive is the trained 2DGS surfel (``model.get_tangent_u`` /
  ``get_tangent_v`` / ``get_normal`` -- read, never eigen-decomposed);
* the local candidate graph is built by
  :func:`osn_gs.surface.torch_coverage_first_subset_partition.build_candidate_graph`
  (kNN spatial adjacency + local-spacing gate + unsigned normal alignment),
  reused VERBATIM so this module and Worklog 96's plain connected-component
  partition are driven by the identical local evidence -- the only variable
  under comparison is the union rule;
* coverage remains non-negotiable: every input surfel gets exactly one final
  subset. What changes is HOW a surfel without a viable region can still be
  covered -- see the structural-core / ownership-propagated / isolated-
  fallback distinction below (SUBSET OWNERSHIP still != TRUSTABILITY: none of
  these roles is a future surface-evidence trust score, see section 14 of the
  architecture directive this module implements).

Sign contract
-------------
Region orientation state uses the outer product ``n n^T``, and
``n n^T == (-n)(-n)^T`` identically -- the representation is exactly as
insensitive to the normal sign gauge as the local pairwise
``|dot(n_i, n_j)|`` test it extends. No normal is ever flipped.
"""

from dataclasses import dataclass
from typing import Any, Callable

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CandidateGraph,
    CoverageFirstPartitionConfig,
    SurfaceOrientationEvidence,
    VERY_SMALL_SUBSET_SIZE,
    build_candidate_graph,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# --- partition roles: every surfel carries exactly one ---
ROLE_STRUCTURAL_CORE = "structural_core_member"
ROLE_OWNERSHIP_PROPAGATED = "ownership_propagated_member"
ROLE_ISOLATED_FALLBACK = "isolated_fallback_member"

PARTITION_ROLES: tuple[str, ...] = (
    ROLE_STRUCTURAL_CORE,
    ROLE_OWNERSHIP_PROPAGATED,
    ROLE_ISOLATED_FALLBACK,
)


@dataclass(frozen=True)
class RegionCoherenceConfig:
    """Every heuristic this module adds, centralized, with the local-graph
    config it wraps unchanged from Worklog 96 (`local`).

    Only ONE new free parameter exists (`structural_weight`), and it is fixed
    at 1.0 (uniform) rather than tuned -- see the field comment. The
    region-coherence floor itself introduces NO new independent parameter: it
    is derived algebraically from `local.normal_compatibility_min_alignment`
    (see :meth:`concentration_floor`).
    """

    local: CoverageFirstPartitionConfig = CoverageFirstPartitionConfig()

    # Per-surfel weight contributed to its region's orientation scatter
    # M_R = sum_i w_i * n_i n_i^T. The architecture directive explicitly
    # forbids introducing future surface-evidence Trust weights in this
    # batch ("if no stronger pre-existing reason exists, use uniform
    # structural weights for the primary replay") -- this is that uniform
    # weight, not a trust score, and every surfel gets exactly the same value.
    structural_weight: float = 1.0

    # OFF by default so every existing caller and test reproduces the exact
    # original Worklog 97 numbers unchanged. When a caller opts in (Worklog
    # 99's region initialization does), an additional REUSED Worklog 98
    # criterion -- the self-normalizing normal-offset/tangential-offset
    # ratio, no new threshold -- gates local edge acceptance alongside
    # spatial adjacency and normal compatibility. Discovered necessary
    # because Worklog 97's own candidate acceptance has no positional test:
    # two nearby PARALLEL sheets with identical normals pass both of its
    # existing gates and get fused into one region, which a downstream
    # region-MERGE step (Worklog 99) can never undo since merging never
    # splits. This uses only `positions` and `surface_normal`, so it does
    # not change what `SurfaceOrientationEvidence` requires.
    require_positional_continuity: bool = False
    parallel_sheet_normal_over_tangent_ratio: float = 1.0

    def concentration_floor(self) -> float:
        """Region-coherence floor, derived from the EXISTING local alignment
        threshold ``a = local.normal_compatibility_min_alignment`` rather than
        an independently swept angle (architecture directive section 5).

        For two unit normals n_i, n_j with |dot(n_i, n_j)| = a exactly, the
        scatter matrix M = n_i n_i^T + n_j n_j^T (uniform weight) has
        eigenvalues 1+a and 1-a in their common plane (and 0 orthogonal to
        it), trace 2, so its concentration is

            C = lambda_max(M) / trace(M) = (1 + a) / 2 .

        The floor used here is exactly that value: a candidate MERGE is
        accepted only if the two regions' UNION remains at least as
        concentrated as a single already-accepted pairwise edge would be by
        itself. This is not an independent angle -- raising/lowering `a`
        raises/lowers this floor by the same algebraic relationship, so the
        codebase still has exactly one normal-compatibility number to reason
        about. Verified analytically and against a numeric eigensolver in
        `tests/test_region_coherent_surfel_partition.py`.
        """

        a = max(-1.0, min(1.0, self.local.normal_compatibility_min_alignment))
        return (1.0 + a) / 2.0

    def payload(self) -> dict[str, Any]:
        return {
            "local": self.local.payload(),
            "structural_weight": self.structural_weight,
            "concentration_floor": self.concentration_floor(),
            "concentration_floor_derivation": "(1 + local.normal_compatibility_min_alignment) / 2",
            "require_positional_continuity": self.require_positional_continuity,
            "parallel_sheet_normal_over_tangent_ratio": self.parallel_sheet_normal_over_tangent_ratio,
        }


@dataclass(frozen=True)
class RegionCoherentPartition:
    """Result of one region-coherent partition run."""

    subset_ids: Any  # (N,) int64 in [0, subset_count) -- exactly one owner per surfel
    subset_count: int
    subset_sizes: Any  # (subset_count,) int64, descending by size
    partition_role: Any  # (N,) int8 index into PARTITION_ROLES
    ambiguous_multi_region: Any  # (N,) bool -- only meaningful where role == OWNERSHIP_PROPAGATED
    region_concentration: Any  # (subset_count,) float -- lambda_max(M_R)/trace(M_R) of the STRUCTURAL CORE only
    region_structural_core_size: Any  # (subset_count,) int64 -- structural-core member count per final subset
    rejected_merge_mask: Any  # (E_c,) bool over graph.candidate_edges -- locally accepted, region-coherence REJECTED
    graph: CandidateGraph
    gaussian_ids: Any
    config: RegionCoherenceConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def anti_chaining_boundary_edges(self) -> Any:
        """Candidate edges locally accepted (spatial + normal-compatible) but
        rejected specifically because the merged region would fail the
        region-coherence floor -- the boundary the review export visualizes."""

        return self.graph.candidate_edges[self.rejected_merge_mask]


def _lambda_max_over_trace_batch(m: Any) -> Any:
    """Closed-form largest-eigenvalue/trace ratio for a batch of symmetric 3x3
    matrices (Smith 1961's trigonometric method), fully vectorized.

    Used only for REGION scatter-matrix diagnostics (concentration/dispersion
    reporting) -- never to derive a per-surfel normal. Verified against
    `torch.linalg.eigvalsh` and against the closed-form two-normal formula in
    `concentration_floor`'s docstring (see the test module).
    """

    torch = require_torch()
    xx, yy, zz = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    xy, xz, yz = m[..., 0, 1], m[..., 0, 2], m[..., 1, 2]
    trace = xx + yy + zz
    safe_trace = trace.clamp_min(_EPS)
    q = trace / 3.0
    bxx, byy, bzz = xx - q, yy - q, zz - q
    p2 = (bxx * bxx + byy * byy + bzz * bzz + 2.0 * (xy * xy + xz * xz + yz * yz)) / 6.0
    degenerate = p2 <= _EPS
    p = torch.sqrt(p2.clamp_min(_EPS))
    ibxx, ibyy, ibzz = bxx / p, byy / p, bzz / p
    ibxy, ibxz, ibyz = xy / p, xz / p, yz / p
    det_b = (
        ibxx * (ibyy * ibzz - ibyz * ibyz)
        - ibxy * (ibxy * ibzz - ibyz * ibxz)
        + ibxz * (ibxy * ibyz - ibyy * ibxz)
    )
    r = (det_b / 2.0).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    two_pi_third = 2.0 * torch.pi / 3.0
    eig1 = q + 2 * p * torch.cos(phi)
    eig3 = q + 2 * p * torch.cos(phi + two_pi_third)
    eig2 = 3 * q - eig1 - eig3
    lambda_max = torch.maximum(torch.maximum(eig1, eig2), eig3)
    concentration = lambda_max / safe_trace
    # A degenerate (near-zero) scatter matrix has no orientation evidence;
    # 1/3 is the isotropic value (equal spread on all three axes), matching
    # what a literal isotropic M would produce were p2 not floored.
    return torch.where(degenerate, torch.full_like(concentration, 1.0 / 3.0), concentration)


def _region_coherent_merge_cpu(
    accepted_left: list[int],
    accepted_right: list[int],
    order: list[int],
    normals_flat: list[float],
    weight: float,
    concentration_floor: float,
    count: int,
) -> tuple[list[int], list[int], list[bool]]:
    """The sequential Kruskal-style merge loop itself (architecture directive
    section 7: deterministic, no thread-scheduling dependence).

    Processes accepted edges in the caller-supplied deterministic `order`
    (descending local normal alignment, ties broken by ascending edge index --
    see the caller). For each edge, if its two endpoints are already in the
    same component, nothing happens (not a rejection -- already resolved). If
    they differ, the MERGED region's concentration is tested using the
    CURRENT scatter state of exactly those two components (never an edge in
    isolation, never a stale/future state) -- this is what makes the test a
    genuine per-merge region-level decision rather than a per-edge local one,
    and what prevents a long chain of individually-passing local edges from
    silently producing one globally-incoherent region: the SAME concentration
    floor is re-applied at every step as the region grows, so a component
    already near the floor rejects a further step that would push it over.

    Pure Python (not torch) deliberately: union-find with dynamically evolving
    per-root state is an inherently sequential fixpoint, not a data-parallel
    reduction -- forcing it into batched/vectorized rounds would either
    require a stale (pre-round) coherence test for multi-way simultaneous
    merges (silently wrong: a 3-way merge could look pairwise-acceptable component and
    still be jointly incoherent) or serialize anyway. Measured at real-scene
    scale (~4M edges) this pure-Python loop runs in single-digit seconds
    (see Worklog 97), so there is no practical need for the vectorized
    approximation.

    Returns ``(parent, component_size, rejected)`` where ``rejected[k]`` says
    whether ``order[k]``'s edge was locally accepted but region-coherence
    rejected (position in the CALLER's own accepted-edge indexing, so the
    caller can scatter it back into full candidate-edge space).
    """

    parent = list(range(count))
    size = [1] * count
    # Per-root scatter matrix, 6 unique entries of a symmetric 3x3 (xx, yy,
    # zz, xy, xz, yz). Every surfel starts as its own singleton region with
    # M = weight * n n^T -- trace = weight, concentration = 1 (a single
    # normal is definitionally "concentrated"), so singleton regions never
    # fail the floor by construction.
    mxx = [0.0] * count
    myy = [0.0] * count
    mzz = [0.0] * count
    mxy = [0.0] * count
    mxz = [0.0] * count
    myz = [0.0] * count
    for i in range(count):
        nx, ny, nz = normals_flat[3 * i], normals_flat[3 * i + 1], normals_flat[3 * i + 2]
        mxx[i] = weight * nx * nx
        myy[i] = weight * ny * ny
        mzz[i] = weight * nz * nz
        mxy[i] = weight * nx * ny
        mxz[i] = weight * nx * nz
        myz[i] = weight * ny * nz

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    import math

    rejected = [False] * len(order)
    for position in order:
        u, v = accepted_left[position], accepted_right[position]
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        xx = mxx[root_u] + mxx[root_v]
        yy = myy[root_u] + myy[root_v]
        zz = mzz[root_u] + mzz[root_v]
        xy = mxy[root_u] + mxy[root_v]
        xz = mxz[root_u] + mxz[root_v]
        yz = myz[root_u] + myz[root_v]
        trace = xx + yy + zz
        if trace <= 1e-15:
            concentration = 1.0 / 3.0
        else:
            q = trace / 3.0
            bxx, byy, bzz = xx - q, yy - q, zz - q
            p2 = (bxx * bxx + byy * byy + bzz * bzz + 2.0 * (xy * xy + xz * xz + yz * yz)) / 6.0
            if p2 <= 1e-18:
                concentration = 1.0 / 3.0
            else:
                p = math.sqrt(p2)
                ibxx, ibyy, ibzz = bxx / p, byy / p, bzz / p
                ibxy, ibxz, ibyz = xy / p, xz / p, yz / p
                det_b = (
                    ibxx * (ibyy * ibzz - ibyz * ibyz)
                    - ibxy * (ibxy * ibzz - ibyz * ibxz)
                    + ibxz * (ibxy * ibyz - ibyy * ibxz)
                )
                r = det_b / 2.0
                r = 1.0 if r > 1.0 else (-1.0 if r < -1.0 else r)
                phi = math.acos(r) / 3.0
                eig1 = q + 2 * p * math.cos(phi)
                eig3 = q + 2 * p * math.cos(phi + 2.0 * math.pi / 3.0)
                eig2 = 3 * q - eig1 - eig3
                lambda_max = max(eig1, eig2, eig3)
                concentration = lambda_max / trace
        if concentration >= concentration_floor:
            # Deterministic union: the SMALLER original index always survives
            # as root, matching the min-label convention Worklog 96's own
            # connected-component solver uses, so region identity is a pure
            # function of membership, never of processing order.
            if root_u > root_v:
                root_u, root_v = root_v, root_u
            parent[root_v] = root_u
            size[root_u] += size[root_v]
            mxx[root_u], myy[root_u], mzz[root_u] = xx, yy, zz
            mxy[root_u], mxz[root_u], myz[root_u] = xy, xz, yz
        else:
            rejected[position] = True

    # Full path compression so every node's `parent` is its true final root.
    for i in range(count):
        find(i)

    return parent, size, rejected


def partition_surfels_region_coherent(
    orientation: SurfaceOrientationEvidence,
    config: RegionCoherenceConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> RegionCoherentPartition:
    """Partition EVERY surfel into exactly one final subset, using region-level
    orientation coherence (not plain transitive connectivity) as the union
    rule. See the module docstring for the full contract.
    """

    torch = require_torch()
    config = config or RegionCoherenceConfig()
    positions = orientation.positions
    normals = orientation.surface_normal
    count = int(positions.shape[0])
    device = positions.device

    graph = build_candidate_graph(orientation, config.local, progress=progress)

    if count == 0:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        return RegionCoherentPartition(
            subset_ids=empty_long,
            subset_count=0,
            subset_sizes=empty_long,
            partition_role=torch.zeros((0,), dtype=torch.int8, device=device),
            ambiguous_multi_region=torch.zeros((0,), dtype=torch.bool, device=device),
            region_concentration=torch.zeros((0,), dtype=torch.float32, device=device),
            region_structural_core_size=empty_long,
            rejected_merge_mask=torch.zeros((0,), dtype=torch.bool, device=device),
            graph=graph,
            gaussian_ids=orientation.gaussian_ids,
            config=config,
        )

    accepted_mask = graph.spatial_edge_mask & graph.normal_compatible_mask
    if config.require_positional_continuity:
        left, right = graph.candidate_edges[:, 0], graph.candidate_edges[:, 1]
        delta_x = positions[right] - positions[left]
        sign = torch.where((normals[left] * normals[right]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
        average_normal = torch.nn.functional.normalize(normals[left] + normals[right] * sign, dim=-1, eps=_EPS)
        normal_offset = (delta_x * average_normal).sum(dim=-1).abs()
        tangential_offset = (delta_x - normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
        normal_offset_ratio = normal_offset / tangential_offset.clamp_min(_EPS)
        accepted_mask = accepted_mask & (normal_offset_ratio <= config.parallel_sheet_normal_over_tangent_ratio)
    accepted_index = torch.nonzero(accepted_mask, as_tuple=False).reshape(-1)
    accepted_edges = graph.candidate_edges[accepted_index]
    accepted_alignment = graph.normal_alignment[accepted_index]

    # Deterministic processing order: most-confident local relationships
    # merge first (descending |dot|), ties broken by ascending (u, v) --
    # composed via three stable sorts (last criterion sorted first), so the
    # result is a pure function of the input, never of hash/thread order.
    edge_left = accepted_edges[:, 0]
    edge_right = accepted_edges[:, 1]
    order = torch.arange(int(accepted_edges.shape[0]), device=device)
    order = order[torch.argsort(edge_right[order], stable=True)]
    order = order[torch.argsort(edge_left[order], stable=True)]
    order = order[torch.argsort(accepted_alignment[order], descending=True, stable=True)]

    if progress is not None:
        progress(
            f"region-coherent merge: candidate_local_edges={int(graph.candidate_edges.shape[0])} "
            f"accepted_local_edges={int(accepted_edges.shape[0])} "
            f"concentration_floor={config.concentration_floor():.6f}"
        )

    parent, component_size, rejected = _region_coherent_merge_cpu(
        edge_left.detach().cpu().tolist(),
        edge_right.detach().cpu().tolist(),
        order.detach().cpu().tolist(),
        normals.detach().cpu().reshape(-1).tolist(),
        float(config.structural_weight),
        float(config.concentration_floor()),
        count,
    )
    if progress is not None:
        rejected_count = sum(rejected)
        progress(f"region-coherent merge done: rejected_merges={rejected_count}")

    parent_t = torch.tensor(parent, dtype=torch.int64, device=device)
    size_t = torch.tensor(component_size, dtype=torch.int64, device=device)
    is_structural = size_t[parent_t] >= 2

    rejected_local = torch.tensor(rejected, dtype=torch.bool, device=device)
    rejected_merge_mask = torch.zeros((int(graph.candidate_edges.shape[0]),), dtype=torch.bool, device=device)
    rejected_merge_mask[accepted_index] = rejected_local

    # --- ownership propagation: fully vectorized, one hop only -----------
    # A solitary surfel (not structural) may be attached to AT MOST one
    # neighbouring structural region via an accepted local edge. This never
    # updates that region's scatter state and never unions two roots, so it
    # cannot bridge two structural regions and cannot be chained through --
    # it is a leaf assignment, never an intermediate link.
    solitary = ~is_structural
    directed_source = torch.cat([edge_left, edge_right])
    directed_target = torch.cat([edge_right, edge_left])
    directed_align = torch.cat([accepted_alignment, accepted_alignment])
    valid = solitary[directed_source] & is_structural[directed_target]
    valid_source = directed_source[valid]
    valid_target = directed_target[valid]
    valid_align = directed_align[valid]
    valid_target_root = parent_t[valid_target]

    partition_role = torch.full((count,), PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE), dtype=torch.int8, device=device)
    ambiguous_multi_region = torch.zeros((count,), dtype=torch.bool, device=device)
    propagated_target_root = torch.full((count,), -1, dtype=torch.int64, device=device)

    if int(valid_source.shape[0]) > 0:
        # Best (highest-alignment, then smallest target node id) candidate
        # per source surfel via two scatter_reduce passes -- exact, not an
        # approximation: first the true max alignment reachable per source,
        # then (restricted to records achieving that exact max) the smallest
        # target id as the documented deterministic tie-break.
        best_align = torch.full((count,), -1.0, dtype=torch.float32, device=device)
        best_align.scatter_reduce_(0, valid_source, valid_align.to(torch.float32), reduce="amax", include_self=True)
        achieves_best = valid_align.to(torch.float32) == best_align[valid_source]
        tie_target = torch.full((count,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=device)
        tie_target.scatter_reduce_(
            0, valid_source[achieves_best], valid_target[achieves_best], reduce="amin", include_self=True
        )
        has_candidate = tie_target < torch.iinfo(torch.int64).max
        chosen_source = torch.nonzero(has_candidate, as_tuple=False).reshape(-1)
        chosen_target = tie_target[chosen_source]
        propagated_target_root[chosen_source] = parent_t[chosen_target]
        partition_role[chosen_source] = PARTITION_ROLES.index(ROLE_OWNERSHIP_PROPAGATED)

        # Ambiguity: does a solitary surfel reach more than one DISTINCT
        # structural region root through ANY of its accepted edges (not just
        # the one ultimately chosen)? Computed via a dedup on (source, root).
        pair_key = valid_source * int(count) + valid_target_root
        unique_pairs = torch.unique(pair_key)
        unique_source = torch.div(unique_pairs, count, rounding_mode="floor")
        distinct_region_count = torch.zeros((count,), dtype=torch.int64, device=device)
        distinct_region_count.index_add_(0, unique_source, torch.ones_like(unique_source))
        ambiguous_multi_region = distinct_region_count > 1

    isolated = solitary & (propagated_target_root < 0)
    partition_role[isolated] = PARTITION_ROLES.index(ROLE_ISOLATED_FALLBACK)

    # --- final subset ids ---------------------------------------------------
    # Structural-core and ownership-propagated members share their region's
    # root; isolated-fallback members are singleton final subsets keyed by
    # their own index (never merged with anything, exactly like Worklog 96's
    # fallback treatment).
    final_group_key = torch.where(partition_role == PARTITION_ROLES.index(ROLE_ISOLATED_FALLBACK),
                                    torch.arange(count, dtype=torch.int64, device=device),
                                    torch.where(propagated_target_root >= 0, propagated_target_root, parent_t))

    unique_keys, inverse, counts = torch.unique(final_group_key, return_inverse=True, return_counts=True)
    order_by_size = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order_by_size)
    subset_id_of_position[order_by_size] = torch.arange(int(order_by_size.shape[0]), dtype=order_by_size.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order_by_size]
    subset_count = int(order_by_size.shape[0])

    # --- region concentration/structural-core-size per FINAL subset --------
    # Computed from the STRUCTURAL CORE ONLY (never ownership-propagated
    # members), matching the contract that propagation never alters region
    # orientation state.
    weight = float(config.structural_weight)
    outer = weight * (normals.unsqueeze(-1) @ normals.unsqueeze(-2))
    core_mask = partition_role == PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)
    region_scatter = torch.zeros((subset_count, 3, 3), dtype=torch.float32, device=device)
    core_subset_ids = subset_ids[core_mask]
    if int(core_subset_ids.shape[0]) > 0:
        region_scatter.index_add_(0, core_subset_ids, outer[core_mask])
    region_structural_core_size = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    if int(core_subset_ids.shape[0]) > 0:
        region_structural_core_size.index_add_(0, core_subset_ids, torch.ones_like(core_subset_ids))
    region_concentration = _lambda_max_over_trace_batch(region_scatter)
    # A subset with zero structural-core members (a pure isolated-fallback
    # singleton) has no region scatter; its own single normal is trivially
    # "concentrated" (matches the singleton-region convention above).
    region_concentration = torch.where(
        region_structural_core_size > 0, region_concentration, torch.ones_like(region_concentration)
    )

    return RegionCoherentPartition(
        subset_ids=subset_ids,
        subset_count=subset_count,
        subset_sizes=subset_sizes,
        partition_role=partition_role,
        ambiguous_multi_region=ambiguous_multi_region,
        region_concentration=region_concentration,
        region_structural_core_size=region_structural_core_size,
        rejected_merge_mask=rejected_merge_mask,
        graph=graph,
        gaussian_ids=orientation.gaussian_ids,
        config=config,
    )


def count_spatially_disconnected_structural_regions(partition: RegionCoherentPartition) -> int:
    """Independent re-derivation: every STRUCTURAL region (excluding
    ownership-propagated/isolated members) must be one connected component
    under its own accepted local edges. Verified, not assumed."""

    torch = require_torch()
    count = len(partition)
    if count == 0:
        return 0
    from osn_gs.surface.torch_coverage_first_subset_partition import _connected_component_roots

    core_mask = partition.partition_role == PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)
    accepted = partition.graph.accepted_edges
    both_core = core_mask[accepted[:, 0]] & core_mask[accepted[:, 1]]
    roots = _connected_component_roots(count, accepted[both_core], partition.config.local)
    core_subset_ids = partition.subset_ids[core_mask]
    core_roots = roots[core_mask]
    unique_pairs = torch.unique(core_subset_ids * int(count) + core_roots)
    subset_of_pair = torch.div(unique_pairs, count, rounding_mode="floor")
    components_per_subset = torch.bincount(subset_of_pair, minlength=max(partition.subset_count, 1))
    return int((components_per_subset > 1).sum())


def region_coherent_accounting(partition: RegionCoherentPartition) -> dict[str, Any]:
    """Full accounting block (architecture directive section 12). Diagnostic
    only -- defines no acceptance threshold."""

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
            "min": int(sorted_sizes[0].item()),
            "median": _percentile(0.5),
            "mean": float(sorted_sizes.mean().item()),
            "p95": _percentile(0.95),
            "max": int(sorted_sizes[-1].item()),
        }

    singleton = int((sizes == 1).sum()) if int(sizes.shape[0]) > 0 else 0
    very_small = int((sizes <= VERY_SMALL_SUBSET_SIZE).sum()) if int(sizes.shape[0]) > 0 else 0
    largest_fraction = (float(sizes[0]) / count) if (int(sizes.shape[0]) > 0 and count) else 0.0

    role_counts = torch.bincount(partition.partition_role.reshape(-1).to(torch.int64), minlength=len(PARTITION_ROLES))
    ambiguous_count = int(partition.ambiguous_multi_region.sum())

    concentration = partition.region_concentration
    has_core = partition.region_structural_core_size > 0

    def _tensor_percentile(values: Any, fraction: float) -> float:
        if int(values.shape[0]) == 0:
            return 0.0
        ordered = torch.sort(values.to(torch.float64)).values
        position = min(int(ordered.shape[0]) - 1, max(0, int(round(fraction * (int(ordered.shape[0]) - 1)))))
        return float(ordered[position].item())

    structural_concentration = concentration[has_core]
    dispersion = 1.0 - structural_concentration  # simple, directly-invertible companion statistic

    largest_order = torch.argsort(sizes, descending=True, stable=True)
    largest_region_orientation = []
    for position in largest_order[: min(16, int(sizes.shape[0]))].tolist():
        largest_region_orientation.append(
            {
                "subset_size": int(sizes[position].item()),
                "structural_core_size": int(partition.region_structural_core_size[position].item()),
                "concentration": float(concentration[position].item()),
            }
        )

    return {
        "input_surfel_count": count,
        "assigned_surfel_count": assigned,
        "unassigned_surfel_count": count - assigned,
        "multiply_owned_surfel_count": 0,
        "subset_id_out_of_range_count": count - in_range,
        "subset_sizes_match_ownership_map": sizes_match,
        "coverage_identity_holds": bool(
            assigned == count
            and in_range == count
            and sizes_match
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
        "spatially_disconnected_structural_region_count": count_spatially_disconnected_structural_regions(partition),
        "local_candidate_edge_count": int(partition.graph.candidate_edges.shape[0]),
        "local_normal_compatible_edge_count": int((partition.graph.spatial_edge_mask & partition.graph.normal_compatible_mask).sum()),
        "region_coherence_rejected_merge_count": int(partition.rejected_merge_mask.sum()),
        "partition_role_counts": {name: int(role_counts[index]) for index, name in enumerate(PARTITION_ROLES)},
        "structural_core_surfel_count": int(role_counts[PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)]),
        "ownership_propagated_surfel_count": int(role_counts[PARTITION_ROLES.index(ROLE_OWNERSHIP_PROPAGATED)]),
        "isolated_fallback_surfel_count": int(role_counts[PARTITION_ROLES.index(ROLE_ISOLATED_FALLBACK)]),
        "ambiguous_multi_region_ownership_count": ambiguous_count,
        "region_orientation": {
            "concentration_median": _tensor_percentile(structural_concentration, 0.5),
            "concentration_p05": _tensor_percentile(structural_concentration, 0.05),
            "concentration_p95": _tensor_percentile(structural_concentration, 0.95),
            "dispersion_median": _tensor_percentile(dispersion, 0.5),
            "dispersion_p95": _tensor_percentile(dispersion, 0.95),
            "structural_region_count": int(has_core.sum()),
        },
        "largest_subsets_orientation": largest_region_orientation,
        "partition_parameters": partition.config.payload(),
    }
