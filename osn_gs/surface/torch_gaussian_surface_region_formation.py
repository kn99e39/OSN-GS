from __future__ import annotations

"""Consensus-aware surface-region formation (worklog 116).

Canonical principle: a pairwise ``same_surface`` relation (worklog 111/113/115)
is evidence a Gaussian PAIR could belong to the same manifold -- it is not, by
itself, proof that a whole REGION should be merged. A single edge (however it
was classified) must never be allowed to fuse two otherwise-independent
surface patches; a stable region is one where membership is explained by
multiple mutually-reinforcing pieces of evidence: shared-neighbor consensus,
local tangent-frame path consistency, and the absence of contradictory
(crease/parallel-separated/rejected) evidence nearby.

This module is explicitly NOT:
  - object segmentation (no semantic/instance notion at all)
  - ordered world-space boundary chain/loop/half-edge extraction
  - a connection to the existing Boundary-first builder, NURBS patch/control
    grid construction, the default dispatcher, or any production/trainer path

It only consumes the worklog 113/115 contract (covariance frame, structural
reliability, manifold-affinity graph) and produces a richer, still-isolated
region-candidate diagnostic. Every threshold here is a configurable policy
default, not a confirmed canonical constant.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    CANDIDATE_STATUS_CANDIDATE,
    RELATION_CREASE,
    RELATION_PARALLEL_SEPARATE,
    RELATION_REJECTED,
    RELATION_SAME_SURFACE,
    ManifoldAffinityGraph,
)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    CONTEXTUAL_CONSISTENT,
    CONTEXTUAL_MIXED,
    INTRINSIC_RELIABLE,
    INTRINSIC_REJECTED,
    StructuralReliabilityResult,
)
from osn_gs.utils.torch_ops import require_torch

POLICY_VERSION = "worklog116_v1"

# --- Region state (§3) -- NOT production_ready/eligible in this round ---
REGION_CORE = "core_region"
REGION_GROWING = "growing_region"
REGION_STABLE = "stable_region"
REGION_REVIEW_REQUIRED = "review_required"
REGION_REJECTED = "rejected_region"
REGION_SMALL_REVIEW = "small_review_region"

# --- Node membership state (§3) ---
MEMBER_CORE = "core_member"
MEMBER_CONSENSUS_ATTACHED = "consensus_attached"
MEMBER_AMBIGUOUS_UNASSIGNED = "ambiguous_unassigned"
MEMBER_CONFLICT_BOUNDARY = "conflict_boundary_candidate"
MEMBER_REJECTED = "rejected_structural_node"

# --- Edge consensus state (§5) ---
CONSENSUS_WELL_SUPPORTED = "well_supported"
CONSENSUS_WEAK = "weak"
CONSENSUS_CONTRADICTED = "contradicted"

# --- Bridge-edge veto state (§6) ---
BRIDGE_WELL_SUPPORTED = "well_supported_connection"
BRIDGE_WEAK_CANDIDATE = "weak_bridge_candidate"
BRIDGE_CONTRADICTED = "contradicted_bridge"
BRIDGE_NOT_CHECKED = "not_checked"

# --- Tangent/path consistency state (§7) ---
PATH_CONSISTENT = "path_consistent"
PATH_PHASE_ALIAS = "phase_alias_or_shortcut_candidate"
PATH_NOT_APPLICABLE = "not_applicable"

# --- Region internal contradiction state (§10) ---
CONTRADICTION_STABLE = "stable_internally_consistent"
CONTRADICTION_POSSIBLE_OVER_MERGE = "possible_over_merge"
CONTRADICTION_POSSIBLE_FRAGMENTATION = "possible_fragmentation"
CONTRADICTION_UNRESOLVED_MIXED = "unresolved_mixed_region"

# --- Region merge decision (§9) ---
MERGE_ACCEPTED = "accepted"
MERGE_REJECTED = "rejected"
MERGE_REVIEW = "review"


@dataclass(frozen=True)
class RegionFormationConfig:
    """Configurable policy (worklog 116 §7-style discipline: no canonical-final numbers)."""

    # Core-eligible edge / core seeding (§4).
    core_min_shared_neighbor_support: int = 1
    core_min_same_surface_degree: int = 2
    core_region_typical_min_size: int = 4  # LABEL only: core_region/stable_region vs small_review_region

    # Local consensus (§5).
    consensus_contradiction_ratio_threshold: float = 0.34

    # Bridge-edge veto (§6).
    bridge_min_shared_neighbor_for_well_supported: int = 2
    bridge_min_independent_cross_edges: int = 2
    bridge_local_cut_min_component_size: int = 3
    bridge_tangent_divergence_threshold: float = 0.5
    bridge_normal_separation_with_parallel_veto: float = 4.0
    bridge_borderline_tangent_residual_veto: float = 0.20
    local_backbone_max_normalized_distance: float = 4.0
    nonlocal_shortcut_mode: str = "auto"  # off | auto | force
    enable_nonlocal_shortcut_filter: bool = False  # deprecated force projection

    # Tangent-frame / path consistency (§7).
    path_max_hops: int = 6
    path_direct_small_rotation: float = 0.15
    path_shortcut_distance_ratio: float = 2.5

    # Region growth (§8).
    growth_min_support_count: int = 2
    growth_min_support_ratio: float = 1.5
    growth_max_rounds: int = 8

    # Region merge (§9).
    merge_min_independent_cross_edges: int = 2
    merge_min_independent_endpoint_coverage: int = 2
    merge_max_contradiction_ratio: float = 0.25

    # Internal contradiction diagnostic (§10).
    internal_contradiction_ratio_threshold: float = 0.15

    # Threshold-boundary proximity guard (§6 "edge가 candidate-radius/threshold
    # 경계에 가까움"): a same_surface edge whose OWN pairwise metrics sit right
    # at the manifold-affinity classifier's decision boundary is inherently
    # less trustworthy evidence than one that clears it comfortably -- a
    # cluster of such marginal edges can mutually "support" each other into a
    # false well_supported reading even when each one individually is a
    # coin-flip. Expressed as a fraction of the same_surface thresholds.
    near_threshold_margin_ratio: float = 0.15

    # --- Canonical semantics, exposed as named policy switches (worklog 37) ---
    # Both default to True: this IS current production behavior, not an
    # opt-in experiment. Named for what they mean, not which worklog
    # introduced them (worklog 36 originally named these
    # `enable_worklog34_growth_weak_bridge_exemption` /
    # `enable_worklog35_parallel_veto_nearby_evidence_gate` -- useful for
    # authoritative ablation, but it ties production semantics to an
    # implementation-history number). Diagnostic/devtool code may still flip
    # these to False for ablation replay; no production caller does.
    #
    # `allow_weak_bridge_only_growth_support`: a single-node GROWTH
    # attachment (join one still-unassigned node to an EXISTING region) is
    # a different operation from a component MERGE (fusing two
    # already-distinct core clusters) -- a same_surface edge vetoed ONLY for
    # being a "weak bridge" (insufficient independent cross-support to merge
    # two SEPARATE clusters) says nothing about whether it is good enough
    # evidence to attach one loose node to a cluster it is already part of.
    # True (canonical): growth ignores weak-bridge-only vetoes (still
    # respects edge-intrinsic vetoes: contradicted consensus, phase-alias,
    # oversized-footprint-with-parallel-evidence). False: pre-existing
    # (blanket) behavior, for ablation only.
    allow_weak_bridge_only_growth_support: bool = True
    # `require_nearby_parallel_evidence_for_parallel_veto`: the core-merge
    # parallel-shortcut override must only fire when there is ACTUAL nearby
    # parallel_separate evidence (`contradicting_parallel_neighbor_count>0`,
    # already computed by the local consensus check), not merely because a
    # raw, footprint-scale-normalized metric crosses a fixed threshold --
    # that raw metric was measured to NOT discriminate same_surface from
    # parallel_separate edges on real long-horizon-trained data. True
    # (canonical): the nearby-evidence gate is required. False: pre-existing
    # (metric-only) behavior, for ablation only.
    require_nearby_parallel_evidence_for_parallel_veto: bool = True
    # `exempt_intra_raw_component_unions_from_bridge_veto`: DIAGNOSTIC-ONLY,
    # default False (worklog 38). Worklog 37 shipped this as True believing
    # it separated seed existence from component merge; it is provably a
    # tautology (raw components are the connected components OF the same
    # edge set the veto iterates, so every core-eligible edge is exempt by
    # construction -- measured 2092/2092 on the 3k checkpoint, bridge veto
    # evaluated 0 edges, 47 articulation bridges unioned regardless). Kept
    # only so that behavior can be reproduced in ablation replay.
    exempt_intra_raw_component_unions_from_bridge_veto: bool = False
    # `separate_seed_and_merge_phases` (worklog 38, canonical): run core
    # seeding as an explicit TWO-phase DSU instead of a single sequential
    # pass that conflates the two questions.
    #   Phase 1 (seed): union ONLY over `seed_strong_edge`s -- edges whose
    #     OWN local evidence (well-supported consensus, i.e. genuine shared
    #     same_surface neighbor support, plus path consistency and no
    #     edge-intrinsic veto) makes them coherent surface interior. Each
    #     resulting component is an independently valid seed, and stays one
    #     even if it never merges with anything.
    #   Phase 2 (merge): collect the remaining (weak) cross-edges BETWEEN
    #     distinct phase-1 components and evaluate them as a component PAIR
    #     -- aggregate distinct-endpoint support, not one edge at a time --
    #     so a genuinely well-supported multi-edge junction can merge while
    #     a single fragile bridge cannot. Crucially, refusing a merge never
    #     deletes either side's seed.
    # False: single-pass legacy behavior, for ablation only.
    separate_seed_and_merge_phases: bool = True
    # Minimum number of DISTINCT endpoints on each side of a component pair
    # that must carry merge-supporting cross-edges before two independently
    # seeded components may merge. This is a COMPONENT-PAIR aggregate, a
    # different question from `bridge_min_shared_neighbor_for_well_supported`
    # (which stays at its canonical 2 and still governs per-EDGE local
    # support); worklog 38 section 9 keeps the two explicitly separate.
    merge_min_distinct_endpoint_support: int = 2


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class EdgeConsensusMetrics:
    """Multi-edge local consensus evidence for one same_surface candidate edge (§5)."""

    shared_same_surface_neighbor_count: int
    mutual_neighborhood_agreement: float
    contradicting_crease_neighbor_count: int
    contradicting_parallel_neighbor_count: int
    rejected_neighbor_contamination_count: int
    local_density_support: int
    independent_supporting_path_count: int
    supported_triangle_count: int
    tangent_transport_residual: float
    bridge_likeness: float
    consensus_state: str
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "shared_same_surface_neighbor_count": self.shared_same_surface_neighbor_count,
            "mutual_neighborhood_agreement": self.mutual_neighborhood_agreement,
            "contradicting_crease_neighbor_count": self.contradicting_crease_neighbor_count,
            "contradicting_parallel_neighbor_count": self.contradicting_parallel_neighbor_count,
            "rejected_neighbor_contamination_count": self.rejected_neighbor_contamination_count,
            "local_density_support": self.local_density_support,
            "independent_supporting_path_count": self.independent_supporting_path_count,
            "supported_triangle_count": self.supported_triangle_count,
            "tangent_transport_residual": self.tangent_transport_residual,
            "bridge_likeness": self.bridge_likeness,
            "consensus_state": self.consensus_state,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BridgeVetoResult:
    """Bridge-edge veto for a candidate cross-component connection (§6)."""

    bridge_state: str
    shared_neighbor_support: int
    local_cut_splits_large_components: bool
    tangent_frame_divergence: float
    cross_component_edge_count: int
    near_candidate_threshold_boundary: bool
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "bridge_state": self.bridge_state,
            "shared_neighbor_support": self.shared_neighbor_support,
            "local_cut_splits_large_components": self.local_cut_splits_large_components,
            "tangent_frame_divergence": self.tangent_frame_divergence,
            "cross_component_edge_count": self.cross_component_edge_count,
            "near_candidate_threshold_boundary": self.near_candidate_threshold_boundary,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PathConsistencyResult:
    """Tangent-frame transport / path-consistency evidence for a candidate edge (§7)."""

    path_found: bool
    path_length: int
    direct_rotation: float
    path_cumulative_rotation: float
    rotation_axis_consistency: float
    tangent_plane_displacement_consistency: float
    path_status: str
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "path_found": self.path_found,
            "path_length": self.path_length,
            "direct_rotation": self.direct_rotation,
            "path_cumulative_rotation": self.path_cumulative_rotation,
            "rotation_axis_consistency": self.rotation_axis_consistency,
            "tangent_plane_displacement_consistency": self.tangent_plane_displacement_consistency,
            "path_status": self.path_status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SurfaceRegionCandidate:
    """A stable-surface-region candidate (§3). NOT a production chart/patch."""

    region_id: int
    member_ids: tuple[Any, ...]
    core_member_ids: tuple[Any, ...]
    attached_ambiguous_member_ids: tuple[Any, ...]
    rejected_excluded_ids: tuple[Any, ...]
    internal_accepted_edge_ids: tuple[tuple[Any, Any], ...]
    internal_ambiguous_edge_ids: tuple[tuple[Any, Any], ...]
    boundary_conflict_edge_ids: tuple[tuple[Any, Any], ...]
    intrinsic_reliability_stats: dict[str, Any]
    contextual_consistency_stats: dict[str, Any]
    tangent_frame_consistency_stats: dict[str, Any]
    scale_stats: dict[str, Any]
    region_confidence: float
    region_state: str
    formation_reason: tuple[str, ...]
    unresolved_reasons: tuple[str, ...]
    policy_version: str

    def payload(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "member_ids": list(self.member_ids),
            "core_member_ids": list(self.core_member_ids),
            "attached_ambiguous_member_ids": list(self.attached_ambiguous_member_ids),
            "rejected_excluded_ids": list(self.rejected_excluded_ids),
            "internal_accepted_edge_ids": [list(e) for e in self.internal_accepted_edge_ids],
            "internal_ambiguous_edge_ids": [list(e) for e in self.internal_ambiguous_edge_ids],
            "boundary_conflict_edge_ids": [list(e) for e in self.boundary_conflict_edge_ids],
            "intrinsic_reliability_stats": self.intrinsic_reliability_stats,
            "contextual_consistency_stats": self.contextual_consistency_stats,
            "tangent_frame_consistency_stats": self.tangent_frame_consistency_stats,
            "scale_stats": self.scale_stats,
            "region_confidence": self.region_confidence,
            "region_state": self.region_state,
            "formation_reason": list(self.formation_reason),
            "unresolved_reasons": list(self.unresolved_reasons),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class RegionFormationResult:
    node_region_id: tuple[int, ...]
    node_membership_state: tuple[str, ...]
    unresolved_membership_ids: tuple[Any, ...]
    regions: tuple[SurfaceRegionCandidate, ...]
    config: RegionFormationConfig

    def payload(self) -> dict[str, Any]:
        return {
            "node_region_id": list(self.node_region_id),
            "node_membership_state": list(self.node_membership_state),
            "unresolved_membership_ids": list(self.unresolved_membership_ids),
            "regions": [r.payload() for r in self.regions],
            "policy_version": POLICY_VERSION,
        }


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: always attach the larger-index root under the
            # smaller-index root, independent of call order.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _build_relation_adjacency(count: int, graph: ManifoldAffinityGraph) -> tuple[
    list[set[int]], list[set[int]], list[set[int]], list[set[int]], dict[tuple[int, int], Any]
]:
    same_surface: list[set[int]] = [set() for _ in range(count)]
    crease: list[set[int]] = [set() for _ in range(count)]
    parallel_separate: list[set[int]] = [set() for _ in range(count)]
    candidate_neighbors: list[set[int]] = [set() for _ in range(count)]
    by_pair: dict[tuple[int, int], Any] = {}
    for edge in graph.edges:
        if edge.candidate_status != CANDIDATE_STATUS_CANDIDATE:
            continue
        by_pair[_pair_key(edge.source, edge.target)] = edge
        candidate_neighbors[edge.source].add(edge.target)
        candidate_neighbors[edge.target].add(edge.source)
        if edge.manifold_relation == RELATION_SAME_SURFACE:
            same_surface[edge.source].add(edge.target)
            same_surface[edge.target].add(edge.source)
        elif edge.manifold_relation == RELATION_CREASE:
            crease[edge.source].add(edge.target)
            crease[edge.target].add(edge.source)
        elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
            parallel_separate[edge.source].add(edge.target)
            parallel_separate[edge.target].add(edge.source)
    return same_surface, crease, parallel_separate, candidate_neighbors, by_pair


def _compute_edge_consensus(
    a: int, b: int, same_surface: list[set[int]], crease: list[set[int]], parallel_separate: list[set[int]],
    candidate_neighbors: list[set[int]], by_pair: dict[tuple[int, int], Any],
    reliability: StructuralReliabilityResult, frame: GaussianCovarianceFrame, config: RegionFormationConfig,
) -> EdgeConsensusMetrics:
    torch = require_torch()
    neighbors_a = same_surface[a] - {b}
    neighbors_b = same_surface[b] - {a}
    shared = neighbors_a & neighbors_b
    union_size = len(neighbors_a | neighbors_b)
    agreement = (len(shared) / union_size) if union_size else 0.0

    # Contradiction is scoped to nodes that are candidates of BOTH a and b --
    # a common candidate n where a calls it same_surface but b calls it
    # crease/parallel_separate (or vice versa) is genuine local disagreement
    # about whether a and b describe the same sheet. Counting ANY crease/
    # parallel edge merely touching a or b (regardless of relevance to the
    # (a, b) pair) was found to over-trigger: a node correctly seeing a
    # nearby-but-different surface as parallel_but_separate is expected,
    # unrelated evidence, not a contradiction of ITS OWN same_surface edges.
    common_candidates = (candidate_neighbors[a] & candidate_neighbors[b]) - {a, b}
    contradicting_crease = 0
    contradicting_parallel = 0
    for n in common_candidates:
        rel_an = by_pair.get(_pair_key(a, n))
        rel_bn = by_pair.get(_pair_key(b, n))
        values = {rel_an.manifold_relation if rel_an else None, rel_bn.manifold_relation if rel_bn else None}
        if RELATION_SAME_SURFACE in values:
            if RELATION_CREASE in values:
                contradicting_crease += 1
            if RELATION_PARALLEL_SEPARATE in values:
                contradicting_parallel += 1
    rejected_count = sum(
        1 for n in common_candidates if reliability.intrinsic.intrinsic_class[n] == INTRINSIC_REJECTED
    )

    local_density = len(neighbors_a) + len(neighbors_b)
    independent_paths = len(shared)  # each shared node is a distinct 2-hop supporting path a-n-b
    triangle_count = len(shared)

    # Tangent-transport residual proxy: do a and b agree, ON AVERAGE, about
    # their alignment with the same shared same-surface neighbors? A large
    # disagreement means a/b are not really describing the same local frame
    # even though the direct pair passed the same_surface classifier.
    if shared:
        residuals = []
        for n in shared:
            align_a = float((frame.normal_candidate[a] * frame.normal_candidate[n]).sum().abs())
            align_b = float((frame.normal_candidate[b] * frame.normal_candidate[n]).sum().abs())
            residuals.append(abs(align_a - align_b))
        tangent_transport_residual = sum(residuals) / len(residuals)
    else:
        tangent_transport_residual = 1.0  # no shared evidence -> cannot confirm consistency

    total_touch = max(len(common_candidates), 1)
    contradiction_ratio = (contradicting_crease + contradicting_parallel) / total_touch
    bridge_likeness = 1.0 / (1.0 + len(shared))

    reasons: list[str] = []
    if len(shared) >= config.core_min_shared_neighbor_support and contradiction_ratio <= config.consensus_contradiction_ratio_threshold:
        consensus_state = CONSENSUS_WELL_SUPPORTED
        reasons.append("sufficient_shared_same_surface_neighbor_support")
    elif contradiction_ratio > config.consensus_contradiction_ratio_threshold:
        consensus_state = CONSENSUS_CONTRADICTED
        reasons.append("crease_or_parallel_separate_contradiction_near_edge")
    else:
        consensus_state = CONSENSUS_WEAK
        reasons.append("insufficient_shared_same_surface_neighbor_support")

    return EdgeConsensusMetrics(
        shared_same_surface_neighbor_count=len(shared),
        mutual_neighborhood_agreement=agreement,
        contradicting_crease_neighbor_count=contradicting_crease,
        contradicting_parallel_neighbor_count=contradicting_parallel,
        rejected_neighbor_contamination_count=rejected_count,
        local_density_support=local_density,
        independent_supporting_path_count=independent_paths,
        supported_triangle_count=triangle_count,
        tangent_transport_residual=tangent_transport_residual,
        bridge_likeness=bridge_likeness,
        consensus_state=consensus_state,
        reasons=tuple(reasons),
    )


def _local_cut_disconnects_large_components(
    a: int, b: int, same_surface: list[set[int]], min_component_size: int, max_hops: int,
) -> bool:
    """Bounded local check: within ``max_hops`` of (a, b), does removing the
    direct a-b same_surface edge separate the local neighborhood into two
    components each at least ``min_component_size`` -- i.e. is this edge
    locally load-bearing rather than one of several redundant connections?"""

    def bounded_component(start: int, avoid_edge: tuple[int, int]) -> set[int]:
        visited = {start}
        frontier = [start]
        depth = {start: 0}
        while frontier:
            node = frontier.pop()
            if depth[node] >= max_hops:
                continue
            for neighbor in same_surface[node]:
                if _pair_key(node, neighbor) == avoid_edge:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    depth[neighbor] = depth[node] + 1
                    frontier.append(neighbor)
        return visited

    avoid = _pair_key(a, b)
    side_a = bounded_component(a, avoid)
    side_b = bounded_component(b, avoid)
    if b in side_a or a in side_b:
        # Still connected without the direct edge -- there is another local path.
        return False
    return len(side_a) >= min_component_size and len(side_b) >= min_component_size


def _evaluate_bridge_veto(
    a: int, b: int, consensus: EdgeConsensusMetrics, same_surface: list[set[int]], frame: GaussianCovarianceFrame,
    cross_component_edge_count: int, config: RegionFormationConfig,
) -> BridgeVetoResult:
    torch = require_torch()
    neighbors_a = same_surface[a] - {b}
    neighbors_b = same_surface[b] - {a}
    if neighbors_a:
        mean_normal_a = torch.stack([frame.normal_candidate[n] for n in neighbors_a]).mean(dim=0)
    else:
        mean_normal_a = frame.normal_candidate[a]
    if neighbors_b:
        mean_normal_b = torch.stack([frame.normal_candidate[n] for n in neighbors_b]).mean(dim=0)
    else:
        mean_normal_b = frame.normal_candidate[b]
    mean_normal_a = torch.nn.functional.normalize(mean_normal_a, dim=0)
    mean_normal_b = torch.nn.functional.normalize(mean_normal_b, dim=0)
    tangent_frame_divergence = 1.0 - float((mean_normal_a * mean_normal_b).sum().abs())

    splits = _local_cut_disconnects_large_components(
        a, b, same_surface, config.bridge_local_cut_min_component_size, config.path_max_hops,
    )
    near_threshold = consensus.shared_same_surface_neighbor_count in (
        config.core_min_shared_neighbor_support, config.core_min_shared_neighbor_support - 1,
    )

    reasons: list[str] = []
    if consensus.consensus_state == CONSENSUS_CONTRADICTED:
        bridge_state = BRIDGE_CONTRADICTED
        reasons.append("crease_or_parallel_separate_contradiction_near_connection")
    elif (
        consensus.shared_same_surface_neighbor_count >= config.bridge_min_shared_neighbor_for_well_supported
        and not splits
        and tangent_frame_divergence <= config.bridge_tangent_divergence_threshold
        and cross_component_edge_count >= 1
    ):
        bridge_state = BRIDGE_WELL_SUPPORTED
        reasons.append("sufficient_shared_neighbor_support_and_no_local_cut_dependency")
    elif splits or tangent_frame_divergence > config.bridge_tangent_divergence_threshold or consensus.shared_same_surface_neighbor_count < config.bridge_min_shared_neighbor_for_well_supported:
        bridge_state = BRIDGE_WEAK_CANDIDATE
        if splits:
            reasons.append("removing_edge_splits_local_neighborhood_into_two_large_components")
        if tangent_frame_divergence > config.bridge_tangent_divergence_threshold:
            reasons.append("local_tangent_frame_divergence_between_endpoints_neighborhoods")
        if consensus.shared_same_surface_neighbor_count < config.bridge_min_shared_neighbor_for_well_supported:
            reasons.append("too_few_shared_same_surface_neighbors_for_confident_bridge")
    else:
        bridge_state = BRIDGE_WELL_SUPPORTED
        reasons.append("passes_all_bridge_checks")

    return BridgeVetoResult(
        bridge_state=bridge_state,
        shared_neighbor_support=consensus.shared_same_surface_neighbor_count,
        local_cut_splits_large_components=splits,
        tangent_frame_divergence=tangent_frame_divergence,
        cross_component_edge_count=cross_component_edge_count,
        near_candidate_threshold_boundary=near_threshold,
        reasons=tuple(reasons),
    )


def _shortest_same_surface_path(
    a: int, b: int, same_surface: list[set[int]], avoid_edge: tuple[int, int], max_hops: int,
) -> list[int] | None:
    if a == b:
        return [a]
    visited = {a}
    frontier = [[a]]
    while frontier:
        next_frontier = []
        for path in frontier:
            node = path[-1]
            if len(path) - 1 >= max_hops:
                continue
            for neighbor in sorted(same_surface[node]):
                if _pair_key(node, neighbor) == avoid_edge:
                    continue
                if neighbor == b:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(path + [neighbor])
        frontier = next_frontier
    return None


def _evaluate_path_consistency(
    a: int, b: int, same_surface: list[set[int]], frame: GaussianCovarianceFrame, config: RegionFormationConfig,
) -> PathConsistencyResult:
    torch = require_torch()
    path = _shortest_same_surface_path(a, b, same_surface, _pair_key(a, b), config.path_max_hops)
    direct_alignment = float((frame.normal_candidate[a] * frame.normal_candidate[b]).sum().abs())
    direct_rotation = 1.0 - direct_alignment
    if path is None or len(path) < 3:
        return PathConsistencyResult(
            path_found=path is not None,
            path_length=(len(path) - 1) if path is not None else 0,
            direct_rotation=direct_rotation,
            path_cumulative_rotation=0.0,
            rotation_axis_consistency=1.0,
            tangent_plane_displacement_consistency=1.0,
            path_status=PATH_NOT_APPLICABLE,
            reasons=("no_alternate_same_surface_path_within_hop_limit",) if path is None else ("path_too_short_for_path_evidence",),
        )

    hop_rotations = []
    axes = []
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        align = float((frame.normal_candidate[n1] * frame.normal_candidate[n2]).sum().abs())
        hop_rotations.append(1.0 - align)
        axis = torch.linalg.cross(frame.normal_candidate[n1], frame.normal_candidate[n2])
        norm = float(torch.linalg.vector_norm(axis))
        axes.append(axis / norm if norm > 1e-8 else None)

    path_cumulative_rotation = sum(hop_rotations)

    valid_axes = [ax for ax in axes if ax is not None]
    if len(valid_axes) >= 2:
        pairwise = []
        for i in range(len(valid_axes) - 1):
            pairwise.append(float((valid_axes[i] * valid_axes[i + 1]).sum().abs()))
        rotation_axis_consistency = sum(pairwise) / len(pairwise)
    else:
        rotation_axis_consistency = 1.0

    tangential_path_length = 0.0
    torch_positions_available = hasattr(frame, "tangent_major_scale")
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        scale = float((frame.tangent_major_scale[n1] + frame.tangent_major_scale[n2]) / 2.0)
        tangential_path_length += max(scale, 1e-9)
    direct_scale = float((frame.tangent_major_scale[a] + frame.tangent_major_scale[b]) / 2.0)
    tangent_plane_displacement_consistency = (direct_scale * len(path) / max(tangential_path_length, 1e-9)) if tangential_path_length else 1.0

    reasons: list[str] = []
    looks_aligned_directly = direct_rotation <= config.path_direct_small_rotation
    path_suggests_large_separation = (
        path_cumulative_rotation > config.path_direct_small_rotation * max(len(path) - 1, 1)
        and rotation_axis_consistency < 0.5
    )
    if looks_aligned_directly and path_suggests_large_separation:
        path_status = PATH_PHASE_ALIAS
        reasons.append("direct_pair_looks_aligned_but_path_shows_incoherent_cumulative_rotation")
    else:
        path_status = PATH_CONSISTENT
        reasons.append("direct_pair_agrees_with_path_evidence")

    return PathConsistencyResult(
        path_found=True,
        path_length=len(path) - 1,
        direct_rotation=direct_rotation,
        path_cumulative_rotation=path_cumulative_rotation,
        rotation_axis_consistency=rotation_axis_consistency,
        tangent_plane_displacement_consistency=tangent_plane_displacement_consistency,
        path_status=path_status,
        reasons=tuple(reasons),
    )


# --- Typed core edge categories (worklog 38 §4) ---
EDGE_SEED_STRONG = "seed_strong_edge"
EDGE_WEAK_BRIDGE = "weak_bridge_edge"
EDGE_CONSENSUS_CONTRADICTED = "consensus_contradicted_edge"
EDGE_PHASE_ALIAS = "phase_alias_edge"
EDGE_OVERSIZED_FOOTPRINT = "oversized_footprint_edge"
EDGE_MERGE_SUPPORTED = "merge_supported_edge"


def _classify_core_edge(
    a: int, b: int, consensus: EdgeConsensusMetrics, path_result: PathConsistencyResult | None,
    by_pair: dict[tuple[int, int], Any], config: RegionFormationConfig,
) -> str:
    """Worklog 38 §4: assign one typed category per core-eligible edge.

    A ``seed_strong_edge`` is an edge whose OWN local evidence is strong
    enough to declare its two endpoints interior to one coherent surface
    fragment -- specifically ``CONSENSUS_WELL_SUPPORTED`` (which already
    means "at least ``core_min_shared_neighbor_support`` genuine shared
    same_surface neighbours AND contradiction ratio under threshold", i.e.
    real multi-edge local support, not merely "the classifier said
    same_surface") plus path consistency and no edge-intrinsic veto.
    Everything else that survives the intrinsic vetoes is a
    ``weak_bridge_edge``: usable as component-to-component ADJACENCY
    evidence, never as a seed union.
    """
    if consensus.consensus_state == CONSENSUS_CONTRADICTED:
        return EDGE_CONSENSUS_CONTRADICTED
    if path_result is not None and path_result.path_status == PATH_PHASE_ALIAS:
        return EDGE_PHASE_ALIAS
    direct_metrics = by_pair[(a, b)].metrics
    if (
        direct_metrics is not None
        and direct_metrics.normal_direction_separation_over_thickness
        > config.bridge_normal_separation_with_parallel_veto
        and direct_metrics.mutual_tangent_residual
        > config.bridge_borderline_tangent_residual_veto
        and (
            not config.require_nearby_parallel_evidence_for_parallel_veto
            or consensus.contradicting_parallel_neighbor_count > 0
        )
    ):
        return EDGE_OVERSIZED_FOOTPRINT
    if consensus.consensus_state == CONSENSUS_WELL_SUPPORTED:
        return EDGE_SEED_STRONG
    return EDGE_WEAK_BRIDGE


def _seed_core_components_two_phase(
    count: int, core_eligible: list[tuple[int, int]], same_surface: list[set[int]],
    crease: list[set[int]], parallel_separate: list[set[int]], candidate_neighbors: list[set[int]],
    by_pair: dict[tuple[int, int], Any], reliability: StructuralReliabilityResult,
    frame: GaussianCovarianceFrame, config: RegionFormationConfig, uf: _UnionFind,
    consensus_by_pair: dict[tuple[int, int], EdgeConsensusMetrics],
    bridge_by_pair: dict[tuple[int, int], BridgeVetoResult],
    path_by_pair: dict[tuple[int, int], PathConsistencyResult],
    boundary_conflict_edges: set[tuple[int, int]],
):
    """Worklog 38 §5: explicit two-phase seed/merge separation.

    Phase 1 unions ONLY ``seed_strong_edge``s, so every resulting component
    is an independently valid seed that exists on its own local evidence.
    Phase 2 then evaluates the remaining weak cross-edges as component
    PAIRS (aggregate distinct-endpoint support), so a well-supported
    multi-edge junction can still merge while a single fragile bridge
    cannot -- and, critically, a refused merge leaves BOTH components
    seeded rather than dissolving them.

    This replaces worklog 37's tautological
    ``exempt_intra_raw_component_unions_from_bridge_veto``, which disabled
    the bridge veto entirely (100% of core-eligible edges exempt by
    construction) instead of separating the two questions.
    """
    edge_category: dict[tuple[int, int], str] = {}

    # --- Phase 1: seed formation over strong edges only ---
    for a, b in core_eligible:
        consensus = consensus_by_pair[(a, b)]
        path_result = None
        if consensus.consensus_state != CONSENSUS_CONTRADICTED:
            path_result = _evaluate_path_consistency(a, b, same_surface, frame, config)
            path_by_pair[(a, b)] = path_result
        category = _classify_core_edge(a, b, consensus, path_result, by_pair, config)
        edge_category[(a, b)] = category
        if category in (EDGE_CONSENSUS_CONTRADICTED, EDGE_PHASE_ALIAS, EDGE_OVERSIZED_FOOTPRINT):
            boundary_conflict_edges.add((a, b))
            continue
        if category == EDGE_SEED_STRONG:
            uf.union(a, b)

    # --- Phase 2: component-pair merge over the remaining weak edges ---
    # Group weak cross-edges by the phase-1 component pair they connect.
    cross_edges_by_component_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b), category in edge_category.items():
        if category != EDGE_WEAK_BRIDGE:
            continue
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue  # already in one seed component; nothing to merge
        cross_edges_by_component_pair.setdefault(_pair_key(ra, rb), []).append((a, b))

    for component_key, edges in sorted(cross_edges_by_component_pair.items()):
        # Aggregate component-pair support: how many DISTINCT endpoints on
        # each side carry a cross-edge. One fragile edge gives 1/1 and can
        # never satisfy `merge_min_distinct_endpoint_support`; a genuine
        # multi-edge junction gives >=2/2.
        side_a_endpoints = set()
        side_b_endpoints = set()
        root_a, _root_b = component_key
        for a, b in edges:
            if uf.find(a) == root_a:
                side_a_endpoints.add(a)
                side_b_endpoints.add(b)
            else:
                side_a_endpoints.add(b)
                side_b_endpoints.add(a)
        endpoint_support = min(len(side_a_endpoints), len(side_b_endpoints))

        # The per-EDGE bridge veto still runs (unchanged semantics,
        # `bridge_min_shared_neighbor_for_well_supported` untouched); the
        # component-pair aggregate is an ADDITIONAL requirement, so this can
        # only ever be stricter than the legacy single-edge path.
        best_bridge = None
        for a, b in edges:
            bridge = _evaluate_bridge_veto(
                a, b, consensus_by_pair[(a, b)], same_surface, frame, len(edges), config,
            )
            bridge_by_pair[(a, b)] = bridge
            if bridge.bridge_state == BRIDGE_WELL_SUPPORTED:
                best_bridge = (a, b)

        if best_bridge is not None and endpoint_support >= config.merge_min_distinct_endpoint_support:
            edge_category[best_bridge] = EDGE_MERGE_SUPPORTED
            uf.union(*best_bridge)
            for a, b in edges:
                if (a, b) != best_bridge:
                    boundary_conflict_edges.add((a, b))
        else:
            # Refused merge: both components KEEP their seeds. The edges
            # stay recorded as conflict/adjacency evidence only.
            for a, b in edges:
                boundary_conflict_edges.add((a, b))

    return uf, consensus_by_pair, bridge_by_pair, path_by_pair, boundary_conflict_edges


def _seed_core_components(
    count: int, same_surface: list[set[int]], crease: list[set[int]], parallel_separate: list[set[int]],
    candidate_neighbors: list[set[int]], by_pair: dict[tuple[int, int], Any],
    reliability: StructuralReliabilityResult, frame: GaussianCovarianceFrame,
    config: RegionFormationConfig,
) -> tuple[_UnionFind, dict[tuple[int, int], EdgeConsensusMetrics], dict[tuple[int, int], BridgeVetoResult], dict[tuple[int, int], PathConsistencyResult], set[tuple[int, int]]]:
    """Union-Find over core-eligible same_surface edges, with a bridge veto
    applied specifically to edges that would merge two currently-distinct
    components (worklog 116 §4/§6)."""
    uf = _UnionFind(count)
    consensus_by_pair: dict[tuple[int, int], EdgeConsensusMetrics] = {}
    bridge_by_pair: dict[tuple[int, int], BridgeVetoResult] = {}
    path_by_pair: dict[tuple[int, int], PathConsistencyResult] = {}
    boundary_conflict_edges: set[tuple[int, int]] = set()

    # Core-eligible edge (worklog 116 §4): endpoints must be INTRINSICALLY
    # reliable -- deliberately NOT gated on the upstream pairwise
    # ``endpoint_status``/``relation_confidence`` fields, which (worklog
    # 114/115) fold CONTEXTUAL consistency into "confidence" too. That
    # conflation is correct for pairwise classification but wrong here: two
    # legitimately separate-but-nearby surfaces (e.g. close_parallel_sheets)
    # can make EVERY node's raw-kNN contextual neighborhood "mixed" even
    # though the covariance-guided same_surface edges are perfectly clean --
    # this module's own multi-edge consensus check (below) is the real
    # confidence signal for core formation, not the coarser pairwise one.
    intrinsic_class_lookup = reliability.intrinsic.intrinsic_class
    core_eligible: list[tuple[int, int]] = []
    for key, edge in by_pair.items():
        if edge.manifold_relation != RELATION_SAME_SURFACE:
            continue
        if ((config.nonlocal_shortcut_mode == "force" or config.enable_nonlocal_shortcut_filter) and (edge.metrics is None or edge.metrics.normalized_distance > config.local_backbone_max_normalized_distance)) or (config.nonlocal_shortcut_mode == "auto" and config.enable_nonlocal_shortcut_filter and edge.metrics is not None and edge.metrics.normalized_distance > config.local_backbone_max_normalized_distance):
            continue
        a, b = key
        if intrinsic_class_lookup[a] != INTRINSIC_RELIABLE or intrinsic_class_lookup[b] != INTRINSIC_RELIABLE:
            continue
        core_eligible.append(key)

    # Worklog 38: worklog 37 introduced an
    # `exempt_intra_raw_component_unions_from_bridge_veto` flag that computed
    # "raw same_surface connected components" FROM ``core_eligible`` itself
    # and then skipped the bridge veto for any edge whose endpoints shared a
    # raw component. That is a TAUTOLOGY: connected components of an edge set
    # contain, by construction, both endpoints of every edge in that set, so
    # 100% of core-eligible edges were exempt (measured: 2092/2092 on the 3k
    # checkpoint, bridge veto evaluated on exactly 0 edges vs 1244 without
    # the flag) and 47 genuine articulation ("single fragile edge, >=3 nodes
    # on each side") bridges were unioned anyway. The flag disabled the
    # bridge veto outright rather than separating seed existence from merge.
    # It is reverted to False here and kept only as a diagnostic opt-in;
    # the real separation is implemented as an explicit two-phase DSU below
    # (`_seed_strong_edge_components` -> component-pair merge evaluation).

    for a, b in core_eligible:
        consensus_by_pair[(a, b)] = _compute_edge_consensus(a, b, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, reliability, frame, config)

    # Deterministic priority: well-supported first, more shared support first,
    # then by (a, b) index for tie-breaking (positions here are already the
    # caller's stable-ordered index space -- see form_surface_regions).
    def priority(key: tuple[int, int]) -> tuple[int, int, tuple[int, int]]:
        metrics = consensus_by_pair[key]
        rank = {CONSENSUS_WELL_SUPPORTED: 0, CONSENSUS_WEAK: 1, CONSENSUS_CONTRADICTED: 2}[metrics.consensus_state]
        return (rank, -metrics.shared_same_surface_neighbor_count, key)

    core_eligible.sort(key=priority)

    if config.separate_seed_and_merge_phases:
        return _seed_core_components_two_phase(
            count, core_eligible, same_surface, crease, parallel_separate, candidate_neighbors,
            by_pair, reliability, frame, config, uf, consensus_by_pair, bridge_by_pair,
            path_by_pair, boundary_conflict_edges,
        )

    pending_cross_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in core_eligible:
        consensus = consensus_by_pair[(a, b)]
        if consensus.consensus_state == CONSENSUS_CONTRADICTED:
            boundary_conflict_edges.add((a, b))
            continue
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue  # already-connected: no bridge check needed (not_checked)

        path_result = _evaluate_path_consistency(a, b, same_surface, frame, config)
        path_by_pair[(a, b)] = path_result
        if path_result.path_status == PATH_PHASE_ALIAS:
            boundary_conflict_edges.add((a, b))
            continue

        # A direct pair that has a large normal-direction separation AND
        # nearby parallel-separated evidence is a close-parallel shortcut,
        # even when false same_surface triangles make it look locally dense.
        # Edge-intrinsic (like CONTRADICTED/PHASE_ALIAS above) -- evaluated
        # unconditionally, regardless of raw-component membership below.
        #
        # Worklog 35: `normal_direction_separation_over_thickness` is
        # normalized by `frame.normal_thickness` -- an individual Gaussian
        # FOOTPRINT scale, the same category of quantity worklog 30-33 found
        # unusable for real long-horizon-trained data (median 12-16x off).
        # Direct measurement on the real 3k/5k/10k checkpoints: this metric's
        # distribution for edges the affinity graph ALREADY classified
        # `same_surface` (median 108, range 0.09-4453) heavily OVERLAPS the
        # distribution for genuine `parallel_separate` edges (median 645) --
        # it does not discriminate at any fixed threshold on this data, let
        # alone the literal "4.0" configured here. 30% of real cross-
        # component pairs with >=2 same_surface bridge edges (56/184 on the
        # 3k checkpoint) had at least one edge individually pass
        # `_evaluate_bridge_veto` as well-supported, only to be vetoed again
        # here by this raw metric -- silently keeping otherwise-well-
        # evidenced core components fragmented.
        #
        # The comment above ("AND nearby parallel-separated evidence") states
        # the intended guard correctly, but the code never actually checked
        # for nearby parallel evidence -- only the direct pair's own
        # mis-scaled metrics. `consensus.contradicting_parallel_neighbor_count`
        # (already computed above, worklog 116) is exactly that missing
        # signal: a genuine count of NEARBY parallel_separate relations
        # shared between a and b's own candidate neighborhoods. Requiring it
        # to be nonzero makes this override do what its own comment always
        # claimed, using the affinity graph's OWN already-correctly-scaled
        # same_surface/parallel_separate classification as the discriminator
        # instead of re-deriving a second, differently-thresholded check on
        # an unnormalized metric.
        direct_metrics = by_pair[(a, b)].metrics
        if (
            direct_metrics is not None
            and direct_metrics.normal_direction_separation_over_thickness
            > config.bridge_normal_separation_with_parallel_veto
            and direct_metrics.mutual_tangent_residual
            > config.bridge_borderline_tangent_residual_veto
            and (
                not config.require_nearby_parallel_evidence_for_parallel_veto
                or consensus.contradicting_parallel_neighbor_count > 0
            )
        ):
            boundary_conflict_edges.add((a, b))
            continue

        if config.exempt_intra_raw_component_unions_from_bridge_veto:
            # Diagnostic-only (worklog 38): reproduces worklog 37's
            # tautological behavior for ablation replay. Never enabled in
            # production -- see the note above `for a, b in core_eligible`.
            uf.union(a, b)
            continue

        component_key = _pair_key(ra, rb)
        crossing_count = len(pending_cross_edges.get(component_key, [])) + 1
        bridge = _evaluate_bridge_veto(a, b, consensus, same_surface, frame, crossing_count, config)
        bridge_by_pair[(a, b)] = bridge

        if bridge.bridge_state == BRIDGE_WELL_SUPPORTED:
            uf.union(a, b)
            pending_cross_edges.pop(component_key, None)
        else:
            pending_cross_edges.setdefault(component_key, []).append((a, b))
            boundary_conflict_edges.add((a, b))

    # Pending weak bridges remain explicit conflict evidence. They are reconsidered only by the final region-merge policy, which requires both well-supported consensus and a well-supported bridge veto result.

    return uf, consensus_by_pair, bridge_by_pair, path_by_pair, boundary_conflict_edges


def form_surface_regions(
    positions: Any,
    frame: GaussianCovarianceFrame,
    reliability: StructuralReliabilityResult,
    graph: ManifoldAffinityGraph,
    *,
    config: RegionFormationConfig | None = None,
    ids: Sequence[Any] | None = None,
) -> RegionFormationResult:
    """Consensus-aware surface-region formation (worklog 116).

    NOT a connected-component pass over ``same_surface`` edges: region
    membership requires shared-neighbor consensus, a bridge-edge veto for any
    edge that would merge two independent components, and tangent-frame path
    consistency to catch phase-aliased/shortcut pairs. Explicitly a
    diagnostic-level foundation -- no ordered boundary graph, no NURBS patch
    construction, no dispatcher/production connection.
    """
    config = config or RegionFormationConfig()
    count = int(reliability.intrinsic.conditioning_score.shape[0])
    ids = tuple(ids) if ids is not None else tuple(range(count))
    if len(ids) != count:
        raise ValueError("ids must have the same length as the reliability result.")

    same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, graph)
    if config.nonlocal_shortcut_mode == "auto":
        broad_candidate_graph = (sum(len(neighbors) for neighbors in candidate_neighbors) / max(count, 1)) > 12.0
        config = replace(config, enable_nonlocal_shortcut_filter=broad_candidate_graph)

    uf, consensus_by_pair, bridge_by_pair, path_by_pair, boundary_conflict_edges = _seed_core_components(
        count, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, reliability, frame, config,
    )

    # Component membership, keyed by union-find root.
    components: dict[int, list[int]] = {}
    for node in range(count):
        components.setdefault(uf.find(node), []).append(node)

    intrinsic_class = reliability.intrinsic.intrinsic_class
    contextual_class = reliability.contextual.contextual_class

    # A node only counts as a genuine CORE seed if its intrinsic evidence is
    # reliable AND it has enough core-eligible same_surface degree within its
    # own component -- guards against a lone unsupported pair masquerading as
    # a seed (worklog 116 §4).
    core_degree = [0] * count
    for (a, b), consensus in consensus_by_pair.items():
        if consensus.consensus_state != CONSENSUS_CONTRADICTED and uf.find(a) == uf.find(b):
            core_degree[a] += 1
            core_degree[b] += 1

    # Sparse-but-continuous sampling can legitimately have no triangles at
    # all: every direct relation is therefore a weak bridge candidate even
    # though the whole local sheet is mutually supported as a network. Do not
    # promote one such pair. When the strict pass produced no core anywhere,
    # a sufficiently sized, contradiction-free same-surface network is a
    # reviewable fallback core. Close-parallel/crease neighborhoods cannot use
    # it because their local contradictory relations veto it; a two-node pair
    # still fails the ordinary minimum-degree condition below.
    has_strict_core = any(
        intrinsic_class[n] == INTRINSIC_RELIABLE
        and core_degree[n] >= config.core_min_same_surface_degree
        for n in range(count)
    )
    if not has_strict_core:
        fallback_uf = _UnionFind(count)
        for a, neighbors in enumerate(same_surface):
            if intrinsic_class[a] != INTRINSIC_RELIABLE or crease[a] or parallel_separate[a]:
                continue
            for b in neighbors:
                if (
                    a < b
                    and intrinsic_class[b] == INTRINSIC_RELIABLE
                    and not crease[b]
                    and not parallel_separate[b]
                ):
                    fallback_uf.union(a, b)
        uf = fallback_uf
        components = {}
        for node in range(count):
            components.setdefault(uf.find(node), []).append(node)
        core_degree = [
            sum(
                1
                for neighbor in same_surface[node]
                if intrinsic_class[neighbor] == INTRINSIC_RELIABLE
                and not crease[node]
                and not parallel_separate[node]
                and not crease[neighbor]
                and not parallel_separate[neighbor]
            )
            for node in range(count)
        ]

    node_region_id = [-1] * count
    node_membership_state = [MEMBER_REJECTED] * count
    region_of_component: dict[int, int] = {}
    regions: list[SurfaceRegionCandidate] = []

    ordered_roots = sorted(components.keys(), key=lambda r: ids[r] if not isinstance(ids[r], (int, float)) else ids[r])
    for root in ordered_roots:
        members = components[root]
        core_members = [
            n for n in members
            if intrinsic_class[n] == INTRINSIC_RELIABLE and core_degree[n] >= config.core_min_same_surface_degree
        ]
        if not core_members:
            for n in members:
                node_membership_state[n] = (
                    MEMBER_REJECTED if intrinsic_class[n] == INTRINSIC_REJECTED else MEMBER_AMBIGUOUS_UNASSIGNED
                )
            continue
        region_id = len(regions)
        region_of_component[root] = region_id
        for n in core_members:
            node_region_id[n] = region_id
            node_membership_state[n] = MEMBER_CORE
        regions.append(None)  # placeholder, filled in below once growth completes

    # --- Region growth (§8): attach ambiguous/insufficient nodes by consensus. ---
    unresolved_membership_ids: list[Any] = []
    for _round in range(config.growth_max_rounds):
        changed = False
        unassigned = [
            n for n in range(count)
            if node_region_id[n] == -1 and intrinsic_class[n] != INTRINSIC_REJECTED
        ]
        unassigned.sort(key=lambda n: str(ids[n]))
        for n in unassigned:
            support: dict[int, int] = {}
            for neighbor in same_surface[n]:
                region_id = node_region_id[neighbor]
                if region_id == -1:
                    continue
                edge = by_pair.get(_pair_key(n, neighbor))
                if edge is None or edge.manifold_relation != RELATION_SAME_SURFACE:
                    continue
                if ((config.nonlocal_shortcut_mode == "force" or config.enable_nonlocal_shortcut_filter) and (edge.metrics is None or edge.metrics.normalized_distance > config.local_backbone_max_normalized_distance)) or (config.nonlocal_shortcut_mode == "auto" and config.enable_nonlocal_shortcut_filter and edge.metrics is not None and edge.metrics.normalized_distance > config.local_backbone_max_normalized_distance):
                    continue
                pair_key = _pair_key(n, neighbor)
                if pair_key in boundary_conflict_edges:
                    # Worklog 34: `boundary_conflict_edges` (built in
                    # `_seed_core_components`) conflates two different
                    # concepts. (1) Edge-intrinsic low-trust signals --
                    # CONSENSUS_CONTRADICTED, PATH_PHASE_ALIAS, and the
                    # oversized-footprint-with-parallel-veto check -- are
                    # real per-edge quality problems, valid regardless of
                    # context, and must still block growth here. (2) A
                    # WEAK/not-well-supported BRIDGE veto specifically
                    # answers "should two ALREADY-DISTINCT core components
                    # merge through this edge without enough independent
                    # cross-support" -- a component-MERGE question that does
                    # not apply to single-node GROWTH (attaching one still-
                    # unassigned node `n` to an existing region is not a
                    # component merge). Reusing the blanket set here
                    # silently vetoed real long-horizon-checkpoint growth
                    # candidates whose own same-surface support to their
                    # target region was otherwise sufficient (worklog 34:
                    # 93/93 checked real-snapshot candidates were blocked
                    # this way, explaining consensus_attached=1 despite
                    # 1079 representatives having same_surface degree>=2).
                    if not config.allow_weak_bridge_only_growth_support:
                        # Pre-fix behavior (ablation-only): blanket block,
                        # weak-bridge-only vetoes included.
                        continue
                    consensus = consensus_by_pair.get(pair_key)
                    path_result = path_by_pair.get(pair_key)
                    is_contradicted = consensus is not None and consensus.consensus_state == CONSENSUS_CONTRADICTED
                    is_phase_alias = path_result is not None and path_result.path_status == PATH_PHASE_ALIAS
                    is_oversized_footprint_veto = (
                        edge.metrics is not None
                        and edge.metrics.normal_direction_separation_over_thickness
                        > config.bridge_normal_separation_with_parallel_veto
                        and edge.metrics.mutual_tangent_residual
                        > config.bridge_borderline_tangent_residual_veto
                    )
                    if is_contradicted or is_phase_alias or is_oversized_footprint_veto:
                        continue
                    # else: the ONLY reason this edge is in
                    # boundary_conflict_edges is a weak-bridge (component-
                    # merge-specific) veto -- does not block single-node growth.
                support[region_id] = support.get(region_id, 0) + 1
            if not support:
                node_membership_state[n] = (
                    MEMBER_AMBIGUOUS_UNASSIGNED if intrinsic_class[n] != INTRINSIC_REJECTED else MEMBER_REJECTED
                )
                continue
            ranked = sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))
            top_region, top_support = ranked[0]
            runner_up_support = ranked[1][1] if len(ranked) > 1 else 0
            has_crease_conflict = any(
                by_pair.get(_pair_key(n, nb)) is not None and by_pair[_pair_key(n, nb)].manifold_relation == RELATION_CREASE
                for nb in crease[n]
            )
            if (
                top_support >= config.growth_min_support_count
                and (runner_up_support == 0 or top_support >= config.growth_min_support_ratio * runner_up_support)
            ):
                node_region_id[n] = top_region
                node_membership_state[n] = (
                    MEMBER_CONFLICT_BOUNDARY if has_crease_conflict else MEMBER_CONSENSUS_ATTACHED
                )
                changed = True
            elif len(ranked) > 1 and runner_up_support > 0 and top_support < config.growth_min_support_ratio * runner_up_support:
                node_membership_state[n] = MEMBER_CONFLICT_BOUNDARY
                if ids[n] not in unresolved_membership_ids:
                    unresolved_membership_ids.append(ids[n])
            else:
                node_membership_state[n] = MEMBER_AMBIGUOUS_UNASSIGNED
        if not changed:
            break

    for n in range(count):
        if intrinsic_class[n] == INTRINSIC_REJECTED:
            node_membership_state[n] = MEMBER_REJECTED
            node_region_id[n] = -1

    # --- Region merge (§9): multi-edge cross-region evidence only. ---
    cross_region_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b), edge in by_pair.items():
        if edge.manifold_relation != RELATION_SAME_SURFACE:
            continue
        ra, rb = node_region_id[a], node_region_id[b]
        if ra == -1 or rb == -1 or ra == rb:
            continue
        cross_region_edges.setdefault(_pair_key(ra, rb), []).append((a, b))

    region_merge_map = {i: i for i in range(len(regions))}

    def merged_root(i: int) -> int:
        while region_merge_map[i] != i:
            i = region_merge_map[i]
        return i

    for (ra, rb), edges in cross_region_edges.items():
        # A later merge must not re-introduce an edge the core seeding pass
        # marked as weak/contradicted bridge evidence. In particular, this
        # prevents gap=0.02 false same_surface pairs from bypassing the bridge
        # veto merely because their count is greater than one.
        merge_edges = [
            (a, b)
            for a, b in edges
            if (a, b) not in boundary_conflict_edges
            and (a, b) in consensus_by_pair
            and consensus_by_pair[(a, b)].consensus_state == CONSENSUS_WELL_SUPPORTED
            and bridge_by_pair[(a, b)].bridge_state == BRIDGE_WELL_SUPPORTED
        ]
        if len(merge_edges) < config.merge_min_independent_cross_edges:
            continue
        edges = merge_edges
        side_a_nodes = {e[0] for e in edges} | {e[1] for e in edges if node_region_id[e[1]] == ra}
        side_b_nodes = {e[1] for e in edges} | {e[0] for e in edges if node_region_id[e[0]] == rb}
        endpoint_coverage = min(len(side_a_nodes), len(side_b_nodes))
        contradiction_count = 0
        for a, b in edges:
            for nb in crease[a] | crease[b] | parallel_separate[a] | parallel_separate[b]:
                if node_region_id[nb] in (ra, rb):
                    contradiction_count += 1
        contradiction_ratio = contradiction_count / max(len(edges), 1)
        if endpoint_coverage >= config.merge_min_independent_endpoint_coverage and contradiction_ratio <= config.merge_max_contradiction_ratio:
            root_a, root_b = merged_root(ra), merged_root(rb)
            if root_a != root_b:
                lo, hi = (root_a, root_b) if root_a < root_b else (root_b, root_a)
                region_merge_map[hi] = lo

    final_region_ids = sorted({merged_root(i) for i in range(len(regions))})
    remap = {old: new for new, old in enumerate(final_region_ids)}
    for n in range(count):
        if node_region_id[n] != -1:
            node_region_id[n] = remap[merged_root(node_region_id[n])]

    # --- Assemble final SurfaceRegionCandidate entities. ---
    torch = require_torch()
    final_regions: list[SurfaceRegionCandidate] = []
    for new_id in range(len(final_region_ids)):
        member_nodes = [n for n in range(count) if node_region_id[n] == new_id]
        if not member_nodes:
            continue
        core_nodes = [n for n in member_nodes if node_membership_state[n] == MEMBER_CORE]
        attached_nodes = [n for n in member_nodes if node_membership_state[n] == MEMBER_CONSENSUS_ATTACHED]

        internal_accepted: list[tuple[Any, Any]] = []
        internal_ambiguous: list[tuple[Any, Any]] = []
        boundary_conflict: list[tuple[Any, Any]] = []
        crease_count = 0
        parallel_count = 0
        rejected_relation_count = 0
        member_set = set(member_nodes)
        for (a, b), edge in by_pair.items():
            if a not in member_set or b not in member_set:
                continue
            id_pair = (ids[a], ids[b])
            if edge.manifold_relation == RELATION_SAME_SURFACE:
                if ((config.nonlocal_shortcut_mode == "force" or config.enable_nonlocal_shortcut_filter) and (edge.metrics is None or edge.metrics.normalized_distance > config.local_backbone_max_normalized_distance)) or (config.nonlocal_shortcut_mode == "auto" and config.enable_nonlocal_shortcut_filter and edge.metrics is not None and edge.metrics.normalized_distance > config.local_backbone_max_normalized_distance):
                    internal_ambiguous.append(id_pair)
                    continue
                if (a, b) in boundary_conflict_edges:
                    internal_ambiguous.append(id_pair)
                else:
                    internal_accepted.append(id_pair)
            elif edge.manifold_relation == RELATION_CREASE:
                crease_count += 1
                boundary_conflict.append(id_pair)
            elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
                parallel_count += 1
                boundary_conflict.append(id_pair)
            elif edge.manifold_relation == RELATION_REJECTED:
                rejected_relation_count += 1

        total_internal_relations = max(len(internal_accepted) + len(internal_ambiguous) + crease_count + parallel_count + rejected_relation_count, 1)
        contradiction_ratio = (crease_count + parallel_count + rejected_relation_count) / total_internal_relations

        # Disconnected-core-subgraph check (fragmentation signal).
        core_set = set(core_nodes) | set(attached_nodes)
        fragmented = False
        if len(core_set) > 1:
            start = next(iter(core_set))
            visited = {start}
            frontier = [start]
            while frontier:
                node = frontier.pop()
                for neighbor in same_surface[node]:
                    if neighbor in core_set and neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append(neighbor)
            fragmented = visited != core_set

        if contradiction_ratio > config.internal_contradiction_ratio_threshold and fragmented:
            internal_state = CONTRADICTION_UNRESOLVED_MIXED
        elif fragmented:
            internal_state = CONTRADICTION_POSSIBLE_FRAGMENTATION
        elif contradiction_ratio > config.internal_contradiction_ratio_threshold:
            internal_state = CONTRADICTION_POSSIBLE_OVER_MERGE
        else:
            internal_state = CONTRADICTION_STABLE

        intrinsic_reliable_count = sum(1 for n in member_nodes if intrinsic_class[n] == INTRINSIC_RELIABLE)
        intrinsic_stats = {
            "reliable_fraction": intrinsic_reliable_count / len(member_nodes),
            "member_count": len(member_nodes),
        }
        contextual_consistent_count = sum(1 for n in member_nodes if contextual_class[n] == CONTEXTUAL_CONSISTENT)
        contextual_mixed_count = sum(1 for n in member_nodes if contextual_class[n] == CONTEXTUAL_MIXED)
        contextual_stats = {
            "consistent_fraction": contextual_consistent_count / len(member_nodes),
            "mixed_fraction": contextual_mixed_count / len(member_nodes),
        }
        if core_set:
            mean_normal = torch.stack([frame.normal_candidate[n] for n in core_set]).mean(dim=0)
            mean_normal = torch.nn.functional.normalize(mean_normal, dim=0)
            spread = sum(
                1.0 - float((frame.normal_candidate[n] * mean_normal).sum().abs()) for n in core_set
            ) / len(core_set)
        else:
            spread = 0.0
        tangent_stats = {"mean_normal_spread": spread, "internal_state": internal_state}
        scale_values = [float(frame.tangent_major_scale[n]) for n in member_nodes]
        scale_stats = {
            "tangent_major_scale_mean": sum(scale_values) / len(scale_values),
            "tangent_major_scale_min": min(scale_values),
            "tangent_major_scale_max": max(scale_values),
        }

        if internal_state in (CONTRADICTION_UNRESOLVED_MIXED,):
            region_state = REGION_REVIEW_REQUIRED
        elif not core_nodes:
            region_state = REGION_REVIEW_REQUIRED
        elif len(member_nodes) < config.core_region_typical_min_size:
            region_state = REGION_SMALL_REVIEW
        elif attached_nodes:
            region_state = REGION_GROWING if len(attached_nodes) > len(core_nodes) // 2 + 1 else REGION_STABLE
        else:
            region_state = REGION_CORE if len(member_nodes) == len(core_nodes) else REGION_STABLE

        formation_reason = ["consensus_aware_core_seeding"]
        if attached_nodes:
            formation_reason.append("consensus_based_growth_attachment")
        unresolved_reasons: list[str] = []
        if internal_state != CONTRADICTION_STABLE:
            unresolved_reasons.append(internal_state)

        region_confidence = max(0.0, min(1.0, intrinsic_stats["reliable_fraction"] * (1.0 - contradiction_ratio)))

        final_regions.append(
            SurfaceRegionCandidate(
                region_id=new_id,
                member_ids=tuple(ids[n] for n in member_nodes),
                core_member_ids=tuple(ids[n] for n in core_nodes),
                attached_ambiguous_member_ids=tuple(ids[n] for n in attached_nodes),
                rejected_excluded_ids=tuple(
                    ids[n] for n in range(count)
                    if node_membership_state[n] == MEMBER_REJECTED and any(nb in member_set for nb in same_surface[n])
                ),
                internal_accepted_edge_ids=tuple(internal_accepted),
                internal_ambiguous_edge_ids=tuple(internal_ambiguous),
                boundary_conflict_edge_ids=tuple(boundary_conflict),
                intrinsic_reliability_stats=intrinsic_stats,
                contextual_consistency_stats=contextual_stats,
                tangent_frame_consistency_stats=tangent_stats,
                scale_stats=scale_stats,
                region_confidence=region_confidence,
                region_state=region_state,
                formation_reason=tuple(formation_reason),
                unresolved_reasons=tuple(unresolved_reasons),
                policy_version=POLICY_VERSION,
            )
        )

    return RegionFormationResult(
        node_region_id=tuple(node_region_id),
        node_membership_state=tuple(node_membership_state),
        unresolved_membership_ids=tuple(unresolved_membership_ids),
        regions=tuple(final_regions),
        config=config,
    )
