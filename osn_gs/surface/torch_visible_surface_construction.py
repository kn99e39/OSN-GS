from __future__ import annotations

"""Canonical experimental Gaussian-to-visible-NURBS construction path.

This module deliberately composes the covariance-guided evidence modules
without changing any builder, dispatcher, trainer, renderer, or checkpoint
path.  Its result retains every canonical intermediate for review.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates, normalize_continuation_candidates
from osn_gs.surface.torch_termination_neighborhood_scale import resolve_termination_neighborhood_scale
from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput, build_continuation_shells_from_input
from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame, extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import ManifoldAffinityConfig, ManifoldAffinityGraph, build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import StructuralReliabilityConfig, StructuralReliabilityResult, evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationConfig, RegionFormationResult, form_surface_regions
from osn_gs.surface.torch_ordered_world_boundary_graph import BoundaryCompatibilityEdge, OrderedBoundaryComponent, build_boundary_compatibility, recover_ordered_boundary_components
from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
from osn_gs.surface.torch_visible_boundary_materialization_adapter import VisibleBoundaryMaterializationResult, materialize_visible_boundary_component
from osn_gs.surface.torch_visible_boundary_region_status import (
    STATUS_AMBIGUOUS, STATUS_ELIGIBLE_CLOSED, STATUS_INSUFFICIENT, STATUS_OPEN_FRAGMENT, STATUS_REJECTED_UNSAFE,
    RegionBoundaryStatus, classify_all_region_boundary_statuses, find_inconsistent_eligible_component_ids,
)
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate, extract_world_space_boundary_halfedge_candidates


SCHEMA_VERSION = "visible_surface_construction_worklog121_v1"


@dataclass(frozen=True)
class VisibleSurfaceConstructionConfig:
    reliability: StructuralReliabilityConfig = field(default_factory=StructuralReliabilityConfig)
    affinity: ManifoldAffinityConfig = field(default_factory=ManifoldAffinityConfig)
    regions: RegionFormationConfig = field(default_factory=RegionFormationConfig)
    support_termination_sectors: int = 8
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class VisibleSurfaceConstructionResult:
    covariance_frame: GaussianCovarianceFrame
    reliability: StructuralReliabilityResult
    manifold_affinity: ManifoldAffinityGraph
    surface_regions: RegionFormationResult
    accepted_local_topology: tuple[tuple[Any, Any], ...]
    nonlocal_phase_alias_diagnostics: tuple[tuple[Any, Any], ...]
    boundary_halfedge_candidates: tuple[WorldSpaceBoundaryHalfEdgeCandidate, ...]
    boundary_compatibility: tuple[BoundaryCompatibilityEdge, ...]
    ordered_boundary_components: tuple[OrderedBoundaryComponent, ...]
    boundary_role_candidates: tuple[OrderedBoundaryComponent, ...]
    region_boundary_statuses: tuple[RegionBoundaryStatus, ...]
    materialization_attempts: tuple[VisibleBoundaryMaterializationResult, ...]
    materialized_visible_nurbs_surfaces: tuple[VisibleBoundaryMaterializationResult, ...]
    review_results: tuple[VisibleBoundaryMaterializationResult, ...]
    diagnostic_summary: dict[str, Any]
    policy_schema_version: str
    coverage_semantics: str
    construction_state: str

    def eligible_materialized_surfaces(self) -> tuple[VisibleBoundaryMaterializationResult, ...]:
        """Worklog 55: the ONLY sanctioned source of visible-surface geometry
        for any downstream consumer -- occluded-chart/uncertain-Gaussian
        continuation (currently an isolated, non-production-wired
        foundation; see docs/worklogs/1-3), stream/PLY exporters, or any
        future integration. Identical to `materialized_visible_nurbs_surfaces`
        (already restricted to `eligible_closed_boundary` regions' approved
        components -- worklog 54), exposed under this name so downstream code
        has one clearly-labeled entry point instead of reading
        `ordered_boundary_components` / `boundary_role_candidates` directly,
        which include open/insufficient/ambiguous/rejected regions and must
        NEVER be a source of downstream geometry.
        """
        return self.materialized_visible_nurbs_surfaces


def _orient_normals_along_accepted_topology(normals: Any, accepted_edges: Sequence[tuple[Any, Any]], ids: Sequence[Any]):
    import torch
    oriented = torch.as_tensor(normals).clone(); index = {item: i for i, item in enumerate(ids)}
    adjacency = {i: [] for i in range(len(ids))}
    for a, b in accepted_edges:
        if a in index and b in index:
            adjacency[index[a]].append(index[b]); adjacency[index[b]].append(index[a])
    seen = set()
    for root in sorted(range(len(ids)), key=lambda item: str(ids[item])):
        if root in seen: continue
        stack = [root]; seen.add(root)
        while stack:
            source = stack.pop()
            for target in sorted(adjacency[source], key=lambda item: str(ids[item])):
                if target in seen: continue
                if float((oriented[source] * oriented[target]).sum()) < 0.0: oriented[target] = -oriented[target]
                seen.add(target); stack.append(target)
    return oriented
def _state(attempts: Sequence[VisibleBoundaryMaterializationResult], components: Sequence[OrderedBoundaryComponent], regions: RegionFormationResult) -> str:
    if any(item.state == "materialized" for item in attempts):
        if all(item.state == "materialized" for item in attempts):
            # Completed construction is only claimed for one unambiguous source
            # surface. Multiple separate regions remain explicitly reviewable.
            return "constructed" if len(regions.regions) == 1 else "review_required"
        return "partially_constructed"
    if not regions.regions:
        return "no_admissible_region"
    if not components:
        return "boundary_recovery_failed"
    if any(item.state == "fit_failed" for item in attempts):
        return "materialization_failed"
    if any(item.state == "validation_failed" for item in attempts):
        return "validation_failed"
    return "review_required"


def construct_visible_nurbs_from_gaussians(
    positions: Any,
    *,
    covariance: Any | None = None,
    log_scales: Any | None = None,
    rotations: Any | None = None,
    stable_ids: Sequence[Any] | None = None,
    config: VisibleSurfaceConstructionConfig | None = None,
    reliability: StructuralReliabilityResult | None = None,
    continuation_input: ContinuationShellInput | None = None,
    candidate_scale: Any | None = None,
    residual_scale: Any | None = None,
) -> VisibleSurfaceConstructionResult:
    """Run the one experimental path from Gaussian evidence to NURBS.

    Exactly one covariance representation is required: covariance, or the
    log-scale plus quaternion pair.  Unsupported topology is retained as a
    review result; it never receives a synthetic closure.

    ``reliability`` is an optional pre-computed override (worklog 129): when
    provided, the internal representative-only
    :func:`evaluate_structural_reliability` call is skipped and this result is
    used instead -- e.g. a caller that aggregated contextual evidence from the
    full observed Gaussian cloud rather than just this bounded representative
    set (see ``torch_full_neighborhood_evidence.py``). This is the reliability
    injection point; boundary recovery and materialization are unchanged and
    still run exclusively on ``positions``/``covariance`` UNLESS
    ``continuation_input`` is also supplied. Not exposed as a CLI selector --
    callers pass this internally, production code always populates it with an
    evidence source, never a stand-in constructor.

    ``continuation_input`` is a second, independent optional override
    (worklog 130): when provided, boundary-candidate generation
    (``extract_support_termination_candidates``) uses a continuous circular
    full-cloud support-gap query per node instead of the representative-only
    8-sector histogram -- see ``torch_full_cloud_continuation_shell.py``. Also
    not exposed as a CLI selector.

    ``candidate_scale``/``residual_scale`` are a third, independent optional
    override pair (worklog 33): passed straight through to
    :func:`build_manifold_affinity_graph`. ``None`` preserves the prior
    ``frame.tangent_major_scale``-based behavior exactly.
    """
    import torch
    from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation

    config = config or VisibleSurfaceConstructionConfig()
    positions = torch.as_tensor(positions)
    count = int(positions.shape[0])
    ids = tuple(range(count)) if stable_ids is None else tuple(stable_ids)
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("stable_ids must be unique and match positions.")
    if covariance is None:
        if log_scales is None or rotations is None:
            raise ValueError("provide covariance or both log_scales and rotations.")
        covariance = covariance_from_scale_rotation(torch.exp(torch.as_tensor(log_scales)), rotations)

    frame = extract_covariance_frame(covariance)
    if reliability is None:
        reliability = evaluate_structural_reliability(positions, frame, config=config.reliability)
    graph = build_manifold_affinity_graph(
        positions, frame, reliability, config=config.affinity, ids=ids,
        candidate_scale=candidate_scale, residual_scale=residual_scale,
    )
    regions = form_surface_regions(positions, frame, reliability, graph, config=config.regions, ids=ids)
    accepted = tuple(sorted(
        (edge for region in regions.regions for edge in region.internal_accepted_edge_ids),
        key=lambda pair: (str(pair[0]), str(pair[1])),
    ))
    phase_alias = tuple(sorted(((edge.source_id, edge.target_id) for edge in graph.edges if edge.relation_reason == "phase_alias_or_shortcut_candidate"), key=lambda pair: (str(pair[0]), str(pair[1]))))
    oriented_normals = _orient_normals_along_accepted_topology(frame.normal_candidate, accepted, ids)
    canonical_frames = construct_canonical_region_tangent_frames(positions, frame, reliability, regions, ids=ids)
    relation_halfedges = extract_world_space_boundary_halfedge_candidates(positions, oriented_normals, regions, graph, ids=ids)
    continuation = (
        build_continuation_shells_from_input(continuation_input, positions, frame, ids, regions, canonical_frames)
        if continuation_input is not None
        else None
    )
    # Worklog 41 (task section 13, Case C): the local-support radius here must
    # be measured in the same scale that governs which accepted edges can
    # exist in the first place -- the resolved representative graph scale
    # (worklog 33's candidate_scale, defaulting to frame.tangent_major_scale
    # exactly as build_manifold_affinity_graph resolves it), not the raw
    # per-Gaussian footprint (frame.equivalent_tangent_scale). On real
    # checkpoints representative spacing is far larger than an individual
    # Gaussian's own footprint (measured median ratio ~7x on a 3k snapshot),
    # so the footprint-scaled radius excluded almost every genuine accepted
    # neighbour and starved the gate of local support before it ever reached
    # the angular-histogram logic. Multiplier (4.0) is unchanged.
    resolved_candidate_scale = resolve_termination_neighborhood_scale(
        candidate_scale=candidate_scale, tangent_major_scale=frame.tangent_major_scale,
    )
    termination_halfedges = extract_support_termination_candidates(positions, oriented_normals, resolved_candidate_scale, regions, ids=ids, sectors=config.support_termination_sectors, canonical_frames=canonical_frames, continuation=continuation, affinity_graph=graph)
    termination_halfedges = normalize_continuation_candidates(termination_halfedges)
    # Relation-derived candidates stay diagnostic-only: they must never close a loop.
    halfedges = tuple(sorted(termination_halfedges, key=lambda item: item.half_edge_id))
    compatibility = build_boundary_compatibility(halfedges)
    _, components = recover_directed_boundary_components(halfedges, accepted)

    # Worklog 54: the region-status contract is the SOLE gate deciding which
    # component reaches materialization -- a validated, safety-checked simple
    # closed loop, and nothing else. Every other region is recorded with a
    # stable typed reason (`region_boundary_statuses`) instead of silently
    # producing an empty or failed materialization attempt.
    region_statuses = classify_all_region_boundary_statuses(
        tuple(region.region_id for region in regions.regions), components, halfedges,
    )
    # Worklog 55: fail-closed consistency check -- an `eligible_component_id`
    # that does not correspond to an actual `ordered_closed_loop` component is
    # never trusted implicitly. This can only fire on a future refactor bug
    # (the classifier derives IDs directly from `components`); when it does,
    # the affected ID is excluded from materialization rather than silently
    # materialized, and the count is surfaced in diagnostics.
    inconsistent_component_ids = find_inconsistent_eligible_component_ids(region_statuses, components)
    eligible_component_ids = frozenset(
        component_id for status in region_statuses for component_id in status.eligible_component_ids
    ) - frozenset(inconsistent_component_ids)
    status_by_eligible_component_id = {
        component_id: status for status in region_statuses for component_id in status.eligible_component_ids
    }

    id_to_index = {item: index for index, item in enumerate(ids)}
    attempts = []
    for component in components:
        if component.component_id not in eligible_component_ids:
            continue
        boundary_ids = tuple(item for item in component.ordered_source_ids if item in id_to_index)
        if not boundary_ids:
            continue
        boundary_points = positions[torch.tensor([id_to_index[item] for item in boundary_ids], device=positions.device)]
        region = next((item for item in regions.regions if item.region_id == component.region_id), None)
        interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item not in set(boundary_ids) and item in id_to_index)
        if not interior_ids:
            interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item in id_to_index)
        interior_points = positions[torch.tensor([id_to_index[item] for item in interior_ids], device=positions.device)] if interior_ids else boundary_points
        region_status = status_by_eligible_component_id.get(component.component_id)
        attempts.append(materialize_visible_boundary_component(
            component, boundary_points, interior_points, boundary_ids=boundary_ids, interior_ids=interior_ids,
            region_status=region_status.status if region_status else "",
            region_status_reason=region_status.reason if region_status else "",
            boundary_role_scope=region_status.boundary_role_scope if region_status else "",
            supporting_source_ids=region_status.supporting_source_ids if region_status else (),
        ))
    attempts = tuple(attempts)
    materialized = tuple(item for item in attempts if item.state == "materialized")
    review = tuple(item for item in attempts if item.state != "materialized")

    # Boundary-pipeline yield, stage by stage (worklog 130 item 2) -- never
    # collapse straight to a single boundary_component_count=0 number.
    genuine_candidates = sum(item.boundary_reason == "observed_support_termination" for item in halfedges)
    reliability_frontier_candidates = sum(item.reliability_frontier for item in halfedges)
    sampling_gap_candidates = sum(item.sampling_gap for item in halfedges)
    crease_candidates = sum(item.boundary_reason == "crease_discontinuity" for item in halfedges)
    parallel_conflict_candidates = sum(item.boundary_reason == "parallel_sheet_conflict" for item in halfedges)
    ambiguous_candidates = sum(item.boundary_reason == "ambiguous_continuation" for item in halfedges)
    # Worklog 41 (task section 6): a candidate reclassified because the
    # surface demonstrably CONTINUES into a neighbouring region is not a
    # reliability problem, so it gets its own typed count instead of being
    # folded into `reliability_frontier` (which means "support exists but is
    # too ambiguous to trust"). Keeps candidate accounting exact.
    smooth_cross_region_candidates = sum(
        item.boundary_reason == "smooth_cross_region_continuation" for item in halfedges
    )
    closed_components = sum(item.ordering_state == "ordered_closed_loop" for item in components)
    open_components = sum(item.ordering_state == "ordered_open_chain" for item in components)
    branching_components = sum(item.ordering_state == "branching_boundary_graph" for item in components)
    ambiguous_components = sum(item.ordering_state == "ambiguous_ordering" for item in components)
    isolated_components = sum(item.ordering_state == "isolated_boundary_candidate" for item in components)

    # A/B/C failure-stage classification -- distinct from `_state()` below,
    # which describes the OVERALL construction outcome; this specifically
    # locates where the boundary pipeline stopped producing yield.
    if genuine_candidates == 0:
        boundary_failure_stage = "A_candidate_generation_failed"
    elif not components:
        boundary_failure_stage = "B_candidate_linking_failed"
    elif closed_components == 0:
        boundary_failure_stage = "C_component_admission_failed"
    else:
        boundary_failure_stage = "not_failed"

    # Reliability-stage classification (worklog 135) -- distinguishes WHY the
    # final reliable_count is low without collapsing every upstream cause
    # into the single downstream `no_admissible_region` construction_state.
    # Additive-only: does not change `_state()`/`construction_state`.
    final_reliable_count = sum(item == "reliable_structural_evidence" for item in reliability.reliability_class)
    intrinsic_reliable_count = sum(item == "intrinsic_reliable" for item in reliability.intrinsic.intrinsic_class)
    if intrinsic_reliable_count == 0:
        reliability_failure_stage = "intrinsic_reliability_collapse"
    elif final_reliable_count == 0:
        reliability_failure_stage = "contextual_reliability_collapse"
    elif final_reliable_count < intrinsic_reliable_count:
        reliability_failure_stage = "partial_contextual_reliability_collapse"
    else:
        reliability_failure_stage = "not_failed"

    summary = {
        "input_gaussian_count": count,
        "reliable_count": final_reliable_count,
        "ambiguous_count": sum(item == "ambiguous_structural_evidence" for item in reliability.reliability_class),
        "rejected_count": sum(item == "rejected_structural_evidence" for item in reliability.reliability_class),
        "intrinsic_reliable_count": intrinsic_reliable_count,
        "intrinsic_ambiguous_count": sum(item == "intrinsic_ambiguous" for item in reliability.intrinsic.intrinsic_class),
        "intrinsic_rejected_count": sum(item == "intrinsic_rejected" for item in reliability.intrinsic.intrinsic_class),
        "reliability_failure_stage": reliability_failure_stage,
        "region_count": len(regions.regions), "boundary_component_count": len(components),
        "admissible_component_count": sum(item.ordering_state == "ordered_closed_loop" and item.role_candidate == "outer_boundary_candidate" and not item.branch_node_ids for item in components),
        "materialization_attempt_count": len(attempts), "materialized_surface_count": len(materialized),
        "review_count": len(review), "unresolved_ambiguity_count": len(regions.unresolved_membership_ids),
        "source_region_to_surface": {item.input.source_region_id: item.input.adapter_id for item in materialized},
        "boundary_failure_stage": boundary_failure_stage,
        "boundary_candidate_count": len(halfedges),
        "boundary_genuine_termination_candidate_count": genuine_candidates,
        "boundary_reliability_frontier_candidate_count": reliability_frontier_candidates,
        "boundary_smooth_cross_region_candidate_count": smooth_cross_region_candidates,
        "boundary_sampling_gap_candidate_count": sampling_gap_candidates,
        "boundary_crease_candidate_count": crease_candidates,
        "boundary_parallel_conflict_candidate_count": parallel_conflict_candidates,
        "boundary_ambiguous_candidate_count": ambiguous_candidates,
        "boundary_component_closed_count": closed_components,
        "boundary_component_open_count": open_components,
        "boundary_component_branching_count": branching_components,
        "boundary_component_ambiguous_count": ambiguous_components,
        "boundary_component_isolated_count": isolated_components,
        # Worklog 54: the region-status production contract, region-by-region.
        "region_boundary_status_count": len(region_statuses),
        "region_boundary_eligible_closed_count": sum(item.status == STATUS_ELIGIBLE_CLOSED for item in region_statuses),
        "region_boundary_open_fragment_count": sum(item.status == STATUS_OPEN_FRAGMENT for item in region_statuses),
        "region_boundary_insufficient_observation_count": sum(item.status == STATUS_INSUFFICIENT for item in region_statuses),
        "region_boundary_ambiguous_count": sum(item.status == STATUS_AMBIGUOUS for item in region_statuses),
        "region_boundary_rejected_unsafe_count": sum(item.status == STATUS_REJECTED_UNSAFE for item in region_statuses),
        "region_boundary_multiple_closed_loops_count": sum(len(item.eligible_component_ids) > 1 for item in region_statuses),
        "region_boundary_status_inconsistency_count": len(inconsistent_component_ids),
        "region_boundary_statuses": [item.payload() for item in region_statuses],
    }
    return VisibleSurfaceConstructionResult(frame, reliability, graph, regions, accepted, phase_alias, halfedges, compatibility, components, components, region_statuses, attempts, materialized, review, summary, config.schema_version, "reliable_core_only", _state(attempts, components, regions))



