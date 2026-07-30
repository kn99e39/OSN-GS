from __future__ import annotations

"""Canonical experimental Gaussian-to-visible-NURBS construction path.

This module deliberately composes the covariance-guided evidence modules
without changing any builder, dispatcher, trainer, renderer, or checkpoint
path.  Its result retains every canonical intermediate for review.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates
from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame, extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import ManifoldAffinityConfig, ManifoldAffinityGraph, build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import StructuralReliabilityConfig, StructuralReliabilityResult, evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationConfig, RegionFormationResult, form_surface_regions
from osn_gs.surface.torch_ordered_world_boundary_graph import BoundaryCompatibilityEdge, OrderedBoundaryComponent, build_boundary_compatibility, recover_ordered_boundary_components
from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
from osn_gs.surface.torch_visible_boundary_materialization_adapter import VisibleBoundaryMaterializationResult, materialize_visible_boundary_component
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
    materialization_attempts: tuple[VisibleBoundaryMaterializationResult, ...]
    materialized_visible_nurbs_surfaces: tuple[VisibleBoundaryMaterializationResult, ...]
    review_results: tuple[VisibleBoundaryMaterializationResult, ...]
    diagnostic_summary: dict[str, Any]
    policy_schema_version: str
    coverage_semantics: str
    construction_state: str


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
    set (see ``torch_full_neighborhood_evidence.py``). This is the ONLY
    injection point; every other canonical stage (affinity, region formation,
    boundary recovery, materialization) is unchanged and still runs
    exclusively on ``positions``/``covariance``. Not exposed as a CLI
    selector -- callers pass this internally, production code always
    populates it with an evidence source, never a stand-in constructor.
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
    graph = build_manifold_affinity_graph(positions, frame, reliability, config=config.affinity, ids=ids)
    regions = form_surface_regions(positions, frame, reliability, graph, config=config.regions, ids=ids)
    accepted = tuple(sorted(
        (edge for region in regions.regions for edge in region.internal_accepted_edge_ids),
        key=lambda pair: (str(pair[0]), str(pair[1])),
    ))
    phase_alias = tuple(sorted(((edge.source_id, edge.target_id) for edge in graph.edges if edge.relation_reason == "phase_alias_or_shortcut_candidate"), key=lambda pair: (str(pair[0]), str(pair[1]))))
    oriented_normals = _orient_normals_along_accepted_topology(frame.normal_candidate, accepted, ids)
    canonical_frames = construct_canonical_region_tangent_frames(positions, frame, reliability, regions, ids=ids)
    relation_halfedges = extract_world_space_boundary_halfedge_candidates(positions, oriented_normals, regions, graph, ids=ids)
    termination_halfedges = extract_support_termination_candidates(positions, oriented_normals, frame.equivalent_tangent_scale, regions, ids=ids, sectors=config.support_termination_sectors, canonical_frames=canonical_frames)
    # Relation-derived candidates stay diagnostic-only: they must never close a loop.
    halfedges = tuple(sorted(termination_halfedges, key=lambda item: item.half_edge_id))
    compatibility = build_boundary_compatibility(halfedges)
    _, components = recover_directed_boundary_components(halfedges, accepted)

    id_to_index = {item: index for index, item in enumerate(ids)}
    attempts = []
    for component in components:
        boundary_ids = tuple(item for item in component.ordered_source_ids if item in id_to_index)
        if not boundary_ids:
            continue
        boundary_points = positions[torch.tensor([id_to_index[item] for item in boundary_ids], device=positions.device)]
        region = next((item for item in regions.regions if item.region_id == component.region_id), None)
        interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item not in set(boundary_ids) and item in id_to_index)
        if not interior_ids:
            interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item in id_to_index)
        interior_points = positions[torch.tensor([id_to_index[item] for item in interior_ids], device=positions.device)] if interior_ids else boundary_points
        attempts.append(materialize_visible_boundary_component(component, boundary_points, interior_points, boundary_ids=boundary_ids, interior_ids=interior_ids))
    attempts = tuple(attempts)
    materialized = tuple(item for item in attempts if item.state == "materialized")
    review = tuple(item for item in attempts if item.state != "materialized")
    summary = {
        "input_gaussian_count": count,
        "reliable_count": sum(item == "reliable_structural_evidence" for item in reliability.reliability_class),
        "ambiguous_count": sum(item == "ambiguous_structural_evidence" for item in reliability.reliability_class),
        "rejected_count": sum(item == "rejected_structural_evidence" for item in reliability.reliability_class),
        "region_count": len(regions.regions), "boundary_component_count": len(components),
        "admissible_component_count": sum(item.ordering_state == "ordered_closed_loop" and item.role_candidate == "outer_boundary_candidate" and not item.branch_node_ids for item in components),
        "materialization_attempt_count": len(attempts), "materialized_surface_count": len(materialized),
        "review_count": len(review), "unresolved_ambiguity_count": len(regions.unresolved_membership_ids),
        "source_region_to_surface": {item.input.source_region_id: item.input.adapter_id for item in materialized},
    }
    return VisibleSurfaceConstructionResult(frame, reliability, graph, regions, accepted, phase_alias, halfedges, compatibility, components, components, attempts, materialized, review, summary, config.schema_version, "reliable_core_only", _state(attempts, components, regions))

