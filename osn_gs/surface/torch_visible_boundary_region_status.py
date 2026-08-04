from __future__ import annotations

"""Region-level visible-boundary eligibility contract (worklog 54).

Worklog 47-53 hardened the candidate/classification/ordering pipeline through
six independent audit rounds and established that the remaining real-checkpoint
closed-loop gap is candidate-evidence density, not a pipeline defect (see
``docs/Urgent_Work/OSN_GS_Urgent_Work_Master.md`` section 2). This module does
not change any of that machinery -- candidate/continuation thresholds, the
matching objective, gap interpolation, and NURBS fitting are all untouched.

What it adds: a single, explicit PRODUCTION CONTRACT classifying every region
into exactly one of five states, so that only a validated safe closed loop is
ever handed to materialization, and every other region carries a stable,
typed reason instead of silently becoming an empty/failed result.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate

SCHEMA_VERSION = "visible_boundary_region_status_worklog55_v2"

STATUS_ELIGIBLE_CLOSED = "eligible_closed_boundary"
STATUS_OPEN_FRAGMENT = "open_observed_fragment"
STATUS_INSUFFICIENT = "insufficient_observation"
STATUS_AMBIGUOUS = "ambiguous_boundary"
STATUS_REJECTED_UNSAFE = "rejected_unsafe"

REASON_NO_BOUNDARY_EVIDENCE = "no_boundary_evidence_in_region"
REASON_NO_PHYSICAL_CANDIDATES = "no_physical_termination_candidates_only_typed_nonphysical_evidence"
REASON_ONLY_ISOLATED = "only_isolated_physical_candidates_no_linked_fragment"
REASON_OPEN_FRAGMENT = "best_available_downstream_valid_open_path"
REASON_VALIDATED_CLOSED = "validated_simple_closed_loop"
REASON_SELF_INTERSECTION = "closed_loop_failed_self_intersection_check"
REASON_CAPACITY_EXCEEDED = "region_candidate_count_exceeds_exact_matching_capacity"
REASON_BUDGET_EXHAUSTED = "two_cycle_branch_budget_exhausted"
# Worklog 55: this pipeline has never implemented inner/hole-loop detection
# (`_recover_directed_boundary_components` hardcodes every closed loop's
# `role_candidate` to `outer_boundary_candidate` -- see
# `torch_directed_boundary_ordering.py`). A region CAN legitimately carry more
# than one independently-validated closed loop (measured: thin_slab's
# front+back sharing one region) with no evidence-based way to tell an outer
# perimeter from an interior hole between them. Excluding the extras
# regressed real materialization (thin_slab 3->2) with no principled way to
# pick which one is "the" outer boundary, so every validated loop stays
# eligible and is materialized independently, exactly as before this
# contract existed -- this reason only makes that ambiguity EXPLICIT in
# diagnostics instead of leaving it implicit.
REASON_MULTIPLE_CLOSED_LOOPS_OUTER_ONLY = "multiple_closed_loops_found_outer_boundary_only_scope_no_hole_merge_attempted"

BOUNDARY_ROLE_SCOPE_OUTER_ONLY = "outer_boundary_only"


@dataclass(frozen=True)
class RegionBoundaryStatus:
    """One authoritative status per region -- the production contract this
    module exists to establish. ``eligible_component_ids`` is the ONLY
    channel materialization may read component IDs from; every other status
    means "do not materialize", never "silently empty".

    ``boundary_role_scope`` is always ``BOUNDARY_ROLE_SCOPE_OUTER_ONLY``:
    this contract has no inner/hole-boundary classification and must never be
    read as implying one (worklog 55).
    """

    region_id: int
    status: str
    reason: str
    candidate_count: int
    component_ordering_states: tuple[str, ...]
    supporting_source_ids: tuple[Any, ...]
    eligible_component_ids: tuple[str, ...]
    boundary_role_scope: str = BOUNDARY_ROLE_SCOPE_OUTER_ONLY
    schema_version: str = SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "status": self.status,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "component_ordering_states": list(self.component_ordering_states),
            "supporting_source_ids": list(self.supporting_source_ids),
            "eligible_component_ids": list(self.eligible_component_ids),
            "boundary_role_scope": self.boundary_role_scope,
        }


def _is_validated_closed(component: OrderedBoundaryComponent, world_position_by_id: Mapping[Any, tuple[float, float, float]]) -> bool:
    if component.ordering_state != "ordered_closed_loop":
        return False
    if component.branch_node_ids:
        return False
    if component.role_candidate != "outer_boundary_candidate":
        return False
    points = [world_position_by_id[half_edge_id] for half_edge_id in component.ordered_half_edge_ids]
    return validate_simple_closed_loop(points).is_simple_polygon


def classify_region_boundary_status(
    region_id: int,
    region_components: Sequence[OrderedBoundaryComponent],
    region_candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    world_position_by_id: Mapping[Any, tuple[float, float, float]],
) -> RegionBoundaryStatus:
    """Classify one region into exactly one of the five contract states.

    ``region_components`` are this region's :class:`OrderedBoundaryComponent`
    entries (closed loops, open fragments, isolated candidates, or a capacity/
    budget fail-closed marker). ``region_candidates`` are ALL boundary
    half-edge candidates generated for this region regardless of type (worklog
    47-50's typed states: ``observed_support_termination``,
    ``parallel_sheet_conflict``, ``crease_discontinuity``,
    ``ambiguous_continuation``, ``reliability_frontier``,
    ``unresolved_sampling_gap``, ``smooth_cross_region_continuation``) -- used
    to distinguish "no evidence at all" from "evidence exists but never became
    a physical candidate".
    """
    component_states = tuple(component.ordering_state for component in region_components)
    supporting_source_ids = tuple(sorted(
        {source_id for component in region_components for source_id in component.ordered_source_ids},
        key=str,
    ))
    physical_candidate_count = sum(
        1 for candidate in region_candidates if candidate.boundary_reason == "observed_support_termination"
    )

    if not region_candidates:
        return RegionBoundaryStatus(
            region_id, STATUS_INSUFFICIENT, REASON_NO_BOUNDARY_EVIDENCE, 0, (), (), (),
        )

    # Worklog 54: a budget-exhausted or capacity-exceeded region can never be
    # eligible, regardless of what `ordering_state` its components carry --
    # the safety/optimality exploration for it did not run to completion.
    if any(
        "two_cycle_branch_budget_exhausted" in component.unresolved_reasons
        for component in region_components
    ):
        return RegionBoundaryStatus(
            region_id, STATUS_REJECTED_UNSAFE, REASON_BUDGET_EXHAUSTED, physical_candidate_count,
            component_states, supporting_source_ids, (),
        )
    if any(component.ordering_state == "ordering_capacity_exceeded" for component in region_components):
        return RegionBoundaryStatus(
            region_id, STATUS_REJECTED_UNSAFE, REASON_CAPACITY_EXCEEDED, physical_candidate_count,
            component_states, supporting_source_ids, (),
        )

    if physical_candidate_count == 0:
        # Real, typed evidence exists (crease/parallel/ambiguous/frontier/
        # sampling-gap/cross-region) but none of it ever became a physical
        # termination candidate -- distinct from having no evidence at all.
        return RegionBoundaryStatus(
            region_id, STATUS_AMBIGUOUS, REASON_NO_PHYSICAL_CANDIDATES, 0, component_states, supporting_source_ids, (),
        )

    validated_closed = [
        component for component in region_components
        if _is_validated_closed(component, world_position_by_id)
    ]
    if validated_closed:
        # Worklog 55: no inner/hole detection exists in this pipeline (every
        # closed loop's `role_candidate` is hardcoded to
        # `outer_boundary_candidate` -- there is no evidence-based signal to
        # tell an outer perimeter from an interior hole). Rather than
        # invent one now (out of scope) or silently drop extra loops
        # (regressed real materialization -- box/thin_slab fixtures proved
        # a region CAN legitimately carry more than one independent closed
        # loop, e.g. thin_slab's front+back sharing one region), every
        # validated closed loop is kept eligible and materialized
        # independently, exactly as before this contract existed. When more
        # than one exists, the reason is explicitly flagged so nothing about
        # "which is the outer boundary" is claimed or hidden.
        eligible_component_ids = tuple(sorted(component.component_id for component in validated_closed))
        reason = REASON_VALIDATED_CLOSED if len(validated_closed) == 1 else REASON_MULTIPLE_CLOSED_LOOPS_OUTER_ONLY
        return RegionBoundaryStatus(
            region_id, STATUS_ELIGIBLE_CLOSED, reason, physical_candidate_count,
            component_states, supporting_source_ids, eligible_component_ids,
        )

    unsafe_closed = any(
        component.ordering_state == "ordered_closed_loop" and not _is_validated_closed(component, world_position_by_id)
        for component in region_components
    )
    if unsafe_closed:
        return RegionBoundaryStatus(
            region_id, STATUS_REJECTED_UNSAFE, REASON_SELF_INTERSECTION, physical_candidate_count,
            component_states, supporting_source_ids, (),
        )

    if any(component.ordering_state == "ambiguous_ordering" for component in region_components):
        return RegionBoundaryStatus(
            region_id, STATUS_OPEN_FRAGMENT, REASON_OPEN_FRAGMENT, physical_candidate_count,
            component_states, supporting_source_ids, (),
        )

    # Only isolated_boundary_candidate components remain: real physical
    # candidates exist but never linked into even one open fragment.
    return RegionBoundaryStatus(
        region_id, STATUS_INSUFFICIENT, REASON_ONLY_ISOLATED, physical_candidate_count,
        component_states, supporting_source_ids, (),
    )


def find_inconsistent_eligible_component_ids(
    statuses: Sequence[RegionBoundaryStatus],
    components: Sequence[OrderedBoundaryComponent],
) -> tuple[str, ...]:
    """Worklog 55: defensive fail-closed check. Every ``eligible_component_id``
    a status claims MUST exist among ``components`` with ``ordering_state ==
    "ordered_closed_loop"``. This can never fire if
    :func:`classify_region_boundary_status` is internally correct (component
    IDs are derived directly from ``components`` in the same call), but the
    downstream materialization gate must not blindly trust that invariant --
    this makes any future drift or refactor bug LOUD (an explicit mismatch
    count) instead of silently materializing something never actually
    validated as a closed loop.
    """
    valid_closed_ids = {
        component.component_id for component in components
        if component.ordering_state == "ordered_closed_loop"
    }
    inconsistent = {
        component_id
        for status in statuses
        for component_id in status.eligible_component_ids
        if component_id not in valid_closed_ids
    }
    return tuple(sorted(inconsistent))


def classify_all_region_boundary_statuses(
    region_ids: Sequence[int],
    components: Sequence[OrderedBoundaryComponent],
    candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
) -> tuple[RegionBoundaryStatus, ...]:
    """Classify every region in ``region_ids`` (deterministic, sorted order)."""
    world_position_by_id = {candidate.half_edge_id: candidate.world_position for candidate in candidates}
    components_by_region: dict[int, list[OrderedBoundaryComponent]] = {}
    for component in components:
        components_by_region.setdefault(component.region_id, []).append(component)
    candidates_by_region: dict[int, list[WorldSpaceBoundaryHalfEdgeCandidate]] = {}
    for candidate in candidates:
        candidates_by_region.setdefault(candidate.source_region_id, []).append(candidate)

    return tuple(
        classify_region_boundary_status(
            region_id,
            components_by_region.get(region_id, ()),
            candidates_by_region.get(region_id, ()),
            world_position_by_id,
        )
        for region_id in sorted(region_ids)
    )
