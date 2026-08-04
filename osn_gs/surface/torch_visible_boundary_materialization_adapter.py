from __future__ import annotations

"""Review-only ordered-boundary to evaluable visible NURBS adapter."""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_nurbs import TorchNURBSSurface, fit_torch_visible_surface_lsq
from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent


@dataclass(frozen=True)
class VisibleBoundaryMaterializationInput:
    adapter_id: str
    source_region_id: int
    source_boundary_component_id: str
    ordered_boundary_point_ids: tuple[Any, ...]
    ordered_boundary_points: Any
    interior_reliable_point_ids: tuple[Any, ...]
    interior_points: Any
    coverage_semantics: str
    materialization_state: str
    reasons: tuple[str, ...]
    # Worklog 55: region-level boundary-status provenance (see
    # torch_visible_boundary_region_status.py), threaded through so any
    # downstream consumer of a materialized result has the eligibility
    # status/reason/scope directly, without re-deriving it or cross-
    # referencing a separate `region_boundary_statuses` list. Defaults keep
    # every pre-existing caller (tests, devtools scripts) unaffected.
    region_status: str = ""
    region_status_reason: str = ""
    boundary_role_scope: str = ""
    supporting_source_ids: tuple[Any, ...] = ()


@dataclass(frozen=True)
class VisibleBoundaryMaterializationResult:
    input: VisibleBoundaryMaterializationInput
    surface: TorchNURBSSurface | None
    state: str
    boundary_residual: float | None
    interior_residual: float | None
    review_reasons: tuple[str, ...]


def materialize_visible_boundary_component(
    component: OrderedBoundaryComponent, boundary_points: Any, interior_points: Any, *,
    boundary_ids: tuple[Any, ...], interior_ids: tuple[Any, ...],
    region_status: str = "", region_status_reason: str = "", boundary_role_scope: str = "",
    supporting_source_ids: tuple[Any, ...] = (),
) -> VisibleBoundaryMaterializationResult:
    """Materialize only a single, non-branching observed outer loop.

    The existing canonical TorchNURBSSurface is returned and remains directly
    evaluable.  Open/branch/ambiguous components never receive a synthetic
    rectangular closure.

    ``region_status``/``region_status_reason``/``boundary_role_scope``/
    ``supporting_source_ids`` (worklog 55, all optional) are pure provenance
    pass-through from the caller's region-status classification -- this
    function's own admissibility/self-intersection gate is unchanged by them.
    """
    admissible = component.ordering_state == "ordered_closed_loop" and component.role_candidate == "outer_boundary_candidate" and not component.branch_node_ids
    self_intersection_reasons: tuple[str, ...] = ()
    if admissible:
        # Worklog 36 (task section 9): total turning angle and planar
        # z-standard-deviation alone cannot prove a source loop is a simple
        # (non-self-intersecting) polygon -- explicit non-adjacent segment
        # crossing validation, projected to the loop's own local tangent
        # plane. A proper self-intersection, non-adjacent collinear overlap,
        # repeated vertex, or zero-area cycle fails closed here, before any
        # NURBS fitting is attempted.
        world_points = [tuple(float(v) for v in row) for row in boundary_points.detach().cpu().tolist()]
        report = validate_simple_closed_loop(world_points)
        if not report.is_simple_polygon:
            admissible = False
            self_intersection_reasons = ("self_intersection_check_failed",) + report.reasons
    state = "materialized" if admissible else "unsupported_topology"
    adapter = VisibleBoundaryMaterializationInput(
        adapter_id=f"adapter:{component.component_id}", source_region_id=component.region_id,
        source_boundary_component_id=component.component_id, ordered_boundary_point_ids=boundary_ids,
        ordered_boundary_points=boundary_points, interior_reliable_point_ids=interior_ids,
        interior_points=interior_points, coverage_semantics="reliable_core_only", materialization_state=state,
        reasons=self_intersection_reasons if self_intersection_reasons else (() if admissible else (component.ordering_state, component.role_candidate)),
        region_status=region_status, region_status_reason=region_status_reason,
        boundary_role_scope=boundary_role_scope, supporting_source_ids=supporting_source_ids,
    )
    if not admissible:
        return VisibleBoundaryMaterializationResult(adapter, None, state, None, None, adapter.reasons)
    try:
        import torch
        observed = torch.cat((boundary_points, interior_points), dim=0)
        surface, _ = fit_torch_visible_surface_lsq(observed, resolution_u=6, resolution_v=6, degree_u=2, degree_v=2)
        grid = torch.linspace(0.0, 1.0, 9, dtype=observed.dtype, device=observed.device)
        sample_u, sample_v = torch.meshgrid(grid, grid, indexing="ij")
        sampled = surface.evaluate(torch.stack((sample_u.reshape(-1), sample_v.reshape(-1)), dim=1))
        if not torch.isfinite(sampled).all():
            return VisibleBoundaryMaterializationResult(adapter, None, "validation_failed", None, None, ("non_finite_evaluate",))
        boundary_residual = float(torch.cdist(boundary_points, sampled).min(dim=1).values.mean())
        interior_residual = float(torch.cdist(interior_points, sampled).min(dim=1).values.mean())
        return VisibleBoundaryMaterializationResult(adapter, surface, "materialized", boundary_residual, interior_residual, ())
    except Exception as exc:
        return VisibleBoundaryMaterializationResult(adapter, None, "fit_failed", None, None, (type(exc).__name__,))
