from __future__ import annotations

"""Review-only ordered-boundary to evaluable visible NURBS adapter."""

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class VisibleBoundaryMaterializationResult:
    input: VisibleBoundaryMaterializationInput
    surface: TorchNURBSSurface | None
    state: str
    boundary_residual: float | None
    interior_residual: float | None
    review_reasons: tuple[str, ...]


def materialize_visible_boundary_component(component: OrderedBoundaryComponent, boundary_points: Any, interior_points: Any, *, boundary_ids: tuple[Any, ...], interior_ids: tuple[Any, ...]) -> VisibleBoundaryMaterializationResult:
    """Materialize only a single, non-branching observed outer loop.

    The existing canonical TorchNURBSSurface is returned and remains directly
    evaluable.  Open/branch/ambiguous components never receive a synthetic
    rectangular closure.
    """
    admissible = component.ordering_state == "ordered_closed_loop" and component.role_candidate == "outer_boundary_candidate" and not component.branch_node_ids
    state = "materialized" if admissible else "unsupported_topology"
    adapter = VisibleBoundaryMaterializationInput(
        adapter_id=f"adapter:{component.component_id}", source_region_id=component.region_id,
        source_boundary_component_id=component.component_id, ordered_boundary_point_ids=boundary_ids,
        ordered_boundary_points=boundary_points, interior_reliable_point_ids=interior_ids,
        interior_points=interior_points, coverage_semantics="reliable_core_only", materialization_state=state,
        reasons=() if admissible else (component.ordering_state, component.role_candidate),
    )
    if not admissible:
        return VisibleBoundaryMaterializationResult(adapter, None, state, None, None, adapter.reasons)
    try:
        import torch
        observed = torch.cat((boundary_points, interior_points), dim=0)
        surface, _ = fit_torch_visible_surface_lsq(observed, resolution_u=6, resolution_v=6, degree_u=2, degree_v=2)
        sampled = surface.evaluate(torch.tensor([[0.5, 0.5]], dtype=observed.dtype, device=observed.device))
        if not torch.isfinite(sampled).all():
            return VisibleBoundaryMaterializationResult(adapter, None, "validation_failed", None, None, ("non_finite_evaluate",))
        residual = float(torch.cdist(interior_points, sampled.reshape(1, 3)).min(dim=1).values.mean())
        return VisibleBoundaryMaterializationResult(adapter, surface, "materialized", None, residual, ())
    except Exception as exc:
        return VisibleBoundaryMaterializationResult(adapter, None, "fit_failed", None, None, (type(exc).__name__,))
