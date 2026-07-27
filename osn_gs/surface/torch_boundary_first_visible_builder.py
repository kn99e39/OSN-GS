from __future__ import annotations

"""Observed-boundary-first visible construction with no box fallback.

The builder derives explicit boundary roles from evidence, then materializes the
same Boundary-first contract.  Loop counts are evidence decoding only; they
never choose a separate surface-construction methodology.
"""
from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_boundary_central_cap import (
    build_boundary_central_cap,
    select_observed_interior_anchor,
)
from osn_gs.surface.torch_boundary_constrained_surface import (
    BoundaryConstrainedSurfaceResult,
    build_boundary_constrained_surface,
)
from osn_gs.surface.torch_boundary_multi_loop import assess_multi_loop_correspondence
from osn_gs.surface.torch_boundary_review_geometry import observed_evidence_entity
from osn_gs.surface.torch_boundary_role_evidence import BoundaryRoleEvidence, boundary_role_evidence
from osn_gs.surface.torch_boundary_source_fidelity import measure_observed_boundary_source_fidelity
from osn_gs.surface.torch_boundary_support_network import (
    build_boundary_support_curve_network,
    observed_boundary_curves_from_annulus_component,
    observed_outer_boundary_curve_from_component,
)


def _anchor_ray_support_coverage(
    boundary_result: Any,
    outer_curve: Any,
    anchor: Any,
    samples: int = 12,
    support_tolerance_cells: int = 1,
) -> float:
    """Reject interior anchors whose rays leave observed support beyond raster tolerance."""
    from osn_gs.utils.torch_ops import require_torch

    torch = require_torch()
    mask = torch.as_tensor(getattr(boundary_result, "refined_mask"), dtype=torch.bool)
    if not isinstance(support_tolerance_cells, int) or support_tolerance_cells < 0:
        raise ValueError("support_tolerance_cells must be a non-negative integer.")
    if support_tolerance_cells:
        tolerance = support_tolerance_cells
        mask = torch.nn.functional.max_pool2d(
            mask.float()[None, None], kernel_size=2 * tolerance + 1, stride=1, padding=tolerance
        )[0, 0] > 0.5
    frame = getattr(boundary_result, "frame")
    boundary = torch.as_tensor(outer_curve.world, dtype=anchor.point.dtype, device=anchor.point.device)
    if boundary.shape[0] > 1 and bool(torch.allclose(boundary[0], boundary[-1])):
        boundary = boundary[:-1]
    steps = torch.linspace(0.05, 0.95, samples, dtype=boundary.dtype, device=boundary.device)
    rays = anchor.point[None, None, :] * (1.0 - steps[None, :, None]) + boundary[:, None, :] * steps[None, :, None]
    uv = frame.apply(rays.reshape(-1, 3))
    h, w = int(mask.shape[0]), int(mask.shape[1])
    cells_u = torch.clamp((uv[:, 0] * h).long(), 0, h - 1)
    cells_v = torch.clamp((uv[:, 1] * w).long(), 0, w - 1)
    return float(mask[cells_u, cells_v].float().mean())


@dataclass(frozen=True)
class BoundaryFirstVisibleSurfaceResult:
    state: str
    topology: str
    reason: str | None
    surface_result: BoundaryConstrainedSurfaceResult | None
    provenance: dict[str, Any]

    @property
    def materialization_state(self) -> str:
        return "materialized" if self.surface_result is not None else "not_materialized"

    @property
    def quality_state(self) -> str:
        if self.state == "unsupported":
            return "unsupported"
        if self.state == "review_required":
            return "review_required"
        return "review_required"


def _observed_boundary_entity_payload(curve: Any, *, role: str) -> dict[str, Any]:
    """Provenance-bearing observed-evidence entity (never a control/evaluated-curve claim)."""
    provenance = dict(getattr(curve, "provenance", {}) or {})
    return observed_evidence_entity(
        curve.world,
        entity_id=f"{curve.boundary_id}:{role}",
        role=role,
        source_component_id=provenance.get("component_id"),
        source_loop_id=provenance.get("loop_label"),
        closed=bool(getattr(curve, "closed", True)),
    ).payload()


def _observed_anchor_entity_payload(anchor: Any, *, component_id: int) -> dict[str, Any]:
    from osn_gs.utils.torch_ops import require_torch

    torch = require_torch()
    point = torch.as_tensor(anchor.point).reshape(1, 3)
    provenance = dict(getattr(anchor, "provenance", {}) or {})
    return observed_evidence_entity(
        point,
        entity_id=f"component:{component_id}:interior_anchor:{provenance.get('source_point_index')}",
        role="interior_anchor",
        source_component_id=component_id,
        source_anchor_id=provenance.get("source_point_index"),
        closed=False,
        source_point_indices=(int(anchor.source_point_index),),
    ).payload()


def _materialize_boundary_role_network(
    boundary_result: Any,
    roles: tuple[BoundaryRoleEvidence, BoundaryRoleEvidence],
    *,
    curve_count: int,
    samples_per_curve: int,
    reverse_inner_boundary: bool,
    inner_phase: float,
    provenance: dict[str, Any],
) -> BoundaryFirstVisibleSurfaceResult:
    """Materialize the one Boundary-first contract from explicit roles only."""
    outer_role, interior_role = roles
    role_provenance = {**provenance, "boundary_roles": [role.role for role in roles]}
    component_id = provenance.get("component_id")
    if interior_role.role == "interior_boundary":
        network = build_boundary_support_curve_network(
            interior_role.curve,
            outer_role.curve,
            curve_count=curve_count,
            samples_per_curve=samples_per_curve,
            reverse_boundary_b=reverse_inner_boundary,
            boundary_b_phase=inner_phase,
        )
        surface_result = build_boundary_constrained_surface(network)
        return BoundaryFirstVisibleSurfaceResult(
            "constructed",
            "boundary_role_network",
            None,
            surface_result,
            {
                **role_provenance,
                "network": network.payload(),
                "observed_outer_boundary": _observed_boundary_entity_payload(outer_role.curve, role="outer_boundary"),
                "observed_inner_boundary": _observed_boundary_entity_payload(interior_role.curve, role="interior_boundary"),
            },
        )
    if interior_role.role == "interior_anchor":
        anchor = interior_role.anchor
        coverage = _anchor_ray_support_coverage(boundary_result, outer_role.curve, anchor)
        observed_layers = {
            "observed_outer_boundary": _observed_boundary_entity_payload(outer_role.curve, role="outer_boundary"),
            "observed_interior_anchor": _observed_anchor_entity_payload(anchor, component_id=component_id),
        }
        if coverage < 0.99:
            return BoundaryFirstVisibleSurfaceResult(
                "unsupported",
                "boundary_role_network",
                "interior_support_crosses_unobserved_region",
                None,
                {
                    **role_provenance,
                    **anchor.provenance,
                    "anchor_ray_support_coverage": coverage,
                    **observed_layers,
                },
            )
        cap = build_boundary_central_cap(outer_role.curve, anchor, segment_count=curve_count)
        return BoundaryFirstVisibleSurfaceResult(
            "constructed",
            "boundary_role_network",
            None,
            cap,
            {
                **role_provenance,
                **cap.provenance,
                "anchor_ray_support_coverage": coverage,
                "minimum_anchor_ray_support_coverage": 0.99,
                "anchor_ray_support_tolerance_cells": 1,
                **observed_layers,
            },
        )
    raise ValueError(f"unsupported Boundary-first interior role: {interior_role.role}")


def build_boundary_first_visible_surface(
    boundary_result: Any,
    *,
    component_points: Any | None = None,
    source_indices: Any | None = None,
    curve_count: int = 8,
    samples_per_curve: int = 8,
    reverse_inner_boundary: bool = False,
    inner_phase: float = 0.0,
    minimum_hole_area_ratio: float = 0.02,
) -> BoundaryFirstVisibleSurfaceResult:
    """Construct from observed outer boundary and explicit interior evidence.

    A nested loop supplies an ``interior_boundary`` role; a hole-free component
    supplies an observed ``interior_anchor`` role.  Both enter the same role
    network materialization contract.  Ambiguous multi-loop evidence remains
    review-only and no rectangle fallback is permitted.
    """
    outer = list(getattr(boundary_result, "outer_loops", ()))
    holes = list(getattr(boundary_result, "hole_loops", ()))
    component_id = int(getattr(boundary_result, "component_id"))
    provenance = {
        "component_id": component_id,
        "outer_loop_count": len(outer),
        "hole_loop_count": len(holes),
        "construction": "boundary_first_role_network",
    }
    if not isinstance(minimum_hole_area_ratio, (int, float)) or not 0.0 < float(minimum_hole_area_ratio) < 1.0:
        raise ValueError("minimum_hole_area_ratio must lie in (0, 1).")

    if len(outer) != 1:
        if len(outer) == 0:
            return BoundaryFirstVisibleSurfaceResult("unsupported", "boundary_role_network", "outer_boundary_missing", None, provenance)
        return BoundaryFirstVisibleSurfaceResult("unsupported", "boundary_role_network", "outer_boundary_ambiguous", None, provenance)
    if len(holes) > 1:
        correspondence = assess_multi_loop_correspondence(boundary_result)
        return BoundaryFirstVisibleSurfaceResult(
            correspondence.state,
            "boundary_role_network",
            correspondence.reason,
            None,
            {**provenance, **correspondence.provenance},
        )

    try:
        outer_curve = observed_outer_boundary_curve_from_component(boundary_result)
        fidelity_provenance = ({"source_boundary_fidelity": measure_observed_boundary_source_fidelity(outer_curve, component_points).payload()} if component_points is not None else {})
        if len(holes) == 1:
            hole_ratio = float(holes[0].area_world) / max(float(outer[0].area_world), 1e-12)
            if hole_ratio < float(minimum_hole_area_ratio):
                return BoundaryFirstVisibleSurfaceResult("unsupported", "boundary_role_network", "hole_area_ratio_too_small", None, provenance)
            _, interior_curve = observed_boundary_curves_from_annulus_component(boundary_result)
            roles = boundary_role_evidence(outer=outer_curve, interior=interior_curve)
        else:
            if component_points is None:
                return BoundaryFirstVisibleSurfaceResult("unsupported", "boundary_role_network", "interior_support_network_required", None, provenance)
            anchor = select_observed_interior_anchor(
                component_points,
                getattr(boundary_result, "frame"),
                source_indices=source_indices,
                support_mask=getattr(boundary_result, "refined_mask"),
            )
            roles = boundary_role_evidence(outer=outer_curve, anchor=anchor)
    except ValueError as error:
        return BoundaryFirstVisibleSurfaceResult("unsupported", "boundary_role_network", "ordered_boundary_required", None, {**provenance, "detail": str(error)})

    return _materialize_boundary_role_network(
        boundary_result,
        roles,
        curve_count=curve_count,
        samples_per_curve=samples_per_curve,
        reverse_inner_boundary=reverse_inner_boundary,
        inner_phase=inner_phase,
        provenance={**provenance, **fidelity_provenance},
    )