from __future__ import annotations

"""Worklog 98 -- U/V integral curves and fit-ready UV derived from one
coherent :class:`~osn_gs.surface.torch_latent_surface_tangent_frame_field.TangentFrameFieldComponent`.

The synchronized field already assigns every node in a coherent component a
consistent, tree-arc-length ``(u, v)`` (Worklog 98 section 6's
"reconcile UV at intersections/correspondences" is satisfied by the
holonomy-checked tree integration itself). This module additionally traces
EXPLICIT U/V integral curves -- bidirectional walks that follow the
synchronized field's own ``e_u``/``e_v`` direction at each step (never an
independently re-picked direction) -- for reporting (curve counts,
per-family residuals) and as an additional, explicitly ordered lattice
structure, while the actual NURBS fit input is the full coherent
component's point set with its field-derived ``(u, v)`` (a strict superset
of, and numerically consistent with, the traced curves).
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_tracer import propagate_tangent_onto_plane
from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.surface.torch_latent_surface_tangent_frame_field import TangentFrameFieldComponent
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

DEFAULT_INTEGRAL_CURVE_STEP_COUNT = 12
DEFAULT_INTEGRAL_CURVE_ANCHOR_COUNT = 4  # fixed, matches Worklog 96's own evenly-spaced-sample convention


@dataclass(frozen=True)
class IntegralCurve:
    family: str  # "u" or "v"
    anchor_node_index: int  # index into the component's own node_indices
    points: Any  # (N, 3)
    uv: Any  # (N, 2)


@dataclass(frozen=True)
class CurveLatticeUV:
    valid: bool
    invalid_reason: str | None
    points: Any | None = None  # (M, 3), the full coherent component
    uv: Any | None = None  # (M, 2)
    u_curves: tuple[IntegralCurve, ...] = ()
    v_curves: tuple[IntegralCurve, ...] = ()


def _nearest_field_node(query_position: Any, component: TangentFrameFieldComponent) -> int:
    torch = require_torch()
    distances = torch.cdist(query_position.reshape(1, 3), component.positions).reshape(-1)
    return int(distances.argmin().item())


def _trace_integral_curve(
    component: TangentFrameFieldComponent, support: LatentSurfaceSupport, start_local: int,
    family: str, *, step_count: int, step_size: float,
) -> IntegralCurve:
    """Bidirectional walk that looks up the SYNCHRONIZED field's own
    direction at the nearest field node each step (never independently
    re-picks a direction from a fresh local eigendecomposition)."""

    torch = require_torch()

    def _walk(sign: float) -> list[Any]:
        current = component.positions[start_local]
        current_uv = torch.stack([component.u[start_local], component.v[start_local]])
        points = [current.reshape(1, 3)]
        uvs = [current_uv.reshape(1, 2)]
        for _step in range(step_count):
            nearest = _nearest_field_node(current, component)
            base_direction = component.e_u[nearest] if family == "u" else component.e_v[nearest]
            direction = sign * base_direction
            result = support.query_batch(current.reshape(1, 3))
            if not bool(result.supported[0]):
                break
            propagated = propagate_tangent_onto_plane(direction, result.normals[0])
            if propagated is None:
                break
            predicted = current + step_size * propagated
            stepped = support.query_batch(predicted.reshape(1, 3))
            if not bool(stepped.supported[0]):
                break
            current = stepped.positions[0]
            # UV at the new position is looked up from the field's own
            # tree-derived potential at the nearest field node -- never
            # re-integrated independently along the traced path, so the
            # curve's UV stays numerically identical to the field it was
            # traced from.
            nearest_after = _nearest_field_node(current, component)
            current_uv = torch.stack([component.u[nearest_after], component.v[nearest_after]])
            points.append(current.reshape(1, 3))
            uvs.append(current_uv.reshape(1, 2))
        return points, uvs

    forward_points, forward_uv = _walk(1.0)
    backward_points, backward_uv = _walk(-1.0)
    backward_points_rev = list(reversed(backward_points[1:]))
    backward_uv_rev = list(reversed(backward_uv[1:]))
    all_points = torch.cat(backward_points_rev + forward_points, dim=0) if backward_points_rev else torch.cat(forward_points, dim=0)
    all_uv = torch.cat(backward_uv_rev + forward_uv, dim=0) if backward_uv_rev else torch.cat(forward_uv, dim=0)
    return IntegralCurve(family, start_local, all_points, all_uv)


def build_curve_lattice(
    component: TangentFrameFieldComponent,
    support: LatentSurfaceSupport,
    *,
    anchor_count: int = DEFAULT_INTEGRAL_CURVE_ANCHOR_COUNT,
    step_count: int = DEFAULT_INTEGRAL_CURVE_STEP_COUNT,
) -> CurveLatticeUV:
    """Fit-ready UV for one coherent field component, plus explicitly
    traced U/V integral curves for reporting. The fit input is the FULL
    component point set with its field-derived, holonomy-checked ``(u, v)``
    -- never PCA, never independently re-estimated."""

    torch = require_torch()
    if not component.coherent:
        return CurveLatticeUV(False, component.incoherence_reason or "incoherent_component")
    count = int(component.positions.shape[0])
    if count < 4:
        return CurveLatticeUV(False, "insufficient_component_size")

    u_extent = float((component.u.max() - component.u.min()).item())
    v_extent = float((component.v.max() - component.v.min()).item())
    if u_extent <= _EPS or v_extent <= _EPS:
        return CurveLatticeUV(False, "degenerate_field_parameter_extent")

    # Anchors evenly spaced by node index (deterministic; index order here
    # only breaks ties among already-geometrically-equivalent choices, it
    # does not define orientation -- that came from the field).
    anchor_step = max(1, count // anchor_count)
    anchor_indices = list(range(0, count, anchor_step))[:anchor_count]

    step_size = support.median_spacing
    u_curves = tuple(
        _trace_integral_curve(component, support, anchor, "u", step_count=step_count, step_size=step_size)
        for anchor in anchor_indices
    )
    v_curves = tuple(
        _trace_integral_curve(component, support, anchor, "v", step_count=step_count, step_size=step_size)
        for anchor in anchor_indices
    )

    uv = torch.stack([component.u, component.v], dim=1)
    # Normalize to [0, 1] for the NURBS parameter domain, same convention
    # PCA-UV already used -- this is a pure affine reparameterization of
    # the field's own coordinates, it does not alter orientation/ordering.
    u_min, u_max = component.u.min(), component.u.max()
    v_min, v_max = component.v.min(), component.v.max()
    normalized_uv = torch.stack([
        (component.u - u_min) / (u_max - u_min).clamp_min(_EPS),
        (component.v - v_min) / (v_max - v_min).clamp_min(_EPS),
    ], dim=1)

    return CurveLatticeUV(True, None, component.positions, normalized_uv, u_curves, v_curves)
