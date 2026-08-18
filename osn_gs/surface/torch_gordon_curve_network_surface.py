from __future__ import annotations

"""Worklog 99 -- Gordon-style curve-network surface construction (candidate C).

**Classical Gordon transfinite interpolation principle** (kept separate
from the OSN-GS approximation below): given a U-curve family ``C_i(v)``
(each at a fixed parameter ``u_i``) and a V-curve family ``D_j(u)`` (each
at a fixed ``v_j``), with known compatible intersections
``P_ij = C_i(v_j) = D_j(u_i)``, the Gordon surface is

    S(u, v) = S1(u, v) + S2(u, v) - S12(u, v)

where ``S1`` lofts (interpolates) across the U-curve family, ``S2`` lofts
across the V-curve family, and ``S12`` is the tensor-product interpolant of
the intersection grid alone -- subtracted so the crossing constraints are
not double-counted. This module treats the supported curves as the PRIMARY
constraint, never collapsing them into an unordered point set first.

**OSN-GS approximation for noisy scattered evidence** (explicitly NOT part
of the classical principle, called out wherever it appears below): OSN-GS
observations do not arrive as clean parametric curves at exact ``u_i``/
``v_j`` locations. This module (1) buckets a coherent Worklog 98 component's
already-computed ``(u, v)`` samples into a small number of discrete U-level
and V-level groups (the count fixed by Worklog 98's own U/V integral-curve
family sizes, never tuned from a fit result), (2) fits a smooth 1D
least-squares curve to each group's points as a function of the transverse
parameter (an actual curve fit, not a raw scatter -- "compatible B-spline/
NURBS curves...using the fixed component parameterization" per the
directive), (3) evaluates the classical loft + intersection-correction
formula on a regular parameter grid using those fitted curves, and (4) uses
the resulting grid values directly as a degree-2 tensor-product B-spline's
control net (a standard practical substitute for solving an exact
Gordon-interpolating B-spline, since a clamped B-spline's control polygon at
a regular parameter grid already closely tracks the values sampled there).

Fails closed (returns ``None`` with a reason) if the coherent component
cannot supply at least 2 distinct, sufficiently populated U-levels and
V-levels -- no fabricated boundary curves, no closed-boundary requirement,
no PCA repair.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9
MIN_LEVEL_POPULATION = 3  # minimum points to fit a meaningful 1D curve per level
MIN_LEVEL_COUNT = 2  # Gordon's loft needs at least 2 curves per family to interpolate between


@dataclass(frozen=True)
class GordonConstructionResult:
    valid: bool
    invalid_reason: str | None
    surface: TorchNURBSSurface | None
    u_level_count: int
    v_level_count: int
    intersection_grid_residual: float | None  # agreement between the two families' independent curve fits at P_ij


def _assign_levels(values: Any, level_count: int) -> Any:
    """Deterministic level assignment: evenly spaced bucket edges over the
    parameter's own observed range (never PCA, never fit-error-driven)."""

    torch = require_torch()
    v_min, v_max = values.min(), values.max()
    extent = (v_max - v_min).clamp_min(_EPS)
    normalized = (values - v_min) / extent
    level = torch.clamp((normalized * level_count).long(), 0, level_count - 1)
    return level


def _fit_1d_curve(transverse_param: Any, positions: Any) -> Any | None:
    """Least-squares quadratic fit of 3D position as a function of one
    scalar transverse parameter -- ``position ~= a + b*t + c*t^2`` per
    coordinate. A real curve fit through noisy scattered samples, not a
    lookup table. Returns the ``(3, 3)`` coefficient matrix or ``None`` if
    underdetermined."""

    torch = require_torch()
    count = int(transverse_param.shape[0])
    if count < MIN_LEVEL_POPULATION:
        return None
    design = torch.stack([torch.ones_like(transverse_param), transverse_param, transverse_param.square()], dim=1)
    try:
        solution = torch.linalg.lstsq(design, positions).solution  # (3, 3)
    except Exception:  # pragma: no cover - defensive
        return None
    return solution


def _evaluate_1d_curve(coefficients: Any, t: Any) -> Any:
    torch = require_torch()
    design = torch.stack([torch.ones_like(t), t, t.square()], dim=-1)
    return design @ coefficients


def construct_gordon_surface(
    points: Any,
    uv: Any,
    *,
    resolution_u: int,
    resolution_v: int,
    u_curve_count: int,
    v_curve_count: int,
    degree_u: int = 2,
    degree_v: int = 2,
) -> GordonConstructionResult:
    """Build a Gordon-style surface from a coherent component's own
    ``(points, uv)`` (Worklog 98's field-derived, holonomy-checked
    parameterization -- never PCA, never re-estimated). ``u_curve_count``/
    ``v_curve_count`` come from Worklog 98's own traced integral-curve
    families (structural, fixed before this call) and set how many
    discrete U-levels/V-levels this construction attempts -- never chosen
    from a fit or held-out result.
    """

    torch = require_torch()
    u_values, v_values = uv[:, 0], uv[:, 1]
    u_level_count = max(MIN_LEVEL_COUNT, min(u_curve_count, resolution_u))
    v_level_count = max(MIN_LEVEL_COUNT, min(v_curve_count, resolution_v))

    u_level = _assign_levels(u_values, u_level_count)
    v_level = _assign_levels(v_values, v_level_count)

    # Fit one 1D curve per U-level (a "V-varying" curve C_i(v), i.e. the
    # classical Gordon U-curve family: fixed u_i, parameterized by v) and
    # one per V-level (D_j(u), fixed v_j, parameterized by u).
    u_level_curves: dict[int, Any] = {}
    u_level_anchor_u: dict[int, float] = {}
    for level in range(u_level_count):
        mask = u_level == level
        if int(mask.sum().item()) < MIN_LEVEL_POPULATION:
            continue
        coefficients = _fit_1d_curve(v_values[mask], points[mask])
        if coefficients is not None:
            u_level_curves[level] = coefficients
            u_level_anchor_u[level] = float(u_values[mask].mean().item())

    v_level_curves: dict[int, Any] = {}
    v_level_anchor_v: dict[int, float] = {}
    for level in range(v_level_count):
        mask = v_level == level
        if int(mask.sum().item()) < MIN_LEVEL_POPULATION:
            continue
        coefficients = _fit_1d_curve(u_values[mask], points[mask])
        if coefficients is not None:
            v_level_curves[level] = coefficients
            v_level_anchor_v[level] = float(v_values[mask].mean().item())

    if len(u_level_curves) < MIN_LEVEL_COUNT or len(v_level_curves) < MIN_LEVEL_COUNT:
        return GordonConstructionResult(
            False, "insufficient_populated_curve_levels", None,
            len(u_level_curves), len(v_level_curves), None,
        )

    sorted_u_levels = sorted(u_level_curves.keys(), key=lambda level: u_level_anchor_u[level])
    sorted_v_levels = sorted(v_level_curves.keys(), key=lambda level: v_level_anchor_v[level])
    anchor_u_values = torch.tensor(
        [u_level_anchor_u[level] for level in sorted_u_levels], dtype=points.dtype, device=points.device,
    )
    anchor_v_values = torch.tensor(
        [v_level_anchor_v[level] for level in sorted_v_levels], dtype=points.dtype, device=points.device,
    )

    # Intersection grid P_ij: average of each family's own independent fit
    # evaluated at the other's anchor -- the two families' agreement here
    # is the intersection/correspondence residual.
    intersection_residuals: list[float] = []
    intersection_grid = torch.zeros(
        (len(sorted_u_levels), len(sorted_v_levels), 3), dtype=points.dtype, device=points.device,
    )
    for i, u_level_id in enumerate(sorted_u_levels):
        for j, v_level_id in enumerate(sorted_v_levels):
            from_u_curve = _evaluate_1d_curve(u_level_curves[u_level_id], anchor_v_values[j : j + 1])[0]
            from_v_curve = _evaluate_1d_curve(v_level_curves[v_level_id], anchor_u_values[i : i + 1])[0]
            intersection_grid[i, j] = 0.5 * (from_u_curve + from_v_curve)
            intersection_residuals.append(float((from_u_curve - from_v_curve).norm().item()))

    def _bracket(anchors: Any, query: float) -> tuple[int, int, float]:
        values = anchors.tolist()
        if query <= values[0]:
            return 0, 0, 0.0
        if query >= values[-1]:
            last = len(values) - 1
            return last, last, 0.0
        for index in range(len(values) - 1):
            if values[index] <= query <= values[index + 1]:
                span = max(values[index + 1] - values[index], _EPS)
                return index, index + 1, (query - values[index]) / span
        return 0, 0, 0.0

    grid_u = torch.linspace(
        float(u_values.min()), float(u_values.max()), resolution_u, dtype=points.dtype, device=points.device,
    )
    grid_v = torch.linspace(
        float(v_values.min()), float(v_values.max()), resolution_v, dtype=points.dtype, device=points.device,
    )
    control_grid = torch.zeros((resolution_u, resolution_v, 3), dtype=points.dtype, device=points.device)

    for a, query_u in enumerate(grid_u.tolist()):
        i0, i1, tu = _bracket(anchor_u_values, query_u)
        for b, query_v in enumerate(grid_v.tolist()):
            j0, j1, tv = _bracket(anchor_v_values, query_v)

            # S1: loft across the U-curve family -- interpolate the two
            # bracketing C_i curves (each evaluated at THIS query_v) across u.
            c_i0 = _evaluate_1d_curve(
                u_level_curves[sorted_u_levels[i0]], torch.tensor([query_v], dtype=points.dtype, device=points.device),
            )[0]
            c_i1 = _evaluate_1d_curve(
                u_level_curves[sorted_u_levels[i1]], torch.tensor([query_v], dtype=points.dtype, device=points.device),
            )[0]
            s1 = (1.0 - tu) * c_i0 + tu * c_i1

            # S2: loft across the V-curve family -- interpolate the two
            # bracketing D_j curves (each evaluated at THIS query_u) across v.
            d_j0 = _evaluate_1d_curve(
                v_level_curves[sorted_v_levels[j0]], torch.tensor([query_u], dtype=points.dtype, device=points.device),
            )[0]
            d_j1 = _evaluate_1d_curve(
                v_level_curves[sorted_v_levels[j1]], torch.tensor([query_u], dtype=points.dtype, device=points.device),
            )[0]
            s2 = (1.0 - tv) * d_j0 + tv * d_j1

            # S12: bilinear interpolation of the 4 bracketing intersection
            # grid points -- the correction term so crossing constraints
            # are not double-counted.
            p00, p01 = intersection_grid[i0, j0], intersection_grid[i0, j1]
            p10, p11 = intersection_grid[i1, j0], intersection_grid[i1, j1]
            s12 = (
                (1.0 - tu) * (1.0 - tv) * p00 + (1.0 - tu) * tv * p01
                + tu * (1.0 - tv) * p10 + tu * tv * p11
            )

            control_grid[a, b] = s1 + s2 - s12

    weights = torch.ones((resolution_u, resolution_v), dtype=points.dtype, device=points.device)
    surface = TorchNURBSSurface(control_grid=control_grid, weights=weights, degree_u=degree_u, degree_v=degree_v)
    mean_intersection_residual = sum(intersection_residuals) / len(intersection_residuals) if intersection_residuals else None
    return GordonConstructionResult(
        True, None, surface, len(u_level_curves), len(v_level_curves), mean_intersection_residual,
    )
