from __future__ import annotations

"""Worklog 99 -- pre-fit parametric domain validity for a Worklog 98
:class:`~osn_gs.surface.torch_latent_surface_curve_lattice.CurveLatticeUV`.

Worklog 98 proves tangent-frame ORIENTATION coherence (no adjacent
transversal direction reversal, no holonomy inconsistency around a cycle).
That is necessary but not sufficient for a well-conditioned parameter
domain: the tree-integrated ``(u, v)`` map can still fold, singularize, or
stretch/compress extremely relative to the actual supported 3D spacing even
when every local orientation agrees. This module characterizes those
failure modes BEFORE any surface is fit, so a candidate patch-construction
architecture is never blamed for an input domain that was already broken.

Every check operates on the coherent lattice's own already-computed
``(u, v)`` -- nothing here repairs or reroutes through PCA, and nothing
here is informed by any downstream fit or held-out result.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# Fixed, structural constants (not tuned from any fit/held-out outcome):
LOCAL_JACOBIAN_NEIGHBOR_COUNT = 6  # matches the minimum needed for a stable 2D affine LSQ fit
DUPLICATE_UV_CELL_FRACTION = 1e-3  # relative to the component's own (u, v) extent
DUPLICATE_INCOMPATIBLE_POSITION_RATIO = 3.0  # multiple of local median 3D spacing


@dataclass(frozen=True)
class ParametricDomainValidityReport:
    valid: bool
    invalid_reasons: tuple[str, ...]
    u_extent: float
    v_extent: float
    duplicate_incompatible_count: int
    fold_fraction: float
    singular_fraction: float
    mean_condition_number: float | None
    max_condition_number: float | None
    stretch_ratio_p95: float | None
    cycle_position_drift_p95_over_spacing: float | None


def _local_jacobian(
    positions: Any, uv: Any, index: int, neighbor_indices: Any,
) -> tuple[Any | None, Any | None]:
    """Least-squares local affine map ``delta_position ~= J @ delta_uv``
    over ``index``'s own UV-space neighborhood. Returns ``(J, singular_values)``,
    or ``(None, None)`` if the local UV neighborhood is degenerate (cannot
    support a 2D affine fit)."""

    torch = require_torch()
    delta_uv = uv[neighbor_indices] - uv[index]
    delta_position = positions[neighbor_indices] - positions[index]
    if int(delta_uv.shape[0]) < 3:
        return None, None
    try:
        solution = torch.linalg.lstsq(delta_uv, delta_position).solution  # (2, 3)
    except Exception:  # pragma: no cover - defensive
        return None, None
    jacobian = solution.T  # (3, 2): columns are d(position)/du, d(position)/dv
    try:
        singular_values = torch.linalg.svdvals(jacobian)
    except Exception:  # pragma: no cover - defensive
        return jacobian, None
    return jacobian, singular_values


def assess_parametric_domain_validity(
    positions: Any, uv: Any, normals: Any, median_spacing: float,
) -> ParametricDomainValidityReport:
    """Characterize one coherent component's already-computed ``(u, v)``
    map. ``normals`` are the SAME per-point normals the tangent frame field
    already carries (never recomputed here, never PCA-derived from
    scratch)."""

    torch = require_torch()
    count = int(positions.shape[0])
    invalid_reasons: list[str] = []

    u_extent = float((uv[:, 0].max() - uv[:, 0].min()).item())
    v_extent = float((uv[:, 1].max() - uv[:, 1].min()).item())
    if u_extent <= _EPS or v_extent <= _EPS:
        invalid_reasons.append("degenerate_uv_extent")

    # Duplicate/incompatible UV assignment: two points landing in the same
    # small UV cell but disagreeing geometrically beyond the local support
    # scale.
    cell_size_u = max(u_extent * DUPLICATE_UV_CELL_FRACTION, _EPS)
    cell_size_v = max(v_extent * DUPLICATE_UV_CELL_FRACTION, _EPS)
    cell_u = torch.round(uv[:, 0] / cell_size_u).to(torch.long)
    cell_v = torch.round(uv[:, 1] / cell_size_v).to(torch.long)
    duplicate_incompatible = 0
    cell_map: dict[tuple[int, int], int] = {}
    for index in range(count):
        key = (int(cell_u[index].item()), int(cell_v[index].item()))
        if key in cell_map:
            other = cell_map[key]
            distance = float((positions[index] - positions[other]).norm().item())
            if distance > DUPLICATE_INCOMPATIBLE_POSITION_RATIO * max(median_spacing, _EPS):
                duplicate_incompatible += 1
        else:
            cell_map[key] = index

    # Local parametric Jacobian: condition number, orientation, singularity.
    k = min(LOCAL_JACOBIAN_NEIGHBOR_COUNT, count - 1)
    fold_count = 0
    singular_count = 0
    condition_numbers: list[float] = []
    stretch_ratios: list[float] = []
    evaluated = 0
    if k >= 3:
        distance_uv = torch.cdist(uv, uv)
        distance_uv.fill_diagonal_(float("inf"))
        _, neighbor_indices_all = torch.topk(distance_uv, k, dim=1, largest=False)
        for index in range(count):
            jacobian, singular_values = _local_jacobian(positions, uv, index, neighbor_indices_all[index])
            if jacobian is None:
                continue
            evaluated += 1
            local_normal_estimate = torch.linalg.cross(jacobian[:, 0], jacobian[:, 1])
            local_normal_norm = local_normal_estimate.norm()
            if float(local_normal_norm.item()) > _EPS:
                orientation = float((local_normal_estimate / local_normal_norm * normals[index]).sum().item())
                if orientation < 0:
                    fold_count += 1
            if singular_values is not None and int(singular_values.numel()) == 2:
                largest, smallest = float(singular_values[0].item()), float(singular_values[1].item())
                if smallest <= _EPS * max(largest, 1.0):
                    singular_count += 1
                else:
                    condition_numbers.append(largest / smallest)
                stretch_ratios.append(largest / max(median_spacing, _EPS))

    fold_fraction = fold_count / evaluated if evaluated else 0.0
    singular_fraction = singular_count / evaluated if evaluated else 0.0
    mean_condition = sum(condition_numbers) / len(condition_numbers) if condition_numbers else None
    max_condition = max(condition_numbers) if condition_numbers else None
    stretch_p95 = (
        float(torch.quantile(torch.tensor(stretch_ratios), 0.95).item()) if stretch_ratios else None
    )

    if fold_fraction > 0.0:
        invalid_reasons.append("uv_orientation_reversal_or_foldover")
    if singular_fraction > 0.0:
        invalid_reasons.append("local_jacobian_singularity")
    if duplicate_incompatible > 0:
        invalid_reasons.append("duplicate_uv_incompatible_geometry")

    return ParametricDomainValidityReport(
        valid=len(invalid_reasons) == 0,
        invalid_reasons=tuple(invalid_reasons),
        u_extent=u_extent, v_extent=v_extent,
        duplicate_incompatible_count=duplicate_incompatible,
        fold_fraction=fold_fraction, singular_fraction=singular_fraction,
        mean_condition_number=mean_condition, max_condition_number=max_condition,
        stretch_ratio_p95=stretch_p95,
        cycle_position_drift_p95_over_spacing=None,  # filled in by the caller from the field's holonomy edges
    )


def cycle_position_drift_p95(
    component: Any, median_spacing: float,
) -> float | None:
    """Position-based companion to Worklog 98's own orientation-based
    holonomy check: for every SUPPORTED cycle-closing edge already tested
    for orientation, additionally measure how far the tree-derived
    ``(u, v)`` at the far endpoint disagrees with what a direct chord-length
    step across that edge (from the near endpoint's own frame) would
    predict. Reported purely as a diagnostic drift magnitude, in units of
    local median 3D spacing."""

    torch = require_torch()
    drifts: list[float] = []
    for edge in component.holonomy_edges:
        a, b = edge.node_a, edge.node_b
        delta = component.positions[b] - component.positions[a]
        predicted_u = component.u[a] + (delta * component.e_u[a]).sum()
        predicted_v = component.v[a] + (delta * component.e_v[a]).sum()
        actual_u, actual_v = component.u[b], component.v[b]
        drift = torch.sqrt((predicted_u - actual_u) ** 2 + (predicted_v - actual_v) ** 2)
        drifts.append(float(drift.item()) / max(median_spacing, _EPS))
    if not drifts:
        return None
    return float(torch.quantile(torch.tensor(drifts), 0.95).item())
