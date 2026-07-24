from __future__ import annotations

"""Single-chart constrained NURBS least-squares solve for Phase F.

docs/Urgent_Work/OSN_GS_Phase_F_Constrained_Occluded_Chart_Design.md sections 8, 10.

Reuses the sanctioned primitives from ``torch_nurbs`` (``_lsq_normal_system``,
``_second_difference_penalty``) WITHOUT modifying them or touching
``fit_coupled_patch_graph_lsq`` (a point-cloud-per-patch fitter unsuited to a
chart that has no point cloud of its own).

The occluded chart is fit to weighted *boundary/connector constraint samples*
(not observed points): support-boundary samples at high weight, connector
samples at low weight, with a fairness (second-difference) penalty on the
interior and a Tikhonov anchor to the Coons seed. Because the rational weights
are all 1 the surface is linear in the control points, so a single regularized
normal-equations solve is exact for the constraint set.
"""

from typing import Any

from osn_gs.surface.torch_nurbs import (
    TorchNURBSSurface,
    _lsq_normal_system,
    _second_difference_penalty,
)
from osn_gs.utils.torch_ops import require_torch


def solve_constrained_chart(
    seed_grid: Any,
    *,
    support_a_uv: Any,
    support_a_points: Any,
    support_b_uv: Any,
    support_b_points: Any,
    connector_uv: Any,
    connector_points: Any,
    degree_u: int,
    degree_v: int,
    support_weight: float,
    connector_weight: float,
    fairness_weight: float,
    interior_seed_weight: float,
    chunk_size: int = 4096,
) -> tuple[TorchNURBSSurface, dict[str, Any]]:
    """Solve for a chart control grid that follows the weighted constraints.

    ``seed_grid`` is the ``(Nu, Nv, 3)`` Coons seed (also the Tikhonov anchor).
    Support samples (both ``v=0`` and ``v=1`` boundaries) are constrained at
    ``support_weight``; connector samples (both ``u=0`` and ``u=1``) at
    ``connector_weight`` (``support_weight >> connector_weight``). Returns the
    fitted surface and a small solve-diagnostics dict.
    """

    torch = require_torch()
    dtype, device = seed_grid.dtype, seed_grid.device
    n_u, n_v = int(seed_grid.shape[0]), int(seed_grid.shape[1])
    n = n_u * n_v

    surface = TorchNURBSSurface(
        control_grid=seed_grid.clone(),
        weights=torch.ones((n_u, n_v), dtype=dtype, device=device),
        degree_u=int(degree_u),
        degree_v=int(degree_v),
    )

    def _cast(t: Any) -> Any:
        return torch.as_tensor(t, dtype=dtype, device=device)

    points = torch.cat([_cast(support_a_points), _cast(support_b_points), _cast(connector_points)], dim=0)
    uv = torch.cat([_cast(support_a_uv), _cast(support_b_uv), _cast(connector_uv)], dim=0)
    weights = torch.cat(
        [
            torch.full((int(support_a_points.shape[0]),), float(support_weight), dtype=dtype, device=device),
            torch.full((int(support_b_points.shape[0]),), float(support_weight), dtype=dtype, device=device),
            torch.full((int(connector_points.shape[0]),), float(connector_weight), dtype=dtype, device=device),
        ],
        dim=0,
    )

    with torch.no_grad():
        matrix, rhs, total_weight = _lsq_normal_system(points, uv, surface, 0.0, 0.0, chunk_size, weights)
        scale = max(total_weight, 1e-8)
        penalty = _second_difference_penalty(n_u, n_v, dtype, device)
        seed_flat = seed_grid.reshape(n, 3)
        system = (
            matrix / scale
            + float(fairness_weight) * penalty
            + float(interior_seed_weight) * torch.eye(n, dtype=dtype, device=device)
        )
        rhs_reg = rhs / scale + float(interior_seed_weight) * seed_flat
        used_lstsq = False
        try:
            solution = torch.linalg.solve(system, rhs_reg)
        except Exception:
            solution = torch.linalg.lstsq(system, rhs_reg).solution
            used_lstsq = True
        surface.control_grid = solution.reshape(n_u, n_v, 3)

    finite = bool(torch.isfinite(surface.control_grid).all())
    diagnostics = {
        "control_point_count": n,
        "total_constraint_weight": float(total_weight),
        "used_lstsq_fallback": used_lstsq,
        "control_grid_finite": finite,
    }
    return surface, diagnostics
