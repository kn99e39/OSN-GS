from __future__ import annotations

"""Worklog 103 -- visualization-only NURBS proxy for a latent support unit.

This representation has exactly one purpose: make the spatial extent of
projected latent geometry easy to inspect. It is COMPLETELY SEPARATE from
any Worklog 101/102 production-NURBS acceptance contract -- there is no
identifiability gate, no domain-validity gate, no safety/extrapolation
filter here. If a numerical fit can be produced at all, it is returned and
must be exported; fit-quality metadata is attached for information only,
never used to hide the result.

Uses the EXISTING, unmodified ``fit_torch_visible_surface_lsq`` (PCA-UV
seed + IDW + regularized LSQ + foot-point correction) -- no new fitting
algorithm, no new parameterization scheme. A small fixed resolution
ladder (never tuned from replay) is tried in order; the first that
produces a finite, non-degenerate control grid is kept.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_nurbs import TorchNURBSSurface, fit_torch_visible_surface_lsq
from osn_gs.utils.torch_ops import require_torch

REPRESENTATION_KIND = "latent_surface_coverage_visualization_nurbs"

# Fixed resolution ladder, coarsest-viable first up to a small fixed cap --
# not tuned from any replay/held-out outcome, purely a "can a numerical fit
# even be produced" ladder.
_RESOLUTION_LADDER: tuple[tuple[int, int, int, int], ...] = (
    (2, 2, 1, 1),  # (resolution_u, resolution_v, degree_u, degree_v)
    (3, 3, 2, 2),
    (4, 4, 2, 2),
)


@dataclass(frozen=True)
class VisualizationNurbsResult:
    unit_id: int
    materialized: bool
    invalid_reason: str | None
    surface: TorchNURBSSurface | None
    resolution_u: int | None
    resolution_v: int | None
    degree_u: int | None
    degree_v: int | None
    mean_residual: float | None  # informational only, never a visibility filter


def fit_visualization_nurbs(unit_id: int, points: Any) -> VisualizationNurbsResult:
    """Attempt every rung of the fixed resolution ladder, coarsest first,
    keeping the first numerically finite result. Never rejected for being
    unsafe/extrapolative/underdetermined -- only a literal numerical
    failure (NaN/Inf/exception) is reported as
    VISUALIZATION_NURBS_MATERIALIZATION_FAILED."""

    torch = require_torch()
    count = int(points.shape[0])
    if count < 3:
        return VisualizationNurbsResult(unit_id, False, "insufficient_points_for_any_surface", None, None, None, None, None, None)

    last_error: str | None = None
    for resolution_u, resolution_v, degree_u, degree_v in _RESOLUTION_LADDER:
        try:
            surface, foot_point_uv = fit_torch_visible_surface_lsq(
                points, resolution_u=resolution_u, resolution_v=resolution_v,
                degree_u=degree_u, degree_v=degree_v,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        control_grid = surface.control_grid
        if bool(torch.isnan(control_grid).any().item()) or bool(torch.isinf(control_grid).any().item()):
            last_error = "non_finite_control_grid"
            continue
        try:
            evaluated, _du, _dv = surface.evaluate_with_derivatives(foot_point_uv)
            residual = float((evaluated - points).norm(dim=1).mean().item())
        except Exception:  # noqa: BLE001
            residual = None
        return VisualizationNurbsResult(
            unit_id, True, None, surface, resolution_u, resolution_v, degree_u, degree_v, residual,
        )

    return VisualizationNurbsResult(
        unit_id, False, last_error or "no_resolution_ladder_rung_succeeded", None, None, None, None, None, None,
    )
