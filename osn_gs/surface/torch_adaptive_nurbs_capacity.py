from __future__ import annotations

"""Worklog 99 -- deterministic, structural control-grid capacity selection
for candidate B (ADAPTIVE_REGULARIZED_NURBS).

Capacity is a fixed function of information already available BEFORE any
fit is attempted: Worklog 98's own U/V integral-curve counts, the
component's total sample count, and its ``(u, v)`` aspect ratio. It is
never chosen from fit residual or held-out error, and this module never
imports the NURBS fitter (verified by an AST test) so that dependency
cannot creep in.
"""

from dataclasses import dataclass
from typing import Any

MIN_RESOLUTION = 4  # a degree-2 clamped B-spline needs >=3 control points per axis; 4 leaves margin
MAX_RESOLUTION = 10  # bounded so this stays a modest capacity change, not an unbounded increase
MIN_TOTAL_BUDGET = 8
MAX_TOTAL_BUDGET = 20


@dataclass(frozen=True)
class AdaptiveCapacity:
    resolution_u: int
    resolution_v: int
    u_curve_count: int
    v_curve_count: int
    sample_count: int
    aspect_ratio: float


def select_adaptive_control_grid_capacity(
    u_curve_count: int, v_curve_count: int, sample_count: int, u_extent: float, v_extent: float,
) -> AdaptiveCapacity:
    """Fixed, deterministic capacity rule -- computed once per component,
    strictly before any fit. Total control-point budget scales with
    sqrt(sample_count) (a standard rule-of-thumb bound keeping control
    density well below the sample density, avoiding overfitting sparse
    components), split between U/V in proportion to the (u, v) aspect
    ratio, then clamped to at least the component's own observed U/V
    curve-family counts (the structural evidence of how many independent
    constraint curves exist in each direction) and to the fixed
    [MIN_RESOLUTION, MAX_RESOLUTION] bounds.
    """

    import math

    budget = max(MIN_TOTAL_BUDGET, min(MAX_TOTAL_BUDGET, round(math.sqrt(max(sample_count, 1)))))
    aspect = max(u_extent, 1e-9) / max(v_extent, 1e-9)
    u_share = math.sqrt(aspect) / (math.sqrt(aspect) + 1.0)
    resolution_u = round(budget * u_share)
    resolution_v = budget - resolution_u

    resolution_u = max(resolution_u, min(u_curve_count, MAX_RESOLUTION) if u_curve_count > 0 else 0)
    resolution_v = max(resolution_v, min(v_curve_count, MAX_RESOLUTION) if v_curve_count > 0 else 0)

    resolution_u = max(MIN_RESOLUTION, min(MAX_RESOLUTION, resolution_u))
    resolution_v = max(MIN_RESOLUTION, min(MAX_RESOLUTION, resolution_v))

    return AdaptiveCapacity(
        resolution_u=resolution_u, resolution_v=resolution_v,
        u_curve_count=u_curve_count, v_curve_count=v_curve_count,
        sample_count=sample_count, aspect_ratio=aspect,
    )
