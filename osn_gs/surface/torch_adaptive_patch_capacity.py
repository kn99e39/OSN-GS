from __future__ import annotations

"""Worklog 102 -- pre-fit adaptive tensor-product B-spline capacity
selection, driven ENTIRELY by :mod:`~osn_gs.surface.torch_patch_identifiability`
(never by fit residual, held-out error, or classification -- verified by
AST in tests).

CANDIDATE_B (:func:`select_adaptive_quadratic_capacity`): degree fixed at
2 (matching the existing Worklog 97/98/100/101 downstream probe). Searches
every tensor-product control lattice from the valid quadratic minimum
(3x3) up to the fixed maximum (6x6), including rectangular grids. Selects
the LARGEST identifiable lattice by control-variable count
(``n_u * n_v``); ties are broken deterministically by how closely the
lattice's own aspect ratio (``n_u / n_v``) matches the chart's own
intrinsic ``(u, v)`` support aspect (``u_extent / v_extent``) -- never by
fit error.

CANDIDATE_C (:func:`select_support_adaptive_capacity`): also searches
degree 1 (minimum 2x2) alongside degree 2 (minimum 3x3), both up to the
same bounded maximum (6x6). Prefers the HIGHEST-order, then
HIGHEST-capacity identifiable lattice -- degree is never promoted or
demoted using fit/held-out performance, only pre-fit identifiability.
Still a tensor-product NURBS representation; no different surface family
is introduced.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_patch_identifiability import (
    PatchIdentifiabilityReport,
    assess_patch_identifiability,
)

MAX_CONTROL_GRID_DIM = 6  # fixed upper bound, matches the existing Worklog 97/98/100/101 probe
QUADRATIC_MIN_CONTROL_GRID_DIM = 3  # smallest valid tensor-product grid for a degree-2 B-spline
LINEAR_MIN_CONTROL_GRID_DIM = 2  # smallest valid tensor-product grid for a degree-1 B-spline


@dataclass(frozen=True)
class CapacitySelection:
    selected: bool
    degree_u: int
    degree_v: int
    control_grid_u: int
    control_grid_v: int
    report: PatchIdentifiabilityReport | None
    candidates_considered: int


def _aspect_tiebreak_key(n_u: int, n_v: int, target_aspect: float) -> float:
    grid_aspect = n_u / max(n_v, 1)
    # Compare in log-space so a 2:1 grid and a 1:2 grid are symmetric
    # deviations from a 1:1 target -- avoids an arbitrary directional bias.
    import math

    return abs(math.log(max(grid_aspect, 1e-6)) - math.log(max(target_aspect, 1e-6)))


def _enumerate_grids(min_dim: int, max_dim: int) -> list[tuple[int, int]]:
    return [(n_u, n_v) for n_u in range(min_dim, max_dim + 1) for n_v in range(min_dim, max_dim + 1)]


def _best_identifiable_grid(
    uv: Any, degree: int, min_dim: int, max_dim: int, target_aspect: float,
) -> tuple[tuple[int, int], PatchIdentifiabilityReport] | None:
    """Largest-capacity identifiable ``(n_u, n_v)`` at fixed ``degree``,
    with the aspect tie-break applied deterministically among equal-capacity
    candidates. Returns ``None`` if no grid in range is identifiable."""

    best: tuple[tuple[int, int], PatchIdentifiabilityReport] | None = None
    best_capacity = -1
    best_tiebreak = float("inf")
    for n_u, n_v in _enumerate_grids(min_dim, max_dim):
        report = assess_patch_identifiability(uv, degree, degree, n_u, n_v)
        if not report.identifiable:
            continue
        capacity = n_u * n_v
        tiebreak = _aspect_tiebreak_key(n_u, n_v, target_aspect)
        if capacity > best_capacity or (capacity == best_capacity and tiebreak < best_tiebreak):
            best = ((n_u, n_v), report)
            best_capacity = capacity
            best_tiebreak = tiebreak
    return best


def select_adaptive_quadratic_capacity(uv: Any) -> CapacitySelection:
    """CANDIDATE_B: degree fixed at 2, largest identifiable lattice from
    3x3 up to 6x6."""

    target_aspect = _target_aspect(uv)
    candidates_considered = len(_enumerate_grids(QUADRATIC_MIN_CONTROL_GRID_DIM, MAX_CONTROL_GRID_DIM))
    found = _best_identifiable_grid(uv, 2, QUADRATIC_MIN_CONTROL_GRID_DIM, MAX_CONTROL_GRID_DIM, target_aspect)
    if found is None:
        return CapacitySelection(False, 2, 2, 0, 0, None, candidates_considered)
    (n_u, n_v), report = found
    return CapacitySelection(True, 2, 2, n_u, n_v, report, candidates_considered)


def select_support_adaptive_capacity(uv: Any) -> CapacitySelection:
    """CANDIDATE_C: prefer the highest-order, then highest-capacity
    identifiable lattice -- degree 2 (min 3x3) tried before degree 1
    (min 2x2), both up to the same bounded maximum (6x6)."""

    target_aspect = _target_aspect(uv)
    quadratic_candidates = len(_enumerate_grids(QUADRATIC_MIN_CONTROL_GRID_DIM, MAX_CONTROL_GRID_DIM))
    linear_candidates = len(_enumerate_grids(LINEAR_MIN_CONTROL_GRID_DIM, MAX_CONTROL_GRID_DIM))
    total_considered = quadratic_candidates + linear_candidates

    found = _best_identifiable_grid(uv, 2, QUADRATIC_MIN_CONTROL_GRID_DIM, MAX_CONTROL_GRID_DIM, target_aspect)
    if found is not None:
        (n_u, n_v), report = found
        return CapacitySelection(True, 2, 2, n_u, n_v, report, total_considered)

    found = _best_identifiable_grid(uv, 1, LINEAR_MIN_CONTROL_GRID_DIM, MAX_CONTROL_GRID_DIM, target_aspect)
    if found is not None:
        (n_u, n_v), report = found
        return CapacitySelection(True, 1, 1, n_u, n_v, report, total_considered)

    return CapacitySelection(False, 2, 2, 0, 0, None, total_considered)


def _target_aspect(uv: Any) -> float:
    u_extent = float((uv[:, 0].max() - uv[:, 0].min()).item()) if int(uv.shape[0]) else 1.0
    v_extent = float((uv[:, 1].max() - uv[:, 1].min()).item()) if int(uv.shape[0]) else 1.0
    if v_extent <= 1e-9:
        return 1.0
    return u_extent / v_extent
