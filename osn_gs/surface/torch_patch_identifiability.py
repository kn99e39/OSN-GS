from __future__ import annotations

"""Worklog 102 -- pre-fit tensor-product B-spline patch identifiability.

Worklog 101's downstream probe rejected any chart below the fixed 6x6
grid's own 36-control-point count as structurally insufficient -- but that
count-based cutoff is not a mathematical requirement of the existing
regularized NURBS solver (:func:`~osn_gs.surface.torch_nurbs._solve_control_grid_lsq`):
a regularized (Tikhonov-anchored, second-difference-penalized) system is
non-singular and solvable even when the RAW data design matrix is
underdetermined. This module replaces the count-based cutoff with an
explicit, pre-fit ALGEBRAIC identifiability contract, built directly from
the exact same tensor-product B-spline basis-table machinery the real
solver uses (:meth:`~osn_gs.surface.torch_nurbs.TorchNURBSSurface._basis_tables`)
-- never a re-implementation of the basis.

Identifiability never uses fit residual, held-out error, extrapolative/
unsafe classification, or rendering metrics (verified by AST in tests). It
is purely a property of the design matrix ``rows[q, i*n_v+j] = N_i(u_q)
N_j(v_q)`` at the chart's own fixed intrinsic ``(u, v)`` samples:

- ``sample_count`` (rows) vs ``control_variable_count`` (columns, ``n_u *
  n_v``).
- The design matrix's numerical rank via SVD, with a tolerance derived
  from the matrix's own scale and numerical precision (``max(shape) *
  eps * largest_singular_value`` -- the same convention
  ``torch.linalg.matrix_rank`` uses by default), never a replay-tuned
  threshold.
- Full condition number / singular-value spectrum for reporting.
- ``u_extent``/``v_extent`` and whether the basis is actually constrained
  (non-degenerate) along each axis independently.

A chart is PATCH_IDENTIFIABLE for a candidate degree/grid configuration iff
the design matrix achieves its OWN maximum possible rank given its shape:
``rank == min(sample_count, control_variable_count)``. When
``sample_count >= control_variable_count`` this is the classical
full-column-rank well-posedness condition. When undersampled
(``sample_count < control_variable_count``, which Worklog 101's small
charts routinely are), this is full ROW rank -- every observation
independently constrains the system, so the remaining null space is filled
by the existing regularizer's principled prior (anchored to the seed grid)
rather than by a degenerate, data-starved collapse. A design matrix that
does NOT reach its own achievable rank (e.g. because all samples collapse
onto a single u or v value, or duplicate UV assignments) is NOT
identifiable regardless of sample count -- that is a genuine algebraic
degeneracy no amount of regularization repairs meaningfully.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


@dataclass(frozen=True)
class PatchIdentifiabilityReport:
    identifiable: bool
    invalid_reason: str | None
    degree_u: int
    degree_v: int
    control_grid_u: int
    control_grid_v: int
    sample_count: int
    control_variable_count: int
    effective_rank: int
    achievable_rank: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    u_extent: float
    v_extent: float
    u_constrained: bool
    v_constrained: bool


def _design_matrix(uv: Any, degree_u: int, degree_v: int, n_u: int, n_v: int) -> Any:
    """The EXACT tensor-product B-spline design matrix the real regularized
    solver assembles (:func:`~osn_gs.surface.torch_nurbs._lsq_normal_system`),
    built via the real, unmodified basis-table method -- never a
    re-implementation of the Cox-de Boor recursion."""

    torch = require_torch()
    dtype, device = uv.dtype, uv.device
    probe_surface = TorchNURBSSurface(
        control_grid=torch.zeros((n_u, n_v, 3), dtype=dtype, device=device),
        weights=torch.ones((n_u, n_v), dtype=dtype, device=device),
        degree_u=degree_u, degree_v=degree_v,
    )
    basis_u, basis_v, _dbasis_u, _dbasis_v = probe_surface._basis_tables(uv)
    rows = torch.einsum("qi,qj->qij", basis_u, basis_v).reshape(int(uv.shape[0]), n_u * n_v)
    return rows


def assess_patch_identifiability(
    uv: Any, degree_u: int, degree_v: int, n_u: int, n_v: int,
) -> PatchIdentifiabilityReport:
    """Pre-fit algebraic identifiability of a tensor-product B-spline
    control grid at the chart's own fixed intrinsic ``(u, v)`` samples.
    Never uses fit residual, held-out error, extrapolative/unsafe
    classification, or rendering metrics."""

    torch = require_torch()
    sample_count = int(uv.shape[0])
    control_variable_count = n_u * n_v

    u_extent = float((uv[:, 0].max() - uv[:, 0].min()).item()) if sample_count else 0.0
    v_extent = float((uv[:, 1].max() - uv[:, 1].min()).item()) if sample_count else 0.0
    u_constrained = u_extent > _EPS
    v_constrained = v_extent > _EPS

    if sample_count == 0:
        return PatchIdentifiabilityReport(
            False, "no_samples", degree_u, degree_v, n_u, n_v, sample_count, control_variable_count,
            0, 0, (), None, u_extent, v_extent, u_constrained, v_constrained,
        )

    design = _design_matrix(uv, degree_u, degree_v, n_u, n_v)
    try:
        singular_values = torch.linalg.svdvals(design)
    except Exception:  # pragma: no cover - defensive
        return PatchIdentifiabilityReport(
            False, "svd_failed", degree_u, degree_v, n_u, n_v, sample_count, control_variable_count,
            0, 0, (), None, u_extent, v_extent, u_constrained, v_constrained,
        )

    largest = float(singular_values[0].item()) if int(singular_values.numel()) else 0.0
    # Same tolerance convention torch.linalg.matrix_rank uses by default:
    # derived purely from the matrix's own scale and floating-point
    # precision, never tuned from any replay/held-out outcome.
    tol = max(design.shape) * torch.finfo(design.dtype).eps * max(largest, _EPS)
    effective_rank = int((singular_values > tol).sum().item())
    achievable_rank = min(sample_count, control_variable_count)
    smallest = float(singular_values[-1].item()) if int(singular_values.numel()) else 0.0
    condition_number = (largest / smallest) if smallest > tol else None

    invalid_reasons: list[str] = []
    if not u_constrained:
        invalid_reasons.append("degenerate_u_extent")
    if not v_constrained:
        invalid_reasons.append("degenerate_v_extent")
    if effective_rank < achievable_rank:
        invalid_reasons.append("rank_deficient_relative_to_achievable")

    return PatchIdentifiabilityReport(
        identifiable=len(invalid_reasons) == 0,
        invalid_reason=invalid_reasons[0] if invalid_reasons else None,
        degree_u=degree_u, degree_v=degree_v, control_grid_u=n_u, control_grid_v=n_v,
        sample_count=sample_count, control_variable_count=control_variable_count,
        effective_rank=effective_rank, achievable_rank=achievable_rank,
        singular_values=tuple(float(value.item()) for value in singular_values),
        condition_number=condition_number,
        u_extent=u_extent, v_extent=v_extent,
        u_constrained=u_constrained, v_constrained=v_constrained,
    )
