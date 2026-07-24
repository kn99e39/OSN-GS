from __future__ import annotations

"""Surface-agnostic parametric-Jacobian and orientation-consistency helpers.

Extracted (Phase D prerequisite, docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md
section 9) from `torch_annulus_chart.py`'s `_jacobian_diagnostics`, which mixed
two responsibilities (singular-value/condition computation, and a single-
reference orientation-consistency check) and required a `TorchNURBSSurface`
object. Both pieces here take raw derivative/normal tensors instead, so a
non-`TorchNURBSSurface` parametric domain (e.g. Phase D's `ContinuationDomain`)
can reuse the exact same math. The two responsibilities stay in separate
functions on purpose -- do not recombine them into one call.

Topology-specific orientation-adjacency logic (e.g. Phase D's own adjacent-
`s`-sample flip check, or `torch_annulus_chart.py`'s own cross-slice ring
holonomy check) is NOT this module's job -- each owns its own wrapper and may
additionally call `compute_orientation_consistency` here for a coarser,
adjacency-independent single-reference sanity check.
"""

from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-8


def compute_parametric_jacobian_metrics(
    deriv_a: Any,
    deriv_b: Any,
    *,
    eps: float = _EPS,
    scale: float = 1.0,
) -> dict[str, Any]:
    """True singular-value/condition metrics of the 3x2 Jacobian ``J = [deriv_a deriv_b]``.

    ``deriv_a``/``deriv_b`` are ``(N, 3)`` first-partial tensors along two
    parametric directions (any parametric surface, not just
    ``TorchNURBSSurface``). Returns per-sample tensors (``sigma_min``/
    ``sigma_max``/``area``/``condition``) alongside aggregate floats, via the
    closed-form eigenvalues of ``J^T J`` -- NOT ``||deriv_a x deriv_b||``
    alone (that area term equals ``sigma_min * sigma_max`` exactly for a 3x2
    matrix but cannot by itself distinguish a collapsed/anisotropic
    parameterization from a healthy one at the same area).

    ``scale`` is a caller-supplied characteristic length used only to report
    a scale-normalized ``min_jacobian_singular_value_normalized`` alongside
    the absolute value (never instead of it), since raw singular values are
    not comparable across differently-scaled patches/domains.
    """

    torch = require_torch()
    a = (deriv_a * deriv_a).sum(dim=1)
    d = (deriv_b * deriv_b).sum(dim=1)
    b = (deriv_a * deriv_b).sum(dim=1)
    trace = a + d
    disc = (trace.square() - 4.0 * (a * d - b * b)).clamp_min(0.0).sqrt()
    sigma_max = ((trace + disc).clamp_min(0.0) * 0.5).sqrt()
    sigma_min = ((trace - disc).clamp_min(0.0) * 0.5).sqrt()
    area = torch.cross(deriv_a, deriv_b, dim=1).norm(dim=1)
    condition = sigma_max / sigma_min.clamp_min(eps)
    sigma_min_normalized = sigma_min / (float(scale) + eps)

    return {
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "area": area,
        "condition": condition,
        "sigma_min_normalized": sigma_min_normalized,
        "min_area_jacobian": float(area.min().cpu()),
        "min_jacobian_singular_value": float(sigma_min.min().cpu()),
        "min_jacobian_singular_value_normalized": float(sigma_min_normalized.min().cpu()),
        "scale": float(scale),
        "jacobian_condition_mean": float(condition.mean().cpu()),
        "jacobian_condition_p95": float(condition.quantile(0.95).cpu()),
        "max_jacobian_condition": float(condition.max().cpu()),
        "near_degenerate_count": int((sigma_min < eps).sum()),
        "sample_count": int(area.shape[0]),
    }


def compute_orientation_consistency(
    normals: Any,
    *,
    valid_mask: Any | None = None,
    eps: float = _EPS,
) -> dict[str, Any]:
    """Single-reference orientation-consistency check over a flat sample set.

    ``normals`` is ``(N, 3)`` (need not be pre-normalized). Builds ONE
    self-consistent reference direction (seeded from the first valid sample,
    majority-sign-aligned across all valid samples), then reports each
    sample's ``orientation_dot`` against that reference and a flip count
    (``orientation_dot < 0``). ``valid_mask`` (``(N,)`` bool, optional)
    excludes samples from both the reference computation and the flip count;
    excluded entries get ``orientation_dot = 0.0`` (not NaN).

    This is exactly `torch_annulus_chart.py`'s existing per-surface check
    (reused as-is, not reinvented) -- a flat/unordered consistency check, NOT
    an adjacency-aware one. Sequential-adjacency checks (e.g. "do NEIGHBORING
    samples along a 1D strip flip sign") are a different algorithm and are
    each topology's own responsibility (see module docstring).
    """

    torch = require_torch()
    n = int(normals.shape[0])
    if valid_mask is None:
        valid_mask = torch.ones((n,), dtype=torch.bool, device=normals.device)
    area = normals.norm(dim=1)
    normal_unit = normals / area.clamp_min(eps)[:, None]

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
    if int(valid_indices.numel()) == 0:
        return {
            "reference_normal": [0.0, 0.0, 0.0],
            "orientation_dot": torch.zeros((n,), dtype=normals.dtype, device=normals.device),
            "orientation_flip_count": 0,
            "valid_sample_count": 0,
        }

    seed = normal_unit[valid_indices[valid_indices.shape[0] // 2]]
    valid_normals = normal_unit[valid_indices]
    aligned = torch.where((valid_normals @ seed < 0.0)[:, None], -valid_normals, valid_normals)
    reference = aligned.mean(dim=0)
    reference = reference / reference.norm().clamp_min(eps)

    orientation_dot = torch.zeros((n,), dtype=normals.dtype, device=normals.device)
    orientation_dot[valid_indices] = normal_unit[valid_indices] @ reference

    return {
        "reference_normal": reference.detach().cpu().tolist(),
        "orientation_dot": orientation_dot,
        "orientation_flip_count": int((orientation_dot[valid_indices] < 0.0).sum()),
        "valid_sample_count": int(valid_indices.numel()),
    }
