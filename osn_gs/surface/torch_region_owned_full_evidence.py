from __future__ import annotations

"""Worklog 67: region-owned full-cloud evidence for NURBS fitting/fidelity
validation only.

Worklog 66 found materialized patches are mostly ``under_supported``: NURBS
fitting only ever sees a region's REPRESENTATIVE members (as few as 3), even
though each representative stands in for a much larger set of full-cloud
Gaussians the density-preserving selection collapsed it from. This module
recovers that full-cloud evidence for FITTING/VALIDATION only -- it must
NEVER influence region formation, boundary ordering, or chart eligibility
(those stay representative-topology-only, exactly as before). The boundary
LOOP geometry is never touched here; only the INTERIOR fitting support set
expands.

Deliberately isolated (same convention as the other ``torch_gaussian_*``/
``torch_region_*`` modules in this package): every function here takes raw
tensors and an already-computed ``propagated_patch_ids`` array -- it does
not recompute the normal-alignment/residual compatibility gate itself. That
gate is `TorchOSNGSPipeline._propagate_with_evidence_gating` (worklog 129
item 10), already production code, already the pipeline's own accepted
mechanism for "does this full-cloud Gaussian's own evidence agree with its
nearest representative's oriented tangent plane" -- reused as the single
source of truth rather than duplicated, so this module cannot drift from it.
Because ownership there is a strict nearest-representative assignment (one
owning representative, hence at most one owning patch, per full-cloud
point), evidence from a different region can never be merged in here, and a
full-cloud point across a crease/parallel-sheet-conflict/ambiguous-frontier
is excluded by the SAME alignment/residual thresholds that already gate
ordinary full-cloud-to-patch assignment elsewhere in production.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_nurbs import TorchNURBSSurface, fit_torch_visible_surface_lsq
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_single_chart_uv_validity import interior_within_boundary
from osn_gs.utils.torch_ops import require_torch

# Same constant as worklog 66's `under_supported` cutoff, itself borrowed
# from `RegionFormationConfig.core_region_typical_min_size` -- kept
# consistent rather than reinvented, so "the existing minimum support
# contract" means the same threshold everywhere it is checked.
MIN_FULL_EVIDENCE_SUPPORT = 4

STATE_MATERIALIZED = "materialized"
STATE_UNDER_SUPPORTED = "under_supported"
STATE_UNSAFE_GEOMETRY = "unsafe_geometry"
STATE_FIT_FAILED = "fit_failed"
# Worklog 79: the chart boundary and the full-evidence fitting set are two
# DIFFERENT scales of the same region -- the boundary comes from the region's
# representative topology while the evidence comes from ownership
# propagation. Nothing previously checked that the boundary actually bounds
# the evidence it is paired with. Measured on baseline_compatible@2900, all 5
# materialized charts bound 3-4 representatives spanning 0.15-0.67 of their
# own owned evidence extent, leaving 89.1-99.8% of that evidence OUTSIDE the
# chart domain; the resulting fits were reported as ordinary surfaces while
# being pure extrapolation over most of the region.
STATE_DOMAIN_NOT_COVERING = "chart_domain_does_not_cover_evidence"

# A chart must bound a MAJORITY of the evidence it is fit to. This is a
# structural eligibility contract, not a tuned quantity: the measured
# violations are 89.1-99.8% outside, so every bound in roughly (0.5, 0.85)
# selects exactly the same set (verified in worklog 79). 0.5 is chosen only
# because "a chart that does not contain most of its own evidence is not that
# evidence's chart" is the weakest defensible statement of the contract.
MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION = 0.5


@dataclass(frozen=True)
class RegionOwnedFullEvidenceFit:
    """Additive companion to a `VisibleBoundaryMaterializationResult` -- the
    representative-only `surface`/`boundary_residual`/`interior_residual`
    on that result are never modified; this is a separate, parallel record."""

    chart_type: str  # "physical" | "parametric"
    source_region_id: int
    representative_support_count: int
    full_evidence_support_count: int
    full_evidence_stable_ids: tuple[Any, ...]
    state: str
    surface: TorchNURBSSurface | None
    boundary_residual: float | None
    full_evidence_interior_residual: float | None
    jacobian_near_degenerate_count: int | None
    reasons: tuple[str, ...]


def collect_region_owned_evidence(
    full_points: Any,
    full_stable_ids: Sequence[Any],
    propagated_patch_ids: Any,
    patch_id: int,
) -> tuple[Any, tuple[Any, ...]]:
    """Return this patch's owned full-cloud evidence: points whose nearest
    representative belongs to this patch AND whose own normal/position
    agree with that representative (``propagated_patch_ids == patch_id``,
    already computed by `_propagate_with_evidence_gating`).

    Exact-duplicate stable IDs are removed defensively -- a strict nearest-
    representative assignment structurally cannot produce them (each
    full-cloud point has exactly one nearest representative), but this is
    kept as an explicit safety net rather than an assumed invariant.
    """

    torch = require_torch()
    mask = propagated_patch_ids == patch_id
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    seen: set[Any] = set()
    unique_indices: list[int] = []
    unique_ids: list[Any] = []
    for local_index in indices:
        stable_id = full_stable_ids[local_index]
        if stable_id in seen:
            continue
        seen.add(stable_id)
        unique_indices.append(local_index)
        unique_ids.append(stable_id)
    if not unique_indices:
        return full_points[:0], ()
    index_tensor = torch.tensor(unique_indices, dtype=torch.long, device=full_points.device)
    return full_points[index_tensor], tuple(unique_ids)


def evidence_outside_chart_domain_fraction(boundary_points: Any, evidence_points: Any) -> float | None:
    """Fraction of ``evidence_points`` lying OUTSIDE the closed boundary loop,
    measured in the LOOP'S OWN best-fit plane.

    The loop defines the chart domain, so the loop's own plane -- not a plane
    refit to the combined set -- is the frame the containment question is
    posed in; refitting to the union would let distant evidence rotate the
    very frame it is being tested against. Returns ``None`` when the test is
    not defined (fewer than 3 boundary points or no evidence), so the caller
    fails open to its existing states rather than on an undefined measurement.
    """

    torch = require_torch()
    n = int(boundary_points.shape[0])
    m = int(evidence_points.shape[0])
    if n < 3 or m == 0:
        return None
    centered = boundary_points - boundary_points.mean(dim=0, keepdim=True)
    # Right-singular vectors of the loop: columns 0/1 span its own plane.
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    axis_u, axis_v = vh[0], vh[1]
    origin = boundary_points.mean(dim=0)
    boundary_uv = torch.stack((
        (boundary_points - origin) @ axis_u, (boundary_points - origin) @ axis_v,
    ), dim=1)
    evidence_uv = torch.stack((
        (evidence_points - origin) @ axis_u, (evidence_points - origin) @ axis_v,
    ), dim=1)
    report = interior_within_boundary(evidence_uv, boundary_uv)
    total = report["interior_total_count"]
    if not total:
        return None
    return report["interior_outside_boundary_count"] / total


def fit_region_owned_full_evidence_patch(
    chart_type: str,
    source_region_id: int,
    boundary_points: Any,
    full_evidence_points: Any,
    full_evidence_stable_ids: tuple[Any, ...],
    representative_support_count: int,
    *,
    min_support: int = MIN_FULL_EVIDENCE_SUPPORT,
    max_evidence_outside_domain_fraction: float = MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
    resolution_u: int = 6,
    resolution_v: int = 6,
    degree_u: int = 2,
    degree_v: int = 2,
    jacobian_sample_resolution: int = 9,
) -> RegionOwnedFullEvidenceFit:
    """Re-fit a patch's NURBS surface using the EXPANDED (boundary + region-
    owned full-evidence) support set. The boundary LOOP itself (and
    therefore boundary segment provenance / chart eligibility) is never
    touched -- only what counts as INTERIOR fitting support changes.

    Fail-closed exactly as worklog 66's classification requires: insufficient
    full-evidence support stays `under_supported`, a degenerate Jacobian or a
    non-finite fit stays `unsafe_geometry`/`fit_failed` -- never silently
    approved.
    """

    torch = require_torch()
    full_evidence_support_count = int(full_evidence_points.shape[0])
    base_kwargs = dict(
        chart_type=chart_type, source_region_id=source_region_id,
        representative_support_count=representative_support_count,
        full_evidence_support_count=full_evidence_support_count,
        full_evidence_stable_ids=full_evidence_stable_ids,
    )
    if full_evidence_support_count < min_support:
        return RegionOwnedFullEvidenceFit(
            **base_kwargs, state=STATE_UNDER_SUPPORTED, surface=None, boundary_residual=None,
            full_evidence_interior_residual=None, jacobian_near_degenerate_count=None,
            reasons=(f"full_evidence_support_count={full_evidence_support_count}<{min_support}",),
        )

    # Worklog 79: chart-domain coverage contract. The boundary loop and the
    # evidence set are paired here for the first time, so this is where the
    # pairing must be shown to be meaningful. Fail closed BEFORE fitting --
    # a surface fit inside a domain that excludes most of its own evidence is
    # extrapolation everywhere else, and reporting it as an ordinary
    # materialized surface overstates visible-surface coverage.
    outside_fraction = evidence_outside_chart_domain_fraction(boundary_points, full_evidence_points)
    if outside_fraction is not None and outside_fraction > max_evidence_outside_domain_fraction:
        return RegionOwnedFullEvidenceFit(
            **base_kwargs, state=STATE_DOMAIN_NOT_COVERING, surface=None, boundary_residual=None,
            full_evidence_interior_residual=None, jacobian_near_degenerate_count=None,
            reasons=(
                f"evidence_outside_chart_domain_fraction={outside_fraction:.4f}"
                f">{max_evidence_outside_domain_fraction}",
            ),
        )

    try:
        observed = torch.cat((boundary_points, full_evidence_points), dim=0)
        surface, _ = fit_torch_visible_surface_lsq(
            observed, resolution_u=resolution_u, resolution_v=resolution_v, degree_u=degree_u, degree_v=degree_v,
        )
        grid = torch.linspace(0.0, 1.0, jacobian_sample_resolution, dtype=observed.dtype, device=observed.device)
        sample_u, sample_v = torch.meshgrid(grid, grid, indexing="ij")
        uv = torch.stack((sample_u.reshape(-1), sample_v.reshape(-1)), dim=1)
        sampled, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
        if not torch.isfinite(sampled).all():
            return RegionOwnedFullEvidenceFit(
                **base_kwargs, state=STATE_FIT_FAILED, surface=None, boundary_residual=None,
                full_evidence_interior_residual=None, jacobian_near_degenerate_count=None,
                reasons=("non_finite_evaluate",),
            )
        # `scale` only affects the reported `sigma_min_normalized` (an
        # informational field) -- `near_degenerate_count` itself compares
        # against a fixed absolute epsilon, unaffected by this estimate.
        # Median nearest-neighbor spacing of the full evidence set, same
        # convention worklog 66's fidelity validation already uses.
        if int(full_evidence_points.shape[0]) >= 2:
            evidence_pairwise = torch.cdist(full_evidence_points, full_evidence_points)
            evidence_pairwise.fill_diagonal_(float("inf"))
            local_scale = float(evidence_pairwise.min(dim=1).values.median().clamp_min(1e-6))
        else:
            local_scale = 1e-6
        jacobian = compute_parametric_jacobian_metrics(deriv_u, deriv_v, scale=local_scale)
        if jacobian["near_degenerate_count"] > 0:
            return RegionOwnedFullEvidenceFit(
                **base_kwargs, state=STATE_UNSAFE_GEOMETRY, surface=None, boundary_residual=None,
                full_evidence_interior_residual=None,
                jacobian_near_degenerate_count=jacobian["near_degenerate_count"],
                reasons=(f"jacobian_near_degenerate_count={jacobian['near_degenerate_count']}",),
            )
        boundary_residual = float(torch.cdist(boundary_points, sampled).min(dim=1).values.mean())
        interior_residual = float(torch.cdist(full_evidence_points, sampled).min(dim=1).values.mean())
        return RegionOwnedFullEvidenceFit(
            **base_kwargs, state=STATE_MATERIALIZED, surface=surface, boundary_residual=boundary_residual,
            full_evidence_interior_residual=interior_residual, jacobian_near_degenerate_count=0, reasons=(),
        )
    except Exception as exc:  # noqa: BLE001
        return RegionOwnedFullEvidenceFit(
            **base_kwargs, state=STATE_FIT_FAILED, surface=None, boundary_residual=None,
            full_evidence_interior_residual=None, jacobian_near_degenerate_count=None,
            reasons=(type(exc).__name__,),
        )
