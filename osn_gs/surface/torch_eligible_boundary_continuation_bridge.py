from __future__ import annotations

"""Worklog 56 — production bridge: eligible visible-boundary surfaces to
bounded occluded-region candidates.

Consumes ONLY ``VisibleSurfaceConstructionResult.eligible_materialized_surfaces()``
(worklog 55) as its input. Does not re-derive, re-rank, or re-validate visible
candidate extraction, directed ordering, or region-boundary eligibility
classification -- those are entirely the upstream construction path's
responsibility and are treated as opaque, trusted facts here.

Also does not perform occluded NURBS fitting or uncertain-Gaussian append
(Phase F/G) -- this module stops at Phase E's geometric, evidence-free
``OccludedRegionCandidate`` objects, exactly like ``torch_occluded_region_candidate``
itself already scopes its own output.

Bridges two structurally different boundary representations:
- The canonical pipeline's ``OrderedBoundaryComponent``/materialized
  ``TorchNURBSSurface`` (world-space ordered loop only, no UV correspondence
  retained after least-squares fitting).
- Phase D/E's ``PatchBoundarySegment`` (UV + world + inner-UV + inner-world +
  tangent/normal), which ``build_continuation_domain``/
  ``build_geometric_region_candidates`` require as their input contract.

The bridge inverts each eligible surface's own ordered boundary/interior
world points back onto its own fitted surface (``project_torch_points_to_nurbs``)
to synthesize a ``PatchBoundarySegment`` -- the inner isocurve is picked as the
nearest *genuine projected interior point* to a small left-offset target, never
a synthetic/interpolated position, so the segment is only as trustworthy as the
eligible region's own already-approved interior evidence.

Region/component/source-ID/status/supporting-ID provenance is carried by
reference, not duplicated into ``ContinuationDomain``/``OccludedRegionCandidate``
fields (which this module does not modify): every synthesized
``PatchBoundarySegment.provenance`` carries the full record, and every
downstream domain/candidate keeps an unbroken ID chain back to it
(``ContinuationDomain.source_boundary_id`` and
``OccludedRegionCandidate.supporting_boundary_ids``), resolvable via this
result's own ``boundaries_by_id`` map.

Worklog 56 follow-up (this file): ``build_continuation_domain``'s own
pre-existing minimum-sample-count floor for a closed boundary (>=4 unique
world samples) is a REPRESENTATION-density convention, not a geometric safety
requirement -- empirically, its own tangent/direction/Jacobian math is fully
well-defined and non-degenerate for a bare 3-vertex triangle (verified
directly against ``_world_arclength_tangent``/``_arclength_metadata`` before
relying on this). So a validated 3-vertex eligible closed loop is
deterministically upsampled (edge midpoint insertion in UV space, evaluated
through the SAME already-fitted, already-approved surface -- never the raw
control polygon) purely to satisfy that pre-existing representation floor.
This adds no new topology, no new gap, and never closes an open chain: it is
the same technique ``torch_patch_boundary.build_rectangular_patch_edge``
already uses (``torch.linspace`` upsampling of an edge, then
``surface.evaluate``), just applied to an already-closed, already-validated
loop instead of a rectangular chart edge. ``build_continuation_domain``'s
threshold itself, and every other Phase D/E acceptance criterion, is
unmodified. Any FAILURE remaining after this resampling is a genuine
continuation-ineligibility of the surface itself (not a sample-count
artifact) and is recorded under ``STATUS_CONTINUATION_INELIGIBLE``
(``"eligible_visible_only_not_continuation_ready"``), never silently folded
into a generic exception string.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.surface.torch_continuation_domain import (
    ContinuationDomain,
    ContinuationDomainBuildError,
    build_continuation_domain,
)
from osn_gs.surface.torch_nurbs import TorchNURBSSurface, project_torch_points_to_nurbs
from osn_gs.surface.torch_occluded_region_candidate import (
    OccludedRegionCandidate,
    build_geometric_region_candidates,
)
from osn_gs.surface.torch_patch_boundary import PatchBoundarySegment, _curve_tangents, _make_record, _signed_area
from osn_gs.surface.torch_visible_boundary_materialization_adapter import VisibleBoundaryMaterializationResult
from osn_gs.surface.torch_visible_boundary_region_status import STATUS_ELIGIBLE_CLOSED
from osn_gs.surface.torch_visible_surface_construction import (
    VisibleSurfaceConstructionConfig,
    VisibleSurfaceConstructionResult,
    construct_visible_nurbs_from_gaussians,
)
from osn_gs.utils.torch_ops import require_torch

SCHEMA_VERSION = "eligible_boundary_continuation_bridge_worklog56_v2"

# Duplicated from `build_continuation_domain`'s own literal closed-boundary
# floor (torch_continuation_domain.py) -- NEVER used to gate/relax that
# module's own acceptance decision, only to pre-emptively upsample a
# validated loop so it clears a floor this bridge does not own or change.
_CLOSED_BOUNDARY_MIN_SAMPLES = 4

STATUS_BRIDGED = "bridged"
STATUS_SKIPPED_NOT_MATERIALIZED = "skipped_not_materialized"
STATUS_SKIPPED_NOT_ELIGIBLE_CLOSED = "skipped_region_status_not_eligible_closed_boundary"
STATUS_BOUNDARY_SEGMENT_BUILD_FAILED = "boundary_segment_build_failed"
STATUS_CONTINUATION_INELIGIBLE = "eligible_visible_only_not_continuation_ready"
BRIDGE_ATTEMPT_STATES = {
    STATUS_BRIDGED,
    STATUS_SKIPPED_NOT_MATERIALIZED,
    STATUS_SKIPPED_NOT_ELIGIBLE_CLOSED,
    STATUS_BOUNDARY_SEGMENT_BUILD_FAILED,
    STATUS_CONTINUATION_INELIGIBLE,
}


@dataclass(frozen=True)
class EligibleContinuationBridgeAttempt:
    """One eligible materialized surface's outcome, whether or not it produced a domain."""

    source_region_id: int
    source_boundary_component_id: str
    region_status: str
    region_status_reason: str
    boundary_role_scope: str
    supporting_source_ids: tuple[Any, ...]
    status: str
    reasons: tuple[str, ...]
    boundary_id: str | None
    continuation_domain_id: str | None

    def __post_init__(self) -> None:
        if self.status not in BRIDGE_ATTEMPT_STATES:
            raise ValueError(f"Unknown bridge attempt status: {self.status!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "source_region_id": int(self.source_region_id),
            "source_boundary_component_id": self.source_boundary_component_id,
            "region_status": self.region_status,
            "region_status_reason": self.region_status_reason,
            "boundary_role_scope": self.boundary_role_scope,
            "supporting_source_ids": list(self.supporting_source_ids),
            "status": self.status,
            "reasons": list(self.reasons),
            "boundary_id": self.boundary_id,
            "continuation_domain_id": self.continuation_domain_id,
        }


@dataclass(frozen=True)
class EligibleBoundaryContinuationBridgeResult:
    attempts: tuple[EligibleContinuationBridgeAttempt, ...]
    boundaries_by_id: dict[str, PatchBoundarySegment]
    continuation_domains: tuple[ContinuationDomain, ...]
    occluded_region_candidates: tuple[OccludedRegionCandidate, ...]
    schema_version: str = SCHEMA_VERSION

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "attempt_count": len(self.attempts),
            "bridged_count": sum(item.status == STATUS_BRIDGED for item in self.attempts),
            "skipped_not_materialized_count": sum(item.status == STATUS_SKIPPED_NOT_MATERIALIZED for item in self.attempts),
            "skipped_not_eligible_closed_count": sum(item.status == STATUS_SKIPPED_NOT_ELIGIBLE_CLOSED for item in self.attempts),
            "boundary_segment_build_failed_count": sum(item.status == STATUS_BOUNDARY_SEGMENT_BUILD_FAILED for item in self.attempts),
            "continuation_ineligible_count": sum(item.status == STATUS_CONTINUATION_INELIGIBLE for item in self.attempts),
            "continuation_domain_count": len(self.continuation_domains),
            "occluded_region_candidate_count": len(self.occluded_region_candidates),
            "schema_version": self.schema_version,
        }


def _nearest_interior_uv(boundary_uv_closed: Any, interior_uv: Any, step: float) -> Any:
    """Nearest genuine projected-interior-point UV to a small left-offset target per boundary sample.

    Mirrors ``torch_patch_boundary._nearest_supported_inner_uv``'s left-offset-then-nearest-real-
    evidence pattern, but the "real evidence" pool is this region's own already-approved interior
    points (projected to UV) instead of a raster support mask -- there is no mask here, only the
    same reliable-core interior points the eligible region's own materialization already used.
    """

    torch = require_torch()
    if int(interior_uv.shape[0]) == 0:
        raise ValueError("Cannot build an inner isocurve without any projected interior points.")
    tangent_uv = _curve_tangents(boundary_uv_closed, closed=True)
    left = torch.stack((-tangent_uv[:, 1], tangent_uv[:, 0]), dim=1)
    target = (boundary_uv_closed + step * left).clamp(0.0, 1.0)
    nearest = torch.cdist(target, interior_uv).argmin(dim=1)
    inner = interior_uv[nearest]
    if int(inner.shape[0]) > 1:
        inner = inner.clone()
        inner[-1] = inner[0]
    return inner


def _resample_closed_uv_loop_to_minimum(uv_closed: Any, min_required: int) -> Any:
    """Deterministically upsample an already-validated closed UV polygon
    (edge-midpoint insertion) until it has >= ``min_required`` unique
    vertices -- a pure resolution increase of the SAME loop, never a new
    vertex position off the existing edges and never a change to vertex
    order/orientation. No-op if already at or above the minimum.
    """

    torch = require_torch()
    unique = uv_closed[:-1]
    n = int(unique.shape[0])
    if n >= min_required:
        return uv_closed
    additional_needed = min_required - n
    vertices = [unique[i] for i in range(n)]
    resampled: list[Any] = []
    for i in range(n):
        resampled.append(vertices[i])
        if i < additional_needed:
            resampled.append(0.5 * (vertices[i] + vertices[(i + 1) % n]))
    stacked = torch.stack(resampled, dim=0)
    return torch.cat([stacked, stacked[:1]], dim=0)


def _build_boundary_segment_for_eligible_surface(
    materialized: VisibleBoundaryMaterializationResult,
) -> PatchBoundarySegment:
    torch = require_torch()
    surface = materialized.surface
    if surface is None:
        raise ValueError("Cannot build a boundary segment without a materialized surface.")
    adapter_input = materialized.input

    boundary_uv = project_torch_points_to_nurbs(adapter_input.ordered_boundary_points, surface)
    interior_uv = project_torch_points_to_nurbs(adapter_input.interior_points, surface)
    boundary_uv_closed = torch.cat([boundary_uv, boundary_uv[:1]], dim=0)
    boundary_uv_closed = _resample_closed_uv_loop_to_minimum(boundary_uv_closed, _CLOSED_BOUNDARY_MIN_SAMPLES)

    n_u, n_v = int(surface.control_grid.shape[0]), int(surface.control_grid.shape[1])
    step = 0.75 * min(1.0 / max(n_u, 1), 1.0 / max(n_v, 1))
    inner_uv_closed = _nearest_interior_uv(boundary_uv_closed, interior_uv, step)

    boundary_id = f"eligible:{int(adapter_input.source_region_id)}:{adapter_input.source_boundary_component_id}"
    provenance = {
        "region_id": int(adapter_input.source_region_id),
        "boundary_component_id": adapter_input.source_boundary_component_id,
        "region_status": adapter_input.region_status,
        "region_status_reason": adapter_input.region_status_reason,
        "boundary_role_scope": adapter_input.boundary_role_scope,
        "supporting_source_ids": [str(item) for item in adapter_input.supporting_source_ids],
        "eligible_boundary_bridge": True,
    }
    return _make_record(
        surface,
        int(adapter_input.source_region_id),
        boundary_id,
        "eligible_visible_boundary",
        boundary_uv_closed,
        inner_uv_closed,
        closed=True,
        orientation="ccw" if _signed_area(boundary_uv_closed) > 0.0 else "cw",
        provenance=provenance,
    )


def build_eligible_boundary_continuation_bridge(
    construction_result: Any,
    *,
    extent_multiplier: float = 1.0,
    correspondence_threshold: float = 1.0,
) -> EligibleBoundaryContinuationBridgeResult:
    """Connect ``eligible_materialized_surfaces()`` to continuation domains and
    bounded occluded-region candidates -- the sole production entry point for
    this bridge.

    Every eligible surface not in the ``eligible_closed_boundary`` region
    status, or whose materialization did not actually succeed, is recorded as
    a skipped attempt (never silently dropped) and contributes zero downstream
    objects -- this is what keeps 3k/10k (0 eligible regions) at zero
    continuation domains and zero candidates while 5k's eligible pair still
    reaches both stages.
    """

    attempts: list[EligibleContinuationBridgeAttempt] = []
    boundaries_by_id: dict[str, PatchBoundarySegment] = {}
    domains: list[ContinuationDomain] = []
    surfaces_by_patch_id: dict[int, TorchNURBSSurface] = {}

    for materialized in construction_result.eligible_materialized_surfaces():
        adapter_input = materialized.input
        common = dict(
            source_region_id=adapter_input.source_region_id,
            source_boundary_component_id=adapter_input.source_boundary_component_id,
            region_status=adapter_input.region_status,
            region_status_reason=adapter_input.region_status_reason,
            boundary_role_scope=adapter_input.boundary_role_scope,
            supporting_source_ids=adapter_input.supporting_source_ids,
        )

        if materialized.state != "materialized" or materialized.surface is None:
            attempts.append(EligibleContinuationBridgeAttempt(
                **common, status=STATUS_SKIPPED_NOT_MATERIALIZED,
                reasons=("materialization_state:" + materialized.state,) + tuple(materialized.review_reasons),
                boundary_id=None, continuation_domain_id=None,
            ))
            continue

        # Fail-closed: `eligible_materialized_surfaces()` is already restricted to
        # `eligible_closed_boundary` components by worklog 54/55's own gate, but this
        # bridge treats that as a fact to VERIFY (defense in depth against a future
        # upstream refactor bug), not to assume -- any component whose own carried
        # provenance disagrees never reaches a continuation domain.
        if adapter_input.region_status != STATUS_ELIGIBLE_CLOSED:
            attempts.append(EligibleContinuationBridgeAttempt(
                **common, status=STATUS_SKIPPED_NOT_ELIGIBLE_CLOSED,
                reasons=(f"region_status={adapter_input.region_status!r}",),
                boundary_id=None, continuation_domain_id=None,
            ))
            continue

        try:
            boundary = _build_boundary_segment_for_eligible_surface(materialized)
        except Exception as exc:  # noqa: BLE001 - fail-closed, never propagate into a partial bridge
            attempts.append(EligibleContinuationBridgeAttempt(
                **common, status=STATUS_BOUNDARY_SEGMENT_BUILD_FAILED,
                reasons=(f"{type(exc).__name__}:{exc}",),
                boundary_id=None, continuation_domain_id=None,
            ))
            continue

        boundaries_by_id[boundary.boundary_id] = boundary
        surfaces_by_patch_id[int(adapter_input.source_region_id)] = materialized.surface

        try:
            domain = build_continuation_domain(
                materialized.surface, boundary, extent_multiplier=extent_multiplier,
            )
        except (ValueError, ContinuationDomainBuildError) as exc:
            # Boundary was already deterministically upsampled past Phase D's
            # own representation-density floor above -- any failure reaching
            # here is a genuine geometric ineligibility of this surface for
            # continuation, not a sample-count artifact, so it gets its own
            # stable typed status instead of a generic exception string.
            attempts.append(EligibleContinuationBridgeAttempt(
                **common, status=STATUS_CONTINUATION_INELIGIBLE,
                reasons=(STATUS_CONTINUATION_INELIGIBLE, f"{type(exc).__name__}:{exc}"),
                boundary_id=boundary.boundary_id, continuation_domain_id=None,
            ))
            continue

        domains.append(domain)
        attempts.append(EligibleContinuationBridgeAttempt(
            **common, status=STATUS_BRIDGED, reasons=(f"domain_state:{domain.state}",),
            boundary_id=boundary.boundary_id, continuation_domain_id=domain.domain_id,
        ))

    candidates = build_geometric_region_candidates(
        domains, boundaries_by_id, surfaces_by_patch_id,
        correspondence_threshold=correspondence_threshold,
    )

    return EligibleBoundaryContinuationBridgeResult(
        attempts=tuple(attempts),
        boundaries_by_id=boundaries_by_id,
        continuation_domains=tuple(domains),
        occluded_region_candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class GaussianToOccludedCandidateBridgeResult:
    """Combined result of running the canonical visible-NURBS construction
    path and this bridge as one production call."""

    construction: VisibleSurfaceConstructionResult
    bridge: EligibleBoundaryContinuationBridgeResult


def run_eligible_boundary_continuation_bridge_from_gaussians(
    positions: Any,
    *,
    covariance: Any | None = None,
    log_scales: Any | None = None,
    rotations: Any | None = None,
    stable_ids: Any | None = None,
    config: VisibleSurfaceConstructionConfig | None = None,
    reliability: Any | None = None,
    continuation_input: Any | None = None,
    candidate_scale: Any | None = None,
    residual_scale: Any | None = None,
    extent_multiplier: float = 1.0,
    correspondence_threshold: float = 1.0,
) -> GaussianToOccludedCandidateBridgeResult:
    """The actual production orchestration entry point: raw Gaussian evidence
    all the way to bounded occluded-region candidates in one call.

    Mirrors ``construct_visible_nurbs_from_gaussians``'s own signature exactly
    (unchanged, called as-is) and feeds its result's
    ``eligible_materialized_surfaces()`` straight into
    ``build_eligible_boundary_continuation_bridge`` -- callers no longer need
    to manually chain the two functions themselves.
    """

    construction = construct_visible_nurbs_from_gaussians(
        positions, covariance=covariance, log_scales=log_scales, rotations=rotations,
        stable_ids=stable_ids, config=config, reliability=reliability,
        continuation_input=continuation_input, candidate_scale=candidate_scale, residual_scale=residual_scale,
    )
    bridge = build_eligible_boundary_continuation_bridge(
        construction, extent_multiplier=extent_multiplier, correspondence_threshold=correspondence_threshold,
    )
    return GaussianToOccludedCandidateBridgeResult(construction=construction, bridge=bridge)
