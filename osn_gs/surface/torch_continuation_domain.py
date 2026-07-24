from __future__ import annotations

"""Phase D — boundary-local world-space sampled continuation strip.

docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md section 6,
detailed design docs/Urgent_Work/OSN_GS_Phase_D_Continuation_Domain_Design.md
(Design Revision 3, user-approved for implementation).

Scope, exactly as approved -- do not extend beyond this without a separate
gate:

- Builds ONE independent sampled continuation strip per input boundary. Does
  NOT pair boundaries, compute overlap, or form bounded occluded-region
  candidates (Phase E's job).
- Does NOT build a final constrained occluded NURBS chart (Phase F's job).
- Does NOT call Phase C's `osn_gs.surface.torch_observation_evidence`
  anything -- `ContinuationDomain.world.reshape(-1, 3)` already matches
  `classify_world_samples`'s input shape, which is as far as the interface
  goes.
- Does NOT extrapolate the source `TorchNURBSSurface`'s control net/knot
  vector past ``[0, 1]``. The continuation strip is a separate, lightweight
  sampled grid seeded from the surface's own analytic derivatives at the
  boundary, not a `TorchNURBSSurface` itself.
- Does NOT re-estimate local geometry from the Gaussian point cloud.
- Canonical geometry is first-order only. Second derivatives feed a
  diagnostic (``second_order_growth_ratio`` et al.), never the returned
  position grid.
- Not wired into any production pipeline/trainer path.
"""

import math
from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.surface.torch_parametric_diagnostics import (
    compute_orientation_consistency,
    compute_parametric_jacobian_metrics,
)
from osn_gs.surface.torch_patch_boundary import BOUNDARY_RECONCILED_INTERNAL, PatchBoundarySegment
from osn_gs.utils.torch_ops import require_torch

STATE_VALID = "valid"
STATE_DEGENERATE = "degenerate"
STATE_REJECTED = "rejected"
CONTINUATION_STATES = {STATE_VALID, STATE_DEGENERATE, STATE_REJECTED}


class ContinuationDomainBuildError(RuntimeError):
    """Grid/AABB could not be constructed at all.

    A distinct category from both input-contract violations (``ValueError``)
    and post-construction quality problems (``ContinuationDomain`` returned
    with ``state=degenerate``/``rejected``) -- see module docstring and
    design doc section 2.3. No ``ContinuationDomain`` object exists when this
    is raised.
    """


@dataclass
class ContinuationDomain:
    """Boundary-local world-space sampled continuation strip.

    The sampled grid (``world``/``s_world``/``t_world``/``tangent_s``/
    ``tangent_t``/``normal``) is the canonical source of truth -- there is no
    continuous analytic ``evaluate(s, t)`` re-evaluation API, matching
    `PatchBoundarySegment`'s own ordered-samples-only representation (see
    ``interpolate_boundary_arclength`` for the one piecewise-linear
    exception, which is explicitly NOT analytic and operates on the source
    boundary, not this grid).

    ``state == STATE_VALID`` means this strip is a numerically valid
    continuation hypothesis -- NOT that it has been approved as an occluded
    surface or as one of Phase E's occluded-region candidates (deliberately
    different state names to avoid that exact confusion).
    """

    domain_id: str
    source_patch_id: int
    source_boundary_id: str
    closed: bool

    s_count: int
    t_count: int
    s_world: Any
    boundary_length: float
    t_world: Any

    world: Any
    tangent_s: Any
    tangent_t: Any
    normal: Any

    outward_tangent_world: Any

    normal_valid_mask: Any
    direction_valid_mask: Any
    sample_valid_mask: Any

    local_surface_scale: float
    continuation_extent: float
    extent_multiplier: float

    aabb_min: Any
    aabb_max: Any

    state: str
    reason: str
    validity: dict[str, Any]
    uncertainty: dict[str, float]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.state not in CONTINUATION_STATES:
            raise ValueError(f"Unknown continuation-domain state: {self.state!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "source_patch_id": int(self.source_patch_id),
            "source_boundary_id": self.source_boundary_id,
            "closed": bool(self.closed),
            "s_count": int(self.s_count),
            "t_count": int(self.t_count),
            "s_world": self.s_world.detach().cpu().tolist(),
            "boundary_length": float(self.boundary_length),
            "t_world": self.t_world.detach().cpu().tolist(),
            "world": self.world.detach().cpu().tolist(),
            "tangent_s": self.tangent_s.detach().cpu().tolist(),
            "tangent_t": self.tangent_t.detach().cpu().tolist(),
            "normal": self.normal.detach().cpu().tolist(),
            "outward_tangent_world": self.outward_tangent_world.detach().cpu().tolist(),
            "normal_valid_mask": self.normal_valid_mask.detach().cpu().tolist(),
            "direction_valid_mask": self.direction_valid_mask.detach().cpu().tolist(),
            "sample_valid_mask": self.sample_valid_mask.detach().cpu().tolist(),
            "local_surface_scale": float(self.local_surface_scale),
            "continuation_extent": float(self.continuation_extent),
            "extent_multiplier": float(self.extent_multiplier),
            "aabb_min": self.aabb_min.detach().cpu().tolist(),
            "aabb_max": self.aabb_max.detach().cpu().tolist(),
            "state": self.state,
            "reason": self.reason,
            "validity": self.validity,
            "uncertainty": self.uncertainty,
            "provenance": dict(self.provenance),
        }


def _strip_closed_duplicate(boundary: PatchBoundarySegment) -> tuple[Any, Any, Any, Any]:
    """Normalization (not validation): drop a closed loop's duplicated closing
    sample (``_canonicalize_closed_loop`` in torch_patch_boundary.py stores
    ``world[-1] == world[0]``). Canonical Phase D representation is ordered
    unique samples with no duplicate closing endpoint (design doc section 2.1).
    """

    torch = require_torch()
    uv, world = boundary.uv, boundary.world
    inner_uv, inner_world = boundary.inner_uv, boundary.inner_world
    if boundary.closed and int(world.shape[0]) > 1 and bool(torch.allclose(world[0], world[-1], atol=1e-9)):
        uv, world = uv[:-1], world[:-1]
        inner_uv, inner_world = inner_uv[:-1], inner_world[:-1]
    return uv, world, inner_uv, inner_world


def _arclength_metadata(world: Any, closed: bool) -> tuple[Any, float]:
    """``(s_world, boundary_length)`` per design doc section 4.1."""

    torch = require_torch()
    n = int(world.shape[0])
    if n < 2:
        s_world = world.new_zeros((max(n, 1),))
        return s_world, 0.0
    segment_lengths = (world[1:] - world[:-1]).norm(dim=1)
    s_world = torch.cat([world.new_zeros(1), torch.cumsum(segment_lengths, dim=0)])
    if closed:
        closing_length = (world[0] - world[-1]).norm()
        boundary_length = float(s_world[-1] + closing_length)
    else:
        boundary_length = float(s_world[-1])
    return s_world, boundary_length


def _world_arclength_tangent(points: Any, s_world: Any, boundary_length: float, closed: bool) -> Any:
    """World-arclength-normalized finite-difference tangent (design doc section 4.1).

    Reused for both the boundary's own world tangent ``T`` and ``d(outward
    direction)/ds`` -- the same central/one-sided/periodic-with-``boundary_length``
    -offset scheme applies to any vector field sampled at the same ``s_world``
    locations.
    """

    torch = require_torch()
    n = int(points.shape[0])
    if n < 2:
        return torch.zeros_like(points)
    if closed:
        idx = torch.arange(n, device=points.device, dtype=torch.long)
        prev_idx = torch.remainder(idx - 1, n)
        next_idx = torch.remainder(idx + 1, n)
        periodic_s_prev = s_world[prev_idx].clone()
        periodic_s_prev[0] = s_world[-1] - boundary_length
        periodic_s_next = s_world[next_idx].clone()
        periodic_s_next[-1] = s_world[0] + boundary_length
        denom = (periodic_s_next - periodic_s_prev).clamp_min(1e-12)
        return (points[next_idx] - points[prev_idx]) / denom[:, None]

    tangent = torch.empty_like(points)
    tangent[0] = (points[1] - points[0]) / (s_world[1] - s_world[0]).clamp_min(1e-12)
    tangent[-1] = (points[-1] - points[-2]) / (s_world[-1] - s_world[-2]).clamp_min(1e-12)
    if n > 2:
        denom = (s_world[2:] - s_world[:-2]).clamp_min(1e-12)
        tangent[1:-1] = (points[2:] - points[:-2]) / denom[:, None]
    return tangent


def interpolate_boundary_arclength(boundary: PatchBoundarySegment, s_query: Any) -> tuple[Any, Any]:
    """Piecewise-linear interpolation of ``boundary.world`` by cumulative world
    arclength.

    Explicitly NOT an analytic closed-form curve -- a straight-line
    approximation between existing ordered samples. Does not replace
    `ContinuationDomain`'s own ``s_world``/``world`` grid (the canonical
    source of truth); this is only a convenience for querying between grid
    samples on the ORIGINAL boundary, not the continuation strip itself.
    """

    torch = require_torch()
    _, world, _, _ = _strip_closed_duplicate(boundary)
    s_world, boundary_length = _arclength_metadata(world, bool(boundary.closed))

    query = torch.as_tensor(s_query, dtype=world.dtype, device=world.device)
    was_scalar = query.dim() == 0
    query = query.reshape(-1)

    if bool(boundary.closed) and boundary_length > 0.0:
        query = torch.remainder(query, boundary_length)
        s_ext = torch.cat([s_world, s_world.new_tensor([boundary_length])])
        world_ext = torch.cat([world, world[:1]], dim=0)
    else:
        query = query.clamp(0.0, float(s_world[-1]))
        s_ext = s_world
        world_ext = world

    upper = torch.searchsorted(s_ext.contiguous(), query.contiguous(), right=False)
    upper = upper.clamp(1, int(s_ext.shape[0]) - 1)
    lower = upper - 1
    s_lo, s_hi = s_ext[lower], s_ext[upper]
    denom = (s_hi - s_lo).clamp_min(1e-12)
    frac = ((query - s_lo) / denom).clamp(0.0, 1.0)
    world_lo, world_hi = world_ext[lower], world_ext[upper]
    position = world_lo + frac[:, None] * (world_hi - world_lo)
    tangent = torch.nn.functional.normalize(world_hi - world_lo, dim=1, eps=1e-12)

    if was_scalar:
        return position[0], tangent[0]
    return position, tangent


def _derive_local_surface_scale(world: Any, closed: bool, inner_world: Any, control_grid: Any) -> float:
    """Canonical ``local_surface_scale`` aggregate (design doc section 4.5.1).

    Raises `ContinuationDomainBuildError` if fewer than two of
    ``{L_boundary, L_inner, L_control}`` are finite and positive.
    """

    torch = require_torch()
    candidates: list[float] = []

    n = int(world.shape[0])
    if n > 1:
        seg = (world[1:] - world[:-1]).norm(dim=1)
        if closed:
            seg = torch.cat([seg, (world[0] - world[-1]).norm().reshape(1)])
        seg_pos = seg[seg > 0]
        if int(seg_pos.numel()) > 0:
            candidates.append(float(torch.quantile(seg_pos, 0.5)))

    inner_dist = (inner_world - world).norm(dim=1)
    inner_pos = inner_dist[inner_dist > 0]
    if int(inner_pos.numel()) > 0:
        candidates.append(float(torch.quantile(inner_pos, 0.5)))

    edge_parts = []
    if int(control_grid.shape[0]) > 1:
        edge_parts.append((control_grid[1:, :, :] - control_grid[:-1, :, :]).norm(dim=-1).reshape(-1))
    if int(control_grid.shape[1]) > 1:
        edge_parts.append((control_grid[:, 1:, :] - control_grid[:, :-1, :]).norm(dim=-1).reshape(-1))
    if edge_parts:
        edges = torch.cat(edge_parts)
        edges_pos = edges[edges > 0]
        if int(edges_pos.numel()) > 0:
            candidates.append(float(torch.quantile(edges_pos, 0.5)))

    valid_scales = [c for c in candidates if math.isfinite(c) and c > 0.0]
    if len(valid_scales) < 2:
        raise ContinuationDomainBuildError(
            f"Cannot derive local_surface_scale automatically: only {len(valid_scales)} of "
            "{L_boundary, L_inner, L_control} are finite and positive (need >= 2)."
        )
    return float(torch.quantile(torch.tensor(valid_scales, dtype=world.dtype), 0.5))


def _second_order_direction_diagnostic(
    Su: Any,
    Sv: Any,
    Suu: Any,
    Suv: Any,
    Svv: Any,
    outward_tangent_world: Any,
    continuation_extent: float,
    eps: float,
) -> dict[str, float]:
    """Directional second-order continuation diagnostic (design doc section 4.3).

    NOT an intrinsic mean/Gaussian/normal/geodesic curvature estimate -- a
    directional second derivative along the (UV-projected, world-metric
    -normalized) outward direction, used only to bound how much a first-order
    strip could be missing at ``continuation_extent``.
    """

    torch = require_torch()
    a = (Su * Su).sum(dim=1)
    d = (Sv * Sv).sum(dim=1)
    b = (Su * Sv).sum(dim=1)
    rhs_u = (Su * outward_tangent_world).sum(dim=1)
    rhs_v = (Sv * outward_tangent_world).sum(dim=1)
    det = (a * d - b * b).clamp_min(eps)
    q_u_raw = (d * rhs_u - b * rhs_v) / det
    q_v_raw = (a * rhs_v - b * rhs_u) / det
    jq_raw = q_u_raw[:, None] * Su + q_v_raw[:, None] * Sv
    norm_jq = jq_raw.norm(dim=1).clamp_min(eps)
    q_u = q_u_raw / norm_jq
    q_v = q_v_raw / norm_jq

    second_order_direction = (
        (q_u * q_u)[:, None] * Suu + 2.0 * (q_u * q_v)[:, None] * Suv + (q_v * q_v)[:, None] * Svv
    )
    displacement = 0.5 * (float(continuation_extent) ** 2) * second_order_direction
    displacement_norm = displacement.norm(dim=1)
    growth_ratio = displacement_norm / max(float(continuation_extent), eps)

    return {
        "growth_ratio_max": float(growth_ratio.max().cpu()),
        "growth_ratio_mean": float(growth_ratio.mean().cpu()),
        "displacement_max_norm": float(displacement_norm.max().cpu()),
    }


def _strip_orientation_flip_count(normal_grid: Any, valid_mask: Any, closed: bool) -> int:
    """Phase D's own strip-adjacency orientation-consistency check.

    Deliberately NOT the shared `compute_orientation_consistency` (which is a
    single-reference, adjacency-independent check reused as-is from
    `torch_annulus_chart.py`) -- a continuation strip's own topology
    (generally an open 1D sequence of boundary samples, optionally closed) is
    different from annulus's ring/pairwise-slice-reference topology, so this
    module owns its own adjacency logic (design doc section 9, prerequisite 2).
    """

    s_count, t_count = int(normal_grid.shape[0]), int(normal_grid.shape[1])
    if s_count < 2:
        return 0
    pairs = [(i, i + 1) for i in range(s_count - 1)]
    if closed:
        pairs.append((s_count - 1, 0))
    flips = 0
    for j in range(t_count):
        column_normal = normal_grid[:, j, :]
        column_valid = valid_mask[:, j]
        for i, k in pairs:
            if bool(column_valid[i]) and bool(column_valid[k]):
                if float((column_normal[i] * column_normal[k]).sum()) < 0.0:
                    flips += 1
    return flips


def build_continuation_domain(
    surface: TorchNURBSSurface,
    boundary: PatchBoundarySegment,
    *,
    expected_patch_id: int | None = None,
    extent_multiplier: float = 1.0,
    local_surface_scale: float | None = None,
    arclength_epsilon: float = 1e-6,
    t_count: int = 5,
    jacobian_eps: float = 1e-8,
    second_order_growth_threshold: float = 0.5,
) -> ContinuationDomain:
    """Build one boundary-local world-space sampled continuation strip.

    Raises ``ValueError`` for input-contract violations, `ContinuationDomainBuildError`
    when a grid/AABB cannot be constructed at all, and otherwise always
    returns a `ContinuationDomain` (``state`` may be ``degenerate``/``rejected``
    for post-construction quality problems -- never silently dropped, see
    design doc sections 2.2-2.4 and 5).
    """

    torch = require_torch()

    # --- Input contract (eager ValueError, design doc section 2.2) ---
    if boundary.state == BOUNDARY_RECONCILED_INTERNAL:
        raise ValueError(
            f"Boundary {boundary.boundary_id!r} is reconciled_internal -- already a resolved "
            "internal seam, not a continuation target."
        )
    if expected_patch_id is not None and int(expected_patch_id) != int(boundary.patch_id):
        raise ValueError(
            f"expected_patch_id={expected_patch_id!r} does not match boundary.patch_id={boundary.patch_id!r}."
        )
    if local_surface_scale is not None and not (math.isfinite(local_surface_scale) and local_surface_scale > 0.0):
        raise ValueError(f"local_surface_scale must be finite and positive, got {local_surface_scale!r}.")
    if not (math.isfinite(extent_multiplier) and extent_multiplier > 0.0):
        raise ValueError(f"extent_multiplier must be finite and positive, got {extent_multiplier!r}.")

    uv, world, inner_uv, inner_world = _strip_closed_duplicate(boundary)
    if not (int(uv.shape[0]) == int(world.shape[0]) == int(inner_uv.shape[0]) == int(inner_world.shape[0])):
        raise ValueError(
            "boundary uv/world/inner_uv/inner_world sample counts do not match "
            f"({uv.shape[0]}, {world.shape[0]}, {inner_uv.shape[0]}, {inner_world.shape[0]})."
        )

    s_count = int(world.shape[0])
    min_required = 4 if boundary.closed else 3
    if s_count < min_required:
        raise ValueError(
            f"Boundary {boundary.boundary_id!r} has {s_count} unique samples, "
            f"below the minimum ({min_required}) for {'closed' if boundary.closed else 'open'} boundaries."
        )

    segment_lengths = (world[1:] - world[:-1]).norm(dim=1) if s_count > 1 else world.new_zeros(0)
    if boundary.closed:
        closing_length = (world[0] - world[-1]).norm()
        all_lengths = torch.cat([segment_lengths, closing_length.reshape(1)])
    else:
        all_lengths = segment_lengths
    if bool((all_lengths <= arclength_epsilon).any()):
        raise ValueError(
            f"Boundary {boundary.boundary_id!r} has an adjacent sample pair (or closing segment) "
            f"with world distance <= arclength_epsilon={arclength_epsilon!r}."
        )

    # --- s_world / boundary_length (section 4.1) ---
    s_world, boundary_length = _arclength_metadata(world, bool(boundary.closed))

    # --- world tangent T via world-arclength differencing (section 4.1) ---
    tangent_world_t = _world_arclength_tangent(world, s_world, boundary_length, bool(boundary.closed))

    # --- analytic surface derivatives at boundary samples ---
    _, su, sv = surface.evaluate_with_derivatives(uv)
    su, sv = su.detach(), sv.detach()

    # --- outward direction, world-space only (section 4.2) ---
    normal_raw = torch.cross(su, sv, dim=1)
    normal_norm = normal_raw.norm(dim=1)
    tangent_norm = tangent_world_t.norm(dim=1)
    surface_normal_unit = normal_raw / normal_norm.clamp_min(jacobian_eps)[:, None]
    cross_candidate_raw = torch.cross(surface_normal_unit, tangent_world_t, dim=1)
    cross_candidate_norm = cross_candidate_raw.norm(dim=1)
    direction_valid_mask = (
        (normal_norm > jacobian_eps) & (tangent_norm > jacobian_eps) & (cross_candidate_norm > jacobian_eps)
    )
    cross_candidate_unit = cross_candidate_raw / cross_candidate_norm.clamp_min(jacobian_eps)[:, None]
    sign = torch.where(
        (cross_candidate_unit * (inner_world - world)).sum(dim=1) > 0.0,
        world.new_tensor(-1.0),
        world.new_tensor(1.0),
    )
    outward_tangent_world = torch.where(
        direction_valid_mask[:, None], sign[:, None] * cross_candidate_unit, torch.zeros_like(cross_candidate_unit)
    )

    if not bool(direction_valid_mask.any()):
        raise ContinuationDomainBuildError(
            f"Boundary {boundary.boundary_id!r}: no valid outward direction could be computed at "
            "ANY sample (surface normal, boundary tangent, or their cross product degenerate "
            "everywhere) -- cannot construct a continuation grid at all."
        )

    # --- local_surface_scale / continuation_extent (section 4.5) ---
    explicit_scale = local_surface_scale is not None
    if not explicit_scale:
        local_surface_scale = _derive_local_surface_scale(world, bool(boundary.closed), inner_world, surface.control_grid)
    continuation_extent = float(extent_multiplier) * float(local_surface_scale)

    # --- second-order direction diagnostic (section 4.3, diagnostic only) ---
    _, _, _, suu, suv, svv = surface.evaluate_with_second_derivatives(uv)
    suu, suv, svv = suu.detach(), suv.detach(), svv.detach()
    second_order = _second_order_direction_diagnostic(
        su, sv, suu, suv, svv, outward_tangent_world, continuation_extent, jacobian_eps
    )

    # --- canonical first-order grid (section 4.4) ---
    d_outward_ds = _world_arclength_tangent(outward_tangent_world, s_world, boundary_length, bool(boundary.closed))
    t_world = torch.linspace(0.0, continuation_extent, int(t_count), dtype=world.dtype, device=world.device)
    world_grid = world[:, None, :] + t_world[None, :, None] * outward_tangent_world[:, None, :]
    tangent_t_grid = outward_tangent_world[:, None, :].expand(-1, int(t_count), -1).clone()
    tangent_s_grid = tangent_world_t[:, None, :] + t_world[None, :, None] * d_outward_ds[:, None, :]

    normal_raw_grid = torch.cross(tangent_s_grid, tangent_t_grid, dim=-1)
    normal_norm_grid = normal_raw_grid.norm(dim=-1)
    normal_valid_mask = normal_norm_grid > jacobian_eps
    normal_grid = torch.where(
        normal_valid_mask[..., None],
        normal_raw_grid / normal_norm_grid.clamp_min(jacobian_eps)[..., None],
        torch.zeros_like(normal_raw_grid),
    )

    finite_mask = (
        torch.isfinite(world_grid).all(dim=-1)
        & torch.isfinite(tangent_s_grid).all(dim=-1)
        & torch.isfinite(tangent_t_grid).all(dim=-1)
    )

    flat_tangent_s = tangent_s_grid.reshape(-1, 3)
    flat_tangent_t = tangent_t_grid.reshape(-1, 3)
    jacobian_metrics = compute_parametric_jacobian_metrics(
        flat_tangent_s, flat_tangent_t, eps=jacobian_eps, scale=local_surface_scale
    )
    sigma_min_grid = jacobian_metrics["sigma_min"].reshape(s_count, int(t_count))
    jacobian_valid_mask = sigma_min_grid >= jacobian_eps

    sample_valid_mask = finite_mask & jacobian_valid_mask & direction_valid_mask[:, None]

    orientation_flip_count = _strip_orientation_flip_count(normal_grid, sample_valid_mask, bool(boundary.closed))
    reference_orientation = compute_orientation_consistency(
        normal_raw_grid.reshape(-1, 3), valid_mask=sample_valid_mask.reshape(-1), eps=jacobian_eps
    )

    norm_s = tangent_s_grid.norm(dim=-1)
    norm_t = tangent_t_grid.norm(dim=-1)
    anisotropy = torch.minimum(norm_s, norm_t) / torch.maximum(norm_s, norm_t).clamp_min(jacobian_eps)
    orthogonality = (tangent_s_grid * tangent_t_grid).sum(dim=-1).abs() / (norm_s * norm_t).clamp_min(jacobian_eps)

    valid_fraction = float(sample_valid_mask.float().mean().cpu())
    growth_ratio_max = second_order["growth_ratio_max"]

    reasons: list[str] = []
    if valid_fraction <= 0.0:
        state, reason = STATE_REJECTED, "non_finite_domain"
    else:
        if valid_fraction < 1.0:
            reasons.append("partial_direction_or_jacobian_degeneracy")
        if orientation_flip_count > 0:
            reasons.append("orientation_inconsistency")
        if growth_ratio_max > second_order_growth_threshold:
            reasons.append("excessive_second_order_growth")
        if reasons:
            state, reason = STATE_DEGENERATE, ";".join(reasons)
        else:
            state, reason = STATE_VALID, "ok"

    if bool(sample_valid_mask.any()):
        valid_points = world_grid[sample_valid_mask]
    else:
        valid_points = world_grid[:, 0, :]
    aabb_min = valid_points.min(dim=0).values.detach()
    aabb_max = valid_points.max(dim=0).values.detach()

    validity = {
        "min_area_jacobian": jacobian_metrics["min_area_jacobian"],
        "min_jacobian_singular_value": jacobian_metrics["min_jacobian_singular_value"],
        "min_jacobian_singular_value_normalized": jacobian_metrics["min_jacobian_singular_value_normalized"],
        "jacobian_condition_mean": jacobian_metrics["jacobian_condition_mean"],
        "jacobian_condition_p95": jacobian_metrics["jacobian_condition_p95"],
        "max_jacobian_condition": jacobian_metrics["max_jacobian_condition"],
        "near_degenerate_count": jacobian_metrics["near_degenerate_count"],
        "orientation_flip_count": orientation_flip_count,
        "reference_orientation_flip_count": reference_orientation["orientation_flip_count"],
        "anisotropy_mean": float(anisotropy.mean().cpu()),
        "anisotropy_min": float(anisotropy.min().cpu()),
        "orthogonality_mean": float(orthogonality.mean().cpu()),
        "orthogonality_max": float(orthogonality.max().cpu()),
        "valid_fraction": valid_fraction,
        "self_intersection_checked": False,
        "visible_surface_penetration_checked": False,
        "adjacent_domain_overlap_checked": False,
    }

    uncertainty = {
        "second_order_growth_ratio_max": second_order["growth_ratio_max"],
        "second_order_growth_ratio_mean": second_order["growth_ratio_mean"],
        "second_order_displacement_at_extent_max_norm": second_order["displacement_max_norm"],
        "boundary_confidence": dict(boundary.confidence),
    }

    provenance = {
        "source_kind": boundary.source_kind,
        "boundary_state": boundary.state,
        "control_edge": boundary.control_edge,
        "local_surface_scale_source": "explicit" if explicit_scale else "automatic",
    }

    return ContinuationDomain(
        domain_id=f"{boundary.boundary_id}:continuation",
        source_patch_id=int(boundary.patch_id),
        source_boundary_id=boundary.boundary_id,
        closed=bool(boundary.closed),
        s_count=s_count,
        t_count=int(t_count),
        s_world=s_world.detach(),
        boundary_length=float(boundary_length),
        t_world=t_world.detach(),
        world=world_grid.detach(),
        tangent_s=tangent_s_grid.detach(),
        tangent_t=tangent_t_grid.detach(),
        normal=normal_grid.detach(),
        outward_tangent_world=outward_tangent_world.detach(),
        normal_valid_mask=normal_valid_mask.detach(),
        direction_valid_mask=direction_valid_mask.detach(),
        sample_valid_mask=sample_valid_mask.detach(),
        local_surface_scale=float(local_surface_scale),
        continuation_extent=float(continuation_extent),
        extent_multiplier=float(extent_multiplier),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        state=state,
        reason=reason,
        validity=validity,
        uncertainty=uncertainty,
        provenance=provenance,
    )
