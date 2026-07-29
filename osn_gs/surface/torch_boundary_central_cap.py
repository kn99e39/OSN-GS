from __future__ import annotations

"""Observed-anchor central caps for disk-like Boundary-first components."""

import math
from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_boundary_support_network import ObservedBoundaryCurve
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class ObservedInteriorAnchor:
    point: Any
    source_point_index: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class BoundaryCentralCapResult:
    state: str
    reason: str | None
    surfaces: tuple[TorchNURBSSurface, ...]
    provenance: dict[str, Any]


def select_observed_interior_anchor(points: Any, frame: Any, *, source_indices: Any | None = None, support_mask: Any | None = None) -> ObservedInteriorAnchor:
    """Choose an actual observed point nearest the component UV medoid.

    The function never synthesizes a centroid as geometry; the selected anchor
    is one of the input observations and retains its original source index.
    """
    torch = require_torch()
    values = torch.as_tensor(points)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 3:
        raise ValueError('Interior anchor requires at least three finite 3-D observations.')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Interior anchor observations must be finite.')
    uv = frame.apply(values, clamp=False)
    medoid_uv = uv.median(dim=0).values
    selection = 'observed_component_medoid'
    clearance = None
    if support_mask is None:
        local = int(torch.argmin(torch.linalg.vector_norm(uv - medoid_uv, dim=1)))
    else:
        mask = torch.as_tensor(support_mask, dtype=torch.bool, device=values.device)
        if mask.ndim != 2 or not bool(mask.any()) or bool(mask.all()):
            raise ValueError('support_mask must contain both supported and unsupported cells.')
        h, w = int(mask.shape[0]), int(mask.shape[1])
        outside = torch.nonzero(~mask, as_tuple=False).to(dtype=uv.dtype)
        outside[:, 0] = (outside[:, 0] + 0.5) / h
        outside[:, 1] = (outside[:, 1] + 0.5) / w
        distances = torch.cdist(uv, outside).min(dim=1).values
        best = distances.max()
        choices = torch.nonzero(torch.isclose(distances, best), as_tuple=False).reshape(-1)
        local = int(choices[torch.argmin(torch.linalg.vector_norm(uv[choices] - medoid_uv, dim=1))])
        clearance = float(distances[local])
        selection = 'observed_max_support_clearance'
    indices = torch.arange(values.shape[0], device=values.device) if source_indices is None else torch.as_tensor(source_indices, device=values.device)
    if int(indices.numel()) != int(values.shape[0]):
        raise ValueError('source_indices must align with interior anchor observations.')
    return ObservedInteriorAnchor(values[local].detach().clone(), int(indices[local]), {
        'source_kind': selection, 'source_point_index': int(indices[local]),
        'uv_medoid': medoid_uv.detach().cpu().tolist(), 'selected_uv': uv[local].detach().cpu().tolist(), 'support_clearance_uv': clearance,
    })


def _anchor_relative_angles(boundary: Any, pole: Any) -> Any:
    """Angle of each boundary sample around ``pole``, in the boundary's own best-fit plane."""
    torch = require_torch()
    centered = boundary - boundary.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis_u, axis_v = vh[0], vh[1]
    relative = boundary - pole
    return torch.atan2(relative @ axis_v, relative @ axis_u)


def validate_star_shaped_boundary(
    boundary: Any, pole: Any, *,
    min_monotonicity_ratio: float = 0.85,
    max_step_degrees: float = 30.0,
    min_total_sweep_degrees: float = 300.0,
) -> dict[str, Any]:
    """Check that the ORDERED boundary sweeps monotonically around ``pole``.

    Equal-arclength resampling silently assumes nothing about angular
    distribution; equal-angle resampling requires the anchor to actually see
    every boundary sample at a distinct, monotonically-progressing angle (a
    star-shaped domain w.r.t. the anchor). A boundary that fails this check
    cannot be safely angle-resampled -- callers must not fall back to naive
    angle sorting in that case, only report insufficient support.

    ``monotonicity_ratio = |net angular sweep| / (sum of |each step|)``: 1.0
    for a boundary that only ever progresses in one direction around the
    anchor, dropping below 1.0 as more of the walk backtracks. This tolerates
    many small near-tangent-ray reversals (their magnitude barely dents the
    ratio) while still catching genuine non-star-shaped structure (a real
    concavity or an anchor outside the loop produces a large, low ratio --
    confirmed empirically: known-concave ``u_shape`` measures ~0.64 and
    degenerate/non-enclosing components measure ~0.0, while every genuinely
    star-shaped scene checked measures >= 0.85). A single large angular jump
    between adjacent RAW samples, or a total net sweep well short of a full
    turn (the anchor isn't actually encircled), are separate hard rejections.
    """
    torch = require_torch()
    angles = _anchor_relative_angles(boundary, pole)
    diffs = angles[1:] - angles[:-1]
    closing = (angles[0] - angles[-1]).unsqueeze(0)
    steps = torch.cat((diffs, closing))
    wrapped = (steps + math.pi) % (2 * math.pi) - math.pi
    net_sweep = float(wrapped.sum())
    total_absolute_variation = float(wrapped.abs().sum())
    monotonicity_ratio = abs(net_sweep) / total_absolute_variation if total_absolute_variation > 1e-12 else 0.0
    direction = 1.0 if net_sweep >= 0 else -1.0
    max_step_value = float(wrapped.abs().max()) * 180.0 / math.pi if int(wrapped.numel()) else 0.0
    total_sweep_degrees = abs(net_sweep) * 180.0 / math.pi
    is_valid = (
        monotonicity_ratio >= min_monotonicity_ratio
        and max_step_value <= max_step_degrees
        and total_sweep_degrees >= min_total_sweep_degrees
    )
    return {
        "is_valid": bool(is_valid),
        "direction": direction,
        "monotonicity_ratio": monotonicity_ratio,
        "max_angular_step_degrees": max_step_value,
        "total_angular_sweep_degrees": total_sweep_degrees,
        "total_steps": int(wrapped.numel()),
        "thresholds": {
            "min_monotonicity_ratio": min_monotonicity_ratio,
            "max_step_degrees": max_step_degrees,
            "min_total_sweep_degrees": min_total_sweep_degrees,
        },
    }


def _resample_closed_by_angle(boundary: Any, pole: Any, count: int, *, direction: float) -> Any:
    """Equal-ANGLE resampling around ``pole`` -- only valid on a star-shaped boundary.

    Same arclength-parameter interpolation pattern as a Euclidean resample,
    substituting per-step angular sweep (in the validated dominant direction)
    for per-step Euclidean length, so fan segments get even angular coverage
    around the anchor instead of whatever an equal-arclength walk happens to
    produce (which can bunch nearly all samples into one narrow wedge and
    dump the rest of the sweep into a single under-constrained segment --
    the confirmed root cause of the crossing defects in worklog 110/112).
    """
    torch = require_torch()
    ends = torch.cat((boundary[1:], boundary[:1]), dim=0)
    angles = _anchor_relative_angles(boundary, pole)
    diffs = angles[1:] - angles[:-1]
    closing = (angles[0] - angles[-1]).unsqueeze(0)
    steps = torch.cat((diffs, closing))
    wrapped = (steps + math.pi) % (2 * math.pi) - math.pi
    lengths = torch.clamp(wrapped * direction, min=1e-9)
    cumulative = torch.cat((torch.zeros((1,), dtype=boundary.dtype, device=boundary.device), lengths.cumsum(0)))
    target = torch.arange(count, dtype=boundary.dtype, device=boundary.device) / float(count) * cumulative[-1]
    indices = (torch.searchsorted(cumulative, target, right=True) - 1).clamp(0, int(lengths.numel()) - 1)
    fraction = ((target - cumulative[indices]) / lengths[indices]).unsqueeze(1)
    return boundary[indices] + fraction * (ends[indices] - boundary[indices])


def build_boundary_central_cap(outer: ObservedBoundaryCurve, anchor: ObservedInteriorAnchor, *, segment_count: int = 8) -> BoundaryCentralCapResult:
    """Materialize observed-anchor support with cubic outer-boundary segments.

    The pole remains explicit, but raster contour cells are not converted into
    one degree-one patch each.  Every segment has a cubic outer boundary and
    exact endpoint seams; quality validation must still treat the pole as a
    singularity.
    """
    torch = require_torch()
    if not outer.closed:
        return BoundaryCentralCapResult('unsupported', 'closed_outer_boundary_required', (), {})
    boundary = torch.as_tensor(outer.world)
    if boundary.ndim != 2 or boundary.shape[1] != 3:
        raise ValueError('Central cap requires a closed outer boundary with shape (N, 3).')
    if boundary.shape[0] > 1 and bool(torch.allclose(boundary[0], boundary[-1])):
        boundary = boundary[:-1]
    if boundary.shape[0] < 3 or segment_count < 3:
        raise ValueError('Central cap requires at least three boundary samples and segments.')
    pole = torch.as_tensor(anchor.point, dtype=boundary.dtype, device=boundary.device)
    star_shape = validate_star_shaped_boundary(boundary, pole)
    if not star_shape['is_valid']:
        return BoundaryCentralCapResult('unsupported', 'insufficient_observed_interior_support', (), {
            'materialization': 'shared_observed_anchor_cubic_fan', 'outer_boundary_id': outer.boundary_id,
            'anchor': dict(anchor.provenance), 'star_shape_validation': star_shape, 'fallback_used': False,
        })
    samples = _resample_closed_by_angle(boundary, pole, int(segment_count), direction=star_shape['direction'])
    surfaces = []
    for index in range(int(segment_count)):
        previous = samples[(index - 1) % segment_count]
        start = samples[index]
        end = samples[(index + 1) % segment_count]
        following = samples[(index + 2) % segment_count]
        outer_controls = torch.stack((start, start + (end - previous) / 6.0, end - (following - start) / 6.0, end))
        grid = torch.stack((torch.stack((pole, pole, pole, pole)), outer_controls), dim=1)
        surfaces.append(TorchNURBSSurface(control_grid=grid, weights=torch.ones((4, 2), dtype=grid.dtype, device=grid.device), degree_u=3, degree_v=1, uv_support_mask=torch.ones((3, 1), dtype=torch.bool, device=grid.device)))
    return BoundaryCentralCapResult('constructed_central_cap', None, tuple(surfaces), {
        'materialization': 'shared_observed_anchor_cubic_fan', 'outer_boundary_id': outer.boundary_id,
        'anchor': dict(anchor.provenance), 'pole_singularity': 'explicit_shared_observed_anchor',
        'patch_count': len(surfaces), 'segment_count': int(segment_count), 'outer_boundary_degree': 3, 'fallback_used': False,
        'boundary_correspondence': 'equal_angle_star_shaped', 'star_shape_validation': star_shape,
    })