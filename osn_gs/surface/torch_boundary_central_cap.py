from __future__ import annotations

"""Observed-anchor central caps for disk-like Boundary-first components."""

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


def _resample_closed(points: Any, count: int) -> Any:
    torch = require_torch()
    ends = torch.cat((points[1:], points[:1]), dim=0)
    lengths = torch.linalg.vector_norm(ends - points, dim=1)
    cumulative = torch.cat((torch.zeros((1,), dtype=points.dtype, device=points.device), lengths.cumsum(0)))
    distances = torch.arange(count, dtype=points.dtype, device=points.device) / float(count) * cumulative[-1]
    indices = (torch.searchsorted(cumulative, distances, right=True) - 1).clamp(0, count - 1)
    indices = indices.clamp_max(int(lengths.numel()) - 1)
    fraction = ((distances - cumulative[indices]) / lengths[indices]).unsqueeze(1)
    return points[indices] + fraction * (ends[indices] - points[indices])


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
    samples = _resample_closed(boundary, int(segment_count))
    pole = torch.as_tensor(anchor.point, dtype=boundary.dtype, device=boundary.device)
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
    })