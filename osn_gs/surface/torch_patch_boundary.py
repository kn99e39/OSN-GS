from __future__ import annotations

"""Stable patch-boundary records for boundary-conditioned NURBS construction.

The Phase-2 support mask is converted into ordered, oriented UV loops whose
supported interior is always on the left. Every record carries a nearby
interior isocurve and NURBS-derived world tangents/normals. This module does
not classify semantic object boundaries or create occluded geometry.
"""

from dataclasses import dataclass, field
import math
from typing import Any

from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.utils.torch_ops import require_torch

BOUNDARY_UNCLASSIFIED = "unclassified"
BOUNDARY_RECONCILED_INTERNAL = "reconciled_internal"
BOUNDARY_UNSUPPORTED = "unsupported"
BOUNDARY_EXTENSION_CANDIDATE = "extension_candidate"
BOUNDARY_STATES = {
    BOUNDARY_UNCLASSIFIED,
    BOUNDARY_RECONCILED_INTERNAL,
    BOUNDARY_UNSUPPORTED,
    BOUNDARY_EXTENSION_CANDIDATE,
}


@dataclass
class PatchBoundarySegment:
    boundary_id: str
    patch_id: int
    source_kind: str
    uv: Any
    world: Any
    inner_uv: Any
    inner_world: Any
    tangent_world: Any
    inward_tangent_world: Any
    normal_world: Any
    closed: bool
    orientation: str
    interior_side: str = "left"
    state: str = BOUNDARY_UNCLASSIFIED
    control_edge: str | None = None
    adjacent_patch_id: int | None = None
    adjacent_boundary_id: str | None = None
    confidence: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in BOUNDARY_STATES:
            raise ValueError(f"Unknown patch-boundary state: {self.state!r}")
        if self.interior_side != "left":
            raise ValueError("Patch boundary records use the invariant interior_side='left'.")

    def payload(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "patch_id": int(self.patch_id),
            "source_kind": self.source_kind,
            "closed": bool(self.closed),
            "orientation": self.orientation,
            "interior_side": self.interior_side,
            "state": self.state,
            "control_edge": self.control_edge,
            "adjacent_patch_id": self.adjacent_patch_id,
            "adjacent_boundary_id": self.adjacent_boundary_id,
            "uv": self.uv.detach().cpu().tolist(),
            "world": self.world.detach().cpu().tolist(),
            "inner_uv": self.inner_uv.detach().cpu().tolist(),
            "inner_world": self.inner_world.detach().cpu().tolist(),
            "tangent_world": self.tangent_world.detach().cpu().tolist(),
            "inward_tangent_world": self.inward_tangent_world.detach().cpu().tolist(),
            "normal_world": self.normal_world.detach().cpu().tolist(),
            "confidence": dict(self.confidence),
            "provenance": dict(self.provenance),
        }


def _edge_turn(incoming: tuple[int, int], outgoing: tuple[int, int]) -> float:
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    return math.atan2(float(cross), float(dot))


def _trace_oriented_mask_loops(mask: Any) -> list[list[tuple[int, int]]]:
    """Trace cell-union boundary loops on integer grid vertices.

    Each directed cell edge is oriented so a supported cell lies on its left.
    Internal shared edges are omitted. At a diagonal ambiguity, the largest
    left turn is selected deterministically, keeping touching regions separate.
    """

    res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for i, j in require_torch().nonzero(mask, as_tuple=False).tolist():
        if i == 0 or not bool(mask[i - 1, j]):
            edges.add(((i, j + 1), (i, j)))
        if i == res_u - 1 or not bool(mask[i + 1, j]):
            edges.add(((i + 1, j), (i + 1, j + 1)))
        if j == 0 or not bool(mask[i, j - 1]):
            edges.add(((i, j), (i + 1, j)))
        if j == res_v - 1 or not bool(mask[i, j + 1]):
            edges.add(((i + 1, j + 1), (i, j + 1)))

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    for values in outgoing.values():
        values.sort()

    unvisited = set(edges)
    loops: list[list[tuple[int, int]]] = []
    while unvisited:
        start, end = min(unvisited)
        unvisited.remove((start, end))
        chain = [start, end]
        while chain[-1] != chain[0]:
            current = chain[-1]
            candidates = [candidate for candidate in outgoing.get(current, []) if (current, candidate) in unvisited]
            if not candidates:
                break
            incoming = (current[0] - chain[-2][0], current[1] - chain[-2][1])
            ranked = []
            for candidate in candidates:
                direction = (candidate[0] - current[0], candidate[1] - current[1])
                ranked.append((-_edge_turn(incoming, direction), candidate))
            _, next_vertex = min(ranked)
            unvisited.remove((current, next_vertex))
            chain.append(next_vertex)
        loops.append(chain)
    return loops


def _signed_area(uv: Any) -> float:
    points = uv[:-1] if int(uv.shape[0]) > 1 and bool((uv[0] == uv[-1]).all()) else uv
    if int(points.shape[0]) < 3:
        return 0.0
    following = points.roll(-1, dims=0)
    return float((0.5 * (points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]).sum()).detach().cpu())


def _canonicalize_closed_loop(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(vertices) < 2 or vertices[0] != vertices[-1]:
        return vertices
    unique = vertices[:-1]
    start_index = min(range(len(unique)), key=lambda index: unique[index])
    ordered = unique[start_index:] + unique[:start_index]
    return ordered + [ordered[0]]


def _curve_tangents(points: Any, closed: bool) -> Any:
    torch = require_torch()
    if int(points.shape[0]) < 2:
        return torch.zeros_like(points)
    if closed:
        unique = points[:-1]
        tangent = unique.roll(-1, dims=0) - unique.roll(1, dims=0)
        tangent = torch.nn.functional.normalize(tangent, dim=1, eps=1e-12)
        return torch.cat((tangent, tangent[:1]), dim=0)
    tangent = torch.empty_like(points)
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    if int(points.shape[0]) > 2:
        tangent[1:-1] = points[2:] - points[:-2]
    return torch.nn.functional.normalize(tangent, dim=1, eps=1e-12)


def _nearest_supported_inner_uv(uv: Any, mask: Any, closed: bool) -> Any:
    torch = require_torch()
    res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
    cells = torch.nonzero(mask, as_tuple=False)
    if int(cells.shape[0]) == 0:
        raise ValueError("Cannot build an inner isocurve from an empty support mask.")
    centers = torch.stack(
        ((cells[:, 0].to(uv.dtype) + 0.5) / res_u, (cells[:, 1].to(uv.dtype) + 0.5) / res_v),
        dim=1,
    ).to(device=uv.device)
    tangent_uv = _curve_tangents(uv, closed)
    left = torch.stack((-tangent_uv[:, 1], tangent_uv[:, 0]), dim=1)
    cell_step = min(1.0 / res_u, 1.0 / res_v)
    target = (uv + 0.75 * cell_step * left).clamp(0.0, 1.0)
    nearest = torch.cdist(target, centers).argmin(dim=1)
    inner = centers[nearest]
    if closed and int(inner.shape[0]) > 1:
        inner[-1] = inner[0]
    return inner


def _make_record(
    surface: TorchNURBSSurface,
    patch_id: int,
    boundary_id: str,
    source_kind: str,
    uv: Any,
    inner_uv: Any,
    closed: bool,
    orientation: str,
    state: str = BOUNDARY_UNCLASSIFIED,
    control_edge: str | None = None,
    adjacent_patch_id: int | None = None,
    adjacent_boundary_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> PatchBoundarySegment:
    torch = require_torch()
    uv = torch.as_tensor(uv, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    inner_uv = torch.as_tensor(inner_uv, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    world = surface.evaluate(uv).detach()
    inner_world = surface.evaluate(inner_uv).detach()
    tangent_world = _curve_tangents(world, closed)
    inward = torch.nn.functional.normalize(inner_world - world, dim=1, eps=1e-12)
    _, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    cross = torch.cross(deriv_u, deriv_v, dim=1)
    normal = torch.nn.functional.normalize(cross, dim=1, eps=1e-12).detach()
    inner_distance = (inner_world - world).norm(dim=1)
    jacobian = cross.norm(dim=1)
    return PatchBoundarySegment(
        boundary_id=boundary_id,
        patch_id=int(patch_id),
        source_kind=source_kind,
        uv=uv.detach(),
        world=world,
        inner_uv=inner_uv.detach(),
        inner_world=inner_world,
        tangent_world=tangent_world,
        inward_tangent_world=inward,
        normal_world=normal,
        closed=bool(closed),
        orientation=orientation,
        state=state,
        control_edge=control_edge,
        adjacent_patch_id=adjacent_patch_id,
        adjacent_boundary_id=adjacent_boundary_id,
        confidence={
            "inner_distance_min": float(inner_distance.min().cpu()),
            "inner_distance_median": float(inner_distance.median().cpu()),
            "jacobian_min": float(jacobian.min().cpu()),
            "jacobian_median": float(jacobian.median().cpu()),
        },
        provenance={} if provenance is None else dict(provenance),
    )


def extract_trimmed_patch_boundaries(
    patch_id: int,
    surface: TorchNURBSSurface,
    source_kind: str = "trim_support_loop",
) -> list[PatchBoundarySegment]:
    """Convert a patch trim mask into stable ordered/oriented boundary loops."""

    torch = require_torch()
    if surface.uv_support_mask is None:
        raise ValueError("Trimmed boundary extraction requires surface.uv_support_mask.")
    mask = torch.as_tensor(surface.uv_support_mask, dtype=torch.bool, device=surface.control_grid.device)
    res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
    traced = [_canonicalize_closed_loop(loop) for loop in _trace_oriented_mask_loops(mask)]
    prepared = []
    for loop in traced:
        if len(loop) < 4 or loop[0] != loop[-1]:
            continue
        uv = torch.tensor(
            [[i / res_u, j / res_v] for i, j in loop],
            dtype=surface.control_grid.dtype,
            device=surface.control_grid.device,
        )
        area = _signed_area(uv)
        prepared.append((uv, area))
    prepared.sort(key=lambda item: (-abs(item[1]), float(item[0][0, 0]), float(item[0][0, 1])))

    records = []
    for loop_index, (uv, area) in enumerate(prepared):
        inner_uv = _nearest_supported_inner_uv(uv, mask, closed=True)
        records.append(
            _make_record(
                surface,
                patch_id,
                f"p{int(patch_id)}:trim:{loop_index}",
                source_kind,
                uv,
                inner_uv,
                closed=True,
                orientation="ccw" if area > 0.0 else "cw",
                provenance={
                    "support_resolution": [res_u, res_v],
                    "signed_uv_area": area,
                    "ordered": True,
                    "interior_on_left": True,
                },
            )
        )
    return records


def build_rectangular_patch_edge(
    patch_id: int,
    surface: TorchNURBSSurface,
    edge: str,
    sample_count: int = 33,
    source_kind: str = "chart_edge",
    state: str = BOUNDARY_UNCLASSIFIED,
    adjacent_patch_id: int | None = None,
    adjacent_boundary_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> PatchBoundarySegment:
    """Create an oriented rectangular chart edge with an interior isocurve."""

    torch = require_torch()
    edge = str(edge).lower()
    if sample_count < 2:
        raise ValueError("A patch edge needs at least two samples.")
    t = torch.linspace(0.0, 1.0, sample_count, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    delta = min(0.25, 1.0 / max(int(surface.control_grid.shape[0]), int(surface.control_grid.shape[1]), 2))
    if edge == "u0":
        uv = torch.stack((torch.zeros_like(t), t.flip(0)), dim=1)
        inner_uv = torch.stack((torch.full_like(t, delta), t.flip(0)), dim=1)
    elif edge == "u1":
        uv = torch.stack((torch.ones_like(t), t), dim=1)
        inner_uv = torch.stack((torch.full_like(t, 1.0 - delta), t), dim=1)
    elif edge == "v0":
        uv = torch.stack((t, torch.zeros_like(t)), dim=1)
        inner_uv = torch.stack((t, torch.full_like(t, delta)), dim=1)
    elif edge == "v1":
        uv = torch.stack((t.flip(0), torch.ones_like(t)), dim=1)
        inner_uv = torch.stack((t.flip(0), torch.full_like(t, 1.0 - delta)), dim=1)
    else:
        raise ValueError(f"Unknown rectangular patch edge: {edge!r}")
    return _make_record(
        surface,
        patch_id,
        f"p{int(patch_id)}:edge:{edge}",
        source_kind,
        uv,
        inner_uv,
        closed=False,
        orientation="open",
        state=state,
        control_edge=edge,
        adjacent_patch_id=adjacent_patch_id,
        adjacent_boundary_id=adjacent_boundary_id,
        provenance={"ordered": True, "interior_on_left": True, **({} if provenance is None else provenance)},
    )


def patch_boundaries_payload(boundaries: list[PatchBoundarySegment]) -> list[dict[str, Any]]:
    return [boundary.payload() for boundary in boundaries]