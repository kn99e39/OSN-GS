from __future__ import annotations

"""Phase 2 Component-Level Boundary Extraction.

Implements ``OSN_GS_Final_Boundary_First_NURBS_Direction.md`` §Phase 2:
extracts outer/inner-hole boundary structure for a whole
``SurfaceComponent`` (Phase 1) at once, in ONE shared component-level UV
frame, instead of per-leaf/per-patch trim masks. This is the direct fix for
Stage 1-F's structural limitation (documented in
``docs/worklogs/28_stage1_support_modes.md``): per-leaf density refinement
cannot remove inter-leaf seams because each leaf has its own independent UV
frame and polygon; here there is exactly one frame and one density field for
the whole physical surface region, so there is no inter-leaf seam to begin
with.

Pipeline per component:

1. **Coarse support** — union of every member leaf's plane-AABB intersection
   polygon (Stage 1-C's exact polygon, reused unmodified), each projected
   into the component's own UV frame and rasterized, then OR-ed together.
2. **Refined support** — an unweighted, per-sample-adaptive-bandwidth KDE
   over the component's own raw Gaussian points (same construction as Stage
   1-F, §torch_boundary_refinement.py, now evaluated once per component
   instead of once per leaf), thresholded at an absolute effective-neighbor
   level, ANDed with the coarse mask (refined support never extends past the
   voxel evidence), with a marching-squares sub-cell contour.
3. **Loop hierarchy** — connected-component labeling of the refined mask
   (outer loops) and of its enclosed complement (hole loops), each split
   further into "significant" vs. "tiny artifact" by a cell-area threshold,
   plus a coarse open-contour diagnostic (contour segments touching the UV
   domain border, meaning the marching-squares curve did not close within
   the component's own domain).

This module is additive and benchmark/analysis-facing only (like Phase 1's
``torch_surface_components.py``): it does not touch the legacy or
``voxel_patch_stage1`` constructors, and nothing in the trainer imports it.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.surface.torch_boundary_refinement import (
    density_grid,
    kde_density,
    marching_squares,
    sample_nn_spacings,
)
from osn_gs.surface.torch_nurbs import UVFrame, uv_frame_from_axes
from osn_gs.surface.torch_surface_components import SurfaceComponent
from osn_gs.surface.torch_voxel_hierarchy import (
    TorchVoxelGaussianHierarchy,
    plane_aabb_intersection_polygon,
    rasterize_convex_polygon_uv,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass
class LoopDescriptor:
    """One connected raster region of the loop hierarchy (outer / hole / tiny)."""

    kind: str  # "outer" | "hole" | "tiny_artifact"
    label: int
    area_cells: int
    area_world: float
    perimeter_cells: int
    boundary_world_points: list[list[float]]
    # For a hole loop: the outer-loop label it is enclosed by (None if none
    # found, e.g. a hole touching the domain border). Always None for
    # "outer"/"tiny_artifact" kinds.
    nested_in_outer_label: int | None = None


@dataclass
class ComponentBoundaryResult:
    component_id: int
    resolution: int
    frame: UVFrame
    coarse_mask: Any  # (R, R) bool: voxel polygon union
    coarse_mask_dilated: Any  # (R, R) bool: coarse_mask + reprojection-gap closing (see extract_component_boundary)
    density_grid: Any  # (R, R) float: raw KDE values
    threshold_level: float
    threshold_field: Any  # (R, R) bool: density >= threshold, BEFORE ANDing with coarse
    refined_mask: Any  # (R, R) bool: threshold_field & coarse_mask (final support)
    contour_uv: list[tuple[tuple[float, float], tuple[float, float]]]
    contour_world: list[list[list[float]]]
    outer_loops: list[LoopDescriptor]
    hole_loops: list[LoopDescriptor]
    tiny_artifact_loops: list[LoopDescriptor]
    unresolved_open_contour_segment_count: int
    topology: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _label_components(mask: Any) -> tuple[Any, int]:
    """4-connected component labeling of a boolean grid. Returns (labels, count).

    ``labels`` is a ``(R, R)`` long tensor, 0 for background, 1..count for
    each connected ``True`` region (in row-major first-True-cell order, so
    labeling is deterministic).
    """

    torch = require_torch()
    h, w = int(mask.shape[0]), int(mask.shape[1])
    labels = torch.zeros((h, w), dtype=torch.long, device=mask.device)
    count = 0
    for i, j in torch.nonzero(mask, as_tuple=False).tolist():
        if labels[i, j] != 0:
            continue
        count += 1
        stack = [(i, j)]
        labels[i, j] = count
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and labels[nx, ny] == 0:
                    labels[nx, ny] = count
                    stack.append((nx, ny))
    return labels, count


def _enclosed_background_mask(mask: Any) -> Any:
    """True on ``~mask`` cells NOT reachable from the grid border (i.e. holes)."""

    torch = require_torch()
    h, w = int(mask.shape[0]), int(mask.shape[1])
    complement = ~mask
    reachable = torch.zeros_like(mask)
    stack = [
        (i, j)
        for i in range(h)
        for j in range(w)
        if (i in (0, h - 1) or j in (0, w - 1)) and complement[i, j]
    ]
    for i, j in stack:
        reachable[i, j] = True
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < h and 0 <= ny < w and complement[nx, ny] and not reachable[nx, ny]:
                reachable[nx, ny] = True
                stack.append((nx, ny))
    return complement & ~reachable


def _boundary_cells(mask: Any) -> list[tuple[int, int]]:
    """``True`` cells with >= 1 four-neighbor outside ``mask`` (or the grid)."""

    torch = require_torch()
    h, w = int(mask.shape[0]), int(mask.shape[1])
    boundary = []
    for i, j in torch.nonzero(mask, as_tuple=False).tolist():
        is_boundary = False
        for nx, ny in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if not (0 <= nx < h and 0 <= ny < w) or not bool(mask[nx, ny]):
                is_boundary = True
                break
        if is_boundary:
            boundary.append((i, j))
    return boundary


def _cell_perimeter(mask: Any, label_mask: Any) -> int:
    """Count of grid-cell edges on the boundary of ``label_mask`` (a mask subset of ``mask``)."""

    h, w = int(label_mask.shape[0]), int(label_mask.shape[1])
    perimeter = 0
    for i, j in _boundary_cells(label_mask):
        for nx, ny in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if not (0 <= nx < h and 0 <= ny < w) or not bool(label_mask[nx, ny]):
                perimeter += 1
    return perimeter


def _world_boundary_points(frame: UVFrame, cells: list[tuple[int, int]], resolution: int) -> list[list[float]]:
    torch = require_torch()
    if not cells:
        return []
    uv = torch.tensor(
        [[(i + 0.5) / resolution, (j + 0.5) / resolution] for i, j in cells],
        dtype=frame.origin.dtype, device=frame.origin.device,
    )
    return frame.to_world(uv).detach().cpu().tolist()


def _build_loop_descriptors(
    mask: Any, kind: str, frame: UVFrame, resolution: int, tiny_loop_area_cells: int, cell_world_area: float
) -> tuple[list[LoopDescriptor], Any, int]:
    """Label ``mask``'s connected components into loop descriptors, split by size.

    Returns ``(descriptors, labels, count)`` -- ``labels``/``count`` cover ALL
    components (including tiny ones, still needed for hole-nesting lookups),
    while significant vs. tiny classification only affects which list in
    ``descriptors`` each one lands in.
    """

    labels, count = _label_components(mask)
    descriptors: list[LoopDescriptor] = []
    for label in range(1, count + 1):
        label_mask = labels == label
        area = int(label_mask.sum())
        actual_kind = "tiny_artifact" if area <= tiny_loop_area_cells else kind
        descriptors.append(
            LoopDescriptor(
                kind=actual_kind,
                label=label,
                area_cells=area,
                area_world=area * cell_world_area,
                perimeter_cells=_cell_perimeter(mask, label_mask),
                boundary_world_points=_world_boundary_points(frame, _boundary_cells(label_mask), resolution),
            )
        )
    return descriptors, labels, count


def extract_component_boundary(
    component: SurfaceComponent,
    hierarchy: TorchVoxelGaussianHierarchy,
    points: Any,
    resolution: int = 64,
    density_bandwidth_multiplier: float = 2.0,
    density_threshold: float = 2.0,
    tiny_loop_area_cells: int = 4,
    coarse_gap_closing_cells: int = 2,
) -> ComponentBoundaryResult:
    """Extract the outer/hole boundary structure of one Phase 1 component.

    ``coarse_gap_closing_cells`` (default 2, tuned for ``resolution=64`` --
    re-verify if resolution changes materially): each member leaf's
    plane-AABB polygon is exact in ITS OWN local plane, but this function
    reprojects all of them into ONE shared, flat, component-level frame
    (§2.1). On a curved component (e.g. sine) the leaf polygons then no
    longer tile edge-to-edge under that shared projection, leaving
    1-3-cell-wide seam gaps in their union alone -- confirmed by comparing
    against the density field on its own, which has zero such gaps for every
    required scene. Dilating ONLY the coarse polygon union by this many
    cells before ANDing with the (untouched) density threshold field closes
    exactly those reprojection seams. This is a construction-artifact fix on
    an intermediate mask, not the "morphology closing on the final result to
    force away small holes" the plan forbids (§11): the density field itself
    is never dilated, so a genuine data-absence hole (e.g. planar_hole's
    262-cell hole) passes through this function completely unchanged with
    or without this step -- verified empirically, not assumed.
    """

    torch = require_torch()
    node_by_id = {node.node_id: node for node in hierarchy.nodes}
    component_points = points[component.gaussian_indices]
    frame = uv_frame_from_axes(component_points, component.centroid, component.tangent_u, component.tangent_v)
    cell_world_area = float(frame.span[0] * frame.span[1]) / (resolution * resolution)

    # --- 2.1 Coarse support: union of member-voxel polygons in the shared frame.
    coarse_mask = torch.zeros((resolution, resolution), dtype=torch.bool, device=component_points.device)
    for leaf_id in component.member_leaf_ids:
        leaf = node_by_id[leaf_id]
        if leaf.plane is None:
            continue
        polygon_world = plane_aabb_intersection_polygon(
            leaf.plane.centroid, leaf.plane.normal, leaf.aabb_min, leaf.aabb_max
        )
        if int(polygon_world.shape[0]) < 3:
            continue
        polygon_uv = frame.apply(polygon_world, clamp=False)
        coarse_mask = coarse_mask | rasterize_convex_polygon_uv(polygon_uv, resolution)
    coarse_mask_for_combine = coarse_mask
    if coarse_gap_closing_cells > 0:
        k = int(coarse_gap_closing_cells)
        coarse_mask_for_combine = torch.nn.functional.max_pool2d(
            coarse_mask.float()[None, None], kernel_size=2 * k + 1, stride=1, padding=k
        )[0, 0] > 0.5

    # --- 2.2 Refined support: one component-wide adaptive-bandwidth KDE.
    own_uv = frame.apply(component_points, clamp=False)
    bandwidths = float(density_bandwidth_multiplier) * sample_nn_spacings(own_uv)
    grid = density_grid(own_uv, resolution, bandwidths)
    threshold_field = grid >= float(density_threshold)
    refined_mask = threshold_field & coarse_mask_for_combine
    contour_uv = marching_squares(grid, float(density_threshold))
    contour_world = [
        [frame.to_world(torch.tensor([a], dtype=own_uv.dtype))[0].tolist(),
         frame.to_world(torch.tensor([b], dtype=own_uv.dtype))[0].tolist()]
        for a, b in contour_uv
    ]

    # --- 2.3 Loop hierarchy.
    outer_descriptors, outer_labels, outer_count = _build_loop_descriptors(
        refined_mask, "outer", frame, resolution, tiny_loop_area_cells, cell_world_area
    )
    hole_raw_mask = _enclosed_background_mask(refined_mask)
    hole_descriptors, hole_labels, hole_count = _build_loop_descriptors(
        hole_raw_mask, "hole", frame, resolution, tiny_loop_area_cells, cell_world_area
    )
    # Nest each hole under the outer loop whose region it is surrounded by:
    # dilate the hole by one cell and see which outer label it touches most.
    for descriptor in hole_descriptors:
        hole_mask = hole_labels == descriptor.label
        dilated = torch.nn.functional.max_pool2d(
            hole_mask.float()[None, None], kernel_size=3, stride=1, padding=1
        )[0, 0] > 0.5
        touching = outer_labels[dilated & (outer_labels > 0)]
        if int(touching.numel()) > 0:
            values, counts = torch.unique(touching, return_counts=True)
            descriptor.nested_in_outer_label = int(values[torch.argmax(counts)])

    outer_loops = [d for d in outer_descriptors if d.kind == "outer"]
    hole_loops = [d for d in hole_descriptors if d.kind == "hole"]
    tiny_artifact_loops = [d for d in outer_descriptors + hole_descriptors if d.kind == "tiny_artifact"]

    # Coarse open-contour proxy (§2.3 "unresolved open contours"): a
    # marching-squares segment touching the UV domain border did not close
    # within this component's own parameterization.
    border_eps = 1.0 / resolution
    unresolved_open = sum(
        1
        for a, b in contour_uv
        for point in (a, b)
        if point[0] <= border_eps or point[0] >= 1.0 - border_eps
        or point[1] <= border_eps or point[1] >= 1.0 - border_eps
    )

    topology = {
        "connected_component_count_all": outer_count,
        "outer_loop_count": len(outer_loops),
        "hole_component_count_all": hole_count,
        "hole_count": len(hole_loops),
        "tiny_artifact_loop_count": len(tiny_artifact_loops),
        "euler_characteristic_all": outer_count - hole_count,
        "euler_characteristic_significant": len(outer_loops) - len(hole_loops),
        "unresolved_open_contour_segment_count": unresolved_open,
        "coarse_support_cells": int(coarse_mask.sum()),
        "refined_support_cells": int(refined_mask.sum()),
        "false_fill_cells": int((coarse_mask & ~refined_mask).sum()),
    }
    diagnostics = {
        "density_bandwidth_multiplier": float(density_bandwidth_multiplier),
        "density_threshold": float(density_threshold),
        "bandwidth_uv_median": float(bandwidths.median()),
        "resolution": int(resolution),
        "member_leaf_count": len(component.member_leaf_ids),
        "component_point_count": int(component_points.shape[0]),
        "coarse_gap_closing_cells": int(coarse_gap_closing_cells),
    }

    return ComponentBoundaryResult(
        component_id=component.component_id,
        resolution=resolution,
        frame=frame,
        coarse_mask=coarse_mask,
        coarse_mask_dilated=coarse_mask_for_combine,
        density_grid=grid,
        threshold_level=float(density_threshold),
        threshold_field=threshold_field,
        refined_mask=refined_mask,
        contour_uv=contour_uv,
        contour_world=contour_world,
        outer_loops=outer_loops,
        hole_loops=hole_loops,
        tiny_artifact_loops=tiny_artifact_loops,
        unresolved_open_contour_segment_count=unresolved_open,
        topology=topology,
        diagnostics=diagnostics,
    )


def threshold_sensitivity(
    component_points_uv: Any, grid: Any, coarse_mask: Any, levels: list[float]
) -> dict[str, int]:
    """Refined-support cell count at each candidate threshold ``levels`` entry.

    A lightweight probe for §14's "threshold sensitivity" report item: how
    much the support area moves for a nearby choice of ``density_threshold``,
    without re-running the full boundary extraction per level.
    """

    return {
        str(level): int(((grid >= float(level)) & coarse_mask).sum())
        for level in levels
    }


def component_boundary_payload(result: ComponentBoundaryResult) -> dict[str, Any]:
    """JSON-serializable provenance of one component's boundary extraction."""

    def _loop_payload(loop: LoopDescriptor) -> dict[str, Any]:
        return {
            "kind": loop.kind,
            "label": loop.label,
            "area_cells": loop.area_cells,
            "area_world": loop.area_world,
            "perimeter_cells": loop.perimeter_cells,
            "boundary_world_points": loop.boundary_world_points,
            "nested_in_outer_label": loop.nested_in_outer_label,
        }

    return {
        "component_id": result.component_id,
        "resolution": result.resolution,
        "frame": {
            "origin": result.frame.origin.detach().cpu().tolist(),
            "axis_u": result.frame.axis_u.detach().cpu().tolist(),
            "axis_v": result.frame.axis_v.detach().cpu().tolist(),
            "coord_min": result.frame.coord_min.detach().cpu().tolist(),
            "span": result.frame.span.detach().cpu().tolist(),
        },
        "coarse_mask": result.coarse_mask.detach().cpu().tolist(),
        "coarse_mask_dilated": result.coarse_mask_dilated.detach().cpu().tolist(),
        "density_grid": result.density_grid.detach().cpu().tolist(),
        "threshold_level": result.threshold_level,
        "threshold_field": result.threshold_field.detach().cpu().tolist(),
        "refined_mask": result.refined_mask.detach().cpu().tolist(),
        "contour_uv": [[list(a), list(b)] for a, b in result.contour_uv],
        "contour_world": result.contour_world,
        "outer_loops": [_loop_payload(loop) for loop in result.outer_loops],
        "hole_loops": [_loop_payload(loop) for loop in result.hole_loops],
        "tiny_artifact_loops": [_loop_payload(loop) for loop in result.tiny_artifact_loops],
        "unresolved_open_contour_segment_count": result.unresolved_open_contour_segment_count,
        "topology": result.topology,
        "diagnostics": result.diagnostics,
    }
