"""Phase 2 outer-boundary estimator bias analysis (diagnostics only).

Investigates WHERE, in the Phase 2 boundary-extraction pipeline, the
outward bias documented in worklog 39 (Step 3: "outer boundary conformance
is systematically poor across every scene") first appears, and whether it
traces to KDE bandwidth, density threshold, grid resolution, or contour
resampling.

This module is read-only against production code: it calls
``build_voxel_gaussian_hierarchy`` / ``build_surface_components`` /
``extract_component_boundary`` (``osn_gs/surface/*``) exactly as they exist,
extracts a per-angle radius profile ``r_stage(theta)`` at each of 7 pipeline
stages, and compares every stage against an ANALYTIC ground-truth radius
``r_gt(theta)`` (a circle/ellipse, so the true boundary is known exactly --
unlike the Step 1-3 annulus scenes, which only had an estimated GT). No
estimator behavior in ``osn_gs/surface/*`` is changed by this module.

Deliberately standalone (like ``scripts/stage1_ablation.py``): this is a
research/diagnostic tool, not a new construction phase reachable from
``osn-gs benchmark`` -- see the "wire into osn-gs benchmark" project
convention, which applies to CONSTRUCTION phases, not one-off analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from osn_gs.surface.torch_boundary_eligibility import (
    ACTIVE_OBSERVED,
    COMPLEX,
    DEFAULT_ELIGIBILITY_THRESHOLDS,
    INACTIVE,
    UNCERTAIN,
    LeafBoundaryProvenance,
    LeafEligibilityResult,
    _axis_nn_spacing,
    _convex_hull_2d,
    _polygon_area_2d,
    _sutherland_hodgman_clip,
    compute_leaf_eligibility,
)
from osn_gs.surface.torch_boundary_refinement import kde_density, sample_nn_spacings
from osn_gs.surface.torch_component_boundary import extract_component_boundary
from osn_gs.surface.torch_nurbs import uv_frame_from_axes
from osn_gs.surface.torch_surface_components import SurfaceComponent, build_surface_components
from osn_gs.surface.torch_voxel_hierarchy import (
    TorchVoxelGaussianHierarchy,
    build_voxel_gaussian_hierarchy,
    compute_leaf_face_adjacency,
    plane_aabb_intersection_polygon,
    rasterize_convex_polygon_uv,
)

from .support_domains import ellipse_radius_at_angle

TWO_PI = 2.0 * math.pi


# --- Scene generation: solid disks/ellipses with a KNOWN analytic boundary. ---

@dataclass(frozen=True)
class BoundaryBiasScene:
    name: str
    points: torch.Tensor  # (N, 3), z=0
    center: torch.Tensor  # (2,)
    a: float
    b: float
    description: str


# name -> (center_xy, a, b)
_ELLIPSE_PARAMS: dict[str, tuple[tuple[float, float], float, float]] = {
    "boundary_bias_circle": ((0.0, 0.0), 0.75, 0.75),
    "boundary_bias_ellipse": ((0.0, 0.0), 0.85, 0.5),
    "boundary_bias_ellipse_high_ecc": ((0.0, 0.0), 0.9, 0.22),
    "boundary_bias_ellipse_offcenter": ((0.2, 0.15), 0.65, 0.42),
    "boundary_bias_ellipse_density_gradient": ((0.0, 0.0), 0.85, 0.5),
    "boundary_bias_ellipse_sparse_outer_rim": ((0.0, 0.0), 0.85, 0.5),
    "boundary_bias_ellipse_anisotropic": ((0.0, 0.0), 0.85, 0.5),
}

BOUNDARY_BIAS_SCENE_NAMES = tuple(_ELLIPSE_PARAMS.keys())


def _sample_uniform_in_ellipse(center: torch.Tensor, a: float, b: float, count: int, generator: torch.Generator) -> torch.Tensor:
    accepted: list[torch.Tensor] = []
    remaining = count
    for _ in range(128):
        if remaining <= 0:
            break
        candidates = (torch.rand((max(64, remaining * 3), 2), generator=generator) * 2.0 - 1.0)
        candidates = candidates * torch.tensor([a, b]) * 1.02 + center
        rel = candidates - center
        inside = ((rel[:, 0] / a).square() + (rel[:, 1] / b).square()) <= 1.0
        take = candidates[inside][:remaining]
        if take.shape[0]:
            accepted.append(take)
            remaining -= take.shape[0]
    if remaining > 0:
        raise RuntimeError("boundary_bias_analysis: could not sample enough in-ellipse points.")
    return torch.cat(accepted, dim=0)


def _sample_radially_biased_in_ellipse(
    center: torch.Tensor, a: float, b: float, count: int, generator: torch.Generator, power: float
) -> torch.Tensor:
    """Radial power-law bias toward the center (``power`` > 1 sparsifies the
    outer rim; larger ``power`` = sparser rim), on the exact analytic
    ``r_gt(theta)`` boundary (not a circle), so it stays a true ellipse."""

    theta = torch.rand(count, generator=generator) * TWO_PI
    u = torch.rand(count, generator=generator)
    r_gt = ellipse_radius_at_angle(theta, a, b)
    r = r_gt * u.pow(power)
    return torch.stack([center[0] + r * torch.cos(theta), center[1] + r * torch.sin(theta)], dim=1)


def _sample_anisotropic_in_ellipse(
    center: torch.Tensor, a: float, b: float, count: int, generator: torch.Generator, oversample: int = 4
) -> torch.Tensor:
    """Uniform-in-ellipse candidates, then weighted resampling favoring
    points near the major (x) axis extremes -- density anisotropy that is
    NOT radially symmetric (unlike the density-gradient/sparse-rim scenes)."""

    pool = _sample_uniform_in_ellipse(center, a, b, count * oversample, generator)
    rel_x = (pool[:, 0] - center[0]).abs() / a
    weights = (0.2 + 0.8 * rel_x)
    idx = torch.multinomial(weights, count, replacement=False, generator=generator)
    return pool[idx]


def generate_boundary_bias_scene(name: str, count: int = 600, seed: int = 0) -> BoundaryBiasScene:
    if name not in _ELLIPSE_PARAMS:
        raise ValueError(f"Unknown boundary-bias scene: {name}")
    (cx, cy), a, b = _ELLIPSE_PARAMS[name]
    center = torch.tensor([cx, cy])
    generator = torch.Generator().manual_seed(seed)
    if name == "boundary_bias_ellipse_density_gradient":
        xy = _sample_radially_biased_in_ellipse(center, a, b, count, generator, power=2.5)
    elif name == "boundary_bias_ellipse_sparse_outer_rim":
        xy = _sample_radially_biased_in_ellipse(center, a, b, count, generator, power=3.5)
    elif name == "boundary_bias_ellipse_anisotropic":
        xy = _sample_anisotropic_in_ellipse(center, a, b, count, generator)
    else:
        xy = _sample_uniform_in_ellipse(center, a, b, count, generator)
    points = torch.cat([xy, torch.zeros((xy.shape[0], 1))], dim=1)
    descriptions = {
        "boundary_bias_circle": "Circle, uniform sampling: baseline, no eccentricity/asymmetry.",
        "boundary_bias_ellipse": "Moderate-eccentricity ellipse, uniform sampling.",
        "boundary_bias_ellipse_high_ecc": "High-eccentricity ellipse, uniform sampling.",
        "boundary_bias_ellipse_offcenter": "Moderate ellipse, off-center, uniform sampling.",
        "boundary_bias_ellipse_density_gradient": "Moderate ellipse, radially inner-biased density (power=2.5).",
        "boundary_bias_ellipse_sparse_outer_rim": "Moderate ellipse, strongly inner-biased density (power=6): sparse near the true boundary, stresses KDE boundary bias.",
        "boundary_bias_ellipse_anisotropic": "Moderate ellipse, 2D-anisotropic density (denser near the major-axis extremes, not radially symmetric).",
    }
    return BoundaryBiasScene(name=name, points=points, center=center, a=a, b=b, description=descriptions[name])


# --- Stage extraction: each stage reduces to r_stage(theta) on a shared angle grid. ---

def _wrap_delta(angle: torch.Tensor, theta: float) -> torch.Tensor:
    return torch.remainder(angle - theta + math.pi, TWO_PI) - math.pi


def _fill_nan_nearest(values: torch.Tensor) -> torch.Tensor:
    out = values.tolist()
    n = len(out)
    if all(v != v for v in out):
        # Entirely empty mask (e.g. zero ACTIVE_OBSERVED leaves) -- no
        # neighbor to copy from at all. Explicit zero radius (no support)
        # is the honest value here, not NaN propagating into every metric.
        return torch.zeros((n,), dtype=values.dtype)
    for i in range(n):
        if out[i] == out[i]:
            continue
        for d in range(1, n):
            lo, hi = (i - d) % n, (i + d) % n
            if out[lo] == out[lo]:
                out[i] = out[lo]
                break
            if out[hi] == out[hi]:
                out[i] = out[hi]
                break
    return torch.tensor(out, dtype=values.dtype)


def _per_angle_max_radius(
    cell_angle: torch.Tensor, cell_radius: torch.Tensor, cell_mask: torch.Tensor, theta_samples: torch.Tensor, window: float
) -> torch.Tensor:
    """Max radius of ``cell_mask``-selected cells within ``window`` of each
    angle in ``theta_samples`` -- the shared "per-angle-bin" technique
    already used by ``_outer_radius_weighted_boundary_angles`` (Phase 4
    hardening Step 4) and ``build_annulus_chart``'s own boundary lookup."""

    out = torch.full((theta_samples.shape[0],), float("nan"), dtype=theta_samples.dtype)
    for i, theta in enumerate(theta_samples.tolist()):
        delta = _wrap_delta(cell_angle, theta)
        near = (delta.abs() <= window) & cell_mask
        if bool(near.any()):
            out[i] = cell_radius[near].max()
    return _fill_nan_nearest(out)


@dataclass
class _Stage3Context:
    """Inputs needed to evaluate the KDE field at arbitrary (not just grid) points."""

    own_uv: torch.Tensor
    bandwidths: torch.Tensor
    threshold_level: float


def _stage3_radial_kde_crossing(
    ctx: _Stage3Context, center_world: torch.Tensor, frame: Any, theta_samples: torch.Tensor, r_max: float, r_steps: int = 80,
) -> torch.Tensor:
    """Per-angle radius where the KDE density crosses ``threshold_level``,
    walking outward along each angle ray IN WORLD SPACE -- a RADIAL
    definition, distinct from stage 4's 2D grid-cell mask boundary, isolating
    whether the two views of "the same" threshold disagree. The walk itself
    happens in world space (so a straight ray at world angle ``theta`` stays
    straight); each world point is converted to the frame's UV space only for
    the density evaluation itself, since ``kde_density``'s inputs (``own_uv``,
    bandwidths) are UV-space quantities and the frame's UV normalization is
    generally anisotropic (``coord_min``/``span`` differ per axis)."""

    radii = torch.linspace(0.0, r_max, r_steps)
    out = torch.zeros((theta_samples.shape[0],), dtype=theta_samples.dtype)
    for i, theta in enumerate(theta_samples.tolist()):
        ray_world_xy = center_world.unsqueeze(0) + radii[:, None] * torch.tensor([math.cos(theta), math.sin(theta)])
        ray_world = torch.cat([ray_world_xy, torch.zeros((ray_world_xy.shape[0], 1))], dim=1)
        ray_uv = frame.apply(ray_world, clamp=False)
        density = kde_density(ray_uv, ctx.own_uv, ctx.bandwidths)
        below = density < ctx.threshold_level
        if bool(below.any()):
            first_below = int(below.float().argmax())
            if first_below == 0:
                out[i] = 0.0
            else:
                d0, d1 = float(density[first_below - 1]), float(density[first_below])
                t = (ctx.threshold_level - d0) / (d1 - d0) if abs(d1 - d0) > 1e-12 else 0.5
                out[i] = radii[first_below - 1] + t * (radii[first_below] - radii[first_below - 1])
        else:
            out[i] = r_max
    return out


def _stage_from_world_points(world_points: list[list[float]], center_world: torch.Tensor, theta_samples: torch.Tensor, resample: bool) -> torch.Tensor:
    """Stage 5 (raw, nearest-point) / Stage 6 (resample=True, bin-averaged)
    per-angle radius from a flat list of world-space boundary points
    (marching-squares contour endpoints)."""

    if not world_points:
        return torch.full((theta_samples.shape[0],), float("nan"))
    pts = torch.tensor(world_points)[:, :2]
    rel = pts - center_world
    angle = torch.remainder(torch.atan2(rel[:, 1], rel[:, 0]), TWO_PI)
    radius = rel.norm(dim=1)
    if resample:
        bin_width = TWO_PI / theta_samples.shape[0]
        out = torch.full((theta_samples.shape[0],), float("nan"))
        for i, theta in enumerate(theta_samples.tolist()):
            delta = _wrap_delta(angle, theta)
            in_bin = delta.abs() <= (0.5 * bin_width)
            if bool(in_bin.any()):
                out[i] = radius[in_bin].mean()
        return _fill_nan_nearest(out)
    out = torch.zeros((theta_samples.shape[0],), dtype=theta_samples.dtype)
    for i, theta in enumerate(theta_samples.tolist()):
        delta = _wrap_delta(angle, theta).abs()
        out[i] = radius[int(delta.argmin())]
    return out


def extract_pipeline_stage_radii(
    scene: BoundaryBiasScene,
    theta_bins: int = 144,
    resolution: int = 64,
    density_bandwidth_multiplier: float = 2.0,
    density_threshold: float = 3.0,
    bandwidth_mode: str = "adaptive",
    voxel_min_gaussian_count: int = 10,
    voxel_max_gaussian_count: int = 150,
    voxel_max_depth: int = 6,
) -> dict[str, torch.Tensor]:
    """Run Stage 1 -> Phase 1 -> Phase 2 on ``scene`` and extract a
    per-angle radius profile at each of the 7 pipeline stages, all on the
    SAME ``theta_bins``-angle grid so every stage is directly comparable.

    ``bandwidth_mode``: ``"adaptive"`` (production default, per-sample NN
    spacing) or ``"fixed"`` (a single scalar bandwidth = the median NN
    spacing x multiplier) -- both call the SAME ``kde_density`` unchanged,
    just with a per-sample tensor vs. a scalar, per the ablation plan.
    """

    hierarchy = build_voxel_gaussian_hierarchy(
        scene.points, voxel_min_gaussian_count=voxel_min_gaussian_count,
        voxel_max_gaussian_count=voxel_max_gaussian_count, voxel_max_depth=voxel_max_depth,
    )
    component_set = build_surface_components(hierarchy, scene.points)
    if component_set.component_count() != 1:
        raise RuntimeError(f"{scene.name}: expected 1 component, got {component_set.component_count()}.")
    component = component_set.components[0]

    boundary = extract_component_boundary(
        component, hierarchy, scene.points, resolution=resolution,
        density_bandwidth_multiplier=density_bandwidth_multiplier, density_threshold=density_threshold,
    )
    frame = boundary.frame
    theta_samples = torch.linspace(0.0, TWO_PI * (theta_bins - 1) / theta_bins, theta_bins, dtype=torch.float32)
    center_world = scene.center

    # Angle/radius for points and cells are computed in WORLD space (relative
    # to the scene's own known center), never in the frame's normalized UV
    # space -- UV normalization is generally anisotropic (independent
    # coord_min/span per axis), so a circle/ellipse's shape and radii are NOT
    # preserved when computed in UV coordinates. Only stage 3's density
    # EVALUATION itself needs UV coordinates (kde_density's own inputs).
    point_rel = scene.points[:, :2] - center_world
    point_angle = torch.remainder(torch.atan2(point_rel[:, 1], point_rel[:, 0]), TWO_PI)
    point_radius = point_rel.norm(dim=1)
    window = 0.5 * (TWO_PI / 24)  # fixed diagnostic window, independent of any Phase-4 wedge count

    own_uv = frame.apply(scene.points, clamp=False)
    resolution_r = int(boundary.refined_mask.shape[0])
    centers = (torch.arange(resolution_r, dtype=own_uv.dtype) + 0.5) / resolution_r
    grid_u, grid_v = torch.meshgrid(centers, centers, indexing="ij")
    cell_uv = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)
    cell_world = frame.to_world(cell_uv)
    cell_rel = cell_world[:, :2] - center_world
    cell_angle = torch.remainder(torch.atan2(cell_rel[:, 1], cell_rel[:, 0]), TWO_PI)
    cell_radius = cell_rel.norm(dim=1)

    stage1 = _per_angle_max_radius(point_angle, point_radius, torch.ones_like(point_angle, dtype=torch.bool), theta_samples, window)
    stage2 = _per_angle_max_radius(cell_angle, cell_radius, boundary.coarse_mask.reshape(-1), theta_samples, window)

    bandwidths = float(density_bandwidth_multiplier) * sample_nn_spacings(own_uv)
    if bandwidth_mode == "fixed":
        bandwidths = torch.full_like(bandwidths, float(bandwidths.median()))
    stage3_ctx = _Stage3Context(own_uv=own_uv, bandwidths=bandwidths, threshold_level=float(density_threshold))
    stage3 = _stage3_radial_kde_crossing(stage3_ctx, center_world, frame, theta_samples, r_max=float(point_radius.max()) * 1.1)

    stage4 = _per_angle_max_radius(cell_angle, cell_radius, boundary.threshold_field.reshape(-1), theta_samples, window)

    stage5 = _stage_from_world_points(
        [pt for seg in boundary.contour_world for pt in seg], center_world, theta_samples, resample=False
    )
    stage6 = _stage_from_world_points(
        [pt for seg in boundary.contour_world for pt in seg], center_world, theta_samples, resample=True
    )

    stage7 = _per_angle_max_radius(cell_angle, cell_radius, boundary.refined_mask.reshape(-1), theta_samples, window)

    return {
        "1_raw_gaussian": stage1,
        "2_voxel_union": stage2,
        "3_kde_radial_crossing": stage3,
        "4_threshold_mask": stage4,
        "5_marching_squares_raw": stage5,
        "6_resampled_contour": stage6,
        "7_phase4_representation": stage7,
        "_theta_samples": theta_samples,
    }


# --- Metrics: every stage's r_stage(theta) vs. the analytic r_gt(theta). ---

def _polar_to_xy(theta: torch.Tensor, r: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return torch.stack([center[0] + r * torch.cos(theta), center[1] + r * torch.sin(theta)], dim=1)


def compute_bias_metrics(
    theta_samples: torch.Tensor, r_stage: torch.Tensor, scene: BoundaryBiasScene, coverage_tolerance: float = 0.02, sectors: int = 8,
) -> dict[str, Any]:
    r_gt = ellipse_radius_at_angle(theta_samples, scene.a, scene.b)
    signed = r_stage - r_gt

    xy_stage = _polar_to_xy(theta_samples, r_stage, scene.center)
    xy_gt = _polar_to_xy(theta_samples, r_gt, scene.center)
    d = torch.cdist(xy_stage, xy_gt)
    edge_to_gt = d.min(dim=1).values
    gt_to_edge = d.min(dim=0).values

    dtheta = TWO_PI / theta_samples.shape[0]
    area_stage = float(0.5 * (r_stage.square() * dtheta).sum())
    area_gt = float(0.5 * (r_gt.square() * dtheta).sum())
    false_fill_area = float(0.5 * (torch.clamp(r_stage.square() - r_gt.square(), min=0.0) * dtheta).sum())
    coverage = float((r_stage >= (r_gt - coverage_tolerance)).float().mean())

    sector_bounds = torch.linspace(0.0, TWO_PI, sectors + 1)
    sector_bias = []
    for s in range(sectors):
        in_sector = (theta_samples >= sector_bounds[s]) & (theta_samples < sector_bounds[s + 1])
        sector_bias.append(float(signed[in_sector].mean()) if bool(in_sector.any()) else float("nan"))

    return {
        "signed_distance_mean": float(signed.mean()),
        "signed_distance_max": float(signed.abs().max()),
        "edge_to_gt_mean": float(edge_to_gt.mean()),
        "gt_to_edge_mean": float(gt_to_edge.mean()),
        "symmetric_chamfer": float(0.5 * (edge_to_gt.mean() + gt_to_edge.mean())),
        "hausdorff": float(torch.maximum(edge_to_gt.max(), gt_to_edge.max())),
        "area_stage": area_stage,
        "area_gt": area_gt,
        "area_error": area_stage - area_gt,
        "area_error_relative": (area_stage - area_gt) / area_gt,
        "false_fill_area": false_fill_area,
        "coverage": coverage,
        "sector_bias": sector_bias,
    }


def analyze_scene(scene: BoundaryBiasScene, **extraction_kwargs: Any) -> dict[str, dict[str, Any]]:
    stages = extract_pipeline_stage_radii(scene, **extraction_kwargs)
    theta_samples = stages.pop("_theta_samples")
    return {name: compute_bias_metrics(theta_samples, r_stage, scene) for name, r_stage in stages.items()}


# --- Step B (prototype, NOT wired into production): boundary-leaf-only
# conservative clipping of the plane-AABB polygon by the convex hull of the
# leaf's own member Gaussians. Interior leaves are always left untouched.
# ``_convex_hull_2d``/``_polygon_area_2d``/``_sutherland_hodgman_clip`` now
# live in ``osn_gs.surface.torch_boundary_eligibility`` (promoted to
# production, worklog 48) and are imported above -- not redefined here. ---


@dataclass
class BoundaryLeafRecord:
    leaf_id: str
    is_boundary: bool
    point_count: int
    center_angle: float
    plane_aabb_polygon_uv: torch.Tensor
    clipped_polygon_uv: torch.Tensor
    plane_aabb_area: float
    clipped_area: float
    occupancy_ratio: float  # clipped_area / plane_aabb_area (1.0 = no change)


def build_boundary_leaf_records(
    component: SurfaceComponent,
    hierarchy: TorchVoxelGaussianHierarchy,
    points: torch.Tensor,
    frame: Any,
    center_world: torch.Tensor,
    min_hull_points: int = 4,
    clip_boundary_leaves: bool = True,
) -> list[BoundaryLeafRecord]:
    """Step A + B: for every member leaf, the EXISTING (unmodified)
    ``plane_aabb_intersection_polygon`` vs. a convex-hull-clipped candidate.
    Interior leaves (not in ``component.boundary_leaf_ids``) are recorded
    but never clipped, even when ``clip_boundary_leaves=True`` -- this
    dataclass IS the "existing voxel hierarchy/component logic unchanged"
    invariant, checked directly by
    ``tests/test_boundary_bias_analysis.py::test_interior_leaf_polygon_never_clipped``.
    """

    node_by_id = {node.node_id: node for node in hierarchy.nodes}
    boundary_ids = set(component.boundary_leaf_ids)
    records: list[BoundaryLeafRecord] = []
    for leaf_id in component.member_leaf_ids:
        leaf = node_by_id[leaf_id]
        if leaf.plane is None:
            continue
        polygon_world = plane_aabb_intersection_polygon(leaf.plane.centroid, leaf.plane.normal, leaf.aabb_min, leaf.aabb_max)
        if int(polygon_world.shape[0]) < 3:
            continue
        polygon_uv = frame.apply(polygon_world, clamp=False)
        is_boundary = leaf_id in boundary_ids
        point_count = int(leaf.gaussian_indices.shape[0]) if leaf.gaussian_indices is not None else 0

        clipped_uv = polygon_uv
        if clip_boundary_leaves and is_boundary and point_count >= min_hull_points:
            member_uv = frame.apply(points[leaf.gaussian_indices], clamp=False)
            hull_uv = _convex_hull_2d(member_uv)
            if int(hull_uv.shape[0]) >= 3:
                candidate = _sutherland_hodgman_clip(polygon_uv, hull_uv)
                if int(candidate.shape[0]) >= 3:
                    clipped_uv = candidate

        leaf_center_world = leaf.plane.centroid
        rel = leaf_center_world[:2] - center_world
        center_angle = float(torch.atan2(rel[1], rel[0]))
        area_before = _polygon_area_2d(polygon_uv)
        area_after = _polygon_area_2d(clipped_uv)
        records.append(
            BoundaryLeafRecord(
                leaf_id=leaf_id, is_boundary=is_boundary, point_count=point_count,
                center_angle=center_angle, plane_aabb_polygon_uv=polygon_uv, clipped_polygon_uv=clipped_uv,
                plane_aabb_area=area_before, clipped_area=area_after,
                occupancy_ratio=(area_after / area_before) if area_before > 1e-12 else 1.0,
            )
        )
    return records


def rasterize_leaf_records(records: list[BoundaryLeafRecord], resolution: int, use_clipped: bool) -> torch.Tensor:
    mask = torch.zeros((resolution, resolution), dtype=torch.bool)
    for r in records:
        polygon = r.clipped_polygon_uv if use_clipped else r.plane_aabb_polygon_uv
        mask = mask | rasterize_convex_polygon_uv(polygon, resolution)
    return mask


def analyze_scene_with_clipping(scene: BoundaryBiasScene, theta_bins: int = 144, resolution: int = 64, min_hull_points: int = 4) -> dict[str, Any]:
    """Step A+B+C for one scene: builds the real Phase-1/Phase-2 pipeline
    (unmodified), then compares the ORIGINAL ``coarse_mask``-equivalent
    stage against the boundary-leaf-clipped candidate -- reusing
    ``threshold_field`` (KDE stage, UNCHANGED, no re-tuning) so only the
    coarse/voxel-union stage differs between "before" and "after"."""

    hierarchy = build_voxel_gaussian_hierarchy(scene.points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6)
    component_set = build_surface_components(hierarchy, scene.points)
    if component_set.component_count() != 1:
        raise RuntimeError(f"{scene.name}: expected 1 component, got {component_set.component_count()}.")
    component = component_set.components[0]
    boundary = extract_component_boundary(component, hierarchy, scene.points, resolution=resolution)
    frame = boundary.frame

    records = build_boundary_leaf_records(component, hierarchy, scene.points, frame, scene.center, min_hull_points=min_hull_points)
    mask_before = rasterize_leaf_records(records, resolution, use_clipped=False)
    mask_after = rasterize_leaf_records(records, resolution, use_clipped=True)
    assert bool((mask_before == boundary.coarse_mask).all()), "rasterize_leaf_records(use_clipped=False) must reproduce production coarse_mask exactly"

    refined_before = boundary.threshold_field & mask_before
    refined_after = boundary.threshold_field & mask_after

    theta_samples = torch.linspace(0.0, TWO_PI * (theta_bins - 1) / theta_bins, theta_bins, dtype=torch.float32)
    cell_world = frame.to_world(_grid_cell_centers_uv(resolution, dtype=frame.origin.dtype))
    cell_rel = cell_world[:, :2] - scene.center
    cell_angle = torch.remainder(torch.atan2(cell_rel[:, 1], cell_rel[:, 0]), TWO_PI)
    cell_radius = cell_rel.norm(dim=1)
    window = 0.5 * (TWO_PI / 24)

    r_before = _per_angle_max_radius(cell_angle, cell_radius, refined_before.reshape(-1), theta_samples, window)
    r_after = _per_angle_max_radius(cell_angle, cell_radius, refined_after.reshape(-1), theta_samples, window)
    r_gt = ellipse_radius_at_angle(theta_samples, scene.a, scene.b)
    under_coverage_before = float((r_before < (r_gt - 0.02)).float().mean())
    under_coverage_after = float((r_after < (r_gt - 0.02)).float().mean())

    boundary_records = [r for r in records if r.is_boundary]
    mean_occupancy = sum(r.occupancy_ratio for r in boundary_records) / len(boundary_records) if boundary_records else 1.0

    return {
        "before": {**compute_bias_metrics(theta_samples, r_before, scene), "under_coverage": under_coverage_before},
        "after": {**compute_bias_metrics(theta_samples, r_after, scene), "under_coverage": under_coverage_after},
        "boundary_leaf_count": len(boundary_records),
        "interior_leaf_count": len(records) - len(boundary_records),
        "mean_boundary_leaf_occupancy_ratio": mean_occupancy,
        "records": records,
    }


def _grid_cell_centers_uv(resolution: int, dtype: torch.dtype) -> torch.Tensor:
    centers = (torch.arange(resolution, dtype=dtype) + 0.5) / resolution
    grid_u, grid_v = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)


# --- Support ELIGIBILITY classification: ACTIVE_OBSERVED / UNCERTAIN /
# INACTIVE / COMPLEX. Now lives in ``osn_gs.surface.torch_boundary_
# eligibility`` (promoted to production, worklog 48) and is imported above
# -- not redefined here. Reframes worklog 46's "under-coverage" finding: a
# sparse near-boundary Gaussian cluster isn't necessarily proof of observed
# surface (OSN-GS already treats certain/observed and uncertain/inferred
# evidence as distinct elsewhere -- [[project_osn_gs_direction]]). ---


def build_boundary_leaf_records_with_eligibility(
    component: SurfaceComponent,
    hierarchy: TorchVoxelGaussianHierarchy,
    points: torch.Tensor,
    frame: Any,
    min_hull_points: int = 4,
    thresholds: dict[str, float] = DEFAULT_ELIGIBILITY_THRESHOLDS,
) -> list[tuple[BoundaryLeafRecord, LeafEligibilityResult]]:
    """Step A+B extended: every boundary leaf gets an eligibility
    classification; the convex-hull clip (worklog 46, UNCHANGED) is applied
    only to leaves classified ``ACTIVE_OBSERVED``. Interior leaves keep
    their original, unclipped polygon and are not run through the
    classifier at all (out of scope -- this pass only re-examines BOUNDARY
    evidence, not leaves already fully surrounded by other active leaves)."""

    node_by_id = {node.node_id: node for node in hierarchy.nodes}
    boundary_ids = set(component.boundary_leaf_ids)
    member_ids = set(component.member_leaf_ids)
    adjacency = compute_leaf_face_adjacency(hierarchy, degenerate_axis_tolerant=True)

    out: list[tuple[BoundaryLeafRecord, LeafEligibilityResult]] = []
    for leaf_id in component.member_leaf_ids:
        leaf = node_by_id[leaf_id]
        if leaf.plane is None:
            continue
        polygon_world = plane_aabb_intersection_polygon(leaf.plane.centroid, leaf.plane.normal, leaf.aabb_min, leaf.aabb_max)
        if int(polygon_world.shape[0]) < 3:
            continue
        polygon_uv = frame.apply(polygon_world, clamp=False)
        is_boundary = leaf_id in boundary_ids
        point_count = int(leaf.gaussian_indices.shape[0]) if leaf.gaussian_indices is not None else 0

        if is_boundary:
            eligibility = compute_leaf_eligibility(leaf, points, frame, polygon_uv, adjacency, node_by_id, member_ids, thresholds)
        else:
            eligibility = None  # interior leaves: not classified, original polygon kept as-is

        clipped_uv = polygon_uv
        if is_boundary and eligibility.final_class == ACTIVE_OBSERVED and point_count >= min_hull_points:
            member_uv = frame.apply(points[leaf.gaussian_indices], clamp=False)
            hull_uv = _convex_hull_2d(member_uv)
            if int(hull_uv.shape[0]) >= 3:
                candidate = _sutherland_hodgman_clip(polygon_uv, hull_uv)
                if int(candidate.shape[0]) >= 3:
                    clipped_uv = candidate

        area_before = _polygon_area_2d(polygon_uv)
        area_after = _polygon_area_2d(clipped_uv)
        record = BoundaryLeafRecord(
            leaf_id=leaf_id, is_boundary=is_boundary, point_count=point_count,
            center_angle=0.0, plane_aabb_polygon_uv=polygon_uv, clipped_polygon_uv=clipped_uv,
            plane_aabb_area=area_before, clipped_area=area_after,
            occupancy_ratio=(area_after / area_before) if area_before > 1e-12 else 1.0,
        )
        out.append((record, eligibility))
    return out


def rasterize_eligibility_masks(
    records: list[tuple[BoundaryLeafRecord, LeafEligibilityResult]], resolution: int,
) -> dict[str, torch.Tensor]:
    """Four DISJOINT masks (by construction: each leaf's polygon goes into
    exactly one of these, never more than one) plus the three REQUIRED
    cumulative coverage views. ``inactive`` leaves contribute no polygon at
    all (excluded, not just unclipped)."""

    active = torch.zeros((resolution, resolution), dtype=torch.bool)
    uncertain = torch.zeros((resolution, resolution), dtype=torch.bool)
    complex_mask = torch.zeros((resolution, resolution), dtype=torch.bool)
    interior = torch.zeros((resolution, resolution), dtype=torch.bool)
    for record, eligibility in records:
        if eligibility is None:  # interior leaf, not classified
            interior = interior | rasterize_convex_polygon_uv(record.plane_aabb_polygon_uv, resolution)
            continue
        cls = eligibility.final_class
        if cls == ACTIVE_OBSERVED:
            active = active | rasterize_convex_polygon_uv(record.clipped_polygon_uv, resolution)
        elif cls == UNCERTAIN:
            uncertain = uncertain | rasterize_convex_polygon_uv(record.plane_aabb_polygon_uv, resolution)
        elif cls == COMPLEX:
            complex_mask = complex_mask | rasterize_convex_polygon_uv(record.plane_aabb_polygon_uv, resolution)
        # INACTIVE: deliberately contributes nothing to any mask.
    return {
        "active": active,
        "uncertain": uncertain,
        "complex": complex_mask,
        "interior": interior,
        "active_only": active | interior,
        "active_plus_uncertain": active | interior | uncertain,
        "active_plus_uncertain_plus_complex": active | interior | uncertain | complex_mask,
    }


def analyze_scene_with_eligibility(
    scene: BoundaryBiasScene, theta_bins: int = 144, resolution: int = 64, min_hull_points: int = 4,
    thresholds: dict[str, float] = DEFAULT_ELIGIBILITY_THRESHOLDS,
) -> dict[str, Any]:
    """Step A+B+C, eligibility-classified version. Reports the three
    cumulative views SEPARATELY, interpreted per the plan review: read
    ``active_only`` for PRECISION (false-fill, outward bias, evidence
    purity -- it is not a defect for this to under-cover a physical GT
    boundary the data doesn't actually evidence), and
    ``active_plus_uncertain`` for RECALL (how much of the true boundary is
    at least in the uncertain pool, not truly lost). Matching GT coverage
    on ``active_only`` is explicitly NOT a target -- doing so would just
    re-promote thin evidence back to "observed", the exact failure mode
    this classifier exists to avoid.
    """

    hierarchy = build_voxel_gaussian_hierarchy(scene.points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6)
    component_set = build_surface_components(hierarchy, scene.points)
    if component_set.component_count() != 1:
        raise RuntimeError(f"{scene.name}: expected 1 component, got {component_set.component_count()}.")
    component = component_set.components[0]
    boundary = extract_component_boundary(component, hierarchy, scene.points, resolution=resolution)
    frame = boundary.frame

    records = build_boundary_leaf_records_with_eligibility(component, hierarchy, scene.points, frame, min_hull_points=min_hull_points, thresholds=thresholds)
    masks = rasterize_eligibility_masks(records, resolution)

    theta_samples = torch.linspace(0.0, TWO_PI * (theta_bins - 1) / theta_bins, theta_bins, dtype=torch.float32)
    cell_world = frame.to_world(_grid_cell_centers_uv(resolution, dtype=frame.origin.dtype))
    cell_rel = cell_world[:, :2] - scene.center
    cell_angle = torch.remainder(torch.atan2(cell_rel[:, 1], cell_rel[:, 0]), TWO_PI)
    cell_radius = cell_rel.norm(dim=1)
    window = 0.5 * (TWO_PI / 24)
    r_gt = ellipse_radius_at_angle(theta_samples, scene.a, scene.b)

    views: dict[str, Any] = {}
    for view_name in ("active_only", "active_plus_uncertain", "active_plus_uncertain_plus_complex"):
        refined = boundary.threshold_field & masks[view_name]
        r_view = _per_angle_max_radius(cell_angle, cell_radius, refined.reshape(-1), theta_samples, window)
        under_coverage = float((r_view < (r_gt - 0.02)).float().mean())
        views[view_name] = {**compute_bias_metrics(theta_samples, r_view, scene), "under_coverage": under_coverage}

    classification_counts: dict[str, int] = {ACTIVE_OBSERVED: 0, UNCERTAIN: 0, INACTIVE: 0, COMPLEX: 0}
    provenance_counts = {"is_root_boundary_leaf": 0, "is_inactive_neighbor_leaf": 0, "is_cross_component_boundary_leaf": 0}
    for _record, eligibility in records:
        if eligibility is None:
            continue
        classification_counts[eligibility.final_class] += 1
        for key in provenance_counts:
            if getattr(eligibility.provenance, key):
                provenance_counts[key] += 1

    return {
        "views": views,
        "classification_counts": classification_counts,
        "provenance_counts": provenance_counts,
        "boundary_leaf_count": sum(1 for _r, e in records if e is not None),
        "interior_leaf_count": sum(1 for _r, e in records if e is None),
        "records": records,
    }
