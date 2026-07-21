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

from osn_gs.surface.torch_boundary_refinement import kde_density, sample_nn_spacings
from osn_gs.surface.torch_component_boundary import extract_component_boundary
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

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
