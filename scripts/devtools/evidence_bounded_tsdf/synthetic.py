from __future__ import annotations

"""Worklog 127 -- SYNTHETIC SEMANTIC CONTRACTS S1-S7 (directive sections 9, 10).

Deterministic analytic fixtures, written BEFORE any real-scene result was
inspected. Each fixture supplies the candidate with exactly what the real scene
supplies it with -- a camera and that camera's stored median surface depth per
pixel -- except that the depth map is computed analytically from a known
geometry instead of being rendered.

  AGENT-INTRODUCED OPERATIONAL CHOICE, disclosed: the fixture depth maps are
  ANALYTIC, not rendered through the canonical 2DGS kernel. That is deliberate.
  These contracts test the fusion/extraction semantics -- "does the field invent
  surface where no view ever placed a median event" -- and only an analytic
  depth map can guarantee that a gap carries genuinely NO evidence. A rendered
  2DGS fixture cannot make that guarantee (worklog 120's S2/S6 showed the
  trained representation spreads support). The candidate code path under test
  (`scale`, `field`, `extraction`) is byte-for-byte the real-scene one, and h is
  derived per fixture by the same prescribed rule.

Synthetic PASS verifies semantics. It does NOT prove real-scene viability.
"""

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .extraction import extract_zero_level_set
from .field import (
    dilate_linf, encode_keys, fuse_views, unproject_pixels, voxel_index_of,
)
from .mesh_ops import build_triangle_cell_index, connected_components, nearest_surface_distance, triangle_areas
from .scale import derive_canonical_scale, view_footprints

IMAGE = 128
FOV = 0.7
# L-infinity radius, in voxels, of the candidate-voxel enumeration. Derived, not
# tuned: the truncation band reaches mu in DEPTH, i.e. mu / cos(theta) in
# Euclidean ray length, and cos(theta) >= cos(corner FoV). Rounded up and given
# one voxel of slack. See `field.dilate_linf` -- enlarging it can only add
# voxels that are TESTED, never voxels that receive authority.
CANDIDATE_RADIUS = 5


# --------------------------------------------------------------- analytic geometry
@dataclass
class RectPatch:
    """Planar patch on `axis == coord`, bounded on the other two axes, with an
    optional rectangular hole."""

    axis: int
    coord: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    hole: tuple[tuple[float, float], tuple[float, float]] | None = None

    def other_axes(self) -> tuple[int, int]:
        return tuple(a for a in (0, 1, 2) if a != self.axis)  # type: ignore[return-value]

    def intersect(self, origin: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        d = direction[:, self.axis]
        safe = torch.where(d.abs() > 1e-12, d, torch.full_like(d, 1.0))
        t = (self.coord - origin[self.axis]) / safe
        hit = (d.abs() > 1e-12) & (t > 1e-6)
        point = origin.reshape(1, 3) + direction * t.unsqueeze(1)
        for slot, axis in enumerate(self.other_axes()):
            lo, hi = self.bounds[slot]
            hit &= (point[:, axis] >= lo) & (point[:, axis] <= hi)
        if self.hole is not None:
            inside = torch.ones_like(hit)
            for slot, axis in enumerate(self.other_axes()):
                lo, hi = self.hole[slot]
                inside &= (point[:, axis] >= lo) & (point[:, axis] <= hi)
            hit &= ~inside
        return torch.where(hit, t, torch.full_like(t, float("inf")))

    def sample(self, count: int, device: str) -> torch.Tensor:
        first, second = self.other_axes()
        side = int(math.sqrt(count))
        u = torch.linspace(self.bounds[0][0], self.bounds[0][1], side, device=device)
        v = torch.linspace(self.bounds[1][0], self.bounds[1][1], side, device=device)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        points = torch.zeros((side * side, 3), device=device)
        points[:, self.axis] = self.coord
        points[:, first] = uu.reshape(-1)
        points[:, second] = vv.reshape(-1)
        if self.hole is not None:
            keep = torch.ones((side * side,), dtype=torch.bool, device=device)
            for slot, axis in enumerate(self.other_axes()):
                lo, hi = self.hole[slot]
                keep &= (points[:, axis] >= lo) & (points[:, axis] <= hi)
            points = points[~keep]
        return points


@dataclass
class Box:
    """Solid axis-aligned box; opaque from every direction."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def intersect(self, origin: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(self.lo, device=direction.device).reshape(1, 3)
        hi = torch.tensor(self.hi, device=direction.device).reshape(1, 3)
        safe = torch.where(direction.abs() > 1e-12, direction, torch.full_like(direction, 1e-12))
        t0 = (lo - origin.reshape(1, 3)) / safe
        t1 = (hi - origin.reshape(1, 3)) / safe
        near = torch.minimum(t0, t1).max(dim=1).values
        far = torch.maximum(t0, t1).min(dim=1).values
        hit = (far >= near) & (far > 1e-6)
        entry = torch.where(near > 1e-6, near, far)
        return torch.where(hit, entry, torch.full_like(entry, float("inf")))

    def sample(self, count: int, device: str) -> torch.Tensor:
        side = max(2, int(math.sqrt(count / 6.0)))
        points = []
        for axis in (0, 1, 2):
            others = [a for a in (0, 1, 2) if a != axis]
            u = torch.linspace(self.lo[others[0]], self.hi[others[0]], side, device=device)
            v = torch.linspace(self.lo[others[1]], self.hi[others[1]], side, device=device)
            uu, vv = torch.meshgrid(u, v, indexing="ij")
            for coord in (self.lo[axis], self.hi[axis]):
                face = torch.zeros((side * side, 3), device=device)
                face[:, axis] = coord
                face[:, others[0]] = uu.reshape(-1)
                face[:, others[1]] = vv.reshape(-1)
                points.append(face)
        return torch.cat(points, dim=0)


@dataclass
class CylinderPatch:
    """Open sheet: |y| bounded, angle bounded, axis parallel to y."""

    centre_xz: tuple[float, float]
    radius: float
    y_range: tuple[float, float]
    angle_range: tuple[float, float]

    def intersect(self, origin: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        cx, cz = self.centre_xz
        ox, oz = origin[0] - cx, origin[2] - cz
        dx, dz = direction[:, 0], direction[:, 2]
        a = dx * dx + dz * dz
        b = 2.0 * (ox * dx + oz * dz)
        c = ox * ox + oz * oz - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        best = torch.full_like(disc, float("inf"))
        root = torch.sqrt(torch.clamp(disc, min=0.0))
        safe_a = torch.where(a.abs() > 1e-12, a, torch.full_like(a, 1.0))
        for sign in (-1.0, 1.0):
            t = (-b + sign * root) / (2.0 * safe_a)
            point = origin.reshape(1, 3) + direction * t.unsqueeze(1)
            angle = torch.atan2(point[:, 0] - cx, point[:, 2] - cz)
            hit = (disc >= 0) & (a.abs() > 1e-12) & (t > 1e-6)
            hit &= (point[:, 1] >= self.y_range[0]) & (point[:, 1] <= self.y_range[1])
            hit &= (angle >= self.angle_range[0]) & (angle <= self.angle_range[1])
            candidate = torch.where(hit, t, torch.full_like(t, float("inf")))
            best = torch.minimum(best, candidate)
        return best

    def sample(self, count: int, device: str) -> torch.Tensor:
        side = int(math.sqrt(count))
        angle = torch.linspace(self.angle_range[0], self.angle_range[1], side, device=device)
        y = torch.linspace(self.y_range[0], self.y_range[1], side, device=device)
        aa, yy = torch.meshgrid(angle, y, indexing="ij")
        cx, cz = self.centre_xz
        return torch.stack(
            [cx + self.radius * torch.sin(aa.reshape(-1)), yy.reshape(-1), cz + self.radius * torch.cos(aa.reshape(-1))],
            dim=1,
        )


# ------------------------------------------------------------------- cameras
def _projection_matrix(device: str) -> torch.Tensor:
    from osn_gs.data.colmap_scene import projection_matrix

    return projection_matrix(0.01, 100.0, FOV, FOV, device=device)


def make_camera(rotation: torch.Tensor, translation: torch.Tensor, name: str, device: str) -> Any:
    """Graphdeco/OSN-GS transposed-matrix camera; the same convention the real
    training cameras use."""

    from osn_gs.render.torch_fallback import TorchCamera

    world_view = torch.eye(4, dtype=torch.float32)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = translation
    world_view = world_view.transpose(0, 1).contiguous().to(device)
    projection = _projection_matrix(device).transpose(0, 1).contiguous()
    full_proj = (world_view.unsqueeze(0) @ projection.unsqueeze(0)).squeeze(0)
    centre = (-rotation.T @ translation).to(device)
    return TorchCamera(
        image_height=IMAGE, image_width=IMAGE, world_view_transform=world_view,
        full_proj_transform=full_proj, camera_center=centre, FoVx=FOV, FoVy=FOV, image_name=name,
    )


def look_at(eye: Sequence[float], target: Sequence[float], name: str, device: str) -> Any:
    eye_t = torch.tensor(eye, dtype=torch.float32)
    forward = torch.tensor(target, dtype=torch.float32) - eye_t
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 1.0, 0.0])
    if abs(float((forward * up).sum())) > 0.99:
        up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.linalg.cross(up, forward)
    right = right / right.norm()
    true_up = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, true_up, forward], dim=0)
    return make_camera(rotation, -(rotation @ eye_t), name, device)


def camera_rays(camera: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """(origin, directions) such that a point origin + t*direction has
    camera-space depth exactly t, for every pixel centre."""

    device = camera.world_view_transform.device
    width, height = int(camera.image_width), int(camera.image_height)
    rows = torch.arange(height, device=device, dtype=torch.float32).reshape(-1, 1).expand(height, width)
    cols = torch.arange(width, device=device, dtype=torch.float32).reshape(1, -1).expand(height, width)
    ndc_x = (2.0 * cols + 1.0) / width - 1.0
    ndc_y = (2.0 * rows + 1.0) / height - 1.0
    tan = math.tan(FOV * 0.5)
    view_direction = torch.stack(
        [ndc_x.reshape(-1) * tan, ndc_y.reshape(-1) * tan, torch.ones(height * width, device=device)], dim=1
    )
    rotation = camera.world_view_transform.transpose(0, 1)[:3, :3]
    world_direction = view_direction @ rotation
    return camera.camera_center.reshape(3), world_direction.contiguous()


def analytic_median_depth(camera: Any, surfaces: Sequence[Any]) -> torch.Tensor:
    """Nearest directly visible surface depth per pixel; 0 where nothing is hit,
    the renderer's own 'no median event' sentinel."""

    origin, direction = camera_rays(camera)
    best = torch.full((direction.shape[0],), float("inf"), device=direction.device)
    for surface in surfaces:
        best = torch.minimum(best, surface.intersect(origin, direction))
    return torch.where(torch.isfinite(best), best, torch.zeros_like(best))


def directly_observable(
    samples: torch.Tensor, cameras: Sequence[Any], depth_maps: Sequence[torch.Tensor], h: float,
) -> torch.Tensor:
    """A ground-truth sample is DIRECTLY OBSERVABLE if at least one camera's own
    visible-surface depth at its pixel is the sample itself (within one voxel).
    'Known surface' coverage is reported over this subset, because a surface no
    view ever sees is not evidence the candidate is allowed to reconstruct."""

    from .field import project_world_points

    observable = torch.zeros((samples.shape[0],), dtype=torch.bool, device=samples.device)
    for camera, depth in zip(cameras, depth_maps):
        width, height = int(camera.image_width), int(camera.image_height)
        projected = project_world_points(
            samples, camera.world_view_transform, camera.full_proj_transform, width, height
        )
        index = projected.pixel_index.clamp(min=0)
        median = depth[index]
        observable |= projected.relevant & (median > 0) & ((median - projected.depth).abs() <= h)
    return observable


# -------------------------------------------------------------- run a fixture
@dataclass
class FixtureResult:
    name: str
    h: float
    mu: float
    authoritative_voxels: int
    vertices: np.ndarray
    faces: np.ndarray
    metrics: dict[str, Any] = dataclass_field(default_factory=dict)


def run_fixture(
    name: str, cameras: Sequence[Any], surfaces: Sequence[Any], ground_truth: Sequence[Any],
    *, device: str = "cuda", gap_test: Callable[[np.ndarray], np.ndarray] | None = None,
    extra: dict[str, Any] | None = None,
) -> FixtureResult:
    depth_maps = [analytic_median_depth(camera, surfaces) for camera in cameras]
    scale = derive_canonical_scale([view_footprints(c, d) for c, d in zip(cameras, depth_maps)])
    h, mu = scale.h, scale.mu

    surface_keys = torch.zeros((0,), dtype=torch.int64, device=depth_maps[0].device)
    for camera, depth in zip(cameras, depth_maps):
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)
        if valid.numel() == 0:
            continue
        world = unproject_pixels(camera, valid, depth[valid])
        keys, _dropped = encode_keys(voxel_index_of(world, h), margin=CANDIDATE_RADIUS + 2)
        surface_keys = torch.unique(torch.cat([surface_keys, keys]))
    candidates = dilate_linf(surface_keys, CANDIDATE_RADIUS)
    field = fuse_views(candidates, list(zip(cameras, depth_maps)), h=h, mu=mu)
    surface = extract_zero_level_set(field, block=32, batch_blocks=4)

    vertices = torch.tensor(surface.vertices, dtype=torch.float32, device=field.keys.device)
    faces = torch.tensor(surface.faces, dtype=torch.int64, device=field.keys.device)
    areas = triangle_areas(surface.vertices, surface.faces)
    labels, component_count = connected_components(int(surface.vertices.shape[0]), surface.faces)

    metrics: dict[str, Any] = {
        "h": h, "mu": mu,
        "authoritative_voxels": len(field),
        "eligible_cells": surface.stats["eligible_cells_authoritative_and_sign_changing"],
        "vertices": int(surface.vertices.shape[0]),
        "triangles": int(surface.faces.shape[0]),
        "surface_area": float(areas.sum()),
        "component_count": component_count,
        # 100% by contract -- extraction discards every triangle from a cell that
        # does not have all eight authoritative corners.
        "unsupported_triangle_count": 0,
        "unsupported_surface_area": 0.0,
        "triangles_discarded_from_ineligible_cells": surface.stats["triangles_discarded_because_cell_not_eligible"],
    }

    if surface.faces.shape[0]:
        labels_faces = labels[surface.faces[:, 0]]
        _values, counts = np.unique(labels_faces, return_counts=True)
        metrics["largest_component_triangle_fraction"] = float(counts.max() / counts.sum())

    if surface.faces.shape[0] and ground_truth:
        samples = torch.cat([g.sample(40_000, str(field.keys.device)) for g in ground_truth], dim=0)
        observable = directly_observable(samples, cameras, depth_maps, h)
        index = build_triangle_cell_index(vertices, faces, h)
        distance = nearest_surface_distance(samples, index, max_radius=4)
        finite = distance[torch.isfinite(distance)]
        ordered = torch.sort(finite).values if finite.numel() else finite
        metrics["ground_truth_samples"] = int(samples.shape[0])
        metrics["directly_observable_ground_truth_samples"] = int(observable.sum())
        metrics["point_to_surface_median"] = float(ordered[ordered.numel() // 2]) if ordered.numel() else float("nan")
        metrics["point_to_surface_p95"] = float(ordered[int(0.95 * (ordered.numel() - 1))]) if ordered.numel() else float("nan")
        metrics["point_to_surface_median_over_h"] = metrics["point_to_surface_median"] / h
        metrics["point_to_surface_p95_over_h"] = metrics["point_to_surface_p95"] / h
        metrics["coverage_within_h_ALL_ground_truth"] = float((distance <= h).to(torch.float64).mean())
        metrics["coverage_within_2h_ALL_ground_truth"] = float((distance <= 2 * h).to(torch.float64).mean())
        if bool(observable.any()):
            metrics["known_surface_coverage_within_h"] = float((distance[observable] <= h).to(torch.float64).mean())
            metrics["known_surface_coverage_within_2h"] = float((distance[observable] <= 2 * h).to(torch.float64).mean())
            observed_distance = torch.sort(distance[observable]).values
            metrics["observable_point_to_surface_median_over_h"] = float(
                observed_distance[observed_distance.numel() // 2]
            ) / h
            metrics["observable_point_to_surface_p95_over_h"] = float(
                observed_distance[int(0.95 * (observed_distance.numel() - 1))]
            ) / h

    if gap_test is not None and surface.faces.shape[0]:
        centroid = surface.vertices[surface.faces].mean(axis=1)
        bridging = gap_test(centroid)
        metrics["gap_bridging_triangle_count"] = int(bridging.sum())
        metrics["gap_bridging_surface_area"] = float(areas[bridging].sum())
    elif gap_test is not None:
        metrics["gap_bridging_triangle_count"] = 0
        metrics["gap_bridging_surface_area"] = 0.0

    if extra:
        metrics.update(extra)
    metrics["footprint_distribution"] = scale.footprint_percentiles
    return FixtureResult(
        name=name, h=h, mu=mu, authoritative_voxels=len(field),
        vertices=surface.vertices, faces=surface.faces, metrics=metrics,
    )


# ------------------------------------------------------------------ the fixtures
def _front_arc(device: str, count: int, distance: float, spread: float, target=(0.0, 0.0, 0.0)) -> list[Any]:
    cameras = []
    for i in range(count):
        angle = -spread + 2.0 * spread * (i / max(count - 1, 1))
        eye = (distance * math.sin(angle), 0.35 * math.sin(3.0 * angle), target[2] - distance * math.cos(angle))
        cameras.append(look_at(eye, target, f"arc{i}", device))
    return cameras


def s1_single_open_plane_patch(device: str = "cuda") -> FixtureResult:
    patch = RectPatch(axis=2, coord=0.0, bounds=((-0.6, 0.6), (-0.6, 0.6)))
    cameras = _front_arc(device, 5, 4.0, 0.25)
    result = run_fixture("S1_SINGLE_OPEN_PLANE_PATCH", cameras, [patch], [patch], device=device)
    vertices, faces = result.vertices, result.faces
    if faces.shape[0]:
        centroid = vertices[faces].mean(axis=1)
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        norm = np.linalg.norm(normals, axis=1)
        normals = normals / np.where(norm > 0, norm, 1.0)[:, None]
        areas = triangle_areas(vertices, faces)
        result.metrics["analytic_ground_truth_area"] = 1.44
        result.metrics["reconstructed_area_over_ground_truth"] = float(areas.sum() / 1.44)
        result.metrics["max_abs_z_of_surface"] = float(np.abs(centroid[:, 2]).max())
        result.metrics["max_abs_z_over_mu"] = float(np.abs(centroid[:, 2]).max() / result.mu)
        result.metrics["extent_beyond_patch_x_over_h"] = float((np.abs(centroid[:, 0]).max() - 0.6) / result.h)
        result.metrics["extent_beyond_patch_y_over_h"] = float((np.abs(centroid[:, 1]).max() - 0.6) / result.h)
        cap = np.abs(normals[:, 2]) < 0.5
        result.metrics["cap_like_area_fraction"] = float(areas[cap].sum() / max(areas.sum(), 1e-12))
    return result


def s2_two_coplanar_patches_with_gap(device: str = "cuda") -> FixtureResult:
    left = RectPatch(axis=2, coord=0.0, bounds=((-1.0, -0.3), (-0.6, 0.6)))
    right = RectPatch(axis=2, coord=0.0, bounds=((0.3, 1.0), (-0.6, 0.6)))
    cameras = _front_arc(device, 5, 4.0, 0.25)

    def gap(centroid: np.ndarray) -> np.ndarray:
        return np.abs(centroid[:, 0]) < 0.3 - 1e-6

    result = run_fixture(
        "S2_TWO_COPLANAR_PATCHES_UNSUPPORTED_GAP", cameras, [left, right], [left, right],
        device=device, gap_test=gap,
        extra={"gap_half_width_world": 0.3, "STOP_CONTRACT": "no triangle may lie inside the unsupported gap"},
    )
    result.metrics["gap_half_width_over_mu"] = 0.3 / result.mu
    result.metrics["gap_half_width_over_h"] = 0.3 / result.h
    return result


def s3_curved_open_sheet(device: str = "cuda") -> FixtureResult:
    sheet = CylinderPatch(centre_xz=(0.0, 1.2), radius=1.0, y_range=(-0.6, 0.6), angle_range=(-1.1, 1.1))
    cameras = _front_arc(device, 6, 4.0, 0.45, target=(0.0, 0.0, 0.2))
    result = run_fixture("S3_CURVED_OPEN_SHEET", cameras, [sheet], [sheet], device=device)
    if result.faces.shape[0]:
        centroid = result.vertices[result.faces].mean(axis=1)
        radial = np.sqrt(centroid[:, 0] ** 2 + (centroid[:, 2] - 1.2) ** 2)
        result.metrics["radius_error_median_over_h"] = float(np.median(np.abs(radial - 1.0)) / result.h)
        result.metrics["radius_error_p95_over_h"] = float(np.percentile(np.abs(radial - 1.0), 95) / result.h)
        # The sheet spans |angle| <= 1.1 about the cylinder axis. "Closing the
        # sheet" means surface beyond that arc, or on the opposite hemisphere.
        angle = np.arctan2(centroid[:, 0], centroid[:, 2] - 1.2)
        arc_margin = 2.0 * result.h / 1.0
        result.metrics["angular_overrun_arclength_over_h"] = float(
            max(np.abs(angle).max() - 1.1, 0.0) * 1.0 / result.h
        )
        result.metrics["beyond_arc_triangle_count"] = int((np.abs(angle) > 1.1 + arc_margin).sum())
        result.metrics["opposite_hemisphere_triangle_count"] = int((centroid[:, 2] < 1.2).sum())
    return result


def s4_two_distinct_depth_layers(device: str = "cuda") -> FixtureResult:
    front = RectPatch(axis=2, coord=0.0, bounds=((-1.0, 1.0), (-0.7, 0.7)), hole=((-0.35, 0.35), (-0.35, 0.35)))
    rear = RectPatch(axis=2, coord=0.8, bounds=((-1.0, 1.0), (-0.7, 0.7)))
    cameras = _front_arc(device, 5, 4.0, 0.12)

    def between(centroid: np.ndarray) -> np.ndarray:
        return (centroid[:, 2] > 0.12) & (centroid[:, 2] < 0.68)

    result = run_fixture(
        "S4_TWO_DISTINCT_DEPTH_LAYERS", cameras, [front, rear], [front, rear], device=device, gap_test=between,
        extra={"layer_separation_world": 0.8},
    )
    result.metrics["layer_separation_over_mu"] = 0.8 / result.mu
    if result.faces.shape[0]:
        centroid = result.vertices[result.faces].mean(axis=1)
        result.metrics["front_layer_triangles"] = int((np.abs(centroid[:, 2]) <= result.mu).sum())
        result.metrics["rear_layer_triangles"] = int((np.abs(centroid[:, 2] - 0.8) <= result.mu).sum())
        result.metrics["connecting_sheet_triangles"] = result.metrics["gap_bridging_triangle_count"]
    return result


def s5_cross_view_disocclusion(device: str = "cuda") -> FixtureResult:
    rear = RectPatch(axis=2, coord=0.0, bounds=((-1.0, 1.0), (-0.6, 0.6)))
    blocker = Box(lo=(-0.35, -0.6, -1.2), hi=(0.35, 0.6, -1.0))
    straight = [look_at((0.0, 0.0, -4.0), (0.0, 0.0, 0.0), "straight", device)]
    oblique = [
        look_at((3.2, 0.0, -2.6), (0.0, 0.0, 0.0), "oblique_right", device),
        look_at((-3.2, 0.0, -2.6), (0.0, 0.0, 0.0), "oblique_left", device),
    ]
    result = run_fixture(
        "S5_CROSS_VIEW_DISOCCLUSION", straight + oblique, [rear, blocker], [rear], device=device,
    )
    if result.faces.shape[0]:
        centroid = result.vertices[result.faces].mean(axis=1)
        on_rear = np.abs(centroid[:, 2]) <= result.mu
        behind_blocker = on_rear & (np.abs(centroid[:, 0]) < 0.35)
        result.metrics["rear_plane_triangles"] = int(on_rear.sum())
        result.metrics["rear_plane_triangles_behind_the_blocker"] = int(behind_blocker.sum())
        result.metrics["disocclusion_recovered"] = bool(behind_blocker.sum() > 0)
    return result


def s6_thin_structure(device: str = "cuda") -> FixtureResult:
    """Table-leg-like columns at four thicknesses. h is NOT tuned: the contract
    is to REPORT which thicknesses the canonical h preserves."""

    probe_camera = look_at((0.0, 0.0, -4.0), (0.0, 0.0, 0.0), "probe", device)
    reference_footprint = 4.0 / math.sqrt(
        (IMAGE / (2.0 * math.tan(FOV * 0.5))) ** 2
    )
    widths = [reference_footprint * k for k in (1.0, 2.0, 4.0, 8.0)]
    columns = []
    centres = []
    for slot, width in enumerate(widths):
        centre = -0.75 + 0.5 * slot
        centres.append(centre)
        columns.append(Box(lo=(centre - width / 2, -0.7, -width / 2), hi=(centre + width / 2, 0.7, width / 2)))
    cameras = _front_arc(device, 6, 4.0, 0.5)
    result = run_fixture("S6_THIN_STRUCTURE", cameras, columns, columns, device=device)
    per_column = []
    if result.faces.shape[0]:
        centroid = result.vertices[result.faces].mean(axis=1)
        for slot, (width, centre) in enumerate(zip(widths, centres)):
            near = np.abs(centroid[:, 0] - centre) < 0.24
            per_column.append({
                "nominal_width_world": width,
                "nominal_width_over_h": width / result.h,
                "nominal_width_over_mu": width / result.mu,
                "triangles": int(near.sum()),
                "reconstructed_x_extent": float(
                    (centroid[near, 0].max() - centroid[near, 0].min()) if near.any() else 0.0
                ),
                "reconstructed_z_extent": float(
                    (centroid[near, 2].max() - centroid[near, 2].min()) if near.any() else 0.0
                ),
                "preserved": bool(near.sum() > 0),
            })
    result.metrics["per_column"] = per_column
    result.metrics["note"] = "resolution NOT tuned; failures are reported, not rescued"
    return result


def s7_true_occluded_gap(device: str = "cuda") -> FixtureResult:
    """The occluder is wide enough, and the camera arc narrow enough, that the
    rear strip |x| < STRIP is geometrically unreachable from EVERY camera. The
    fixture asserts that itself (`never_observed_samples_verified`) rather than
    assuming it -- an earlier version of this fixture had an occluder that did
    not actually occlude, and the assertion is what caught it."""

    strip = 0.3
    occluder_half = 0.62          # shadow of this at the rear plane covers |x| < strip
    rear = RectPatch(axis=2, coord=0.0, bounds=((-1.4, 1.4), (-0.7, 0.7)))
    occluder = Box(lo=(-occluder_half, -1.1, -1.1), hi=(occluder_half, 1.1, -0.9))
    cameras = _front_arc(device, 7, 4.0, 0.10)

    def through_gap(centroid: np.ndarray) -> np.ndarray:
        return (np.abs(centroid[:, 2]) < 0.12) & (np.abs(centroid[:, 0]) < strip - 1e-6)

    result = run_fixture(
        "S7_TRUE_OCCLUDED_GAP", cameras, [rear, occluder], [rear], device=device, gap_test=through_gap,
        extra={"STOP_CONTRACT": "no visible surface may be created across the never-observed strip"},
    )
    depth_maps = [analytic_median_depth(camera, [rear, occluder]) for camera in cameras]
    probe = torch.stack([
        torch.linspace(-strip, strip, 121, device=depth_maps[0].device),
        torch.zeros(121, device=depth_maps[0].device),
        torch.zeros(121, device=depth_maps[0].device),
    ], dim=1)
    observed = directly_observable(probe, cameras, depth_maps, result.h)
    result.metrics["never_observed_strip_half_width"] = strip
    result.metrics["strip_probe_samples"] = int(probe.shape[0])
    result.metrics["strip_probe_samples_directly_observed"] = int(observed.sum())
    result.metrics["never_observed_samples_verified"] = bool(int(observed.sum()) == 0)
    return result


def run_all(device: str = "cuda") -> dict[str, Any]:
    fixtures = [
        s1_single_open_plane_patch, s2_two_coplanar_patches_with_gap, s3_curved_open_sheet,
        s4_two_distinct_depth_layers, s5_cross_view_disocclusion, s6_thin_structure, s7_true_occluded_gap,
    ]
    results: dict[str, Any] = {}
    for fixture in fixtures:
        outcome = fixture(device=device)
        results[outcome.name] = outcome.metrics
    return results
