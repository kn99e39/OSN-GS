from __future__ import annotations

"""Worklog 165 -- analytic blocker audit of frozen Historical Candidate-B.

This module is diagnostic-only.  It does not alter Candidate-B, W160 global
aggregation, the canonical renderer, or any production surface path.  The
only new operation is an analytic first-intersection calculation for three
fixed synthetic surfaces, compared with the canonical renderer median-depth
map produced by the same 2DGS surfel renderer used by Candidate-B.

The report deliberately keeps two coordinate systems separate:

* ``z_star`` is the exact first geometric intersection depth of the analytic
  blocker, expressed in the renderer's camera-space-z convention.
* ``median_depth`` is the frozen canonical renderer median channel.  A valid
  median is never treated as a geometric first hit by this module.

The synthetic fixture constants are fixed in source before execution.  They
are not swept or selected after observing the result.
"""

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for _path in (str(DEVTOOLS), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    STATE_NAMES,
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    project_queries,
)
from observed_occluded.synthetic_contracts import front_camera  # noqa: E402
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel  # noqa: E402
from osn_gs.render.torch_surfel_query_depth_diagnostics import (  # noqa: E402
    render_with_query_depth_probe,
)
from osn_gs.utils.torch_ops import rgb_to_sh_dc  # noqa: E402


IMAGE = 64
FOV = 0.7
BACKGROUND = (0.02, 0.02, 0.02)
SURFEL_OPACITY = 0.93
PLANE_EXTENT_U = 1.10
PLANE_EXTENT_V = 1.10
OBLIQUE_DEGREES = 28.0
SPHERE_RADIUS = 1.0
SPHERE_LATITUDE_COUNT = 9
SPHERE_LONGITUDE_COUNT = 18
SURFEL_SCALE_FACTOR = 0.60
SILHOUETTE_BAND_PIXELS = 2.0
QUERY_BEFORE_OFFSET = 0.50
QUERY_BEHIND_OFFSET = 0.50
NO_HIT_QUERY_OFFSET = 0.50
REAL_LADDER_OFFSETS = (-0.50, 0.0, 0.50, 1.00)
REVIEW_CAMERAS = ("DSC07960.JPG", "DSC08003.JPG", "DSC08043.JPG")
REVIEW_POLYGONS = {
    "tabletop": {
        "DSC08043.JPG": ((200, 215), (235, 213), (236, 235), (201, 236)),
        "DSC07960.JPG": ((383, 188), (415, 189), (413, 200), (385, 199)),
        "DSC08003.JPG": ((240, 158), (280, 158), (279, 171), (242, 170)),
    },
    "table_side_lower_geometry": {
        "DSC08043.JPG": ((220, 264), (385, 260), (383, 280), (222, 283)),
        "DSC07960.JPG": ((215, 257), (375, 253), (373, 275), (217, 278)),
        "DSC08003.JPG": ((205, 259), (380, 256), (378, 277), (207, 280)),
    },
    "vase_foreground_structure": {
        "DSC08043.JPG": ((385, 185), (458, 193), (445, 226), (376, 218)),
        "DSC07960.JPG": ((375, 184), (447, 193), (440, 226), (369, 217)),
        "DSC08003.JPG": ((225, 184), (282, 190), (278, 224), (221, 218)),
    },
}

OBSERVED_RGB = np.asarray((0.10, 0.85, 0.35), dtype=np.float32)
OCCLUDED_RGB = np.asarray((0.92, 0.18, 0.18), dtype=np.float32)
UNRESOLVED_RGB = np.asarray((0.60, 0.60, 0.62), dtype=np.float32)
NON_HIT_RGB = np.asarray((0.36, 0.52, 0.94), dtype=np.float32)
FRONT_RGB = np.asarray((0.98, 0.74, 0.18), dtype=np.float32)
AT_OR_BEHIND_RGB = np.asarray((0.24, 0.76, 0.92), dtype=np.float32)
EQUALITY_RGB = np.asarray((0.72, 0.44, 0.92), dtype=np.float32)
COUNTEREXAMPLE_B_RGB = np.asarray((1.00, 0.48, 0.05), dtype=np.float32)
COUNTEREXAMPLE_C_RGB = np.asarray((0.96, 0.20, 0.72), dtype=np.float32)


def _progress(message: str) -> None:
    print(f"[worklog 165] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _preload_existing_qdepth_binary() -> str:
    """Use an already-built diagnostic binary without changing renderer code.

    The loader normally JIT-builds this unchanged diagnostic sibling.  The
    Windows checkout may not expose ``cl`` even when the exact binary is
    already present in the temporary extension directory, so W165 may preload
    that binary into the existing loader.  No source, kernel, or classifier is
    changed by this compatibility step.
    """

    import osn_gs.render.torch_surfel_query_depth_diagnostics as qdepth

    if qdepth._EXTENSION is not None:
        return "already_loaded"
    binary = Path(tempfile.gettempdir()) / "osn_gs_diff_surfel_rasterization_qdepth" / "osn_gs_diff_surfel_rasterization_qdepth_c.pyd"
    if not binary.exists():
        return "not_found"
    binary_root = str(binary.parent)
    if binary_root not in sys.path:
        sys.path.insert(0, binary_root)
    importlib.import_module("torch")
    qdepth._EXTENSION = importlib.import_module("osn_gs_diff_surfel_rasterization_qdepth_c")
    return "preloaded_existing_binary"


def _safe_reset_output(path: Path) -> None:
    """Remove only the explicitly named current W165 output directory."""

    if path.exists():
        if path.name != "165_historical_candidate_b_arbitrary_point_occlusion_sufficiency_audit":
            raise ValueError(f"refusing to clear unexpected output path: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _rotation_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return a normalized wxyz quaternion for a proper 3x3 frame."""

    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0)
        w = 0.5 * root
        scale = 0.5 / max(root, 1e-12)
        x = (matrix[2, 1] - matrix[1, 2]) * scale
        y = (matrix[0, 2] - matrix[2, 0]) * scale
        z = (matrix[1, 0] - matrix[0, 1]) * scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            x = 0.5 * root
            scale = 0.5 / root
            y = (matrix[0, 1] + matrix[1, 0]) * scale
            z = (matrix[0, 2] + matrix[2, 0]) * scale
            w = (matrix[2, 1] - matrix[1, 2]) * scale
        elif index == 1:
            root = math.sqrt(max(1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2], 1e-12))
            y = 0.5 * root
            scale = 0.5 / root
            x = (matrix[0, 1] + matrix[1, 0]) * scale
            z = (matrix[1, 2] + matrix[2, 1]) * scale
            w = (matrix[0, 2] - matrix[2, 0]) * scale
        else:
            root = math.sqrt(max(1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2], 1e-12))
            z = 0.5 * root
            scale = 0.5 / root
            x = (matrix[0, 2] + matrix[2, 0]) * scale
            y = (matrix[1, 2] + matrix[2, 1]) * scale
            w = (matrix[1, 0] - matrix[0, 1]) * scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)


@dataclass(frozen=True)
class RayGrid:
    origin: np.ndarray
    direction: np.ndarray
    camera_depth: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray


@dataclass(frozen=True)
class SurfaceHit:
    depth: np.ndarray
    hit: np.ndarray
    local_u: np.ndarray
    local_v: np.ndarray
    silhouette: np.ndarray


@dataclass(frozen=True)
class AnalyticSurface:
    name: str
    kind: str
    center: np.ndarray
    tangent_u: np.ndarray | None = None
    tangent_v: np.ndarray | None = None
    normal: np.ndarray | None = None
    half_u: float | None = None
    half_v: float | None = None
    radius: float | None = None

    def intersect(self, rays: RayGrid) -> SurfaceHit:
        shape = rays.camera_depth.shape
        origin = rays.origin.reshape(1, 1, 3)
        direction = rays.direction
        if self.kind == "plane_rectangle":
            assert self.tangent_u is not None and self.tangent_v is not None and self.normal is not None
            assert self.half_u is not None and self.half_v is not None
            denom = np.sum(direction * self.normal.reshape(1, 1, 3), axis=-1)
            numerator = float(np.dot(self.center - rays.origin, self.normal))
            safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
            depth = numerator / safe
            point = origin + depth[..., None] * direction
            relative = point - self.center.reshape(1, 1, 3)
            local_u = np.sum(relative * self.tangent_u.reshape(1, 1, 3), axis=-1)
            local_v = np.sum(relative * self.tangent_v.reshape(1, 1, 3), axis=-1)
            hit = (
                (np.abs(denom) > 1e-12)
                & (depth > 0.0)
                & (np.abs(local_u) <= self.half_u)
                & (np.abs(local_v) <= self.half_v)
            )
            pixel_world = np.maximum(depth, 1e-6) * (2.0 * math.tan(FOV * 0.5) / IMAGE)
            edge_distance = np.minimum(self.half_u - np.abs(local_u), self.half_v - np.abs(local_v))
            silhouette = hit & (edge_distance <= SILHOUETTE_BAND_PIXELS * pixel_world)
            return SurfaceHit(np.where(hit, depth, np.nan), hit, local_u, local_v, silhouette)

        if self.kind == "sphere":
            assert self.radius is not None
            offset = rays.origin.reshape(1, 1, 3) - self.center.reshape(1, 1, 3)
            a = np.sum(direction * direction, axis=-1)
            b = 2.0 * np.sum(direction * offset, axis=-1)
            c = float(np.sum(offset * offset)) - self.radius * self.radius
            discriminant = b * b - 4.0 * a * c
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            safe_a = np.where(a > 0.0, a, 1.0)
            near = (-b - root) / (2.0 * safe_a)
            far = (-b + root) / (2.0 * safe_a)
            depth = np.where((near > 0.0) & valid, near, np.where((far > 0.0) & valid, far, np.nan))
            point = origin + np.where(np.isfinite(depth), depth, 0.0)[..., None] * direction
            relative = point - self.center.reshape(1, 1, 3)
            local_u = np.zeros(shape, dtype=np.float64)
            local_v = np.zeros(shape, dtype=np.float64)
            closest_t = -b / (2.0 * safe_a)
            closest_point = origin + closest_t[..., None] * direction
            closest_distance = np.linalg.norm(closest_point - self.center.reshape(1, 1, 3), axis=-1)
            pixel_world = np.maximum(depth, 1e-6) * (2.0 * math.tan(FOV * 0.5) / IMAGE)
            silhouette = np.isfinite(depth) & ((self.radius - closest_distance) <= SILHOUETTE_BAND_PIXELS * pixel_world)
            return SurfaceHit(depth, np.isfinite(depth), local_u, local_v, silhouette)

        raise ValueError(f"unknown analytic surface kind: {self.kind}")


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    surface: AnalyticSurface
    camera: Any
    sample_density: str
    model_parameters: dict[str, Any]


def _camera_ray_grid(camera: Any) -> RayGrid:
    rows, cols = np.indices((IMAGE, IMAGE), dtype=np.float64)
    pixel_x = cols
    pixel_y = rows
    ndc_x = (2.0 * pixel_x + 1.0) / IMAGE - 1.0
    ndc_y = (2.0 * pixel_y + 1.0) / IMAGE - 1.0
    camera_direction = np.stack(
        (
            ndc_x * math.tan(FOV * 0.5),
            ndc_y * math.tan(FOV * 0.5),
            np.ones_like(ndc_x),
        ),
        axis=-1,
    )
    view = camera.world_view_transform.detach().cpu().numpy()
    view_linear = view[:3, :3]
    inverse_linear = np.linalg.inv(view_linear)
    direction = camera_direction @ inverse_linear
    origin = camera.camera_center.detach().cpu().numpy().astype(np.float64)
    return RayGrid(origin, direction, np.ones_like(ndc_x), pixel_x, pixel_y)


def _world_point_at_camera_depth(camera: Any, row: int, col: int, depth: float) -> np.ndarray:
    ndc_x = (2.0 * float(col) + 1.0) / IMAGE - 1.0
    ndc_y = (2.0 * float(row) + 1.0) / IMAGE - 1.0
    camera_point = np.asarray(
        (
            ndc_x * math.tan(FOV * 0.5) * depth,
            ndc_y * math.tan(FOV * 0.5) * depth,
            depth,
        ),
        dtype=np.float64,
    )
    view = camera.world_view_transform.detach().cpu().numpy()
    inverse_linear = np.linalg.inv(view[:3, :3])
    translation = view[3, :3]
    return (camera_point - translation) @ inverse_linear


def _plane_surface(name: str, oblique: bool = False) -> AnalyticSurface:
    if not oblique:
        tangent_u = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        tangent_v = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    else:
        angle = math.radians(OBLIQUE_DEGREES)
        tangent_u = np.asarray((math.cos(angle), 0.0, -math.sin(angle)), dtype=np.float64)
        tangent_v = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    normal = np.cross(tangent_u, tangent_v)
    return AnalyticSurface(
        name=name,
        kind="plane_rectangle",
        center=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normal=normal,
        half_u=PLANE_EXTENT_U,
        half_v=PLANE_EXTENT_V,
    )


def _sphere_surface() -> AnalyticSurface:
    return AnalyticSurface(
        name="curved_sphere",
        kind="sphere",
        center=np.asarray((0.0, 0.0, 0.35), dtype=np.float64),
        radius=SPHERE_RADIUS,
    )


def _surface_samples(surface: AnalyticSurface, density: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    scales: list[tuple[float, float]] = []
    if surface.kind == "plane_rectangle":
        assert surface.tangent_u is not None and surface.tangent_v is not None
        count = 9 if density == "coarse" else 17
        u_values = np.linspace(-surface.half_u, surface.half_u, count)
        v_values = np.linspace(-surface.half_v, surface.half_v, count)
        spacing_u = float(u_values[1] - u_values[0])
        spacing_v = float(v_values[1] - v_values[0])
        frame = np.column_stack((surface.tangent_u, surface.tangent_v, surface.normal))
        quaternion = _rotation_to_quaternion(frame)
        for u in u_values:
            for v in v_values:
                positions.append(surface.center + u * surface.tangent_u + v * surface.tangent_v)
                rotations.append(quaternion)
                scales.append((SURFEL_SCALE_FACTOR * spacing_u, SURFEL_SCALE_FACTOR * spacing_v))
        metadata = {
            "sampling": "regular tensor grid including rectangle boundary",
            "grid_count_u": count,
            "grid_count_v": count,
            "spacing_u": spacing_u,
            "spacing_v": spacing_v,
        }
    elif surface.kind == "sphere":
        latitude_count = SPHERE_LATITUDE_COUNT if density == "coarse" else 17
        longitude_count = SPHERE_LONGITUDE_COUNT if density == "coarse" else 36
        for latitude_index in range(latitude_count):
            latitude = math.radians(-72.0 + 144.0 * latitude_index / max(latitude_count - 1, 1))
            cos_lat, sin_lat = math.cos(latitude), math.sin(latitude)
            for longitude_index in range(longitude_count):
                longitude = 2.0 * math.pi * longitude_index / longitude_count
                cos_lon, sin_lon = math.cos(longitude), math.sin(longitude)
                normal = np.asarray((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dtype=np.float64)
                tangent_u = np.asarray((-sin_lon, cos_lon, 0.0), dtype=np.float64)
                tangent_v = np.asarray((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), dtype=np.float64)
                frame = np.column_stack((tangent_u, tangent_v, normal))
                positions.append(surface.center + surface.radius * normal)
                rotations.append(_rotation_to_quaternion(frame))
                scales.append((0.30 if density == "coarse" else 0.15, 0.30 if density == "coarse" else 0.15))
        metadata = {
            "sampling": "fixed latitude-longitude rings without poles",
            "latitude_count": latitude_count,
            "longitude_count": longitude_count,
            "latitude_range_degrees": [-72.0, 72.0],
        }
    else:
        raise ValueError(surface.kind)
    return np.asarray(positions, dtype=np.float32), np.asarray(rotations, dtype=np.float32), np.asarray(scales, dtype=np.float32), metadata


def _build_model(surface: AnalyticSurface, density: str, device: str) -> tuple[Any, dict[str, Any]]:
    positions, rotations, scales, sampling = _surface_samples(surface, density)
    model = TorchGaussianSurfelModel(sh_degree=0, device=device)
    model.initialize(
        positions=torch.as_tensor(positions, dtype=torch.float32, device=device),
        colors=torch.tensor([[0.70, 0.58, 0.38]] * len(positions), dtype=torch.float32, device=device),
        opacities=torch.full((len(positions), 1), SURFEL_OPACITY, dtype=torch.float32, device=device),
        scales=torch.as_tensor(scales, dtype=torch.float32, device=device),
        rotations=torch.as_tensor(rotations, dtype=torch.float32, device=device),
    )
    model.active_sh_degree = 0
    return model, {
        "row_count": int(len(positions)),
        "opacity": SURFEL_OPACITY,
        "background": list(BACKGROUND),
        **sampling,
    }


def _fixture(name: str, density: str, device: str) -> SyntheticFixture:
    surfaces = {
        "fronto_parallel_plane": _plane_surface("fronto_parallel_plane", oblique=False),
        "oblique_plane": _plane_surface("oblique_plane", oblique=True),
        "curved_sphere": _sphere_surface(),
    }
    surface = surfaces[name]
    camera = front_camera(device, name=f"W165_{name}_{density}")
    # The model is built in the caller so the fixture stays a small immutable
    # geometry/camera description for report serialization.
    _, parameters = _build_model(surface, density, device)
    return SyntheticFixture(name, surface, camera, density, parameters)


def _render_fixture(fixture: SyntheticFixture, device: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    qdepth_source = _preload_existing_qdepth_binary()
    model, parameters = _build_model(fixture.surface, fixture.sample_density, device)
    with torch.no_grad():
        package = render_with_query_depth_probe(
            fixture.camera,
            model,
            query_depths=None,
            background=torch.as_tensor(BACKGROUND, dtype=torch.float32, device=device),
        )
    render = package["render"].detach().cpu().numpy()
    if render.ndim == 3 and render.shape[0] == 3:
        render = np.transpose(render, (1, 2, 0))
    render = np.clip(render, 0.0, 1.0)
    median = package["out_others"][5].detach().cpu().numpy().reshape(IMAGE, IMAGE).astype(np.float64)
    rays = _camera_ray_grid(fixture.camera)
    hit = fixture.surface.intersect(rays)
    return model, render, median, hit, {**parameters, "renderer_backend": "canonical 2DGS surfel forward path via qdepth diagnostic sibling", "qdepth_binary_source": qdepth_source, "resolution": [IMAGE, IMAGE], "fov": FOV}


def _state_render(model: Any, camera: Any, median: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    _preload_existing_qdepth_binary()
    positions = model.get_xyz.detach()
    geometry = project_queries(camera, positions)
    result = candidate_b.classify_view(
        geometry,
        torch.as_tensor(median.reshape(-1), dtype=torch.float32, device=device),
    )
    states = result["states"].detach().cpu().numpy().astype(np.int8)
    colors = np.tile(UNRESOLVED_RGB.reshape(1, 3), (len(states), 1))
    colors[states == STATE_OBSERVED] = OBSERVED_RGB
    colors[states == STATE_OCCLUDED] = OCCLUDED_RGB
    old_dc = model._features_dc.detach().clone()
    old_rest = model._features_rest.detach().clone()
    try:
        with torch.no_grad():
            model._features_dc.copy_(rgb_to_sh_dc(torch.as_tensor(colors, dtype=torch.float32, device=device)).reshape(-1, 1, 3))
            model._features_rest.zero_()
            package = render_with_query_depth_probe(camera, model, query_depths=None, background=torch.as_tensor(BACKGROUND, dtype=torch.float32, device=device))
        image = package["render"].detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    finally:
        with torch.no_grad():
            model._features_dc.copy_(old_dc)
            model._features_rest.copy_(old_rest)
    return np.clip(image, 0.0, 1.0), states


def _rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(image, dtype=np.float32)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=-1)
    Image.fromarray(np.clip(value * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path, format="PNG", optimize=True)


def _palette_image(values: np.ndarray, valid: np.ndarray, *, low: float | None = None, high: float | None = None) -> np.ndarray:
    finite = valid & np.isfinite(values)
    if low is None:
        low = float(np.nanmin(values[finite])) if np.any(finite) else 0.0
    if high is None:
        high = float(np.nanmax(values[finite])) if np.any(finite) else low + 1.0
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    normalized = np.clip((np.where(finite, values, low) - low) / (high - low), 0.0, 1.0)
    # A compact blue -> cyan -> yellow -> red map, with gray for no data.
    anchors = np.asarray(((0.10, 0.20, 0.70), (0.05, 0.75, 0.90), (0.98, 0.84, 0.18), (0.85, 0.12, 0.08)), dtype=np.float32)
    scaled = normalized * (len(anchors) - 1)
    left = np.floor(scaled).astype(np.int64).clip(0, len(anchors) - 1)
    right = np.minimum(left + 1, len(anchors) - 1)
    fraction = (scaled - left)[..., None]
    image = anchors[left] * (1.0 - fraction) + anchors[right] * fraction
    image[~finite] = np.asarray((0.18, 0.18, 0.20), dtype=np.float32)
    return image


def _categorical_map(hit: SurfaceHit, median: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    valid_median = median > 0.0
    in_front = hit.hit & valid_median & (median < hit.depth)
    at_or_behind = hit.hit & valid_median & (median >= hit.depth)
    equality = hit.hit & valid_median & (median == hit.depth)
    no_hit_valid = (~hit.hit) & valid_median
    image = np.tile(UNRESOLVED_RGB.reshape(1, 1, 3), (IMAGE, IMAGE, 1))
    image[in_front] = FRONT_RGB
    image[at_or_behind] = AT_OR_BEHIND_RGB
    image[equality] = EQUALITY_RGB
    image[no_hit_valid] = NON_HIT_RGB
    return image, {
        "median_in_front_of_blocker": in_front,
        "median_at_or_behind_blocker": at_or_behind,
        "no_blocker_with_valid_median": no_hit_valid,
        "other_invalid": ~(in_front | at_or_behind | no_hit_valid),
        "equality": equality,
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = np.sort(values)
    return {
        "count": int(values.size),
        "min": float(ordered[0]),
        "median": float(np.percentile(ordered, 50)),
        "mean": float(ordered.mean()),
        "p95": float(np.percentile(ordered, 95)),
        "max": float(ordered[-1]),
    }


def _fixture_accounting(hit: SurfaceHit, median: np.ndarray) -> dict[str, Any]:
    valid = median > 0.0
    strict_front = hit.hit & valid & (median < hit.depth)
    equality = hit.hit & valid & (median == hit.depth)
    at_or_behind = hit.hit & valid & (median >= hit.depth)
    no_hit_valid = (~hit.hit) & valid
    band = hit.silhouette
    margins = hit.depth - median
    strict_front_values = margins[strict_front]
    strict_front_interior = strict_front & ~band
    strict_front_silhouette = strict_front & band
    return {
        "total_image_rays_pixels_considered": int(median.size),
        "rays_with_analytic_blocker_hit": int(hit.hit.sum()),
        "rays_without_blocker_hit": int((~hit.hit).sum()),
        "rays_with_valid_renderer_median": int(valid.sum()),
        "blocker_hit_ordering": {
            "m_lt_z_star_median_in_front_of_blocker": int(strict_front.sum()),
            "m_eq_z_star_exact_float_equality": int(equality.sum()),
            "m_ge_z_star_median_at_or_behind_blocker": int(at_or_behind.sum()),
            "fraction_m_lt_z_star_among_hit_valid": float(strict_front.sum() / max(int((hit.hit & valid).sum()), 1)),
        },
        "no_hit_valid_median_count": int(no_hit_valid.sum()),
        "z_star_minus_m_distribution_for_m_lt_z_star": _distribution(strict_front_values),
        "strict_front_attribution": {
            "interior_blocker_rays": int(strict_front_interior.sum()),
            "silhouette_support_band_rays": int(strict_front_silhouette.sum()),
            "silhouette_band_definition": f"analytic hit within {SILHOUETTE_BAND_PIXELS} projected pixels of surface silhouette; diagnostic attribution only",
        },
        "categorical_counts": {
            "MEDIAN_IN_FRONT_OF_BLOCKER": int(strict_front.sum()),
            "MEDIAN_AT_OR_BEHIND_BLOCKER": int(at_or_behind.sum()),
            "NO_BLOCKER_WITH_VALID_MEDIAN": int(no_hit_valid.sum()),
            "OTHER_INVALID": int((~(strict_front | at_or_behind | no_hit_valid)).sum()),
        },
    }


def _candidate_query_record(
    fixture: SyntheticFixture,
    median: np.ndarray,
    row: int,
    col: int,
    query_depth: float,
    label: str,
    gt_label: str,
    device: str,
    blocker_depth: float | None,
) -> dict[str, Any]:
    world = _world_point_at_camera_depth(fixture.camera, row, col, query_depth)
    query = torch.as_tensor(world.reshape(1, 3), dtype=torch.float32, device=device)
    geometry = project_queries(fixture.camera, query)
    result = candidate_b.classify_view(
        geometry,
        torch.as_tensor(median.reshape(-1), dtype=torch.float32, device=device),
    )
    state = int(result["states"][0].item())
    actual_depth = float(geometry.depth[0].item())
    record = {
        "fixture": fixture.name,
        "surface_sampling": fixture.sample_density,
        "label": label,
        "camera": str(fixture.camera.image_name),
        "pixel_ray": {"row": int(row), "col": int(col)},
        "world_xyz": [float(value) for value in world],
        "query_camera_depth": actual_depth,
        "median_depth": float(median[row, col]) if median[row, col] > 0.0 else None,
        "analytic_blocker_depth": None if blocker_depth is None else float(blocker_depth),
        "candidate_b_state": STATE_NAMES.get(state, f"UNKNOWN_{state}"),
        "gt_state": gt_label,
        "projected_pixel": [float(geometry.pixel_x[0].item()), float(geometry.pixel_y[0].item())],
        "strict_query_gt_margins": {
            "z_star_minus_m": None if blocker_depth is None or median[row, col] <= 0.0 else float(blocker_depth - median[row, col]),
            "z_star_minus_z_query": None if blocker_depth is None else float(blocker_depth - actual_depth),
            "z_query_minus_m": None if median[row, col] <= 0.0 else float(actual_depth - median[row, col]),
        },
    }
    record["candidate_b_occluded_confirmation"] = bool(state == STATE_OCCLUDED)
    record["strict_counterexample_confirmed"] = bool(label.startswith("counterexample") and state == STATE_OCCLUDED)
    return record


def _first_pixel(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.argwhere(mask)
    if not len(indices):
        return None
    row, col = indices[len(indices) // 2]
    return int(row), int(col)


def _best_pixel(mask: np.ndarray, score: np.ndarray) -> tuple[int, int] | None:
    candidates = np.where(mask, score, -np.inf)
    if not np.isfinite(candidates).any():
        return None
    row, col = np.unravel_index(int(np.nanargmax(candidates)), candidates.shape)
    return int(row), int(col)


def _build_queries_and_counters(fixture: SyntheticFixture, median: np.ndarray, hit: SurfaceHit, device: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = median > 0.0
    maps = _categorical_map(hit, median)[1]
    records: list[dict[str, Any]] = []

    hit_pixel = _first_pixel(hit.hit & valid)
    if hit_pixel is None:
        raise RuntimeError(f"fixture has no valid analytic hit rays: {fixture.name}")
    row, col = hit_pixel
    z_star = float(hit.depth[row, col])
    records.append(_candidate_query_record(fixture, median, row, col, z_star - QUERY_BEFORE_OFFSET, "A_query_before_blocker", "GT_DIRECT_ACCESS", device, z_star))
    records.append(_candidate_query_record(fixture, median, row, col, z_star + QUERY_BEHIND_OFFSET, "B_query_behind_blocker", "GT_BLOCKED", device, z_star))
    records.append(_candidate_query_record(fixture, median, row, col, z_star, "C_query_exactly_on_blocker", "GT_BOUNDARY", device, z_star))

    strict_pixel = _best_pixel(maps["median_in_front_of_blocker"], hit.depth - median)
    if strict_pixel is not None:
        row_b, col_b = strict_pixel
        midpoint = 0.5 * (float(median[row_b, col_b]) + float(hit.depth[row_b, col_b]))
        records.append(_candidate_query_record(fixture, median, row_b, col_b, midpoint, "counterexample_B_m_lt_z_query_lt_z_star", "GT_DIRECT_ACCESS", device, float(hit.depth[row_b, col_b])))

    no_hit_pixel = _first_pixel(maps["no_blocker_with_valid_median"])
    if no_hit_pixel is not None:
        row_c, col_c = no_hit_pixel
        query_depth = float(median[row_c, col_c]) + NO_HIT_QUERY_OFFSET
        records.append(_candidate_query_record(fixture, median, row_c, col_c, query_depth, "counterexample_C_valid_median_no_geometric_blocker", "GT_DIRECT_ACCESS_NO_BLOCKER", device, None))

    positive_pixel = _first_pixel(maps["median_at_or_behind_blocker"])
    if positive_pixel is not None:
        row_f, col_f = positive_pixel
        z_query = float(median[row_f, col_f]) + QUERY_BEHIND_OFFSET
        records.append(_candidate_query_record(fixture, median, row_f, col_f, z_query, "F_positive_control_behind_m_and_blocker", "GT_BLOCKED", device, float(hit.depth[row_f, col_f])))

    all_counterexample_attempts = [record for record in records if record["label"].startswith("counterexample")]
    counters = [record for record in all_counterexample_attempts if record["strict_counterexample_confirmed"]]
    return records, {
        "contract_cases": records,
        "executable_counterexample_count": int(len(counters)),
        "strict_B_counterexample_count": int(sum(record["label"].startswith("counterexample_B") for record in counters)),
        "strict_C_counterexample_count": int(sum(record["label"].startswith("counterexample_C") for record in counters)),
        "counterexample_records": counters,
        "counterexample_attempt_count": int(len(all_counterexample_attempts)),
        "nonconfirmed_counterexample_attempts": [record for record in all_counterexample_attempts if not record["strict_counterexample_confirmed"]],
    }


def _cross_section_png(path: Path, fixture: SyntheticFixture, records: list[dict[str, Any]]) -> None:
    width, height = 1000, 320
    image = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((24, 18), f"W165 {fixture.name}: camera-ray depth cross-section", fill=(240, 240, 240), font=font)
    relevant = [record for record in records if record["label"].startswith("counterexample")]
    if not relevant:
        draw.text((24, 150), "No strict executable counterexample in this realization", fill=(210, 210, 210), font=font)
    for index, record in enumerate(relevant):
        y = 92 + index * 92
        m = record["median_depth"]
        q = record["query_camera_depth"]
        z = record["analytic_blocker_depth"]
        finite = [value for value in (m, q, z) if value is not None]
        maximum = max(finite + [1.0]) + 0.6
        x0, x1 = 70, 930
        scale = (x1 - x0) / maximum
        draw.line((x0, y, x1, y), fill=(130, 130, 140), width=2)
        draw.ellipse((x0 - 7, y - 7, x0 + 7, y + 7), fill=(240, 240, 240))
        draw.text((x0, y + 10), "camera", fill=(230, 230, 230), font=font)
        if m is not None:
            xm = x0 + m * scale
            draw.line((xm, y - 24, xm, y + 24), fill=(70, 185, 235), width=4)
            draw.text((xm - 28, y - 46), "median", fill=(70, 185, 235), font=font)
        if z is not None:
            xz = x0 + z * scale
            draw.line((xz, y - 24, xz, y + 24), fill=(130, 220, 100), width=4)
            draw.text((xz - 24, y + 28), "z*", fill=(130, 220, 100), font=font)
        xq = x0 + q * scale
        draw.ellipse((xq - 8, y - 8, xq + 8, y + 8), fill=(255, 75, 170) if record["label"].startswith("counterexample_C") else (255, 145, 35))
        draw.text((xq - 25, y - 66), "query", fill=(255, 170, 80), font=font)
        short_label = "B strict counterexample" if record["label"].startswith("counterexample_B") else "C no-hit valid median"
        draw.text((24, y + 40), short_label, fill=(230, 230, 230), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _counterexample_map(path: Path, categorical: np.ndarray, records: list[dict[str, Any]]) -> None:
    image = categorical.copy()
    for record in records:
        if not record["label"].startswith("counterexample"):
            continue
        row, col = record["pixel_ray"]["row"], record["pixel_ray"]["col"]
        colour = COUNTEREXAMPLE_C_RGB if record["label"].startswith("counterexample_C") else COUNTEREXAMPLE_B_RGB
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = row + dr, col + dc
                if 0 <= rr < IMAGE and 0 <= cc < IMAGE and (abs(dr) == 2 or abs(dc) == 2):
                    image[rr, cc] = colour
    _rgb_image(path, image)


def _fixture_readmes(root: Path, fixture: SyntheticFixture, parameters: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("README.md").write_text(
        f"# W165 {fixture.name} {fixture.sample_density} realization\n\n"
        "이 directory는 source에 고정된 한 가지 surfel sampling realization이다. 하위 directory는 visualization type별로 분리되고 각 type README가 shared renderer 조건, palette, review limitation을 설명한다.\n",
        encoding="utf-8",
    )
    common = (
        f"공통 조건: fixture={fixture.name}, sampling={fixture.sample_density}, resolution={IMAGE}x{IMAGE}, "
        f"FOV={FOV}, background={BACKGROUND}, opacity={SURFEL_OPACITY}, row_count={parameters['row_count']}. "
        "analytic depth는 exact ray/surface first intersection이고 renderer median은 canonical 2DGS surfel renderer의 "
        "frozen median channel이다. 이 batch는 Candidate-B 또는 renderer를 수정하지 않는다."
    )
    directories = {
        "original_scene": "Original Scene은 fixed surfel rows의 원래 색상 렌더다. geometry·scale·rotation·opacity·row count를 바꾸지 않는다.",
        "observed_occluded": "Observed/Occluded는 같은 surfel rows에만 frozen Candidate-B center-state 색을 입힌 diagnostic render다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED이며 physical truth가 아니다.",
        "analytic_geometry": "analytic blocker hit mask와 exact z*를 표시한다. 이는 synthetic ground truth 전용이며 production first-hit path가 아니다.",
        "analytic_blocker_depth": "색상은 analytic first-intersection z*의 camera-space depth이고 회색은 NO_HIT이다.",
        "renderer_median_depth": "색상은 canonical renderer median depth m(p)이고 회색은 invalid/no event이다.",
        "signed_depth_difference": "색상은 z*(p)-m(p)이며 analytic hit ray만 유효하다. 양수는 median이 blocker 앞에 있음을 뜻한다.",
        "categorical_ray_map": "주황=MEDIAN_IN_FRONT_OF_BLOCKER, 청록=MEDIAN_AT_OR_BEHIND_BLOCKER, 파랑=NO_BLOCKER_WITH_VALID_MEDIAN, 보라=equality, 회색=other/invalid.",
        "counterexample_rays": "주황 테두리=B strict counterexample, 분홍 테두리=C valid-median/no-hit counterexample이다. case를 제거하지 않는다.",
        "cross_section": "대표 counterexample의 camera→ray depth 순서를 보여준다. cyan=median, green=z*, orange/pink=query이다.",
    }
    for directory, description in directories.items():
        path = root / directory / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# W165 {directory}\n\n{description}\n\n{common}\n\nreview limitation: synthetic analytic truth는 이 fixture의 known surface에만 적용된다. real-scene physical blocker truth, TSDF, Gaussian Region, topology, Boundary First, NURBS, continuation은 판정하지 않는다.\n", encoding="utf-8")


def _write_synthetic_readmes(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("README.md").write_text(
        "# W165 synthetic analytic blocker exports\n\n"
        "이 directory는 고정된 fronto-parallel plane, oblique plane, sphere fixture를 coarse/dense 2DGS surfel realization으로 canonical renderer에 통과시킨 결과다. "
        "각 fixture 아래에는 mandatory `Original Scene`/`Observed-Occluded` pair와 `analytic_geometry`, `analytic_blocker_depth`, `renderer_median_depth`, `signed_depth_difference`, `categorical_ray_map`, `counterexample_rays`, `cross_section` visualization type이 있고 각 type directory에 UTF-8 README가 있다. "
        "coarse와 dense는 사후 튜닝이 아니라 source에 고정된 두 realization이다.\n",
        encoding="utf-8",
    )
    for name in ("fronto_parallel_plane", "oblique_plane", "curved_sphere"):
        root.joinpath(name, "README.md").parent.mkdir(parents=True, exist_ok=True)
        root.joinpath(name, "README.md").write_text(
            f"# W165 synthetic fixture: {name}\n\n"
            "coarse와 dense realization을 같은 analytic surface, camera convention, resolution, opacity, background에서 비교한다. 하위 directory는 visualization type별로 분리되어 있으며, 실패 case와 no-hit support spill을 숨기지 않는다.\n",
            encoding="utf-8",
        )
    root.joinpath("mandatory_gaussian_visualization_pair", "README.md").parent.mkdir(parents=True, exist_ok=True)
    root.joinpath("mandatory_gaussian_visualization_pair", "README.md").write_text(
        "# W165 mandatory Gaussian visualization pair\n\n"
        "모든 synthetic fixture/density에 대해 동일한 rows와 geometry를 사용한 `Original Scene` 및 `Observed-Occluded`를 보존한다. state 색만 바꾸며 marker geometry나 물리 truth를 추가하지 않는다.\n",
        encoding="utf-8",
    )


def _save_fixture_exports(root: Path, fixture: SyntheticFixture, model: Any, render: np.ndarray, median: np.ndarray, hit: SurfaceHit, records: list[dict[str, Any]], device: str) -> None:
    fixture_root = root / fixture.name / fixture.sample_density
    fixture_root.mkdir(parents=True, exist_ok=True)
    categorical, _ = _categorical_map(hit, median)
    observed_occluded, _ = _state_render(model, fixture.camera, median, device)
    analytic_view = _palette_image(hit.depth, hit.hit, low=float(np.nanmin(hit.depth[hit.hit])), high=float(np.nanmax(hit.depth[hit.hit])))
    analytic_view[~hit.hit] = np.asarray((0.12, 0.12, 0.14), dtype=np.float32)
    _rgb_image(fixture_root / "original_scene" / f"{fixture.name}.png", render)
    _rgb_image(fixture_root / "observed_occluded" / f"{fixture.name}.png", observed_occluded)
    _rgb_image(fixture_root / "analytic_geometry" / f"{fixture.name}.png", analytic_view)
    _rgb_image(fixture_root / "analytic_blocker_depth" / f"{fixture.name}.png", _palette_image(hit.depth, hit.hit))
    _rgb_image(fixture_root / "renderer_median_depth" / f"{fixture.name}.png", _palette_image(median, median > 0.0))
    difference = np.where(hit.hit & (median > 0.0), hit.depth - median, np.nan)
    _rgb_image(fixture_root / "signed_depth_difference" / f"{fixture.name}.png", _palette_image(difference, np.isfinite(difference)))
    _rgb_image(fixture_root / "categorical_ray_map" / f"{fixture.name}.png", categorical)
    _counterexample_map(fixture_root / "counterexample_rays" / f"{fixture.name}.png", categorical, records)
    _cross_section_png(fixture_root / "cross_section" / f"{fixture.name}.png", fixture, records)
    _fixture_readmes(fixture_root, fixture, fixture.model_parameters)
    pair_root = root / "mandatory_gaussian_visualization_pair" / fixture.name / fixture.sample_density
    pair_root.mkdir(parents=True, exist_ok=True)
    _rgb_image(pair_root / "Original Scene" / f"{fixture.name}.png", render)
    _rgb_image(pair_root / "Observed-Occluded" / f"{fixture.name}.png", observed_occluded)
    pair_root.joinpath("README.md").write_text(
        "# Mandatory Gaussian Visualization Pair\n\n"
        "Original Scene과 Observed-Occluded는 동일한 synthetic surfel rows, geometry, camera, resolution, background, renderer를 공유한다. Observed-Occluded에서 바뀐 것은 Candidate-B center-state에 따른 display color뿐이며 green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED다. marker row, light, shading, geometry, opacity, scale, rotation은 추가하지 않았다.\n",
        encoding="utf-8",
    )
    pair_root.joinpath("Original Scene", "README.md").write_text(
        "# Original Scene\n\nCanonical renderer가 fixed synthetic surfel rows의 원래 색을 렌더한 baseline이다. Candidate-B state color는 덧씌우지 않는다.\n",
        encoding="utf-8",
    )
    pair_root.joinpath("Observed-Occluded", "README.md").write_text(
        "# Observed-Occluded\n\nOriginal Scene과 동일한 rows/geometry를 사용하고 Candidate-B state color만 표시한다. 이는 physical geometric truth가 아니다.\n",
        encoding="utf-8",
    )


def _run_synthetic_fixture(name: str, density: str, device: str, out: Path) -> dict[str, Any]:
    fixture = _fixture(name, density, device)
    _progress(f"rendering synthetic fixture {name}/{density}")
    model, render, median, hit, parameters = _render_fixture(fixture, device)
    accounting = _fixture_accounting(hit, median)
    records, executable = _build_queries_and_counters(fixture, median, hit, device)
    _save_fixture_exports(out / "synthetic", fixture, model, render, median, hit, records, device)
    _write_raw_fixture(out / "synthetic" / name / density / "raw_depth.npz", {}, hit, median)
    return {
        "fixture": name,
        "sampling_density": density,
        "analytic_surface": {
            "kind": fixture.surface.kind,
            "parameters": {key: value for key, value in fixture.surface.__dict__.items() if value is not None},
        },
        "renderer_realization": parameters,
        "accounting": accounting,
        "executable_queries": executable,
        "median_depth_raw_npz": str((out / "synthetic" / name / density / "raw_depth.npz").resolve()),
    }


def _write_raw_fixture(path: Path, result: dict[str, Any], hit: SurfaceHit, median: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, analytic_depth=hit.depth.astype(np.float32), analytic_hit=hit.hit, silhouette=hit.silhouette, renderer_median_depth=median.astype(np.float32))


def _real_polygon_center(polygon: Any) -> tuple[int, int]:
    vertices = np.asarray(polygon, dtype=np.float64)
    return int(round(float(vertices[:, 1].mean()))), int(round(float(vertices[:, 0].mean())))


def _real_world_point(camera: Any, row: int, col: int, depth: float) -> np.ndarray:
    # Real cameras use the same rasterizer pixel/depth convention as synthetic
    # cameras; their own FOV and image dimensions are read from the camera.
    width, height = int(camera.image_width), int(camera.image_height)
    ndc_x = (2.0 * float(col) + 1.0) / width - 1.0
    ndc_y = (2.0 * float(row) + 1.0) / height - 1.0
    camera_point = np.asarray((ndc_x * math.tan(float(camera.FoVx) * 0.5) * depth, ndc_y * math.tan(float(camera.FoVy) * 0.5) * depth, depth), dtype=np.float64)
    view = camera.world_view_transform.detach().cpu().numpy()
    return (camera_point - view[3, :3]) @ np.linalg.inv(view[:3, :3])


def _real_scene_replay(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    try:
        from maximal_visible_connectivity_export import load_all_train_cameras
    except Exception as exc:  # pragma: no cover - only used by the full replay
        return {"status": "UNAVAILABLE", "reason": f"camera loader import failed: {exc}"}
    if not args.cache.exists():
        return {"status": "UNAVAILABLE", "reason": f"missing frozen W153 cache: {args.cache}"}
    try:
        cameras, camera_meta = load_all_train_cameras(args.source, args.images, args.sparse_dir, args.resolution, args.llffhold, args.device)
        names = [str(camera.image_name) for camera in cameras]
        depth_path = args.cache / "renderer_median_depth_maps.npz"
        with np.load(depth_path, allow_pickle=False) as data:
            depth_maps = np.asarray(data["depth"], dtype=np.float32)
        missing = [name for name in args.real_camera_names if name not in names]
        if missing:
            return {"status": "UNAVAILABLE", "reason": "requested frozen W160 review camera is absent from loaded camera set", "missing_camera_names": missing, "camera_names": names}
    except Exception as exc:  # pragma: no cover - environment/data dependent
        return {"status": "UNAVAILABLE", "reason": f"frozen real-scene load failed: {exc}"}

    projection_root = out / "real_scene" / "median_ladder_projection"
    world_root = out / "real_scene" / "median_ladder_world"
    projection_root.mkdir(parents=True, exist_ok=True)
    world_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    by_camera: dict[str, Any] = {}
    colors = {
        "OBSERVED": np.asarray((0.10, 0.85, 0.35), dtype=np.float32),
        "OCCLUDED": np.asarray((0.92, 0.18, 0.18), dtype=np.float32),
        "UNRESOLVED": np.asarray((0.60, 0.60, 0.62), dtype=np.float32),
    }
    for camera_name in args.real_camera_names:
        index = names.index(camera_name)
        camera = cameras[index]
        width, height = int(camera.image_width), int(camera.image_height)
        median_map = depth_maps[index].reshape(height, width)
        projection = Image.new("RGB", (width, height), (24, 24, 28))
        draw = ImageDraw.Draw(projection)
        all_world: list[list[float]] = []
        camera_records = []
        for roi_name, polygons in REVIEW_POLYGONS.items():
            row, col = _real_polygon_center(polygons[camera_name])
            if not (0 <= row < median_map.shape[0] and 0 <= col < median_map.shape[1]):
                continue
            median = float(median_map[row, col])
            if median <= 0.0:
                continue
            roi_records = []
            for offset in REAL_LADDER_OFFSETS:
                query_depth = median + float(offset)
                world = _real_world_point(camera, row, col, query_depth)
                query = torch.as_tensor(world.reshape(1, 3), dtype=torch.float32, device=args.device)
                geometry = project_queries(camera, query)
                state = int(candidate_b.classify_view(geometry, torch.as_tensor(median_map.reshape(-1), dtype=torch.float32, device=args.device))["states"][0].item())
                state_name = STATE_NAMES.get(state, f"UNKNOWN_{state}")
                record = {
                    "camera": camera_name,
                    "roi": roi_name,
                    "pixel_ray": {"row": row, "col": col},
                    "offset_from_median": float(offset),
                    "median_depth": median,
                    "query_camera_depth": float(geometry.depth[0].item()),
                    "projected_pixel": [float(geometry.pixel_x[0].item()), float(geometry.pixel_y[0].item())],
                    "world_xyz": [float(value) for value in world],
                    "candidate_b_state": state_name,
                    "physical_truth": "NON-ORACLE REVIEW REFERENCE",
                }
                records.append(record)
                camera_records.append(record)
                roi_records.append(record)
                all_world.append(record["world_xyz"])
                # All ladder points lie on the same camera ray, so their true
                # projection is the same pixel.  Use concentric rings at that
                # pixel to make the coincident projection visible without
                # moving the recorded/query geometry.
                px = max(0, min(width - 1, int(round(float(geometry.pixel_x[0].item())))))
                py = max(0, min(height - 1, int(round(float(geometry.pixel_y[0].item())))))
                rgb = (colors.get(state_name, colors["UNRESOLVED"]) * 255.0).astype(np.uint8).tolist()
                radius = 5 + int(round(abs(offset) * 3.0))
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=tuple(rgb), width=2)
            draw.rectangle((col - 8, row - 8, col + 8, row + 8), outline=(170, 170, 170), width=1)
            draw.text((col + 10, row - 8), roi_name[:8], fill=(220, 220, 220))
        projection.save(projection_root / f"{Path(camera_name).stem}.png", format="PNG", optimize=True)
        by_camera[camera_name] = {"query_count": len(camera_records), "records": camera_records}
        _write_json(world_root / f"{Path(camera_name).stem}.json", {"camera": camera_name, "records": camera_records})

    world_image = Image.new("RGB", (1000, 640), (24, 24, 28))
    draw = ImageDraw.Draw(world_image)
    draw.text((24, 18), "W165 real-scene median ladder world positions (NON-ORACLE REVIEW REFERENCE)", fill=(240, 240, 240))
    xyz = np.asarray([record["world_xyz"] for record in records], dtype=np.float64)
    if len(xyz):
        xmin, xmax = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        ymin, ymax = float(xyz[:, 1].min()), float(xyz[:, 1].max())
        span_x, span_y = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
        for record in records:
            point = np.asarray(record["world_xyz"], dtype=np.float64)
            px = int(60 + 880 * (point[0] - xmin) / span_x)
            py = int(600 - 520 * (point[1] - ymin) / span_y)
            state = record["candidate_b_state"]
            rgb = (colors.get(state, colors["UNRESOLVED"]) * 255.0).astype(np.uint8).tolist()
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=tuple(rgb))
    world_image.save(world_root / "world_positions.png", format="PNG", optimize=True)
    projection_root.joinpath("README.md").write_text(
        "# W165 real-scene median ladder projection\n\n"
        "W160에 이미 존재하는 `DSC07960.JPG`, `DSC08003.JPG`, `DSC08043.JPG`와 frozen `tabletop`, `table_side_lower_geometry`, `vase_foreground_structure` ROI의 polygon centroid에서 query ladder를 만들었다. "
        "각 camera PNG는 before/at/behind median의 projection을 보여주며 green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED이다. "
        "이 단계는 `NON-ORACLE REVIEW REFERENCE`이며 physical blocker truth를 선언하지 않는다. camera, renderer, median cache, ROI는 read-only다.\n",
        encoding="utf-8",
    )
    world_root.joinpath("README.md").write_text(
        "# W165 real-scene median ladder world positions\n\n"
        "세 frozen review camera와 기존 ROI의 median-centered query ladder world XYZ를 XY schematic으로 표시한다. "
        "색은 frozen Candidate-B state이며 green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED이다. `NON-ORACLE REVIEW REFERENCE`: real-scene physical geometry나 first-hit truth는 이 export에서 판정하지 않는다.\n",
        encoding="utf-8",
    )
    return {
        "status": "COMPLETE",
        "camera_names": list(args.real_camera_names),
        "camera_meta": camera_meta,
        "roi_names": list(REVIEW_POLYGONS),
        "ladder_offsets": list(REAL_LADDER_OFFSETS),
        "query_count": len(records),
        "by_camera": by_camera,
        "physical_truth_authority": "NONE_NON_ORACLE_REVIEW_REFERENCE",
        "outputs": {"projection": str(projection_root.resolve()), "world": str(world_root.resolve())},
    }


def _historical_candidate_b_fidelity() -> dict[str, Any]:
    source = (REPO_ROOT / "scripts/devtools/observed_occluded/candidate_b_median_depth.py").resolve()
    text = source.read_text(encoding="utf-8")
    required = (
        "observed = valid & (geometry.depth <= median)",
        "occluded = valid & (geometry.depth > median)",
        "valid = relevant & (median > 0.0)",
    )
    return {
        "candidate_b_source": str(source),
        "candidate_b_source_sha256": _sha256_file(source),
        "required_frozen_lines_present": {line: line in text for line in required},
        "candidate_b_modified_by_w165": False,
        "w160_global_aggregation_modified_by_w165": False,
        "production_renderer_modified_by_w165": False,
        "classification_callsite": "observed_occluded.candidate_b_median_depth.classify_view",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _safe_reset_output(args.out)
    synthetic_root = args.out / "synthetic"
    _write_synthetic_readmes(synthetic_root)
    fixed_results: dict[str, Any] = {}
    dense_results: dict[str, Any] = {}
    fixture_names = ("fronto_parallel_plane", "oblique_plane", "curved_sphere")
    for name in fixture_names:
        fixed_results[name] = _run_synthetic_fixture(name, "coarse", args.device, args.out)
        dense_results[name] = _run_synthetic_fixture(name, "dense", args.device, args.out)

    real_scene = _real_scene_replay(args, args.out)
    (args.out / "real_scene").mkdir(parents=True, exist_ok=True)
    (args.out / "real_scene" / "README.md").write_text(
        "# W165 real-scene sanity replay\n\n"
        "이 단계는 W160의 세 fixed camera와 기존 reviewed ROI를 사용한 `NON-ORACLE REVIEW REFERENCE`다. synthetic analytic blocker truth를 real scene에 전이하지 않으며, Candidate-B/renderer/ROI를 변경하지 않는다. 하위 visualization은 type별로 구성한다.\n",
        encoding="utf-8",
    )
    strict_failures = []
    dense_strict_failures = []
    for name in fixture_names:
        for record in fixed_results[name]["executable_queries"]["counterexample_records"]:
            strict_failures.append(record)
        for record in dense_results[name]["executable_queries"]["counterexample_records"]:
            dense_strict_failures.append(record)

    strict_non_silhouette = []
    dense_non_silhouette = []
    for name in fixture_names:
        accounting = fixed_results[name]["accounting"]
        if accounting["strict_front_attribution"]["interior_blocker_rays"] > 0:
            strict_non_silhouette.append(name)
        dense_accounting = dense_results[name]["accounting"]
        if dense_accounting["strict_front_attribution"]["interior_blocker_rays"] > 0:
            dense_non_silhouette.append(name)

    if dense_non_silhouette:
        architecture_result = "BEHIND_MEDIAN_NOT_SUFFICIENT"
        architecture_reason = "fixed dense realization contains an interior strict counterexample, not only a silhouette/support-spill case."
    elif strict_failures and not strict_non_silhouette:
        architecture_result = "SUFFICIENCY_FAILS_AT_RENDERER_SUPPORT_BOUNDARY"
        architecture_reason = "all strict failures are confined to the declared silhouette/support or no-hit support-spill boundary."
    elif strict_failures:
        architecture_result = "REPRESENTATION_DEPENDENT_MIXED"
        architecture_reason = "strict counterexamples occur in the coarse realization but not in the independently fixed dense replay."
    else:
        architecture_result = "BEHIND_MEDIAN_SUFFICIENCY_SUPPORTED_ON_CONTROLS"
        architecture_reason = "no strict synthetic counterexample was found in the fixed coarse or dense controls; this is not universal physical proof."

    report = {
        "status": "COMPLETE_WL165_HISTORICAL_CANDIDATE_B_ARBITRARY_POINT_OCCLUSION_SUFFICIENCY_AUDIT",
        "batch": "Worklog 165 -- Historical Candidate-B Arbitrary-Point Occlusion Sufficiency Audit Against Analytic Blocker Ground Truth",
        "intent_alignment": {"diagnostic_only": True, "sole_question": "whether query_depth > renderer_median_depth is sufficient for arbitrary-point geometric blocking", "candidate_b_redesigned": False, "production_behavior_modified": False, "human_review_required_for_real_scene": True},
        "implementation_fidelity": {"synthetic_renderer": "canonical 2DGS surfel forward path; qdepth sibling used only to expose the unchanged median channel", "analytic_ground_truth": "exact plane-rectangle and sphere first intersections in camera-space-z convention", "query_construction": "world XYZ reconstructed from the same rasterizer pixel center and requested camera-space z", "epsilon_for_counterexamples": False, "h_or_mu_used": False, "opacity_or_scale_sweep": False, "fixed_resolution": [IMAGE, IMAGE], "fixed_fov": FOV, "fixed_background": list(BACKGROUND)},
        "historical_candidate_b_preservation": _historical_candidate_b_fidelity(),
        "analytic_ground_truth_contract": {"GT_DIRECT_ACCESS": "camera-to-query segment has no blocker hit before query", "GT_BLOCKED": "analytic first blocker hit is strictly before query", "GT_BOUNDARY": "query constructed at analytic surface; diagnostic only", "no_hit_label": "GT_DIRECT_ACCESS_NO_BLOCKER", "production_first_hit_path_added": False},
        "synthetic_fixtures": {"fronto_parallel_plane": fixed_results["fronto_parallel_plane"], "oblique_plane": fixed_results["oblique_plane"], "curved_surface": fixed_results["curved_sphere"]},
        "m_vs_z_star_accounting": {"coarse": {name: fixed_results[name]["accounting"] for name in fixture_names}, "dense": {name: dense_results[name]["accounting"] for name in fixture_names}, "strict_ordering_definition": "m < z_star; equality retained separately as exact float equality; no epsilon discard"},
        "strict_counterexamples": {"coarse_count": len(strict_failures), "dense_count": len(dense_strict_failures), "coarse_records": strict_failures, "dense_records": dense_strict_failures, "executable_candidate_b_confirmation": all(record["candidate_b_state"] == "OCCLUDED" for record in strict_failures + dense_strict_failures)},
        "silhouette_support_spill_attribution": {"band_pixels": SILHOUETTE_BAND_PIXELS, "definition": "diagnostic separation only; failure cases are retained", "non_silhouette_or_no_hit_failure_fixtures": strict_non_silhouette},
        "fixed_denser_replay": {"status": "COMPLETE", "realization_is_independently_fixed": True, "coarse_and_dense_parameters": {name: {"coarse": fixed_results[name]["renderer_realization"], "dense": dense_results[name]["renderer_realization"]} for name in fixture_names}, "counterexamples": {"coarse": len(strict_failures), "dense": len(dense_strict_failures)}},
        "real_scene_sanity_replay": real_scene,
        "qualitative_review_exports": {"synthetic_root": str(synthetic_root.resolve()), "mandatory_pair_root": str((synthetic_root / "mandatory_gaussian_visualization_pair").resolve()), "real_scene_root": str((args.out / "real_scene").resolve()), "png_primary": True, "ppm_count": 0, "readme_per_visualization_type": True, "physical_authority": "synthetic only; real scene is NON-ORACLE REVIEW REFERENCE"},
        "architecture_result": {"allowed_verdicts": ["BEHIND_MEDIAN_SUFFICIENCY_SUPPORTED_ON_CONTROLS", "BEHIND_MEDIAN_NOT_SUFFICIENT", "SUFFICIENCY_FAILS_AT_RENDERER_SUPPORT_BOUNDARY", "REPRESENTATION_DEPENDENT_MIXED", "ANALYTIC_GROUND_TRUTH_CONTRACT_GAP", "UNRESOLVED"], "verdict": architecture_result, "reason": architecture_reason, "universal_physical_proof": False},
        "retained_rejected_open": {"retained": ["Historical Candidate-B", "POINT_QUERY_STATE", "W160 per-view and global aggregation", "W161 OCCLUSION_DOMAIN_CONTRACT_GAP and paused status", "W162-W164", "canonical renderer behavior", "t_w", "Gaussian Region", "TSDF", "topology", "Boundary First", "NURBS", "continuation", "Eligibility"], "rejected": ["median replacement", "first-hit production path", "transmittance occlusion", "contributor-aware arbitrary-query classification", "primitive observation as point-query truth", "threshold/tolerance tuning", "TSDF sign as oracle"], "open": ["human architecture review of attribution and dense replay", "real-scene qualitative interpretation", "universal physical sufficiency beyond these controls"]},
        "inputs": {"repo_root": str(REPO_ROOT), "w153_replay_cache_excluded_from_temp_mirror": True, "real_cache": str(args.cache)},
        "outputs": {"report": str((args.out / "worklog_165_report.json").resolve()), "synthetic": str(synthetic_root.resolve()), "real_scene": str((args.out / "real_scene").resolve())},
        "runtime_seconds": {"total": time.time() - started},
    }
    _write_json(args.out / "worklog_165_report.json", report)
    _write_json(args.out / "synthetic_fixture_results.json", {"coarse": fixed_results, "dense": dense_results})
    (args.out / "README.md").write_text(
        "# W165 Historical Candidate-B Arbitrary-Point Occlusion Sufficiency Audit\n\n"
        "이 output은 frozen Historical Candidate-B와 exact analytic blocker intersection을 비교한 diagnostic-only batch다. `synthetic/`는 fixture와 visualization type별 PNG/README를 포함하고, `real_scene/`는 기존 W160 camera/ROI를 사용한 `NON-ORACLE REVIEW REFERENCE`다. \n\n"
        f"Architecture verdict: `{architecture_result}`. 원본 Candidate-B, W160 aggregation, canonical renderer, production geometry path는 변경하지 않았다.\n",
        encoding="utf-8",
    )
    _progress(f"architecture result: {architecture_result}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/165_historical_candidate_b_arbitrary_point_occlusion_sufficiency_audit")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "DATASET")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache")
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--real-camera-names", nargs=3, default=list(REVIEW_CAMERAS))
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_result": report["architecture_result"]["verdict"], "coarse_counterexamples": report["strict_counterexamples"]["coarse_count"], "dense_counterexamples": report["strict_counterexamples"]["dense_count"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
