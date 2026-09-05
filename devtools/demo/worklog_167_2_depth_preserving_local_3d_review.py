"""Worklog 167-2: depth-preserving local 3D review of W167 first hits.

This module is an export-only consumer of the frozen W167 real-scene replay.
It does not rerun ray/triangle intersection, alter the query ladder, change
component attribution, or modify the W167 architecture verdict.  The raw W166
zero-set mesh is read only to make a deterministic local inspection volume.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT / "scripts" / "devtools"), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # Direct script execution.
    from worklog_167_1_image_space_review_exports import target_mask
    from worklog_167_raw_zero_set_ray_blocker_audit import (
        REAL_QUERY_OFFSET,
        REVIEW_CAMERAS,
        STATUS_HIT,
        _real_camera_rays,
    )
except ModuleNotFoundError:  # Package import used by focused tests.
    from devtools.demo.worklog_167_1_image_space_review_exports import target_mask
    from devtools.demo.worklog_167_raw_zero_set_ray_blocker_audit import (
        REAL_QUERY_OFFSET,
        REVIEW_CAMERAS,
        STATUS_HIT,
        _real_camera_rays,
    )


DEFAULT_W167_ROOT = REPO_ROOT / "output/167_raw_zero_set_ray_blocker_audit/real_scene"
DEFAULT_RAW_MESH = REPO_ROOT / "output/confirmed/166_historical_sdf_zero_surface_mesh_export/historical_sdf_zero_surface_raw.npz"
DEFAULT_OUT = DEFAULT_W167_ROOT / "review_views_167_2"
DEFAULT_DATASET = REPO_ROOT / "DATASET"
DEFAULT_IMAGES = "images_8"
DEFAULT_SPARSE_DIR = "sparse/0"
DEFAULT_RESOLUTION = -1
DEFAULT_LLFFHOLD = 8

TARGETS = (
    "tabletop",
    "tabletop_vase_contact",
    "table_side_lower_geometry",
    "vase_foreground_structure",
)
TARGET_LABELS = {
    "tabletop": "tabletop",
    "tabletop_vase_contact": "tabletop-vase contact",
    "table_side_lower_geometry": "table-side / lower",
    "vase_foreground_structure": "vase / curved neighbor",
}
VIEW_KINDS = (
    "local_3d_cutaway",
    "depth_ordered_side",
    "local_3d_perspective",
    "component_first_hit_spotlight",
)

# These are fixed before looking at any result.  They control legibility only.
RAY_DISPLAY_MAX = 32
LOCAL_PADDING_FRACTION = 0.25
MIN_LOCAL_PADDING = 0.05
# Fixed before inspecting the replay.  Large camera-to-hit AABBs can contain
# millions of raw triangles; exact IDs stay in sidecars while PNG strokes stay
# reviewable and bounded.
MAX_RENDER_TRIANGLES = 12_000
CANVAS = (1400, 920)

COLOR = {
    "background": (13, 17, 24),
    "panel": (21, 27, 36),
    "mesh": (117, 131, 145),
    "mesh_faint": (73, 86, 100),
    "camera": (248, 246, 202),
    "ray": (44, 194, 244),
    "behind": (232, 62, 62),
    "major": (40, 224, 106),
    "fragment": (255, 137, 32),
    "query": (255, 214, 46),
    "text": (245, 245, 245),
    "muted": (177, 188, 199),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return value / norm


def select_ray_indices(indices: np.ndarray, maximum: int = RAY_DISPLAY_MAX) -> np.ndarray:
    """Select a deterministic row-major subset; never select by component/status."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(values) <= maximum:
        return values.copy()
    stride = int(math.ceil(len(values) / maximum))
    return values[::stride][:maximum]


def derive_inspection_bounds(
    camera_center: np.ndarray,
    first_hit_points: np.ndarray,
    query_points: Iterable[np.ndarray] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a fixed-padding AABB covering the camera-to-hit local review.

    Including the true camera and the before/surface/behind ray points makes
    the disclosed volume contain the displayed ray corridors.  The volume is
    a display crop; it is not a new geometry or intersection decision.
    """

    points = [np.asarray(camera_center, dtype=np.float64).reshape(1, 3)]
    hits = np.asarray(first_hit_points, dtype=np.float64).reshape(-1, 3)
    if len(hits):
        points.append(hits)
    for query in query_points:
        query_array = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        if len(query_array):
            points.append(query_array)
    population = np.concatenate(points, axis=0)
    lower = population.min(axis=0)
    upper = population.max(axis=0)
    span = upper - lower
    padding = max(float(span.max()) * LOCAL_PADDING_FRACTION, MIN_LOCAL_PADDING, 2.0 * REAL_QUERY_OFFSET)
    lower = lower - padding
    upper = upper + padding
    return lower, upper, np.full(3, padding, dtype=np.float64)


def triangles_in_aabb(vertices: np.ndarray, faces: np.ndarray, lower: np.ndarray, upper: np.ndarray, chunk_size: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Return every raw-mesh triangle whose AABB intersects the display volume."""

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    selected_ids: list[np.ndarray] = []
    selected_triangles: list[np.ndarray] = []
    for start in range(0, len(faces), chunk_size):
        stop = min(start + chunk_size, len(faces))
        triangles = vertices[faces[start:stop]]
        tri_min = triangles.min(axis=1)
        tri_max = triangles.max(axis=1)
        mask = np.all(tri_max >= lower, axis=1) & np.all(tri_min <= upper, axis=1)
        if np.any(mask):
            selected_ids.append(np.flatnonzero(mask).astype(np.int64) + start)
            selected_triangles.append(np.asarray(triangles[mask], dtype=np.float64))
    if not selected_ids:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3, 3), dtype=np.float64)
    return np.concatenate(selected_ids), np.concatenate(selected_triangles, axis=0)


def triangles_in_aabb_display(
    vertices: np.ndarray,
    faces: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    maximum: int = MAX_RENDER_TRIANGLES,
    chunk_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Scan every face but retain only a fixed raw-order raster sample.

    The complete crop count is retained in the report.  First-hit triangle
    identities are separately preserved from W167; the full crop face list is
    optional and intentionally not duplicated into large compressed sidecars.
    """

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    sample_ids: list[np.ndarray] = []
    sample_triangles: list[np.ndarray] = []
    total = 0
    chunk_count = int(math.ceil(len(faces) / chunk_size))
    per_chunk = max(1, int(math.ceil(maximum / max(chunk_count, 1))))
    for start in range(0, len(faces), chunk_size):
        stop = min(start + chunk_size, len(faces))
        triangles = vertices[faces[start:stop]]
        tri_min = triangles.min(axis=1)
        tri_max = triangles.max(axis=1)
        mask = np.all(tri_max >= lower, axis=1) & np.all(tri_min <= upper, axis=1)
        local_ids = np.flatnonzero(mask).astype(np.int64)
        total += int(len(local_ids))
        if len(local_ids):
            chosen_count = min(len(local_ids), per_chunk)
            chosen = local_ids[np.linspace(0, len(local_ids) - 1, chosen_count, dtype=np.int64)]
            sample_ids.append(chosen + start)
            sample_triangles.append(np.asarray(triangles[chosen], dtype=np.float64))
    if not sample_ids:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3, 3), dtype=np.float64), total
    ids = np.concatenate(sample_ids)
    triangles = np.concatenate(sample_triangles, axis=0)
    if len(ids) > maximum:
        keep = np.linspace(0, len(ids) - 1, maximum, dtype=np.int64)
        ids = ids[keep]
        triangles = triangles[keep]
    return ids, triangles, total


def _camera_center(camera: Any) -> np.ndarray:
    value = camera.camera_center
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64).reshape(3)


def _load_cameras(args: argparse.Namespace) -> dict[str, Any]:
    from maximal_visible_connectivity_export import load_all_train_cameras

    cameras, _ = load_all_train_cameras(
        args.dataset,
        args.images,
        args.sparse_dir,
        args.resolution,
        args.llffhold,
        "cpu",
    )
    by_name = {str(camera.image_name): camera for camera in cameras}
    missing = [name for name in REVIEW_CAMERAS if name not in by_name]
    if missing:
        raise KeyError(f"camera metadata missing: {missing}")
    return by_name


def _load_raw_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        vertex_key = "vertices" if "vertices" in keys else "verts"
        face_key = "faces" if "faces" in keys else "triangles"
        if vertex_key not in keys or face_key not in keys:
            raise KeyError(f"raw mesh must contain vertices/faces, found {sorted(keys)}")
        vertices = np.asarray(data[vertex_key], dtype=np.float64)
        faces = np.asarray(data[face_key], dtype=np.int64)
    return vertices, faces


def _record(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _ladder(camera: Any, record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pixels = np.asarray(record["pixel"], dtype=np.int64)
    depth = np.asarray(record["depth"], dtype=np.float64)
    rays = _real_camera_rays(camera, pixels)
    origin = np.asarray(rays.origins, dtype=np.float64)
    direction = np.asarray(rays.directions, dtype=np.float64)
    hit = np.asarray(record["world_xyz"], dtype=np.float64)
    q_before = origin + (depth - REAL_QUERY_OFFSET)[:, None] * direction
    q_surface = origin + depth[:, None] * direction
    q_behind = origin + (depth + REAL_QUERY_OFFSET)[:, None] * direction
    return {
        "origin": origin,
        "direction": direction,
        "q_before": q_before,
        "q_surface": q_surface,
        "q_behind": q_behind,
        "stored_hit": hit,
    }


def _basis(camera_center: np.ndarray, target_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _unit(target_center - camera_center)
    up_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _unit(np.cross(forward, up_seed))
    if abs(float(np.dot(right, right))) < 1e-8:
        right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
    up = _unit(np.cross(right, forward))
    return forward, right, up


def _all_scene_points(camera_center: np.ndarray, triangles: np.ndarray, hit_points: np.ndarray, q_behind: np.ndarray) -> np.ndarray:
    pieces = [np.asarray(camera_center, dtype=np.float64).reshape(1, 3), np.asarray(hit_points, dtype=np.float64), np.asarray(q_behind, dtype=np.float64)]
    if len(triangles):
        pieces.append(np.asarray(triangles, dtype=np.float64).reshape(-1, 3))
    return np.concatenate([piece for piece in pieces if len(piece)], axis=0)


def _project(points: np.ndarray, mode: str, camera_center: np.ndarray, target_center: np.ndarray, domain_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    domain_points = np.asarray(domain_points, dtype=np.float64).reshape(-1, 3)
    forward, right, up = _basis(camera_center, target_center)
    if mode == "side":
        values = points - camera_center
        domain_values = domain_points - camera_center
        x = values @ forward
        y = values @ up
        dx = domain_values @ forward
        dy = domain_values @ up
        label = "horizontal = true camera-ray depth (world units)"
    elif mode == "perspective":
        distance = max(float(np.linalg.norm(target_center - camera_center)), 1e-3)
        eye = camera_center - forward * max(0.2, distance * 0.12)
        values = points - eye
        domain_values = domain_points - eye
        depth = values @ forward
        domain_depth = domain_values @ forward
        x = (values @ right) / np.maximum(depth, 0.05) * distance
        y = (values @ up) / np.maximum(depth, 0.05) * distance
        dx = (domain_values @ right) / np.maximum(domain_depth, 0.05) * distance
        dy = (domain_values @ up) / np.maximum(domain_depth, 0.05) * distance
        label = "local perspective; eye offset is display-only"
    else:
        # Oblique orthographic cutaway: depth is retained as an oblique axis.
        view = _unit(forward * 0.82 + right * 0.42 + up * 0.28)
        screen_right = _unit(np.cross(up, view))
        screen_up = _unit(np.cross(view, screen_right))
        values = points - target_center
        domain_values = domain_points - target_center
        x = values @ screen_right
        y = values @ screen_up
        dx = domain_values @ screen_right
        dy = domain_values @ screen_up
        label = "oblique local 3D cutaway; depth axis retained"
    return np.column_stack((x, y)), np.column_stack((dx, dy)), label


def _screen(coords: np.ndarray, domain_coords: np.ndarray, width: int, height: int) -> np.ndarray:
    all_coords = np.concatenate((coords, domain_coords), axis=0)
    lower = all_coords.min(axis=0)
    upper = all_coords.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    margin_x = max(100.0, width * 0.08)
    margin_y = 105.0
    sx = (width - 2 * margin_x) / span[0]
    sy = (height - margin_y - 48.0) / span[1]
    scale = min(sx, sy)
    origin = np.array([margin_x + (width - 2 * margin_x - span[0] * scale) * 0.5, height - 48.0 - (height - margin_y - 48.0 - span[1] * scale) * 0.5], dtype=np.float64)
    out = np.empty_like(coords)
    out[:, 0] = origin[0] + (coords[:, 0] - lower[0]) * scale
    out[:, 1] = origin[1] - (coords[:, 1] - lower[1]) * scale
    return out


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], size: int = 20) -> None:
    draw.text(xy, text, fill=fill, font=_font(size))


def _triangle_indices(count: int) -> np.ndarray:
    if count <= MAX_RENDER_TRIANGLES:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, MAX_RENDER_TRIANGLES, dtype=np.int64)


def _draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: tuple[int, int, int], shape: str, radius: int = 5) -> None:
    x, y = xy
    if shape == "cross":
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=1)


def render_view(
    path: Path,
    kind: str,
    camera_name: str,
    target: str,
    camera_center: np.ndarray,
    triangles: np.ndarray,
    hit_triangles: dict[int, tuple[np.ndarray, bool]],
    rays: dict[str, np.ndarray],
    first_hit_indices: np.ndarray,
    major_mask: np.ndarray,
    local_bounds: tuple[np.ndarray, np.ndarray],
    selected_indices: np.ndarray,
    triangle_total: int,
    spotlight: bool = False,
) -> dict[str, Any]:
    width, height = CANVAS
    image = Image.new("RGB", (width, height), COLOR["background"])
    draw = ImageDraw.Draw(image)
    target_center = np.asarray(rays["stored_hit"][first_hit_indices], dtype=np.float64).mean(axis=0)
    mode = "side" if kind == "depth_ordered_side" else "perspective" if kind == "local_3d_perspective" else "cutaway"

    # The domain includes the true camera, all first-hit points, all selected
    # ray endpoints, and every triangle retained by the disclosed AABB.
    domain = _all_scene_points(camera_center, triangles, rays["stored_hit"][first_hit_indices], rays["q_behind"][first_hit_indices])
    def project(value: np.ndarray) -> np.ndarray:
        coords, domain_coords, _ = _project(value, mode, camera_center, target_center, domain)
        return _screen(coords, domain_coords, width, height)

    projection_domain, _, axis_label = _project(domain, mode, camera_center, target_center, domain)
    camera_xy = _screen(*_project(camera_center.reshape(1, 3), mode, camera_center, target_center, domain)[:2], width, height)[0]

    # Mesh is rendered in stable raw triangle order.  No triangle is filtered
    # by component or suspicion; a fixed cap is only a raster legibility aid.
    mesh_indices = _triangle_indices(len(triangles))
    if len(triangles):
        tri_xy = project(triangles[mesh_indices].reshape(-1, 3)).reshape(-1, 3, 2)
        for tri in tri_xy:
            pts = [(float(x), float(y)) for x, y in tri]
            draw.line(pts + [pts[0]], fill=COLOR["mesh_faint"], width=1)

    # Highlight exact W167 first-hit triangles, preserving their component
    # state and IDs in the sidecar even when a raster overlap is dense.
    for triangle_id, (triangle, is_major) in hit_triangles.items():
        tri_xy = project(np.asarray(triangle).reshape(-1, 3)).reshape(3, 2)
        pts = [(float(x), float(y)) for x, y in tri_xy]
        draw.line(pts + [pts[0]], fill=COLOR["major"] if is_major else COLOR["fragment"], width=3)

    # Draw deterministic ray corridors. All first-hit points remain visible;
    # only line clutter is subsampled by select_ray_indices().
    for index in selected_indices.tolist():
        line_points = project(np.vstack((rays["origin"][index], rays["stored_hit"][index], rays["q_behind"][index])))
        draw.line([(float(x), float(y)) for x, y in line_points[:2]], fill=COLOR["ray"], width=2)
        draw.line([(float(x), float(y)) for x, y in line_points[1:]], fill=COLOR["behind"], width=2)
        _draw_marker(draw, tuple(line_points[2]), COLOR["behind"], "cross", 4)

    hit_xy = project(rays["stored_hit"][first_hit_indices])
    for point, is_major in zip(hit_xy, major_mask.tolist()):
        _draw_marker(draw, (float(point[0]), float(point[1])), COLOR["major"] if is_major else COLOR["fragment"], "dot" if is_major else "cross", 6 if is_major else 7)
    surface_xy = project(rays["q_surface"][first_hit_indices])
    for point in surface_xy[:: max(1, len(surface_xy) // 80)]:
        _draw_marker(draw, (float(point[0]), float(point[1])), COLOR["query"], "dot", 3)
    _draw_marker(draw, tuple(camera_xy), COLOR["camera"], "cross", 9)

    # Header and legend are deliberately text-only so the RGB image-space
    # reprojection remains secondary and cannot be mistaken for 3D evidence.
    _draw_text(draw, (26, 16), f"W167-2 | {kind} | {camera_name} | {TARGET_LABELS[target]}", COLOR["text"], 25)
    _draw_text(draw, (26, 49), f"camera center shown at true world position | {axis_label}", COLOR["muted"], 16)
    legend_y = height - 72
    legend = [
        (COLOR["mesh"], "raw zero-set mesh in disclosed AABB"),
        (COLOR["ray"], "camera → saved first hit"),
        (COLOR["behind"], "saved ray continuation Q_behind"),
        (COLOR["major"], "top-20 component first hit"),
        (COLOR["fragment"], "non-top-20 component first hit"),
        (COLOR["query"], "Q_surface / exact saved hit"),
    ]
    for item, (color, label) in enumerate(legend):
        row, col = divmod(item, 3)
        x = 26 + col * 455
        y = legend_y + row * 27
        draw.rectangle((x, y, x + 16, y + 16), fill=color)
        _draw_text(draw, (x + 22, y - 2), label, COLOR["muted"], 14)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return {
        "path": str(path),
        "kind": kind,
        "mesh_triangles_in_volume": int(triangle_total),
        "mesh_triangles_rendered": int(len(mesh_indices)),
        "mesh_raster_decimated": bool(len(mesh_indices) < triangle_total),
        "first_hit_points_rendered": int(len(first_hit_indices)),
        "ray_lines_rendered": int(len(selected_indices)),
        "inspection_bounds_min": np.asarray(local_bounds[0]).tolist(),
        "inspection_bounds_max": np.asarray(local_bounds[1]).tolist(),
        "spotlight": bool(spotlight),
    }


def _read_report(root: Path) -> dict[str, Any]:
    path = root / "real_scene_replay_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_readme(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _target_readme(kind: str) -> str:
    return (
        f"# W167-2 `{kind}`\n\n"
        "이 디렉터리는 W167의 저장된 real-scene first-hit record와 W166 raw zero-set mesh를 읽기 전용으로 사용한 depth-preserving local 3D 검토 산출물이다.\n\n"
        f"- Visualization: `{kind}`.\n"
        "- Each PNG keeps the true camera center, deterministic ray corridors, every target-associated saved first-hit point, exact hit triangle highlight, and raw mesh triangles intersecting the disclosed deterministic inspection AABB.\n"
        "- Green = W167 exact top-20 component first hit, orange = W167 non-top-20 component first hit, gray = raw zero-set mesh, cyan = camera-to-hit ray, red = Q_behind continuation, yellow = Q_surface.\n"
        "- Non-top-20 is only W167's disconnected-component attribution label; it is not a false-blocker or physical-semantic verdict.\n"
        "- The same-ray image-space reprojection from W167-1 is only a secondary projection-consistency aid. It cannot establish 3D surface identity because a point reconstructed on the same ray reprojects to the source pixel by construction.\n\n"
        "Review limitation: mesh outside the disclosed AABB is not shown in this local frame; the PNG does not claim that omitted remote geometry is absent. Exact pixel/depth/world_xyz/triangle_id/component_id/barycentric rows and volume metadata are in `../sidecars/`; the PNG mesh strokes use a fixed raw-order raster cap.\n"
    )


def _sidecar_readme() -> str:
    return (
        "# W167-2 sidecars\n\n"
        "각 JSON은 한 카메라·타깃의 W167 저장 행을 보존한다. `pixel`, `depth`, `world_xyz`, `triangle_id`, `component_id`, `barycentric`, camera center/direction, Q_before/Q_surface/Q_behind와 inspection AABB를 포함한다.\n\n"
        "첫-hit 행은 선택적으로 줄이지 않았다. PNG ray line만 고정 row-major 규칙으로 최대 32개를 그리며, 모든 first-hit point와 모든 non-top-20 spotlight 행은 유지한다. JSON에는 PNG mesh sample의 raw triangle IDs와 exact crop count를 함께 기록한다.\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real_report = _read_report(root)
    cameras = _load_cameras(args)
    vertices, faces = _load_raw_mesh(Path(args.raw_mesh))
    crop_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, int]] = {}
    sidecar_dir = out / "sidecars"
    _write_readme(out / "README.md", "# W167-2 depth-preserving local 3D review\n\n" + "W167-2 is a subordinate, display-only review layer over the frozen W167 real-scene first-hit replay. W167 and W167-1 outputs are not overwritten.\n\n" + "Primary evidence is local world-space 3D: the true camera center, selected rays, exact saved first-hit points, deterministic raw-mesh inspection volume, depth-ordered side view, perspective view, and non-top-20 component spotlight. Same-ray image-space reprojection is secondary projection consistency only; it cannot prove surface identity.\n\n" + "Geometry outside each disclosed inspection AABB is not shown in that local frame. A fixed raster cap may decimate mesh strokes for legibility, while exact first-hit rows and mesh triangle IDs remain in sidecars. This batch does not upgrade W167's `REAL_SCENE_REVIEW_REQUIRED` verdict.\n")
    _write_readme(sidecar_dir / "README.md", _sidecar_readme())

    major_ids: set[int] = set()
    for camera_data in real_report.get("per_camera", {}).values():
        for row in camera_data.get("component_provenance", []):
            if bool(row.get("major_component")):
                major_ids.add(int(row["component_id"]))

    summaries: list[dict[str, Any]] = []
    unique_fragment_keys: set[tuple[str, int]] = set()
    covered_fragment_keys: set[tuple[str, int]] = set()

    for camera_name in REVIEW_CAMERAS:
        stem = Path(camera_name).stem
        record = _record(root / f"{stem}_ray_results.npz")
        ladder = _ladder(cameras[camera_name], record)
        status = np.asarray(record["status"]).astype(str)
        hit_mask = status == STATUS_HIT
        all_hit_indices = np.flatnonzero(hit_mask).astype(np.int64)
        fragment_mask_all = ~np.isin(np.asarray(record["component_id"], dtype=np.int64), np.asarray(sorted(major_ids), dtype=np.int64))
        for index in all_hit_indices[fragment_mask_all[all_hit_indices]].tolist():
            unique_fragment_keys.add((camera_name, int(index)))
        camera_center = _camera_center(cameras[camera_name])

        for target in TARGETS:
            target_indices = np.flatnonzero(hit_mask & target_mask(camera_name, record["pixel"], target)).astype(np.int64)
            if not len(target_indices):
                continue
            target_major = np.isin(np.asarray(record["component_id"][target_indices], dtype=np.int64), np.asarray(sorted(major_ids), dtype=np.int64))
            target_fragments = target_indices[~target_major]
            for index in target_fragments.tolist():
                covered_fragment_keys.add((camera_name, int(index)))

            lower, upper, padding = derive_inspection_bounds(
                camera_center,
                record["world_xyz"][target_indices],
                (ladder["q_before"][target_indices], ladder["q_surface"][target_indices], ladder["q_behind"][target_indices]),
            )
            crop_key = tuple(np.round(np.concatenate((lower, upper)), 12).tolist())
            if crop_key not in crop_cache:
                crop_cache[crop_key] = triangles_in_aabb_display(vertices, faces, lower, upper)
            mesh_ids, mesh_triangles, mesh_total = crop_cache[crop_key]
            hit_triangles: dict[int, tuple[np.ndarray, bool]] = {}
            for index in target_indices.tolist():
                triangle_id = int(record["triangle_id"][index])
                hit_triangles[triangle_id] = (vertices[faces[triangle_id]], bool(int(record["component_id"][index]) in major_ids))
            selected = select_ray_indices(target_indices)
            target_dir_names = {
                "local_3d_cutaway": f"local_3d_cutaway_{target}",
                "depth_ordered_side": f"depth_ordered_side_{target}",
                "local_3d_perspective": f"local_3d_perspective_{target}",
                "component_first_hit_spotlight": f"component_first_hit_spotlight_{target}",
            }
            target_summary = {
                "camera": camera_name,
                "target": target,
                "target_label": TARGET_LABELS[target],
                "first_hit_count": int(len(target_indices)),
                "major_first_hit_count": int(target_major.sum()),
                "non_top20_first_hit_count": int((~target_major).sum()),
                "inspection_bounds_min": lower.tolist(),
                "inspection_bounds_max": upper.tolist(),
                "padding_world": padding.tolist(),
                "mesh_triangles_in_volume": int(mesh_total),
                "selected_ray_indices": selected.tolist(),
                "views": [],
            }
            sidecar_base = f"{target}__{stem}"
            np.savez_compressed(
                sidecar_dir / f"{sidecar_base}.npz",
                pixel=np.asarray(record["pixel"][target_indices]),
                depth=np.asarray(record["depth"][target_indices]),
                world_xyz=np.asarray(record["world_xyz"][target_indices]),
                triangle_id=np.asarray(record["triangle_id"][target_indices]),
                component_id=np.asarray(record["component_id"][target_indices]),
                barycentric=np.asarray(record["barycentric"][target_indices]),
                ray_origin=np.asarray(ladder["origin"][target_indices]),
                ray_direction=np.asarray(ladder["direction"][target_indices]),
                q_before=np.asarray(ladder["q_before"][target_indices]),
                q_surface=np.asarray(ladder["q_surface"][target_indices]),
                q_behind=np.asarray(ladder["q_behind"][target_indices]),
            )
            _write_json(sidecar_dir / f"{sidecar_base}.json", {**target_summary, "sidecar_npz": f"{sidecar_base}.npz", "ray_subsampling_rule": "fixed row-major stride, maximum 32 line corridors; all first-hit rows remain in sidecar and all hit points remain in PNG", "mesh_crop_rule": "all raw triangles whose AABB intersects camera-to-hit inspection AABB; no component/status filtering", "mesh_triangle_ids_in_png_sample": mesh_ids.tolist(), "mesh_triangle_sample_rule": "first raw-order triangles up to fixed raster cap; full crop count is exact"})

            for kind in VIEW_KINDS:
                directory = out / target_dir_names[kind]
                _write_readme(directory / "README.md", _target_readme(kind))
                spotlight = kind == "component_first_hit_spotlight"
                spotlight_indices = target_fragments if spotlight and len(target_fragments) else target_indices
                spotlight_major = np.isin(np.asarray(record["component_id"][spotlight_indices], dtype=np.int64), np.asarray(sorted(major_ids), dtype=np.int64))
                spotlight_selected = np.unique(np.concatenate((select_ray_indices(spotlight_indices), select_ray_indices(target_indices)))) if spotlight else selected
                spotlight_triangles: dict[int, tuple[np.ndarray, bool]] = {}
                for index in spotlight_indices.tolist():
                    triangle_id = int(record["triangle_id"][index])
                    spotlight_triangles[triangle_id] = (vertices[faces[triangle_id]], bool(int(record["component_id"][index]) in major_ids))
                # Spotlight keeps every non-top-20 ray when present; context
                # points remain visible in the other three required views.
                if spotlight and len(target_fragments):
                    spotlight_hit_indices = spotlight_indices
                else:
                    spotlight_hit_indices = target_indices
                view_summary = render_view(
                    directory / f"{stem}.png",
                    kind,
                    camera_name,
                    target,
                    camera_center,
                    mesh_triangles,
                    spotlight_triangles if spotlight else hit_triangles,
                    ladder,
                    spotlight_hit_indices,
                    spotlight_major if spotlight and len(target_fragments) else target_major,
                    (lower, upper),
                    spotlight_selected,
                    mesh_total,
                    spotlight=spotlight,
                )
                target_summary["views"].append(view_summary)
            summaries.append(target_summary)

    report = {
        "status": "COMPLETE_W167_2_DEPTH_PRESERVING_LOCAL_3D_REVIEW",
        "parent_worklog": "W167",
        "secondary_parent_review": "W167-1",
        "architecture_verdict_preserved": real_report.get("architecture_verdict", "REAL_SCENE_REVIEW_REQUIRED"),
        "intent_alignment": "display-only local 3D review over frozen W167 saved first-hit records",
        "w167_data_reuse": {
            "ray_result_files": [f"{Path(name).stem}_ray_results.npz" for name in REVIEW_CAMERAS],
            "same_triangle_id_component_id_world_xyz_depth_pixel_identity": True,
            "raw_mesh_source": str(Path(args.raw_mesh)),
            "intersection_rerun": False,
            "query_ladder_changed": False,
            "component_attribution_changed": False,
        },
        "why_w167_1_was_insufficient": "same-ray reprojection confirms projection consistency only; it cannot distinguish which 3D zero-set sheet is physically intended",
        "local_3d_visualization_contract": {
            "primary_evidence": "world-space local mesh and depth-ordered ray geometry",
            "inspection_crop": "camera-to-target AABB with fixed proportional padding",
            "padding_fraction": LOCAL_PADDING_FRACTION,
            "minimum_padding_world": MIN_LOCAL_PADDING,
            "ray_display_max": RAY_DISPLAY_MAX,
            "all_first_hit_points_preserved": True,
            "non_top20_rays_preserved_in_spotlights": True,
            "mesh_display_cap": MAX_RENDER_TRIANGLES,
            "mesh_cap_is_display_only": True,
        },
        "camera_target_coverage": {camera: {target: 0 for target in TARGETS} for camera in REVIEW_CAMERAS},
        "component_spotlight_coverage": {
            "unique_non_top20_first_hit_keys": [[camera, index] for camera, index in sorted(unique_fragment_keys)],
            "covered_unique_non_top20_first_hit_keys": [[camera, index] for camera, index in sorted(covered_fragment_keys)],
            "all_unique_non_top20_hits_covered": unique_fragment_keys <= covered_fragment_keys,
        },
        "generated_review_artifacts": summaries,
        "human_review_questions_now_answerable": [
            "Which local 3D raw zero-set surface or triangle is first along the displayed ray?",
            "Does the first-hit triangle/component appear to be the intended tabletop, contact, lower table, or vase/curved-neighbor physical surface?",
            "Are non-top-20 first hits isolated fragments or coherent local geometry when viewed with depth?",
        ],
        "limitations": [
            "The local PNG omits raw mesh outside its disclosed inspection AABB; omission is a crop boundary, not evidence of absence.",
            "A fixed raster cap can reduce mesh stroke density in PNGs; exact crop triangle IDs remain in sidecars.",
            "No physical hidden-surface ground truth is available, so this review does not upgrade W167's architecture verdict.",
        ],
        "retained_open": "RETAINED_W167_REAL_SCENE_REVIEW_REQUIRED; physical surface intent remains a human-review question",
    }
    for row in summaries:
        report["camera_target_coverage"][row["camera"]][row["target"]] = int(row["first_hit_count"])
    _write_json(out / "worklog_167_2_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_W167_ROOT)
    parser.add_argument("--raw-mesh", type=Path, default=DEFAULT_RAW_MESH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--sparse-dir", default=DEFAULT_SPARSE_DIR)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--llffhold", type=int, default=DEFAULT_LLFFHOLD)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
