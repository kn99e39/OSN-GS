"""Worklog 167-1: close image-space review exports for the frozen W167 replay.

This batch is deliberately an export-layer-only consumer of W167 artifacts.  It
does not load the W166 mesh, rerun ray/triangle intersection, change the query
ladder, or alter the W167 architecture verdict.  The only regenerated values
are Q_before/Q_surface/Q_behind world points from the saved W167 hit depth and
the same camera-ray construction, followed by read-only camera projection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for _path in (str(DEVTOOLS), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # Direct script execution puts devtools/demo on sys.path.
    from worklog_167_raw_zero_set_ray_blocker_audit import (  # noqa: E402
        CONTACT_REGIONS,
        REAL_QUERY_OFFSET,
        REL_BEHIND_ZEROSET_SURFACE,
        REL_IN_FRONT_OF_ZEROSET_SURFACE,
        REL_NO_DECISION,
        REL_ZEROSET_FIRST_SURFACE,
        REVIEW_CAMERAS,
        REVIEW_POLYGONS,
        STATUS_AMBIGUOUS,
        STATUS_HIT,
        STATUS_NO_HIT,
        _polygon_mask,
        _real_camera_rays,
    )
except ModuleNotFoundError:  # Package import is used by focused tests.
    from devtools.demo.worklog_167_raw_zero_set_ray_blocker_audit import (  # noqa: E402
        CONTACT_REGIONS,
        REAL_QUERY_OFFSET,
        REL_BEHIND_ZEROSET_SURFACE,
        REL_IN_FRONT_OF_ZEROSET_SURFACE,
        REL_NO_DECISION,
        REL_ZEROSET_FIRST_SURFACE,
        REVIEW_CAMERAS,
        REVIEW_POLYGONS,
        STATUS_AMBIGUOUS,
        STATUS_HIT,
        STATUS_NO_HIT,
        _polygon_mask,
        _real_camera_rays,
    )


DEFAULT_W167_ROOT = REPO_ROOT / "output/167_raw_zero_set_ray_blocker_audit/real_scene"
DEFAULT_OUT = DEFAULT_W167_ROOT / "review_views"
DEFAULT_DATASET = REPO_ROOT / "DATASET"
DEFAULT_IMAGES = "images_8"
DEFAULT_SPARSE_DIR = "sparse/0"
DEFAULT_RESOLUTION = -1
DEFAULT_LLFFHOLD = 8
MIN_CROP_WIDTH = 96
MIN_CROP_HEIGHT = 64

FULL_DIR = "first_hit_overlay_full"
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

COLOR = {
    "first_hit": (40, 224, 106),
    "top20": (40, 224, 106),
    "non_top20": (255, 137, 32),
    "before": (44, 194, 244),
    "surface": (255, 214, 46),
    "behind": (232, 62, 62),
    "no_decision": (160, 160, 164),
    "text": (245, 245, 245),
    "legend_bg": (12, 16, 22, 225),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ray_record(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _camera_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def project_world_points(camera: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using W167's row-vector full projection matrix."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    clip = homogeneous @ _camera_matrix(camera.full_proj_transform)
    w = clip[:, 3]
    safe_w = np.where(w != 0.0, w, 1.0)
    width = int(camera.image_width)
    height = int(camera.image_height)
    x = ((clip[:, 0] / safe_w + 1.0) * width - 1.0) * 0.5
    y = ((clip[:, 1] / safe_w + 1.0) * height - 1.0) * 0.5
    projected = np.column_stack((x, y))
    valid = np.isfinite(projected).all(axis=1) & np.isfinite(w) & (w != 0.0)
    valid &= (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    return projected, valid


def _target_polygon_points(camera_name: str, target: str) -> np.ndarray:
    if target == "tabletop_vase_contact":
        polygons = [REVIEW_POLYGONS[name][camera_name] for name in CONTACT_REGIONS]
        return np.concatenate((np.asarray(polygons[0], dtype=np.float64), np.asarray(polygons[1], dtype=np.float64)), axis=0)
    return np.asarray(REVIEW_POLYGONS[target][camera_name], dtype=np.float64)


def target_mask(camera_name: str, pixels: np.ndarray, target: str) -> np.ndarray:
    if target == "tabletop_vase_contact":
        return np.logical_or(
            _polygon_mask(pixels, REVIEW_POLYGONS[CONTACT_REGIONS[0]][camera_name]),
            _polygon_mask(pixels, REVIEW_POLYGONS[CONTACT_REGIONS[1]][camera_name]),
        )
    return _polygon_mask(pixels, REVIEW_POLYGONS[target][camera_name])


def derive_crop_box(
    image_size: tuple[int, int],
    support_xy: np.ndarray,
    projected_hit_xy: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Derive one deterministic tight crop from frozen support and optional hit extent."""

    width, height = image_size
    support_xy = np.asarray(support_xy, dtype=np.float64).reshape(-1, 2)
    projected_hit_xy = np.asarray(projected_hit_xy if projected_hit_xy is not None else np.zeros((0, 2)), dtype=np.float64).reshape(-1, 2)
    pieces = [support_xy]
    if len(projected_hit_xy):
        pieces.append(projected_hit_xy)
    extent = np.concatenate(pieces, axis=0)
    xmin, ymin = extent.min(axis=0)
    xmax, ymax = extent.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    padding = max(12.0, math.ceil(0.15 * span))
    x0 = max(0, int(math.floor(xmin - padding)))
    y0 = max(0, int(math.floor(ymin - padding)))
    x1 = min(width, int(math.ceil(xmax + padding + 1.0)))
    y1 = min(height, int(math.ceil(ymax + padding + 1.0)))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)

    def expand_center(start: int, end: int, minimum: int, limit: int) -> tuple[int, int]:
        if end - start >= minimum or limit <= minimum:
            return start, end
        center = 0.5 * (start + end)
        start = int(math.floor(center - minimum * 0.5))
        end = start + minimum
        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit
        return max(0, start), min(limit, end)

    x0, x1 = expand_center(x0, x1, MIN_CROP_WIDTH, width)
    y0, y1 = expand_center(y0, y1, MIN_CROP_HEIGHT, height)
    return x0, y0, x1, y1


def derive_spotlight_box(
    image_size: tuple[int, int],
    suspicious_xy: np.ndarray,
) -> tuple[int, int, int, int]:
    """Derive a deterministic close box around non-top-20 projected hits."""

    width, height = image_size
    points = np.asarray(suspicious_xy, dtype=np.float64).reshape(-1, 2)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    padding = max(18.0, math.ceil(0.50 * span))
    x0 = max(0, int(math.floor(xmin - padding)))
    y0 = max(0, int(math.floor(ymin - padding)))
    x1 = min(width, int(math.ceil(xmax + padding + 1.0)))
    y1 = min(height, int(math.ceil(ymax + padding + 1.0)))
    x0, x1 = max(0, x0), min(width, max(x0 + 1, x1))
    y0, y1 = max(0, y0), min(height, max(y0 + 1, y1))

    def expand_center(start: int, end: int, minimum: int, limit: int) -> tuple[int, int]:
        if end - start >= minimum or limit <= minimum:
            return start, end
        center = 0.5 * (start + end)
        start = int(math.floor(center - minimum * 0.5))
        end = start + minimum
        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit
        return max(0, start), min(limit, end)

    x0, x1 = expand_center(x0, x1, MIN_CROP_WIDTH, width)
    y0, y1 = expand_center(y0, y1, MIN_CROP_HEIGHT, height)
    return x0, y0, x1, y1


def _rgb_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _draw_points(
    image: Any,
    points_xy: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
    alpha: int = 235,
    shape: str = "dot",
) -> Any:
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x, y in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
        bounds = (float(x - radius), float(y - radius), float(x + radius), float(y + radius))
        if shape == "ring":
            draw.ellipse(bounds, outline=(*color, alpha), width=max(1, int(radius // 2)))
        elif shape == "square":
            draw.rectangle(bounds, fill=(*color, alpha))
        elif shape == "cross":
            draw.line((float(x - radius), float(y), float(x + radius), float(y)), fill=(*color, alpha), width=max(1, int(radius // 2)))
            draw.line((float(x), float(y - radius), float(x), float(y + radius)), fill=(*color, alpha), width=max(1, int(radius // 2)))
        else:
            draw.ellipse(bounds, fill=(*color, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _label_image(image: Any, title: str, legend: str) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    base = image.convert("RGB")
    font = ImageFont.load_default(size=7)
    max_chars = max(10, (base.width - 12) // 5)

    def wrap(text: str) -> list[str]:
        words = text.replace("|", " | ").split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    title_lines = wrap(title)
    legend_lines = wrap(legend)
    line_height = 10
    band_height = 4 + line_height * (len(title_lines) + len(legend_lines))
    output = Image.new("RGB", (base.width, base.height + band_height), COLOR["legend_bg"][:3])
    output.paste(base, (0, band_height))
    draw = ImageDraw.Draw(output)
    y = 2
    for line in title_lines:
        draw.text((7, y), line, fill=COLOR["text"], font=font)
        y += line_height
    for line in legend_lines:
        draw.text((7, y), line, fill=(210, 215, 222), font=font)
        y += line_height
    return output.convert("RGB")


def _upscale(image: Any, full_frame: bool = False) -> Any:
    from PIL import Image

    if full_frame:
        scale = 2
    else:
        scale = max(2, min(8, 900 // max(image.width, image.height, 1)))
    if scale == 1:
        return image
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _save_png(path: Path, image: Any, full_frame: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _upscale(image, full_frame=full_frame).save(path, format="PNG", optimize=True)


def _render_full_overlay(image: Any, projected_xy: np.ndarray, camera_name: str) -> Any:
    image = _draw_points(image, projected_xy, COLOR["first_hit"], radius=1, alpha=220)
    return _label_image(
        image,
        f"167-1 FULL HIT | {camera_name}",
        "RGB base | green=first-hit",
    )


def _crop_render(
    image: Any,
    crop_box: tuple[int, int, int, int],
    title: str,
    legend: str,
    overlays: Iterable[tuple[np.ndarray, tuple[int, int, int], int, int, str]],
) -> Any:
    x0, y0, x1, y1 = crop_box
    cropped = image.crop((x0, y0, x1, y1)).convert("RGB")
    for points_xy, color, radius, alpha, shape in overlays:
        local = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
        if len(local):
            local[:, 0] -= x0
            local[:, 1] -= y0
        cropped = _draw_points(cropped, local, color, radius=radius, alpha=alpha, shape=shape)
    return _label_image(cropped, title, legend)


def _shared_readme(path: Path, title: str, kind: str, crop_rule: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("README.md").write_text(
        f"# {title}\n\n"
        "Input/state: read-only W167 real-scene ray-result NPZ, frozen camera metadata, frozen W160-W165 ROI definitions, and the original RGB frame. W167 geometry, ray intersection, query semantics, and architecture result are not regenerated or changed.\n\n"
        f"Visualization: `{kind}`. {crop_rule}\n\n"
        "Colors: green = first-hit surface or top-20 component; orange = non-top-20 component; cyan = Q_before; yellow = Q_surface / exact first-hit point; red = Q_behind. The original RGB image remains the background.\n\n"
        "Query ladder: Q_before is the saved W167 ray at first-hit depth minus the frozen offset; Q_surface is the saved first-hit point; Q_behind is the same ray at first-hit depth plus the frozen offset. Non-top-20 means a disconnected component outside W167's exact top-20-by-vertex-count attribution set; it is an attribution label, not a filtering or false-blocker decision.\n\n"
        "Review limitation: real-scene physical hidden-surface ground truth is unavailable. These images make W167 evidence readable for human surface-alignment review and do not upgrade `REAL_SCENE_REVIEW_REQUIRED`.\n",
        encoding="utf-8",
    )


def _friendly_target(target: str) -> str:
    return TARGET_LABELS[target]


def _camera_image_path(dataset: Path, images: str, camera_name: str) -> Path:
    path = dataset / images / camera_name
    if path.exists():
        return path
    stem = Path(camera_name).stem
    for suffix in (".JPG", ".jpg", ".PNG", ".png"):
        candidate = dataset / images / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


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
    missing = [name for name in args.camera_names if name not in by_name]
    if missing:
        raise KeyError(f"camera metadata missing: {missing}")
    return by_name


def _major_ids(real_report: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for camera in real_report["per_camera"].values():
        for row in camera.get("component_provenance", []):
            if bool(row.get("major_component")):
                ids.add(int(row["component_id"]))
    return ids


def reconstruct_query_ladder(camera: Any, record: dict[str, np.ndarray], offset: float = REAL_QUERY_OFFSET) -> dict[str, Any]:
    """Regenerate only W167's saved-ray ladder points from hit records."""

    pixels = np.asarray(record["pixel"], dtype=np.int64)
    status = np.asarray(record["status"]).astype(str)
    hit_depth = np.asarray(record["depth"], dtype=np.float64)
    first_hit_world = np.asarray(record["world_xyz"], dtype=np.float64)
    hit_mask = status == STATUS_HIT
    rays = _real_camera_rays(camera, pixels)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(hit_mask).tolist():
        depth = float(hit_depth[index])
        for label, delta, relation in (
            ("Q_before", -offset, REL_IN_FRONT_OF_ZEROSET_SURFACE),
            ("Q_surface", 0.0, REL_ZEROSET_FIRST_SURFACE),
            ("Q_behind", offset, REL_BEHIND_ZEROSET_SURFACE),
        ):
            point = rays.origins[index] + (depth + delta) * rays.directions[index]
            if label == "Q_surface":
                point = first_hit_world[index]
            rows.append(
                {
                    "ray_index": int(index),
                    "label": label,
                    "relation": relation,
                    "query_depth": float(depth + delta),
                    "query_world_xyz": point.tolist(),
                    "first_hit_depth": depth,
                    "first_hit_world_xyz": first_hit_world[index].tolist(),
                    "component_id": int(record["component_id"][index]),
                    "triangle_id": int(record["triangle_id"][index]),
                    "offset": float(delta),
                    "candidate_b_median_used": False,
                    "surface_membership_threshold_used": False,
                }
            )
    return {
        "rows": rows,
        "rays": rays,
        "hit_mask": hit_mask,
        "status": status,
    }


def _project_query_rows(camera: Any, rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    groups: dict[str, list[np.ndarray]] = {"Q_before": [], "Q_surface": [], "Q_behind": []}
    for row in rows:
        groups[row["label"]].append(np.asarray(row["query_world_xyz"], dtype=np.float64))
    projected: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for label, values in groups.items():
        points = np.asarray(values, dtype=np.float64).reshape(-1, 3) if values else np.zeros((0, 3), dtype=np.float64)
        xy, valid = project_world_points(camera, points)
        projected[label] = xy
        audits[label] = {"count": int(len(points)), "inside_image_count": int(valid.sum()), "outside_image_count": int((~valid).sum())}
    return projected, audits


def _add_reprojection_error_audit(
    audit: dict[str, Any],
    projected: np.ndarray,
    expected_xy: np.ndarray,
) -> None:
    expected_xy = np.asarray(expected_xy, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if len(projected) != len(expected_xy):
        audit["reprojection_error_pixels"] = {"count": 0, "max": None, "p95": None, "reason": "shape mismatch"}
        return
    error = np.linalg.norm(projected - expected_xy, axis=1) if len(projected) else np.zeros((0,), dtype=np.float64)
    audit["reprojection_error_pixels"] = {
        "count": int(len(error)),
        "max": float(error.max()) if len(error) else None,
        "p95": float(np.percentile(error, 95.0)) if len(error) else None,
    }


def _layout_report(root: Path, dirs: list[Path], camera_names: tuple[str, ...]) -> dict[str, Any]:
    expected = {f"{Path(name).stem}.png" for name in camera_names}
    violations: list[str] = []
    directory_rows: list[dict[str, Any]] = []
    for directory in dirs:
        if not (directory / "README.md").exists():
            violations.append(f"missing README.md: {directory}")
        nested = [str(path.relative_to(root)) for path in directory.iterdir() if path.is_dir()]
        if nested:
            violations.append(f"nested camera directory not allowed: {directory}")
        actual = {path.name for path in directory.glob("*.png")}
        missing = sorted(expected - actual)
        if missing:
            violations.extend(f"missing camera PNG {name}: {directory}" for name in missing)
        directory_rows.append({"path": str(directory), "readme": (directory / "README.md").exists(), "camera_pngs": sorted(actual), "nested_directories": nested})
    return {"root": str(root), "shared_readme_plus_direct_camera_png_rule": not violations, "violations": violations, "directories": directory_rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    w167_root = args.w167_root.resolve()
    parent_report_path = w167_root.parent / "worklog_167_report.json"
    real_report_path = w167_root / "real_scene_replay_report.json"
    parent_report = _read_json(parent_report_path)
    real_report = _read_json(real_report_path)
    if parent_report["architecture_result"]["verdict"] != "REAL_SCENE_REVIEW_REQUIRED":
        raise RuntimeError("W167 architecture verdict changed; 167-1 must not reinterpret it")
    if real_report["camera_names"] != list(args.camera_names):
        raise RuntimeError("camera set differs from frozen W167 real-scene replay")

    cameras = _load_cameras(args)
    args.out.mkdir(parents=True, exist_ok=True)
    _shared_readme(
        args.out,
        "Worklog 167-1 real-scene image-space review exports",
        "parent review_views directory",
        "All child directories use one shared README and direct camera PNG files; no camera subdirectories are used.",
    )
    major_ids = _major_ids(real_report)
    per_camera: dict[str, Any] = {}
    visualization_dirs: list[Path] = []
    target_dirs: dict[str, dict[str, Any]] = {}

    full_dir = args.out / FULL_DIR
    _shared_readme(full_dir, "FIRST_HIT_OVERLAY_FULL", FULL_DIR, "Each original RGB frame is shown at full extent with all valid W167 first-hit pixels projected back into the image.")
    visualization_dirs.append(full_dir)

    for target in TARGETS:
        target_dirs[target] = {}
        target_dirs[target]["first_hit"] = args.out / f"{target}_first_hit_overlay_cropped"
        target_dirs[target]["component"] = args.out / f"{target}_component_provenance_cropped"
        target_dirs[target]["ladder"] = args.out / f"{target}_query_ladder_cropped"
        target_dirs[target]["spotlight"] = args.out / f"{target}_suspicious_hit_spotlight"
        _shared_readme(target_dirs[target]["first_hit"], "FIRST_HIT_OVERLAY_CROPPED", target_dirs[target]["first_hit"].name, "The crop is derived from frozen ROI polygon support only, with 15% padding, a 12-pixel minimum padding, and a 96x64 minimum readable canvas. W167-associated points outside this crop remain disclosed in the report and are not reclassified.")
        _shared_readme(target_dirs[target]["component"], "COMPONENT_PROVENANCE_CROPPED", target_dirs[target]["component"].name, "The crop uses the same deterministic target box as the first-hit crop; green is top-20 and orange is non-top-20.")
        _shared_readme(target_dirs[target]["ladder"], "QUERY_LADDER_CROPPED", target_dirs[target]["ladder"].name, "The crop uses the same deterministic target box and shows the three query points for each identical W167 ray.")
        visualization_dirs.extend([target_dirs[target]["first_hit"], target_dirs[target]["component"], target_dirs[target]["ladder"]])
        spotlight_required = any(
            int(real_report["per_camera"][camera_name]["roi_counts"][target]["fragment_first_hit_count"]) > 0
            for camera_name in args.camera_names
        )
        target_dirs[target]["spotlight_required"] = spotlight_required
        if spotlight_required:
            _shared_readme(target_dirs[target]["spotlight"], "SUSPICIOUS_HIT_SPOTLIGHT", target_dirs[target]["spotlight"].name, "This directory exists because at least one frozen camera has a non-top-20 hit in this target. The suspicious camera uses the non-top-20 projected-hit extent with 50% max-span padding and an 18-pixel minimum; other cameras retain the same target crop and explicitly show that they have no non-top-20 hit.")
            visualization_dirs.append(target_dirs[target]["spotlight"])

    for camera_name in args.camera_names:
        camera = cameras[camera_name]
        record = _load_ray_record(w167_root / f"{Path(camera_name).stem}_ray_results.npz")
        image_path = _camera_image_path(args.dataset, args.images, camera_name)
        image = _rgb_image(image_path)
        pixels = np.asarray(record["pixel"], dtype=np.int64)
        status = np.asarray(record["status"]).astype(str)
        hit_mask = status == STATUS_HIT
        hit_points = np.asarray(record["world_xyz"], dtype=np.float64)
        projected_hits, hit_projected_valid = project_world_points(camera, hit_points[hit_mask])
        expected_hit_xy = pixels[hit_mask][:, [1, 0]].astype(np.float64)
        hit_components = np.asarray(record["component_id"], dtype=np.int64)[hit_mask]
        non_top_hit = ~np.isin(hit_components, np.asarray(sorted(major_ids), dtype=np.int64))
        full_image = _render_full_overlay(image, projected_hits[hit_projected_valid], camera_name)
        _save_png(full_dir / f"{Path(camera_name).stem}.png", full_image, full_frame=True)

        ladder = reconstruct_query_ladder(camera, record)
        projected_ladder, projection_audit = _project_query_rows(camera, ladder["rows"])
        _add_reprojection_error_audit(projection_audit["Q_before"], projected_ladder["Q_before"], expected_hit_xy)
        _add_reprojection_error_audit(projection_audit["Q_surface"], projected_ladder["Q_surface"], expected_hit_xy)
        _add_reprojection_error_audit(projection_audit["Q_behind"], projected_ladder["Q_behind"], expected_hit_xy)
        hit_projection_audit = {"count": int(len(projected_hits)), "inside_image_count": int(hit_projected_valid.sum()), "outside_image_count": int((~hit_projected_valid).sum())}
        _add_reprojection_error_audit(hit_projection_audit, projected_hits, expected_hit_xy)
        real_report_camera = real_report["per_camera"][camera_name]
        target_rows: dict[str, Any] = {}
        for target in TARGETS:
            mask = target_mask(camera_name, pixels, target)
            target_hit = mask & hit_mask
            target_projected_hits, target_hit_valid = project_world_points(camera, hit_points[target_hit])
            target_hit_components = np.asarray(record["component_id"], dtype=np.int64)[target_hit]
            target_non_top = ~np.isin(target_hit_components, np.asarray(sorted(major_ids), dtype=np.int64))
            support_xy = _target_polygon_points(camera_name, target)
            crop_box = derive_crop_box((int(image.width), int(image.height)), support_xy)
            stem = Path(camera_name).stem
            crop_hit_xy = target_projected_hits[target_hit_valid]
            target_non_top_valid = target_non_top[target_hit_valid]
            crop_hit_visible = (
                (crop_hit_xy[:, 0] >= crop_box[0])
                & (crop_hit_xy[:, 0] < crop_box[2])
                & (crop_hit_xy[:, 1] >= crop_box[1])
                & (crop_hit_xy[:, 1] < crop_box[3])
            )
            _save_png(
                target_dirs[target]["first_hit"] / f"{stem}.png",
                _crop_render(
                    image,
                    crop_box,
                    f"167-1 HIT CROP | {camera_name} | {_friendly_target(target)}",
                    "RGB crop | green=first-hit",
                    [(crop_hit_xy, COLOR["first_hit"], 1, 220, "dot")],
                ),
            )
            component_top_xy = crop_hit_xy[~target_non_top_valid]
            component_non_top_xy = crop_hit_xy[target_non_top_valid]
            _save_png(
                target_dirs[target]["component"] / f"{stem}.png",
                _crop_render(
                    image,
                    crop_box,
                    f"167-1 COMPONENT | {camera_name} | {_friendly_target(target)}",
                    "green=top20 | orange=non-top20",
                    [(component_top_xy, COLOR["top20"], 2, 220, "dot"), (component_non_top_xy, COLOR["non_top20"], 2, 245, "dot")],
                ),
            )
            target_rows[target] = {
                "target_present": True,
                "meaningful_in_camera": bool(mask.any()),
                "ray_count": int(mask.sum()),
                "first_hit_count": int(target_hit.sum()),
                "top20_component_hit_count": int((~target_non_top).sum()),
                "non_top20_component_hit_count": int(target_non_top.sum()),
                "crop_box_xyxy_original": list(crop_box),
                "crop_rule": "frozen ROI polygon support only, 15% max-span padding, 12-pixel minimum, 96x64 minimum readable canvas",
                "projected_first_hit_inside_image_count": int(target_hit_valid.sum()),
                "projected_first_hit_outside_image_count": int((~target_hit_valid).sum()),
                "projected_first_hit_inside_crop_count": int(crop_hit_visible.sum()),
                "w167_associated_hit_outside_target_crop_count": int((~crop_hit_visible).sum()),
                "w167_roi_count_match": int(mask.sum()) == int(real_report_camera["roi_counts"][target]["ray_count"]),
            }
            ladder_rows_by_label = {
                label: projected_ladder[label][target_mask(camera_name, pixels[ladder["hit_mask"]], target)]
                for label in ("Q_before", "Q_surface", "Q_behind")
            }
            _save_png(
                target_dirs[target]["ladder"] / f"{stem}.png",
                _crop_render(
                    image,
                    crop_box,
                    f"167-1 LADDER | {camera_name} | {_friendly_target(target)}",
                    "cyan=before | yellow=surface | red=behind",
                    [
                        (ladder_rows_by_label["Q_before"], COLOR["before"], 3, 230, "ring"),
                        (ladder_rows_by_label["Q_surface"], COLOR["surface"], 2, 255, "square"),
                        (ladder_rows_by_label["Q_behind"], COLOR["behind"], 3, 230, "cross"),
                    ],
                ),
            )
            if bool(target_dirs[target]["spotlight_required"]):
                spotlight_dir = target_dirs[target]["spotlight"]
                suspicious_xy = crop_hit_xy[target_non_top_valid]
                has_suspicious = len(suspicious_xy) > 0
                spotlight_box = derive_spotlight_box((int(image.width), int(image.height)), suspicious_xy) if has_suspicious else crop_box
                _save_png(
                    spotlight_dir / f"{stem}.png",
                    _crop_render(
                        image,
                        spotlight_box,
                        f"167-1 SPOTLIGHT | {camera_name} | {_friendly_target(target)}",
                        "orange=non-top20 | green=top20" if has_suspicious else "no non-top20 | green=top20",
                        [(crop_hit_xy[~target_non_top_valid], COLOR["top20"], 2, 180, "dot"), (suspicious_xy, COLOR["non_top20"], 4, 255, "cross")],
                    ),
                )
                target_rows[target]["spotlight"] = {"generated": True, "suspicious_hit_in_this_camera": has_suspicious, "spotlight_box_xyxy_original": list(spotlight_box), "rule": "suspicious projected-hit extent with 50% max-span padding, 18-pixel minimum; target crop when this camera has no suspicious hit"}
            else:
                target_rows[target]["spotlight"] = {"generated": False, "reason": "no non-top-20 component hit in this target"}

        expected_status = {STATUS_HIT: int((status == STATUS_HIT).sum()), STATUS_NO_HIT: int((status == STATUS_NO_HIT).sum()), STATUS_AMBIGUOUS: int((status == STATUS_AMBIGUOUS).sum())}
        per_camera[camera_name] = {
            "image_path": str(image_path.resolve()),
            "image_shape_hw": [int(image.height), int(image.width)],
            "w167_ray_record_count": int(len(pixels)),
            "w167_status_counts": expected_status,
            "projected_first_hit_inside_image_count": int(hit_projected_valid.sum()),
            "projected_first_hit_outside_image_count": int((~hit_projected_valid).sum()),
            "first_hit_projection_audit": hit_projection_audit,
            "query_ladder_regenerated_from_saved_hits": True,
            "query_ladder_offset_world": REAL_QUERY_OFFSET,
            "query_ladder_projection_audit": projection_audit,
            "target_coverage": target_rows,
            "non_top20_total_hit_count": int(non_top_hit.sum()),
        }

    layout = _layout_report(args.out, visualization_dirs, args.camera_names)
    target_coverage = {
        camera: {target: per_camera[camera]["target_coverage"][target] for target in TARGETS}
        for camera in args.camera_names
    }
    non_top_coverage = {
        camera: {target: per_camera[camera]["target_coverage"][target]["non_top20_component_hit_count"] for target in TARGETS}
        for camera in args.camera_names
    }
    outside_crop_cases = [
        {
            "camera": camera,
            "target": target,
            "w167_associated_hit_outside_target_crop_count": int(per_camera[camera]["target_coverage"][target]["w167_associated_hit_outside_target_crop_count"]),
        }
        for camera in args.camera_names
        for target in TARGETS
        if int(per_camera[camera]["target_coverage"][target]["w167_associated_hit_outside_target_crop_count"]) > 0
    ]
    report = {
        "status": "COMPLETE_WL167_1_IMAGE_SPACE_REVIEW_EXPORTS",
        "batch": "167-1",
        "parent_worklog": 167,
        "intent_alignment": {
            "export_only": True,
            "w167_geometry_changed": False,
            "w167_ray_mesh_intersection_changed": False,
            "w167_query_ladder_semantics_changed": False,
            "w167_architecture_result": parent_report["architecture_result"]["verdict"],
            "production_behavior_modified": False,
        },
        "input_reuse_vs_regeneration": {
            "reused_directly": [
                "W167 per-camera *_ray_results.npz pixel/status/depth/world_xyz/triangle_id/component_id/barycentric records",
                "W167 real_scene_replay_report.json component provenance and camera/ROI accounting",
                "W167 frozen camera set and W160-W165 ROI polygon definitions imported from the W167 export module",
                "DATASET/images_8 original RGB frames",
            ],
            "regenerated_read_only": [
                "Q_before/Q_surface/Q_behind world points from saved first-hit depth and W167 camera-ray construction",
                "camera projection of saved and regenerated points using full_proj_transform",
                "deterministic target crop boxes from frozen ROI support plus projected hit extent",
            ],
            "not_rerun": ["W166 mesh load", "ray-triangle intersection", "screen-tile broad phase", "component labeling", "architecture evaluation"],
            "roi_count_match_all_cameras": all(row[target]["w167_roi_count_match"] for row in target_coverage.values() for target in TARGETS),
        },
        "visualization_directory_structure": layout,
        "generated_visualization_types": {
            "first_hit_overlay_full": True,
            "first_hit_overlay_cropped": True,
            "component_provenance_cropped": True,
            "query_ladder_cropped": True,
            "suspicious_hit_spotlight": any(per_camera[camera]["target_coverage"][target]["spotlight"]["generated"] for camera in args.camera_names for target in TARGETS),
            "png_primary": True,
            "shared_readme_no_camera_subdirectories": layout["shared_readme_plus_direct_camera_png_rule"],
        },
        "target_roi_coverage_per_camera": target_coverage,
        "non_top20_component_hit_coverage": non_top_coverage,
        "non_top20_unique_w167_first_hit_total": int(sum(per_camera[camera]["non_top20_total_hit_count"] for camera in args.camera_names)),
        "human_review_relevant_observations": {
            "first_hit_projection_inside_image_for_all_expected_hits": all(per_camera[camera]["projected_first_hit_outside_image_count"] == 0 for camera in args.camera_names),
            "query_ladder_projection_inside_image_for_all_expected_queries": all(
                projection["outside_image_count"] == 0
                for camera in args.camera_names
                for projection in per_camera[camera]["query_ladder_projection_audit"].values()
            ),
            "w167_associated_hits_outside_tight_target_crop": outside_crop_cases,
            "observations": [
                "The full-frame overlays show first-hit locations in the actual RGB image rather than only in distant common-world coordinates.",
                "The four target crop families keep the frozen ROI intent and expose tabletop, contact, lower table, and vase/curved-neighbor alignment at readable scale.",
                "Component provenance separates top-20 attribution from non-top-20 disconnected hits without filtering or semantic reinterpretation.",
                "Query ladder colors preserve before/surface/behind ordering for identical rays in image space.",
                "Some W167-associated rays remain outside the tight polygon-support crop in specific camera/target pairs; they are disclosed rather than silently reclassified or used to widen the crop.",
                "Human review is still required for physical-surface alignment; no automatic success claim is made.",
            ],
        },
        "retained_rejected_open": {
            "retained": ["W167 raw zero-set geometry", "W167 ray-hit records", "W167 query offset and relation meanings", "W167 component provenance", "W167 camera set and ROI intent", "REAL_SCENE_REVIEW_REQUIRED"],
            "rejected": ["geometry improvement", "mesh repair/filtering", "new blocker semantics", "new component threshold", "architecture upgrade", "automatic success claim"],
            "open": ["human judgment of physical-surface alignment in the generated RGB overlays", "whether any highlighted non-top-20 hit is meaningful structure or nuisance fragment"],
        },
    }
    _write_json(args.out / "worklog_167_1_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w167-root", type=Path, default=DEFAULT_W167_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--sparse-dir", default=DEFAULT_SPARSE_DIR)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--llffhold", type=int, default=DEFAULT_LLFFHOLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera-names", nargs=3, default=list(REVIEW_CAMERAS))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    report = run(args)
    print(json.dumps({"status": report["status"], "architecture_result_preserved": report["intent_alignment"]["w167_architecture_result"], "layout_ok": report["visualization_directory_structure"]["shared_readme_plus_direct_camera_png_rule"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
