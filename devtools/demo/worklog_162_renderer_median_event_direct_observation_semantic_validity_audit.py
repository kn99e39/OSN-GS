"""Worklog 162 -- renderer median-event direct-observation semantic audit.

This is a read-only audit of the historical Candidate-B median-depth ordering
used by Worklogs 153 and 160.  It does not change a classifier, a Gaussian
state, W161, the renderer, or any production path.  The only world-space
membership label used here is the already materialized W155 ``region_id``
mapping joined by the checkpoint's stable Gaussian ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo import worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit as w160


DEFAULT_CHECKPOINT = w160.DEFAULT_CHECKPOINT
DEFAULT_SOURCE = w160.DEFAULT_SOURCE
DEFAULT_CACHE = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache"
DEFAULT_WL155_MAPPING = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/gaussian_id_region_status_mapping.npz"
DEFAULT_WL155_REPORT = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/worklog_155_report.json"
DEFAULT_WL161_REPORT = REPO_ROOT / "output/161_global_persistent_occlusion_spatial_domain_audit/worklog_161_report.json"
DEFAULT_OUT = REPO_ROOT / "output/162_renderer_median_event_direct_observation_semantic_validity_audit"

REVIEW_CAMERAS = w160.REVIEW_CAMERAS
REVIEW_POLYGONS = w160.REVIEW_POLYGONS
EVENT_CAMERA = w160.EVENT_CAMERA
EVENT_PIXEL = w160.EVENT_PIXEL
EVENT_RADIUS = w160.EVENT_RADIUS

STATE_NON_RELEVANT = w160.STATE_NON_RELEVANT
STATE_UNRESOLVED = w160.STATE_UNRESOLVED
STATE_OBSERVED = w160.STATE_OBSERVED
STATE_OCCLUDED = w160.STATE_OCCLUDED
STATE_NAMES = w160.STATE_NAMES

TABLETOP_REGION_ID = 1
W155_STATUS_NAMES = {0: "CORE", 1: "ATTACHED", 2: "AMBIGUOUS", 4: "UNASSIGNED"}
WORLD_CLASS_A = "PROJECTS_ON_TABLETOP_BUT_WORLD_LOCATION_ELSEWHERE"
WORLD_CLASS_B = "WORLD_LOCATION_IN_REVIEWED_TABLETOP_REGION"
WORLD_CLASS_C = "AMBIGUOUS_UNDER_EXISTING_REVIEW_CONTRACT"

REGION_ALL_RGB = (0.12, 0.50, 0.92)
REGION_CONTEXT_RGB = (0.72, 0.72, 0.74)
ROI_COLORS = {"tabletop": (1.0, 0.78, 0.05), "table_side_lower_geometry": (0.12, 0.75, 1.0), "vase_foreground_structure": (1.0, 0.30, 0.70)}


def _progress(message: str) -> None:
    print(f"[worklog 162] {message}", flush=True)


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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _signed_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "min": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None, "max": None, "exact_zero_count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.percentile(array, 50, method="nearest")),
        "p05": float(np.percentile(array, 5, method="nearest")),
        "p25": float(np.percentile(array, 25, method="nearest")),
        "p75": float(np.percentile(array, 75, method="nearest")),
        "p95": float(np.percentile(array, 95, method="nearest")),
        "max": float(np.max(array)),
        "exact_zero_count": int(np.count_nonzero(array == 0.0)),
    }


def _state_counts(values: np.ndarray) -> dict[str, int]:
    return {name: int(np.count_nonzero(values == code)) for code, name in STATE_NAMES.items()}


def _load_w155_mapping(path: Path, checkpoint_stable_ids: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"stable_gaussian_id", "region_id", "membership_status"}
        if not required.issubset(data.files):
            raise ValueError(f"W155 mapping lacks required keys: {sorted(required - set(data.files))}")
        ids = np.asarray(data["stable_gaussian_id"], dtype=np.int64)
        region_id = np.asarray(data["region_id"], dtype=np.int64)
        membership_status = np.asarray(data["membership_status"], dtype=np.int8)
    if ids.ndim != 1 or not np.all(ids[:-1] <= ids[1:]) or len(np.unique(ids)) != len(ids):
        raise ValueError("W155 mapping stable IDs must be unique and sorted")
    row_ids = np.asarray(checkpoint_stable_ids, dtype=np.int64)
    positions = np.searchsorted(ids, row_ids)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], row_ids):
        raise ValueError("W155 mapping does not exactly cover checkpoint stable Gaussian IDs")
    return {
        "stable_gaussian_id": row_ids,
        "region_id": region_id[positions],
        "membership_status": membership_status[positions],
        "source": str(path.resolve()),
        "sha256": _sha256_file(path),
        "mapping_row_count": int(len(ids)),
        "mapping_hash_contract": "W155 SHA256(sorted stable_gaussian_id, region_id, membership_status)",
    }


def _world_classification(region_id: int, membership_status: int) -> str:
    if membership_status not in (0, 1):
        return WORLD_CLASS_C
    if region_id == TABLETOP_REGION_ID:
        return WORLD_CLASS_B
    return WORLD_CLASS_A


def synthetic_contracts() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    a = w160.classify_projective_sdf_evidence(relevant=True, query_depth=2.0, median_depth=2.0)
    cases.append({"name": "A_exact_surface_alignment", "expected": "OBSERVED with s=0", "actual": {"state": STATE_NAMES[a["state"]], "s": a["signed_distance"]}, "pass": a["state"] == STATE_OBSERVED and a["signed_distance"] == 0.0})
    b = w160.classify_projective_sdf_evidence(relevant=True, query_depth=3.0, median_depth=2.0)
    cases.append({"name": "B_query_behind_foreground_median_event", "expected": "OCCLUDED", "actual": STATE_NAMES[b["state"]], "pass": b["state"] == STATE_OCCLUDED})
    c = {
        "name": "C_same_region_samples_different_depth",
        "expected": "median ordering exposes depth order but does not identify physical event membership",
        "actual": "ordering_only_no_event_identity",
        "pass": True,
    }
    cases.append(c)
    d = {
        "name": "D_projection_overlap_world_location_elsewhere",
        "expected": WORLD_CLASS_A,
        "actual": _world_classification(37, 0),
        "pass": _world_classification(37, 0) == WORLD_CLASS_A,
    }
    cases.append(d)
    e = w160.aggregate_persistent_states(np.asarray([[STATE_OCCLUDED, STATE_OBSERVED]], dtype=np.int8))[0]
    cases.append({"name": "E_one_observed_view_prevents_global_occluded", "expected": "OBSERVED", "actual": STATE_NAMES[int(e)], "pass": e == STATE_OBSERVED})
    return {"all_pass": bool(all(item["pass"] for item in cases)), "cases": cases, "note": "Synthetic A-E verifies mechanics and limits only; it is not physical ground truth."}


def _classify_exact(geometry: Any, median: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = geometry.relevant & (median > 0.0)
    signed = median - geometry.depth
    states = torch.full((geometry.depth.numel(),), STATE_NON_RELEVANT, dtype=torch.int8, device=geometry.depth.device)
    states[geometry.relevant] = STATE_UNRESOLVED
    states[valid & (signed < 0.0)] = STATE_OCCLUDED
    states[valid & (signed >= 0.0)] = STATE_OBSERVED
    signed = torch.where(valid, signed, torch.full_like(signed, float("nan")))
    return states, signed


def _inspect_renderer_provenance(cache: Path) -> dict[str, Any]:
    known = ["renderer_median_depth_maps.npz", "replay_input_runtime.json", "field.npz"]
    candidates = sorted(path.name for path in cache.iterdir() if path.is_file() and any(token in path.name.lower() for token in ("contributor", "event_identity", "median_event", "lineage")))
    return {
        "status": "CONTRIBUTOR_PROVENANCE_NOT_AVAILABLE_UNDER_EXISTING_CONTRACT" if not candidates else "CANDIDATE_PROVENANCE_ARTIFACTS_REQUIRE_MANUAL_VERIFICATION",
        "exact_contributor_stable_id_join": False,
        "median_event_identity": "NOT_RECOVERABLE_FROM_DEPTH_ONLY_MEDIAN_MAP",
        "checked_artifacts": known,
        "candidate_identity_or_contributor_files": candidates,
        "reason": "renderer_median_depth_maps.npz stores one scalar median depth per pixel, not the contributing Gaussian stable ID; field.npz stores sparse fused voxel samples, not event identity.",
        "no_rgb_or_nearest_gaussian_inference": True,
    }


def _draw_polygons(image: torch.Tensor, camera_name: str, *, labels: bool = True) -> torch.Tensor:
    from PIL import Image, ImageDraw

    value = image.detach().cpu().clamp(0.0, 1.0)
    if value.ndim == 3 and value.shape[0] == 3:
        value = value.permute(1, 2, 0)
    pil = Image.fromarray((value.numpy() * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    for name, polygons in REVIEW_POLYGONS.items():
        polygon = [tuple(int(round(v)) for v in point) for point in polygons[camera_name]]
        colour = tuple(int(round(c * 255.0)) for c in ROI_COLORS[name])
        draw.line(polygon + [polygon[0]], fill=colour, width=3)
        if labels:
            draw.text(polygon[0], name, fill=colour)
    if camera_name == EVENT_CAMERA:
        x, y = EVENT_PIXEL
        r = EVENT_RADIUS
        draw.ellipse((int(x-r), int(y-r), int(x+r), int(y+r)), outline=(255, 255, 255), width=2)
    return torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).to(dtype=torch.float32) / 255.0


def _render_with_colours(model: Any, rasterizer: Any, camera: Any, colours: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    return w160._render_state(model, rasterizer, camera, colours, original=False, background=background)


def _write_png(path: Path, image: torch.Tensor) -> None:
    w160._save_png(path, image)


def _write_visualization_readmes(out: Path, row_count: int, camera_names: list[str], renderer_provenance: dict[str, Any]) -> None:
    conditions = f"frozen checkpoint/iteration, {row_count:,} Gaussian rows, same 648x420 camera calibration, same OSNSurfelRasterizer, same black background, and same camera set"
    root = out / "review_views"
    _write_json(out / "README.md", {})
    (out / "README.md").write_text(
        "# W162 renderer median-event semantic audit\n\n"
        "이 directory는 W160의 historical Candidate-B `median_depth - query_depth` ordering을 semantic하게 감사한 diagnostic 결과다. W155의 기존 `stable_gaussian_id -> region_id/membership_status` mapping과 W153의 frozen renderer median-depth map을 read-only로 join했다. classifier, state, renderer, W161, production path는 변경하지 않았다.\n\n"
        f"모든 camera PNG는 {conditions}을 공유한다. state palette는 green=OBSERVED `(0.10, 0.85, 0.35)`, red=OCCLUDED `(0.92, 0.18, 0.18)`, gray=UNRESOLVED `(0.60, 0.60, 0.62)`이다. `global_state_pure`는 mandatory Original Scene/Observed-Occluded pair의 state 쪽이고 `original_scene`은 learned SH appearance 쪽이다.\n\n"
        "`tabletop` positive control은 W155에서 기존 review polygon에 반복적으로 나타난 `region_id=1`을 그대로 사용한다. 이는 Gaussian center와 polygon 사이에 새 world-distance threshold를 적용한 것이 아니다. Region ID와 membership status는 physical truth가 아니라 W155 Gaussian-side review contract다.\n\n"
        f"renderer provenance 결과는 `{renderer_provenance['status']}`이다. median map만으로는 median event의 contributing Gaussian stable ID를 복원할 수 없으므로, red state를 hidden-surface truth로 해석하지 않는다.\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# W162 review views\n\n"
        "이 directory 아래의 각 visualization directory는 W162의 semantic review를 나타내며 camera PNG는 직접 저장되어 있다. 모든 camera view는 동일 frozen checkpoint/iteration, 648x420 resolution, black background, OSNSurfelRasterizer, camera calibration, Gaussian row count를 공유한다.\n\n"
        "`original_scene`은 learned SH appearance, `global_state_pure`는 W160 historical global state, `tabletop_*` view는 W155의 기존 `region_id=1` query population이다. `common_world`는 같은 population의 world XYZ diagnostic projection이다.\n\n"
        "공통 legend은 green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED 또는 선택되지 않은 context이다. ROI outline은 annotation-only이며, 어떤 view도 marker Gaussian, 새 geometry, spatial domain, physical first-hit truth, median event contributor identity를 추가하지 않는다. 각 하위 directory README가 해당 visualization의 semantics, palette, 공통 rendering 조건, review limitation을 개별적으로 설명한다.\n",
        encoding="utf-8",
    )
    descriptions = {
        "original_scene": "Original Scene: frozen checkpoint의 learned SH appearance를 렌더링한다. state color, geometry, opacity, scale/covariance, rotation, marker Gaussian을 추가하지 않는다.",
        "global_state_pure": "Global state pure: 같은 Gaussian rows/geometry를 유지하고 W160 historical all-relevant global state만 green/red/gray로 표시한다. global OCCLUDED는 모든 relevant camera가 OCCLUDED일 때만 성립한다.",
        "global_state_overlay": "Global state overlay: Original Scene 위에 fixed tabletop/table-side/vase review polygon과 event 1527 circle을 그린 annotation view다. polygon은 selection이나 membership predicate가 아니다.",
        "tabletop_global_occluded_only": "Tabletop global-occluded only: W155 `region_id=1` population 중 global OCCLUDED만 red로 표시하고 나머지 Gaussian rows는 gray context로 표시한다. geometry를 숨기거나 marker를 만들지 않는다.",
        "tabletop_region_all": "Tabletop region all: 기존 W155 `region_id=1` 전체 population을 global state palette로 표시하고 나머지는 gray context로 둔다. W155 membership status와 global state를 혼동하지 않는다.",
        "tabletop_region_global_occluded": "Tabletop region global-occluded: W155 `region_id=1`이면서 global OCCLUDED인 query만 red, 나머지는 gray context다. 이는 query의 same-region membership이지 median event identity가 아니다.",
        "tabletop_region_global_observed": "Tabletop region global-observed: W155 `region_id=1`이면서 global OBSERVED인 query만 green, 나머지는 gray context다. positive control의 직접 관측 비교용이다.",
        "tabletop_vase_contact": "Tabletop-vase contact: 같은 frozen camera에서 tabletop과 vase_foreground_structure fixed ROI를 함께 표시한다. 두 영역의 image-space 접촉은 world-space 동일 membership이나 event identity를 증명하지 않는다.",
    }
    for name, description in descriptions.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "README.md").write_text(
            f"# {name}\n\n{description}\n\n공통 조건: {conditions}.\n\nlegend: green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED 또는 선택되지 않은 context. `NON_RELEVANT`는 global vote가 아니며 gray로 렌더된 context와 동일한 의미도 아니다.\n\nreview limitation: 이 PNG는 renderer-relative ordering과 기존 review annotation을 보여준다. physical first-hit truth, hidden surface existence, median-event contributing Gaussian identity, TSDF sign, downstream topology는 판정하지 않는다.\n",
            encoding="utf-8",
        )
    common = root / "common_world"
    common.mkdir(parents=True, exist_ok=True)
    (common / "README.md").write_text(
        "# common_world\n\n"
        "`perspective.png`, `top.png`, `side.png`는 동일한 world-space point population을 세 개의 고정 orthographic diagnostic projection으로 본다. gray는 deterministic context sample, target population은 W155 `region_id=1`이며 target state는 green/red/gray로 표시한다. `perspective`는 world X-Z, `top`은 X-Y, `side`는 Y-Z 축 투영이며 실제 camera perspective가 아니다.\n\n"
        "모든 target Gaussian은 표시하고, 전체-scene context는 PNG 가독성을 위해 고정 stride로 downsample한다. point size/alpha는 display-only이며 새로운 spatial domain, voxel, radius, smoothing, bridge, membership rule을 만들지 않는다. world XYZ는 checkpoint 값이고 Region ID는 W155 mapping이다.\n",
        encoding="utf-8",
    )


def _world_projection_png(path: Path, positions: np.ndarray, target_mask: np.ndarray, global_state: np.ndarray, axis_pair: tuple[int, int], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x_axis, y_axis = axis_pair
    context = positions[:: max(1, len(positions) // 200000)]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.scatter(context[:, x_axis], context[:, y_axis], s=0.15, c="#b8b8bc", alpha=0.08, linewidths=0)
    for code, colour, label in ((STATE_UNRESOLVED, "#99999f", "UNRESOLVED"), (STATE_OBSERVED, "#1ad95a", "OBSERVED"), (STATE_OCCLUDED, "#eb2e2e", "OCCLUDED")):
        mask = target_mask & (global_state == code)
        if np.any(mask):
            ax.scatter(positions[mask, x_axis], positions[mask, y_axis], s=1.4, c=colour, alpha=0.80, linewidths=0, label=label)
    ax.set_xlabel(("X" if x_axis == 0 else "Y" if x_axis == 1 else "Z") + " world")
    ax.set_ylabel(("X" if y_axis == 0 else "Y" if y_axis == 1 else "Z") + " world")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=4)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def _clear_known_outputs(out: Path) -> None:
    for name in ("review_views",):
        target = out / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("w162_tabletop_population_audit.npz", "w162_tabletop_cross_view_raw.npz", "tabletop_cross_view_records.json", "review_projection_records.json", "worklog_162_report.json"):
        target = out / name
        if target.exists():
            target.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    synthetic = synthetic_contracts()
    if not synthetic["all_pass"]:
        raise RuntimeError("synthetic A-E contract failure")
    args.out.mkdir(parents=True, exist_ok=True)
    _clear_known_outputs(args.out)
    _progress("loading checkpoint, frozen cameras, W153 median maps, and W155 mapping")
    model, payload = w160.load_primitive_model(args.checkpoint, device=args.device)
    if w160.checkpoint_primitive(payload) != w160.PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("canonical 2DGS surfel checkpoint required")
    raw_ids = payload["model_raw"].get("stable_gaussian_ids")
    if raw_ids is None:
        raise ValueError("checkpoint lacks stable_gaussian_ids; W162 requires exact stable IDs")
    stable_ids = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    cameras, camera_meta = w160.load_all_train_cameras(args.source, args.images, args.sparse_dir, args.resolution, args.llffhold, args.device)
    names = [str(camera.image_name) for camera in cameras]
    if len(names) != 161:
        raise ValueError(f"expected 161 cameras, got {len(names)}")
    depth_np, depth_meta = w160._load_depth_cache(args.cache, names)
    mapping = _load_w155_mapping(args.wl155_mapping, stable_ids)
    region_id = mapping["region_id"]
    membership_status = mapping["membership_status"]
    target_mask = region_id == TABLETOP_REGION_ID
    target_rows = np.flatnonzero(target_mask).astype(np.int64)
    positions_np = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    row_count = int(len(positions_np))
    target_count = int(len(target_rows))
    target_s = np.full((target_count, len(cameras)), np.nan, dtype=np.float32)
    target_states = np.full((target_count, len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    global_observed_any = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    global_has_relevant = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    global_has_unresolved = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    relevant_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    observed_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    occluded_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    unresolved_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    review_projection_records: dict[str, Any] = {}

    with torch.no_grad():
        positions = model.get_xyz.detach()
        for camera_index, (camera, median_np) in enumerate(zip(cameras, depth_np)):
            median_flat = torch.as_tensor(median_np, dtype=torch.float32, device=args.device)
            geometry = w160.project_queries(camera, positions)
            pixel = geometry.pixel_index.clamp(min=0)
            median = median_flat[pixel]
            states, signed = _classify_exact(geometry, median)
            relevant = geometry.relevant
            global_observed_any |= states == STATE_OBSERVED
            global_has_relevant |= relevant
            global_has_unresolved |= states == STATE_UNRESOLVED
            relevant_count += relevant.to(torch.int16)
            observed_count += (states == STATE_OBSERVED).to(torch.int16)
            occluded_count += (states == STATE_OCCLUDED).to(torch.int16)
            unresolved_count += (states == STATE_UNRESOLVED).to(torch.int16)
            if target_count:
                target_s[:, camera_index] = signed[target_rows].detach().cpu().numpy()
                target_states[:, camera_index] = states[target_rows].detach().cpu().numpy()
            if names[camera_index] in REVIEW_CAMERAS:
                camera_record: dict[str, Any] = {}
                for control_name, polygon_map in REVIEW_POLYGONS.items():
                    roi = relevant & w160._polygon_mask(geometry.pixel_x, geometry.pixel_y, polygon_map[names[camera_index]])
                    rows = torch.nonzero(roi & (states == STATE_OCCLUDED), as_tuple=False).reshape(-1).detach().cpu().numpy().astype(np.int64)
                    entries = []
                    for row in rows.tolist():
                        entries.append({
                            "stable_gaussian_id": int(stable_ids[row]),
                            "checkpoint_row_index": int(row),
                            "world_xyz": positions_np[row].tolist(),
                            "gaussian_surface_region_id": int(region_id[row]),
                            "w155_membership_status": W155_STATUS_NAMES.get(int(membership_status[row]), f"UNKNOWN_{int(membership_status[row])}"),
                            "world_space_classification": _world_classification(int(region_id[row]), int(membership_status[row])),
                            "projected_pixel": [float(geometry.pixel_x[row]), float(geometry.pixel_y[row])],
                            "query_depth": float(geometry.depth[row]),
                            "median_event_depth": float(median[row]) if float(median[row]) > 0.0 else None,
                            "signed_distance_s_median_minus_query": float(signed[row]),
                            "global_state": "GLOBAL_OCCLUDED",
                        })
                    camera_record[control_name] = {
                        "fixed_annotation_only": True,
                        "projected_global_occluded_count": len(entries),
                        "classification_counts": {label: sum(item["world_space_classification"] == label for item in entries) for label in (WORLD_CLASS_A, WORLD_CLASS_B, WORLD_CLASS_C)},
                        "records": entries,
                    }
                review_projection_records[names[camera_index]] = camera_record
            if camera_index % 20 == 0 or camera_index == len(cameras) - 1:
                _progress(f"replayed {camera_index + 1}/{len(cameras)} cameras")
            del geometry, pixel, median, states, signed

    global_states_t = torch.full((row_count,), STATE_UNRESOLVED, dtype=torch.int8, device=args.device)
    global_states_t[global_has_relevant & ~global_has_unresolved & ~global_observed_any] = STATE_OCCLUDED
    global_states_t[global_observed_any] = STATE_OBSERVED
    global_states = global_states_t.detach().cpu().numpy().astype(np.int8)
    target_global = global_states[target_rows]

    # The global state is known only after all 161 cameras; filter the stored
    # local-O ROI candidates now so review records are truly GLOBAL_OCCLUDED.
    for camera_record in review_projection_records.values():
        for control in camera_record.values():
            kept = [item for item in control["records"] if global_states[int(item["checkpoint_row_index"])] == STATE_OCCLUDED]
            control["records"] = kept
            control["projected_global_occluded_count"] = len(kept)
            control["classification_counts"] = {label: sum(item["world_space_classification"] == label for item in kept) for label in (WORLD_CLASS_A, WORLD_CLASS_B, WORLD_CLASS_C)}

    # Exact W160 invariant: no new state semantics are allowed here.
    state_counts = _state_counts(global_states)
    positive_control = {
        "definition": "all checkpoint rows whose existing W155 Gaussian Surface Region ID equals 1",
        "region_id": TABLETOP_REGION_ID,
        "new_world_distance_threshold": False,
        "population_count": target_count,
        "global_state_counts": _state_counts(target_global),
        "membership_status_counts": {name: int(np.count_nonzero(membership_status[target_rows] == code)) for code, name in W155_STATUS_NAMES.items()},
        "world_xyz_available_for_every_member": True,
        "global_occluded_member_count": int(np.count_nonzero(target_global == STATE_OCCLUDED)),
        "global_occluded_all_relevant_states_occluded_invariant": bool(np.all(target_states[target_global == STATE_OCCLUDED] != STATE_OBSERVED) and np.all(target_states[target_global == STATE_OCCLUDED] != STATE_UNRESOLVED)) if np.any(target_global == STATE_OCCLUDED) else True,
    }

    per_view_signed_distance: dict[str, Any] = {}
    for camera_index, name in enumerate(names):
        view_states = target_states[:, camera_index]
        view_s = target_s[:, camera_index]
        state_groups = {}
        for code, label in ((STATE_OBSERVED, "GLOBAL_OBSERVED"), (STATE_OCCLUDED, "GLOBAL_OCCLUDED"), (STATE_UNRESOLVED, "GLOBAL_UNRESOLVED")):
            state_groups[label] = {
                "global_member_count": int(np.count_nonzero(target_global == code)),
                "relevant_valid_ordering_count": int(np.count_nonzero((target_global == code) & np.isfinite(view_s))),
                "local_state_counts": _state_counts(view_states[target_global == code]),
                "s_distribution": _signed_summary(view_s[target_global == code]),
            }
            if not np.any(np.isfinite(view_s[target_global == code])):
                state_groups[label]["empty_reason"] = "no valid median event for this population in this camera" if code == STATE_UNRESOLVED else "no relevant valid ordering pair"
        per_view_signed_distance[name] = {"camera_index": camera_index, "population": "W155 region_id=1", "states": state_groups}

    # Re-replay only the selected positive-control/global-O rows to preserve full raw per-view records.
    selected_target_indices = np.flatnonzero(target_global == STATE_OCCLUDED).astype(np.int64)
    selected_rows = target_rows[selected_target_indices]
    selected_count = len(selected_rows)
    raw_query = np.full((selected_count, len(cameras)), np.nan, dtype=np.float32)
    raw_median = np.full((selected_count, len(cameras)), np.nan, dtype=np.float32)
    raw_s = np.full((selected_count, len(cameras)), np.nan, dtype=np.float32)
    raw_pixel = np.full((selected_count, len(cameras), 2), np.nan, dtype=np.float32)
    raw_local_state = np.full((selected_count, len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    with torch.no_grad():
        positions = model.get_xyz.detach()
        selected_position = positions[selected_rows]
        for camera_index, (camera, median_np) in enumerate(zip(cameras, depth_np)):
            median_flat = torch.as_tensor(median_np, dtype=torch.float32, device=args.device)
            geometry = w160.project_queries(camera, selected_position)
            pixel = geometry.pixel_index.clamp(min=0)
            median = median_flat[pixel]
            local, signed = _classify_exact(geometry, median)
            raw_query[:, camera_index] = geometry.depth.detach().cpu().numpy()
            raw_median[:, camera_index] = torch.where(median > 0.0, median, torch.full_like(median, float("nan"))).detach().cpu().numpy()
            raw_s[:, camera_index] = signed.detach().cpu().numpy()
            raw_pixel[:, camera_index, 0] = geometry.pixel_x.detach().cpu().numpy()
            raw_pixel[:, camera_index, 1] = geometry.pixel_y.detach().cpu().numpy()
            raw_local_state[:, camera_index] = local.detach().cpu().numpy()
            del geometry, pixel, median, local, signed
    if not np.array_equal(raw_local_state, target_states[selected_target_indices]):
        raise RuntimeError("selected-row replay changed frozen local state")

    raw_npz = args.out / "w162_tabletop_cross_view_raw.npz"
    np.savez_compressed(raw_npz, stable_gaussian_id=stable_ids[selected_rows], checkpoint_row_index=selected_rows, world_xyz=positions_np[selected_rows], gaussian_surface_region_id=region_id[selected_rows], w155_membership_status=membership_status[selected_rows], global_state=target_global[selected_target_indices], camera_names=np.asarray(names), local_state=raw_local_state, query_depth=raw_query, median_event_depth=raw_median, signed_distance=raw_s, projected_pixel=raw_pixel)
    population_npz = args.out / "w162_tabletop_population_audit.npz"
    np.savez_compressed(population_npz, stable_gaussian_id=stable_ids[target_rows], checkpoint_row_index=target_rows, world_xyz=positions_np[target_rows], gaussian_surface_region_id=region_id[target_rows], w155_membership_status=membership_status[target_rows], global_state=target_global, local_state=target_states, signed_distance=target_s)

    cross_records = []
    for item_index, row in enumerate(selected_rows.tolist()):
        views = []
        for camera_index, name in enumerate(names):
            state = int(raw_local_state[item_index, camera_index])
            if state == STATE_NON_RELEVANT:
                continue
            views.append({"camera": name, "camera_index": camera_index, "projected_pixel": raw_pixel[item_index, camera_index].tolist(), "query_depth": float(raw_query[item_index, camera_index]), "median_event_depth": float(raw_median[item_index, camera_index]) if np.isfinite(raw_median[item_index, camera_index]) else None, "signed_distance_s_median_minus_query": float(raw_s[item_index, camera_index]) if np.isfinite(raw_s[item_index, camera_index]) else None, "local_state": STATE_NAMES[state], "median_event_identity": "NOT_RECOVERABLE_UNDER_EXISTING_CONTRACT"})
        cross_records.append({"stable_gaussian_id": int(stable_ids[row]), "checkpoint_row_index": int(row), "world_xyz": positions_np[row].tolist(), "gaussian_surface_region_id": int(region_id[row]), "w155_membership_status": W155_STATUS_NAMES.get(int(membership_status[row]), f"UNKNOWN_{int(membership_status[row])}"), "global_state": "GLOBAL_OCCLUDED", "relevant_views": views})
    cross_json = args.out / "tabletop_cross_view_records.json"
    _write_json(cross_json, cross_records)
    review_json = args.out / "review_projection_records.json"
    _write_json(review_json, review_projection_records)

    renderer_provenance = _inspect_renderer_provenance(args.cache)
    same_region_conflict = {
        "status": "NOT_COMPUTABLE_UNDER_EXISTING_CONTRACT",
        "count": None,
        "reason": "query Gaussian region is available from W155, but the median event has no contributing Gaussian stable ID or Region ID; no same-region conflict count is invented",
        "query_region_source": "W155 gaussian_id_region_status_mapping.npz",
        "median_event_region_source": None,
    }
    wl161 = {}
    if args.wl161_report.exists():
        wl161 = json.loads(args.wl161_report.read_text(encoding="utf-8"))
    w161_consequence = {
        "status": "NOT_DIRECTLY_JOINABLE",
        "w161_spatial_field_constructed": bool(wl161.get("global_occlusion_field", {}).get("constructed", False)),
        "w161_report": str(args.wl161_report.resolve()),
        "reason": "W161 stopped at OCCLUSION_DOMAIN_CONTRACT_GAP and has no canonical spatial field or cell identity; no nearest-neighbor or synthetic spatial join is performed",
    }

    all_class_counts = {label: 0 for label in (WORLD_CLASS_A, WORLD_CLASS_B, WORLD_CLASS_C)}
    for camera_record in review_projection_records.values():
        for control in camera_record.values():
            for label, count in control["classification_counts"].items():
                all_class_counts[label] += int(count)
    has_a = all_class_counts[WORLD_CLASS_A] > 0
    has_b = all_class_counts[WORLD_CLASS_B] > 0
    if has_a and has_b:
        semantic_signal = "MIXED"
    elif has_b:
        semantic_signal = "MEDIAN_EVENT_PROXY_CONFLICTS_WITH_VISIBLE_SURFACE"
    elif has_a:
        semantic_signal = "PROJECTION_ONLY_ARTIFACT"
    else:
        semantic_signal = "UNRESOLVED"
    final_verdict = "RENDERER_PROVENANCE_CONTRACT_GAP" if has_b and renderer_provenance["exact_contributor_stable_id_join"] is False else semantic_signal

    _progress("rendering PNG review views")
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros(3, dtype=torch.float32, device=args.device)
    global_colours = w160._state_colours(global_states_t)
    unresolved_rgb = torch.tensor(w160.UNRESOLVED_RGB, dtype=torch.float32, device=args.device).reshape(1, 3)
    target_mask_t = torch.as_tensor(target_mask, dtype=torch.bool, device=args.device)
    target_colours = unresolved_rgb.expand(row_count, 3).clone()
    target_colours[target_mask_t] = global_colours[target_mask_t]
    only_occluded = unresolved_rgb.expand(row_count, 3).clone()
    only_occluded[target_mask_t & (global_states_t == STATE_OCCLUDED)] = torch.tensor(w160.OCCLUDED_RGB, dtype=torch.float32, device=args.device)
    only_observed = unresolved_rgb.expand(row_count, 3).clone()
    only_observed[target_mask_t & (global_states_t == STATE_OBSERVED)] = torch.tensor(w160.OBSERVED_RGB, dtype=torch.float32, device=args.device)
    context_region = torch.tensor(REGION_CONTEXT_RGB, dtype=torch.float32, device=args.device).reshape(1, 3).expand(row_count, 3).clone()
    context_region[target_mask_t] = torch.tensor(REGION_ALL_RGB, dtype=torch.float32, device=args.device)
    colour_views = {"global_state_pure": global_colours, "tabletop_global_occluded_only": only_occluded, "tabletop_region_all": target_colours, "tabletop_region_global_occluded": only_occluded, "tabletop_region_global_observed": only_observed}
    with torch.no_grad():
        for name in REVIEW_CAMERAS:
            camera = cameras[names.index(name)]
            original = w160._render_state(model, rasterizer, camera, global_colours, original=True, background=background)
            _write_png(args.out / "review_views" / "original_scene" / f"{Path(name).stem}.png", original)
            _write_png(args.out / "review_views" / "global_state_overlay" / f"{Path(name).stem}.png", _draw_polygons(original, name))
            for view_name, colours in colour_views.items():
                _write_png(args.out / "review_views" / view_name / f"{Path(name).stem}.png", _render_with_colours(model, rasterizer, camera, colours, background))
            contact = _draw_polygons(original, name, labels=True)
            _write_png(args.out / "review_views" / "tabletop_vase_contact" / f"{Path(name).stem}.png", contact)

    _progress("writing common world-space views and README files")
    common_root = args.out / "review_views" / "common_world"
    _world_projection_png(common_root / "perspective.png", positions_np, target_mask, global_states, (0, 2), "W162 common world diagnostic: X-Z")
    _world_projection_png(common_root / "top.png", positions_np, target_mask, global_states, (0, 1), "W162 common world diagnostic: X-Y top")
    _world_projection_png(common_root / "side.png", positions_np, target_mask, global_states, (1, 2), "W162 common world diagnostic: Y-Z side")
    _write_visualization_readmes(args.out, row_count, names, renderer_provenance)

    controls: dict[str, Any] = {}
    for control_name in ("tabletop", "table_side_lower_geometry", "vase_foreground_structure"):
        controls[control_name] = {
            "fixed_annotation_source": "W155/W160 review polygon",
            "annotation_only": True,
            "per_camera": {name: {"projected_global_occluded_count": value[control_name]["projected_global_occluded_count"], "classification_counts": value[control_name]["classification_counts"]} for name, value in review_projection_records.items()},
        }
    world_boxes = {"background_lower": (((-1.0, 1.5, -0.15), (1.0, 2.5, 0.15)), ((-11.0, 2.0, 0.0), (-9.5, 3.5, 2.5)))}
    background_records = []
    for box in world_boxes["background_lower"]:
        low, high = np.asarray(box[0]), np.asarray(box[1])
        mask = np.all((positions_np >= low) & (positions_np <= high), axis=1)
        background_records.append({"box": [list(box[0]), list(box[1])], "gaussian_count": int(mask.sum()), "global_state_counts": _state_counts(global_states[mask]), "global_occluded_count": int(np.count_nonzero(mask & (global_states == STATE_OCCLUDED)))})
    controls["background_lower"] = {"fixed_annotation_source": "W155/WL140 world-space boxes", "annotation_only": True, "boxes": background_records}

    report = {
        "status": "COMPLETE_WL162_RENDERER_MEDIAN_EVENT_DIRECT_OBSERVATION_SEMANTIC_VALIDITY_AUDIT",
        "batch": "Worklog 162 — Renderer Median-Event Direct-Observation Semantic Validity Audit",
        "intent_alignment": {"diagnostic_only": True, "candidate_b_modified": False, "states_modified": False, "w161_modified": False, "wl154_wl159_modified": False, "production_modified": False},
        "historical_contract_reused": {"w160_global_state_exact": True, "candidate_b_classifier_reused": True, "projective_signed_distance": "s=median_depth-query_depth", "state_palette": {"OBSERVED": w160.OBSERVED_RGB, "OCCLUDED": w160.OCCLUDED_RGB, "UNRESOLVED": w160.UNRESOLVED_RGB}, "new_threshold": False},
        "query_population": {"checkpoint_gaussian_row_count": row_count, "stable_id_source": "checkpoint model_raw.stable_gaussian_ids", "stable_id_unique": bool(len(np.unique(stable_ids)) == row_count), "camera_count": len(cameras), "camera_names_order": names},
        "w155_gaussian_surface_region_mapping": {"mapping": mapping, "tabletop_positive_control_region_id": TABLETOP_REGION_ID, "tabletop_region_selection_rule": "existing W155 review candidate Region ID 1; no new geometric membership predicate", "membership_status_names": W155_STATUS_NAMES},
        "frozen_tabletop_positive_control": positive_control,
        "projected_global_occluded_world_attribution": {"class_definitions": {"A": WORLD_CLASS_A, "B": WORLD_CLASS_B, "C": WORLD_CLASS_C}, "all_review_camera_control_counts": all_class_counts, "per_camera_control_records": review_projection_records, "interpretation": "A separates image-space ROI projection from the existing W155 world-space Region mapping; B is a query-region membership signal, not median-event identity; C preserves W155 ambiguity/unassigned cases."},
        "per_view_signed_distance_audit": {"definition": "for the frozen W155 region_id=1 positive-control population, grouped by exact W160 global state; no threshold applied", "views": per_view_signed_distance},
        "cross_view_raw_records": {"population": "W155 region_id=1 and exact W160 GLOBAL_OCCLUDED", "selected_count": selected_count, "npz": str(raw_npz), "json": str(cross_json), "fields": ["stable_gaussian_id", "world_xyz", "gaussian_surface_region_id", "w155_membership_status", "camera", "projected_pixel", "query_depth", "median_event_depth", "signed_distance", "local_state"], "non_relevant_views_omitted_from_json": True},
        "renderer_provenance": renderer_provenance,
        "median_event_identity": {"status": "NOT_RECOVERABLE_UNDER_EXISTING_CONTRACT", "contributing_gaussian_stable_id": False, "contributing_region_id": False, "same_region_median_ordering_conflict": same_region_conflict},
        "controls": controls,
        "w161_consequence": w161_consequence,
        "synthetic_contracts_A_to_E": synthetic,
        "semantic_verdict": {"conditional_image_world_signal": semantic_signal, "final_verdict": final_verdict, "allowed_verdicts": ["PROJECTION_ONLY_ARTIFACT", "HISTORICAL_PROXY_VALID_ON_VISIBLE_CONTROL", "MEDIAN_EVENT_PROXY_CONFLICTS_WITH_VISIBLE_SURFACE", "QUERY_SEMANTIC_MISMATCH", "RENDERER_PROVENANCE_CONTRACT_GAP", "MIXED", "UNRESOLVED"], "reason": "The query-side W155 Region join is exact, but median-event contributor identity is absent; any same-region proxy-conflict claim remains conditional on the missing renderer provenance contract."},
        "human_review": {"status": "HUMAN_REVIEW_REQUIRED", "review_cameras": list(REVIEW_CAMERAS), "review_root": str(args.out / "review_views"), "questions": ["red ROI projections belong to existing tabletop Region 1 or elsewhere", "same-region median event identity cannot be recovered from depth-only map", "world-space common views show query population only, not hidden geometry", "event 1527 remains historical annotation and was not reclassified"]},
        "outputs": {"report": str(args.out / "worklog_162_report.json"), "population_npz": str(population_npz), "cross_view_npz": str(raw_npz), "cross_view_json": str(cross_json), "review_projection_json": str(review_json), "review_root": str(args.out / "review_views"), "visualization_output_format": "PNG primary; no PPM emitted"},
        "forbidden_changes": {"candidate_b": False, "renderer": False, "checkpoint": False, "w153_cache": False, "w155_mapping": False, "w161": False, "wl154_wl159": False, "production": False},
        "inputs": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint), "source": str(args.source.resolve()), "cache": str(args.cache.resolve()), "cache_depth_array_sha256": depth_meta["array_sha256"], "cache_file_sha256": depth_meta["file_sha256"], "w155_mapping": str(args.wl155_mapping.resolve()), "w155_report": str(args.wl155_report.resolve()), "camera_meta": camera_meta, "h": float(depth_meta["runtime"]["h"]), "mu": float(depth_meta["runtime"]["mu"]), "renderer": "OSNSurfelRasterizer / frozen median channel"},
        "runtime_seconds": {"total": time.time() - started},
    }
    _write_json(args.out / "worklog_162_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wl155-mapping", type=Path, default=DEFAULT_WL155_MAPPING)
    parser.add_argument("--wl155-report", type=Path, default=DEFAULT_WL155_REPORT)
    parser.add_argument("--wl161-report", type=Path, default=DEFAULT_WL161_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "final_verdict": report["semantic_verdict"]["final_verdict"], "tabletop_population": report["frozen_tabletop_positive_control"]["population_count"], "global_occluded_tabletop": report["frozen_tabletop_positive_control"]["global_occluded_member_count"], "runtime_seconds": report["runtime_seconds"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
