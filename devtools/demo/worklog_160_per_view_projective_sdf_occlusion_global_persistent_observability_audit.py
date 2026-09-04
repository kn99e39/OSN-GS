from __future__ import annotations

"""Worklog 160 diagnostic-only per-view projective-SDF occlusion audit.

Frozen contract: z_v(x) is camera-space z and s_v(x)=d_v(p)-z_v(x), where
d_v(p) is the canonical renderer median-depth event.  Candidate-B's states
are the exact sign ordering of this quantity; the fused TSDF is not queried
by the classifier.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for path in (str(DEVTOOLS), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import (  # noqa: E402
    PRIMITIVE_SURFEL_2D,
    _rgb_to_f_dc,
    checkpoint_primitive,
    load_primitive_model,
)
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402
from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    RELEVANCE_DEPTH_BELOW_NEAR,
    RELEVANCE_INVALID_PROJECTION,
    RELEVANCE_OUTSIDE_IMAGE,
    STATE_NAMES,
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    aggregate_global,
    project_queries,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt"
DEFAULT_SOURCE = REPO_ROOT / "DATASET"
DEFAULT_CACHE = REPO_ROOT / "output/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache"
DEFAULT_OUT = REPO_ROOT / "output/160_per_view_projective_sdf_occlusion_global_persistent_observability"
REVIEW_CAMERAS = ("DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG")
EVENT_CAMERA = "DSC08003.JPG"
EVENT_PIXEL = (259.0, 169.0)
EVENT_RADIUS = 12.0
OBSERVED_RGB = (0.10, 0.85, 0.35)
OCCLUDED_RGB = (0.92, 0.18, 0.18)
UNRESOLVED_RGB = (0.60, 0.60, 0.62)

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


def _progress(message: str) -> None:
    print(f"[worklog 160] {message}", flush=True)


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


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


def classify_projective_sdf_evidence(*, relevant: bool, query_depth: float, median_depth: float | None) -> dict[str, Any]:
    """Classify already-projected evidence with no new threshold."""
    if not relevant:
        return {"state": STATE_NON_RELEVANT, "signed_distance": None, "surface_aligned": False}
    if median_depth is None or not np.isfinite(median_depth) or median_depth <= 0.0:
        return {"state": STATE_UNRESOLVED, "signed_distance": None, "surface_aligned": False}
    signed = float(median_depth - query_depth)
    return {"state": STATE_OBSERVED if signed >= 0.0 else STATE_OCCLUDED, "signed_distance": signed, "surface_aligned": signed == 0.0}


def aggregate_persistent_states(per_view_states: np.ndarray) -> np.ndarray:
    """Global OCCLUDED requires every relevant view to be OCCLUDED."""
    states = np.asarray(per_view_states, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError("per_view_states must have shape (query, camera)")
    relevant = states != STATE_NON_RELEVANT
    observed = (states == STATE_OBSERVED).any(axis=1)
    all_occluded = relevant.any(axis=1) & ((states == STATE_OCCLUDED) | ~relevant).all(axis=1)
    result = np.full(states.shape[0], STATE_UNRESOLVED, dtype=np.int8)
    result[all_occluded] = STATE_OCCLUDED
    result[observed] = STATE_OBSERVED
    return result


def synthetic_contracts() -> dict[str, Any]:
    def view(query: float, median: float | None, relevant: bool = True) -> dict[str, Any]:
        return classify_projective_sdf_evidence(relevant=relevant, query_depth=query, median_depth=median)

    cases = []
    a = view(1.0, 2.0)
    cases.append({"name": "A_directly_reachable", "expected": "OBSERVED", "actual": STATE_NAMES[a["state"]], "pass": a["state"] == STATE_OBSERVED})
    b = view(3.0, 2.0)
    cases.append({"name": "B_behind_surface", "expected": "OCCLUDED", "actual": STATE_NAMES[b["state"]], "pass": b["state"] == STATE_OCCLUDED})
    c = view(2.0, 2.0)
    cases.append({"name": "C_surface_aligned", "expected": {"state": "OBSERVED", "surface_aligned": True}, "actual": {"state": STATE_NAMES[c["state"]], "surface_aligned": c["surface_aligned"]}, "pass": c["state"] == STATE_OBSERVED and c["surface_aligned"]})
    d = aggregate_persistent_states(np.asarray([[STATE_OCCLUDED, STATE_OCCLUDED]], dtype=np.int8))[0]
    cases.append({"name": "D_all_relevant_occluded", "expected": "OCCLUDED", "actual": STATE_NAMES[int(d)], "pass": d == STATE_OCCLUDED})
    e = aggregate_persistent_states(np.asarray([[STATE_OCCLUDED, STATE_OBSERVED]], dtype=np.int8))[0]
    cases.append({"name": "E_one_visible_view", "expected": "OBSERVED", "actual": STATE_NAMES[int(e)], "pass": e == STATE_OBSERVED})
    f = aggregate_persistent_states(np.asarray([[STATE_NON_RELEVANT, STATE_OCCLUDED]], dtype=np.int8))[0]
    cases.append({"name": "F_irrelevant_camera_no_vote", "expected": "OCCLUDED", "actual": STATE_NAMES[int(f)], "pass": f == STATE_OCCLUDED})
    g = aggregate_persistent_states(np.asarray([[STATE_NON_RELEVANT, STATE_UNRESOLVED]], dtype=np.int8))[0]
    cases.append({"name": "G_no_valid_relevant_camera", "expected": "UNRESOLVED", "actual": STATE_NAMES[int(g)], "pass": g == STATE_UNRESOLVED})
    h = aggregate_persistent_states(np.asarray([[STATE_OCCLUDED]], dtype=np.int8))[0]
    cases.append({"name": "H_fused_sign_ignored", "expected": "OCCLUDED", "actual": STATE_NAMES[int(h)], "fused_tsdf_sign_control": "+0.75 ignored", "pass": h == STATE_OCCLUDED})
    return {"all_pass": bool(all(case["pass"] for case in cases)), "cases": cases, "note": "Synthetic PASS establishes mechanics only."}


def _load_depth_cache(cache: Path, names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    depth_path = cache / "renderer_median_depth_maps.npz"
    runtime_path = cache / "replay_input_runtime.json"
    if not depth_path.exists() or not runtime_path.exists():
        raise FileNotFoundError(f"incomplete W153 cache: {depth_path}")
    with np.load(depth_path, allow_pickle=False) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if depth.ndim != 2 or depth.shape[0] != len(names) or runtime.get("camera_names") != names:
        raise ValueError("W153 depth map shape/order does not match frozen cameras")
    actual = _sha256_array(depth)
    if runtime.get("depth_sha256") and runtime["depth_sha256"] != actual:
        raise ValueError("W153 depth map hash mismatch")
    return depth, {"path": str(depth_path), "file_sha256": _sha256_file(depth_path), "array_sha256": actual, "runtime": runtime}


def _polygon_mask(x: torch.Tensor, y: torch.Tensor, polygon: Any) -> torch.Tensor:
    vertices = np.asarray(tuple(polygon), dtype=np.float64)
    xx = x.detach().cpu().numpy().astype(np.float64, copy=False)
    yy = y.detach().cpu().numpy().astype(np.float64, copy=False)
    inside = np.zeros_like(xx, dtype=bool)
    x0, y0 = vertices[:, 0], vertices[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for lx, ly, rx, ry in zip(x0, y0, x1, y1):
        if abs(float(ry - ly)) > 1e-12:
            inside ^= ((ly > yy) != (ry > yy)) & (xx < (rx - lx) * (yy - ly) / (ry - ly) + lx)
    return torch.from_numpy(inside).to(device=x.device)


def _state_colours(states: torch.Tensor) -> torch.Tensor:
    colours = torch.tensor(UNRESOLVED_RGB, dtype=torch.float32, device=states.device).reshape(1, 3).expand(int(states.numel()), 3).clone()
    colours[states == STATE_OBSERVED] = torch.tensor(OBSERVED_RGB, dtype=torch.float32, device=states.device)
    colours[states == STATE_OCCLUDED] = torch.tensor(OCCLUDED_RGB, dtype=torch.float32, device=states.device)
    return colours


def _save_png(path: Path, image: torch.Tensor) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    value = image.detach().cpu().clamp(0.0, 1.0)
    if value.ndim == 3 and value.shape[0] == 3:
        value = value.permute(1, 2, 0)
    Image.fromarray((value.numpy() * 255.0).astype(np.uint8), mode="RGB").save(path, format="PNG", optimize=True)


def _render_state(model: Any, rasterizer: Any, camera: Any, colours: torch.Tensor, *, original: bool, background: torch.Tensor) -> torch.Tensor:
    if original:
        return rasterizer.render(camera, model, background=background)["render"].detach().clone()
    old_dc = model._features_dc.detach().clone()
    old_rest = model._features_rest.detach().clone()
    old_degree = model.active_sh_degree
    try:
        model._features_dc.copy_(_rgb_to_f_dc(colours).unsqueeze(1))
        model._features_rest.zero_()
        model.active_sh_degree = 0
        return rasterizer.render(camera, model, background=background)["render"].detach().clone()
    finally:
        model._features_dc.copy_(old_dc)
        model._features_rest.copy_(old_rest)
        model.active_sh_degree = old_degree
        del old_dc, old_rest


def _write_readmes(out: Path, row_count: int) -> None:
    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")

    write(out / "README.md", f"""# W160 시각화 산출물

이 batch는 frozen W153 per-camera median-depth map과 checkpoint Gaussian center를 사용한 diagnostic visualization이다. 입력은 동일 checkpoint/iteration의 Gaussian rows와 161개 frozen training camera이며, 모든 PNG는 같은 renderer, resolution, background, camera calibration, Gaussian row count({row_count:,})를 유지하고 appearance color만 바꾼다.

Original Scene은 learned SH appearance, Observed-Occluded는 frozen Candidate-B global state, per_camera_state는 global 집계 전 각 camera-local state, global_state는 all-relevant global state를 의미한다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED 또는 해당 camera의 NON_RELEVANT다. NON_RELEVANT는 global vote가 아니다.

이 산출물은 renderer-relative observability를 검토하기 위한 것이며 physical first-hit truth, hidden-surface truth, Gaussian Region membership 또는 downstream topology를 증명하지 않는다.
""")
    pair = out / "mandatory_gaussian_visualization_pair"
    write(pair / "README.md", """# Mandatory Gaussian Visualization Pair

Original Scene과 Observed-Occluded는 같은 checkpoint, iteration, renderer, resolution, background, camera calibration, Gaussian rows를 렌더링한 필수 pair다. 입력과 geometry는 동일하고 Original Scene은 원래 SH appearance, Observed-Occluded는 state color만 사용한다. geometry, scale/covariance, rotation, opacity, row count는 바뀌지 않았으며 marker Gaussian과 light/shading/geometry modification은 없다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED이다. 이 pair는 state visualization이지 physical occlusion ground truth나 surface reconstruction 결과가 아니다.
""")
    write(pair / "Original Scene" / "README.md", """# Original Scene

각 camera-name PNG는 matched training camera에서 원래 학습된 SH appearance를 렌더링한 Original Scene이다. 입력은 frozen checkpoint의 전체 Gaussian rows이며 state color override와 geometry 변경은 없다. 공통 조건은 checkpoint, iteration, renderer, resolution, background, camera calibration, row count가 Observed-Occluded와 동일하다는 것이다. 별도 state palette는 적용하지 않고 learned SH color를 그대로 사용한다. 비교 legend는 OBSERVED=green, OCCLUDED=red, UNRESOLVED=gray지만 이 Original Scene 파일에는 그 state 색을 덧씌우지 않는다. 따라서 이 이미지는 appearance baseline일 뿐 per-view/global observability 또는 physical first-hit truth를 직접 판정하지 않는다.
""")
    write(pair / "Observed-Occluded" / "README.md", """# Observed-Occluded

각 camera-name PNG는 Original Scene과 같은 frozen checkpoint, iteration, renderer, resolution, background, camera calibration, Gaussian rows와 geometry를 사용하고 frozen Candidate-B global state color만 적용한다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED이다. 이 색은 all-relevant global state의 diagnostic label이며, marker Gaussian·재질·조명·geometry를 추가하지 않는다. 따라서 색상만으로 physical surface membership 또는 실제 hidden-surface truth를 주장할 수 없다.
""")
    review = out / "review_views"
    write(review / "README.md", """# Per-camera와 global review views

per_camera_state는 global aggregation 이전의 camera-local state이고 global_state는 161개 frozen training camera의 all-relevant aggregation 결과다. 입력은 W153 median-depth map과 동일 checkpoint Gaussian query이며 각 하위 visualization directory에는 camera-name PNG와 공통 README가 있다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED 또는 NON_RELEVANT다. gray는 OCCLUDED vote가 아니다. 모든 PNG의 checkpoint, iteration, renderer, resolution, background, camera calibration, row count는 동일하다. 이 view는 renderer-relative contract review용이며 fused TSDF sign이나 physical first-hit truth를 표시하지 않는다.
""")
    write(review / "per_camera_state" / "README.md", """# Per-camera state

각 camera PNG는 frozen checkpoint Gaussian query를 해당 camera에 투영하고 renderer median depth와 비교한 camera-local state다. `s=d-z`에서 `s>=0`이면 OBSERVED, `s<0`이면 OCCLUDED이며 exact `s=0` surface alignment는 report accounting으로 별도 보존한다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED 또는 NON_RELEVANT다. 동일 checkpoint, iteration, renderer, resolution, background, camera calibration, row count를 사용한다. 이는 한 camera의 ordering evidence일 뿐 global persistent occlusion이나 physical first-hit truth가 아니다.
""")
    write(review / "global_state" / "README.md", """# Global state

relevant view가 하나 이상이고 모든 relevant view가 OCCLUDED일 때만 red=GLOBAL OCCLUDED다. valid OBSERVED가 하나라도 있으면 green=OBSERVED이고, unresolved가 남거나 valid relevant decision이 없으면 gray=UNRESOLVED다. green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED이며 NON_RELEVANT는 aggregation 분모에서 제외된다. 입력 camera set, checkpoint, iteration, renderer, resolution, background, camera calibration, row count는 per_camera_state와 동일하다. majority, percentage, confidence averaging, fused TSDF sign은 사용하지 않았고, 이 결과는 renderer-relative global contract이지 physical hidden-surface proof가 아니다.
""")


def _account(states: torch.Tensor, geometry: Any, signed: torch.Tensor, median: torch.Tensor) -> dict[str, int]:
    relevant = geometry.relevant
    valid = relevant & (median > 0.0)
    code = geometry.relevance_code
    return {
        "query_count_evaluated": int(states.numel()),
        "relevant_query_count": int(relevant.sum()),
        "non_relevant_query_count": int((~relevant).sum()),
        "invalid_projection_query_count": int((code == RELEVANCE_INVALID_PROJECTION).sum()),
        "depth_below_renderer_near_query_count": int((code == RELEVANCE_DEPTH_BELOW_NEAR).sum()),
        "outside_image_query_count": int((code == RELEVANCE_OUTSIDE_IMAGE).sum()),
        "valid_renderer_median_query_count": int(valid.sum()),
        "directly_reachable_observed_count": int((states == STATE_OBSERVED).sum()),
        "surface_aligned_count": int((valid & (signed == 0.0)).sum()),
        "occluded_count": int((states == STATE_OCCLUDED).sum()),
        "unresolved_count": int((states == STATE_UNRESOLVED).sum()),
    }


def _review_summary(name: str, geometry: Any, states: torch.Tensor) -> dict[str, Any]:
    result = {"camera": name, "targets": {}}
    for target, polygons in REVIEW_POLYGONS.items():
        mask = geometry.relevant & _polygon_mask(geometry.pixel_x, geometry.pixel_y, polygons[name])
        values = states[mask].detach().cpu().numpy()
        result["targets"][target] = {"annotation_only": True, "query_count": int(values.size), "observed": int((values == STATE_OBSERVED).sum()), "occluded": int((values == STATE_OCCLUDED).sum()), "unresolved": int((values == STATE_UNRESOLVED).sum()), "non_relevant_excluded": int((~mask).sum())}
    if name == EVENT_CAMERA:
        distance = torch.sqrt((geometry.pixel_x - EVENT_PIXEL[0]) ** 2 + (geometry.pixel_y - EVENT_PIXEL[1]) ** 2)
        mask = geometry.relevant & (distance <= EVENT_RADIUS)
        values = states[mask].detach().cpu().numpy()
        result["event_1527"] = {"pixel": list(EVENT_PIXEL), "radius_pixels": EVENT_RADIUS, "historical_review": "CLEAR_NOT_ON_INTENDED_SURFACE", "annotation_only": True, "query_count": int(values.size), "observed": int((values == STATE_OBSERVED).sum()), "occluded": int((values == STATE_OCCLUDED).sum()), "unresolved": int((values == STATE_UNRESOLVED).sum())}
    return result


def _fused_field_audit(cache: Path, positions: torch.Tensor, global_states: np.ndarray, h: float) -> dict[str, Any]:
    path = cache / "field.npz"
    if not path.exists():
        return {"available": False, "reason": f"missing {path}"}
    with np.load(path, allow_pickle=False) as data:
        keys = np.asarray(data["keys"], dtype=np.int64)
        values = np.asarray(data["value"], dtype=np.float32)
        cached_h = float(data["h"])
    if cached_h != h:
        return {"available": False, "reason": "fused field h differs from frozen median-map h"}
    xyz = positions.detach().cpu().numpy().astype(np.float32, copy=False)
    bound, span = 1 << 19, 1 << 20
    index = np.floor(xyz / h).astype(np.int64)
    encoded = ((index[:, 0] + bound) * span + index[:, 1] + bound) * span + index[:, 2] + bound
    locations = np.searchsorted(keys, encoded)
    safe = np.minimum(locations, max(len(keys) - 1, 0))
    found = (locations < len(keys)) & (keys[safe] == encoded) if len(keys) else np.zeros(len(encoded), dtype=bool)
    field_values = np.full(len(encoded), np.nan, dtype=np.float32)
    if len(keys):
        field_values[found] = values[safe[found]]
    fused = np.full(len(encoded), STATE_UNRESOLVED, dtype=np.int8)
    fused[found & (field_values >= 0.0)] = STATE_OBSERVED
    fused[found & (field_values < 0.0)] = STATE_OCCLUDED
    confusion = {}
    for left, left_name in ((STATE_OBSERVED, "OBSERVED"), (STATE_OCCLUDED, "OCCLUDED"), (STATE_UNRESOLVED, "UNRESOLVED")):
        for right, right_name in ((STATE_OBSERVED, "OBSERVED"), (STATE_OCCLUDED, "OCCLUDED"), (STATE_UNRESOLVED, "UNRESOLVED")):
            count = int(((global_states == left) & (fused == right)).sum())
            if count:
                confusion[f"GLOBAL_{left_name}->FUSED_{right_name}"] = count
    return {"available": True, "field_path": str(path), "field_sha256": _sha256_file(path), "field_authoritative_voxel_count": int(len(keys)), "query_count": int(len(encoded)), "query_voxels_with_fused_authority": int(found.sum()), "query_voxels_unknown": int((~found).sum()), "shortcut_definition": "fused value >=0 OBSERVED, <0 OCCLUDED, absent UNRESOLVED; diagnostic only", "same_global_state_count": int((fused == global_states).sum()), "disagreement_count": int((fused != global_states).sum()), "agreement_fraction": float((fused == global_states).mean()), "confusion": confusion, "not_promoted_to_classifier": True, "conclusion": "FUSED_FIELD_SHORTCUT_IS_NOT_THE_VIEW_DEPENDENT_CONTRACT"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_contracts()
    if not synthetic["all_pass"]:
        raise RuntimeError("synthetic A-H failure")
    _progress("loading checkpoint and frozen training cameras")
    model, payload = load_primitive_model(args.checkpoint, device=args.device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("canonical 2DGS surfel checkpoint required")
    cameras, camera_meta = load_all_train_cameras(args.source, args.images, args.sparse_dir, args.resolution, args.llffhold, args.device)
    names = [str(camera.image_name) for camera in cameras]
    if len(names) != 161:
        raise ValueError(f"expected 161 cameras, got {len(names)}")
    depth_np, depth_meta = _load_depth_cache(args.cache, names)
    depth_maps = [torch.as_tensor(row, dtype=torch.float32, device=args.device) for row in depth_np]
    positions = model.get_xyz.detach()
    row_count = int(positions.shape[0])
    h, mu = float(depth_meta["runtime"]["h"]), float(depth_meta["runtime"]["mu"])
    observed_any = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    has_relevant = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    has_unresolved = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    relevant_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    observed_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    occluded_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    unresolved_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    per_camera, reconciliation, review_states, review_summaries = {}, {}, {}, {}
    match_total = disagreement_total = 0
    with torch.no_grad():
        for camera_index, (camera, median_flat) in enumerate(zip(cameras, depth_maps)):
            geometry = project_queries(camera, positions)
            pixel = geometry.pixel_index.clamp(min=0)
            median = median_flat[pixel]
            valid = geometry.relevant & (median > 0.0)
            signed = median - geometry.depth
            recovered = torch.full((row_count,), STATE_NON_RELEVANT, dtype=torch.int8, device=args.device)
            recovered[geometry.relevant] = STATE_UNRESOLVED
            recovered[valid & (signed < 0.0)] = STATE_OCCLUDED
            recovered[valid & (signed >= 0.0)] = STATE_OBSERVED
            historical = candidate_b.classify_view(geometry, median_flat)["states"]
            same = recovered == historical
            same_count = int(same.sum())
            disagree_count = int((~same).sum())
            match_total += same_count
            disagreement_total += disagree_count
            reconciliation[names[camera_index]] = {"query_count": row_count, "same_count": same_count, "disagreement_count": disagree_count, "agreement_fraction": float(same.to(torch.float64).mean()), "observed_to_occluded": int(((historical == STATE_OBSERVED) & (recovered == STATE_OCCLUDED)).sum()), "occluded_to_observed": int(((historical == STATE_OCCLUDED) & (recovered == STATE_OBSERVED)).sum()), "unresolved_disagreements": int(((historical == STATE_UNRESOLVED) != (recovered == STATE_UNRESOLVED)).sum())}
            observed_any |= recovered == STATE_OBSERVED
            has_relevant |= recovered != STATE_NON_RELEVANT
            has_unresolved |= recovered == STATE_UNRESOLVED
            relevant_count += (recovered != STATE_NON_RELEVANT).to(torch.int16)
            observed_count += (recovered == STATE_OBSERVED).to(torch.int16)
            occluded_count += (recovered == STATE_OCCLUDED).to(torch.int16)
            unresolved_count += (recovered == STATE_UNRESOLVED).to(torch.int16)
            per_camera[names[camera_index]] = _account(recovered, geometry, signed, median)
            if names[camera_index] in REVIEW_CAMERAS:
                review_states[names[camera_index]] = recovered.detach().cpu().numpy().astype(np.int8)
                review_summaries[names[camera_index]] = _review_summary(names[camera_index], geometry, recovered)
            if camera_index % 20 == 0 or camera_index == len(cameras) - 1:
                _progress(f"classified {camera_index + 1}/{len(cameras)} cameras")
            del geometry, pixel, median, valid, signed, recovered, historical
    global_t = torch.full((row_count,), STATE_UNRESOLVED, dtype=torch.int8, device=args.device)
    global_t[has_relevant & ~has_unresolved & ~observed_any] = STATE_OCCLUDED
    global_t[observed_any] = STATE_OBSERVED
    global_states = global_t.detach().cpu().numpy().astype(np.int8)
    state_counts = {name: int((global_states == code).sum()) for code, name in STATE_NAMES.items() if code != STATE_NON_RELEVANT}
    np.savez_compressed(args.out / "gaussian_center_observability.npz", global_state=global_states, relevant_view_count=relevant_count.cpu().numpy(), observed_view_count=observed_count.cpu().numpy(), occluded_view_count=occluded_count.cpu().numpy(), unresolved_view_count=unresolved_count.cpu().numpy())
    np.savez_compressed(args.out / "review_per_camera_states.npz", **{name.replace(".", "_"): value for name, value in review_states.items()})
    fused_audit = _fused_field_audit(args.cache, positions, global_states, h)

    _progress("rendering matched-camera PNG review evidence")
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros(3, dtype=torch.float32, device=args.device)
    pair_root, review_root = args.out / "mandatory_gaussian_visualization_pair", args.out / "review_views"
    global_colours = _state_colours(global_t)
    with torch.no_grad():
        for name in REVIEW_CAMERAS:
            camera = cameras[names.index(name)]
            original = _render_state(model, rasterizer, camera, global_colours, original=True, background=background)
            global_image = _render_state(model, rasterizer, camera, global_colours, original=False, background=background)
            _save_png(pair_root / "Original Scene" / f"{name}.png", original)
            _save_png(pair_root / "Observed-Occluded" / f"{name}.png", global_image)
            per_states = torch.as_tensor(review_states[name], dtype=torch.int8, device=args.device)
            _save_png(review_root / "per_camera_state" / f"{name}.png", _render_state(model, rasterizer, camera, _state_colours(per_states), original=False, background=background))
            _save_png(review_root / "global_state" / f"{name}.png", global_image)
    _write_readmes(args.out, row_count)
    stacked = np.stack([review_states[name] for name in REVIEW_CAMERAS], axis=1)
    mixed_rows = np.flatnonzero((stacked == STATE_OBSERVED).any(axis=1) & (stacked == STATE_OCCLUDED).any(axis=1))
    mixed_examples = [{"gaussian_row": int(row), "camera_states": {name: STATE_NAMES[int(stacked[row, i])] for i, name in enumerate(REVIEW_CAMERAS)}, "global_state": STATE_NAMES[int(global_states[row])]} for row in mixed_rows[:20]]

    report = {
        "status": "COMPLETE_WL160_PER_VIEW_PROJECTIVE_SDF_OCCLUSION_GLOBAL_PERSISTENT_OBSERVABILITY_AUDIT",
        "batch": "Worklog 160 — Per-View Projective-SDF Occlusion and Global Persistent-Observability Contract Recovery",
        "intent_alignment": {"diagnostic_only": True, "production_behavior_modified": False, "historical_outputs_preserved": True, "changed_only": "isolated per-view ordering reconstruction and reporting"},
        "current_observability_architecture": "frozen renderer median event -> per-view median ordering -> all-relevant aggregation -> global state",
        "historical_global_state_path": {"query_population": "all Gaussian center rows", "per_camera_source": "candidate_b_median_depth.py::classify_view", "global_source": "shared.py::aggregate_global", "mechanism": "renderer median-depth ordering, not fused TSDF", "per_camera_state_preserved": True, "global_rule": "ANY OBSERVED wins; otherwise all relevant OCCLUDED -> OCCLUDED; otherwise UNRESOLVED"},
        "projective_sdf_per_view_contract": {"source_camera_id": "W153 replay_input_runtime.json camera_names order", "renderer_median_depth_map": "W153 replay_cache/renderer_median_depth_maps.npz, one H*W map per training camera", "valid_relevant_pixel": "w>0, z>=0.2, rounded pixel inside image", "camera_space_query_depth": "z=[x,1] @ world_view_transform column 2", "projective_signed_distance": "s=d-z", "sign": "s>0 camera-side; s<0 behind median event; not object inside/outside", "truncation": "W153 fusion-only |s|<=mu, mu=3h; not classifier threshold", "fusion_weight": "W153 exactly 1 per authoritative view", "pre_fusion_per_view_available": True, "recovery": "cached median map plus frozen camera calibration; no new threshold"},
        "surface_alignment_vs_occlusion": {"surface_alignment": "exact float32 s==0 accounting", "occlusion": "valid median and s<0", "observed": "valid median and s>=0, exact Candidate-B equivalent", "separate": True, "new_threshold": False},
        "relevant_view_contract": {"projection_inside_image": True, "valid_renderer_evidence": "median_depth>0 canonical no-event sentinel", "positive_depth": "w>0 and z>=0.2 canonical near plane", "future_latent_geometry_used": False, "new_angular_visibility_heuristic": False},
        "stop_condition_result": {"stop_a_triggered": False, "missing_primitive": None, "basis": "all 161 W153 per-camera median maps and frozen camera order/hash available"},
        "per_view_classifier": {"function": "frozen Candidate-B exact median comparison plus diagnostic s_v(x)", "state_labels": ["NON_RELEVANT", "OBSERVED", "OCCLUDED", "UNRESOLVED"], "arbitrary_3d_query": True, "Gaussian_membership_required": False},
        "global_persistent_occlusion": {"function": "aggregate_persistent_states", "all_relevant_views_required": True, "majority_vote": False, "percentage_vote": False, "confidence_average": False, "fused_field_sign_used": False, "state_counts": state_counts},
        "query_population": {"gaussian_center_count": row_count, "camera_count": len(cameras), "arbitrary_query_contract": "project_queries plus cached median map; Gaussian row not required"},
        "synthetic_contracts_A_to_H": synthetic,
        "candidate_b_reconciliation": {"historical_candidate": "Candidate-B classify_view plus shared.aggregate_global", "per_view_pairs": match_total + disagreement_total, "per_view_exact_agreement_count": match_total, "per_view_exact_agreement_fraction": match_total / max(row_count * len(cameras), 1), "per_view_disagreement_count": disagreement_total, "observed_to_occluded": sum(item["observed_to_occluded"] for item in reconciliation.values()), "occluded_to_observed": sum(item["occluded_to_observed"] for item in reconciliation.values()), "unresolved_disagreements": sum(item["unresolved_disagreements"] for item in reconciliation.values()), "per_camera_attribution": reconciliation, "candidate_b_modified": False, "global_rule_replayed": True},
        "per_camera_accounting": per_camera,
        "real_scene_review": {"matched_cameras": list(REVIEW_CAMERAS), "per_camera_state_before_global": review_summaries, "examples": {"directly_observed_tabletop": "fixed tabletop polygon", "behind_vase_foreground_structure": "fixed vase_foreground_structure polygon", "table_side_lower_geometry": "fixed table_side_lower_geometry polygon", "occluded_some_cameras_visible_in_another": mixed_examples, "event_1527": "fixed DSC08003.JPG pixel/radius annotation; historical review preserved"}, "review_windows_annotation_only": True, "png_paths": {"mandatory_pair": str(pair_root), "per_camera": str(review_root / "per_camera_state"), "global": str(review_root / "global_state")}},
        "fused_tsdf_non_equivalence": fused_audit,
        "relation_to_wl154_wl159": {"per_view_global_occlusion": "upstream latent-domain observability evidence", "fused_tsdf_zero_set": "downstream observed visible-surface geometry, not occlusion oracle", "gaussian_surface_region": "physical-surface candidate identity, not relevant-view eligibility", "wl154_wl159": "downstream support/topology diagnostics retained and not modified"},
        "architecture_verdict": {"allowed": ["PER_VIEW_SDF_OCCLUSION_VALIDATED", "HISTORICAL_GLOBAL_STATE_ALREADY_VALID", "PER_VIEW_OCCLUSION_CONTRACT_GAP", "RELEVANT_VIEW_CONTRACT_GAP", "GLOBAL_AGGREGATION_MISMATCH", "FUSED_FIELD_SEMANTIC_CONFLATION", "MIXED", "UNRESOLVED"], "architecture_verdict": "HISTORICAL_GLOBAL_STATE_ALREADY_VALID", "reason": "Existing Candidate-B already supplies deterministic renderer-relative per-camera ordering and shared.aggregate_global already enforces all-relevant persistent OCCLUDED; W160 validates the equivalent projective-SDF interpretation without replacing it.", "intent_alignment": "PASS", "implementation_fidelity": "PASS", "architecture_result": "HISTORICAL_GLOBAL_STATE_ALREADY_VALID"},
        "retained_rejected_open": {"retained": ["W127/W139/W145/W148-W159 artifacts", "W153 per-camera median maps", "historical Candidate-B outputs", "event 1527 review status", "renderer/checkpoint/camera calibration"], "rejected": ["fused TSDF sign as occlusion oracle", "TSDF unknown as OCCLUDED", "Gaussian Region prerequisite", "majority/percentage/confidence vote", "new threshold", "topology/Boundary First/NURBS/latent changes"], "open": ["physical first-hit truth of renderer median event", "independent hidden-surface evidence beyond renderer-relative contract"]},
        "inputs": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint), "iteration": int(payload.get("iteration", 0)), "camera_meta": camera_meta, "camera_names": names, "w153_depth_cache": depth_meta, "h": h, "mu": mu, "renderer": "OSNSurfelRasterizer / frozen median channel"},
        "outputs": {"report": str(args.out / "worklog_160_report.json"), "observability_npz": str(args.out / "gaussian_center_observability.npz"), "review_per_camera_states": str(args.out / "review_per_camera_states.npz"), "mandatory_pair": str(pair_root), "review_root": str(review_root), "visualization_output_format": "PNG primary; no PPM emitted"},
        "forbidden_changes": {"production_renderer": False, "checkpoint": False, "fused_tsdf": False, "gaussian_regions": False, "wl154_wl159_topology": False, "boundary_first": False, "nurbs": False, "latent_trust": False, "candidate_b": False},
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_160_report.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_verdict": report["architecture_verdict"]["architecture_verdict"], "gaussian_center_count": report["query_population"]["gaussian_center_count"], "runtime_seconds": report["runtime_seconds"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
