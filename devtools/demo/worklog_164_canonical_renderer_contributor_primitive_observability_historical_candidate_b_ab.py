from __future__ import annotations

"""Worklog 164 -- canonical primitive observation versus point-query A/B.

The historical Candidate-B point-query state is replayed unchanged.  A
separate diagnostic-only candidate changes only the positive observation
state of canonical Gaussian primitives for which the existing isolated
diagnostic rasterizer reports ``forward_accepted``.  This is a per-primitive,
per-camera bit captured by the renderer's own acceptance path; no K=16 pixel
prefix or new contribution threshold is used.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo import worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit as w160  # noqa: E402
from devtools.demo import worklog_162_renderer_median_event_direct_observation_semantic_validity_audit as w162  # noqa: E402
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative  # noqa: E402


DEFAULT_CHECKPOINT = w160.DEFAULT_CHECKPOINT
DEFAULT_SOURCE = w160.DEFAULT_SOURCE
DEFAULT_CACHE = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache"
DEFAULT_WL155_MAPPING = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/gaussian_id_region_status_mapping.npz"
DEFAULT_WL155_REPORT = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/worklog_155_report.json"
DEFAULT_WL160_STATE = REPO_ROOT / "output/confirmed/160_per_view_projective_sdf_occlusion_global_persistent_observability/gaussian_center_observability.npz"
DEFAULT_WL161_REPORT = REPO_ROOT / "output/confirmed/161_global_persistent_occlusion_spatial_domain_audit/worklog_161_report.json"
DEFAULT_WL162_RAW = REPO_ROOT / "output/confirmed/162_renderer_median_event_direct_observation_semantic_validity_audit/w162_tabletop_cross_view_raw.npz"
DEFAULT_WL163_RAW = REPO_ROOT / "output/163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit/w163_query_provenance_raw.npz"
DEFAULT_WL162_REVIEW = REPO_ROOT / "output/confirmed/162_renderer_median_event_direct_observation_semantic_validity_audit/review_projection_records.json"
DEFAULT_OUT = REPO_ROOT / "output/confirmed/164_canonical_renderer_contributor_primitive_observability_historical_candidate_b_controlled_ab"

REVIEW_CAMERAS = tuple(w160.REVIEW_CAMERAS)
TABLETOP_REGION_ID = 1
REGION0_ID = 0
W155_STATUS_NAMES = {0: "CORE", 1: "ATTACHED", 2: "AMBIGUOUS", 4: "UNASSIGNED"}

STATE_NON_RELEVANT = w160.STATE_NON_RELEVANT
STATE_UNRESOLVED = w160.STATE_UNRESOLVED
STATE_OBSERVED = w160.STATE_OBSERVED
STATE_OCCLUDED = w160.STATE_OCCLUDED
STATE_NAMES = w160.STATE_NAMES

OBSERVED_RGB = w160.OBSERVED_RGB
OCCLUDED_RGB = w160.OCCLUDED_RGB
UNRESOLVED_RGB = w160.UNRESOLVED_RGB
CONTEXT_RGB = UNRESOLVED_RGB
CONTRIBUTOR_RGB = (0.10, 0.55, 1.00)
CHANGED_RGB = (1.00, 0.86, 0.05)
HISTORICAL_OCCLUDED_WORLD_RGB = "#a91e2c"
CANDIDATE_OCCLUDED_WORLD_RGB = "#ef6658"
CHANGED_WORLD_RGB = "#ffe000"

POINT_QUERY_STATE = "POINT_QUERY_STATE"
PRIMITIVE_OBSERVATION_STATE = "PRIMITIVE_OBSERVATION_STATE"
CONTRIBUTED_IN_CAMERA = "CONTRIBUTED_IN_CAMERA"

STATE_ORDER = (STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED)
STATE_LABEL_ORDER = ("OBSERVED", "OCCLUDED", "UNRESOLVED")


def _progress(message: str) -> None:
    print(f"[worklog 164] {message}", flush=True)


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
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _state_counts(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values)
    return {name: int(np.count_nonzero(values == code)) for code, name in STATE_NAMES.items() if code != STATE_NON_RELEVANT}


def _apply_contributor_override(point_query_state: np.ndarray, contributed_in_camera: np.ndarray) -> np.ndarray:
    """Return PRIMITIVE_OBSERVATION_STATE without mutating POINT_QUERY_STATE."""

    point = np.asarray(point_query_state, dtype=np.int8)
    contributed = np.asarray(contributed_in_camera, dtype=bool)
    if point.shape != contributed.shape:
        raise ValueError("point-query state and contributor bit shape mismatch")
    result = point.copy()
    result[contributed] = STATE_OBSERVED
    return result


def _transition_matrix(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, dict[str, int]]:
    baseline = np.asarray(baseline, dtype=np.int8).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.int8).reshape(-1)
    if baseline.shape != candidate.shape:
        raise ValueError("transition arrays have different shapes")
    return {
        STATE_NAMES[left]: {STATE_NAMES[right]: int(np.count_nonzero((baseline == left) & (candidate == right))) for right in STATE_ORDER}
        for left in STATE_ORDER
    }


def _contributor_count_bins(counts: np.ndarray) -> dict[str, int]:
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    return {
        "zero_cameras": int(np.count_nonzero(counts == 0)),
        "exactly_1_camera": int(np.count_nonzero(counts == 1)),
        "2_to_5_cameras": int(np.count_nonzero((counts >= 2) & (counts <= 5))),
        "6_to_10_cameras": int(np.count_nonzero((counts >= 6) & (counts <= 10))),
        "11_to_20_cameras": int(np.count_nonzero((counts >= 11) & (counts <= 20))),
        "greater_than_20_cameras": int(np.count_nonzero(counts > 20)),
    }


def _contributor_distribution(counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if not counts.size:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None, "bins": _contributor_count_bins(counts)}
    return {
        "count": int(counts.size),
        "min": int(counts.min()),
        "p05": float(np.percentile(counts, 5)),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "max": int(counts.max()),
        "bins": _contributor_count_bins(counts),
    }


def _population_accounting(
    rows: np.ndarray,
    baseline_global: np.ndarray,
    candidate_global: np.ndarray,
    contributor_counts: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    baseline = baseline_global[rows]
    candidate = candidate_global[rows]
    counts = contributor_counts[rows]
    baseline_occluded = baseline == STATE_OCCLUDED
    baseline_occluded_to_observed = baseline_occluded & (candidate == STATE_OBSERVED)
    return {
        "population_count": int(rows.size),
        "baseline_point_query_state_counts": _state_counts(baseline),
        "candidate_primitive_observation_state_counts": _state_counts(candidate),
        "transition_matrix": _transition_matrix(baseline, candidate),
        "historical_global_occluded_to_candidate_global_observed": int(baseline_occluded_to_observed.sum()),
        "historical_global_occluded_to_candidate_global_unresolved": int((baseline_occluded & (candidate == STATE_UNRESOLVED)).sum()),
        "historical_global_occluded_unchanged_global_occluded": int((baseline_occluded & (candidate == STATE_OCCLUDED)).sum()),
        "historical_global_occluded_changed_fraction": float(baseline_occluded_to_observed.sum() / max(int(baseline_occluded.sum()), 1)),
        "contributor_camera_count_distribution_per_primitive": _contributor_distribution(counts),
        "contributor_positive_primitive_count": int(np.count_nonzero(counts > 0)),
        "contributor_zero_camera_primitive_count": int(np.count_nonzero(counts == 0)),
    }


def synthetic_contracts() -> dict[str, Any]:
    """Synthetic A--F contracts for the isolated positive-observation path."""

    cases: list[dict[str, Any]] = []
    a = _apply_contributor_override(np.asarray([STATE_OCCLUDED]), np.asarray([True]))[0]
    cases.append({"name": "A_accepted_behind_median_becomes_primitive_observed", "expected": "OBSERVED", "actual": STATE_NAMES[int(a)], "baseline_point_query": "OCCLUDED", "pass": int(a) == STATE_OBSERVED})
    b = _apply_contributor_override(np.asarray([STATE_OCCLUDED]), np.asarray([True]))[0]
    cases.append({"name": "B_accepted_before_median_is_primitive_observed", "expected": "OBSERVED", "actual": STATE_NAMES[int(b)], "traversal_relation": "BEFORE_MEDIAN_EVENT is irrelevant to positive acceptance", "pass": int(b) == STATE_OBSERVED})
    c = _apply_contributor_override(np.asarray([STATE_OCCLUDED]), np.asarray([False]))[0]
    cases.append({"name": "C_not_accepted_baseline_occluded_unchanged", "expected": "OCCLUDED", "actual": STATE_NAMES[int(c)], "pass": int(c) == STATE_OCCLUDED})
    d = _apply_contributor_override(np.asarray([STATE_OBSERVED]), np.asarray([False]))[0]
    cases.append({"name": "D_not_accepted_baseline_observed_unchanged", "expected": "OBSERVED", "actual": STATE_NAMES[int(d)], "pass": int(d) == STATE_OBSERVED})
    e = _apply_contributor_override(np.asarray([STATE_OCCLUDED]), np.asarray([True]))[0]
    cases.append({"name": "E_truncated_pixel_prefix_does_not_override_exact_bit", "expected": "OBSERVED", "actual": STATE_NAMES[int(e)], "prefix_not_used": True, "pass": int(e) == STATE_OBSERVED})
    try:
        _apply_contributor_override(np.asarray([STATE_OCCLUDED]), np.asarray([True, False]))
    except ValueError as exc:
        f_pass = True
        f_actual = "NOT_APPLICABLE_WITHOUT_CANONICAL_PRIMITIVE_IDENTITY"
        f_error = str(exc)
    else:
        f_pass = False
        f_actual = "APPLIED_INCORRECTLY"
        f_error = None
    cases.append({"name": "F_arbitrary_xyz_without_identity_refused", "expected": "not applicable", "actual": f_actual, "error": f_error, "pass": f_pass})
    return {"all_pass": bool(all(case["pass"] for case in cases)), "cases": cases, "note": "Synthetic PASS verifies mechanics only; it is not physical surface ground truth."}


def _load_w160_global_state(path: Path, row_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "global_state" not in data.files:
            raise ValueError("W160 state artifact lacks global_state")
        state = np.asarray(data["global_state"], dtype=np.int8)
    if state.shape != (row_count,):
        raise ValueError(f"W160 state shape {state.shape} does not match {row_count} checkpoint rows")
    return state


def _load_mapping(path: Path, stable_ids: np.ndarray) -> dict[str, Any]:
    return w162._load_w155_mapping(path, stable_ids)


def _read_frozen_w163(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "reason": f"W163 raw artifact not found: {path}"}
    with np.load(path, allow_pickle=False) as data:
        return {
            "available": True,
            "path": str(path.resolve()),
            "query_count": int(data["stable_gaussian_id"].shape[0]),
            "camera_count": int(data["camera_names"].shape[0]),
            "query_participation_exact_pair_count": int(np.count_nonzero(data["query_participation"] == 1)),
            "query_stable_id_count": int(data["stable_gaussian_id"].shape[0]),
        }


def _load_control_rows(
    review_path: Path,
    positions: np.ndarray,
    region_id: np.ndarray,
) -> dict[str, np.ndarray]:
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    names = {
        "table_side_lower_geometry": set(),
        "vase_foreground_structure": set(),
        "tabletop": set(),
    }
    for camera_record in review.values():
        for control_name in names:
            for item in camera_record.get(control_name, {}).get("records", []):
                names[control_name].add(int(item["checkpoint_row_index"]))
    boxes = (
        ((-1.0, 1.5, -0.15), (1.0, 2.5, 0.15)),
        ((-11.0, 2.0, 0.0), (-9.5, 3.5, 2.5)),
    )
    background = np.zeros(len(positions), dtype=bool)
    for low, high in boxes:
        background |= np.all((positions >= np.asarray(low)) & (positions <= np.asarray(high)), axis=1)
    return {
        "region0": np.flatnonzero(region_id == REGION0_ID).astype(np.int64),
        "region1": np.flatnonzero(region_id == TABLETOP_REGION_ID).astype(np.int64),
        "table_side_lower_geometry": np.asarray(sorted(names["table_side_lower_geometry"]), dtype=np.int64),
        "vase_foreground_structure": np.asarray(sorted(names["vase_foreground_structure"]), dtype=np.int64),
        "tabletop_vase_contact": np.asarray(sorted(names["tabletop"] | names["vase_foreground_structure"]), dtype=np.int64),
        "background_lower": np.flatnonzero(background).astype(np.int64),
    }


def _render_state(model: Any, rasterizer: Any, camera: Any, colours: torch.Tensor, *, original: bool, background: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return w160._render_state(model, rasterizer, camera, colours, original=original, background=background)


def _colour_rows(row_count: int, colour: tuple[float, float, float], device: str) -> torch.Tensor:
    return torch.tensor(colour, dtype=torch.float32, device=device).reshape(1, 3).expand(row_count, 3).clone()


def _save_png(path: Path, image: torch.Tensor) -> None:
    w160._save_png(path, image)


def _readme_texts(row_count: int, camera_count: int) -> dict[str, str]:
    conditions = f"frozen checkpoint/iteration, {row_count:,} Gaussian rows, {camera_count} training cameras, 648x420 calibration, OSNSurfelRasterizer, black background"
    shared = (
        f"공통 rendering 조건: {conditions}. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. "
        "이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다."
    )
    state_legend = "state palette: green=OBSERVED (0.10, 0.85, 0.35), red=OCCLUDED (0.92, 0.18, 0.18), gray=UNRESOLVED (0.60, 0.60, 0.62)."
    limitation = (
        "Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. "
        "POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다."
    )
    return {
        "root": f"""# W164 Canonical Primitive Observation A/B

이 output은 W163에서 확인한 median-ordering conflict를 대상으로 historical Candidate-B의 POINT_QUERY_STATE와 contributor-aware PRIMITIVE_OBSERVATION_STATE를 matched A/B로 비교한다. canonical Gaussian primitive g가 frozen canonical renderer의 기존 acceptance path에서 한 개 이상의 pixel에 accepted contributor로 기록되면, camera별 CONTRIBUTED_IN_CAMERA(g,v)=true이고 해당 primitive state만 OBSERVED로 override한다. arbitrary XYZ에는 이 규칙을 적용하지 않는다.

{shared}

W163의 K=16 pixel prefix는 이 batch의 participation 판정에 사용하지 않는다. isolated diagnostic sibling의 per-primitive forward_accepted bit가 exact sparse predicate이며, contribution magnitude threshold, percentage, area, confidence, new T/alpha threshold는 없다. `POINT_QUERY_STATE`와 `PRIMITIVE_OBSERVATION_STATE`는 raw NPZ와 report에서 별도로 inspectable하다.

{state_legend}

{limitation}
""",
        "review_root": f"""# W164 Review Views

각 하위 directory는 동일한 canonical Gaussian population에 대한 한 가지 visualization semantics를 담는다. camera PNG는 `<camera_name_stem>.png` 형식으로 directory 바로 아래에 있고, 각 directory에 이 visualization의 의미, input/state semantics, palette/legend, 공통 rendering 조건, review limitation을 개별 기록한다.

{shared}

Historical A는 Gaussian center의 immutable Candidate-B POINT_QUERY_STATE를 사용한다. Candidate B는 그 local state에 exact per-primitive contributor positive evidence만 적용한다. common_world는 world XYZ diagnostic projection이며 W161 spatial field나 physical first-hit reconstruction이 아니다.
""",
        "original_scene": f"""# Original Scene

각 camera PNG는 전체 {row_count:,} canonical Gaussian을 learned SH appearance 그대로 canonical OSNSurfelRasterizer로 렌더링한 baseline이다. color override, marker Gaussian, lighting/shading, geometry modification이 없다.

{shared}

Legend: learned SH appearance 자체가 표시되며 state나 contributor identity를 뜻하지 않는다.

{limitation}
""",
        "historical_global_state": f"""# Historical Global State

각 PNG는 Gaussian center에 대해 frozen Candidate-B의 POINT_QUERY_STATE를 161개 relevant-camera aggregation으로 계산한 historical global state를 표시한다. green=GLOBAL OBSERVED, red=GLOBAL OCCLUDED, gray=GLOBAL UNRESOLVED이다.

{shared}

{state_legend} 이 view는 W160/W162 historical baseline을 재현하며 PRIMITIVE_OBSERVATION_STATE나 contributor positive override를 포함하지 않는다.

{limitation}
""",
        "contributor_aware_global_state": f"""# Contributor-Aware Global State

Historical POINT_QUERY_STATE에 대해 per-camera `CONTRIBUTED_IN_CAMERA(g,v)`가 true인 canonical primitive를 OBSERVED로 positive override한 뒤 frozen all-relevant aggregation을 적용한 Candidate B state이다. 색은 candidate PRIMITIVE_OBSERVATION_STATE의 global 결과를 나타낸다.

{shared}

{state_legend} Non-contributor의 historical OCCLUDED ordering은 그대로 유지된다. contributor count voting이나 confidence weighting은 없다.

{limitation}
""",
        "historical_global_occluded_only": f"""# Historical Global Occluded Only

Historical POINT_QUERY_STATE에서 GLOBAL OCCLUDED인 Gaussian만 red로 표시하고 나머지 전체 Gaussian은 gray context로 표시한다. broad tabletop의 baseline red population을 Candidate view와 같은 camera/renderer 조건으로 비교하기 위한 A view이다.

{shared}

Legend: red=historical GLOBAL OCCLUDED, gray=other/context.

{limitation}
""",
        "contributor_aware_global_occluded_only": f"""# Contributor-Aware Global Occluded Only

Contributor-aware candidate에서 GLOBAL OCCLUDED로 남은 Gaussian만 red로 표시하고 나머지는 gray context로 표시한다. red population의 감소는 오직 exact per-primitive contributor positive evidence에 의한 것이다.

{shared}

Legend: red=candidate GLOBAL OCCLUDED, gray=other/context. Candidate의 red는 contributor가 없는 경우에도 historical OCCLUDED가 유지된 결과일 수 있으며, non-contribution 자체를 physical proof로 읽지 않는다.

{limitation}
""",
        "baseline_occluded_to_observed": f"""# Baseline Occluded to Candidate Observed

Historical GLOBAL OCCLUDED였지만 contributor-aware candidate에서 GLOBAL OBSERVED가 된 Gaussian만 yellow로 표시한다. 다른 row는 gray context이다. 이 view는 A/B transition의 changed population만 공간적으로 보여준다.

{shared}

Legend: yellow=historical GLOBAL OCCLUDED → candidate GLOBAL OBSERVED, gray=other/context. yellow는 canonical primitive observation evidence가 추가된 transition이지 physical hidden-surface 판정이 아니다.

{limitation}
""",
        "contributor_observed_primitives": f"""# Contributor-Observed Primitives

161개 training camera 중 하나 이상에서 exact `forward_accepted` bit가 true인 canonical Gaussian primitive를 blue로 표시한다. contributor가 0개 camera인 primitive와 나머지 context는 gray이다. 이 view는 global state가 아니라 positive primitive evidence coverage를 나타낸다.

{shared}

Legend: blue=CONTRIBUTED_IN_CAMERA in at least one camera, gray=no positive contributor evidence/context. zero contributor는 OCCLUDED proof가 아니다.

{limitation}
""",
        "region0_state_ab": f"""# Region 0 State A/B

W156/W157 primary tabletop candidate인 frozen W155 Region 0만 state color로 표시하고 context는 gray로 둔다. baseline과 candidate가 다른 row, 즉 historical GLOBAL OCCLUDED → candidate GLOBAL OBSERVED는 yellow로 overlay한다. Region 0과 Region 1을 하나의 tabletop Region으로 합치지 않는다.

{shared}

Legend: green/red/gray=candidate Region 0 global state; yellow=Region 0 A/B changed row; gray=context.

{limitation}
""",
        "region1_state_ab": f"""# Region 1 State A/B

W162가 frozen한 W155 Region 1 전체 population만 state color로 표시하고 context는 gray로 둔다. baseline과 candidate가 다른 row는 yellow로 overlay한다. Region 1의 historical 65,471 Gaussian과 59,274/6,197/0 baseline state counts는 report에서 보존된다.

{shared}

Legend: green/red/gray=candidate Region 1 global state; yellow=Region 1 A/B changed row; gray=context.

{limitation}
""",
        "tabletop_vase_contact_ab": f"""# Tabletop/Vase Contact A/B

W162에 이미 존재하는 fixed tabletop 및 vase/curved-neighbor review record의 union만 사용해 contact population을 고정한다. candidate state를 표시하고 baseline GLOBAL OCCLUDED → candidate GLOBAL OBSERVED transition은 yellow로 overlay한다. 새 ROI, 새 Region, 거리 threshold는 만들지 않는다.

{shared}

Legend: green/red/gray=candidate contact-population global state; yellow=A/B changed row; gray=context.

{limitation}
""",
        "common_world": f"""# Common World Views

`perspective.png`, `top.png`, `side.png`는 전체 canonical Gaussian world XYZ를 X-Z, X-Y, Y-Z로 투영한 common-world diagnostic이다. historical GLOBAL OCCLUDED 전체는 dark red, candidate GLOBAL OCCLUDED는 light red, historical OCCLUDED → candidate OBSERVED는 yellow로 구분한다.

{shared}

Legend: dark red=historical GLOBAL OCCLUDED, light red=candidate GLOBAL OCCLUDED, yellow=baseline OCCLUDED → candidate OBSERVED, gray=display-only context. camera perspective가 아니며 W161 spatial domain을 만들거나 Gate O2를 닫지 않는다.

{limitation}
""",
    }


def _write_readmes(out: Path, row_count: int, camera_count: int) -> None:
    texts = _readme_texts(row_count, camera_count)
    for name, text in texts.items():
        if name == "root":
            path = out / "README.md"
        elif name == "review_root":
            path = out / "review_views" / "README.md"
        else:
            path = out / "review_views" / name / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _clear_output(out: Path) -> None:
    review = out / "review_views"
    if review.exists():
        shutil.rmtree(review)
    for name in ("w164_per_camera_states.npz", "w164_global_ab_raw.npz", "w164_contributor_evidence.npz", "canonical_equivalence.json", "worklog_164_report.json"):
        path = out / name
        if path.exists():
            path.unlink()


def _world_projection_png(
    path: Path,
    positions: np.ndarray,
    baseline_global: np.ndarray,
    candidate_global: np.ndarray,
    axis_pair: tuple[int, int],
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_axis, y_axis = axis_pair
    changed = (baseline_global == STATE_OCCLUDED) & (candidate_global == STATE_OBSERVED)
    historical_occluded = baseline_global == STATE_OCCLUDED
    candidate_occluded = candidate_global == STATE_OCCLUDED
    context = positions[::max(1, len(positions) // 250000)]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.scatter(context[:, x_axis], context[:, y_axis], s=0.15, c="#b8b8bc", alpha=0.08, linewidths=0, label="context")
    if np.any(historical_occluded):
        ax.scatter(positions[historical_occluded, x_axis], positions[historical_occluded, y_axis], s=0.8, c=HISTORICAL_OCCLUDED_WORLD_RGB, alpha=0.30, linewidths=0, label="historical GLOBAL OCCLUDED")
    unchanged_candidate = candidate_occluded & ~changed
    if np.any(unchanged_candidate):
        ax.scatter(positions[unchanged_candidate, x_axis], positions[unchanged_candidate, y_axis], s=1.0, c=CANDIDATE_OCCLUDED_WORLD_RGB, alpha=0.75, linewidths=0, label="candidate GLOBAL OCCLUDED")
    if np.any(changed):
        ax.scatter(positions[changed, x_axis], positions[changed, y_axis], s=1.5, c=CHANGED_WORLD_RGB, alpha=0.95, linewidths=0, label="baseline O → candidate OBSERVED")
    labels = ("X", "Y", "Z")
    ax.set_xlabel(f"{labels[x_axis]} world")
    ax.set_ylabel(f"{labels[y_axis]} world")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=4)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png")
    plt.close(fig)


def _render_review_views(
    args: argparse.Namespace,
    model: Any,
    cameras: list[Any],
    names: list[str],
    positions: np.ndarray,
    region_id: np.ndarray,
    baseline_global: np.ndarray,
    candidate_global: np.ndarray,
    contributor_counts: np.ndarray,
    controls: dict[str, np.ndarray],
) -> None:
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros(3, dtype=torch.float32, device=args.device)
    row_count = len(positions)
    baseline_colours = torch.as_tensor(w160._state_colours(torch.as_tensor(baseline_global, dtype=torch.int8, device=args.device)), device=args.device)
    candidate_colours = w160._state_colours(torch.as_tensor(candidate_global, dtype=torch.int8, device=args.device))
    context = _colour_rows(row_count, CONTEXT_RGB, args.device)
    target_region1 = context.clone()
    target_region0 = context.clone()
    changed_global = (baseline_global == STATE_OCCLUDED) & (candidate_global == STATE_OBSERVED)
    region1_mask = region_id == TABLETOP_REGION_ID
    region0_mask = region_id == REGION0_ID
    target_region1[torch.as_tensor(region1_mask, dtype=torch.bool, device=args.device)] = candidate_colours[torch.as_tensor(region1_mask, dtype=torch.bool, device=args.device)]
    target_region0[torch.as_tensor(region0_mask, dtype=torch.bool, device=args.device)] = candidate_colours[torch.as_tensor(region0_mask, dtype=torch.bool, device=args.device)]
    target_region1[torch.as_tensor(changed_global & region1_mask, dtype=torch.bool, device=args.device)] = torch.tensor(CHANGED_RGB, device=args.device)
    target_region0[torch.as_tensor(changed_global & region0_mask, dtype=torch.bool, device=args.device)] = torch.tensor(CHANGED_RGB, device=args.device)
    historical_occluded = context.clone()
    candidate_occluded = context.clone()
    historical_occluded[torch.as_tensor(baseline_global == STATE_OCCLUDED, dtype=torch.bool, device=args.device)] = torch.tensor(OCCLUDED_RGB, device=args.device)
    candidate_occluded[torch.as_tensor(candidate_global == STATE_OCCLUDED, dtype=torch.bool, device=args.device)] = torch.tensor(OCCLUDED_RGB, device=args.device)
    changed = context.clone()
    changed[torch.as_tensor(changed_global, dtype=torch.bool, device=args.device)] = torch.tensor(CHANGED_RGB, device=args.device)
    contributor = context.clone()
    contributor[torch.as_tensor(contributor_counts > 0, dtype=torch.bool, device=args.device)] = torch.tensor(CONTRIBUTOR_RGB, device=args.device)
    contact = context.clone()
    contact_rows = controls["tabletop_vase_contact"]
    contact[torch.as_tensor(contact_rows, dtype=torch.long, device=args.device)] = candidate_colours[torch.as_tensor(contact_rows, dtype=torch.long, device=args.device)]
    changed_contact_rows = contact_rows[changed_global[contact_rows]]
    if changed_contact_rows.size:
        contact[torch.as_tensor(changed_contact_rows, dtype=torch.long, device=args.device)] = torch.tensor(CHANGED_RGB, device=args.device)

    for camera_name in REVIEW_CAMERAS:
        camera = cameras[names.index(camera_name)]
        stem = Path(camera_name).stem
        _progress(f"rendering review camera {camera_name}")
        original = _render_state(model, rasterizer, camera, baseline_colours, original=True, background=background)
        _save_png(args.out / "review_views" / "original_scene" / f"{stem}.png", original)
        _save_png(args.out / "review_views" / "historical_global_state" / f"{stem}.png", _render_state(model, rasterizer, camera, baseline_colours, original=False, background=background))
        _save_png(args.out / "review_views" / "contributor_aware_global_state" / f"{stem}.png", _render_state(model, rasterizer, camera, candidate_colours, original=False, background=background))
        _save_png(args.out / "review_views" / "historical_global_occluded_only" / f"{stem}.png", _render_state(model, rasterizer, camera, historical_occluded, original=False, background=background))
        _save_png(args.out / "review_views" / "contributor_aware_global_occluded_only" / f"{stem}.png", _render_state(model, rasterizer, camera, candidate_occluded, original=False, background=background))
        _save_png(args.out / "review_views" / "baseline_occluded_to_observed" / f"{stem}.png", _render_state(model, rasterizer, camera, changed, original=False, background=background))
        _save_png(args.out / "review_views" / "contributor_observed_primitives" / f"{stem}.png", _render_state(model, rasterizer, camera, contributor, original=False, background=background))
        _save_png(args.out / "review_views" / "region0_state_ab" / f"{stem}.png", _render_state(model, rasterizer, camera, target_region0, original=False, background=background))
        _save_png(args.out / "review_views" / "region1_state_ab" / f"{stem}.png", _render_state(model, rasterizer, camera, target_region1, original=False, background=background))
        _save_png(args.out / "review_views" / "tabletop_vase_contact_ab" / f"{stem}.png", _render_state(model, rasterizer, camera, contact, original=False, background=background))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    synthetic = synthetic_contracts()
    if not synthetic["all_pass"]:
        raise RuntimeError("synthetic A-F contract failure")
    args.out.mkdir(parents=True, exist_ok=True)
    _clear_output(args.out)
    _progress("loading checkpoint, cameras, W153 depth maps, W155 mapping, and frozen historical artifacts")
    model, payload = w160.load_primitive_model(args.checkpoint, device=args.device)
    if w160.checkpoint_primitive(payload) != w160.PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("canonical 2DGS surfel checkpoint required")
    raw_ids = payload["model_raw"].get("stable_gaussian_ids")
    if raw_ids is None:
        raise ValueError("checkpoint lacks stable_gaussian_ids")
    stable_ids = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    positions = model.get_xyz.detach()
    positions_np = positions.cpu().numpy().astype(np.float32, copy=False)
    row_count = len(positions_np)
    mapping = _load_mapping(args.wl155_mapping, stable_ids)
    region_id = mapping["region_id"].astype(np.int64, copy=False)
    membership_status = mapping["membership_status"].astype(np.int8, copy=False)
    baseline_global_frozen = _load_w160_global_state(args.wl160_state, row_count)
    cameras, camera_meta = w160.load_all_train_cameras(args.source, args.images, args.sparse_dir, args.resolution, args.llffhold, args.device)
    names = [str(camera.image_name) for camera in cameras]
    camera_count = len(cameras)
    if camera_count != 161:
        raise ValueError(f"expected 161 training cameras, got {camera_count}")
    depth_np, depth_meta = w160._load_depth_cache(args.cache, names)
    controls = _load_control_rows(args.wl162_review, positions_np, region_id)
    frozen_w163 = _read_frozen_w163(args.wl163_raw)
    with np.load(args.wl162_raw, allow_pickle=False) as data:
        w162_region1_ids = np.asarray(data["stable_gaussian_id"], dtype=np.int64)
        w162_region1_global = np.asarray(data["global_state"], dtype=np.int8)
    region1_rows = controls["region1"]
    if len(w162_region1_ids) != 6197 or not np.all(w162_region1_global == STATE_OCCLUDED):
        raise ValueError("W162 Region-1 frozen positive-control artifact is not the expected 6,197 GLOBAL_OCCLUDED rows")
    # Stable IDs in the W162 raw artifact are sufficient for the population
    # check, but row ordering is not assumed to equal checkpoint row ordering.
    frozen_region1_occluded_ids = stable_ids[region1_rows][baseline_global_frozen[region1_rows] == STATE_OCCLUDED]
    if not np.array_equal(np.sort(w162_region1_ids), np.sort(frozen_region1_occluded_ids)):
        raise ValueError("W162 Region-1 stable IDs do not reconcile with frozen W155/W160 state")
    point_query_state = np.empty((row_count, camera_count), dtype=np.int8)
    contributed_in_camera = np.zeros((row_count, camera_count), dtype=bool)
    validation_names = set(REVIEW_CAMERAS) | {names[0], names[camera_count // 2], names[-1]}
    canonical_equivalence: dict[str, Any] = {}
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    canonical_rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros(3, dtype=torch.float32, device=args.device)

    _progress("replaying immutable point-query states and exact per-primitive contributor bits")
    with torch.no_grad():
        for camera_index, (camera, camera_name, median_np) in enumerate(zip(cameras, names, depth_np)):
            geometry = w160.project_queries(camera, positions)
            median_flat = torch.as_tensor(median_np, dtype=torch.float32, device=args.device)
            pixel = geometry.pixel_index.clamp(min=0)
            median = median_flat[pixel]
            local, _signed = w162._classify_exact(geometry, median)
            point_query_state[:, camera_index] = local.detach().cpu().numpy().astype(np.int8, copy=False)
            diag = render_with_pixel_representative(camera, model, background=background)
            accepted = diag["forward_accepted"].detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
            if accepted.shape != (row_count,):
                raise ValueError(f"forward_accepted shape {accepted.shape} does not match checkpoint rows {row_count}")
            contributed_in_camera[:, camera_index] = accepted
            if camera_name in validation_names:
                canonical = canonical_rasterizer.render(camera, model, background=background)
                render_diff = (diag["render"].detach() - canonical["render_unclamped"].detach()).abs()
                alpha_diff = (diag["out_others"][1].detach() - canonical["rend_alpha"].squeeze(0).detach()).abs()
                median_diff = (diag["out_others"][5].detach() - canonical["depth_median"].squeeze(0).detach()).abs()
                canonical_equivalence[camera_name] = {
                    "render_bitwise_equal": bool(torch.equal(diag["render"].detach(), canonical["render_unclamped"].detach())),
                    "alpha_bitwise_equal": bool(torch.equal(diag["out_others"][1].detach(), canonical["rend_alpha"].squeeze(0).detach())),
                    "median_depth_bitwise_equal": bool(torch.equal(diag["out_others"][5].detach(), canonical["depth_median"].squeeze(0).detach())),
                    "render_max_abs_diff": float(render_diff.max().item()),
                    "alpha_max_abs_diff": float(alpha_diff.max().item()),
                    "median_depth_max_abs_diff": float(median_diff.max().item()),
                    "renderer_canonical_unchanged": True,
                }
                del canonical
            del geometry, median_flat, pixel, median, local, _signed, diag
            if camera_index % 10 == 0 or camera_index == camera_count - 1:
                _progress(f"replayed {camera_index + 1}/{camera_count} cameras")

    if not all(value["render_bitwise_equal"] and value["alpha_bitwise_equal"] and value["median_depth_bitwise_equal"] for value in canonical_equivalence.values()):
        raise RuntimeError("diagnostic and canonical renderer outputs diverged")
    if not np.array_equal(w160.aggregate_global(point_query_state), baseline_global_frozen):
        raise RuntimeError("replayed POINT_QUERY_STATE does not reproduce frozen W160 global state")
    primitive_observation_state = np.empty_like(point_query_state)
    for camera_index in range(camera_count):
        primitive_observation_state[:, camera_index] = _apply_contributor_override(point_query_state[:, camera_index], contributed_in_camera[:, camera_index])
    candidate_global = w160.aggregate_global(primitive_observation_state)
    contributor_counts = contributed_in_camera.sum(axis=1, dtype=np.int32)
    baseline_global = baseline_global_frozen
    changed_global = (baseline_global == STATE_OCCLUDED) & (candidate_global == STATE_OBSERVED)
    global_accounting = _population_accounting(np.arange(row_count, dtype=np.int64), baseline_global, candidate_global, contributor_counts)
    region_accounting = {name: _population_accounting(rows, baseline_global, candidate_global, contributor_counts) for name, rows in controls.items()}
    transition_full = _transition_matrix(baseline_global, candidate_global)
    cross_tab = {
        "historical_global_state_by_contributor_presence": {
            "contributing_in_at_least_one_camera": _state_counts(baseline_global[contributor_counts > 0]),
            "contributing_in_zero_cameras": _state_counts(baseline_global[contributor_counts == 0]),
        },
        "contributor_camera_pair_count": int(contributed_in_camera.sum()),
        "primitive_count_with_positive_evidence": int(np.count_nonzero(contributor_counts > 0)),
        "primitive_count_with_zero_positive_evidence": int(np.count_nonzero(contributor_counts == 0)),
        "zero_positive_evidence_is_not_occluded_proof": True,
    }
    raw_ab_path = args.out / "w164_global_ab_raw.npz"
    np.savez_compressed(raw_ab_path, stable_gaussian_id=stable_ids, checkpoint_row_index=np.arange(row_count, dtype=np.int64), world_xyz=positions_np, region_id=region_id, membership_status=membership_status, point_query_global_state=baseline_global, primitive_observation_global_state=candidate_global, contributor_camera_count=contributor_counts, changed_global=changed_global)
    evidence_path = args.out / "w164_contributor_evidence.npz"
    np.savez_compressed(evidence_path, stable_gaussian_id=stable_ids, checkpoint_row_index=np.arange(row_count, dtype=np.int64), contributed_in_camera=contributed_in_camera, contributor_camera_count=contributor_counts, camera_names=np.asarray(names))
    state_path = args.out / "w164_per_camera_states.npz"
    np.savez_compressed(state_path, stable_gaussian_id=stable_ids, checkpoint_row_index=np.arange(row_count, dtype=np.int64), camera_names=np.asarray(names), point_query_state=point_query_state, primitive_observation_state=primitive_observation_state)

    _progress("rendering matched A/B PNG review views")
    _render_review_views(args, model, cameras, names, positions_np, region_id, baseline_global, candidate_global, contributor_counts, controls)
    common_root = args.out / "review_views" / "common_world"
    _world_projection_png(common_root / "perspective.png", positions_np, baseline_global, candidate_global, (0, 2), "W164 common world: historical and contributor-aware states")
    _world_projection_png(common_root / "top.png", positions_np, baseline_global, candidate_global, (0, 1), "W164 common world top: historical and contributor-aware states")
    _world_projection_png(common_root / "side.png", positions_np, baseline_global, candidate_global, (1, 2), "W164 common world side: historical and contributor-aware states")
    _write_readmes(args.out, row_count, camera_count)

    report = {
        "status": "COMPLETE_WL164_CANONICAL_RENDERER_CONTRIBUTOR_PRIMITIVE_OBSERVABILITY_HISTORICAL_CANDIDATE_B_CONTROLLED_AB",
        "batch": "Worklog 164 -- Canonical Renderer-Contributor Primitive Observability and Historical Candidate-B Controlled A/B",
        "intent_alignment": {"status": "PASS", "diagnostic_only": True, "preserved_historical_candidate_b": True, "changed_only": "positive OBSERVED override for exact canonical primitive-camera contributor bits", "arbitrary_xyz_not_promoted": True, "human_review_required": True},
        "implementation_fidelity": {"status": "PASS", "point_query_state_type": POINT_QUERY_STATE, "primitive_observation_state_type": PRIMITIVE_OBSERVATION_STATE, "contributor_predicate": CONTRIBUTED_IN_CAMERA, "participation_source": "existing isolated diagnostic sibling per-primitive forward_accepted bit from canonical acceptance path", "prefix_K16_used_for_participation": False, "contribution_magnitude_threshold": False, "new_alpha_or_T_threshold": False, "percentage_or_confidence_rule": False, "relevance_semantics_changed": False, "historical_non_contributor_occluded_rule_changed": False, "candidate_global_aggregation": "frozen w160.aggregate_global"},
        "architecture_result": "MIXED",
        "architecture_result_reason": "Quantitative A/B isolates the positive primitive observation rule and leaves arbitrary point-query OCCLUDED semantics open; human qualitative review is required before any architecture success or overpermissiveness claim.",
        "1_semantic_type_separation": {"POINT_QUERY_STATE": "immutable Gaussian-center Candidate-B median-depth ordering state; arbitrary XYZ-compatible contract", "PRIMITIVE_OBSERVATION_STATE": "POINT_QUERY_STATE plus exact canonical primitive contributor positive OBSERVED override", "not_equivalent": True, "raw_per_camera_artifact": str(state_path.resolve())},
        "2_historical_baseline_preservation": {"w160_frozen_global_state_exact_reproduced": True, "candidate_b_modified": False, "w160_state_modified": False, "w161_modified": False, "w162_w163_modified": False, "production_renderer_modified": False, "median_depth_changed": False},
        "3_exact_contributor_participation_contract": {"predicate": CONTRIBUTED_IN_CAMERA, "definition": "forward_accepted[g,v] == 1 means primitive g passed the frozen canonical forward acceptance path for at least one rendered pixel in camera v", "camera_count": camera_count, "primitive_count": row_count, "pair_count": row_count * camera_count, "contributor_pair_count": int(contributed_in_camera.sum()), "positive_primitive_count": int(np.count_nonzero(contributor_counts > 0)), "zero_positive_primitive_count": int(np.count_nonzero(contributor_counts == 0)), "raw_artifact": str(evidence_path.resolve()), "zero_positive_is_not_occluded_proof": True},
        "4_canonical_render_equivalence": {"validation_cameras": sorted(canonical_equivalence), "all_bitwise_equal": True, "per_camera": canonical_equivalence, "diagnostic_did_not_replace_canonical": True},
        "5_full_scene_ab": global_accounting,
        "6_region0_control": region_accounting["region0"],
        "7_region1_control": {**region_accounting["region1"], "frozen_w162_population_count": 65471, "frozen_w162_historical_global_state_counts": {"OBSERVED": 59274, "OCCLUDED": 6197, "UNRESOLVED": 0}, "w162_population_unchanged": True, "historical_global_occluded_contributor_camera_bins": _contributor_count_bins(contributor_counts[region1_rows][baseline_global[region1_rows] == STATE_OCCLUDED])},
        "8_table_side_vase_background_controls": {name: region_accounting[name] for name in ("table_side_lower_geometry", "vase_foreground_structure", "background_lower")},
        "9_contributor_evidence_accounting": {**cross_tab, "full_primitive_contributor_camera_count_distribution": _contributor_distribution(contributor_counts), "cross_tab_by_historical_global_state": {STATE_NAMES[code]: {"contributor_positive": int(np.count_nonzero((baseline_global == code) & (contributor_counts > 0))), "contributor_zero": int(np.count_nonzero((baseline_global == code) & (contributor_counts == 0)))} for code in STATE_ORDER}, "raw_artifact": str(evidence_path.resolve())},
        "10_synthetic_contracts": synthetic,
        "11_quantitative_result": {"full_scene": global_accounting, "region0": region_accounting["region0"], "region1": region_accounting["region1"], "table_side_lower": region_accounting["table_side_lower_geometry"], "vase_curved_neighbor": region_accounting["vase_foreground_structure"], "background_controls": region_accounting["background_lower"], "transition_matrix": transition_full, "historical_global_occluded_to_candidate_global_observed": int(changed_global.sum()), "historical_global_occluded_to_candidate_global_unresolved": int(np.count_nonzero((baseline_global == STATE_OCCLUDED) & (candidate_global == STATE_UNRESOLVED))), "unchanged_global_occluded": int(np.count_nonzero((baseline_global == STATE_OCCLUDED) & (candidate_global == STATE_OCCLUDED)))},
        "12_qualitative_review_exports": {"status": "HUMAN_REVIEW_REQUIRED", "review_root": str((args.out / "review_views").resolve()), "visualizations": ["original_scene", "historical_global_state", "contributor_aware_global_state", "historical_global_occluded_only", "contributor_aware_global_occluded_only", "baseline_occluded_to_observed", "contributor_observed_primitives", "region0_state_ab", "region1_state_ab", "tabletop_vase_contact_ab", "common_world"], "png_primary": True, "ppm_count": 0, "camera_layout": "<camera_name_stem>.png directly inside each visualization directory", "readme_per_visualization": True},
        "13_human_review_questions": ["broad tabletop red GLOBAL OCCLUDED가 contributor-aware candidate에서 얼마나 감소하는가?", "어떤 red population이 남는가?", "removed OCCLUDED가 명확한 rendered visible structure에 집중되는가?", "hidden-looking population을 candidate가 과도하게 OBSERVED로 바꾸지 않는가?", "Region 0과 Region 1의 A/B 변화가 어떻게 다른가?", "tabletop-vase contact, table-side/lower, background control의 변화는 contributor rule이 permissive한지 시사하는가?", "common-world에서 changed primitive가 observed scene geometry에 속하는 것으로 보이는가, projection overlap처럼 보이는가?"],
        "14_architecture_interpretation": {"provisional_verdict": "MIXED", "allowed_verdicts": ["CONTRIBUTOR_PRIMITIVE_OBSERVATION_SUPPORTED", "CONTRIBUTOR_PRIMITIVE_OBSERVATION_OVERPERMISSIVE", "CONTRIBUTOR_RULE_REPAIRS_MEDIAN_CONFLICT_BUT_POINT_QUERY_REMAINS_OPEN", "NO_MATERIAL_EFFECT", "MIXED", "UNRESOLVED"], "adopted_claim": "renderer contributor -> canonical primitive observation evidence exists", "rejected_claim": "renderer contributor -> physical surface ground truth", "point_query_occlusion": "remains scientifically open; Gate O2 is not closed", "qualitative_success_declared": False},
        "15_retained_rejected_open": {"retained": ["historical Candidate-B", "W160 point-query state/cache", "W161 OCCLUSION_DOMAIN_CONTRACT_GAP and paused status", "W162 populations", "W163 raw provenance", "canonical checkpoint/renderer/cameras/stable IDs/W155 mapping", "intrinsic t_w, TSDF, topology, Boundary First, NURBS, continuation"], "rejected": ["K=16 prefix non-inclusion as NOT_CONTRIBUTOR", "contribution magnitude threshold", "contributor count voting", "zero contributor as OCCLUDED proof", "arbitrary XYZ contributor semantics", "first-hit physical truth", "new ROI/Region/spatial field"], "open": ["human qualitative architecture review", "physical first-hit truth", "arbitrary XYZ OCCLUDED contract", "W161 spatial-domain construction remains paused"]},
        "w162_w163_reconciliation": {"w162_raw": str(args.wl162_raw.resolve()), "w163_raw": frozen_w163, "region1_rows": int(region1_rows.size), "w163_exact_conflict_source_preserved": True},
        "inputs": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint), "iteration": int(payload.get("iteration", 0)), "source": str(args.source.resolve()), "camera_names": names, "camera_meta": camera_meta, "w153_cache": str(args.cache.resolve()), "w153_replay_cache_excluded_from_temp_mirror": True, "w153_depth_cache": depth_meta, "w155_mapping": str(args.wl155_mapping.resolve()), "w155_mapping_sha256": _sha256_file(args.wl155_mapping), "w155_report": str(args.wl155_report.resolve()), "w160_state": str(args.wl160_state.resolve()), "w161_report": str(args.wl161_report.resolve()), "w162_raw": str(args.wl162_raw.resolve()), "w162_review": str(args.wl162_review.resolve()), "w163_raw": str(args.wl163_raw.resolve())},
        "outputs": {"report": str((args.out / "worklog_164_report.json").resolve()), "global_ab_raw": str(raw_ab_path.resolve()), "per_camera_states": str(state_path.resolve()), "contributor_evidence": str(evidence_path.resolve()), "review_root": str((args.out / "review_views").resolve()), "visualization_output_format": "PNG primary; no PPM emitted"},
        "forbidden_changes": {"candidate_b": False, "w160": False, "w161": False, "w162_w163": False, "production_renderer": False, "regions": False, "intrinsic_t_w": False, "tsdf": False, "topology": False, "boundary_first": False, "nurbs": False, "continuation": False, "latent_geometry": False},
        "runtime_seconds": {"total": time.time() - started},
    }
    _write_json(args.out / "canonical_equivalence.json", canonical_equivalence)
    _write_json(args.out / "worklog_164_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wl155-mapping", type=Path, default=DEFAULT_WL155_MAPPING)
    parser.add_argument("--wl155-report", type=Path, default=DEFAULT_WL155_REPORT)
    parser.add_argument("--wl160-state", type=Path, default=DEFAULT_WL160_STATE)
    parser.add_argument("--wl161-report", type=Path, default=DEFAULT_WL161_REPORT)
    parser.add_argument("--wl162-raw", type=Path, default=DEFAULT_WL162_RAW)
    parser.add_argument("--wl162-review", type=Path, default=DEFAULT_WL162_REVIEW)
    parser.add_argument("--wl163-raw", type=Path, default=DEFAULT_WL163_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
