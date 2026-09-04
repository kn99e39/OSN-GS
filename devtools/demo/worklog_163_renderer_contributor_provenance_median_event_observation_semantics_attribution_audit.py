from __future__ import annotations

"""Worklog 163 -- renderer contributor provenance and median-event audit.

This is a diagnostic-only continuation of W162.  The canonical renderer,
Candidate-B, W160 state arrays, W161, the W155 region mapping, and all
production paths remain unchanged.  The existing diagnostic sibling renderer
is replayed only to expose the renderer's own median representative and its
bounded accepted-contributor prefix.  A missing contributor from a truncated
prefix is deliberately recorded as unavailable, never as a negative result.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo import worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit as w160  # noqa: E402
from devtools.demo import worklog_162_renderer_median_event_direct_observation_semantic_validity_audit as w162  # noqa: E402
from osn_gs.render.torch_surfel_representative_diagnostics import (  # noqa: E402
    render_with_pixel_representative,
)


DEFAULT_CHECKPOINT = w160.DEFAULT_CHECKPOINT
DEFAULT_SOURCE = w160.DEFAULT_SOURCE
DEFAULT_CACHE = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache"
DEFAULT_WL155_MAPPING = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/gaussian_id_region_status_mapping.npz"
DEFAULT_WL155_REPORT = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/worklog_155_report.json"
DEFAULT_WL161_REPORT = REPO_ROOT / "output/confirmed/161_global_persistent_occlusion_spatial_domain_audit/worklog_161_report.json"
DEFAULT_WL162_RAW = REPO_ROOT / "output/confirmed/162_renderer_median_event_direct_observation_semantic_validity_audit/w162_tabletop_cross_view_raw.npz"
DEFAULT_WL162_REVIEW = REPO_ROOT / "output/confirmed/162_renderer_median_event_direct_observation_semantic_validity_audit/review_projection_records.json"
DEFAULT_WL160_STATE = REPO_ROOT / "output/confirmed/160_per_view_projective_sdf_occlusion_global_persistent_observability/gaussian_center_observability.npz"
DEFAULT_OUT = REPO_ROOT / "output/163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit"

REVIEW_CAMERAS = tuple(w160.REVIEW_CAMERAS)
TABLETOP_REGION_ID = 1
SLOT_CAPACITY = 16
W155_STATUS_NAMES = {0: "CORE", 1: "ATTACHED", 2: "AMBIGUOUS", 4: "UNASSIGNED"}

QUERY_IS_EXACT_CONTRIBUTOR = "QUERY_IS_EXACT_CONTRIBUTOR"
QUERY_NOT_CONTRIBUTOR = "QUERY_NOT_CONTRIBUTOR"
QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE = "QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE"

MEDIAN_SAME_GAUSSIAN = "MEDIAN_SAME_GAUSSIAN"
MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN = "MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN"
MEDIAN_DIFFERENT_REGION = "MEDIAN_DIFFERENT_REGION"
MEDIAN_IDENTITY_UNAVAILABLE = "MEDIAN_IDENTITY_UNAVAILABLE"

BEFORE_MEDIAN_EVENT = "BEFORE_MEDIAN_EVENT"
AT_MEDIAN_EVENT = "AT_MEDIAN_EVENT"
AFTER_MEDIAN_EVENT = "AFTER_MEDIAN_EVENT"
ORDER_UNAVAILABLE = "ORDER_UNAVAILABLE"

PROVENANCE_EXACT_RGB = (0.10, 0.55, 1.00)
PROVENANCE_UNAVAILABLE_RGB = (1.00, 0.55, 0.05)
MEDIAN_SAME_RGB = (1.00, 0.82, 0.05)
MEDIAN_DIFFERENT_RGB = (0.95, 0.15, 0.75)
CONFLICT_RGB = (1.00, 0.92, 0.05)
CONTEXT_RGB = w160.UNRESOLVED_RGB


def _progress(message: str) -> None:
    print(f"[worklog 163] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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
    return {name: int(np.count_nonzero(values == code)) for code, name in w160.STATE_NAMES.items()}


def _query_participation(
    query_row: int,
    contrib_ids: Iterable[int],
    contrib_count: int,
    slot_capacity: int = SLOT_CAPACITY,
) -> str:
    """Classify identity using the captured prefix and the uncapped count.

    Inclusion in the prefix is exact.  Non-inclusion is exact only when the
    uncapped count does not exceed the prefix capacity; otherwise the result
    is explicitly unavailable.
    """

    ids = np.asarray(list(contrib_ids), dtype=np.int64).reshape(-1)
    if np.any(ids == int(query_row)):
        return QUERY_IS_EXACT_CONTRIBUTOR
    if int(contrib_count) <= int(slot_capacity):
        return QUERY_NOT_CONTRIBUTOR
    return QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE


def _median_identity(
    query_row: int,
    query_region: int,
    median_row: int,
    median_region: int,
) -> str:
    if int(median_row) < 0 or int(median_region) < 0:
        return MEDIAN_IDENTITY_UNAVAILABLE
    if int(median_row) == int(query_row):
        return MEDIAN_SAME_GAUSSIAN
    if int(median_region) == int(query_region):
        return MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN
    return MEDIAN_DIFFERENT_REGION


def _order_relation(query_row: int, median_row: int, query_slot: int, post_median: int) -> str:
    """Use the diagnostic kernel's exact traversal relation; no epsilon."""

    if int(query_slot) < 0 or int(median_row) < 0:
        return ORDER_UNAVAILABLE
    if int(query_row) == int(median_row):
        return AT_MEDIAN_EVENT
    if int(post_median) == 1:
        return AFTER_MEDIAN_EVENT
    if int(post_median) == 0:
        return BEFORE_MEDIAN_EVENT
    return ORDER_UNAVAILABLE


def synthetic_contracts() -> dict[str, Any]:
    """Synthetic A--F contracts for exact category mechanics and limitations."""

    cases: list[dict[str, Any]] = []
    a = _median_identity(2, 1, 2, 1)
    cases.append({"name": "A_same_gaussian_at_median", "expected": MEDIAN_SAME_GAUSSIAN, "actual": a, "pass": a == MEDIAN_SAME_GAUSSIAN})
    b = _median_identity(3, 1, 2, 1)
    cases.append({"name": "B_same_region_different_gaussian", "expected": MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN, "actual": b, "pass": b == MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN})
    c_status = _query_participation(3, [2, 3], 2)
    c_order = _order_relation(3, 2, 1, 1)
    cases.append({"name": "C_exact_query_after_median", "expected": [QUERY_IS_EXACT_CONTRIBUTOR, AFTER_MEDIAN_EVENT], "actual": [c_status, c_order], "pass": c_status == QUERY_IS_EXACT_CONTRIBUTOR and c_order == AFTER_MEDIAN_EVENT})
    d = _query_participation(3, [2, 4], 2)
    cases.append({"name": "D_exact_not_contributor_without_truncation", "expected": QUERY_NOT_CONTRIBUTOR, "actual": d, "pass": d == QUERY_NOT_CONTRIBUTOR})
    e = _median_identity(3, 1, 2, 0)
    cases.append({"name": "E_median_different_region", "expected": MEDIAN_DIFFERENT_REGION, "actual": e, "pass": e == MEDIAN_DIFFERENT_REGION})
    f = _query_participation(99, list(range(16)), 17)
    cases.append({"name": "F_truncated_prefix_is_unavailable", "expected": QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE, "actual": f, "pass": f == QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE})
    return {
        "all_pass": bool(all(case["pass"] for case in cases)),
        "cases": cases,
        "no_contribution_threshold": True,
        "note": "Synthetic PASS verifies labels only; it is not physical or scene evidence.",
    }


def _load_w160_state(path: Path, row_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "global_state" not in data.files:
            raise ValueError("W160 observability NPZ lacks global_state")
        states = np.asarray(data["global_state"], dtype=np.int8)
    if states.shape != (row_count,):
        raise ValueError(f"W160 global_state shape {states.shape} does not match checkpoint rows {row_count}")
    return states


def _load_raw_w162(path: Path) -> dict[str, np.ndarray]:
    required = {
        "stable_gaussian_id", "checkpoint_row_index", "world_xyz",
        "gaussian_surface_region_id", "w155_membership_status", "global_state",
        "camera_names", "local_state", "query_depth", "median_event_depth",
        "signed_distance", "projected_pixel",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"W162 raw NPZ lacks {sorted(missing)}")
        result = {key: np.asarray(data[key]) for key in required}
    if result["local_state"].ndim != 2:
        raise ValueError("W162 local_state must have query x camera shape")
    if result["local_state"].shape[0] != result["checkpoint_row_index"].size:
        raise ValueError("W162 local state and row count disagree")
    return result


def _region_lineage(mapping: dict[str, Any], report_path: Path) -> dict[str, Any]:
    ids = mapping["stable_gaussian_id"]
    regions = mapping["region_id"]
    statuses = mapping["membership_status"]
    region_sets = {region: set(ids[regions == region].tolist()) for region in (0, 1)}
    stats: dict[str, Any] = {}
    for region in (0, 1):
        mask = regions == region
        stats[str(region)] = {
            "stable_gaussian_count": int(mask.sum()),
            "membership_status_counts": {name: int(np.count_nonzero(statuses[mask] == code)) for code, name in W155_STATUS_NAMES.items()},
            "accepted_core_or_attached_count": int(np.count_nonzero(mask & np.isin(statuses, [0, 1]))),
            "role": "primary_tabletop_review_candidate" if region == 0 else "frozen_tabletop_positive_control",
        }
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return {
        "source": str(report_path.resolve()),
        "w155_report_available": report_path.exists(),
        "region_stats": stats,
        "region_0_region_1_stable_id_overlap": int(len(region_sets[0] & region_sets[1])),
        "w155_role": "W155 fixed image-space tabletop review included candidate IDs from multiple regions, including 0 and 1; it did not define one broad physical tabletop population.",
        "w156_role": "Region 0 was the primary_tabletop_review_candidate; Region 2 and Region 5 were historical controls.",
        "w157_role": "Region 0 was the primary same-region TSDF separation audit; Region 2 and Region 5 remained controls.",
        "w162_role": "Region 1 was frozen as the tabletop positive control and its full 65,471-row population was audited.",
        "population_displayed": "W163 query provenance displays exactly W162 GLOBAL_OCCLUDED rows from W155 region_id=1; control views retain the frozen W162 ROI records and W155/W160 background boxes.",
    }


def _class_code(status: str) -> int:
    return {
        QUERY_NOT_CONTRIBUTOR: 0,
        QUERY_IS_EXACT_CONTRIBUTOR: 1,
        QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE: 2,
    }[status]


def _median_code(label: str) -> int:
    return {
        MEDIAN_IDENTITY_UNAVAILABLE: -1,
        MEDIAN_SAME_GAUSSIAN: 0,
        MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN: 1,
        MEDIAN_DIFFERENT_REGION: 2,
    }[label]


def _order_code(label: str) -> int:
    return {
        ORDER_UNAVAILABLE: -1,
        BEFORE_MEDIAN_EVENT: 0,
        AT_MEDIAN_EVENT: 1,
        AFTER_MEDIAN_EVENT: 2,
    }[label]


def _state_colour_tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return w160._state_colours(torch.as_tensor(values, dtype=torch.int8, device=device))


def _colour_rows(row_count: int, default: tuple[float, float, float], device: str) -> torch.Tensor:
    return torch.tensor(default, dtype=torch.float32, device=device).reshape(1, 3).expand(row_count, 3).clone()


def _save_png(path: Path, image: torch.Tensor) -> None:
    w160._save_png(path, image)


def _render_state(model: Any, rasterizer: Any, camera: Any, colours: torch.Tensor, *, original: bool, background: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return w160._render_state(model, rasterizer, camera, colours, original=original, background=background)

def _world_projection_png(
    path: Path,
    positions: np.ndarray,
    target_rows: np.ndarray,
    global_states: np.ndarray,
    conflict_rows: np.ndarray,
    axis_pair: tuple[int, int],
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_axis, y_axis = axis_pair
    context = positions[::max(1, len(positions) // 200000)]
    target_mask = np.zeros(len(positions), dtype=bool)
    target_mask[target_rows] = True
    conflict_mask = np.zeros(len(positions), dtype=bool)
    conflict_mask[conflict_rows] = True
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.scatter(context[:, x_axis], context[:, y_axis], s=0.15, c="#b8b8bc", alpha=0.08, linewidths=0, label="context")
    for code, colour, label in ((w160.STATE_OBSERVED, "#1ad95a", "GLOBAL OBSERVED"), (w160.STATE_OCCLUDED, "#eb2e2e", "GLOBAL OCCLUDED"), (w160.STATE_UNRESOLVED, "#99999f", "GLOBAL UNRESOLVED")):
        mask = target_mask & (global_states == code) & ~conflict_mask
        if np.any(mask):
            ax.scatter(positions[mask, x_axis], positions[mask, y_axis], s=1.4, c=colour, alpha=0.82, linewidths=0, label=label)
    if np.any(conflict_mask):
        ax.scatter(positions[conflict_mask, x_axis], positions[conflict_mask, y_axis], s=3.0, c="#ffe600", alpha=0.95, linewidths=0, label="CONFLICT: global O + exact contributor")
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


def _readme_texts(row_count: int, target_count: int, camera_count: int) -> dict[str, str]:
    conditions = f"frozen checkpoint/iteration, {row_count:,} Gaussian rows, {camera_count} training cameras, 648x420 camera calibration, OSNSurfelRasterizer, black background"
    common = (
        f"공통 rendering 조건: {conditions}. 모든 PNG는 camera 이름의 stem을 파일명으로 사용한다. "
        "Gaussian의 position, scale/covariance, rotation, opacity, row count와 geometry는 변경하지 않고 display color만 바꾼다. "
        "이 batch는 diagnostic이며 production renderer, Candidate-B, W160 state, W161, TSDF/topology/NURBS를 변경하지 않는다."
    )
    limitation = (
        "Review limitation: state 또는 provenance 색은 physical first-hit truth, hidden-surface existence, "
        "TSDF sign, topology, Boundary First ownership을 증명하지 않는다. `contrib_count > 16`인 pixel에서 "
        "captured prefix에 query가 없으면 결과는 NOT_CONTRIBUTOR가 아니라 PROVENANCE_UNAVAILABLE이다."
    )
    return {
        "root": f"""# W163 Renderer Contributor Provenance Audit

이 output은 W162의 renderer provenance gap을 existing isolated diagnostic CUDA sibling으로 재현한 read-only audit이다. W155/W156/W157의 Region lineage를 변경하지 않고, W162가 frozen한 W155 `region_id=1` positive-control 중 GLOBAL_OCCLUDED 6,197 Gaussian을 exact stable ID로 join한다. 전체 query population은 {target_count:,}행이고 {camera_count}개 camera의 per-pixel provenance를 기록한다.

{common}

Canonical renderer는 per-pixel contributor ID를 외부에 내보내지 않는다. 기존 diagnostic sibling은 kernel의 동일한 `T > 0.5` median crossing에서 `representative_id`를, accepted contributor의 최대 16-slot prefix와 uncapped `contrib_count`를 내보낸다. 따라서 truncation이 없는 경우에만 non-inclusion을 exact NOT_CONTRIBUTOR로 판정한다. `forward_accepted`는 view-level flag라 query-pixel evidence로 사용하지 않는다.

{limitation}
""",
        "review_root": f"""# W163 Review Views

이 directory는 각 visualization의 camera PNG와 개별 README를 보관한다. 각 하위 directory는 해당 visualization이 무엇을 나타내는지, input/state semantics, palette/legend, {conditions}, review limitation을 자체적으로 설명한다. camera subdirectory나 `render.png`는 사용하지 않는다.

Target population은 W162 frozen GLOBAL_OCCLUDED ∩ W155 `region_id=1`이며, common world view에서는 그 population 전체를 표시한다. `common_world`의 context는 전체 checkpoint를 읽기 쉽게 표시하기 위한 display-only downsample이다.
""",
        "original_scene": f"""# Original Scene

각 camera PNG는 frozen checkpoint의 전체 {row_count:,} Gaussian을 canonical `OSNSurfelRasterizer`로 원래 learned SH appearance 그대로 렌더링한 baseline이다. state color override, geometry 변경, marker Gaussian, light/shading 추가가 없다.

{common}

Legend: learned appearance 자체가 palette이다. 이 view의 색만으로 OBSERVED/OCCLUDED나 contributor identity를 판정하지 않는다.

{limitation}
""",
        "observed_occluded_global_state": f"""# Observed/Occluded Global State

각 PNG는 Original Scene과 같은 checkpoint, Gaussian rows, geometry, camera, renderer, resolution, background를 사용하고 W160 frozen global state만 display color로 바꾼 mandatory pair의 Observed/Occluded 쪽이다. green=`OBSERVED` `(0.10, 0.85, 0.35)`, red=`OCCLUDED` `(0.92, 0.18, 0.18)`, gray=`UNRESOLVED` `(0.60, 0.60, 0.62)`이다. `NON_RELEVANT`는 global vote가 아니므로 gray context로 남긴다.

{common}

이 색은 renderer-relative historical state이며 median event의 exact contributor나 physical hidden surface를 의미하지 않는다.

{limitation}
""",
        "tabletop_control_region": f"""# Tabletop Control Region

W155 frozen `region_id=1`의 전체 population을 blue `(0.12, 0.50, 0.92)`로 표시하고 나머지 Gaussian은 gray context로 표시한다. 이는 W162가 positive control로 고정한 world Region membership을 보여주는 control이다. 표시 population은 region 전체이며 GLOBAL_OCCLUDED subset만 뜻하지 않는다.

{common}

Legend: blue=`W155 REGION 1 MEMBERSHIP`, gray=`non-target context`. Region membership은 observation evidence나 median contributor identity가 아니다.

{limitation}
""",
        "global_occluded_tabletop": f"""# Global Occluded Tabletop

W155 `region_id=1` population 중 W160/W162 frozen global state가 `GLOBAL_OCCLUDED`인 Gaussian만 red `(0.92, 0.18, 0.18)`로 표시하고 나머지는 gray context로 표시한다. 이 view는 W162의 6,197-row query population을 전체 scene geometry 위에 위치시킨다.

{common}

Legend: red=`GLOBAL_OCCLUDED`, gray=`other/context`. red는 161 relevant training view의 median ordering aggregation 결과이지 physical first-hit truth가 아니다.

{limitation}
""",
        "query_renderer_contributor": f"""# Query Renderer Contributor

각 camera에서 W162 target query pixel의 accepted-contributor prefix에 query Gaussian stable ID가 exact로 포함되면 blue `(0.10, 0.55, 1.00)`로 표시한다. `contrib_count > 16`이고 prefix에 없으면 orange `(1.00, 0.55, 0.05)`=`PROVENANCE_UNAVAILABLE`, truncation이 없고 prefix에 없으면 gray=`QUERY_NOT_CONTRIBUTOR`이다. 비-query context도 gray이다.

{common}

이 view의 blue는 해당 camera/pixel에서 renderer가 exact contributor로 captured했다는 뜻이다. 이는 global O와 공존할 수 있으며 그 경우 central conflict이다.

{limitation}
""",
        "median_contributor_same_region": f"""# Median Contributor Same Region

각 target query pixel에서 diagnostic renderer가 반환한 `representative_id`가 query와 다른 Gaussian이면서 W155 Region ID가 같은 경우, 그 median representative row를 yellow `(1.00, 0.82, 0.05)`로 표시한다. gray는 해당 category가 아닌 context이다. `MEDIAN_SAME_GAUSSIAN`은 이 category에 포함하지 않는다.

{common}

Legend: yellow=`MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN`, gray=`other/context`. 색은 renderer median event identity와 W155 stable-ID join을 나타내며 physical surface ownership을 추가하지 않는다.

{limitation}
""",
        "median_contributor_different_region": f"""# Median Contributor Different Region

각 target query pixel의 median `representative_id`가 유효하고 그 representative의 W155 Region ID가 query Region 1과 다르면 해당 representative row를 magenta `(0.95, 0.15, 0.75)`로 표시한다. gray는 category 외 context이다.

{common}

Legend: magenta=`MEDIAN_DIFFERENT_REGION`, gray=`other/context`. 이는 exact renderer representative와 existing Region mapping의 관계만 보여준다.

{limitation}
""",
        "occluded_query_contributor_conflict": f"""# Occluded Query Contributor Conflict

W162 global-O tabletop query가 같은 camera의 같은 pixel에서 diagnostic renderer accepted-contributor prefix에 exact로 포함된 query row를 yellow `(1.00, 0.92, 0.05)`로 표시한다. 이는 `OCCLUDED_QUERY_RENDERER_CONTRIBUTOR_CONFLICT`라는 mechanical attribution이다. orange는 truncation 때문에 provenance가 unavailable인 target, gray는 그 밖의 context이다.

{common}

Legend: yellow=`GLOBAL_OCCLUDED + exact query renderer contributor` conflict, orange=`PROVENANCE_UNAVAILABLE`, gray=`other/context`. conflict는 physical false positive라고 부르지 않고, historical global-O label과 renderer contributor participation이 동시에 관측되었다는 뜻으로만 기록한다.

{limitation}
""",
        "common_world": f"""# Common World Views

`perspective.png`, `top.png`, `side.png`는 W155 Region 1과 W162 global state가 join된 frozen world XYZ를 각각 X-Z, X-Y, Y-Z orthographic diagnostic projection으로 표시한다. W155 Region 1 full population 65,471개는 모두 표시한다. `GLOBAL OBSERVED` green, `GLOBAL OCCLUDED` red, `GLOBAL UNRESOLVED` gray이며, 그 위에 exact conflict subset을 yellow로 표시한다.

{common}

Legend: green=`GLOBAL OBSERVED`, red=`GLOBAL OCCLUDED`, gray=`GLOBAL UNRESOLVED`, yellow=`OCCLUDED_QUERY_RENDERER_CONTRIBUTOR_CONFLICT`. 전체 checkpoint context는 display-only stride로 downsample하며 target population은 숨기지 않는다. 이 이미지는 실제 camera perspective나 W161 spatial field가 아니다.

{limitation}
""",
    }


def _write_readmes(out: Path, row_count: int, target_count: int, camera_count: int) -> None:
    texts = _readme_texts(row_count, target_count, camera_count)
    for name, text in texts.items():
        target = out / ("README.md" if name == "root" else "review_views/README.md" if name == "review_root" else f"review_views/{name}/README.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip() + "\n", encoding="utf-8")


def _clear_output(out: Path) -> None:
    if (out / "review_views").exists():
        shutil.rmtree(out / "review_views")
    for name in ("w163_query_provenance_raw.npz", "w163_control_provenance.npz", "control_provenance_records.json", "canonical_equivalence.json", "worklog_163_report.json"):
        path = out / name
        if path.exists():
            path.unlink()


def _pixel_metadata(diag: dict[str, Any]) -> dict[str, np.ndarray]:
    representative = diag["representative_id"].detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    contrib_ids = diag["contrib_ids"].detach().cpu().numpy().astype(np.int64, copy=False)
    post = diag["contrib_post_median"].detach().cpu().numpy().astype(np.int8, copy=False)
    count = diag["contrib_count"].detach().cpu().numpy().astype(np.int32, copy=False).reshape(-1)
    median_depth = diag["out_others"][5].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if contrib_ids.ndim != 3 or contrib_ids.shape[2] != SLOT_CAPACITY:
        raise ValueError(f"diagnostic slot shape {contrib_ids.shape} does not match K={SLOT_CAPACITY}")
    if post.shape != contrib_ids.shape or representative.shape != count.shape or representative.size != contrib_ids.shape[0] * contrib_ids.shape[1]:
        raise ValueError("diagnostic output shapes disagree")
    return {
        "representative": representative,
        "contrib_ids": contrib_ids.reshape(-1, SLOT_CAPACITY),
        "post": post.reshape(-1, SLOT_CAPACITY),
        "count": count,
        "median_depth": median_depth,
    }


def _record_at_pixel(
    row: int,
    flat_pixel: int,
    meta: dict[str, np.ndarray],
    stable_ids: np.ndarray,
    region_id: np.ndarray,
    membership_status: np.ndarray,
) -> dict[str, Any]:
    representative = int(meta["representative"][flat_pixel])
    query_region = int(region_id[row])
    median_region = int(region_id[representative]) if representative >= 0 else -1
    status = _query_participation(row, meta["contrib_ids"][flat_pixel], int(meta["count"][flat_pixel]))
    slots = np.flatnonzero(meta["contrib_ids"][flat_pixel] == int(row))
    slot = int(slots[0]) if len(slots) else -1
    post = int(meta["post"][flat_pixel, slot]) if slot >= 0 else -1
    identity = _median_identity(row, query_region, representative, median_region)
    order = _order_relation(row, representative, slot, post)
    return {
        "checkpoint_row_index": int(row),
        "stable_gaussian_id": int(stable_ids[row]),
        "query_region_id": query_region,
        "query_membership_status": W155_STATUS_NAMES.get(int(membership_status[row]), f"UNKNOWN_{int(membership_status[row])}"),
        "median_checkpoint_row_index": representative,
        "median_stable_gaussian_id": int(stable_ids[representative]) if representative >= 0 else -1,
        "median_region_id": median_region,
        "query_participation": status,
        "query_slot": slot,
        "query_post_median_flag": post,
        "median_identity": identity,
        "order_relation": order,
        "contrib_count": int(meta["count"][flat_pixel]),
        "contrib_slots_truncated": bool(int(meta["count"][flat_pixel]) > SLOT_CAPACITY),
        "median_depth": float(meta["median_depth"][flat_pixel]),
    }


def _classify_arrays(
    rows: np.ndarray,
    pixels: np.ndarray,
    meta: dict[str, np.ndarray],
    region_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(rows)
    participation = np.full(count, -1, dtype=np.int8)
    query_slot = np.full(count, -1, dtype=np.int16)
    median_row = np.full(count, -1, dtype=np.int32)
    median_region = np.full(count, -1, dtype=np.int32)
    median_identity = np.full(count, -1, dtype=np.int8)
    order = np.full(count, -1, dtype=np.int8)
    contrib_count = np.zeros(count, dtype=np.int32)
    for index, (row, pixel) in enumerate(zip(rows.tolist(), pixels.tolist())):
        pixel = int(pixel)
        representative = int(meta["representative"][pixel])
        ids = meta["contrib_ids"][pixel]
        total = int(meta["count"][pixel])
        status = _query_participation(int(row), ids, total)
        participation[index] = _class_code(status)
        slots = np.flatnonzero(ids == int(row))
        slot = int(slots[0]) if len(slots) else -1
        query_slot[index] = slot
        median_row[index] = representative
        mregion = int(region_id[representative]) if representative >= 0 else -1
        median_region[index] = mregion
        identity = _median_identity(int(row), int(region_id[row]), representative, mregion)
        median_identity[index] = _median_code(identity)
        post = int(meta["post"][pixel, slot]) if slot >= 0 else -1
        order[index] = _order_code(_order_relation(int(row), representative, slot, post))
        contrib_count[index] = total
    return participation, query_slot, median_row, median_region, median_identity, order, contrib_count


def _category_counts(
    participation: np.ndarray,
    median_identity: np.ndarray,
    order: np.ndarray,
    contrib_count: np.ndarray,
) -> dict[str, int]:
    exact_not = (participation == _class_code(QUERY_NOT_CONTRIBUTOR)) & (contrib_count <= SLOT_CAPACITY)
    exact = participation == _class_code(QUERY_IS_EXACT_CONTRIBUTOR)
    return {
        "A_REGION_MEMBER_BUT_NOT_RENDERER_CONTRIBUTOR": int(exact_not.sum()),
        "B_QUERY_RENDERER_CONTRIBUTOR_AFTER_MEDIAN": int((exact & (order == _order_code(AFTER_MEDIAN_EVENT))).sum()),
        "C_QUERY_RENDERER_CONTRIBUTOR_AT_OR_BEFORE_MEDIAN": int((exact & np.isin(order, [_order_code(AT_MEDIAN_EVENT), _order_code(BEFORE_MEDIAN_EVENT)])).sum()),
        "D_MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN": int((median_identity == _median_code(MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN)).sum()),
        "E_MEDIAN_DIFFERENT_REGION": int((median_identity == _median_code(MEDIAN_DIFFERENT_REGION)).sum()),
        "F_PROVENANCE_UNAVAILABLE": int((participation == _class_code(QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE)).sum()),
        "MEDIAN_SAME_GAUSSIAN": int((median_identity == _median_code(MEDIAN_SAME_GAUSSIAN)).sum()),
    }


def _control_records_for_camera(
    camera_name: str,
    review_data: dict[str, Any],
    meta: dict[str, np.ndarray],
    width: int,
    stable_ids: np.ndarray,
    region_id: np.ndarray,
    membership_status: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for control_name, control in review_data.items():
        for item in control.get("records", []):
            x, y = item["projected_pixel"]
            col, row = int(round(float(x))), int(round(float(y)))
            flat = row * width + col
            if flat < 0 or flat >= len(meta["representative"]):
                continue
            record = _record_at_pixel(int(item["checkpoint_row_index"]), flat, meta, stable_ids, region_id, membership_status)
            record.update({"camera": camera_name, "control": control_name, "source": "W162 review_projection_records.json"})
            records.append(record)
    return records


def _controls_from_background(
    camera_name: str,
    geometry: Any,
    meta: dict[str, np.ndarray],
    global_states: np.ndarray,
    positions: np.ndarray,
    stable_ids: np.ndarray,
    region_id: np.ndarray,
    membership_status: np.ndarray,
) -> list[dict[str, Any]]:
    boxes = (
        ((-1.0, 1.5, -0.15), (1.0, 2.5, 0.15)),
        ((-11.0, 2.0, 0.0), (-9.5, 3.5, 2.5)),
    )
    selected = np.zeros(len(positions), dtype=bool)
    for low, high in boxes:
        selected |= np.all((positions >= np.asarray(low)) & (positions <= np.asarray(high)), axis=1)
    selected &= global_states == w160.STATE_OCCLUDED
    rows = np.flatnonzero(selected & geometry.relevant.detach().cpu().numpy()).astype(np.int64)
    pixels = geometry.pixel_index.detach().cpu().numpy().astype(np.int64)[rows]
    records: list[dict[str, Any]] = []
    for row, pixel in zip(rows.tolist(), pixels.tolist()):
        record = _record_at_pixel(int(row), int(pixel), meta, stable_ids, region_id, membership_status)
        record.update({"camera": camera_name, "control": "background_lower", "source": "W160 frozen REVIEW_WORLD_BOXES"})
        records.append(record)
    return records


def _control_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_control[str(record["control"])].append(record)
    result: dict[str, Any] = {}
    for control, items in sorted(by_control.items()):
        result[control] = {
            "record_count": len(items),
            "camera_count": len({item["camera"] for item in items}),
            "query_participation": dict(Counter(item["query_participation"] for item in items)),
            "median_identity": dict(Counter(item["median_identity"] for item in items)),
            "order_relation": dict(Counter(item["order_relation"] for item in items)),
            "truncated_pixel_count": int(sum(item["contrib_slots_truncated"] for item in items)),
            "contrib_count_distribution": _distribution([item["contrib_count"] for item in items]),
        }
    return result


def _distribution(values: Iterable[int | float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _render_review_views(
    args: argparse.Namespace,
    model: Any,
    cameras: list[Any],
    names: list[str],
    positions: np.ndarray,
    region_id: np.ndarray,
    global_states: np.ndarray,
    target_rows: np.ndarray,
    target_index_by_row: dict[int, int],
    participation: np.ndarray,
    median_identity: np.ndarray,
    median_rows: np.ndarray,
    conflict_rows_by_camera: dict[str, np.ndarray],
    unavailable_rows_by_camera: dict[str, np.ndarray],
) -> None:
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros(3, dtype=torch.float32, device=args.device)
    row_count = len(positions)
    state_colours = _state_colour_tensor(global_states, args.device)
    target_mask = np.zeros(row_count, dtype=bool)
    target_mask[target_rows] = True
    target_mask_t = torch.as_tensor(target_mask, dtype=torch.bool, device=args.device)
    target_region = _colour_rows(row_count, CONTEXT_RGB, args.device)
    target_region[target_mask_t] = torch.tensor((0.12, 0.50, 0.92), device=args.device)
    global_occluded = _colour_rows(row_count, CONTEXT_RGB, args.device)
    global_occluded[torch.as_tensor(target_rows, dtype=torch.long, device=args.device)[torch.as_tensor(global_states[target_rows] == w160.STATE_OCCLUDED, device=args.device)]] = torch.tensor(w160.OCCLUDED_RGB, device=args.device)
    for camera_name in REVIEW_CAMERAS:
        camera = cameras[names.index(camera_name)]
        stem = Path(camera_name).stem
        _progress(f"rendering review camera {camera_name}")
        original = _render_state(model, rasterizer, camera, state_colours, original=True, background=background)
        _save_png(args.out / "review_views" / "original_scene" / f"{stem}.png", original)
        _save_png(args.out / "review_views" / "observed_occluded_global_state" / f"{stem}.png", _render_state(model, rasterizer, camera, state_colours, original=False, background=background))
        _save_png(args.out / "review_views" / "tabletop_control_region" / f"{stem}.png", _render_state(model, rasterizer, camera, target_region, original=False, background=background))
        _save_png(args.out / "review_views" / "global_occluded_tabletop" / f"{stem}.png", _render_state(model, rasterizer, camera, global_occluded, original=False, background=background))

        query_colours = _colour_rows(row_count, CONTEXT_RGB, args.device)
        exact_rows = target_rows[(participation[:, names.index(camera_name)] == _class_code(QUERY_IS_EXACT_CONTRIBUTOR))]
        unavailable_rows = unavailable_rows_by_camera[camera_name]
        if len(exact_rows):
            query_colours[torch.as_tensor(exact_rows, dtype=torch.long, device=args.device)] = torch.tensor(PROVENANCE_EXACT_RGB, device=args.device)
        if len(unavailable_rows):
            query_colours[torch.as_tensor(unavailable_rows, dtype=torch.long, device=args.device)] = torch.tensor(PROVENANCE_UNAVAILABLE_RGB, device=args.device)
        _save_png(args.out / "review_views" / "query_renderer_contributor" / f"{stem}.png", _render_state(model, rasterizer, camera, query_colours, original=False, background=background))

        same_rows = np.unique(median_rows[:, names.index(camera_name)][median_identity[:, names.index(camera_name)] == _median_code(MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN)])
        different_rows = np.unique(median_rows[:, names.index(camera_name)][median_identity[:, names.index(camera_name)] == _median_code(MEDIAN_DIFFERENT_REGION)])
        same_colours = _colour_rows(row_count, CONTEXT_RGB, args.device)
        different_colours = _colour_rows(row_count, CONTEXT_RGB, args.device)
        same_rows = same_rows[same_rows >= 0]
        different_rows = different_rows[different_rows >= 0]
        if len(same_rows):
            same_colours[torch.as_tensor(same_rows, dtype=torch.long, device=args.device)] = torch.tensor(MEDIAN_SAME_RGB, device=args.device)
        if len(different_rows):
            different_colours[torch.as_tensor(different_rows, dtype=torch.long, device=args.device)] = torch.tensor(MEDIAN_DIFFERENT_RGB, device=args.device)
        _save_png(args.out / "review_views" / "median_contributor_same_region" / f"{stem}.png", _render_state(model, rasterizer, camera, same_colours, original=False, background=background))
        _save_png(args.out / "review_views" / "median_contributor_different_region" / f"{stem}.png", _render_state(model, rasterizer, camera, different_colours, original=False, background=background))

        conflict_colours = _colour_rows(row_count, CONTEXT_RGB, args.device)
        conflicts = conflict_rows_by_camera[camera_name]
        if len(conflicts):
            conflict_colours[torch.as_tensor(conflicts, dtype=torch.long, device=args.device)] = torch.tensor(CONFLICT_RGB, device=args.device)
        if len(unavailable_rows):
            conflict_colours[torch.as_tensor(unavailable_rows, dtype=torch.long, device=args.device)] = torch.tensor(PROVENANCE_UNAVAILABLE_RGB, device=args.device)
        _save_png(args.out / "review_views" / "occluded_query_contributor_conflict" / f"{stem}.png", _render_state(model, rasterizer, camera, conflict_colours, original=False, background=background))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    synthetic = synthetic_contracts()
    if not synthetic["all_pass"]:
        raise RuntimeError("synthetic A-F contract failure")
    args.out.mkdir(parents=True, exist_ok=True)
    _clear_output(args.out)
    _progress("loading W162 raw population, checkpoint, cameras, and W155 mapping")
    model, payload = w160.load_primitive_model(args.checkpoint, device=args.device)
    if w160.checkpoint_primitive(payload) != w160.PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("canonical 2DGS surfel checkpoint required")
    stable_ids = payload["model_raw"].get("stable_gaussian_ids")
    if stable_ids is None:
        raise ValueError("checkpoint lacks stable_gaussian_ids")
    stable_ids_np = stable_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    mapping = w162._load_w155_mapping(args.wl155_mapping, stable_ids_np)
    region_id = mapping["region_id"]
    membership_status = mapping["membership_status"]
    positions_np = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    row_count = len(positions_np)
    global_states = _load_w160_state(args.wl160_state, row_count)
    raw = _load_raw_w162(args.wl162_raw)
    target_rows = raw["checkpoint_row_index"].astype(np.int64, copy=False)
    target_stable_ids = raw["stable_gaussian_id"].astype(np.int64, copy=False)
    if not np.array_equal(target_stable_ids, stable_ids_np[target_rows]):
        raise ValueError("W162 stable IDs do not exactly join checkpoint row indices")
    if not np.array_equal(raw["gaussian_surface_region_id"].astype(np.int64), region_id[target_rows]):
        raise ValueError("W162 region IDs do not exactly join W155 mapping")
    if not np.all(raw["global_state"] == w160.STATE_OCCLUDED) or not np.all(global_states[target_rows] == w160.STATE_OCCLUDED):
        raise ValueError("W163 requires W162 GLOBAL_OCCLUDED target population")
    camera_names = [str(item) for item in raw["camera_names"].tolist()]
    cameras, camera_meta = w160.load_all_train_cameras(args.source, args.images, args.sparse_dir, args.resolution, args.llffhold, args.device)
    names = [str(camera.image_name) for camera in cameras]
    if names != camera_names:
        raise ValueError("W162 and current frozen camera order differ")
    target_count = len(target_rows)
    camera_count = len(cameras)
    target_index_by_row = {int(row): index for index, row in enumerate(target_rows.tolist())}
    provenance = np.full((target_count, camera_count), -1, dtype=np.int8)
    query_slot = np.full((target_count, camera_count), -1, dtype=np.int16)
    median_rows = np.full((target_count, camera_count), -1, dtype=np.int32)
    median_regions = np.full((target_count, camera_count), -1, dtype=np.int32)
    median_identity = np.full((target_count, camera_count), -1, dtype=np.int8)
    order_relation = np.full((target_count, camera_count), -1, dtype=np.int8)
    contrib_count = np.zeros((target_count, camera_count), dtype=np.int32)
    target_representative_stable_id = np.full((target_count, camera_count), -1, dtype=np.int64)
    target_contrib_ids = np.full((target_count, camera_count, SLOT_CAPACITY), -1, dtype=np.int64)
    target_contrib_stable_ids = np.full((target_count, camera_count, SLOT_CAPACITY), -1, dtype=np.int64)
    target_contrib_post = np.full((target_count, camera_count, SLOT_CAPACITY), -1, dtype=np.int8)
    median_depth_diag = np.full((target_count, camera_count), np.nan, dtype=np.float32)
    control_records: list[dict[str, Any]] = []
    canonical_equivalence: dict[str, Any] = {}
    conflict_rows_by_camera: dict[str, np.ndarray] = {}
    unavailable_rows_by_camera: dict[str, np.ndarray] = {}
    review_records = json.loads(args.wl162_review.read_text(encoding="utf-8")) if args.wl162_review.exists() else {}

    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    canonical_rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros(3, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        for camera_index, (camera, camera_name) in enumerate(zip(cameras, names)):
            diag = render_with_pixel_representative(camera, model, background=background)
            meta = _pixel_metadata(diag)
            geometry = w160.project_queries(camera, model.get_xyz.detach())
            pixels = geometry.pixel_index.detach().cpu().numpy().astype(np.int64, copy=False)[target_rows]
            if np.any(pixels < 0):
                raise ValueError(f"W162 target row became non-relevant in {camera_name}")
            values = _classify_arrays(target_rows, pixels, meta, region_id)
            provenance[:, camera_index], query_slot[:, camera_index], median_rows[:, camera_index], median_regions[:, camera_index], median_identity[:, camera_index], order_relation[:, camera_index], contrib_count[:, camera_index] = values
            target_contrib_ids[:, camera_index, :] = meta["contrib_ids"][pixels]
            row_slots = target_contrib_ids[:, camera_index, :]
            valid_slots = row_slots >= 0
            safe_slots = np.where(valid_slots, row_slots, 0)
            target_contrib_stable_ids[:, camera_index, :] = np.where(valid_slots, stable_ids_np[safe_slots], -1)
            target_contrib_post[:, camera_index, :] = meta["post"][pixels]
            median_depth_diag[:, camera_index] = meta["median_depth"][pixels]
            valid_median = median_rows[:, camera_index] >= 0
            target_representative_stable_id[valid_median, camera_index] = stable_ids_np[median_rows[valid_median, camera_index]]
            exact = provenance[:, camera_index] == _class_code(QUERY_IS_EXACT_CONTRIBUTOR)
            conflict_rows_by_camera[camera_name] = target_rows[exact]
            unavailable_rows_by_camera[camera_name] = target_rows[provenance[:, camera_index] == _class_code(QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE)]
            if camera_name in review_records:
                control_records.extend(_control_records_for_camera(camera_name, review_records[camera_name], meta, int(camera.image_width), stable_ids_np, region_id, membership_status))
                control_records.extend(_controls_from_background(camera_name, geometry, meta, global_states, positions_np, stable_ids_np, region_id, membership_status))
            canonical = canonical_rasterizer.render(camera, model, background=background)
            diag_render = diag["render"].detach()
            canonical_render = canonical["render_unclamped"].detach()
            alpha_diag = diag["out_others"][1].detach()
            alpha_canonical = canonical["rend_alpha"].squeeze(0).detach()
            median_canonical = canonical["depth_median"].squeeze(0).detach()
            diag_median = diag["out_others"][5].detach()
            canonical_equivalence[camera_name] = {
                "render_bitwise_equal": bool(torch.equal(diag_render, canonical_render)),
                "alpha_bitwise_equal": bool(torch.equal(alpha_diag, alpha_canonical)),
                "median_depth_bitwise_equal": bool(torch.equal(diag_median, median_canonical)),
                "render_max_abs_diff": float((diag_render - canonical_render).abs().max().item()),
                "alpha_max_abs_diff": float((alpha_diag - alpha_canonical).abs().max().item()),
                "median_depth_max_abs_diff": float((diag_median - median_canonical).abs().max().item()),
                "renderer_canonical_unchanged": True,
            }
            if camera_index % 10 == 0 or camera_index == camera_count - 1:
                _progress(f"replayed diagnostic provenance {camera_index + 1}/{camera_count} cameras")
            del diag, meta, geometry, canonical

    if not all(item["render_bitwise_equal"] and item["alpha_bitwise_equal"] and item["median_depth_bitwise_equal"] for item in canonical_equivalence.values()):
        raise RuntimeError("diagnostic sibling is not bitwise equivalent to canonical output")
    if not np.all(raw["local_state"] == w160.STATE_OCCLUDED):
        raise ValueError("W162 raw local state unexpectedly contains non-OCCLUDED rows")
    diag_median_valid = np.isfinite(median_depth_diag) & (median_depth_diag > 0.0)
    cached_median = raw["median_event_depth"].astype(np.float32)
    median_match = np.isclose(median_depth_diag[diag_median_valid], cached_median[diag_median_valid], rtol=0.0, atol=0.0)
    if not np.all(median_match):
        raise RuntimeError("diagnostic median depth is not bitwise equal to W162 cached median at valid target pixels")

    raw_path = args.out / "w163_query_provenance_raw.npz"
    np.savez_compressed(
        raw_path,
        stable_gaussian_id=target_stable_ids,
        checkpoint_row_index=target_rows,
        world_xyz=raw["world_xyz"].astype(np.float32),
        gaussian_surface_region_id=region_id[target_rows],
        w155_membership_status=membership_status[target_rows],
        global_state=global_states[target_rows],
        camera_names=np.asarray(camera_names),
        query_participation=provenance,
        query_slot=query_slot,
        median_checkpoint_row_index=median_rows,
        median_stable_gaussian_id=target_representative_stable_id,
        median_region_id=median_regions,
        median_identity=median_identity,
        order_relation=order_relation,
        contrib_count=contrib_count,
        contrib_ids=target_contrib_stable_ids,
        contrib_checkpoint_row_index=target_contrib_ids,
        contrib_post_median=target_contrib_post,
        diagnostic_median_depth=median_depth_diag,
        w162_median_event_depth=cached_median,
    )
    control_path = args.out / "w163_control_provenance.npz"
    _write_json(args.out / "control_provenance_records.json", control_records)
    np.savez_compressed(
        control_path,
        checkpoint_row_index=np.asarray([item["checkpoint_row_index"] for item in control_records], dtype=np.int64),
        stable_gaussian_id=np.asarray([item["stable_gaussian_id"] for item in control_records], dtype=np.int64),
        median_checkpoint_row_index=np.asarray([item["median_checkpoint_row_index"] for item in control_records], dtype=np.int64),
        median_stable_gaussian_id=np.asarray([item["median_stable_gaussian_id"] for item in control_records], dtype=np.int64),
        camera=np.asarray([item["camera"] for item in control_records]),
        control=np.asarray([item["control"] for item in control_records]),
        query_participation=np.asarray([item["query_participation"] for item in control_records]),
        median_identity=np.asarray([item["median_identity"] for item in control_records]),
        order_relation=np.asarray([item["order_relation"] for item in control_records]),
        contrib_count=np.asarray([item["contrib_count"] for item in control_records], dtype=np.int32),
    )

    all_category_counts = _category_counts(provenance.reshape(-1), median_identity.reshape(-1), order_relation.reshape(-1), contrib_count.reshape(-1))
    participation_counts = {QUERY_NOT_CONTRIBUTOR: int(np.count_nonzero(provenance == _class_code(QUERY_NOT_CONTRIBUTOR))), QUERY_IS_EXACT_CONTRIBUTOR: int(np.count_nonzero(provenance == _class_code(QUERY_IS_EXACT_CONTRIBUTOR))), QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE: int(np.count_nonzero(provenance == _class_code(QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE)))}
    order_counts = {ORDER_UNAVAILABLE: int(np.count_nonzero(order_relation == _order_code(ORDER_UNAVAILABLE))), BEFORE_MEDIAN_EVENT: int(np.count_nonzero(order_relation == _order_code(BEFORE_MEDIAN_EVENT))), AT_MEDIAN_EVENT: int(np.count_nonzero(order_relation == _order_code(AT_MEDIAN_EVENT))), AFTER_MEDIAN_EVENT: int(np.count_nonzero(order_relation == _order_code(AFTER_MEDIAN_EVENT)))}
    median_identity_counts = {MEDIAN_IDENTITY_UNAVAILABLE: int(np.count_nonzero(median_identity == _median_code(MEDIAN_IDENTITY_UNAVAILABLE))), MEDIAN_SAME_GAUSSIAN: int(np.count_nonzero(median_identity == _median_code(MEDIAN_SAME_GAUSSIAN))), MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN: int(np.count_nonzero(median_identity == _median_code(MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN))), MEDIAN_DIFFERENT_REGION: int(np.count_nonzero(median_identity == _median_code(MEDIAN_DIFFERENT_REGION)))}
    conflict_pairs = np.argwhere(provenance == _class_code(QUERY_IS_EXACT_CONTRIBUTOR))
    conflict_rows = np.unique(target_rows[conflict_pairs[:, 0]]) if len(conflict_pairs) else np.empty(0, dtype=np.int64)
    conflict_stable_ids = stable_ids_np[conflict_rows] if len(conflict_rows) else np.empty(0, dtype=np.int64)
    category_by_camera = {name: _category_counts(provenance[:, index], median_identity[:, index], order_relation[:, index], contrib_count[:, index]) for index, name in enumerate(camera_names)}

    _progress("rendering PNG review views and common world plots")
    _render_review_views(args, model, cameras, names, positions_np, region_id, global_states, target_rows, target_index_by_row, provenance, median_identity, median_rows, conflict_rows_by_camera, unavailable_rows_by_camera)
    region1_rows = np.flatnonzero(region_id == TABLETOP_REGION_ID).astype(np.int64)
    common_root = args.out / "review_views" / "common_world"
    _world_projection_png(common_root / "perspective.png", positions_np, region1_rows, global_states, conflict_rows, (0, 2), "W163 common world: X-Z diagnostic perspective")
    _world_projection_png(common_root / "top.png", positions_np, region1_rows, global_states, conflict_rows, (0, 1), "W163 common world: X-Y top")
    _world_projection_png(common_root / "side.png", positions_np, region1_rows, global_states, conflict_rows, (1, 2), "W163 common world: Y-Z side")
    _write_readmes(args.out, row_count, target_count, camera_count)

    lineage = _region_lineage(mapping, args.wl155_report)
    exact_conflict_verdict = "OCCLUDED_QUERY_RENDERER_CONTRIBUTOR_CONFLICT" if len(conflict_pairs) else "NO_EXACT_CONFLICT_RECOVERED"
    if len(conflict_pairs):
        architecture_result = "MIXED"
        architecture_reason = "Exact conflict pairs exist, but bounded contributor slots leave a larger provenance-unavailable population; the result is mixed, not a global success claim."
    elif participation_counts[QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE]:
        architecture_result = "CONTRIBUTOR_OBSERVATION_SEMANTIC_GAP"
        architecture_reason = "No exact conflict was recovered, but truncated prefixes prevent a negative contributor claim for unavailable pixels."
    else:
        architecture_result = "MEDIAN_EVENT_PROXY_SUPPORTED_BY_RENDERER_PROVENANCE"
        architecture_reason = "All relevant query pixels have complete bounded provenance and no conflict was found."
    report = {
        "status": "COMPLETE_WL163_RENDERER_CONTRIBUTOR_PROVENANCE_MEDIAN_EVENT_OBSERVATION_SEMANTICS_ATTRIBUTION_AUDIT",
        "batch": "Worklog 163 -- Renderer Contributor Provenance and Median-Event Observation-Semantics Attribution Audit",
        "intent_alignment": {"diagnostic_only": True, "production_behavior_modified": False, "candidate_b_modified": False, "w160_state_modified": False, "w161_modified": False, "w155_w162_modified": False, "new_observation_semantics": False, "human_review_required": True},
        "implementation_fidelity": {"source_of_median_identity": "existing isolated diff_surfel_rasterization_diag sibling at canonical kernel T > 0.5 crossing", "source_of_contributor_sequence": "same diagnostic forward execution accepted-contributor prefix", "slot_capacity": SLOT_CAPACITY, "uncapped_count_used": True, "stable_id_join": "checkpoint row index -> model_raw.stable_gaussian_ids -> W155 mapping", "no_epsilon_order": True, "no_contribution_threshold": True, "canonical_renderer_untouched": True},
        "architecture_result": architecture_result,
        "architecture_reason": architecture_reason,
        "mechanical_conflict_verdict": exact_conflict_verdict,
        "w162_reconciliation": {"raw_npz": str(args.wl162_raw.resolve()), "target_count": target_count, "target_region_id": TABLETOP_REGION_ID, "global_state_counts": _state_counts(global_states[target_rows]), "all_local_states_are_occluded": bool(np.all(raw["local_state"] == w160.STATE_OCCLUDED)), "same_stable_id_join": True, "same_pixel_depth_bitwise_equal": bool(np.all(median_match))},
        "tabletop_control_lineage_reconciliation": lineage,
        "renderer_provenance_capability": {"canonical_production": {"contributor_id_externally_available": False, "median_event_identity": "not returned by canonical Python binding", "per_pixel_alpha_times_T": False}, "existing_diagnostic_sibling": {"available": True, "representative_id": "exact global surfel row index at T > 0.5 median crossing", "contrib_ids": "captured accepted contributor prefix, exported as stable Gaussian IDs; row-index slots are also retained in the raw NPZ", "contrib_post_median": "0 at-or-before, 1 strictly after", "contrib_count": "true uncapped accepted contributor count", "forward_accepted": "view-level only; not used as query-pixel evidence"}, "stop_condition_A": "not triggered; deterministic isolated diagnostic rerender was recoverable without new production semantics"},
        "stop_condition_result": {"stop_a_triggered": False, "verdict": "DIAGNOSTIC_PROVENANCE_RECOVERABLE", "basis": "existing diagnostic sibling exposes exact median representative identity and bounded contributor prefix; truncation is explicitly detectable"},
        "query_contributor_provenance": {"population": "W162 GLOBAL_OCCLUDED rows in W155 region_id=1", "query_count": target_count, "camera_count": camera_count, "pair_count": target_count * camera_count, "status_counts": participation_counts, "per_camera_category_counts": category_by_camera, "raw_npz": str(raw_path.resolve()), "meaning": {QUERY_IS_EXACT_CONTRIBUTOR: "query row appears in captured accepted-contributor slots", QUERY_NOT_CONTRIBUTOR: "query absent and uncapped count <= 16", QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE: "query absent but uncapped count > 16; no negative claim"}},
        "median_event_contributor_provenance": {"representative_identity_source": "diagnostic representative_id", "median_identity_counts": median_identity_counts, "median_stable_id_array": "w163_query_provenance_raw.npz::median_stable_gaussian_id", "same_gaussian": MEDIAN_SAME_GAUSSIAN, "same_region_different_gaussian": MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN, "different_region": MEDIAN_DIFFERENT_REGION, "unavailable": MEDIAN_IDENTITY_UNAVAILABLE},
        "contributor_order_relation": {"counts": order_counts, "definitions": {BEFORE_MEDIAN_EVENT: "exact contributor slot has post flag 0 and is not median representative", AT_MEDIAN_EVENT: "query stable ID equals median representative stable ID", AFTER_MEDIAN_EVENT: "exact contributor slot has post flag 1", ORDER_UNAVAILABLE: "query not exact, no median identity, or truncated/noncaptured provenance"}},
        "tabletop_conflict_accounting": {
            "classification": exact_conflict_verdict,
            "exact_conflict_pair_count": int(len(conflict_pairs)),
            "unique_query_stable_id_count": int(len(conflict_stable_ids)),
            "unique_query_stable_ids_sample": conflict_stable_ids[:50].tolist(),
            "camera_pair_counts": {camera_names[index]: int(np.count_nonzero(provenance[:, index] == _class_code(QUERY_IS_EXACT_CONTRIBUTOR))) for index in range(camera_count)},
            "order_within_exact_query_contributor": dict(Counter(order_relation[provenance == _class_code(QUERY_IS_EXACT_CONTRIBUTOR)].tolist())),
            "interpretation": "This is a mechanical coexistence of W162 GLOBAL_OCCLUDED and exact renderer contributor participation at the relevant pixel; it is not labeled a physical false positive.",
            "median_same_gaussian_conflicts": int(np.count_nonzero((provenance == _class_code(QUERY_IS_EXACT_CONTRIBUTOR)) & (median_identity == _median_code(MEDIAN_SAME_GAUSSIAN)))),
            "median_same_region_different_conflicts": int(np.count_nonzero((provenance == _class_code(QUERY_IS_EXACT_CONTRIBUTOR)) & (median_identity == _median_code(MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN)))),
            "median_different_region_conflicts": int(np.count_nonzero((provenance == _class_code(QUERY_IS_EXACT_CONTRIBUTOR)) & (median_identity == _median_code(MEDIAN_DIFFERENT_REGION)))),
        },        "control_population_results": {"records": len(control_records), "summary": _control_summary(control_records), "sources": ["W162 review_projection_records.json", "W160 frozen REVIEW_WORLD_BOXES"], "new_roi_or_region": False},
        "continuous_contribution_limitation": {"available": False, "per_pixel_per_primitive_alpha_times_T": False, "reason": "diagnostic sibling exposes IDs, order flags, and uncapped count but not per-primitive alpha*T magnitude", "forward_accepted_not_substitute": True, "no_threshold_applied": True},
        "synthetic_contracts_A_to_F": synthetic,
        "quantitative_result": {"category_counts_all_target_camera_pairs": all_category_counts, "contrib_count_distribution_all_target_camera_pairs": _distribution(contrib_count.reshape(-1)), "truncated_pair_count": int(np.count_nonzero(contrib_count > SLOT_CAPACITY)), "canonical_equivalence": canonical_equivalence, "canonical_equivalence_all_bitwise": True, "median_depth_exact_reconciliation": {"valid_pair_count": int(diag_median_valid.sum()), "bitwise_equal_pair_count": int(median_match.sum()), "max_abs_difference": 0.0}},
        "human_qualitative_review_exports": {"review_root": str((args.out / "review_views").resolve()), "visualizations": ["original_scene", "observed_occluded_global_state", "tabletop_control_region", "global_occluded_tabletop", "query_renderer_contributor", "median_contributor_same_region", "median_contributor_different_region", "occluded_query_contributor_conflict", "common_world"], "png_primary": True, "ppm_count": 0, "camera_file_layout": "<camera_name_stem>.png directly inside each visualization directory", "readme_per_visualization": True},
        "human_review_questions": ["W162 GLOBAL_OCCLUDED tabletop Gaussians 중 어떤 rows가 relevant training-view rendering에 실제로 accepted contributor로 참여하는가?", "각 query pixel의 median event를 생성한 exact renderer contributor stable ID는 무엇인가?", "exact conflict pair는 median representative와 같은 Gaussian인가, 같은 Region의 다른 Gaussian인가, 또는 다른 Region인가?", "bounded slots 때문에 unavailable population이 시각적으로 어떻게 분포하는가?"],
        "w161_consequence": {"status": "REMAINS_PAUSED", "report": str(args.wl161_report.resolve()), "reason": "W163 does not construct or join a W161 spatial field; W161 OCCLUSION_DOMAIN_CONTRACT_GAP is retained."},
        "retained_rejected_open": {"retained": ["W155 Region 0/1 lineage and stable-ID mapping", "W156/W157 historical roles", "W160 global state", "W161 paused spatial-domain gap", "W162 raw query population and review controls", "canonical renderer and diagnostic sibling provenance"], "rejected": ["query non-inclusion under truncation as NOT_CONTRIBUTOR", "nearest-Gaussian inference", "RGB/alpha heuristic", "per-primitive contribution threshold", "fused TSDF sign as observation evidence", "new region/ROI/spatial join", "production renderer or Candidate-B change"], "open": ["per-pixel per-primitive alpha*T magnitude", "complete contributor sequence for truncated pixels", "physical first-hit truth of a renderer median event", "W161 spatial join remains paused"]},
        "inputs": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint), "iteration": int(payload.get("iteration", 0)), "source": str(args.source.resolve()), "camera_names": names, "camera_meta": camera_meta, "w155_mapping": str(args.wl155_mapping.resolve()), "w155_mapping_sha256": _sha256_file(args.wl155_mapping), "w153_cache": str(args.cache.resolve()), "w153_replay_cache_excluded_from_temp_mirror": True, "w160_state": str(args.wl160_state.resolve()), "w162_raw": str(args.wl162_raw.resolve()), "w162_review": str(args.wl162_review.resolve()) if args.wl162_review.exists() else None, "renderer": "canonical OSNSurfelRasterizer plus existing isolated diff_surfel_rasterization_diag", "slot_capacity": SLOT_CAPACITY},
        "outputs": {"report": str((args.out / "worklog_163_report.json").resolve()), "raw_npz": str(raw_path.resolve()), "control_npz": str(control_path.resolve()), "control_json": str((args.out / "control_provenance_records.json").resolve()), "review_root": str((args.out / "review_views").resolve()), "visualization_output_format": "PNG primary; no PPM emitted"},
        "forbidden_changes": {"production_renderer": False, "candidate_b": False, "w160": False, "w161": False, "w155_w162": False, "tsdf": False, "topology": False, "boundary_first": False, "nurbs": False, "continuation": False},
        "summary_rule": "Exact conflict existence is reported mechanically; truncation/unavailable provenance prevents a blanket success or blanket negative claim. architecture_result=MIXED when exact conflicts and unavailable pairs coexist.",
        "runtime_seconds": {"total": time.time() - started},
    }
    _write_json(args.out / "canonical_equivalence.json", canonical_equivalence)
    _write_json(args.out / "worklog_163_report.json", report)
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
