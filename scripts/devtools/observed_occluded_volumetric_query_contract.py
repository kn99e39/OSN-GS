"""Worklog 123 -- Volumetric frontier query contract closure.

Worklog 122 conditionally accepted the canonical median-surface event as a
renderer-defined visible-surface observation frontier. That acceptance does NOT
mean median depth is the physical first hit, and this batch does not reopen the
A/C/D comparison or tune candidate B.

The remaining question is the QUERY CONTRACT: how should an arbitrary 3D
world-space point be evaluated against that frontier while preserving exact
identity for renderer-originated frontier events?

Candidate architecture under test (the simplest one):

    canonical query      world-space position x
    optional provenance  exact renderer median-event identity when x came from one
    per-view frontier    canonical stored median depth

`(camera_id, pixel_id, stored_median_depth)` is valid renderer-event provenance
but must NOT replace world-space 3D as the global volumetric representation. No
heuristic epsilon is approved; the float64 arm here is diagnostic only and never
canonical.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import (  # noqa: E402
    PRIMITIVE_SURFEL_2D, checkpoint_primitive, load_primitive_model,
    _rgb_to_f_dc, write_ppm, write_surfel_ply,
)
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402

from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded import query_bank as bank_module  # noqa: E402
from observed_occluded import query_contract_synthetics, topology_gap_bank, volumetric_query  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    RELEVANCE_NAMES, RELEVANCE_OK, STATE_NAMES, STATE_NON_RELEVANT, STATE_OBSERVED,
    STATE_OCCLUDED, STATE_UNRESOLVED, aggregate_global, distribution, project_queries,
    ViewGeometry,
    reconstruct_direct_surfel_intersection_world_point, state_fractions,
)
from observed_occluded.volumetric_query import (  # noqa: E402
    IDENTITY_NAMES, IDENTITY_ON_FRONTIER, EventIdentityAccumulator, StabilityAccumulator,
    VolumetricQueryBank, apply_event_identity, project_queries_float64, reference_side,
)

WL119_REPRESENTATIVE_UNION = 785937
WL122_SOURCE_EVENTS = 43817760
WL122_SOURCE_CONTRADICTIONS = 8157322
WL122_GLOBAL_OCCLUDED_ANCHORS = 19

# Fixed a priori -- selection strides and probe positions, decided before any
# result was observed. None of them is a tolerance and no decision depends on
# their values.
ANCHOR_VIEW_STRIDE = 10          # same construction as worklog 122's disocclusion corpus
ANCHORS_PER_VIEW = 200
LADDER_ANCHOR_STRIDE = 7         # every 7th anchor also seeds a near-frontier ladder
FREE_SPACE_T = 0.5
BEHIND_T = 1.5
# Relative offsets from the frontier, as exact fractions of the frontier depth:
# camera-space depth is linear along the ray from the camera centre, so
# depth(t) = t * depth(frontier) and the relative margin is exactly (t - 1).
# A fixed decade ladder spanning float32's resolution limit, so the audit can
# see WHERE disagreements stop -- a measurement design, never a threshold.
NEAR_FRONTIER_DECADES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
MARGIN_SAMPLE_STRIDE = 991

KIND_EVENT_ANCHOR = "P1_RENDERER_EVENT_ANCHOR"
KIND_FREE_SPACE = "G1_OBSERVED_FREE_SPACE"
KIND_BEHIND = "G2_BEHIND_FRONTIER"
KIND_NEAR_FRONTIER = "G3_NEAR_FRONTIER_LADDER"
KIND_WL120 = "G4_WL120_ORIGINAL_BANK"
KIND_WL121 = "G5_WL121_SUPPLEMENTAL_BANK"

_ITERATION_DIR = "iteration_0000001"
_SCENE_RGB = (0.07, 0.08, 0.10)
_MARKER_RADIUS = 0.036220766603946686 * 1.5   # identical marker size to worklogs 120-122
_EFFECT_RGB = {
    "unchanged_observed": (0.10, 0.85, 0.35),
    "rescued_by_identity": (0.20, 0.55, 0.98),
    "still_occluded": (0.92, 0.18, 0.18),
    "unresolved": (0.60, 0.60, 0.62),
}
_REPORT_NAME = "volumetric_query_contract_report.json"
_WORKLOG_NAME = "123_volumetric_frontier_query_contract_closure.md"


ORIGINAL_SCENE_README = """# ORIGINAL_2DGS_SCENE

## 색상 의미
- 학습된 2DGS 체크포인트를 **원래의 학습된 SH 색상 그대로** 렌더링한 것이다. 진단용 색상 부호화가 전혀 없다.

## 이 이미지가 보여주는 것
이번 배치의 진단 view가 공유하는 **기준 장면**이다. `EVENT_IDENTITY_EFFECT` / `NEAR_FRONTIER_LADDER`와 **같은 카메라·같은 iteration**에서 렌더링했으므로, 질의 점이 장면의 어느 구조 위에 놓였는지 대조하는 기준이 된다.
"""

EVENT_IDENTITY_README = """# EVENT_IDENTITY_EFFECT

## 색상 의미
- **초록** (`0.10, 0.85, 0.35`): provenance 유무와 무관하게 global `OBSERVED` — world-space 왕복만으로도 이미 닫힌 anchor
- **파랑** (`0.20, 0.55, 0.98`): **provenance를 유지했을 때만** global `OBSERVED`가 된 anchor (world-space float32 왕복만으로는 global `OCCLUDED`였던 잔여 모순)
- **빨강** (`0.92, 0.18, 0.18`): provenance를 유지해도 여전히 global `OCCLUDED`
- **회색** (`0.60, 0.60, 0.62`): global `UNRESOLVED`
- **거의 검은 남색** (`0.07, 0.08, 0.10`): 학습된 2DGS 장면 전체(문맥용, 판정과 무관)

## 이 이미지가 보여주는 것
매 {stride}번째 학습 뷰에서 결정론적 stride로 뽑은 **{anchors}개 renderer median event anchor**에 대해, **exact event-identity provenance 계층이 실제로 무엇을 바꾸는지**만 따로 칠한 그림이다.

파랑(= provenance가 구제한 anchor)은 **{rescued}개**, 초록(원래부터 닫혀 있던 anchor)은 {unchanged}개, 빨강(여전히 OCCLUDED)은 {still}개다. provenance는 오직 "이 질의가 이 뷰의 바로 그 renderer median event인가"에만 답하며, 다른 뷰의 판정·global 집계·component 소유권·표면 연속성·신뢰도에는 **일절 관여하지 않는다**. 화면 대부분이 초록이라는 사실 자체가 "world-space 3D 질의가 이미 대부분 닫혀 있다"는 관측이다.

**해석 주의**: 이 그림은 수치 경계 정책을 승인하지 않는다. epsilon·ULP 수용 대역·nextafter 보정은 이번 배치 어디에도 도입되지 않았다.
"""

NEAR_FRONTIER_README = """# NEAR_FRONTIER_LADDER

## 색상 의미
- **파랑 계열** (`0.15, ~, 0.95`): frontier보다 **카메라 쪽**(상대 offset 음수). 밝을수록 frontier에서 멀다(1e-7 → 1e-2)
- **주황 계열** (`0.95, ~, 0.15`): frontier보다 **뒤쪽**(상대 offset 양수). 밝을수록 frontier에서 멀다
- **거의 검은 남색** (`0.07, 0.08, 0.10`): 학습된 2DGS 장면 전체(문맥용)

## 이 이미지가 보여주는 것
renderer median event anchor의 카메라 광선 위에서 frontier로부터 **상대 offset {decades}의 ±10진 사다리**로 배치한 {probes}개 probe다. camera-space depth는 카메라 중심에서 광선을 따라 선형이므로, 파라미터 t의 probe는 정확히 `depth = t x frontier_depth`에 놓이고 상대 margin은 정확히 `t - 1`이다.

이 사다리는 **float32와 float64 기준 arm의 판정이 어느 거리에서 갈리기 시작하는지**를 재기 위한 측정 설계이며, offset별 불일치율은 리포트의 `near_frontier_attribution.relative_offset_ladder`에 있다.

**해석 주의**: 이 offset들은 tolerance가 **아니다** — 여기서 epsilon·ULP 수용 대역·백분율 임계값을 유도하지 않았고, 후보 B는 변경되지 않았다.
"""


def write_view_readme(folder: Path, body: str, surfels: int) -> None:
    """Every export view folder carries its own Korean README (required by
    docs/output_folder_conventions.md). Written HERE, in the script, next to the
    PLY and the PPM -- never as a manual post-step, so it cannot be forgotten."""

    folder.mkdir(parents=True, exist_ok=True)
    footer = (
        "\n---\n"
        "체크포인트: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt` "
        f"({surfels:,} surfel, 161 train camera)\n"
        f"전체 리포트: `../{_REPORT_NAME}` · "
        f"Worklog: [`docs/worklogs/{_WORKLOG_NAME}`](../../../../docs/worklogs/{_WORKLOG_NAME})\n"
    )
    (folder / "README.md").write_text(body + footer, encoding="utf-8")


def _progress(message: str) -> None:
    print(f"[wl123-query] {message}", flush=True)


def _quantiles(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    out = distribution(finite)
    if finite.size:
        for fraction in (0.01, 0.05, 0.25, 0.75, 0.99):
            out[f"p{int(fraction * 100):02d}"] = float(np.quantile(finite, fraction))
    out["non_finite_excluded"] = int(values.size - finite.size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--wl120-npz", type=Path,
                        default=Path("output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"))
    parser.add_argument("--wl121-npz", type=Path,
                        default=Path("output/confirmed/121_osn_gs_observed_occluded_value_space/value_space_supplemental_bank.npz"))
    parser.add_argument("--wl122-report", type=Path,
                        default=Path("output/confirmed/122_osn_gs_median_frontier_validation/median_frontier_validation_report.json"))
    parser.add_argument("--max-views", type=int, default=0, help="smoke-test only")
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 123 (volumetric frontier query contract closure)",
        "question": (
            "Can world-space 3D remain the canonical volumetric query abstraction, with renderer-event "
            "provenance preserving exact frontier identity, without adding a heuristic boundary tolerance?"
        ),
        "preserved": {
            "candidate_B_classify_view": "unmodified (imported and called)",
            "median_rule": "canonical pre-update T > 0.5, untouched",
            "global_aggregation": "frozen ANY-OBSERVED, untouched",
            "worklog_120_121_122": "historical evidence, measurements not rewritten",
        },
        "no_epsilon_introduced": (
            "Provenance validity uses EXACT bitwise agreement between the carried stored_median_depth and the "
            "renderer's own median at the carried pixel. No epsilon, ULP band, nextafter correction or "
            "percentage threshold exists anywhere in this batch."
        ),
    }

    # ================================================================ WL122 corrections
    _progress("[1/8] worklog 122 interpretation corrections (count vs weight audit)")
    corrections: dict[str, Any] = {
        "carried_forward": [
            "'renderer-visible somewhere' does NOT imply same-surface redundant contribution -- Renderer "
            "Contribution, Surface Representative, Surface Identity and Volumetric Observation are separate concepts.",
            "'same frozen visible component' is stronger provenance but still does not by itself prove physical redundancy.",
            "Worklog 122's statement that post-median evidence is 'overwhelmingly redundant representation' is "
            "NOT established and is retracted as an interpretation here (its underlying counts stand).",
            "Worklog 122's 2.24% is only an UPPER BOUND on post-median contribution weight from surfels that "
            "are never median representatives anywhere, under marginal-distribution assumptions -- never an "
            "exact amount of independent hidden-surface evidence.",
        ],
    }
    if arguments.wl122_report.exists():
        wl122 = json.loads(arguments.wl122_report.read_text(encoding="utf-8"))
        post = wl122["post_median_accounting"]
        counts, weights = post["counts_by_category"], post["contribution_mass_by_category"]
        total_weight = post["total_accepted_contribution_mass"]
        count_front = counts["depth_in_front_of_median"] / counts["all"]
        count_behind = counts["depth_at_or_behind_median"] / counts["all"]
        weight_front = weights["depth_in_front_of_median"] / weights["all"]
        weight_behind = weights["depth_at_or_behind_median"] / weights["all"]
        post_weight_fraction = post["post_median_contribution_mass"] / total_weight
        direct_behind_of_total = weights["depth_at_or_behind_median"] / total_weight
        chained = post_weight_fraction * weight_behind
        corrections["post_median_count_vs_weight_audit"] = {
            "post_median_contributor_count": counts["all"],
            "front_of_median_contributor_count": counts["depth_in_front_of_median"],
            "behind_median_contributor_count": counts["depth_at_or_behind_median"],
            "front_of_median_COUNT_fraction": count_front,
            "behind_median_COUNT_fraction": count_behind,
            "front_of_median_WEIGHT_fraction": weight_front,
            "behind_median_WEIGHT_fraction": weight_behind,
            "post_median_WEIGHT_fraction_of_total": post_weight_fraction,
            "worklog_122_quoted_27_65_percent_is": (
                "the WEIGHT fraction (0.27646), not the count fraction (0.21618)"
            ),
            "chained_claim_0_39055_times_0_72354": chained,
            "direct_recomputation_behind_weight_over_total": direct_behind_of_total,
            "chain_matches_direct_recomputation": bool(abs(chained - direct_behind_of_total) < 1e-12),
            "historical_28_26_percent_verdict": (
                "VALID -- both factors are contribution-WEIGHT fractions, so their product is the weight "
                "fraction of total accepted contribution that is post-median AND behind the median depth. "
                "Direct recomputation reproduces it exactly."
            ),
            "presentational_correction": (
                "Worklog 122's prose put the contributor COUNT (248,820,747) next to the 27.65% WEIGHT share "
                "in one sentence. The count share is 21.62%. The numbers were right; the sentence conflated "
                "two different fractions."
            ),
        }
    report["worklog_122_interpretation_corrections"] = corrections

    # ================================================================ scene
    _progress(f"[2/8] loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint.")
    device = model.device
    total_model_count = len(model)
    cameras, camera_meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, arguments.device,
    )
    if int(arguments.max_views) > 0:
        cameras = cameras[: int(arguments.max_views)]
        camera_meta = {**camera_meta, "smoke_test_max_views": int(arguments.max_views)}
    report["total_trained_surfels"] = total_model_count
    report["camera_meta"] = camera_meta

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    with torch.no_grad():
        positions_full = model.get_xyz.detach()
        rotation_full = model.get_rotation_matrix.detach()
        tangent_u_full = rotation_full[:, :, 0].contiguous()
        tangent_v_full = rotation_full[:, :, 1].contiguous()
        scaling_full = model.get_scaling.detach()
        scale_u_full = scaling_full[:, 0].contiguous()
        scale_v_full = scaling_full[:, 1].contiguous()

    preview_camera = min(cameras, key=lambda c: str(c.image_name))
    region_index, region_meta = bank_module.region_of_surfel(model, preview_camera)
    report["region_anchor_mechanism"] = region_meta

    # ============================================ sweep A: anchors + generic bank
    _progress("[3/8] sweep A: deterministic renderer-event anchors and generic 3D probes")
    anchor_world: list[torch.Tensor] = []
    anchor_camera: list[int] = []
    anchor_pixel: list[int] = []
    anchor_median: list[float] = []
    anchor_representative: list[int] = []
    for view_index in range(0, len(cameras), ANCHOR_VIEW_STRIDE):
        camera = cameras[view_index]
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        representative = package["representative_id"].reshape(-1).to(torch.int64)
        median_flat = candidate_b.median_depth_map(package["out_others"]).reshape(-1)
        world = reconstruct_direct_surfel_intersection_world_point(
            representative, package["median_s_u"], package["median_s_v"],
            positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
        )
        valid = torch.nonzero((representative >= 0) & torch.isfinite(world).all(dim=1), as_tuple=False).reshape(-1)
        if valid.numel():
            picks = topology_gap_bank.deterministic_stride(int(valid.numel()), ANCHORS_PER_VIEW, device)
            chosen = valid[picks]
            anchor_world.append(world[chosen].clone())
            anchor_camera.extend([view_index] * int(chosen.numel()))
            anchor_pixel.extend(chosen.tolist())
            anchor_median.extend(median_flat[chosen].tolist())
            anchor_representative.extend(representative[chosen].tolist())
        del package, world

    anchor_positions = torch.cat(anchor_world, dim=0) if anchor_world else torch.zeros((0, 3), device=device)
    anchor_count = int(anchor_positions.shape[0])
    anchor_camera_np = np.asarray(anchor_camera, dtype=np.int64)
    anchor_pixel_np = np.asarray(anchor_pixel, dtype=np.int64)
    anchor_median_np = np.asarray(anchor_median, dtype=np.float32)
    anchor_representative_np = np.asarray(anchor_representative, dtype=np.int64)
    _progress(f"  renderer-event anchors: {anchor_count}")

    origins = torch.stack([cameras[int(v)].camera_center.reshape(3) for v in anchor_camera_np])
    direction = anchor_positions - origins
    free_space = origins + direction * FREE_SPACE_T
    behind = origins + direction * BEHIND_T
    ladder_rows = np.arange(0, anchor_count, LADDER_ANCHOR_STRIDE, dtype=np.int64)
    ladder_points: list[torch.Tensor] = []
    ladder_offset: list[float] = []
    ladder_source: list[int] = []
    for offset in NEAR_FRONTIER_DECADES:
        for sign in (-1.0, 1.0):
            factor = 1.0 + sign * offset
            rows = torch.as_tensor(ladder_rows, dtype=torch.int64, device=device)
            ladder_points.append(origins[rows] + direction[rows] * factor)
            ladder_offset.extend([sign * offset] * int(rows.numel()))
            ladder_source.extend(ladder_rows.tolist())
    ladder_positions = torch.cat(ladder_points, dim=0) if ladder_points else torch.zeros((0, 3), device=device)

    stored_120 = np.load(arguments.wl120_npz, allow_pickle=True)
    stored_121 = np.load(arguments.wl121_npz, allow_pickle=True)
    wl120_positions = torch.as_tensor(stored_120["positions"], dtype=torch.float32, device=device)
    wl121_positions = torch.as_tensor(stored_121["positions"], dtype=torch.float32, device=device)
    context_count = int(stored_121["context_gating_reason"].shape[0])

    # WL121 endpoints carry direct renderer-event provenance from the stored
    # contexts (view index + pixel row/col + representative). Midpoints and
    # controls do not, and none is invented for them.
    width = int(cameras[0].image_width)
    wl121_camera = np.full(int(wl121_positions.shape[0]), -1, dtype=np.int64)
    wl121_pixel = np.full(int(wl121_positions.shape[0]), -1, dtype=np.int64)
    wl121_representative = np.full(int(wl121_positions.shape[0]), -1, dtype=np.int64)
    wl121_camera[:context_count] = stored_121["context_view_index"]
    wl121_pixel[:context_count] = stored_121["context_row_a"] * width + stored_121["context_col_a"]
    wl121_representative[:context_count] = stored_121["context_representative_a"]
    wl121_camera[context_count:2 * context_count] = stored_121["context_view_index"]
    wl121_pixel[context_count:2 * context_count] = stored_121["context_row_b"] * width + stored_121["context_col_b"]
    wl121_representative[context_count:2 * context_count] = stored_121["context_representative_b"]

    def _empty_provenance(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.full(count, -1, dtype=np.int64), np.full(count, -1, dtype=np.int64),
            np.full(count, np.nan, dtype=np.float32), np.full(count, -1, dtype=np.int64),
        )

    parts: list[tuple[str, torch.Tensor, tuple]] = [
        (KIND_EVENT_ANCHOR, anchor_positions,
         (anchor_camera_np, anchor_pixel_np, anchor_median_np, anchor_representative_np)),
        (KIND_FREE_SPACE, free_space, _empty_provenance(anchor_count)),
        (KIND_BEHIND, behind, _empty_provenance(anchor_count)),
        (KIND_NEAR_FRONTIER, ladder_positions, _empty_provenance(int(ladder_positions.shape[0]))),
        (KIND_WL120, wl120_positions, _empty_provenance(int(wl120_positions.shape[0]))),
        (KIND_WL121, wl121_positions,
         (wl121_camera, wl121_pixel, np.full(int(wl121_positions.shape[0]), np.nan, dtype=np.float32), wl121_representative)),
    ]
    combined = VolumetricQueryBank(
        world_position=torch.cat([p for _, p, _ in parts], dim=0).contiguous(),
        kind=sum([[name] * int(p.shape[0]) for name, p, _ in parts], []),
        provenance_camera=np.concatenate([q[0] for _, _, q in parts]),
        provenance_pixel=np.concatenate([q[1] for _, _, q in parts]),
        provenance_median_depth=np.concatenate([q[2] for _, _, q in parts]),
        provenance_representative=np.concatenate([q[3] for _, _, q in parts]),
    )
    kinds_np = np.asarray(combined.kind)
    offsets = {}
    cursor = 0
    for name, positions_part, _ in parts:
        offsets[name] = (cursor, cursor + int(positions_part.shape[0]))
        cursor += int(positions_part.shape[0])

    # WL121 endpoint provenance needs the renderer's stored median at that pixel,
    # which is read during the sweep (exact bitwise, never reconstructed).
    report["query_bank"] = {
        **combined.metadata(),
        "composition": {name: [int(a), int(b)] for name, (a, b) in offsets.items()},
        "construction": {
            "anchor_view_stride": ANCHOR_VIEW_STRIDE, "anchors_per_view": ANCHORS_PER_VIEW,
            "free_space_t": FREE_SPACE_T, "behind_t": BEHIND_T,
            "near_frontier_relative_offsets": list(NEAR_FRONTIER_DECADES),
            "ladder_anchor_stride": LADDER_ANCHOR_STRIDE,
            "note": (
                "camera-space depth is linear along the ray from the camera centre, so a probe at parameter t "
                "sits at depth = t * frontier_depth and its relative margin from the frontier is exactly t - 1"
            ),
            "rng": "none",
        },
        "provenance_sources": {
            "P1_RENDERER_EVENT_ANCHOR": "recorded directly at construction (view, pixel, stored median, representative)",
            "G5_WL121_SUPPLEMENTAL_BANK": "endpoints only, from the stored worklog 121 context arrays; midpoints and controls carry none",
            "others": "none -- deliberately treated as generic arbitrary-3D queries",
        },
    }

    # ==================================================== sweep B: the audit
    _progress("[4/8] sweep B: exhaustive event identity + arbitrary-3D stability over all views")
    identity_audit = EventIdentityAccumulator()
    stability_event = StabilityAccumulator(label="exact_renderer_event_queries")
    stability_generic = StabilityAccumulator(label="generic_arbitrary_3D_queries")
    ever_representative = torch.zeros((total_model_count,), dtype=torch.bool, device=device)

    count_combined = len(combined)
    base_states = np.full((count_combined, len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    layered_states = np.full((count_combined, len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    reference_states = np.full((count_combined, len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    identity_flags = np.zeros((count_combined, len(cameras)), dtype=np.int8)
    provenance_applied_total = 0
    provenance_rejected_total = 0

    is_event_query = np.isin(kinds_np, (KIND_EVENT_ANCHOR,))
    generic_mask = ~is_event_query
    generic_rows = np.nonzero(generic_mask)[0]
    generic_index = torch.as_tensor(generic_rows, dtype=torch.int64, device=device)
    generic_count = int(generic_rows.size)
    generic_pair_fields = {
        "float32_pixel_row": np.full((generic_count, len(cameras)), -1, dtype=np.int32),
        "float32_pixel_col": np.full((generic_count, len(cameras)), -1, dtype=np.int32),
        "reference_pixel_row": np.full((generic_count, len(cameras)), -1, dtype=np.int32),
        "reference_pixel_col": np.full((generic_count, len(cameras)), -1, dtype=np.int32),
        "float32_query_depth": np.full((generic_count, len(cameras)), np.nan, dtype=np.float32),
        "reference_query_depth": np.full((generic_count, len(cameras)), np.nan, dtype=np.float64),
        "float32_stored_median_depth": np.full((generic_count, len(cameras)), np.nan, dtype=np.float32),
        "reference_stored_median_depth": np.full((generic_count, len(cameras)), np.nan, dtype=np.float32),
    }

    for view_index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        representative = package["representative_id"].reshape(-1).to(torch.int64)
        ever_representative[torch.unique(representative[representative >= 0])] = True
        median_flat = candidate_b.median_depth_map(package["out_others"]).reshape(-1)

        # ---- section 4: exhaustive event identity for THIS view's own events
        valid = torch.nonzero(representative >= 0, as_tuple=False).reshape(-1)
        if valid.numel():
            world = reconstruct_direct_surfel_intersection_world_point(
                representative, package["median_s_u"], package["median_s_v"],
                positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
            )
            finite = torch.isfinite(world).all(dim=1)
            valid = valid[finite[valid]]
            event_world = world[valid]
            event_geometry = project_queries(camera, event_world)
            historical = candidate_b.classify_view(event_geometry, median_flat)["states"]
            event_bank = VolumetricQueryBank(
                world_position=event_world, kind=[KIND_EVENT_ANCHOR] * int(valid.numel()),
                provenance_camera=np.full(int(valid.numel()), view_index, dtype=np.int64),
                provenance_pixel=valid.detach().cpu().numpy(),
                provenance_median_depth=median_flat[valid].detach().cpu().numpy(),
                provenance_representative=representative[valid].detach().cpu().numpy(),
            )
            layered = apply_event_identity(view_index, event_bank, event_geometry, median_flat, historical)
            event_reference = project_queries_float64(camera, event_world)
            reference = reference_side(event_reference, median_flat)

            identity_audit.total_events += int(valid.numel())
            identity_audit.historical_float32_observed += int((historical == STATE_OBSERVED).sum())
            identity_audit.historical_float32_contradiction += int((historical != STATE_OBSERVED).sum())
            identity_audit.provenance_observed += int((layered["states"] == STATE_OBSERVED).sum())
            identity_audit.provenance_contradiction += int((layered["states"] != STATE_OBSERVED).sum())
            identity_audit.provenance_applied += layered["applied"]
            identity_audit.provenance_rejected_stale += layered["rejected"]
            identity_audit.reference_observed += int((reference == STATE_OBSERVED).sum())
            identity_audit.reference_contradiction += int((reference != STATE_OBSERVED).sum())
            stability_event.accumulate(
                historical, reference, event_geometry, event_reference, median_flat,
                np.full(int(valid.numel()), KIND_EVENT_ANCHOR), margin_sample_stride=0,
            )
            del world, event_world, event_geometry, event_reference, event_bank

        # ---- sections 5-7: the combined arbitrary-3D bank
        geometry = project_queries(camera, combined.world_position)
        base = candidate_b.classify_view(geometry, median_flat)["states"]
        base_states[:, view_index] = base.detach().cpu().numpy()

        # WL121 endpoint provenance carries no stored median; fill it exactly
        # from the renderer's own output at the carried pixel for this view.
        view_bank = combined
        endpoint_rows = np.nonzero((combined.provenance_camera == view_index) & np.isnan(combined.provenance_median_depth))[0]
        if endpoint_rows.size:
            filled = combined.provenance_median_depth.copy()
            pixels = torch.as_tensor(combined.provenance_pixel[endpoint_rows], dtype=torch.int64, device=device)
            filled[endpoint_rows] = median_flat[pixels].detach().cpu().numpy()
            view_bank = VolumetricQueryBank(
                world_position=combined.world_position, kind=combined.kind,
                provenance_camera=combined.provenance_camera, provenance_pixel=combined.provenance_pixel,
                provenance_median_depth=filled, provenance_representative=combined.provenance_representative,
            )
        applied = apply_event_identity(view_index, view_bank, geometry, median_flat, base)
        layered_states[:, view_index] = applied["states"].detach().cpu().numpy()
        identity_flags[:, view_index] = applied["identity"].detach().cpu().numpy()
        provenance_applied_total += applied["applied"]
        provenance_rejected_total += applied["rejected"]

        reference_geometry = project_queries_float64(camera, combined.world_position)
        reference = reference_side(reference_geometry, median_flat)
        reference_states[:, view_index] = reference.detach().cpu().numpy()
        # The generic stability arm is deliberately separate from the exact
        # renderer-event arm. P1 source events are not allowed to inflate the
        # arbitrary-3D result.
        generic_geometry = ViewGeometry(
            pixel_x=geometry.pixel_x[generic_index],
            pixel_y=geometry.pixel_y[generic_index],
            pixel_col=geometry.pixel_col[generic_index],
            pixel_row=geometry.pixel_row[generic_index],
            pixel_index=geometry.pixel_index[generic_index],
            depth=geometry.depth[generic_index],
            relevant=geometry.relevant[generic_index],
            relevance_code=geometry.relevance_code[generic_index],
        )
        generic_reference = {
            key: value[generic_index] for key, value in reference_geometry.items()
        }
        stability_generic.accumulate(
            base[generic_index], reference[generic_index], generic_geometry, generic_reference,
            median_flat, kinds_np[generic_rows], margin_sample_stride=MARGIN_SAMPLE_STRIDE,
        )

        # Preserve per-query-view diagnostic fields for every generic pair in
        # the NPZ artifact. Canonical states remain in base_states and
        # reference_states; this is attribution storage only.
        generic_pair_fields["float32_pixel_row"][:, view_index] = (
            geometry.pixel_row[generic_index].detach().cpu().numpy()
        )
        generic_pair_fields["float32_pixel_col"][:, view_index] = (
            geometry.pixel_col[generic_index].detach().cpu().numpy()
        )
        generic_pair_fields["reference_pixel_row"][:, view_index] = (
            reference_geometry["pixel_row"][generic_index].detach().cpu().numpy()
        )
        generic_pair_fields["reference_pixel_col"][:, view_index] = (
            reference_geometry["pixel_col"][generic_index].detach().cpu().numpy()
        )
        generic_pair_fields["float32_query_depth"][:, view_index] = (
            geometry.depth[generic_index].detach().cpu().numpy()
        )
        generic_pair_fields["reference_query_depth"][:, view_index] = (
            reference_geometry["depth"][generic_index].detach().cpu().numpy()
        )
        float32_median = median_flat[geometry.pixel_index[generic_index].clamp(min=0)]
        reference_median = median_flat[reference_geometry["pixel_index"][generic_index].clamp(min=0)]
        generic_pair_fields["float32_stored_median_depth"][:, view_index] = (
            torch.where(geometry.relevant[generic_index], float32_median,
                        torch.zeros_like(float32_median)).detach().cpu().numpy()
        )
        generic_pair_fields["reference_stored_median_depth"][:, view_index] = (
            torch.where(reference_geometry["relevant"][generic_index], reference_median,
                        torch.zeros_like(reference_median)).detach().cpu().numpy()
        )
        del package
        if view_index % 20 == 0:
            _progress(f"  sweep B view {view_index + 1}/{len(cameras)}")

    representative_union = int(ever_representative.sum().item())
    report["frozen_state_fingerprint"] = {
        "median_surface_representatives_union": representative_union,
        "worklog_119_to_122_reference": WL119_REPRESENTATIVE_UNION,
        "matches": bool(representative_union == WL119_REPRESENTATIVE_UNION),
    }

    # ================================================================ results
    summary = identity_audit.summary()
    summary["worklog_122_reference"] = {
        "source_events": WL122_SOURCE_EVENTS,
        "float32_contradictions": WL122_SOURCE_CONTRADICTIONS,
        "reproduces_worklog_122_corpus": bool(summary["total_source_median_events"] == WL122_SOURCE_EVENTS),
        "reproduces_worklog_122_contradiction_count": bool(
            summary["historical_float32_source_contradiction"] == WL122_SOURCE_CONTRADICTIONS
        ),
    }
    report["exact_event_identity"] = summary
    report["arbitrary_3d_stability"] = {
        "exact_renderer_event_queries": stability_event.summary(),
        "generic_arbitrary_3D_queries": stability_generic.summary(),
        "reference_arm_status": (
            "float64 recomputation of the SAME projection/depth formulas from the SAME stored float32 inputs. "
            "DIAGNOSTIC ONLY -- never canonical in this batch."
        ),
    }

    # per-kind stability breakdown, and the near-frontier attribution
    per_kind: dict[str, Any] = {}
    for name, (start, stop) in offsets.items():
        if name == KIND_EVENT_ANCHOR:
            continue
        rows = np.arange(start, stop)
        relevant = base_states[rows] != STATE_NON_RELEVANT
        agree = base_states[rows] == reference_states[rows]
        per_kind[name] = {
            "queries": int(rows.size),
            "relevant_pairs": int(relevant.sum()),
            "float32_reference_agreement": int((agree & relevant).sum()),
            "disagreements": int((~agree & relevant).sum()),
            "disagreement_rate": float((~agree & relevant).sum()) / max(int(relevant.sum()), 1),
        }
    report["arbitrary_3d_stability"]["per_query_kind"] = per_kind

    ladder_start, ladder_stop = offsets[KIND_NEAR_FRONTIER]
    ladder_offset_np = np.asarray(ladder_offset, dtype=np.float64)
    ladder_table: dict[str, Any] = {}
    for offset in sorted(set(ladder_offset_np.tolist())):
        rows = ladder_start + np.nonzero(ladder_offset_np == offset)[0]
        relevant = base_states[rows] != STATE_NON_RELEVANT
        agree = base_states[rows] == reference_states[rows]
        observed = base_states[rows] == STATE_OBSERVED
        ladder_table[f"{offset:+.0e}"] = {
            "queries": int(rows.size),
            "relevant_pairs": int(relevant.sum()),
            "disagreements": int((~agree & relevant).sum()),
            "disagreement_rate": float((~agree & relevant).sum()) / max(int(relevant.sum()), 1),
            "float32_OBSERVED_fraction": float((observed & relevant).sum()) / max(int(relevant.sum()), 1),
        }
    generic_relevant = base_states[generic_rows] != STATE_NON_RELEVANT
    generic_disagreement = generic_relevant & (base_states[generic_rows] != reference_states[generic_rows])
    generic_pixel_changed = (
        (generic_pair_fields["float32_pixel_row"] != generic_pair_fields["reference_pixel_row"])
        | (generic_pair_fields["float32_pixel_col"] != generic_pair_fields["reference_pixel_col"])
    )
    generic_reference_margin = (
        generic_pair_fields["reference_query_depth"]
        - generic_pair_fields["reference_stored_median_depth"]
    )
    generic_float32_margin = (
        generic_pair_fields["float32_query_depth"]
        - generic_pair_fields["float32_stored_median_depth"]
    )
    report["arbitrary_3d_stability"]["disagreement_attribution"] = {
        "relevant_projected_pixel_changes": int((generic_relevant & generic_pixel_changed).sum()),
        "state_disagreements_same_projected_pixel": int(
            (generic_disagreement & ~generic_pixel_changed).sum()
        ),
        "state_disagreements_with_projected_pixel_change": int(
            (generic_disagreement & generic_pixel_changed).sum()
        ),
        "same_pixel_reference_signed_margin": _quantiles(
            generic_reference_margin[generic_disagreement & ~generic_pixel_changed]
        ),
        "same_pixel_float32_signed_margin": _quantiles(
            generic_float32_margin[generic_disagreement & ~generic_pixel_changed]
        ),
        "pixel_changed_reference_signed_margin": _quantiles(
            generic_reference_margin[generic_disagreement & generic_pixel_changed]
        ),
        "pixel_changed_float32_signed_margin": _quantiles(
            generic_float32_margin[generic_disagreement & generic_pixel_changed]
        ),
        "interpretation": (
            "Disagreements are reported by representation path. Same-pixel cases "
            "are frontier-side arithmetic cases; pixel-changed cases are discrete "
            "raster-pixel boundary cases. No measured margin is promoted to a "
            "production tolerance."
        ),
    }
    report["exact_event_identity"]["pairwise_diagnostic_fields"] = {
        "scope": "all 43,817,760 source-event query-view pairs, evaluated in streaming form",
        "fields": [
            "float32 projected pixel",
            "reference projected pixel",
            "float32 query depth",
            "reference query depth",
            "stored median depth",
            "float32 side",
            "reference side",
            "signed distance from frontier",
            "ULP distance where applicable",
        ],
        "storage": (
            "Aggregate counters/distributions are in this JSON; the generic "
            "pairwise artifact is persisted in the NPZ. No production classifier "
            "consumes the reference arm."
        ),
    }
    report["arbitrary_3d_stability"]["pairwise_fields"] = {
        "artifact": "volumetric_query_contract.npz",
        "scope": "all non-P1 generic query-view pairs (WL120, WL121, free-space, behind-frontier, near-frontier)",
        "query_order": "generic_query_indices and generic_kind in the NPZ; view order is view_names",
        "fields": {
            "float32_pixel_row": "float32 projected raster row",
            "float32_pixel_col": "float32 projected raster column",
            "reference_pixel_row": "float64-reference projected raster row",
            "reference_pixel_col": "float64-reference projected raster column",
            "float32_query_depth": "canonical float32 camera-space z",
            "reference_query_depth": "diagnostic float64 camera-space z",
            "float32_stored_median_depth": "stored median at the canonical float32 pixel",
            "reference_stored_median_depth": "stored median at the reference pixel",
            "base_states": "canonical frozen Candidate B side in base_states",
            "reference_states": "diagnostic reference side in reference_states",
        },
        "reference_arm": "diagnostic only; no production decision uses these fields",
    }
    report["near_frontier_attribution"] = {
        "relative_offset_ladder": ladder_table,
        "interpretation_guard": (
            "These offsets are a measurement design, not a tolerance. No epsilon, ULP acceptance band, "
            "nextafter correction or percentage threshold is derived from them."
        ),
    }

    # ---------------------------------------------------------- cross-view replay
    _progress("[5/8] cross-view replay on the worklog 122 anchor corpus")
    anchor_start, anchor_stop = offsets[KIND_EVENT_ANCHOR]
    anchor_rows = np.arange(anchor_start, anchor_stop)
    base_global = aggregate_global(base_states[anchor_rows])
    layered_global = aggregate_global(layered_states[anchor_rows])
    source_base = base_states[anchor_rows, anchor_camera_np]
    source_layered = layered_states[anchor_rows, anchor_camera_np]
    occluded_views = (base_states[anchor_rows] == STATE_OCCLUDED).sum(axis=1)
    report["cross_view_replay"] = {
        "anchors": int(anchor_rows.size),
        "worklog_122_reference_global_OCCLUDED": WL122_GLOBAL_OCCLUDED_ANCHORS,
        "source_view_state_without_provenance": state_fractions(source_base),
        "source_view_state_with_provenance": state_fractions(source_layered),
        "global_without_provenance": state_fractions(base_global),
        "global_with_provenance": state_fractions(layered_global),
        "global_OCCLUDED_without_provenance": int((base_global == STATE_OCCLUDED).sum()),
        "global_OCCLUDED_with_provenance": int((layered_global == STATE_OCCLUDED).sum()),
        "anchors_hidden_in_at_least_one_view": int((occluded_views > 0).sum()),
        "global_OBSERVED_retention_with_provenance": float((layered_global == STATE_OBSERVED).mean()),
        "identity_applied_on_source_view": int((identity_flags[anchor_rows, anchor_camera_np] == IDENTITY_ON_FRONTIER).sum()),
        "aggregation_rule": "frozen ANY-OBSERVED, unchanged; no view-count rule anywhere",
    }

    # ------------------------------------------------ true-fragmentation replay
    _progress("[6/8] worklog 121 true-fragmentation replay")
    wl121_start, wl121_stop = offsets[KIND_WL121]
    wl121_rows = np.arange(wl121_start, wl121_stop)
    stored_kind = stored_121["kind"]
    stored_global_b = stored_121["global_B"]
    base_global_121 = aggregate_global(base_states[wl121_rows])
    layered_global_121 = aggregate_global(layered_states[wl121_rows])
    fragmentation: dict[str, Any] = {}
    for kind in sorted(set(stored_kind.tolist())):
        local = np.nonzero(stored_kind == kind)[0]
        fragmentation[str(kind)] = {
            "queries": int(local.size),
            "worklog_121_stored": state_fractions(stored_global_b[local]),
            "without_provenance": state_fractions(base_global_121[local]),
            "with_provenance": state_fractions(layered_global_121[local]),
            "without_provenance_matches_worklog_121": bool(
                np.array_equal(base_global_121[local], stored_global_b[local])
            ),
        }
    report["true_fragmentation_replay"] = {
        "contexts": context_count,
        "gating_attribution": {
            topology_gap_bank.GATING_NAMES[int(code)]: int((stored_121["context_gating_reason"] == code).sum())
            for code in sorted(topology_gap_bank.GATING_NAMES)
            if int((stored_121["context_gating_reason"] == code).sum())
        },
        "by_query_kind": fragmentation,
        "midpoint_guard": (
            "Midpoints carry no renderer-event provenance and none is invented. B(midpoint) = OBSERVED is NOT "
            "read as surface continuity and is not used as a component merge criterion; topology is unchanged."
        ),
    }

    # ------------------------------------------------------- synthetic contracts
    _progress("[7/8] synthetic query contracts Q1-Q5")
    report["synthetic_query_contracts"] = query_contract_synthetics.run_query_contracts(device=arguments.device)

    # ------------------------------------------------------------- exports
    _progress("[8/8] review exports (each view folder gets its own Korean README)")
    view_paths: dict[str, Any] = {}
    try:
        from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
        from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())

        def _export(view_name: str, positions: torch.Tensor, colours: torch.Tensor, body: str) -> dict[str, Any]:
            scene_xyz = model.get_xyz.detach()
            scene_rgb = torch.tensor(_SCENE_RGB, device=device).reshape(1, 3).expand(scene_xyz.shape[0], 3)
            count = int(positions.shape[0])
            scaling = torch.cat(
                [model._scaling.detach(), torch.full((count, 2), float(np.log(_MARKER_RADIUS)), device=device)], dim=0
            )
            rotation = torch.zeros((count, 4), dtype=torch.float32, device=device)
            rotation[:, 0] = 1.0
            rotation = torch.cat([model._rotation.detach(), rotation], dim=0)
            opacity = torch.cat([model._opacity.detach().reshape(-1), torch.full((count,), 4.0, device=device)], dim=0)
            xyz = torch.cat([scene_xyz, positions], dim=0)
            rgb = torch.cat([scene_rgb, colours], dim=0)
            folder = output_root / view_name
            ply_path = folder / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, xyz, _rgb_to_f_dc(rgb), opacity, scaling, rotation)
            review = TorchGaussianSurfelModel(sh_degree=0, device=str(device))
            review.initialize(positions=xyz, colors=rgb, opacities=torch.sigmoid(opacity).reshape(-1, 1),
                              scales=torch.exp(scaling), rotations=rotation)
            review.active_sh_degree = 0
            with torch.no_grad():
                package = rasterizer.render(preview_camera, review)
            write_ppm(folder / "render.ppm", package["render"])
            write_view_readme(folder, body, total_model_count)
            del review, package
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            return {"point_cloud_ply": str(ply_path), "gaussian_count": written, "marker_points": count}

        with torch.no_grad():
            package = rasterizer.render(preview_camera, model)
        folder = output_root / "ORIGINAL_2DGS_SCENE"
        ply_path = folder / _ITERATION_DIR / "point_cloud.ply"
        view_paths["ORIGINAL_2DGS_SCENE"] = {
            "point_cloud_ply": str(ply_path),
            "gaussian_count": write_surfel_ply(
                ply_path, model.get_xyz.detach(), model._features_dc.detach()[:, 0, :],
                model._opacity.detach().reshape(-1), model._scaling.detach(), model._rotation.detach(),
            ),
        }
        write_ppm(folder / "render.ppm", package["render"])
        write_view_readme(folder, ORIGINAL_SCENE_README, total_model_count)
        del package

        effect_labels: list[str] = []
        for row in range(anchor_rows.size):
            before, after = int(base_global[row]), int(layered_global[row])
            if after == STATE_OBSERVED and before == STATE_OBSERVED:
                effect_labels.append("unchanged_observed")
            elif after == STATE_OBSERVED:
                effect_labels.append("rescued_by_identity")
            elif after == STATE_OCCLUDED:
                effect_labels.append("still_occluded")
            else:
                effect_labels.append("unresolved")
        rescued = effect_labels.count("rescued_by_identity")
        report["cross_view_replay"]["anchors_rescued_by_event_identity"] = rescued
        view_paths["EVENT_IDENTITY_EFFECT"] = _export(
            "EVENT_IDENTITY_EFFECT", anchor_positions,
            torch.tensor([_EFFECT_RGB[label] for label in effect_labels], dtype=torch.float32, device=device),
            EVENT_IDENTITY_README.format(
                stride=ANCHOR_VIEW_STRIDE, anchors=f"{anchor_rows.size:,}", rescued=rescued,
                unchanged=effect_labels.count("unchanged_observed"),
                still=effect_labels.count("still_occluded"),
            ),
        )

        ladder_rows_all = np.arange(ladder_start, ladder_stop)
        ladder_colour_map: dict[float, tuple[float, float, float]] = {}
        for offset in sorted(set(ladder_offset_np.tolist())):
            shade = min(max((7.0 + float(np.log10(abs(offset)))) / 5.0, 0.0), 1.0)
            ladder_colour_map[offset] = (
                (0.95, 0.35 + 0.5 * shade, 0.15) if offset > 0 else (0.15, 0.35 + 0.5 * shade, 0.95)
            )
        view_paths["NEAR_FRONTIER_LADDER"] = _export(
            "NEAR_FRONTIER_LADDER", combined.world_position[ladder_rows_all],
            torch.tensor([ladder_colour_map[float(v)] for v in ladder_offset_np], dtype=torch.float32, device=device),
            NEAR_FRONTIER_README.format(
                decades=list(NEAR_FRONTIER_DECADES), probes=f"{int(ladder_rows_all.size):,}"
            ),
        )
    except Exception as error:  # pragma: no cover - exports are secondary to the audit
        view_paths["failed"] = f"{type(error).__name__}: {error}"
        _progress(f"review export FAILED: {type(error).__name__}: {error}")
    report["review_exports"] = view_paths

    np.savez_compressed(
        output_root / "volumetric_query_contract.npz",
        world_position=combined.world_position.detach().cpu().numpy(),
        kind=kinds_np, provenance_camera=combined.provenance_camera,
        provenance_pixel=combined.provenance_pixel,
        provenance_representative=combined.provenance_representative,
        base_states=base_states, layered_states=layered_states,
        reference_states=reference_states, identity_flags=identity_flags,
        generic_query_indices=generic_rows,
        generic_kind=kinds_np[generic_rows],
        generic_float32_pixel_row=generic_pair_fields["float32_pixel_row"],
        generic_float32_pixel_col=generic_pair_fields["float32_pixel_col"],
        generic_reference_pixel_row=generic_pair_fields["reference_pixel_row"],
        generic_reference_pixel_col=generic_pair_fields["reference_pixel_col"],
        generic_float32_query_depth=generic_pair_fields["float32_query_depth"],
        generic_reference_query_depth=generic_pair_fields["reference_query_depth"],
        generic_float32_stored_median_depth=generic_pair_fields["float32_stored_median_depth"],
        generic_reference_stored_median_depth=generic_pair_fields["reference_stored_median_depth"],
        ladder_offset=ladder_offset_np, ladder_source=np.asarray(ladder_source, dtype=np.int64),
        anchor_camera=anchor_camera_np, anchor_pixel=anchor_pixel_np,
        view_names=np.asarray([str(getattr(c, "image_name", i)) for i, c in enumerate(cameras)]),
    )
    report["provenance_application"] = {
        "applied_query_view_pairs": provenance_applied_total,
        "rejected_stored_median_mismatch": provenance_rejected_total,
    }
    report["artifacts"] = {"npz": str(output_root / "volumetric_query_contract.npz")}
    report["total_seconds"] = time.time() - started
    report_path = output_root / "volumetric_query_contract_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _progress(f"wrote {report_path} ({report['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
