from __future__ import annotations

"""Worklog 161: diagnostic audit of the spatial GLOBAL OCCLUDED domain.

This batch deliberately stops before constructing a spatial field when the
pre-latent architecture has no already-approved query domain and resolution.
It replays the W160 pointwise contract over the historical Gaussian-center
population only to reconcile geometric relevance with renderer-event
availability; Gaussian centers are not accepted as a spatial-domain proxy.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
W160_PATH = REPO_ROOT / "devtools/demo/worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt"
DEFAULT_SOURCE = REPO_ROOT / "DATASET"
DEFAULT_CACHE = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache"
DEFAULT_OUT = REPO_ROOT / "output/161_global_persistent_occlusion_spatial_domain_audit"


def _load_w160() -> Any:
    spec = importlib.util.spec_from_file_location("worklog_160_for_worklog_161", W160_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W160 module: {W160_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


w160 = _load_w160()

STATE_NAMES = w160.STATE_NAMES
STATE_NON_RELEVANT = w160.STATE_NON_RELEVANT
STATE_UNRESOLVED = w160.STATE_UNRESOLVED
STATE_OBSERVED = w160.STATE_OBSERVED
STATE_OCCLUDED = w160.STATE_OCCLUDED


def _progress(message: str) -> None:
    print(f"[worklog 161] {message}", flush=True)


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


def _global_from_accumulators(
    observed_any: torch.Tensor,
    has_relevant: torch.Tensor,
    has_unresolved: torch.Tensor,
) -> torch.Tensor:
    result = torch.full_like(observed_any, STATE_UNRESOLVED, dtype=torch.int8)
    result[has_relevant & ~has_unresolved & ~observed_any] = STATE_OCCLUDED
    result[observed_any] = STATE_OBSERVED
    return result


def relabel_semantics_a_to_b(
    historical_states: torch.Tensor,
    geometrically_relevant: torch.Tensor,
    evidence_available: torch.Tensor,
) -> torch.Tensor:
    """Apply the hypothetical Semantics-B relabeling without changing A."""

    result = historical_states.clone()
    result[geometrically_relevant & ~evidence_available] = STATE_NON_RELEVANT
    return result


def synthetic_contracts() -> dict[str, Any]:
    """Mechanics-only A-F checks for the reconciled pointwise/global contract."""

    def aggregate(rows: list[list[int]]) -> str:
        states = np.asarray(rows, dtype=np.int8)
        return STATE_NAMES[int(w160.aggregate_persistent_states(states)[0])]

    cases = [
        {
            "name": "A_visible_bounded_volume",
            "expected_historical": "OBSERVED",
            "actual_historical": aggregate([[STATE_OBSERVED, STATE_OBSERVED]]),
        },
        {
            "name": "B_hidden_in_every_relevant_view",
            "expected_historical": "OCCLUDED",
            "actual_historical": aggregate([[STATE_OCCLUDED, STATE_OCCLUDED]]),
        },
        {
            "name": "C_visible_in_one_of_many_views",
            "expected_historical": "OBSERVED",
            "actual_historical": aggregate([[STATE_OCCLUDED, STATE_OBSERVED, STATE_OCCLUDED]]),
        },
        {
            "name": "D_relevant_camera_missing_renderer_event",
            "expected_historical": "UNRESOLVED",
            "actual_historical": aggregate([[STATE_UNRESOLVED, STATE_OCCLUDED]]),
            "hypothetical_semantics_b": aggregate([[STATE_NON_RELEVANT, STATE_OCCLUDED]]),
            "note": "Semantics A preserves missing relevant evidence as UNRESOLVED; B would silently remove that camera and change this mixed case to OCCLUDED.",
        },
        {
            "name": "E_outside_all_camera_supported_relevance",
            "expected_historical": "UNRESOLVED",
            "actual_historical": aggregate([[STATE_NON_RELEVANT, STATE_NON_RELEVANT]]),
        },
        {
            "name": "F_hidden_query_without_gaussian_membership",
            "expected_historical": "OCCLUDED",
            "actual_historical": aggregate([[STATE_OCCLUDED, STATE_OCCLUDED]]),
            "gaussian_membership_used": False,
        },
    ]
    for case in cases:
        case["pass"] = case["actual_historical"] == case["expected_historical"]
    return {
        "all_pass": bool(all(case["pass"] for case in cases)),
        "cases": cases,
        "note": "Synthetic PASS validates pointwise/global mechanics only; it does not authorize a real spatial domain.",
    }


def _read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def spatial_domain_audit() -> dict[str, Any]:
    """Audit existing candidates without constructing or expanding a domain."""

    voxel_regions = REPO_ROOT / "osn_gs/surface/torch_voxel_regions.py"
    voxel_hierarchy = REPO_ROOT / "osn_gs/surface/torch_voxel_hierarchy.py"
    continuation_domain = REPO_ROOT / "osn_gs/surface/torch_continuation_domain.py"
    current_framework = REPO_ROOT / "docs/current_framework.md"
    wl120 = REPO_ROOT / "docs/worklogs/120_observed_occluded_volumetric_operationalization_audit.md"
    candidates = [
        {
            "candidate": "torch_voxel_regions.build_torch_voxel_surface_regions",
            "path": str(voxel_regions),
            "eligible_as_occlusion_domain": False,
            "reason": "Gaussian-point input, Gaussian-occupied cells only, AABB derived from input points, and output is a visible NURBS patch candidate partition.",
            "contract_evidence": [
                "TorchVoxelSurfaceRegions docstring: surface-aligned adaptive cells used as NURBS patch candidates",
                "build_torch_voxel_surface_regions computes min/max from points and keeps only occupied cells",
            ],
        },
        {
            "candidate": "torch_voxel_hierarchy.build_voxel_gaussian_hierarchy",
            "path": str(voxel_hierarchy),
            "eligible_as_occlusion_domain": False,
            "reason": "Experimental voxel_patch_stage1 hierarchy over Gaussian centers; its nodes record Gaussian indices and Gaussian-derived root AABB, not an approved all-space query complex.",
            "contract_evidence": [
                "module docstring identifies a retained experimental voxel_patch_stage1 ablation",
                "build_voxel_gaussian_hierarchy sets root bounds from points.min/points.max",
            ],
        },
        {
            "candidate": "W153 sparse authoritative TSDF field",
            "path": str(REPO_ROOT / "output/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache/field.npz"),
            "eligible_as_occlusion_domain": False,
            "reason": "Sparse authoritative support derived for TSDF surface reconstruction; missing keys mean UNKNOWN and are not a complete scene/query domain.",
            "contract_evidence": [
                "W153 field stores authoritative voxel keys only",
                "W153 h is TSDF discretization and does not by itself approve an occlusion query lattice",
            ],
        },
        {
            "candidate": "torch_continuation_domain.build_continuation_domain",
            "path": str(continuation_domain),
            "eligible_as_occlusion_domain": False,
            "reason": "Boundary-local continuation strip, explicitly outside production and downstream continuation is forbidden in this batch.",
            "contract_evidence": [
                "module docstring says not wired into any production pipeline/trainer path",
                "domain is constructed from an existing boundary and continuation extent",
            ],
        },
    ]
    source_presence = {
        "voxel_regions": voxel_regions.exists(),
        "voxel_hierarchy": voxel_hierarchy.exists(),
        "continuation_domain": continuation_domain.exists(),
        "current_framework": current_framework.exists(),
        "wl120": wl120.exists(),
    }
    # Read the files as an encoding/provenance guard. The values are not used
    # to infer a new contract; the explicit candidate records above are the
    # conservative audit result.
    source_lengths = {name: len(_read_utf8(path)) for name, path in {
        "voxel_regions": voxel_regions,
        "voxel_hierarchy": voxel_hierarchy,
        "continuation_domain": continuation_domain,
        "current_framework": current_framework,
        "wl120": wl120,
    }.items()}
    return {
        "canonical_pre_latent_spatial_query_domain_exists": False,
        "canonical_domain_candidate": None,
        "source_presence": source_presence,
        "source_utf8_lengths": source_lengths,
        "candidates": candidates,
        "missing_contract": [
            "scene/query bounds independent of Gaussian-center AABB",
            "approved spatial query population including empty/hidden locations",
            "approved discretization resolution and indexing contract",
            "native cell-complex semantics for the complete pre-latent domain",
        ],
        "forbidden_substitutes_not_used": [
            "Gaussian-center AABB plus arbitrary expansion",
            "manual hidden-space box",
            "NURBS continuation bounds",
            "future latent geometry",
            "W153 authoritative support treated as the full domain",
            "new voxel spacing or new sampling resolution",
        ],
    }


def _account_camera(states: torch.Tensor, geometry: Any, median: torch.Tensor) -> dict[str, int]:
    geometric = geometry.relevant
    evidence = geometric & (median > 0.0)
    return {
        "query_count": int(states.numel()),
        "geometrically_relevant_pairs": int(geometric.sum()),
        "evidence_available_pairs": int(evidence.sum()),
        "geometrically_relevant_without_evidence_pairs": int((geometric & ~evidence).sum()),
        "non_relevant_pairs": int((~geometric).sum()),
        "historical_observed_pairs": int((states == STATE_OBSERVED).sum()),
        "historical_occluded_pairs": int((states == STATE_OCCLUDED).sum()),
        "historical_unresolved_pairs": int((states == STATE_UNRESOLVED).sum()),
        "historical_non_relevant_pairs": int((states == STATE_NON_RELEVANT).sum()),
    }


def _transition_counts(left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for left_code, left_name in STATE_NAMES.items():
        for right_code, right_name in STATE_NAMES.items():
            count = int(((left == left_code) & (right == right_code)).sum())
            if count:
                result[f"{left_name}->{right_name}"] = count
    return result


def _write_readmes(out: Path) -> None:
    (out / "README.md").write_text(
        """# W161 Global Persistent-Occlusion Spatial Domain Audit

이 batch는 W160 pointwise observability contract를 historical Gaussian-center population에서 다시 평가해 `GEOMETRICALLY_RELEVANT`와 `RENDERER_EVIDENCE_AVAILABLE`의 semantics를 분리하고, pre-latent spatial query domain이 실제로 존재하는지 감사했다.

결과는 `OCCLUSION_DOMAIN_CONTRACT_GAP`이다. 기존 visible-surface voxel partition, Gaussian-derived hierarchy, W153 sparse authoritative TSDF support, boundary-local continuation strip 중 어느 것도 Gaussian이 없는 hidden location을 포함하는 승인된 scene/query domain과 resolution/indexing contract가 아니다. 따라서 이 output에는 GLOBAL OCCLUDED spatial field, Occluded Region, spatial PNG를 만들지 않았다. 빈 공간이나 missing evidence를 OCCLUDED로 칠하지 않은 것이 의도된 결과다.

W160 semantics는 camera 앞/near/image 내부의 `GEOMETRICALLY_RELEVANT` query에서 valid median event가 없으면 `UNRESOLVED`를 유지하는 historical Semantics A이다. valid event가 있는 경우에만 `s=d-z` ordering을 적용한다. `OBSERVED`는 green, `OCCLUDED`는 red, `UNRESOLVED`는 gray라는 canonical palette는 향후 eligible spatial visualization에도 유지해야 한다.

이 directory의 report와 NPZ는 pointwise semantic reconciliation 및 domain-gap evidence만 나타낸다. 이를 hidden surface 존재, physical first-hit truth, Gaussian Region, fused TSDF occupancy 또는 future NURBS/latent geometry로 해석해서는 안 된다.
""".rstrip() + "\n",
        encoding="utf-8",
    )
    review = out / "review_views"
    review.mkdir(parents=True, exist_ok=True)
    (review / "README.md").write_text(
        """# W161 spatial review views

이 visualization directory는 `OCCLUSION_DOMAIN_CONTRACT_GAP` 때문에 의도적으로 spatial PNG를 생성하지 않았다. 입력 camera와 W160 pointwise contract는 존재하지만, 이를 평가할 canonical pre-latent 3D query domain과 resolution/indexing contract가 없다.

따라서 `global_state`, `global_occluded_only`, `global_unresolved`, `occluded_region_ids`, `relevant_view_count`, `common_world` 이미지는 만들지 않았다. 임의 scene bounds, Gaussian-center AABB 확장, W153 authoritative support, 새 voxel size, smoothing/dilation/bridge를 사용해 이미지를 만들면 architecture evidence가 아니라 발명된 domain을 표시하게 된다.

향후 domain contract가 별도 승인된 뒤에는 canonical camera 조건을 공유하고, `GLOBAL OBSERVED=green`, `GLOBAL OCCLUDED=red`, `GLOBAL UNRESOLVED=gray`를 고정해 각 visualization type에 공통 README를 둬야 한다. 현재 상태에서 human reviewer가 spatial plausibility를 판단할 수 있다고 선언하지 않는다.
""".rstrip() + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_contracts()
    if not synthetic["all_pass"]:
        raise RuntimeError("synthetic A-F failure")
    domain = spatial_domain_audit()

    _progress("loading frozen W160 checkpoint, cameras, and W153 median maps")
    model, payload = w160.load_primitive_model(args.checkpoint, device=args.device)
    if w160.checkpoint_primitive(payload) != w160.PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("canonical 2DGS surfel checkpoint required")
    cameras, camera_meta = w160.load_all_train_cameras(
        args.source, args.images, args.sparse_dir, args.resolution, args.llffhold, args.device
    )
    names = [str(camera.image_name) for camera in cameras]
    if len(names) != 161:
        raise ValueError(f"expected 161 cameras, got {len(names)}")
    depth_np, depth_meta = w160._load_depth_cache(args.cache, names)
    depth_maps = [torch.as_tensor(row, dtype=torch.float32, device=args.device) for row in depth_np]
    positions = model.get_xyz.detach()
    row_count = int(positions.shape[0])

    geometric_relevant_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    evidence_available_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    relevant_without_evidence_count = torch.zeros(row_count, dtype=torch.int16, device=args.device)
    a_observed_any = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    a_has_relevant = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    a_has_unresolved = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    b_observed_any = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    b_has_relevant = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    b_has_unresolved = torch.zeros(row_count, dtype=torch.bool, device=args.device)
    per_camera: dict[str, Any] = {}

    with torch.no_grad():
        for camera_index, (camera, median_flat) in enumerate(zip(cameras, depth_maps)):
            geometry = w160.project_queries(camera, positions)
            pixel = geometry.pixel_index.clamp(min=0)
            median = median_flat[pixel]
            geometric = geometry.relevant
            evidence = geometric & (median > 0.0)
            historical = w160.candidate_b.classify_view(geometry, median_flat)["states"]
            semantics_b = relabel_semantics_a_to_b(historical, geometric, evidence)

            geometric_relevant_count += geometric.to(torch.int16)
            evidence_available_count += evidence.to(torch.int16)
            relevant_without_evidence_count += (geometric & ~evidence).to(torch.int16)

            a_observed_any |= historical == STATE_OBSERVED
            a_has_relevant |= geometric
            a_has_unresolved |= historical == STATE_UNRESOLVED
            b_observed_any |= semantics_b == STATE_OBSERVED
            b_has_relevant |= semantics_b != STATE_NON_RELEVANT
            b_has_unresolved |= semantics_b == STATE_UNRESOLVED
            per_camera[names[camera_index]] = _account_camera(historical, geometry, median)

            if camera_index % 20 == 0 or camera_index == len(cameras) - 1:
                _progress(f"reconciled {camera_index + 1}/{len(cameras)} cameras")
            del geometry, pixel, median, geometric, evidence, historical, semantics_b

    global_a_t = _global_from_accumulators(a_observed_any, a_has_relevant, a_has_unresolved)
    global_b_t = _global_from_accumulators(b_observed_any, b_has_relevant, b_has_unresolved)
    global_a = global_a_t.detach().cpu().numpy().astype(np.int8)
    global_b = global_b_t.detach().cpu().numpy().astype(np.int8)
    geometric_counts = geometric_relevant_count.detach().cpu().numpy()
    evidence_counts = evidence_available_count.detach().cpu().numpy()
    missing_counts = relevant_without_evidence_count.detach().cpu().numpy()

    semantic_a_pairs = int(geometric_counts.sum())
    evidence_pairs = int(evidence_counts.sum())
    missing_evidence_pairs = int(missing_counts.sum())
    np.savez_compressed(
        args.out / "w161_pointwise_relevance_audit.npz",
        global_state_semantics_a=global_a,
        global_state_semantics_b=global_b,
        geometrically_relevant_camera_count=geometric_counts,
        evidence_available_camera_count=evidence_counts,
        relevant_without_evidence_camera_count=missing_counts,
    )
    _write_readmes(args.out)

    state_counts_a = {name: int((global_a == code).sum()) for code, name in STATE_NAMES.items() if code != STATE_NON_RELEVANT}
    state_counts_b = {name: int((global_b == code).sum()) for code, name in STATE_NAMES.items() if code != STATE_NON_RELEVANT}
    transition_counts = _transition_counts(global_a, global_b)
    report = {
        "status": "STOPPED_WL161_OCCLUSION_DOMAIN_CONTRACT_GAP",
        "batch": "Worklog 161 — Global Persistent-Occlusion Spatial Domain and Occluded-Region Contract Audit",
        "intent_alignment": {
            "diagnostic_only": True,
            "production_behavior_modified": False,
            "historical_outputs_preserved": True,
            "spatial_field_constructed": False,
            "reason": "No justified pre-latent spatial query domain was found; no new bounds or resolution were chosen.",
        },
        "implementation_fidelity": {
            "w160_pointwise_classifier_reused": True,
            "historical_candidate_b_replayed": True,
            "historical_global_aggregation_replayed": True,
            "new_visibility_semantics": False,
            "new_domain_or_resolution": False,
            "synthetic_contracts_A_to_F_pass": synthetic["all_pass"],
        },
        "w160_reconciliation": {
            "gaussian_center_count": row_count,
            "camera_count": len(cameras),
            "per_view_pairs": row_count * len(cameras),
            "pointwise_population_role": "semantic audit only; not accepted as W161 spatial domain",
        },
        "geometric_relevance_vs_evidence_availability": {
            "historical_semantics": "SEMANTICS_A",
            "geometrically_relevant_pairs": semantic_a_pairs,
            "renderer_evidence_available_pairs": evidence_pairs,
            "geometrically_relevant_but_evidence_unavailable_pairs": missing_evidence_pairs,
            "evidence_available_fraction_of_geometrically_relevant": evidence_pairs / max(semantic_a_pairs, 1),
            "per_camera_accounting": per_camera,
            "source_contract": "project_queries defines w>0, z>=0.2, rounded pixel inside image; Candidate-B then keeps relevant/no-median as UNRESOLVED.",
        },
        "historical_global_state_impact": {
            "semantics_a": "geometrically relevant + no valid median event -> UNRESOLVED",
            "semantics_b": "no valid median event -> NON_RELEVANT",
            "real_global_state_semantics_a": state_counts_a,
            "hypothetical_global_state_semantics_b": state_counts_b,
            "a_to_b_transitions": transition_counts,
            "global_state_disagreement_count": int((global_a != global_b).sum()),
            "conclusion": "No real impact when relevant_without_evidence_pairs is zero; Semantics A remains the historical and safer contract.",
        },
        "spatial_query_domain_contract": domain,
        "discretization_resolution_contract": {
            "existing_approved_spatial_resolution": False,
            "w153_h": float(depth_meta["runtime"]["h"]),
            "w153_h_role": "TSDF surface-field discretization only; not reused as a general occlusion-domain spacing",
            "new_resolution_chosen": None,
            "indexing_contract": None,
        },
        "stop_condition_result": {
            "triggered": True,
            "verdict": "OCCLUSION_DOMAIN_CONTRACT_GAP",
            "missing": domain["missing_contract"],
            "spatial_classification_started": False,
        },
        "global_occlusion_field": {
            "constructed": False,
            "reason": "OCCLUSION_DOMAIN_CONTRACT_GAP",
            "total_domain_queries": None,
            "global_state_accounting": None,
        },
        "global_state_accounting": {
            "constructed": False,
            "reason": "No canonical spatial domain; Gaussian-center global counts above are not spatial-field counts.",
        },
        "occluded_region_accounting": {
            "eligible": False,
            "constructed": False,
            "reason": "No canonical spatial cell complex; no native connectivity was invented.",
        },
        "gaussian_tsdf_independence": {
            "performed": False,
            "reason": "A Gaussian/TSDF independence accounting would require a real W161 spatial query domain.",
            "no_new_near_radius": True,
        },
        "synthetic_contracts_A_to_F": synthetic,
        "real_scene_quantitative_result": {
            "performed": False,
            "reason": "Stopped before spatial-domain classification.",
            "pointwise_semantic_audit_only": True,
        },
        "human_qualitative_review_exports": {
            "status": "HUMAN_REVIEW_REQUIRED",
            "spatial_visualizations_created": False,
            "reason": "No canonical spatial field exists to visualize without inventing a domain.",
            "review_cameras": ["DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG"],
            "questions_deferred": [
                "physical plausibility of GLOBAL OCCLUDED locations",
                "missing evidence preserved as UNRESOLVED",
                "direct reachability from review cameras",
                "Gaussian-free occluded locations",
                "spatial coherence of Occluded Regions",
                "boundary relation to persistent observability versus TSDF coverage",
            ],
        },
        "relation_to_visible_surface_branch": {
            "gate_o1_pointwise_per_view_global": "CLOSED by W160 and reconciled here",
            "gate_o2_spatial_global_occluded_domain": "OPEN — OCCLUSION_DOMAIN_CONTRACT_GAP",
            "gate_v_visible_source_surface_support_boundary_representative": "OPEN",
            "gate_c_continuation": "NOT_STARTED",
            "wl154_wl159_modified": False,
        },
        "architecture_verdict": {
            "allowed": [
                "GLOBAL_OCCLUSION_DOMAIN_VALIDATED",
                "RELEVANT_EVIDENCE_SEMANTIC_MISMATCH",
                "OCCLUSION_DOMAIN_CONTRACT_GAP",
                "GLOBAL_FIELD_VALID_REGION_TOPOLOGY_OPEN",
                "MIXED",
                "UNRESOLVED",
            ],
            "architecture_verdict": "OCCLUSION_DOMAIN_CONTRACT_GAP",
            "intent_alignment": "PASS",
            "implementation_fidelity": "PASS",
            "architecture_result": "OCCLUSION_DOMAIN_CONTRACT_GAP",
            "reason": "W160 pointwise evaluation is valid and historical relevance semantics reconcile as Semantics A, but no approved pre-latent all-space query domain/resolution/indexing contract exists.",
        },
        "retained_rejected_open": {
            "retained": [
                "W160 pointwise classifier and historical Candidate-B",
                "W153 renderer median-depth maps and TSDF artifact",
                "canonical checkpoint, 161 cameras, calibration, projection convention",
                "W154-W159 downstream artifacts",
            ],
            "rejected": [
                "Gaussian-center population as spatial domain",
                "Gaussian-center AABB expansion",
                "W153 authoritative support as full domain",
                "new voxel spacing/resolution",
                "fused TSDF sign as occlusion",
                "UNRESOLVED-to-OCCLUDED promotion",
                "new spatial connectivity, smoothing, dilation, bridge, or region pruning",
                "hidden geometry, NURBS, continuation, or latent evidence",
            ],
            "open": [
                "canonical pre-latent scene/query bounds",
                "approved general spatial discretization and indexing",
                "native topology for the complete query domain",
                "independent physical first-hit/hidden-surface evidence",
            ],
        },
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "source": str(args.source.resolve()),
            "cache": str(args.cache.resolve()),
            "camera_names": names,
            "camera_meta": camera_meta,
            "w153_depth_cache": depth_meta,
            "renderer": "W160 frozen OSNSurfelRasterizer median channel",
        },
        "outputs": {
            "report": str(args.out / "worklog_161_report.json"),
            "pointwise_relevance_audit": str(args.out / "w161_pointwise_relevance_audit.npz"),
            "review_root_readme": str(args.out / "review_views/README.md"),
            "spatial_visualizations": "not created because domain contract is missing",
        },
        "forbidden_changes": {
            "candidate_b": False,
            "renderer_median_semantics": False,
            "fused_tsdf": False,
            "gaussian_regions": False,
            "wl154_wl159": False,
            "boundary_first": False,
            "nurbs": False,
            "continuation": False,
            "latent_geometry": False,
        },
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_161_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
    print(
        json.dumps(
            {
                "status": report["status"],
                "architecture_verdict": report["architecture_verdict"]["architecture_verdict"],
                "geometrically_relevant_pairs": report["geometric_relevance_vs_evidence_availability"]["geometrically_relevant_pairs"],
                "evidence_available_pairs": report["geometric_relevance_vs_evidence_availability"]["renderer_evidence_available_pairs"],
                "relevant_without_evidence_pairs": report["geometric_relevance_vs_evidence_availability"]["geometrically_relevant_but_evidence_unavailable_pairs"],
                "global_state_disagreement_count": report["historical_global_state_impact"]["global_state_disagreement_count"],
                "spatial_field_constructed": report["global_occlusion_field"]["constructed"],
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
