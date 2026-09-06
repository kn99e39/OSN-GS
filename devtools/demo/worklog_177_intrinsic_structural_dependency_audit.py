"""Worklog 177: audit whether W150 structure admits an intrinsic-normal swap.

This is a read-only, source-level contract audit.  It deliberately stops at
the composition gate: no W150/W154 production module, checkpoint, TSDF
artifact, or visual review is changed or replayed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CLASSIFICATIONS = {
    "NORMAL_SOURCE_INDEPENDENT",
    "GENERIC_NORMAL_CONSUMER",
    "COVARIANCE_FRAME_DEPENDENT",
    "SEMANTICALLY_INCOMPATIBLE_WITH_SIMPLE_SUBSTITUTION",
}


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _refs(text: str, pattern: str) -> list[int]:
    expression = re.compile(pattern)
    return [index for index, line in enumerate(text.splitlines(), start=1) if expression.search(line)]


def _source(file: str, text: str, *patterns: str) -> dict[str, Any]:
    return {"file": file, "lines": sorted({line for pattern in patterns for line in _refs(text, pattern)}), "patterns": list(patterns)}


def _dependency_audit(root: Path) -> list[dict[str, Any]]:
    frame_rel = "osn_gs/surface/torch_gaussian_covariance_frame.py"
    reliability_rel = "osn_gs/surface/torch_gaussian_structural_reliability.py"
    affinity_rel = "osn_gs/surface/torch_gaussian_manifold_affinity.py"
    formation_rel = "osn_gs/surface/torch_gaussian_surface_region_formation.py"
    frame = _read(root, frame_rel)
    reliability = _read(root, reliability_rel)
    affinity = _read(root, affinity_rel)
    formation = _read(root, formation_rel)

    rows = [
        {
            "quantity": "normal",
            "semantic_role": "표면 orientation line의 sign-independent 비교",
            "source": _source(frame_rel, frame, r"normal_candidate: Any", r"normal_candidate = eigenvectors"),
            "depends_on": "covariance eigenvector lambda3; W96/W97 대안은 learned intrinsic t_w",
            "intrinsic_tw_preserves_meaning": "부분적으로 yes: normal-only dot/alignment의 의미는 보존 가능",
            "classification": "GENERIC_NORMAL_CONSUMER",
            "transitive_note": "그러나 W150 downstream metric 전체는 normal만으로 정의되지 않는다.",
        },
        {
            "quantity": "tangent axes",
            "semantic_role": "local covariance principal frame의 tangent basis",
            "source": _source(frame_rel, frame, r"tangent_u: Any", r"tangent_v: Any", r"tangent_u = eigenvectors", r"tangent_v = eigenvectors"),
            "depends_on": "lambda1/lambda2 eigenvectors",
            "intrinsic_tw_preserves_meaning": "no: t_w만 바꾸면 covariance tangent axes와 하나의 orthonormal frame을 이루지 않을 수 있음",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "tangent axes를 intrinsic-derived basis로 다시 만들면 normal source만의 치환이 아니다.",
        },
        {
            "quantity": "spatial relation",
            "semantic_role": "candidate generation 및 local support의 spatial/scale relation",
            "source": _source(affinity_rel, affinity, r"candidate_scale = .*frame\.tangent_major_scale", r"equivalent_tangent_scale = frame\.equivalent_tangent_scale", r"within_radius"),
            "depends_on": "positions plus tangent_major_scale/equivalent_tangent_scale",
            "intrinsic_tw_preserves_meaning": "no: same positions라도 covariance footprint scale gate가 유지되어 mixed semantics가 됨",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "새 intrinsic scale을 만들면 역사적 scale contract가 변경된다.",
        },
        {
            "quantity": "tangent-plane displacement",
            "semantic_role": "normal projection을 제거한 local tangent residual",
            "source": _source(affinity_rel, affinity, r"tangent_component = displacement", r"mutual_tangent_residual", r"tangent_direction_displacement_ratio"),
            "depends_on": "normal_candidate plus residual_scale (default tangent_major_scale)",
            "intrinsic_tw_preserves_meaning": "projection 부분은 가능하지만 normalization/footprint semantics는 보존되지 않음",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "같은 normal을 넣어도 covariance scale이 metric의 단위를 결정한다.",
        },
        {
            "quantity": "normal separation",
            "semantic_role": "parallel sheet의 normal-direction gap을 local thickness로 정규화",
            "source": _source(affinity_rel, affinity, r"normal_direction_gap", r"normal_direction_separation_over_thickness", r"normal_thickness"),
            "depends_on": "normal_candidate plus normal_thickness=sqrt(lambda3)",
            "intrinsic_tw_preserves_meaning": "no: t_w에는 historical covariance thickness의 one-sided meaning이 없음",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "intrinsic t_w normal로 바꾸고 covariance thickness를 남기면 서로 다른 frame/evidence를 혼합한다.",
        },
        {
            "quantity": "reliability",
            "semantic_role": "intrinsic shape validity와 contextual neighborhood reliability",
            "source": _source(reliability_rel, reliability, r"def evaluate_intrinsic_reliability", r"frame\.tangent_major_scale", r"frame\.normal_thickness", r"def evaluate_contextual_consistency"),
            "depends_on": "eigenvalue/shape class, tangent scales, normal thickness, neighbor normal agreement",
            "intrinsic_tw_preserves_meaning": "no: intrinsic reliability는 covariance shape/conditioning/scale evidence 자체를 평가",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "reliability class는 seed eligibility와 ambiguity에 직접 공급된다.",
        },
        {
            "quantity": "affinity",
            "semantic_role": "candidate pair를 same_surface/crease/parallel/ambiguous로 분류",
            "source": _source(affinity_rel, affinity, r"def _compute_pair_metrics", r"def _classify_relation", r"frame\.equivalent_tangent_scale", r"frame\.elongation"),
            "depends_on": "normal alignment, tangent residual, footprint/aniso ratio, normal separation and scale gates",
            "intrinsic_tw_preserves_meaning": "no: normal authority만 교체해도 pair relation의 입력 metric 일부가 covariance-specific",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "form_surface_regions는 affinity graph를 이미 계산된 사실로 소비하지만 graph의 의미가 바뀐다.",
        },
        {
            "quantity": "parallel conflict",
            "semantic_role": "aligned normals라도 distinct parallel sheet이면 분리하는 veto",
            "source": _source(affinity_rel, affinity, r"parallel_separate_min_normal_gap_over_thickness", r"RELATION_PARALLEL_SEPARATE", r"normal_direction_separation_over_thickness"),
            "depends_on": "normal alignment + thickness-normalized gap + residual/footprint",
            "intrinsic_tw_preserves_meaning": "no: t_w가 surface normal이어도 Gaussian thickness bound를 제공하지 않음",
            "classification": "SEMANTICALLY_INCOMPATIBLE_WITH_SIMPLE_SUBSTITUTION",
            "transitive_note": "이 safeguard가 W150의 close-parallel separation 의미를 보존하는 핵심 차단점이다.",
        },
        {
            "quantity": "shared-neighbor consensus",
            "semantic_role": "두 endpoint를 잇는 independent same_surface neighbor support",
            "source": _source(formation_rel, formation, r"def _compute_edge_consensus", r"contradiction_ratio", r"shared_same_surface_neighbor_count"),
            "depends_on": "same_surface graph, intrinsic/contextual reliability, neighbor normal comparisons",
            "intrinsic_tw_preserves_meaning": "direct count는 generic하지만 upstream graph/reliability semantics는 유지되지 않음",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "t_w substitution 뒤 같은 edge set이라는 전제가 성립하지 않는다.",
        },
        {
            "quantity": "seed classification",
            "semantic_role": "reliable core와 seed eligibility를 정하는 초기 분류",
            "source": _source(formation_rel, formation, r"def _seed_core_components", r"INTRINSIC_RELIABLE", r"core_min_same_surface_degree"),
            "depends_on": "reliability class and affinity degree/relation",
            "intrinsic_tw_preserves_meaning": "algorithm branch는 generic하지만 입력 class가 바뀌므로 identity-preserving substitution이 아님",
            "classification": "GENERIC_NORMAL_CONSUMER",
            "transitive_note": "directly normal을 읽지는 않지만 covariance-dependent inputs를 통해 의미가 결정된다.",
        },
        {
            "quantity": "component-pair support",
            "semantic_role": "candidate component pair를 지지하는 independent cross edges의 개수",
            "source": _source(formation_rel, formation, r"bridge_min_independent_cross_edges", r"endpoint_support", r"component"),
            "depends_on": "edge incidence/counting; no new geometric quantity by itself",
            "intrinsic_tw_preserves_meaning": "counting operation 자체는 보존 가능하지만 counted edges의 identity는 보존되지 않음",
            "classification": "NORMAL_SOURCE_INDEPENDENT",
            "transitive_note": "normal-independent인 것은 집계 연산뿐이며 W150 certificate 전체가 independent라는 뜻은 아니다.",
        },
        {
            "quantity": "bridge veto",
            "semantic_role": "single fragile bridge가 서로 다른 core를 union하지 못하게 하는 veto",
            "source": _source(formation_rel, formation, r"def _evaluate_bridge_veto", r"bridge_tangent_divergence_threshold", r"tangent_frame_divergence", r"BRIDGE_WELL_SUPPORTED"),
            "depends_on": "shared support, local normal divergence, local cut, tangent/path evidence",
            "intrinsic_tw_preserves_meaning": "no: tangent-frame divergence and support graph are not normal-only",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "W38에서 보존한 bridge veto semantics의 핵심 입력이 섞인다.",
        },
        {
            "quantity": "path/tangent transport",
            "semantic_role": "direct shortcut과 multi-hop local path의 rotation/scale consistency",
            "source": _source(formation_rel, formation, r"def _evaluate_path_consistency", r"frame\.normal_candidate", r"frame\.tangent_major_scale", r"tangent_plane_displacement_consistency"),
            "depends_on": "frame normal rotations plus tangent-major scale transport",
            "intrinsic_tw_preserves_meaning": "no: t_w-only replacement leaves path scale/normal from different authorities",
            "classification": "SEMANTICALLY_INCOMPATIBLE_WITH_SIMPLE_SUBSTITUTION",
            "transitive_note": "path safeguard는 단일 normal scalar seam으로 분리되지 않는다.",
        },
        {
            "quantity": "ambiguity",
            "semantic_role": "contradictory/insufficient evidence를 forced ownership 대신 ambiguity로 보존",
            "source": _source(formation_rel, formation, r"CONSENSUS_CONTRADICTED", r"RELATION_AMBIGUOUS", r"contextual_class"),
            "depends_on": "reliability, affinity relation, consensus and path/bridge outcomes",
            "intrinsic_tw_preserves_meaning": "policy shape is generic, but evidence meaning changes with the substituted frame",
            "classification": "COVARIANCE_FRAME_DEPENDENT",
            "transitive_note": "ambiguity를 보존한다는 정책과 ambiguity를 산출하는 evidence semantics를 구분해야 한다.",
        },
    ]
    for row in rows:
        if row["classification"] not in CLASSIFICATIONS:
            raise ValueError(row["classification"])
    return rows


def build_report(root: Path) -> dict[str, Any]:
    runner_rel = "devtools/demo/candidate_f_gaussian_region_owned_tsdf.py"
    constructor_rel = "osn_gs/surface/torch_visible_surface_construction.py"
    formation_rel = "osn_gs/surface/torch_gaussian_surface_region_formation.py"
    frame_rel = "osn_gs/surface/torch_gaussian_covariance_frame.py"
    reliability_rel = "osn_gs/surface/torch_gaussian_structural_reliability.py"
    affinity_rel = "osn_gs/surface/torch_gaussian_manifold_affinity.py"
    runner = _read(root, runner_rel)
    constructor = _read(root, constructor_rel)
    formation = _read(root, formation_rel)
    frame = _read(root, frame_rel)
    reliability = _read(root, reliability_rel)
    affinity = _read(root, affinity_rel)

    dependency_audit = _dependency_audit(root)
    w154_w97_lines = _refs(runner, r"partition_surfels_region_coherent\(active_orientation, RegionCoherenceConfig\(\)")
    w154_w150_lines = _refs(runner, r"form_surface_regions")
    w150_call_lines = _refs(constructor, r"regions\s*=\s*form_surface_regions\(")
    constructor_inputs = _refs(constructor, r"frame = extract_covariance_frame|reliability = evaluate_structural_reliability|graph = build_manifold_affinity_graph")
    formation_definition = _refs(formation, r"def form_surface_regions\(")

    return {
        "worklog": "W177",
        "title": "Intrinsic-Normal Structural Local Surface Decomposition: Controlled Contract Composition Audit",
        "read_only_audit": True,
        "production_behavior_modified": False,
        "intent_alignment": {
            "preserved": ["W97/W154 Baseline A", "W10/W31-W38/W150 structural Baseline B", "historical thresholds", "W154/W155 checkpoint identity", "W171 support and W174 witness lineage"],
            "not_done": ["Candidate C implementation", "W97/W150 production replacement", "threshold tuning", "TSDF connectivity or identity change", "NURBS/continuation/Eligibility/UNRESOLVED", "new visualization replay"],
        },
        "architecture_question": "W150 structural safeguards와 learned intrinsic t_w normal authority가 normal source만 바꾸는 clean composition인지 판정",
        "baseline_a_w97": {
            "status": "REPRODUCED_BY_PRESERVED_W154_W155_EVIDENCE",
            "normal_authority": "learned intrinsic t_w",
            "constructor": "partition_surfels_region_coherent",
            "exact_call_lines": w154_w97_lines,
            "active_checkpoint": "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000",
            "active_surfels": 1190469,
            "raw_regions": 104977,
            "accepted_regions": 64892,
            "accepted_population": 1146852,
            "unassigned_isolated_fallback": 40085,
            "largest_region_fraction": 0.216319,
            "top_1_5_10_fraction": [0.216319, 0.330285, 0.358279],
            "source": "W155 exact replay and W175 lineage audit",
        },
        "baseline_b_w150": {
            "status": "NOT_COMPARABLE",
            "normal_authority": "covariance eigenframe normal_candidate",
            "constructor": "extract_covariance_frame -> evaluate_structural_reliability -> build_manifold_affinity_graph -> form_surface_regions",
            "constructor_input_lines": constructor_inputs,
            "form_surface_regions_call_lines": w150_call_lines,
            "form_surface_regions_definition_lines": formation_definition,
            "reason": "W154/W155 checkpoint에 W150 covariance frame, reliability, affinity artifacts가 동일 row order로 보존되어 있지 않다. W97 checkpoint에 대한 approximation은 하지 않는다.",
            "w154_runner_form_surface_regions_lines": w154_w150_lines,
        },
        "dependency_audit": dependency_audit,
        "parameter_contract": {
            "status": "PRESERVED_NOT_REMAPPED",
            "historical_sources": {
                "covariance_frame": "extract_covariance_frame(covariance)",
                "candidate_scale_default": "frame.tangent_major_scale",
                "residual_scale_default": "frame.tangent_major_scale",
                "footprint_scale": "frame.equivalent_tangent_scale",
                "parallel_gap_scale": "frame.normal_thickness",
                "normal_relation": "abs(dot(frame.normal_candidate[a], frame.normal_candidate[b]))",
                "path_scale": "frame.tangent_major_scale",
            },
            "mathematical_identity": {
                "tangent_projection": "d_T = d - (d·n)n",
                "normal_gap": "|d·n_bar| / tau_n",
                "candidate_distance": "||d|| / s_t",
                "footprint_overlap": "uses equivalent_tangent_scale",
            },
            "threshold_action": "Candidate C가 구현되지 않아 값/threshold/scale identity를 변경하지 않았다.",
        },
        "clean_composition_gate": {
            "status": "NO_CLEAN_NORMAL_SUBSTITUTION",
            "passed": False,
            "required_condition": "substitution이 authoritative surface normal만 바꾸고 모든 downstream quantity의 semantic meaning을 보존해야 함",
            "blocking_dependencies": [
                "GaussianCovarianceFrame.tangent_u/tangent_v 및 tangent scales가 normal과 같은 covariance eigenframe에서 유도됨",
                "evaluate_intrinsic_reliability가 covariance shape/conditioning/scale을 primary evidence로 판정함",
                "build_manifold_affinity_graph가 tangent residual, footprint/aniso ratio, normal-thickness gap을 함께 사용함",
                "form_surface_regions의 bridge/path safeguards가 frame.normal_candidate와 frame.tangent_major_scale을 함께 사용함",
            ],
            "two_invalid_substitutions": [
                "covariance tangents/scales를 유지하고 normal만 t_w로 바꾸면 하나의 coherent orthonormal frame이 아닌 mixed frame이 된다.",
                "t_w로 tangent basis/scale까지 다시 만들면 normal authority만 바꾸는 것이 아니며 historical covariance scale/threshold semantics를 재정의한다.",
            ],
            "exact_stop": "핵심 safeguard인 parallel conflict 및 tangent/path transport가 simple normal substitution으로 의미 보존되지 않으므로 Candidate C를 구현하지 않고 중단",
        },
        "candidate_c": {
            "status": "NOT_IMPLEMENTED_CLEAN_GATE_FAILED",
            "separate_module_created": False,
            "historical_synthetic_fixtures": "NOT_RUN_CLEAN_GATE_FAILED",
            "real_scene": "NOT_RUN_CLEAN_GATE_FAILED",
            "production_semantics_changed": False,
        },
        "w174_witness": {
            "status": "BASELINE_A_ONLY",
            "row_4043": {"stable_gaussian_id": 4937175, "subset_id": 1, "membership_role": "core", "ambiguous": False, "path_subset_crossing": 0},
            "row_4051": {"stable_gaussian_id": 3929355, "subset_id": 1, "membership_role": "core", "ambiguous": False, "path_subset_crossing": 0},
            "interpretation": "두 row의 ID가 다르다는 사실이 아니라 W97의 existing core membership와 stored path owner가 이미 보존되어 있다는 점이 safeguard evidence이다.",
            "baseline_b_and_c": "NOT_COMPARABLE_NO_W150_CROSSWALK / NO_CANDIDATE_C",
        },
        "tabletop_fixed_aabb": {
            "status": "BASELINE_A_ONLY",
            "complete_support_rows": 17965,
            "selected_support_rows": 15189,
            "support_subset_count": 1,
            "support_by_subset": {"1": {"complete": 17965, "selected": 15189}},
            "distinct_owner_gaussians": {"complete": 3315, "selected": 2887},
            "candidate_c": "NOT_RUN_CLEAN_GATE_FAILED",
            "forbidden_identity_changes": ["TSDF components", "W174 triangles", "W173 loops"],
        },
        "zero_set_projection": {
            "status": "NOT_RUN_NO_CANDIDATE",
            "association_rule": "existing stored W154 nearest-Gaussian association; no nearest recomputation",
            "candidate_c_subset_projection": "not available",
            "tsdf_connectivity_used_for_identity": False,
        },
        "scene_wide_accounting": {
            "baseline_a": {
                "active_gaussians": 1190469,
                "subsets": 104977,
                "largest_fraction": 0.216319,
                "top_k_fraction": {"1": 0.216319, "5": 0.330285, "10": 0.358279},
                "singleton_subsets": 40085,
                "ambiguous_support_state_count": 3532,
                "rejected_count": 0,
                "unassigned_isolated_fallback": 40085,
                "mechanism_attribution": "W97 region-concentration and singleton fallback; not W150 bridge/path",
            },
            "baseline_b": "NOT_COMPARABLE",
            "candidate_c": "NOT_RUN_CLEAN_GATE_FAILED",
        },
        "historical_lineages": {
            "parallel_sheet": "W10/W31-W38 covariance-frame controls retained; W150 same-checkpoint replay NOT_COMPARABLE",
            "phase_alias_shortcut": "W10/W31-W38 path safeguard retained historically; Candidate C NOT_RUN",
            "weak_bridge": "W38 articulation bridge control retained historically; Candidate C NOT_RUN",
            "w99_w100_patio_to_hedge": "NOT_COMPARABLE exact stable mapping for this W177 arm; no reinterpretation",
            "curved_real_positive_control": "REAL_CURVED_POSITIVE_CONTROL_UNAVAILABLE (W171 curved/vase invalid; synthetic historical controls only)",
        },
        "visualization": {
            "status": "NOT_GENERATED_CLEAN_GATE_FAILED",
            "reason": "W177 visualization A-G is conditional on Candidate C reaching real scene; clean composition gate failed before Candidate C.",
            "preserved_prior_artifacts": "W154/W155/W171-W175 artifacts unchanged",
            "tsdf_component_coloring": False,
        },
        "verdict": "NO_CLEAN_NORMAL_SUBSTITUTION",
        "architecture_interpretation": "W150 structural safeguards are not a normal-only wrapper around form_surface_regions. Their evidence contract couples covariance normal, tangent frame, footprint/thickness scales, reliability and affinity. Intrinsic t_w remains a valid W97/W154 baseline authority, but cannot be substituted into W150 without a new contract.",
        "retained_rejected_open": {
            "retained": ["W97/W154/W155 exact identity lineage", "W150 covariance structural lineage", "W174 witness baseline", "Gaussian WHO vs TSDF WHERE separation", "all historical thresholds and production behavior"],
            "rejected": ["automatic W150+t_w composition", "largest-component or trusted-subset filtering", "new tolerance/threshold", "TSDF component identity", "automatic architecture promotion"],
            "open": ["separate future architecture decision for a genuinely intrinsic-normal structural contract", "same-checkpoint W97↔W150 crosswalk with covariance/reliability/affinity artifacts", "whether to define new tangent/scale semantics rather than claim a normal-only substitution"],
        },
        "source_assertions": {
            "w154_w97_call_detected": w154_w97_lines,
            "w154_w150_call_detected": w154_w150_lines,
            "w150_constructor_call_detected": w150_call_lines,
            "frame_has_covariance_tangent_and_scale_fields": all(_refs(frame, pattern) for pattern in [r"tangent_u: Any", r"tangent_v: Any", r"tangent_major_scale: Any", r"normal_thickness: Any", r"equivalent_tangent_scale: Any"]),
            "reliability_uses_frame_scales": bool(_refs(reliability, r"frame\.tangent_major_scale") and _refs(reliability, r"frame\.normal_thickness")),
            "affinity_uses_frame_scales": bool(_refs(affinity, r"frame\.tangent_major_scale") and _refs(affinity, r"frame\.normal_thickness") and _refs(affinity, r"frame\.equivalent_tangent_scale")),
            "classification_values_valid": all(row["classification"] in CLASSIFICATIONS for row in dependency_audit),
        },
    }


def _write_output(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "worklog_177_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Worklog 177 감사 산출물\n\n"
        "이 디렉터리는 W150 structural constructor에 learned intrinsic `t_w` normal authority를 단순 치환할 수 있는지 정적·계약 수준에서 감사한 결과입니다.\n\n"
        "- `worklog_177_report.json`: dependency audit, clean composition gate, Baseline A/B와 Candidate C의 실행 상태, witness/tabletop/scene-wide accounting.\n"
        "- 이 audit은 read-only이며 W97/W150/W154 production, checkpoint, TSDF, NURBS, existing visualization을 변경하지 않았습니다.\n"
        "- Candidate C는 clean gate 실패로 구현·replay·real-scene visualization을 수행하지 않았습니다.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.repo.resolve())
    if args.out is not None:
        _write_output(report, args.out.resolve())
    print(json.dumps({"worklog": report["worklog"], "verdict": report["verdict"], "candidate_c": report["candidate_c"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
