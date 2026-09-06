"""Worklog 176: reconstruct the historical Local Surface contract.

This module is deliberately read-only with respect to production behavior.  It
audits preserved worklogs and source text, records the competing historical
contracts, and stops before a same-checkpoint control when no single approved
implementation satisfies the stated intent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STATUS_VALUES = {"IMPLEMENTED", "OPTIONAL", "ABSENT", "DIAGNOSTIC_ONLY"}


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _line_refs(text: str, pattern: str) -> list[int]:
    expression = re.compile(pattern)
    return [index for index, line in enumerate(text.splitlines(), start=1) if expression.search(line)]


def _cell(**values: str) -> dict[str, str]:
    missing = set(values) - {
        "gaussian_normal_source",
        "local_geometric_neighborhood",
        "knn_definition",
        "distance_locality_gate",
        "sign_independent_normal_consistency",
        "positional_continuity",
        "normal_direction_separation",
        "tangent_mutual_tangent_consistency",
        "discontinuity_first_cuts",
        "shared_neighbor_multi_edge_consensus",
        "region_orientation_concentration",
        "interface_consistency",
        "bridge_veto",
        "path_tangent_transport_consistency",
        "singleton_propagation",
        "ambiguity_handling",
        "deterministic_tie_breaking",
    }
    if missing:
        raise ValueError(f"contract matrix fields missing: {sorted(missing)}")
    invalid = {key: value for key, value in values.items() if value not in STATUS_VALUES}
    if invalid:
        raise ValueError(f"invalid contract matrix values: {invalid}")
    return values


MATRIX_FIELDS = (
    "gaussian_normal_source",
    "local_geometric_neighborhood",
    "knn_definition",
    "distance_locality_gate",
    "sign_independent_normal_consistency",
    "positional_continuity",
    "normal_direction_separation",
    "tangent_mutual_tangent_consistency",
    "discontinuity_first_cuts",
    "shared_neighbor_multi_edge_consensus",
    "region_orientation_concentration",
    "interface_consistency",
    "bridge_veto",
    "path_tangent_transport_consistency",
    "singleton_propagation",
    "ambiguity_handling",
    "deterministic_tie_breaking",
)


def _lineage() -> list[dict[str, Any]]:
    """Return the historical decision record in chronological order."""

    return [
        {
            "worklog": "W10",
            "document": "docs/worklogs/10_consensus_aware_surface_region_formation_foundation.md",
            "hypothesis": "pairwise same_surface만으로는 독립 표면을 region으로 안전하게 설명할 수 없다.",
            "mechanism_tested": "covariance frame 기반 consensus, shared-neighbor evidence, bridge veto, tangent/path consistency",
            "positive_evidence": "평면·smooth curved sheet·close-parallel sheet에서 false merge를 막고 stable/core 상태를 보존했다.",
            "negative_evidence": "foundation은 isolated diagnostic이며 real production quality를 주장하지 않았다.",
            "promoted_mechanism": "후속 W31–W38에서 이 consensus/bridge/path 계열이 canonical constructor semantics로 정제됐다.",
            "retained_mechanism": "ambiguity/rejected state, stable IDs, multi-edge support, bridge/path provenance",
            "rejected_mechanism": "단순 connected-component와 isolated pair의 core 승격",
            "explicit_stop_condition": "ordered boundary, NURBS, default dispatcher, production integration은 실행하지 않음",
            "later_status": "W31–W38 및 W124/W150의 covariance-based constructor lineage로 이어짐",
        },
        {
            "worklog": "W31–W38",
            "document": "docs/worklogs/31_per_representative_reliability_gate_trace.md; docs/worklogs/38_seed_merge_semantics_correction_and_candidate_recall_audit.md",
            "hypothesis": "region 실패와 false bridge의 원인은 reliability·candidate·seed/merge semantics를 분리하면 추적할 수 있다.",
            "mechanism_tested": "covariance-frame reliability/affinity, explicit two-phase DSU, component-pair support, bridge veto",
            "positive_evidence": "articulation bridge union이 47에서 0으로 줄고 weak bridge가 seed를 제거하지 않음을 검증했다.",
            "negative_evidence": "real termination candidate recall 및 일부 cylinder cap 문제는 남았고 scene-tuned 보정은 원복했다.",
            "promoted_mechanism": "separate_seed_and_merge_phases=True 및 bridge veto를 의미 기반 canonical semantics로 유지",
            "retained_mechanism": "shared-neighbor consensus, parallel-separation evidence, tangent/path checks, explicit ambiguity",
            "rejected_mechanism": "raw-component 내부 bridge exemption 및 threshold를 맞춘 scene-specific repair",
            "explicit_stop_condition": "candidate recall 병목을 해결했다고 주장하지 않음",
            "later_status": "W150 constructor/source contract와 W124 current canonical inventory에 반영",
        },
        {
            "worklog": "W96",
            "document": "docs/worklogs/96_2dgs_coverage_first_surfel_partition.md",
            "hypothesis": "2DGS surfel의 학습된 intrinsic t_w를 surface evidence authority로 쓰면 volumetric normal 문제를 줄일 수 있다.",
            "mechanism_tested": "intrinsic t_w + local kNN/spacing-gated coverage-first connected components",
            "positive_evidence": "2DGS intrinsic normal source를 covariance/eigen 재구성 없이 직접 사용하고 coverage identity를 보존했다.",
            "negative_evidence": "pairwise single-linkage가 real scene에서 74.70% giant subset을 만들었다.",
            "promoted_mechanism": "intrinsic t_w source와 2DGS branch의 evidence orientation",
            "retained_mechanism": "local graph, deterministic ownership, complete coverage",
            "rejected_mechanism": "plain pairwise connected component을 최종 anti-chaining 해법으로 승격",
            "explicit_stop_condition": "architecture success/failure는 review 뒤로 유보",
            "later_status": "W97에서 union rule만 교체된 실험 baseline으로 보존",
        },
        {
            "worklog": "W97",
            "document": "docs/worklogs/97_region_coherent_surfel_partition.md",
            "hypothesis": "region-level orientation concentration이 normal-only single-linkage chaining을 막을 수 있다.",
            "mechanism_tested": "intrinsic t_w, identical local graph, concentration-gated deterministic Kruskal merge, non-bridging singleton propagation",
            "positive_evidence": "largest subset이 74.70%에서 21.20%로 줄고 accumulated-drift chain이 한 region이 되지 않았다.",
            "negative_evidence": "real curved surface가 과분할됐고, 문서가 architecture adoption을 결정하지 않았다.",
            "promoted_mechanism": "W154에서는 historical baseline으로 호출됨; final architecture 승인으로 기록되지는 않음",
            "retained_mechanism": "region concentration, intrinsic t_w, deterministic ownership roles",
            "rejected_mechanism": "W97 자체를 universal final contract라고 선언",
            "explicit_stop_condition": "subset-local trust 및 다음 architecture 결정 전 human review",
            "later_status": "W98–W100에서 별도 대체 후보가 실험됐고 W175가 W154 active path임을 확인",
        },
        {
            "worklog": "W98",
            "document": "docs/worklogs/98_discontinuity_first_surfel_partition.md",
            "hypothesis": "smooth curvature와 true discontinuity를 normal concentration만으로 구분할 수 없다.",
            "mechanism_tested": "intrinsic tangent frame shape-operator residual plus positional normal-vs-tangent cut",
            "positive_evidence": "synthetic cylinder는 유지되고 crease/parallel sheets는 분리됐다.",
            "negative_evidence": "real scene에서 per-edge cuts가 graph percolation을 막지 못해 94.51% giant subset이 재발했다.",
            "promoted_mechanism": "diagnostic evidence로 shape residual/positional cut을 보존",
            "retained_mechanism": "intrinsic t_w and explicit discontinuity diagnostics",
            "rejected_mechanism": "per-edge discontinuity partition을 final architecture로 승격",
            "explicit_stop_condition": "real-scene negative result 뒤 architecture decision 없음",
            "later_status": "W99/W100 interface candidates의 evidence source로만 재사용",
        },
        {
            "worklog": "W99",
            "document": "docs/worklogs/99_interface_coherent_region_merge.md",
            "hypothesis": "W97 안전 초기화 후 W98 evidence를 complete interface에 적용하면 curvature 복원과 anti-percolation을 함께 얻을 수 있다.",
            "mechanism_tested": "positional-gated W97 init, full interface aggregation, majority smoothness, round-based merge",
            "positive_evidence": "synthetic cylinder/crease/parallel/zigzag controls와 tabletop curve recovery가 통과했다.",
            "negative_evidence": "real patio/grass/hedge가 53.86% giant subset으로 다시 merge됐다.",
            "promoted_mechanism": "complete-interface accounting과 provenance를 diagnostic으로 유지",
            "retained_mechanism": "W97 init and W98 shape/positional evidence as an experiment",
            "rejected_mechanism": "W99 merge rule의 architecture promotion",
            "explicit_stop_condition": "mixed result; no architecture decision",
            "later_status": "W100이 bilateral/region-conditioned A/B로 대체 검증",
        },
        {
            "worklog": "W100",
            "document": "docs/worklogs/100_bilateral_interface_region_merge.md",
            "hypothesis": "W99 percolation은 편측 residual 허용과 boundary-contaminated fit에서 왔을 수 있다.",
            "mechanism_tested": "region-conditioned local shape operator and bilateral AND certificate",
            "positive_evidence": "largest subset이 53.86%에서 22.91%로 줄고 W99 patio→hedge lineage 5개 merge가 모두 기각됐다.",
            "negative_evidence": "작은 region의 median+MAD 한계가 남았고 문서가 architecture final decision을 내리지 않았다.",
            "promoted_mechanism": "bilateral attribution result를 diagnostic evidence로 보존",
            "retained_mechanism": "W99 interface structure and W98 shape evidence in isolated module",
            "rejected_mechanism": "W100을 W154 production/local-surface identity contract로 연결",
            "explicit_stop_condition": "architecture review 전 stop; tuning 없음",
            "later_status": "W175가 W98/W99/W100은 W154 뒤에 붙지 않은 별도 계열임을 명시",
        },
        {
            "worklog": "W115–W116",
            "document": "docs/worklogs/115_design_intent_specification_implementation_traceability_audit.md; docs/worklogs/116_visible_nurbs_representation_contract_recovery_audit.md",
            "hypothesis": "renderer-native visible topology와 NURBS representation semantics를 분리하면 historical chart 실패를 과대해석하지 않을 수 있다.",
            "mechanism_tested": "renderer median representative, image-space topology, non-representative ambiguity separation, existing NURBS/trimming/coupled-fit audit",
            "positive_evidence": "visible topology evidence와 NURBS materialized evidence를 별도 보고해야 한다는 canonical semantic contract를 정리했다.",
            "negative_evidence": "Gaussian intrinsic-t_w local decomposition을 최종 W154 contract로 결정하지 않았고, renderer-native topology와 Gaussian region identity는 다른 축으로 남았다.",
            "promoted_mechanism": "renderer-native topology/query semantics, not a Gaussian region constructor",
            "retained_mechanism": "representation seam != physical boundary, ambiguity is not forced ownership",
            "rejected_mechanism": "blob=chart, full-rank gate, fixed capacity를 universal architecture law로 해석",
            "explicit_stop_condition": "새 chart decomposition/coupling은 별도 architecture question으로 유보",
            "later_status": "W124 current canonical inventory의 visible/provisional 분류로 이어짐",
        },
        {
            "worklog": "W124",
            "document": "docs/worklogs/124_canonical_architecture_reduction_audit.md",
            "hypothesis": "현재 실행 graph와 historical diagnostic/candidate를 분리해 smallest defensible canonical core를 만들 수 있다.",
            "mechanism_tested": "static import/call graph and architecture inventory",
            "positive_evidence": "covariance/affinity/reliability/region-formation source를 current canonical A 분류로, W97 region-coherent partition을 provisional B로 구분했다.",
            "negative_evidence": "visible topology/NURBS는 provisional이고 occluded/continuation은 open으로 남았다.",
            "promoted_mechanism": "current constructor lineage including form_surface_regions",
            "retained_mechanism": "world-space query plus optional renderer-event provenance",
            "rejected_mechanism": "historical experiment을 자동으로 canonical downstream semantics로 승격",
            "explicit_stop_condition": "new topology repair, occluded surface, trust, uncertain promotion 없음",
            "later_status": "W150/W175의 contract distinction을 해석하는 상위 inventory evidence",
        },
        {
            "worklog": "W150",
            "document": "docs/worklogs/150_boundary_first_local_surface_decomposition_contract_trace.md",
            "hypothesis": "Boundary First가 fit 전에 local surface ownership과 region-owned boundary를 사용해야 한다.",
            "mechanism_tested": "form_surface_regions after covariance frame/reliability/affinity, then boundary ownership/order and fit",
            "positive_evidence": "canonical constructor의 intended Local Surface contract를 source call graph로 명시했다.",
            "negative_evidence": "W139/W145 pooled PCA path는 이 constructor를 fit 전에 호출하지 않아 bypass였다.",
            "promoted_mechanism": "covariance-derived Gaussian frame + consensus/bridge/path region candidate as intended constructor contract",
            "retained_mechanism": "region-owned boundary and pre-fit ownership",
            "rejected_mechanism": "global PCA pooled ownership and post-fit boundary substitution",
            "explicit_stop_condition": "renderer event를 새 membership로 연결하거나 production을 수정하지 않음",
            "later_status": "W175가 W154의 W97 path와 W150 contract가 다름을 확인; same-checkpoint crosswalk는 없음",
        },
        {
            "worklog": "W151–W155",
            "document": "docs/worklogs/151_renderer_event_canonical_surface_compatibility_audit.md; docs/worklogs/155_intrinsic_normal_gaussian_region_viability_audit.md",
            "hypothesis": "renderer/TSDF downstream evidence가 기존 Gaussian local-surface identity를 의미 변경 없이 소비할 수 있는가.",
            "mechanism_tested": "compatibility and exact W154/W155 replay/mapping audits",
            "positive_evidence": "W155는 W97 stable-ID/region/status exact mapping을 재현했다.",
            "negative_evidence": "renderer event는 physical-sheet membership를 제공하지 않고 W154는 W150 constructor를 호출하지 않는다.",
            "promoted_mechanism": "none; exact lineage preservation only",
            "retained_mechanism": "W97 baseline mapping and W150 distinction",
            "rejected_mechanism": "event normal/row ID를 region identity로 재해석",
            "explicit_stop_condition": "compatibility gap이면 candidate adapter/control을 실행하지 않음",
            "later_status": "W175에서 active W154 path를 W97로 제한해 판정",
        },
        {
            "worklog": "W171–W175",
            "document": "docs/worklogs/171_zero_set_derived_visible_structural_nurbs_audit.md; docs/worklogs/175_gaussian_normal_local_surface_lineage_audit.md",
            "hypothesis": "W154 zero-set/fragmentation/NURBS 결과의 원인을 Gaussian identity, TSDF topology, structural fit으로 분리할 수 있다.",
            "mechanism_tested": "read-only ownership lineage and downstream attribution",
            "positive_evidence": "W175에서 W97 subset→W154 region→W171 ownership identity가 exact 보존됐다.",
            "negative_evidence": "W150 richer constructor와 W97 active path의 equivalence는 증명되지 않았고 W171 real support는 fragmentation gate에서 멈췄다.",
            "promoted_mechanism": "none; attribution only",
            "retained_mechanism": "Gaussian WHO / TSDF WHERE separation; native component is not a new identity",
            "rejected_mechanism": "largest-component filtering, TSDF component을 Gaussian identity로 승격",
            "explicit_stop_condition": "architecture repair, decomposition replacement, NURBS redesign 없음",
            "later_status": "W176 historical contract reconstruction으로 이관",
        },
    ]


def _matrix() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "W96 intrinsic-normal coverage-first",
            "status": "DIAGNOSTIC_ONLY",
            "contract": _cell(
                gaussian_normal_source="IMPLEMENTED", local_geometric_neighborhood="IMPLEMENTED", knn_definition="IMPLEMENTED",
                distance_locality_gate="IMPLEMENTED", sign_independent_normal_consistency="IMPLEMENTED", positional_continuity="ABSENT",
                normal_direction_separation="ABSENT", tangent_mutual_tangent_consistency="ABSENT", discontinuity_first_cuts="ABSENT",
                shared_neighbor_multi_edge_consensus="ABSENT", region_orientation_concentration="ABSENT", interface_consistency="ABSENT",
                bridge_veto="ABSENT", path_tangent_transport_consistency="ABSENT", singleton_propagation="IMPLEMENTED",
                ambiguity_handling="ABSENT", deterministic_tie_breaking="IMPLEMENTED",
            ),
        },
        {
            "candidate": "W97 region-coherent intrinsic-normal partition",
            "status": "DIAGNOSTIC_ONLY",
            "contract": _cell(
                gaussian_normal_source="IMPLEMENTED", local_geometric_neighborhood="IMPLEMENTED", knn_definition="IMPLEMENTED",
                distance_locality_gate="IMPLEMENTED", sign_independent_normal_consistency="IMPLEMENTED", positional_continuity="OPTIONAL",
                normal_direction_separation="ABSENT", tangent_mutual_tangent_consistency="ABSENT", discontinuity_first_cuts="ABSENT",
                shared_neighbor_multi_edge_consensus="ABSENT", region_orientation_concentration="IMPLEMENTED", interface_consistency="ABSENT",
                bridge_veto="ABSENT", path_tangent_transport_consistency="ABSENT", singleton_propagation="IMPLEMENTED",
                ambiguity_handling="IMPLEMENTED", deterministic_tie_breaking="IMPLEMENTED",
            ),
        },
        {
            "candidate": "W98 discontinuity-first intrinsic-normal partition",
            "status": "DIAGNOSTIC_ONLY",
            "contract": _cell(
                gaussian_normal_source="IMPLEMENTED", local_geometric_neighborhood="IMPLEMENTED", knn_definition="IMPLEMENTED",
                distance_locality_gate="IMPLEMENTED", sign_independent_normal_consistency="IMPLEMENTED", positional_continuity="IMPLEMENTED",
                normal_direction_separation="IMPLEMENTED", tangent_mutual_tangent_consistency="IMPLEMENTED", discontinuity_first_cuts="IMPLEMENTED",
                shared_neighbor_multi_edge_consensus="ABSENT", region_orientation_concentration="ABSENT", interface_consistency="ABSENT",
                bridge_veto="ABSENT", path_tangent_transport_consistency="ABSENT", singleton_propagation="ABSENT",
                ambiguity_handling="ABSENT", deterministic_tie_breaking="IMPLEMENTED",
            ),
        },
        {
            "candidate": "W99 interface-coherent intrinsic-normal merge",
            "status": "DIAGNOSTIC_ONLY",
            "contract": _cell(
                gaussian_normal_source="IMPLEMENTED", local_geometric_neighborhood="IMPLEMENTED", knn_definition="IMPLEMENTED",
                distance_locality_gate="IMPLEMENTED", sign_independent_normal_consistency="IMPLEMENTED", positional_continuity="IMPLEMENTED",
                normal_direction_separation="IMPLEMENTED", tangent_mutual_tangent_consistency="IMPLEMENTED", discontinuity_first_cuts="IMPLEMENTED",
                shared_neighbor_multi_edge_consensus="OPTIONAL", region_orientation_concentration="IMPLEMENTED", interface_consistency="IMPLEMENTED",
                bridge_veto="ABSENT", path_tangent_transport_consistency="ABSENT", singleton_propagation="ABSENT",
                ambiguity_handling="IMPLEMENTED", deterministic_tie_breaking="IMPLEMENTED",
            ),
        },
        {
            "candidate": "W100 bilateral region-conditioned intrinsic-normal merge",
            "status": "DIAGNOSTIC_ONLY",
            "contract": _cell(
                gaussian_normal_source="IMPLEMENTED", local_geometric_neighborhood="IMPLEMENTED", knn_definition="IMPLEMENTED",
                distance_locality_gate="IMPLEMENTED", sign_independent_normal_consistency="IMPLEMENTED", positional_continuity="IMPLEMENTED",
                normal_direction_separation="IMPLEMENTED", tangent_mutual_tangent_consistency="IMPLEMENTED", discontinuity_first_cuts="IMPLEMENTED",
                shared_neighbor_multi_edge_consensus="OPTIONAL", region_orientation_concentration="IMPLEMENTED", interface_consistency="IMPLEMENTED",
                bridge_veto="ABSENT", path_tangent_transport_consistency="ABSENT", singleton_propagation="ABSENT",
                ambiguity_handling="IMPLEMENTED", deterministic_tie_breaking="IMPLEMENTED",
            ),
        },
        {
            "candidate": "W10/W31-W38/W150 form_surface_regions",
            "status": "IMPLEMENTED",
            "contract": _cell(
                gaussian_normal_source="IMPLEMENTED", local_geometric_neighborhood="IMPLEMENTED", knn_definition="IMPLEMENTED",
                distance_locality_gate="IMPLEMENTED", sign_independent_normal_consistency="IMPLEMENTED", positional_continuity="IMPLEMENTED",
                normal_direction_separation="IMPLEMENTED", tangent_mutual_tangent_consistency="IMPLEMENTED", discontinuity_first_cuts="OPTIONAL",
                shared_neighbor_multi_edge_consensus="IMPLEMENTED", region_orientation_concentration="ABSENT", interface_consistency="OPTIONAL",
                bridge_veto="IMPLEMENTED", path_tangent_transport_consistency="IMPLEMENTED", singleton_propagation="ABSENT",
                ambiguity_handling="IMPLEMENTED", deterministic_tie_breaking="IMPLEMENTED",
            ),
        },
    ]


def build_report(root: Path) -> dict[str, Any]:
    runner_rel = "devtools/demo/candidate_f_gaussian_region_owned_tsdf.py"
    constructor_rel = "osn_gs/surface/torch_visible_surface_construction.py"
    formation_rel = "osn_gs/surface/torch_gaussian_surface_region_formation.py"
    w175_rel = "docs/worklogs/175_gaussian_normal_local_surface_lineage_audit.md"
    runner = _read(root, runner_rel)
    constructor = _read(root, constructor_rel)
    formation = _read(root, formation_rel)
    w175 = _read(root, w175_rel)

    w154_w97_refs = _line_refs(runner, r"partition_surfels_region_coherent\(active_orientation, RegionCoherenceConfig\(\)")
    w154_w150_refs = _line_refs(runner, r"form_surface_regions")
    w150_constructor_refs = _line_refs(constructor, r"regions\s*=\s*form_surface_regions\(")
    formation_refs = _line_refs(formation, r"def form_surface_regions\(")

    active_path_exact = bool(w154_w97_refs) and not w154_w150_refs
    w175_explicitly_separates = "W150" in w175 and "W97" in w175 and "호출하지 않는다" in w175

    witness = {
        "w97_baseline": {
            "row_4043": {"subset_id": 1, "membership_role": "core", "ambiguous": False},
            "row_4051": {"subset_id": 1, "membership_role": "core", "ambiguous": False},
            "path_subset_boundary_crossing": 0,
            "source": "W175 exact stored-ID join",
        },
        "comparison_contract": "NOT_COMPARABLE_NO_SINGLE_APPROVED_MATCH",
        "classification": "NOT_COMPARABLE",
    }

    report: dict[str, Any] = {
        "worklog": "W176",
        "title": "Historical Local Surface Decomposition Contract Reconstruction and W154 Architecture-Drift Attribution",
        "architecture_result": "CONTRACT_SPLIT_ACROSS_MULTIPLE_HISTORICAL_BRANCHES",
        "read_only_audit": True,
        "production_behavior_modified": False,
        "intent_alignment": {
            "preserved": ["W96/W97/W98/W99/W100", "W150", "W154/W155", "W171-W175", "historical parameters", "current production behavior"],
            "not_done": ["W97 tuning", "W150 adoption", "new hybrid", "TSDF ownership change", "NURBS fit", "continuation", "Eligibility", "UNRESOLVED"],
        },
        "historical_lineage": _lineage(),
        "decomposition_contract_matrix": _matrix(),
        "normal_alone_failure_safeguard_history": [
            {"worklog": "W96", "failure": "local pairwise normal compatibility single-linkage chains distant/orientation-changing surfaces", "safeguard": "none; failure measured", "result": "74.70% giant subset", "promotion": "not final"},
            {"worklog": "W97", "failure": "accumulated normal drift and anti-chaining", "safeguard": "region-level concentration floor derived from pairwise alignment", "result": "synthetic pass; real curved-surface over-fragmentation", "promotion": "not architecture-approved; retained as W154 baseline"},
            {"worklog": "W98", "failure": "smooth curvature is falsely cut by normal-only concentration; independent edge cuts still percolate", "safeguard": "shape-operator residual and positional normal-vs-tangent cut", "result": "synthetic pass; 94.51% real giant subset", "promotion": "diagnostic only"},
            {"worklog": "W99", "failure": "per-edge evidence merged across a complete interface and reintroduced chaining", "safeguard": "full-interface majority merge after positional-gated W97 init", "result": "mixed; 53.86% giant subset", "promotion": "diagnostic only"},
            {"worklog": "W100", "failure": "one-sided/boundary-contaminated interface evidence", "safeguard": "region-conditioned bilateral AND certificate", "result": "22.91% largest subset and all traced W99 lineage merges rejected", "promotion": "diagnostic only; no final architecture decision"},
            {"worklog": "W10/W31-W38", "failure": "pairwise affinity/weak bridge can fuse independent sheets", "safeguard": "shared-neighbor consensus, bridge veto, tangent/path checks, two-phase seed/merge", "result": "articulation bridge union 47→0 on controls", "promotion": "implemented in form_surface_regions lineage, but covariance-frame source"},
        ],
        "intended_final_local_surface_contract": {
            "requirements": {
                "A_intrinsic_tw_authority": "W96/W97/W98/W99/W100 satisfy; W10/W150 form_surface_regions uses covariance-derived frame instead",
                "B_gaussian_before_tsdf": "W97/W154 and W150 constructor are Gaussian-side; TSDF does not define the W97 ID",
                "C_approved_structural_safeguard_against_normal_only_aliasing": "W10/W150 satisfies structurally; W96-W100 safeguards are experimental branches without final promotion",
                "D_tsdf_not_surface_identity": "preserved in W154/W175 lineage; TSDF component is downstream support organization, not a new Gaussian identity",
            },
            "outcome": "CONTRACT_SPLIT_ACROSS_MULTIPLE_HISTORICAL_BRANCHES",
            "reason": "A+B+D are present in W96/W97 active lineage while C is supplied by the W10/W150 covariance-based lineage; no preserved implementation satisfies A-D together.",
            "do_not_compose": True,
        },
        "why_w154_uses_w97": {
            "active_runner": runner_rel,
            "w154_w97_call_lines": w154_w97_refs,
            "w154_form_surface_regions_call_lines": w154_w150_refs,
            "w150_constructor_call_lines": w150_constructor_refs,
            "formation_definition_lines": formation_refs,
            "exact_finding": "W154 explicitly chooses the W96/W97 intrinsic-t_w partition path; it does not call W150 form_surface_regions.",
            "approval_evidence": "No later preserved Worklog explicitly promotes W97 as the final contract. W175 records baseline identity preservation, not contract equivalence or architecture approval.",
            "drift_point": "W154 candidate runner's Gaussian identity constructor call at the W97 partition line, rather than the W150 form_surface_regions constructor path.",
            "determination": "EXPLICIT_PATH_DIFFERENCE; REASON_FOR_SELECTION_NOT_ARCHITECTURE-APPROVED_IN_PRESERVED_EVIDENCE",
        },
        "conditional_same_checkpoint_control": {
            "status": "NOT_RUN",
            "reason": "No single already-approved implementation matches intrinsic t_w authority plus the approved structural safeguard; running W150 or W100 alone would test a different contract or invent a hybrid interpretation.",
            "checkpoint_preserved": True,
        },
        "w174_witness_result": witness,
        "tabletop_region_result": {
            "w97_baseline": {
                "fixed_w171_support_rows": 17965,
                "selected_w171_support_rows": 15189,
                "subset_count_in_fixed_support": 1,
                "support_by_subset": {"1": {"complete_rows": 17965, "selected_rows": 15189}},
                "distinct_owner_gaussians": {"complete": 3315, "selected": 2887},
            },
            "comparison_contract": "NOT_RUN_NO_SINGLE_APPROVED_MATCH",
            "zero_set_ownership_recomputed": False,
        },
        "zero_set_semantic_contract": {
            "gaussian_local_surface_decomposition": "WHO",
            "tsdf_zero_set_samples": "WHERE",
            "same_subset_may_be_spatially_sparse": True,
            "native_tsdf_component_is_new_identity": False,
            "structural_nurbs_carrier_equals_observed_membership": False,
        },
        "visual_review": {
            "status": "NOT_GENERATED_NO_COMPARABLE_APPROVED_CONTRACT",
            "retained_existing_visuals": "W175 W97 lineage review exports and prior W150/W154-W175 artifacts",
            "reason": "The requested A/B comparison requires a single approved historical comparison contract; the audit found a split, so generating a visual comparison would imply a contract that does not exist.",
        },
        "architecture_attribution": {
            "result": "W154_CONSUMES_OLDER_OR_DIFFERENT_W97_CONTRACT_THAN_W150_FORM_SURFACE_REGIONS",
            "identity_lineage": "W97 subset_id -> W154 region_id -> W171 zero-set ownership is preserved",
            "architecture_drift": "W154 active identity path is W97 intrinsic-t_w region-coherent partition; W150 covariance/consensus/bridge/path constructor is not consumed.",
            "interpretation": "This is an architecture contract split/drift finding, not proof that W150 should replace W97 and not proof that W97 identity is wrong.",
        },
        "promoted_retained_rejected_open": {
            "promoted": ["W10/W31-W38 form_surface_regions structural semantics in its own covariance-based lineage", "W96/W97 intrinsic t_w as W154 active baseline identity only"],
            "retained": ["exact W97/W154/W171 lineage", "Gaussian WHO vs TSDF WHERE", "W98-W100 diagnostic evidence", "W150 bypass evidence"],
            "rejected": ["W97 declared final merely from W175 identity preservation", "W150 automatically substituted", "W98/W99/W100 automatically promoted", "new hybrid decomposition", "TSDF component-based identity"],
            "open": ["architecture decision for an intrinsic-t_w contract with approved structural safeguards", "same-checkpoint W97↔W150 per-Gaussian crosswalk", "whether W150 covariance frame or 2DGS intrinsic t_w is the final orientation authority"],
        },
        "source_assertions": {
            "w154_active_path_detected": active_path_exact,
            "w175_preserves_w150_w97_distinction": w175_explicitly_separates,
            "matrix_status_values_valid": all(value in STATUS_VALUES for row in _matrix() for value in row["contract"].values()),
        },
    }
    return report


def _write_output(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "worklog_176_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Worklog 176 감사 산출물\n\n"
        "이 디렉터리는 역사적 Local Surface Decomposition 계약을 읽기 전용으로 재구성한 report를 담습니다.\n\n"
        "- `worklog_176_report.json`: W96–W175 계보, decomposition contract matrix, W154 active path, same-checkpoint control 보류 사유.\n"
        "- 이번 결과는 visualization 비교가 아니라 contract audit입니다. 단일 승인 계약이 확인되지 않아 새 비교 시각화를 만들지 않았습니다.\n"
        "- production code, checkpoint, W154/W175 artifact, TSDF ownership, NURBS 동작은 변경하지 않았습니다.\n",
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
    print(json.dumps({"architecture_result": report["architecture_result"], "control": report["conditional_same_checkpoint_control"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
