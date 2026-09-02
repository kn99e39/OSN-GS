"""Worklog 151: renderer-event compatibility audit.

This is a diagnostic-only follow-up to WL150.  It extracts the executable
canonical Local Surface Decomposition / Boundary First input contract and
compares it with the frozen WL145 renderer-median event schema.  The audit is
intentionally fail-closed: because the event evidence lacks canonical region,
adjacency, and physical-sheet ownership semantics, it does not implement a
restoration candidate or invent a mapping.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from devtools.demo import boundary_first_local_surface_decomposition_contract_trace as wl150


REPO_ROOT = wl150.REPO_ROOT
OUTPUT_ROOT = REPO_ROOT / "output" / "151_renderer_event_canonical_surface_compatibility_audit"
TEMP_ROOT = REPO_ROOT / "temp" / "151_renderer_event_canonical_surface_compatibility_audit"

CANONICAL_CONSTRUCTION = wl150.CANONICAL_CONSTRUCTION
CANONICAL_PIPELINE = wl150.CANONICAL_PIPELINE
BOUNDARY_BUILDER = wl150.BOUNDARY_BUILDER
BOUNDARY_ADAPTER = wl150.BOUNDARY_ADAPTER
REGION_FORMATION = wl150.REGION_FORMATION
DECOMPOSITION_DIAGNOSTIC = wl150.DECOMPOSITION_DIAGNOSTIC
WL139_MODULE = wl150.WL139_MODULE
WL144_MODULE = wl150.WL144_MODULE
WL145_MODULE = wl150.WL145_MODULE
WL148_MODULE = wl150.WL148_MODULE
WL149_MODULE = wl150.WL149_MODULE


def _source_inventory() -> dict[str, Any]:
    return {
        "canonical_constructor": wl150._source_ref(
            CANONICAL_CONSTRUCTION,
            "construct_visible_nurbs_from_gaussians",
            ("covariance", "stable_ids", "evaluate_structural_reliability", "build_manifold_affinity_graph", "form_surface_regions", "extract_world_space_boundary_halfedge_candidates", "materialize_visible_boundary_component"),
        ),
        "canonical_initialize": wl150._source_ref(
            CANONICAL_PIPELINE,
            "_initialize_canonical",
            ("construction_points", "stable_ids", "materialized_visible_nurbs_surfaces", "_assign_uv_support_masks"),
        ),
        "canonical_region_formation": wl150._source_ref(
            REGION_FORMATION,
            "form_surface_regions",
            ("positions", "frame", "reliability", "graph", "ids", "node_region_id", "member_ids"),
        ),
        "canonical_boundary_builder": wl150._source_ref(
            BOUNDARY_BUILDER,
            "build_boundary_first_visible_surface",
            ("outer_loops", "_materialize_boundary_role_network", "component_points"),
        ),
        "canonical_boundary_adapter": wl150._source_ref(
            BOUNDARY_ADAPTER,
            "materialize_visible_boundary_component",
            ("ordered_closed_loop", "branch_node_ids", "source_region_id", "boundary_ids", "interior_ids", "fit_torch_visible_surface_lsq"),
        ),
        "diagnostic_decomposition_only": wl150._source_ref(
            DECOMPOSITION_DIAGNOSTIC,
            "build_proxy_surface_components_diagnostics",
            ("Diagnostics-only", "not called by the production boundary-first constructor"),
        ),
        "renderer_event_generation": wl150._source_ref(
            WL144_MODULE,
            "_build_per_view_event_cloud",
            ("renderer_median_depth", "event_points_xyz", "local_normals", "source_camera"),
        ),
        "renderer_event_save": wl150._source_ref(
            WL145_MODULE,
            "_save_candidate_outputs",
            ("event_cloud_with_provenance.npz", "source_camera"),
        ),
        "wl145_union": wl150._source_ref(
            WL145_MODULE,
            "run_audit",
            ("clean_points = np.concatenate", "_pca_chart_config", "RAW_VISIBLE_SURFACE"),
        ),
        "wl139_fitter": wl150._source_ref(
            WL139_MODULE,
            "fit_physical_chart_surface",
            ("fit_indices = deterministic_indices", "_fixed_physical_uv", "fit_uv, fit_n"),
        ),
        "wl148_replay": wl150._source_ref(
            WL148_MODULE,
            "_load_frozen_baseline",
            ("frozen", "representative"),
        ),
        "wl149_attribution": wl150._source_ref(
            WL149_MODULE,
            "run_audit",
            ("full_per_point_influence", "event_id", "v_min"),
        ),
    }


def _canonical_local_surface_contract(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_name": "canonical Local Surface Decomposition",
        "population": "bounded construction-point population supplied to the canonical constructor; stable IDs must be unique and match the population",
        "node_entity_type": "Gaussian/evidence node represented by world position plus covariance-derived structural frame and stable source identity",
        "geometric_quantities_required": [
            "world positions",
            "Gaussian covariance, or log_scales plus rotations (exactly one representation)",
            "covariance-derived normal candidate, tangent scales, and thickness",
            "resolved local graph scale and residual scale",
        ],
        "adjacency_topology_required": [
            "manifold affinity graph",
            "same_surface / crease / parallel_separate / rejected relation evidence",
            "shared-neighbor consensus and bridge-edge/path consistency checks",
        ],
        "normal_orientation_required": "covariance frame normal candidates, then orientation along accepted topology before boundary extraction",
        "visibility_evidence_state_required": "structural reliability classes and contextual consistency; rejected/ambiguous/core membership states are explicit",
        "ownership_provenance_required": [
            "unique stable IDs aligned to input positions",
            "node_region_id",
            "SurfaceRegionCandidate.region_id",
            "member_ids, core_member_ids, attached and rejected IDs",
        ],
        "graph_construction_source": "build_manifold_affinity_graph from positions, covariance frame, reliability, fixed canonical affinity policy, and stable IDs",
        "fixed_policy_or_scales": "canonical RegionFormationConfig/Affinity config and covariance-derived scales; no new values may be invented by this audit",
        "output_region_identity": "RegionFormationResult.regions with region_id, region state, confidence, contradiction/consistency diagnostics, and provenance",
        "output_member_identity": "RegionFormationResult.node_region_id and SurfaceRegionCandidate member/core/attached IDs",
        "mechanics_required_for_invocation": [
            "positions",
            "covariance or log_scales+rotations",
            "unique stable IDs",
            "reliability result or the canonical reliability computation",
            "manifold affinity graph or its canonical construction inputs",
        ],
        "semantics_required_to_mean_one_local_physical_surface": [
            "same-surface relation rather than mere spatial/camera proximity",
            "coherent local topology and region ownership",
            "normal/tangent consistency with crease and parallel-separate distinctions",
            "provenance that permits every member to be assigned, rejected, or left ambiguous",
        ],
        "source_evidence": inventory,
    }


def _canonical_boundary_first_contract(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_name": "canonical Boundary First ownership/materialization",
        "required_region_input": "accepted SurfaceRegionCandidate regions plus canonical tangent frames and oriented normals",
        "boundary_candidate_source": "world-space boundary halfedge candidates derived from positions, oriented normals, regions, graph, stable IDs, and support/termination semantics",
        "boundary_adjacency_order_semantics": "directed compatible halfedges are recovered into OrderedBoundaryComponent objects with ordered source IDs",
        "closed_open_branch_ambiguous_semantics": {
            "ordered_closed_loop_outer": "eligible for pre-fit materialization after simple-loop validation",
            "open": "unsupported/review; no synthetic rectangle",
            "branch": "unsupported/review; no synthetic rectangle",
            "ambiguous": "unsupported/review; no synthetic rectangle",
        },
        "interior_core_requirement": "region-core interior IDs/points are supplied alongside ordered boundary points; reliable-core-only coverage is explicit",
        "ownership_requirement": [
            "source region ID",
            "source boundary component ID",
            "ordered boundary point IDs",
            "interior reliable point IDs",
            "supporting source IDs and region status provenance",
        ],
        "exact_prefit_adapter_input": "materialize_visible_boundary_component(component, boundary_points, interior_points, boundary_ids, interior_ids, region status/provenance)",
        "domain_control_timing": "PRE_FIT for eligible boundary-plus-interior input; later support occupancy is a distinct post-fit concern",
        "mechanics_required_for_invocation": [
            "region-owned boundary halfedges",
            "ordered directed component",
            "valid closed-loop state and no branch nodes",
            "world boundary points",
            "region-core interior points",
            "stable boundary/interior IDs",
        ],
        "semantics_required_to_mean_physical_boundary": [
            "boundary is owned by the correct local physical region",
            "adjacency/order expresses one observed surface boundary",
            "closed/open/branch/ambiguous state is meaningful for that region",
            "interior/core support belongs to the same region",
        ],
        "source_evidence": inventory,
    }


def _renderer_event_contract() -> dict[str, Any]:
    return {
        "contract_name": "frozen WL145 renderer median-event evidence",
        "source_artifact": "output/confirmed/145_genuine_physical_sheet_oracle_clean_support_representative_audit/tabletop_broad_planar_clean/per_view_renderer_median_events/*/event_cloud_with_provenance.npz",
        "fields_directly_present": [
            "world XYZ (event_points_xyz)",
            "renderer median event depth",
            "source camera",
            "source pixel",
            "event normal (local_normals)",
            "per-view camera provenance",
            "unique frozen union row/event ID after deterministic WL149 row-order replay",
        ],
        "fields_absent": [
            "primitive/contributor identity",
            "visible-surface topology identity",
            "canonical local region identity",
            "canonical node_region_id",
            "boundary halfedge/component ownership",
            "ordered boundary relation",
            "Gaussian covariance/log_scales/rotations aligned to each event",
            "canonical structural reliability result",
            "canonical manifold affinity graph",
            "physical-sheet identity as an executable field",
        ],
        "semantic_warnings": [
            "renderer contribution provenance != physical-sheet membership",
            "manual source polygon/control label != physical-sheet identity",
            "camera co-visibility or common image control != surface topology",
            "event normal != region identity",
            "spatial proximity != ownership",
            "event row ID != canonical region/member ownership",
        ],
        "event_1527_human_update": "HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE",
        "source_evidence": wl150._source_ref(WL144_MODULE, "_build_per_view_event_cloud", ("renderer_median_depth", "event_points_xyz", "local_normals")),
    }


def _compatibility_matrix(canonical_local: dict[str, Any], canonical_boundary: dict[str, Any], renderer: dict[str, Any]) -> list[dict[str, Any]]:
    # Classification is intentionally about semantic availability, not about
    # whether a new inference rule could be invented to fill the gap.
    rows = [
        ("local node positions", "world XYZ", "COMPATIBLE", "event_points_xyz is directly present"),
        ("canonical covariance/frame", "covariance, log_scales, rotations, tangent scale, thickness", "INCOMPATIBLE_MISSING", "renderer event schema has no Gaussian covariance representation"),
        ("unique stable row identity", "frozen event/row ID", "COMPATIBLE_BY_EXISTING_DETERMINISTIC_MAPPING", "WL149 row order gives a deterministic unique ID, but this is not ownership"),
        ("structural reliability", "intrinsic/contextual reliability classes", "INCOMPATIBLE_MISSING", "no canonical reliability result is present in event artifacts"),
        ("manifold affinity graph", "same_surface/crease/parallel_separate/rejected edge relations", "INCOMPATIBLE_MISSING", "no graph or already-established event-to-graph mapping exists"),
        ("normal candidate", "event normal", "SEMANTICALLY_DIFFERENT", "a normal is present but does not establish coherent region identity/orientation along topology"),
        ("visibility/evidence state", "renderer median depth and camera/pixel provenance", "SEMANTICALLY_DIFFERENT", "renderer event provenance is not structural reliability or visible-topology state"),
        ("physical-sheet identity", "manual control label / human review", "SEMANTICALLY_DIFFERENT", "manual polygon label is not executable membership; event 1527 is explicitly reviewed off-sheet"),
        ("region ownership", "node_region_id / SurfaceRegionCandidate", "INCOMPATIBLE_MISSING", "no region ID is created or carried by WL145 event clouds"),
        ("member provenance", "member/core/attached/rejected IDs", "INCOMPATIBLE_MISSING", "camera/pixel provenance cannot replace region member state"),
        ("boundary candidate source", "region-owned world halfedges", "INCOMPATIBLE_MISSING", "event evidence has no boundary relation or region-owned candidate"),
        ("boundary adjacency/order", "directed compatible halfedges → ordered component", "INCOMPATIBLE_MISSING", "no topology/adjacency/order is present"),
        ("closed/open/branch/ambiguous state", "OrderedBoundaryComponent state", "INCOMPATIBLE_MISSING", "component state cannot be derived without boundary graph semantics"),
        ("region-core interior", "reliable core IDs/points for same region", "INCOMPATIBLE_MISSING", "no core membership is present"),
        ("pre-fit adapter ownership", "source_region_id + component + ordered boundary/interior IDs", "INCOMPATIBLE_MISSING", "required adapter ownership tuple is absent"),
        ("renderer depth → world position", "camera/pixel/depth reconstruction", "COMPATIBLE_BY_EXISTING_DETERMINISTIC_MAPPING", "already materialized by the frozen renderer-event extraction; no new mapping needed"),
    ]
    return [{"canonical_requirement": req, "renderer_availability": available, "classification": classification, "reason": reason} for req, available, classification, reason in rows]


def _adapter_gate(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    incompatible = [row for row in matrix if row["classification"].startswith("INCOMPATIBLE")]
    semantic = [row for row in matrix if row["classification"] == "SEMANTICALLY_DIFFERENT"]
    return {
        "status": "CONTRACT_GAP",
        "eligible": False,
        "stop_condition": "A",
        "reason": "Canonical Local Surface Decomposition / Boundary First invocation would require missing ownership/topology/covariance semantics, or reinterpret renderer/manual fields as those semantics.",
        "incompatible_missing_requirements": incompatible,
        "incompatible_semantics_requirements": semantic,
        "new_mapping_needed": [
            "physical-sheet membership",
            "local region ownership",
            "manifold adjacency/topology",
            "boundary ownership/order",
            "Gaussian covariance/reliability inputs aligned to renderer events",
        ],
        "forbidden_invented_mechanisms": [
            "distance/KNN threshold",
            "normal-angle threshold",
            "correspondence heuristic",
            "primitive/contributor inference",
            "physical-sheet classifier",
            "largest-component or outlier rule",
        ],
        "candidate_c_not_implemented": True,
        "synthetic_contracts_not_run": True,
        "real_scene_replay_not_run": True,
        "qualitative_exports_not_run": True,
    }


def _extrema_owner_accounting(baseline: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    report = wl150._read_json(wl150.WL149_REPORT)
    rows = report["EXTREMA OWNERSHIP"]["owner_provenance"]
    return {
        "source": "frozen WL149 EXTREMA OWNERSHIP and full_per_point_influence.json",
        "owners": [
            {
                "event_id": int(row["event_id"]),
                "source_camera": row["source_camera"],
                "chart_u": row["chart_u"],
                "chart_v": row["chart_v"],
                "historical_role": "u_min/u_max/v_min/v_max owner",
                "candidate_c_assignment": "NOT_RUN_DUE_TO_CONTRACT_GAP",
                "canonical_region_assignment": "NOT_REPRESENTABLE_FROM_RENDERER_EVENT_SCHEMA",
            }
            for row in rows
        ],
        "event_1527": {
            "human_review": wl150.HUMAN_REVIEW_PHYSICAL_SHEET_STATUS,
            "historical_v_min_owner": True,
            "candidate_c_assignment": "NOT_RUN_DUE_TO_CONTRACT_GAP",
            "remains_present": True,
        },
        "no_four_event_rejection_rule": True,
        "baseline_event_count": baseline["event_count"],
    }


def _source_history() -> dict[str, Any]:
    result = wl150._source_history_manifest()
    result["worklog151_source_module"] = {"path": wl150._relative(Path(__file__)), "sha256": wl150._sha256_file(Path(__file__))}
    result["prior_contract_trace"] = "temp/150_boundary_first_local_surface_decomposition_contract_trace"
    result["compatibility_audit_only"] = True
    return result


def build_audit() -> dict[str, Any]:
    baseline = wl150._load_wl149_baseline()
    event = wl150._load_event_1527(baseline)
    inventory = _source_inventory()
    canonical_local = _canonical_local_surface_contract(inventory)
    canonical_boundary = _canonical_boundary_first_contract(inventory)
    renderer = _renderer_event_contract()
    matrix = _compatibility_matrix(canonical_local, canonical_boundary, renderer)
    gate = _adapter_gate(matrix)
    owners = _extrema_owner_accounting(baseline, event)
    return {
        "baseline_reconciliation": {
            "wl149": baseline,
            "wl150_contract_trace": "temp/150_boundary_first_local_surface_decomposition_contract_trace",
            "exact": True,
        },
        "canonical_local_surface_contract": canonical_local,
        "canonical_boundary_first_contract": canonical_boundary,
        "renderer_event_evidence_contract": renderer,
        "compatibility_matrix": matrix,
        "adapter_eligibility_verdict": gate,
        "conditional_candidate_status": {
            "candidate": "Candidate C — Canonical Pre-Fit Ownership Restoration",
            "status": "NOT_IMPLEMENTED_STOP_CONDITION_A",
            "reason": "Compatibility gate failed; implementing it would require new semantics rather than representation plumbing.",
        },
        "extrema_owner_accounting": owners,
        "synthetic_contract_status": {"status": "NOT_RUN", "reason": "STOP CONDITION A / CONTRACT_GAP"},
        "real_scene_replay_status": {"status": "NOT_RUN", "reason": "STOP CONDITION A / CONTRACT_GAP", "historical_arms_preserved": ["WL149 Arm A", "WL148 Arm B"]},
        "qualitative_review_status": {"status": "NOT_RUN", "reason": "No Candidate C geometry is legal under the compatibility gate; existing WL149 visuals remain historical."},
        "source_history_manifest": _source_history(),
        "architecture_verdict": {
            "verdict": "A. CONTRACT_GAP",
            "compatibility": False,
            "why": "Renderer events contain geometric/rendering provenance but not the canonical local-region, topology, boundary, covariance/reliability, or physical-sheet ownership semantics.",
            "event_1527": "not representable under the canonical input contract; not manually removed",
            "candidate_c": "not implemented",
            "canonical_production_modified": False,
        },
        "report": {
            "1. CURRENT QUESTION": "Can frozen renderer-event evidence legitimately enter the existing canonical Local Surface Decomposition / Boundary First ownership contract without inventing semantics?",
            "2. WL150 BASELINE RECONCILIATION": {"exact": True, "event_count": baseline["event_count"], "event_union_sha256": baseline["event_union_sha256"], "event_1527": event, "representative_shape": baseline["representative_shape"], "support_vertices": baseline["support_vertices"], "support_mask_sha256": baseline["support_mask_sha256"], "all_four_cells": baseline["fully_supported_cells"]},
            "3. CANONICAL LOCAL-SURFACE CONTRACT": canonical_local,
            "4. CANONICAL BOUNDARY-FIRST CONTRACT": canonical_boundary,
            "5. RENDERER-EVENT EVIDENCE CONTRACT": renderer,
            "6. COMPATIBILITY MATRIX": matrix,
            "7. ADAPTER ELIGIBILITY VERDICT": gate,
            "8. CONDITIONAL CANDIDATE IMPLEMENTATION": {"status": "NOT_IMPLEMENTED", "reason": "CONTRACT_GAP stop condition A"},
            "9. EXTREMA OWNER ACCOUNTING": owners,
            "10. SYNTHETIC CONTRACTS": {"status": "NOT_RUN", "reason": "CONTRACT_GAP"},
            "11. REAL-SCENE QUANTITATIVE RESULT": {"status": "NOT_RUN", "reason": "CONTRACT_GAP; no Candidate C replay"},
            "12. REAL-SCENE QUALITATIVE REVIEW": {"status": "NOT_RUN", "reason": "CONTRACT_GAP; no Candidate C geometry/export"},
            "13. ARCHITECTURE VERDICT": "A. CONTRACT_GAP",
            "14. RETAINED / REJECTED / OPEN": {
                "RETAINED": ["WL139/WL145/WL148/WL149/WL150", "1586 event union and event 1527", "renderer median semantics", "canonical Local Surface Decomposition and Boundary First source", "PCA/representative/support historical results"],
                "REJECTED": ["blind constructor reconnection", "event deletion", "new membership/correspondence/adjacency heuristic", "Candidate C implementation under missing semantics", "synthetic and real-scene claims based on an invented adapter"],
                "OPEN": ["a separate architecture batch defining physical-sheet membership and renderer-event-to-region semantics", "whether Gaussian covariance/reliability can be aligned to renderer events without changing meaning", "how canonical boundary ownership should consume future validated evidence"],
            },
            "INTENT ALIGNMENT": {"diagnostic_only": True, "stop_condition_A_honored": True, "no_geometry_change": True, "no_threshold_tuning": True, "event_1527_removed": False, "candidate_c_implemented": False},
            "IMPLEMENTATION FIDELITY": {"frozen_wl149_replayed": True, "source_history_inspected": True, "canonical_source_modified": False, "historical_artifacts_modified": False, "new_semantics_invented": False, "new_heuristic_added": False},
            "ARCHITECTURE RESULT": {"verdict": "A. CONTRACT_GAP", "compatibility": False, "conditional_sections_skipped": ["Candidate C", "synthetic contracts", "real-scene replay", "qualitative Candidate C exports"]},
        },
    }


def write_audit() -> dict[str, Any]:
    audit = build_audit()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "baseline_reconciliation.json": audit["baseline_reconciliation"],
        "canonical_local_surface_contract.json": audit["canonical_local_surface_contract"],
        "canonical_boundary_first_contract.json": audit["canonical_boundary_first_contract"],
        "renderer_event_evidence_contract.json": audit["renderer_event_evidence_contract"],
        "compatibility_matrix.json": audit["compatibility_matrix"],
        "adapter_eligibility_verdict.json": audit["adapter_eligibility_verdict"],
        "conditional_candidate_status.json": audit["conditional_candidate_status"],
        "extrema_owner_accounting.json": audit["extrema_owner_accounting"],
        "synthetic_contract_status.json": audit["synthetic_contract_status"],
        "real_scene_replay_status.json": audit["real_scene_replay_status"],
        "qualitative_review_status.json": audit["qualitative_review_status"],
        "source_history_manifest.json": audit["source_history_manifest"],
        "architecture_verdict.json": audit["architecture_verdict"],
        "renderer_event_canonical_surface_compatibility_audit_report.json": audit["report"],
    }
    for name, value in payloads.items():
        wl150._write_json(OUTPUT_ROOT / name, value)
    matrix_lines = ["# WL151 compatibility matrix", "", "| Canonical requirement | Renderer availability | Classification |", "|---|---|---|"]
    matrix_lines.extend(f"| {row['canonical_requirement']} | {row['renderer_availability']} | **{row['classification']}** |" for row in audit["compatibility_matrix"])
    matrix_lines.extend(["", "## Gate", "", "**CONTRACT_GAP** — Stop Condition A. Candidate C, synthetic contracts, and real-scene replay were not run.", "", "`HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE` applies only to event 1527."])
    (OUTPUT_ROOT / "compatibility_matrix.md").write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")
    readme = """# Worklog 151 — Renderer-event / canonical ownership compatibility audit

Diagnostic-only, fail-closed audit. WL149/WL150 are replayed exactly. The
renderer-event schema is missing canonical local-region, topology, boundary,
covariance/reliability, and physical-sheet ownership semantics.

- Verdict: **A. CONTRACT_GAP**
- Candidate C: not implemented
- Synthetic contracts: not run
- Real-scene replay: not run
- Event 1527: preserved; `HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE`

See `compatibility_matrix.md` and the numbered JSON files.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_ROOT.iterdir():
        destination = TEMP_ROOT / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)
    return {"output_root": wl150._relative(OUTPUT_ROOT), "temp_root": wl150._relative(TEMP_ROOT), "files": sorted(payloads) + ["compatibility_matrix.md", "README.md"], "verdict": audit["architecture_verdict"]["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit renderer-event compatibility with canonical ownership")
    parser.add_argument("--write", action="store_true", help="write numbered output and temp artifacts")
    arguments = parser.parse_args()
    result = write_audit() if arguments.write else build_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=wl150._json_default))


if __name__ == "__main__":
    main()
