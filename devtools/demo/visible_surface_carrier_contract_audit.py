"""Worklog 152: audit Raw Visible Surface as a topology carrier.

The audit is intentionally read-only with respect to the canonical pipeline
and historical outputs.  It reconciles WL149--WL151, inspects the available
WL127 artifact and committed TSDF source, and stops before any Candidate D
when the actual carrier/provenance contract is unavailable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from devtools.demo import boundary_first_local_surface_decomposition_contract_trace as wl150


REPO_ROOT = wl150.REPO_ROOT
OUTPUT_ROOT = REPO_ROOT / "output" / "152_visible_surface_carrier_contract_audit"
TEMP_ROOT = REPO_ROOT / "temp" / "152_visible_surface_carrier_contract_audit"

RAW_VISIBLE_SURFACE = (
    REPO_ROOT
    / "output"
    / "confirmed"
    / "127_osn_gs_evidence_bounded_projective_tsdf"
    / "RENDERER_MEDIAN_SURFACE_POINTS"
    / "iteration_0000001"
    / "point_cloud.ply"
)
WL127_ROOT = REPO_ROOT / "output" / "confirmed" / "127_osn_gs_evidence_bounded_projective_tsdf"
WL127_DOC = REPO_ROOT / "docs" / "worklogs" / "127_evidence_bounded_projective_tsdf.md"
WL127_EXTRACTION = REPO_ROOT / "scripts" / "devtools" / "evidence_bounded_tsdf" / "extraction.py"
WL127_MESH_OPS = REPO_ROOT / "scripts" / "devtools" / "evidence_bounded_tsdf" / "mesh_ops.py"
WL127_DRIVER = REPO_ROOT / "scripts" / "devtools" / "evidence_bounded_tsdf_stages.py"
WL149_OUTPUT_ROOT = REPO_ROOT / "output" / "149_physical_sheet_evidence_vs_chart_extent_failure_attribution"


def _load_wl149_baseline_from_preserved_output() -> dict[str, Any]:
    """Replay WL149 from the available numbered output without copying or
    rewriting any historical folder.

    WL150's loader is intentionally reused, but its paths are temporarily
    pointed at the existing output mirror because the working tree currently
    lacks the WL149 temp report while the numbered output mirror is present.
    """

    original = (wl150.WL149_ROOT, wl150.WL149_REPORT, wl150.WL149_INFLUENCE, wl150.WL149_INFLUENCE_NPZ)
    wl150.WL149_ROOT = WL149_OUTPUT_ROOT
    wl150.WL149_REPORT = WL149_OUTPUT_ROOT / "physical_sheet_evidence_vs_chart_extent_failure_attribution_report.json"
    wl150.WL149_INFLUENCE = WL149_OUTPUT_ROOT / "full_per_point_influence.json"
    wl150.WL149_INFLUENCE_NPZ = WL149_OUTPUT_ROOT / "full_per_point_influence.npz"
    try:
        return wl150._load_wl149_baseline()
    finally:
        wl150.WL149_ROOT, wl150.WL149_REPORT, wl150.WL149_INFLUENCE, wl150.WL149_INFLUENCE_NPZ = original


def _parse_ply_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise AssertionError("Raw Visible Surface PLY has no end_header")
    header_text = raw[: end + len(marker)].decode("ascii")
    elements: dict[str, int] = {}
    properties: dict[str, list[str]] = {}
    current: str | None = None
    for line in header_text.splitlines():
        words = line.split()
        if len(words) >= 3 and words[0] == "element":
            current = words[1]
            elements[current] = int(words[2])
            properties[current] = []
        elif len(words) >= 3 and words[0] == "property" and current is not None:
            properties[current].append(words[-1])
    return {
        "path": wl150._relative(path),
        "sha256": wl150._sha256_file(path),
        "bytes": path.stat().st_size,
        "format": next((line for line in header_text.splitlines() if line.startswith("format ")), None),
        "elements": elements,
        "properties": properties,
        "header": header_text,
        "has_faces": "face" in elements and elements["face"] > 0,
    }


def _source_inventory() -> dict[str, Any]:
    return {
        "wl127_extracted_surface_entity": {
            "path": wl150._relative(WL127_EXTRACTION),
            "matched_lines": wl150._line_matches(WL127_EXTRACTION, ("class ExtractedSurface", "vertices: np.ndarray", "faces: np.ndarray", "vertex_support_count", "vertex_field_value")),
            "extract_function": wl150._source_ref(WL127_EXTRACTION, "extract_zero_level_set", ("corner_ok", "marching_cubes", "weld_block_seams", "faces=")),
        },
        "wl127_mesh_connectivity": wl150._source_ref(WL127_MESH_OPS, "connected_components", ("faces.shape[0] == 0", "Vertex-connectivity components")),
        "wl127_mesh_export": wl150._source_ref(WL127_MESH_OPS, "write_mesh_ply", ("element face", "vertex_indices", "support")),
        "wl127_stage_replay": wl150._source_ref(WL127_DRIVER, "replay_historical_visible_nurbs", ("mesh", "surface", "historical")),
        "wl127_semantics": {"path": wl150._relative(WL127_DOC), "sha256": wl150._sha256_file(WL127_DOC), "matched_lines": wl150._line_matches(WL127_DOC, ("raw renderer median surface point cloud", "추출 mesh component", "renderer-evidence"))},
        "wl145_raw_surface_reference": wl150._source_ref(wl150.WL145_MODULE, "run_audit", ("RAW_VISIBLE_SURFACE", "_load_xyz_ply")),
    }


def _baseline_reconciliation() -> dict[str, Any]:
    baseline = _load_wl149_baseline_from_preserved_output()
    event = wl150._load_event_1527(baseline)
    mesh_cache = WL127_ROOT / "_cache" / "mesh.npz"
    field_cache = WL127_ROOT / "_cache" / "field.npz"
    raw_artifact = _parse_ply_header(RAW_VISIBLE_SURFACE) if RAW_VISIBLE_SURFACE.exists() else {"exists": False}
    return {
        "wl149_wl151_exact": True,
        "event_count": baseline["event_count"],
        "event_union_sha256": baseline["event_union_sha256"],
        "event_1527": {
            "event_id": event["event_id"],
            "source_camera": event["source_camera"],
            "source_pixel": [event["source_pixel_x"], event["source_pixel_y"]],
            "world_xyz": event["world_xyz"],
            "v_min_owner": event["pca_v_min_owner"],
            "human_review": wl150.HUMAN_REVIEW_PHYSICAL_SHEET_STATUS,
        },
        "historical_extrema_owner_ids": [795, 947, 1104, 1527],
        "representative_shape": baseline["representative_shape"],
        "support_vertices": baseline["support_vertices"],
        "support_mask_sha256": baseline["support_mask_sha256"],
        "all_four_supported_cells": baseline["fully_supported_cells"],
        "raw_visible_surface_artifact": raw_artifact,
        "raw_visible_surface_exact_replay": {
            "expected_source_commit": "943a764",
            "mesh_cache_path": wl150._relative(mesh_cache),
            "mesh_cache_exists": mesh_cache.exists(),
            "field_cache_path": wl150._relative(field_cache),
            "field_cache_exists": field_cache.exists(),
            "available_artifact_is_the_extracted_tsdf_mesh": False,
            "status": "AVAILABLE_POINT_CLOUD_ONLY_TSDF_MESH_REPLAY_UNAVAILABLE",
        },
    }


def _renderer_event_contract() -> dict[str, Any]:
    return {
        "scientific_meaning": "renderer-grounded observation event from a camera pixel and renderer median-depth reconstruction",
        "direct_properties": ["world XYZ", "renderer median-event depth", "source camera", "source pixel", "event normal", "camera provenance"],
        "does_not_prove": ["physical-sheet identity", "visible topology", "local region ownership", "boundary ownership"],
        "event_1527": "HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE",
        "source": "WL145 per-view event_cloud_with_provenance.npz / WL144 renderer event helper",
    }


def _canonical_gaussian_region_contract() -> dict[str, Any]:
    return {
        "entity": "Gaussian construction node",
        "properties": ["position", "covariance or log_scales+rotations", "covariance frame", "structural reliability", "manifold affinity graph", "stable source ID"],
        "outputs": ["node_region_id", "SurfaceRegionCandidate.region_id", "member/core/attached/rejected IDs", "region state/confidence"],
        "boundary_relation": "region-owned topology and boundary candidate/order/status, not renderer event membership",
        "not_available_on_renderer_events": True,
    }


def _raw_visible_surface_contract(header: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    properties = header.get("properties", {}).get("vertex", [])
    return {
        "artifact": header,
        "actual_entity": "vertex-only PLY point artifact under RENDERER_MEDIAN_SURFACE_POINTS; not the committed ExtractedSurface mesh entity",
        "property_contract": {
            "XYZ": {"status": "DIRECTLY_PRESENT", "evidence": properties[:3]},
            "mesh_vertex": {"status": "SEMANTICALLY_DIFFERENT", "evidence": "vertex record exists, but no face graph establishes a mesh vertex"},
            "mesh_face": {"status": "ABSENT", "evidence": "PLY header has no element face"},
            "zero_set_sample": {"status": "SEMANTICALLY_DIFFERENT", "evidence": "available artifact is renderer-median point output; TSDF ExtractedSurface cache is absent"},
            "connectivity": {"status": "ABSENT", "evidence": "no faces/edges in available artifact"},
            "face_edge_adjacency": {"status": "ABSENT", "evidence": "no face list"},
            "connected_component_identity": {"status": "ABSENT", "evidence": "cannot call face-based connected_components on this artifact"},
            "topological_boundary_edges": {"status": "ABSENT", "evidence": "no edge graph; boundary count is not meaningful"},
            "source_TSDF_cell": {"status": "ABSENT", "evidence": "no cell key/voxel ID in PLY properties"},
            "source_camera_evidence_provenance": {"status": "ABSENT", "evidence": "no camera/event ID fields in PLY properties"},
            "observation_support": {"status": "SEMANTICALLY_DIFFERENT", "evidence": "extra opacity/scale fields are not TSDF authority or renderer-event support"},
            "confidence_reliability": {"status": "SEMANTICALLY_DIFFERENT", "evidence": "opacity is not canonical structural reliability"},
            "ownership": {"status": "ABSENT", "evidence": "no region/surface/boundary owner field"},
            "canonical_extracted_surface_fields": {"status": "ABSENT", "evidence": "ExtractedSurface vertices/faces/support/value/h are not present as a typed replay artifact"},
        },
        "source_contract": source,
    }


def _provenance_audit(baseline: dict[str, Any], raw_contract: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "renderer_observation_to_TSDF": {"status": "DETERMINISTICALLY_DERIVABLE", "scope": "WL127 source defines projective TSDF field construction from renderer queries, but current field cache is unavailable"},
        "TSDF_to_zero_set": {"status": "DETERMINISTICALLY_DERIVABLE", "scope": "committed extract_zero_level_set uses all-eight-corner authority and marching-cubes cell ownership"},
        "zero_set_to_mesh_vertex_face": {"status": "DETERMINISTICALLY_DERIVABLE", "scope": "available in committed ExtractedSurface implementation via faces and seam welding"},
        "mesh_element_to_renderer_event": {"status": "ABSENT", "scope": "no event/camera ID or source-cell provenance is present in the available PLY or ExtractedSurface schema"},
        "mesh_element_to_camera_set": {"status": "ABSENT", "scope": "no deterministic camera contributor field"},
        "mesh_membership_meaning": "evidence-bounded geometric extraction/zero-set support, not proof of one physical sheet",
        "event_1527_trace": {
            "status": "NOT_MAPPABLE_UNDER_EXISTING_CONTRACT",
            "reason": "event 1527 is present in WL149 renderer union, but no source-cell/mesh-element provenance maps it into the available Raw Visible Surface artifact",
            "human_review": wl150.HUMAN_REVIEW_PHYSICAL_SHEET_STATUS,
            "blacklist": False,
        },
        "provenance_gap": True,
        "source_evidence": source,
        "baseline_preserved": baseline["wl149_wl151_exact"],
        "raw_contract_provenance_status": raw_contract["property_contract"]["source_camera_evidence_provenance"]["status"],
    }


def _connectivity_diagnostic(header: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    vertex_count = int(header.get("elements", {}).get("vertex", 0))
    face_count = int(header.get("elements", {}).get("face", 0))
    return {
        "artifact": header["path"],
        "vertex_count": vertex_count,
        "face_count": face_count,
        "edge_count": 0 if face_count == 0 else "NOT_COMPUTED",
        "connected_components": "NOT_DEFINED_WITHOUT_FACES",
        "component_size_distribution": "NOT_DEFINED_WITHOUT_FACES",
        "topological_boundary_component_count": "NOT_MEANINGFUL_WITHOUT_FACES",
        "open_boundary_count": "NOT_MEANINGFUL_WITHOUT_FACES",
        "closed_boundary_count": "NOT_MEANINGFUL_WITHOUT_FACES",
        "no_component_filtering": True,
        "no_largest_component_selection": True,
        "source_operation": source["wl127_mesh_connectivity"],
        "interpretation": "The available file cannot answer mesh connectivity questions; treating each point as a component would invent point adjacency and is not performed.",
    }


def _carrier_eligibility(connectivity: dict[str, Any], provenance: dict[str, Any], raw_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": "INELIGIBLE",
        "architecture_verdict": "INELIGIBLE_CARRIER",
        "available_artifact_scope": "WL127 RENDERER_MEDIAN_SURFACE_POINTS vertex-only PLY",
        "candidate_extracted_mesh_scope": "NOT_EVALUATED_DUE_TO_MISSING_REPLAY_ARTIFACT",
        "requirements": {
            "local_surface_connectivity": "INCOMPATIBLE_MISSING",
            "local_region_identity": "INCOMPATIBLE_MISSING",
            "physical_boundary_candidates": "INCOMPATIBLE_MISSING",
            "pre_fit_support_domain": "INCOMPATIBLE_MISSING",
            "evidence_provenance": "INCOMPATIBLE_MISSING",
        },
        "stop_conditions": [
            "RAW_SURFACE_PROVENANCE_GAP",
            "PHYSICAL_SHEET_MEMBERSHIP_GAP",
            "RAW_SURFACE_BASELINE_REPLAY_UNAVAILABLE",
        ],
        "reason": "A mesh/zero-set topology carrier is not available in the current confirmed artifact, and the available point artifact has neither connectivity nor event/source-cell provenance. It cannot own Visible Surface Topology or Boundary ownership without a new contract.",
        "connectivity_result": connectivity["connected_components"],
        "provenance_result": provenance["mesh_element_to_renderer_event"]["status"],
        "raw_surface_membership_result": raw_contract["property_contract"]["ownership"]["status"],
        "no_candidate_d": True,
    }


def _physical_sheet_result(carrier: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": "E. INSUFFICIENT_EVIDENCE",
        "question": "Can native Raw Visible Surface connectivity separate tabletop from side/vase/background/lower geometry?",
        "answer": "Not testable from the available vertex-only artifact; no native connectivity or deterministic physical-sheet identity exists to inspect.",
        "not_inferred": ["no claim of B connected physical sheets", "no claim of C fragmented physical sheet", "no component-size or geometric threshold introduced"],
        "carrier_verdict": carrier["architecture_verdict"],
    }


def _source_history(source: dict[str, Any]) -> dict[str, Any]:
    result = wl150._source_history_manifest()
    result["wl127_commit"] = {"commit": "943a764", "subject": "Worklog 127: evidence-bounded projective TSDF for direct Visible Surface construction"}
    result["wl127_sources"] = [
        {"path": wl150._relative(path), "exists": path.exists(), "sha256": wl150._sha256_file(path) if path.exists() else None}
        for path in (WL127_EXTRACTION, WL127_MESH_OPS, WL127_DRIVER, WL127_DOC)
    ]
    result["wl152_source_inventory"] = source
    result["no_canonical_modification"] = True
    return result


def build_audit() -> dict[str, Any]:
    baseline = _baseline_reconciliation()
    source = _source_inventory()
    renderer = _renderer_event_contract()
    canonical_gaussian = _canonical_gaussian_region_contract()
    raw_header = baseline["raw_visible_surface_artifact"]
    raw_contract = _raw_visible_surface_contract(raw_header, source)
    provenance = _provenance_audit(baseline, raw_contract, source)
    connectivity = _connectivity_diagnostic(raw_header, source)
    carrier = _carrier_eligibility(connectivity, provenance, raw_contract)
    physical = _physical_sheet_result(carrier)
    return {
        "baseline_reconciliation": baseline,
        "renderer_event_contract": renderer,
        "canonical_gaussian_region_contract": canonical_gaussian,
        "raw_visible_surface_contract": raw_contract,
        "evidence_to_raw_surface_provenance": provenance,
        "connectivity_diagnostic": connectivity,
        "topology_carrier_eligibility": carrier,
        "physical_sheet_connectivity_result": physical,
        "boundary_first_eligibility": {
            "status": "STOPPED",
            "reason": "INELIGIBLE_CARRIER / provenance and physical-sheet ownership gaps",
            "candidate_d_eligible": False,
            "candidate_d_implemented": False,
            "candidate_d_tested": False,
        },
        "conditional_candidate_d": {"status": "NOT_RUN", "reason": "Eligibility gate failed before candidate construction"},
        "id_1527_probe": {
            "status": "NOT_MAPPABLE_UNDER_EXISTING_CONTRACT",
            "human_review": wl150.HUMAN_REVIEW_PHYSICAL_SHEET_STATUS,
            "blacklist": False,
            "historical_v_min_owner": True,
            "raw_surface_contribution": "UNRESOLVED_NO_PROVENANCE_MAPPING",
        },
        "synthetic_results": {"status": "NOT_RUN", "reason": "Candidate D ineligible"},
        "real_scene_quantitative_result": {
            "status": "LIMITED_ARTIFACT_ACCOUNTING_ONLY",
            "available_artifact": {"vertices": raw_header["elements"].get("vertex", 0), "faces": raw_header["elements"].get("face", 0)},
            "component_accounting": connectivity,
            "boundary_accounting": "NOT_MEANINGFUL_WITHOUT_FACES",
            "evidence_provenance_accounting": provenance,
            "intended_sheet_accounting": physical,
            "candidate_d_metrics": "NOT_RUN",
        },
        "real_scene_qualitative_result": {"status": "NOT_RUN_FOR_CANDIDATE_D", "existing_historical_review": "WL145/WL149 retained; no new mapping or visualization"},
        "source_history_manifest": _source_history(source),
        "architecture_verdict": {
            "verdict": "INELIGIBLE_CARRIER",
            "secondary_gaps": ["RAW_SURFACE_PROVENANCE_GAP", "PHYSICAL_SHEET_MEMBERSHIP_GAP"],
            "why": "The available Raw Visible Surface artifact is a vertex-only renderer-event point artifact, not a face-connected extracted surface. The committed ExtractedSurface mesh contract exists in source but its matching replay artifact is unavailable; neither representation carries deterministic renderer-event-to-surface-element provenance or physical-sheet ownership.",
            "candidate_d": "not implemented",
            "event_1527": "preserved and not blacklisted; contribution is not mappable under the existing contract",
            "canonical_production_modified": False,
        },
        "report": {
            "1. CURRENT ARCHITECTURE QUESTION": "Can existing Raw Visible Surface Geometry legitimately own Visible Surface Topology and Boundary ownership between renderer evidence and representative fitting?",
            "2. WL151 BASELINE RECONCILIATION": baseline,
            "3. RENDERER-EVENT CONTRACT": renderer,
            "4. CANONICAL GAUSSIAN-REGION CONTRACT": canonical_gaussian,
            "5. RAW VISIBLE SURFACE CONTRACT": raw_contract,
            "6. EVIDENCE → RAW SURFACE PROVENANCE": provenance,
            "7. TOPOLOGY CARRIER ELIGIBILITY": carrier,
            "8. PHYSICAL-SHEET CONNECTIVITY RESULT": physical,
            "9. BOUNDARY-FIRST ELIGIBILITY": {
                "status": "STOPPED",
                "candidate_d_eligible": False,
                "stop_conditions": carrier["stop_conditions"],
            },
            "10. CONDITIONAL CANDIDATE D": {"status": "NOT_RUN", "reason": "Candidate D requires all eligibility gates; no new membership/provenance mechanism is introduced"},
            "11. ID 1527 TRACE": {
                "status": "NOT_MAPPABLE_UNDER_EXISTING_CONTRACT",
                "human_review": wl150.HUMAN_REVIEW_PHYSICAL_SHEET_STATUS,
                "historical_v_min_owner": True,
                "blacklist": False,
            },
            "12. SYNTHETIC RESULTS": {"status": "NOT_RUN", "reason": "Candidate D ineligible"},
            "13. REAL-SCENE QUANTITATIVE RESULT": {
                "status": "LIMITED_ARTIFACT_ACCOUNTING_ONLY",
                "vertex_count": raw_header["elements"].get("vertex", 0),
                "face_count": raw_header["elements"].get("face", 0),
                "candidate_d_metrics": "NOT_RUN",
            },
            "14. REAL-SCENE QUALITATIVE RESULT": {"status": "NOT_RUN_FOR_CANDIDATE_D"},
            "15. ARCHITECTURE VERDICT": "INELIGIBLE_CARRIER",
            "16. RETAINED / REJECTED / OPEN": {
                "RETAINED": ["WL139/WL145/WL148/WL149/WL150/WL151", "1586 event union and event 1527", "renderer median semantics", "canonical TSDF source and canonical ownership source", "existing WL127 point artifact without mutation"],
                "REJECTED": ["treating a vertex-only PLY as mesh topology", "nearest/event attribution invented for 1527", "manual polygons as production membership", "Candidate D implementation", "connectivity repair or component filtering"],
                "OPEN": ["reproduce and freeze the matching WL127 ExtractedSurface mesh artifact", "preserve deterministic TSDF-cell/event provenance if required", "define physical-sheet membership for the raw surface", "derive defensible local boundary ownership without a new heuristic"],
            },
            "INTENT ALIGNMENT": {"diagnostic_only": True, "baseline_preserved": True, "candidate_d_skipped_on_gate": True, "event_1527_removed": False, "new_membership_mechanism": False, "canonical_behavior_changed": False},
            "IMPLEMENTATION FIDELITY": {"wl149_wl151_replayed": True, "wl127_source_inspected": True, "raw_artifact_hash_recorded": True, "raw_surface_geometry_modified": False, "canonical_source_modified": False, "new_thresholds": False, "new_correspondence": False, "largest_component_filter": False, "candidate_d": False},
            "ARCHITECTURE RESULT": {"verdict": "INELIGIBLE_CARRIER", "secondary_gaps": ["RAW_SURFACE_PROVENANCE_GAP", "PHYSICAL_SHEET_MEMBERSHIP_GAP"], "exact_missing_contract": ["face/edge topology in available artifact", "event/TSDF-cell to surface-element provenance", "physical-sheet/local-region identity", "boundary ownership"], "conditional_sections_skipped": ["Candidate D", "synthetic contracts", "Candidate D real-scene metrics", "Candidate D qualitative exports"]},
        },
    }


def write_audit() -> dict[str, Any]:
    audit = build_audit()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "baseline_reconciliation.json": audit["baseline_reconciliation"],
        "renderer_event_contract.json": audit["renderer_event_contract"],
        "canonical_gaussian_region_contract.json": audit["canonical_gaussian_region_contract"],
        "raw_visible_surface_contract.json": audit["raw_visible_surface_contract"],
        "evidence_to_raw_surface_provenance.json": audit["evidence_to_raw_surface_provenance"],
        "connectivity_diagnostic.json": audit["connectivity_diagnostic"],
        "topology_carrier_eligibility.json": audit["topology_carrier_eligibility"],
        "physical_sheet_connectivity_result.json": audit["physical_sheet_connectivity_result"],
        "boundary_first_eligibility.json": audit["boundary_first_eligibility"],
        "conditional_candidate_d.json": audit["conditional_candidate_d"],
        "id_1527_probe.json": audit["id_1527_probe"],
        "synthetic_results.json": audit["synthetic_results"],
        "real_scene_quantitative_result.json": audit["real_scene_quantitative_result"],
        "real_scene_qualitative_result.json": audit["real_scene_qualitative_result"],
        "source_history_manifest.json": audit["source_history_manifest"],
        "architecture_verdict.json": audit["architecture_verdict"],
        "visible_surface_carrier_contract_audit_report.json": audit["report"],
    }
    for name, value in payloads.items():
        wl150._write_json(OUTPUT_ROOT / name, value)
    matrix = """# WL152 Raw Visible Surface carrier result

## Verdict

**INELIGIBLE_CARRIER**

The available `RENDERER_MEDIAN_SURFACE_POINTS/point_cloud.ply` is a
vertex-only point artifact with no faces, edges, connected components, or
boundary graph. The matching WL127 TSDF `ExtractedSurface` replay cache is not
available. Its source contract also has no deterministic renderer-event or
TSDF-cell provenance into mesh elements.

Secondary gaps: `RAW_SURFACE_PROVENANCE_GAP`, `PHYSICAL_SHEET_MEMBERSHIP_GAP`.

Candidate D was not implemented. No connectivity repair, membership rule,
event filtering, or attribution heuristic was introduced.
"""
    (OUTPUT_ROOT / "carrier_verdict.md").write_text(matrix, encoding="utf-8")
    readme = """# Worklog 152 — Visible Surface carrier contract audit

Diagnostic-only audit of renderer evidence → Raw Visible Surface → local
ownership. WL139–WL151 and event 1527 are preserved.

- Verdict: **INELIGIBLE_CARRIER**
- Secondary gaps: `RAW_SURFACE_PROVENANCE_GAP`, `PHYSICAL_SHEET_MEMBERSHIP_GAP`
- Available WL127 artifact: vertex-only PLY, 1,212,365 vertices, no faces
- Candidate D: not implemented
- Event 1527: not blacklisted; contribution not mappable under the existing contract

See `carrier_verdict.md` and the numbered JSON reports.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_ROOT.iterdir():
        destination = TEMP_ROOT / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)
    return {"output_root": wl150._relative(OUTPUT_ROOT), "temp_root": wl150._relative(TEMP_ROOT), "files": sorted(payloads) + ["carrier_verdict.md", "README.md"], "verdict": audit["architecture_verdict"]["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Raw Visible Surface as a topology carrier")
    parser.add_argument("--write", action="store_true", help="write numbered output and temp artifacts")
    arguments = parser.parse_args()
    result = write_audit() if arguments.write else build_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=wl150._json_default))


if __name__ == "__main__":
    main()
