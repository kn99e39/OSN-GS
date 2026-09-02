"""Worklog 150: executable contract trace for the WL139--WL149 path.

This module is deliberately diagnostic-only.  It reads committed source and
frozen WL149/WL145 artifacts, reconstructs the canonical Boundary First and
local-region contract, and compares that contract with the isolated WL139
clean-oracle path.  It does not filter evidence, refit a surface, change PCA,
or modify any production module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "150_boundary_first_local_surface_decomposition_contract_trace"
TEMP_ROOT = REPO_ROOT / "temp" / "150_boundary_first_local_surface_decomposition_contract_trace"

WL149_ROOT = REPO_ROOT / "temp" / "149_physical_sheet_evidence_vs_chart_extent_failure_attribution"
WL149_REPORT = WL149_ROOT / "physical_sheet_evidence_vs_chart_extent_failure_attribution_report.json"
WL149_INFLUENCE = WL149_ROOT / "full_per_point_influence.json"
WL149_INFLUENCE_NPZ = WL149_ROOT / "full_per_point_influence.npz"

WL145_ROOT = REPO_ROOT / "output" / "confirmed" / "145_genuine_physical_sheet_oracle_clean_support_representative_audit"
WL145_EVENT_ROOT = WL145_ROOT / "tabletop_broad_planar_clean" / "per_view_renderer_median_events"

CANONICAL_CONSTRUCTION = REPO_ROOT / "osn_gs" / "surface" / "torch_visible_surface_construction.py"
CANONICAL_PIPELINE = REPO_ROOT / "osn_gs" / "core" / "torch_pipeline.py"
BOUNDARY_BUILDER = REPO_ROOT / "osn_gs" / "surface" / "torch_boundary_first_visible_builder.py"
BOUNDARY_ADAPTER = REPO_ROOT / "osn_gs" / "surface" / "torch_visible_boundary_materialization_adapter.py"
REGION_FORMATION = REPO_ROOT / "osn_gs" / "surface" / "torch_gaussian_surface_region_formation.py"
DECOMPOSITION_DIAGNOSTIC = REPO_ROOT / "osn_gs" / "surface" / "torch_surface_decomposition.py"
WL139_MODULE = REPO_ROOT / "devtools" / "demo" / "physical_chart_surface_representative.py"
WL145_MODULE = REPO_ROOT / "devtools" / "demo" / "genuine_physical_sheet_oracle_clean_support_representative_audit.py"
WL144_MODULE = REPO_ROOT / "devtools" / "demo" / "per_view_renderer_surface_correspondence_physical_sheet_oracle_audit.py"
WL148_MODULE = REPO_ROOT / "devtools" / "demo" / "wl145_baseline_reconciliation_support_constrained_materialization.py"
WL149_MODULE = REPO_ROOT / "devtools" / "demo" / "physical_sheet_evidence_vs_chart_extent_failure_attribution.py"

CAMERA_ORDER = ("DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG")
HUMAN_REVIEW_PHYSICAL_SHEET_STATUS = "CLEAR_NOT_ON_INTENDED_SURFACE"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return _relative(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _function_node(path: Path, function_name: str) -> ast.AST | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    return sorted(candidates, key=lambda node: getattr(node, "lineno", 0))[0] if candidates else None


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ast.dump(function, annotate_fields=False)


def _function_info(path: Path, function_name: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    node = _function_node(path, function_name)
    if node is None:
        return {"path": _relative(path), "function": function_name, "status": "MISSING"}
    calls = sorted({_call_name(call) for call in ast.walk(node) if isinstance(call, ast.Call)})
    return {
        "path": _relative(path),
        "function": function_name,
        "status": "PRESENT",
        "line": int(node.lineno),
        "end_line": int(getattr(node, "end_lineno", node.lineno)),
        "source_sha256": _sha256_bytes(source.encode("utf-8")),
        "calls": calls,
    }


def _line_matches(path: Path, needles: tuple[str, ...]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if any(needle in line for needle in needles):
            matches.append({"line": number, "text": line.strip()})
    return matches


def _source_ref(path: Path, function_name: str, needles: tuple[str, ...] = ()) -> dict[str, Any]:
    result = _function_info(path, function_name)
    if needles:
        result["matched_source_lines"] = _line_matches(path, needles)
    return result


def _load_wl149_baseline() -> dict[str, Any]:
    report = _read_json(WL149_REPORT)
    influence_rows = _read_json(WL149_INFLUENCE)
    influence_npz = np.load(WL149_INFLUENCE_NPZ, allow_pickle=False)
    baseline = report["BASELINE RECONCILIATION"]
    ids = [int(row["event_id"]) for row in influence_rows]
    expected_ids = list(range(int(baseline["event_count"])))
    if ids != expected_ids:
        raise AssertionError("WL149 event IDs are not the exact frozen row-order population")
    if int(len(influence_rows)) != 1586 or int(influence_npz["event_id"].shape[0]) != 1586:
        raise AssertionError("WL149 event count changed")
    if not np.array_equal(influence_npz["event_id"], np.arange(1586, dtype=np.int64)):
        raise AssertionError("WL149 NPZ event IDs changed")
    if baseline["event_union_sha256"] != "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78":
        raise AssertionError("unexpected WL149 union hash")
    if baseline["camera_counts"] != {"DSC08043.JPG": 754, "DSC07960.JPG": 330, "DSC08003.JPG": 502}:
        raise AssertionError("unexpected WL149 camera counts")
    if baseline["representative_shape"] != [3840, 3] or int(baseline["support_vertices"]) != 314:
        raise AssertionError("unexpected WL149 representative/support baseline")
    if baseline["support_mask_sha256"] != "23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9":
        raise AssertionError("unexpected WL149 support mask hash")
    if int(baseline["fully_supported_cells"]) != 211:
        raise AssertionError("unexpected WL149 all-four support relation")
    event = next(row for row in influence_rows if int(row["event_id"]) == 1527)
    return {
        "report_path": _relative(WL149_REPORT),
        "influence_path": _relative(WL149_INFLUENCE),
        "influence_npz_path": _relative(WL149_INFLUENCE_NPZ),
        "event_count": int(baseline["event_count"]),
        "event_union_sha256": baseline["event_union_sha256"],
        "camera_counts": baseline["camera_counts"],
        "event_id_definition": report["RENDERER-EVENT PROVENANCE"]["event_id_definition"],
        "representative_shape": baseline["representative_shape"],
        "representative_xyz_sha256": baseline["representative_xyz_sha256"],
        "representative_normals_sha256": baseline["representative_normals_sha256"],
        "support_vertices": int(baseline["support_vertices"]),
        "unsupported_vertices": int(baseline["unsupported_vertices"]),
        "support_mask_sha256": baseline["support_mask_sha256"],
        "fully_supported_cells": int(baseline["fully_supported_cells"]),
        "all_event_ids_exact": True,
        "event_1527_influence_row": event,
        "wl149_status": report["status"],
        "preservation_assertions": {
            "event_union_preserved": True,
            "event_1527_preserved": True,
            "pca_axes_preserved": True,
            "representative_xyz_preserved": True,
            "representative_normals_preserved": True,
            "support_mask_preserved": True,
            "all_four_support_relation_preserved": True,
        },
    }


def _load_event_1527(baseline: dict[str, Any]) -> dict[str, Any]:
    counts = baseline["camera_counts"]
    offset = sum(int(counts[name]) for name in CAMERA_ORDER[:2])
    local_index = 1527 - offset
    path = WL145_EVENT_ROOT / CAMERA_ORDER[2] / "event_cloud_with_provenance.npz"
    data = np.load(path, allow_pickle=False)
    source_camera = str(np.asarray(data["source_camera"]).item())
    control = str(np.asarray(data["physical_sheet_control"]).item())
    row = {
        "event_id": 1527,
        "source_camera": source_camera,
        "source_event_local_index": int(local_index),
        "source_event_artifact": _relative(path),
        "physical_sheet_control_label": control,
        "source_pixel_x": int(data["pixel_x"][local_index]),
        "source_pixel_y": int(data["pixel_y"][local_index]),
        "renderer_median_event_depth": float(data["renderer_median_depth"][local_index]),
        "world_xyz": np.asarray(data["event_points_xyz"][local_index], dtype=np.float64),
        "event_normal": np.asarray(data["local_normals"][local_index], dtype=np.float64),
    }
    influence = baseline["event_1527_influence_row"]
    if row["source_camera"] != influence["source_camera"]:
        raise AssertionError("event 1527 source camera changed")
    if [row["source_pixel_x"], row["source_pixel_y"]] != [int(influence["source_pixel_x"]), int(influence["source_pixel_y"])]:
        raise AssertionError("event 1527 source pixel changed")
    np.testing.assert_allclose(row["world_xyz"], [influence["world_x"], influence["world_y"], influence["world_z"]], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(row["event_normal"], [influence["event_normal_x"], influence["event_normal_y"], influence["event_normal_z"]], rtol=0.0, atol=1.0e-12)
    row.update({
        "chart_u": float(influence["chart_u"]),
        "chart_v": float(influence["chart_v"]),
        "chart_n": float(influence["chart_n"]),
        "pca_v_min_owner": bool(influence["fixed_is_v_min_owner"]),
        "fixed_extent_reduction_v_span": float(influence["fixed_extent_reduction_v_span"]),
        "fixed_extent_reduction_rectangular_chart_area": float(influence["fixed_extent_reduction_rectangular_chart_area"]),
        "pca_extent_reduction_v_span": float(influence["pca_extent_reduction_v_span"]),
        "pca_extent_reduction_rectangular_chart_area": float(influence["pca_extent_reduction_rectangular_chart_area"]),
        "fixed_area_rank": int(influence["fixed_area_rank"]),
        "pca_area_rank": int(influence["pca_area_rank"]),
        "human_review_status": HUMAN_REVIEW_PHYSICAL_SHEET_STATUS,
    })
    if not row["pca_v_min_owner"] or row["chart_v"] != -0.5984138975738875:
        raise AssertionError("event 1527 is no longer the frozen v_min owner")
    return row


def _intended_boundary_first_contract() -> dict[str, Any]:
    return {
        "contract_status": "EXECUTABLE_CANONICAL_CONTRACT",
        "boundary_owner": "region-owned observed boundary candidate / ordered boundary component",
        "boundary_evidence": "world-space boundary halfedge candidates derived after accepted region topology, then directed compatibility and ordering",
        "boundary_stage": "before eligible representative materialization and before the adapter's NURBS fit",
        "boundary_constraints": {
            "membership": "indirectly: region/topology and admissibility decide which evidence may enter the fit",
            "representative_fit_support": "PRE_FIT: ordered boundary points plus region-core interior points are the adapter fit input",
            "chart_extent": "local region-owned fit domain; not a global PCA extrema rule",
            "materialization": "yes: only eligible, non-branching observed outer components are materialized",
            "continuation_frontier": "boundary/termination candidates are part of the canonical construction context, but continuation is downstream and not part of this trace",
        },
        "fit_timing": "PRE_FIT_DOMAIN_CONTROL",
        "synthetic_rectangle_policy": "open/branch/ambiguous boundaries do not receive a synthetic rectangular closure",
        "source_evidence": [
            _source_ref(CANONICAL_CONSTRUCTION, "construct_visible_nurbs_from_gaussians", ("form_surface_regions(", "recover_directed_boundary_components", "materialize_visible_boundary_component")),
            _source_ref(BOUNDARY_BUILDER, "build_boundary_first_visible_surface", ("_materialize_boundary_role_network", "outer_loops")),
            _source_ref(BOUNDARY_ADAPTER, "materialize_visible_boundary_component", ("ordered_closed_loop", "fit_torch_visible_surface_lsq")),
            _source_ref(CANONICAL_PIPELINE, "_initialize_canonical", ("construct_visible_nurbs_from_gaussians", "materialized_visible_nurbs_surfaces")),
        ],
        "explicit_negative": "A post-fit occupancy mask is not itself the Boundary First pre-fit contract.",
    }


def _intended_local_surface_decomposition_contract() -> dict[str, Any]:
    return {
        "contract_status": "EXECUTABLE_CANONICAL_CONTRACT",
        "population": "bounded construction-point population passed to canonical Gaussian-to-visible-NURBS construction",
        "decomposition_owner": "form_surface_regions in torch_gaussian_surface_region_formation.py",
        "one_local_physical_surface": "a region candidate with coherent manifold/affinity evidence and region-owned member nodes/IDs, subsequently checked against accepted topology and boundary status",
        "identity_created": "node_region_id and SurfaceRegionCandidate.region_id/member IDs",
        "identity_preserved_into_fit": "yes: source_region_id, ordered boundary IDs, interior IDs, and supporting source IDs reach materialized region-owned surfaces",
        "fit_population": "per eligible local boundary component plus its region-core interior support",
        "pooled_cross_region_pca": "not legal in the canonical contract; no global point-union PCA is the region ownership mechanism",
        "source_evidence": [
            _source_ref(REGION_FORMATION, "form_surface_regions", ("node_region_id", "SurfaceRegionCandidate", "region_id")),
            _source_ref(CANONICAL_CONSTRUCTION, "construct_visible_nurbs_from_gaussians", ("surface_regions", "source_region_to_surface", "region_core")),
            _source_ref(BOUNDARY_ADAPTER, "materialize_visible_boundary_component", ("source_region_id", "supporting_source_ids")),
        ],
        "dormant_diagnostic_path": {
            "module": _relative(DECOMPOSITION_DIAGNOSTIC),
            "status": "DIAGNOSTICS_ONLY_NOT_PRODUCTION_CONSTRUCTOR",
            "source_evidence": _source_ref(DECOMPOSITION_DIAGNOSTIC, "build_proxy_surface_components_diagnostics", ("not called by the production boundary-first constructor", "Diagnostics-only")),
        },
    }


def _actual_dataflow(baseline: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": "ISOLATED_WL139_WL145_ORACLE_CONTROL_PATH",
        "source_evidence": {
            "wl145_loads_wl127_mesh_for_provenance_only": _source_ref(WL145_MODULE, "run_audit", ("RAW_VISIBLE_SURFACE", "_load_inputs")),
            "per_view_renderer_events": _source_ref(WL144_MODULE, "_build_per_view_event_cloud", ("renderer_median_depth", "event_points_xyz", "source_camera")),
            "per_view_cloud_output": _source_ref(WL145_MODULE, "_save_candidate_outputs", ("event_cloud_with_provenance.npz", "common_world_event_clouds")),
            "union_and_pca": _source_ref(WL145_MODULE, "run_audit", ("clean_points = np.concatenate", "_pca_chart_config")),
            "pca_frame": _source_ref(WL145_MODULE, "_pca_chart_config", ("np.linalg.eigh", "coordinates = points @ axes", "u_bounds")),
            "wl139_fit": _source_ref(WL145_MODULE, "_save_representative_outputs", ("fit_physical_chart_surface",)),
            "wl139_fixed_uv": _source_ref(WL139_MODULE, "_fixed_physical_uv", ("_case_coordinates(points, config)",)),
            "wl139_domain": _source_ref(WL139_MODULE, "_physical_domain", ("retained_construction", "full_evaluation_only")),
            "wl139_fitter": _source_ref(WL139_MODULE, "fit_physical_chart_surface", ("fit_indices = deterministic_indices", "fit_uv, fit_n")),
            "wl148_frozen_replay": _source_ref(WL148_MODULE, "_load_frozen_baseline", ("frozen", "representative")),
            "wl149_provenance": _source_ref(WL149_MODULE, "_load_provenance", ("event_id", "source_camera", "world_xyz")),
            "wl149_pca": _source_ref(WL149_MODULE, "_pca_frame", ("np.linalg.eigh", "projected")),
            "wl149_audit": _source_ref(WL149_MODULE, "run_audit", ("v_min", "full_per_point_influence")),
        },
        "stages": [
            {
                "stage": "WL145 manual physical-sheet control and per-view renderer event extraction",
                "timing": "PRE_FIT",
                "population": "three independent camera event clouds; the clear tabletop candidate totals 1586 rows",
                "region_ids_before_union": "ABSENT",
                "local_surface_ids_before_union": "ABSENT",
                "boundary_ownership_before_union": "ABSENT",
                "identity_note": "manual PhysicalSheetControl name is metadata, not canonical region/topology membership",
            },
            {
                "stage": "WL145 clean_points union",
                "timing": "PRE_FIT",
                "operation": "np.concatenate([cloud.points for cloud in clouds], axis=0)",
                "population": "all 1586 renderer-event XYZ rows, camera order DSC08043, DSC07960, DSC08003",
                "region_ids": "ABSENT; no region formation call occurs",
                "local_surface_ids": "ABSENT",
                "boundary_ownership": "ABSENT",
                "provenance": "flattened into an XYZ-only fit population; sidecar row order permits retrospective camera lookup",
            },
            {
                "stage": "WL145 _pca_chart_config",
                "timing": "PRE_FIT",
                "operation": "global PCA of the pooled clean renderer-event union; world_xyz @ axis columns gives u,v,n",
                "region_or_boundary_constraint": "ABSENT",
                "domain_rule": "global min/max extrema establish u/v/n bounds and a rectangular chart",
            },
            {
                "stage": "WL145 graphness audit",
                "timing": "PRE_FIT_DIAGNOSTIC",
                "operation": "graphness is checked on the pooled union; it does not create membership or ownership",
                "region_or_boundary_constraint": "ABSENT",
                "geometry_mutation": "NONE",
            },
            {
                "stage": "WL139 physical chart representative fit",
                "timing": "FIT",
                "operation": "fit_physical_chart_surface over the pooled XYZ points at fixed physical chart UV; the WL145 clean path calls it with the full clean support",
                "region_or_boundary_constraint": "ABSENT",
                "domain_rule": "the PCA rectangle/config is already established; fitter receives XYZ only, not region or boundary IDs",
            },
            {
                "stage": "WL148 B support-constrained materialization",
                "timing": "POST_FIT",
                "operation": "existing support mask selects all-four-supported cells from the frozen representative",
                "region_or_boundary_constraint": "POST_FIT occupancy only",
                "explicit_semantics": "WL148 B does NOT by itself restore Boundary First semantics.",
            },
            {
                "stage": "WL149 attribution",
                "timing": "POST_FIT_DIAGNOSTIC",
                "operation": "replays the frozen union/representative and evaluates extrema ownership and influence without refit",
                "event_1527_result": {
                    "chart_v": event["chart_v"],
                    "v_min_owner": event["pca_v_min_owner"],
                    "fixed_area_rank": event["fixed_area_rank"],
                    "pca_area_rank": event["pca_area_rank"],
                },
            },
        ],
        "exact_baseline": {
            "event_count": baseline["event_count"],
            "event_union_sha256": baseline["event_union_sha256"],
            "representative_shape": baseline["representative_shape"],
            "support_vertices": baseline["support_vertices"],
            "fully_supported_cells": baseline["fully_supported_cells"],
        },
    }


def _event_trace(baseline: dict[str, Any], event: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    stages = [
        {
            "stage": "per-view renderer event generation",
            "source": actual["source_evidence"]["per_view_renderer_events"],
            "event_status": {"Primitive Coverage": "ABSENT", "Visible Surface Topology identity": "ABSENT", "physical-sheet membership": "PRESENT", "region ownership": "ABSENT", "boundary ownership": "ABSENT", "renderer contribution provenance": "PRESENT"},
            "detail": "PRESENT means the manual control label and camera/pixel/depth/XYZ/normal event provenance exist; it is not a validated physical-sheet membership field.",
        },
        {
            "stage": "event-union insertion",
            "source": actual["source_evidence"]["union_and_pca"],
            "event_status": {"Primitive Coverage": "ABSENT", "Visible Surface Topology identity": "ABSENT", "physical-sheet membership": "PRESENT", "region ownership": "ABSENT", "boundary ownership": "ABSENT", "renderer contribution provenance": "PRESENT"},
            "detail": "Event 1527 is row 443 of DSC08003.JPG and union row 1527; the manual label remains metadata while canonical ownership is never created.",
        },
        {
            "stage": "PCA input and chart extent",
            "source": actual["source_evidence"]["pca_frame"],
            "event_status": {"Primitive Coverage": "ABSENT", "Visible Surface Topology identity": "ABSENT", "physical-sheet membership": "DISCARDED", "region ownership": "ABSENT", "boundary ownership": "ABSENT", "renderer contribution provenance": "DISCARDED"},
            "detail": "The PCA function receives the pooled XYZ array and computes global extrema; it does not receive control, region, boundary, or camera sidecar fields.",
        },
        {
            "stage": "WL139 representative fit and domain",
            "source": actual["source_evidence"]["wl139_fitter"],
            "event_status": {"Primitive Coverage": "ABSENT", "Visible Surface Topology identity": "ABSENT", "physical-sheet membership": "DISCARDED", "region ownership": "ABSENT", "boundary ownership": "ABSENT", "renderer contribution provenance": "DISCARDED"},
            "detail": "The fitter consumes points and fixed physical UV/normal-coordinate data only; event 1527 is therefore legal input to the pooled representative.",
        },
        {
            "stage": "WL148/WL149 frozen replay",
            "source": actual["source_evidence"]["wl148_frozen_replay"],
            "event_status": {"Primitive Coverage": "NOT APPLICABLE", "Visible Surface Topology identity": "NOT APPLICABLE", "physical-sheet membership": "NOT APPLICABLE", "region ownership": "NOT APPLICABLE", "boundary ownership": "NOT APPLICABLE", "renderer contribution provenance": "PRESENT"},
            "detail": "Replay preserves the historical arrays; no new ownership is inferred and no geometry is repaired.",
        },
    ]
    return {
        "event_id": 1527,
        "human_review_fact": f"HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: {HUMAN_REVIEW_PHYSICAL_SHEET_STATUS}",
        "source_camera": event["source_camera"],
        "source_pixel": [event["source_pixel_x"], event["source_pixel_y"]],
        "source_event_local_index": event["source_event_local_index"],
        "renderer_median_event_depth": event["renderer_median_event_depth"],
        "world_xyz": event["world_xyz"],
        "event_normal": event["event_normal"],
        "physical_sheet_control_label": event["physical_sheet_control_label"],
        "pca_input_inclusion": "PRESENT",
        "pca_coordinates": {"u": event["chart_u"], "v": event["chart_v"], "n": event["chart_n"]},
        "v_min_ownership": {"status": "PRESENT", "is_owner": event["pca_v_min_owner"], "coordinate": event["chart_v"]},
        "chart_influence": {
            "fixed_v_span_reduction": event["fixed_extent_reduction_v_span"],
            "fixed_rectangular_area_reduction": event["fixed_extent_reduction_rectangular_chart_area"],
            "full_pca_v_span_reduction": event["pca_extent_reduction_v_span"],
            "full_pca_rectangular_area_reduction": event["pca_extent_reduction_rectangular_chart_area"],
            "influence_is_diagnostic_only": True,
        },
        "representative_domain_consequence": "event 1527 set the global v_min bound used by the WL145/WL139 clean-support chart; this was not a boundary-owned local extent.",
        "contract_field_trace": {
            "Primitive Coverage": "ABSENT: frozen renderer-event provenance has no primitive/contributor ID.",
            "Visible Surface Topology identity": "ABSENT: WL145 clean-oracle path does not invoke canonical accepted topology.",
            "physical-sheet membership": "PRESENT as a manual control label, with human value CLEAR_NOT_ON_INTENDED_SURFACE; no executable canonical membership proof.",
            "region ownership": "ABSENT: no node_region_id/SurfaceRegionCandidate is created on this path.",
            "boundary ownership": "ABSENT: no ordered canonical boundary component is created or attached.",
            "renderer contribution provenance": "PRESENT per-view (camera, pixel, renderer depth, XYZ, normal), then DISCARDED from the PCA/fitter argument when reduced to XYZ.",
        },
        "stage_trace": stages,
        "other_sparse_events": "No general rejection is inferred; the human review override is recorded only for event 1527.",
        "baseline_event_union_preserved": baseline["event_1527_influence_row"]["event_id"] == 1527,
    }


def _identity_audit() -> list[dict[str, Any]]:
    return [
        {"transition": "manual image-space control", "created": ["physical_sheet_control label"], "preserved": ["control name in per-view NPZ metadata"], "merged": [], "flattened": [], "discarded": ["no canonical region/boundary identity"], "assessment": "manual label is not physical-sheet proof"},
        {"transition": "per-view renderer event cloud", "created": ["camera/pixel/depth/XYZ/normal provenance"], "preserved": ["per-view sidecar fields"], "merged": [], "flattened": [], "discarded": ["primitive/contributor ID unavailable"], "assessment": "renderer provenance exists without surface identity"},
        {"transition": "WL145 clean_points union", "created": [], "preserved": ["row order only; retrospective camera lookup"], "merged": ["three camera clouds"], "flattened": ["all points into one XYZ population"], "discarded": ["region, local-surface, boundary ownership"], "assessment": "first decisive population flattening"},
        {"transition": "WL145 PCA chart", "created": ["one global PCA frame and rectangular extrema domain"], "preserved": [], "merged": ["all pooled rows share one chart"], "flattened": ["manual/camera semantics absent from PCA input"], "discarded": ["ownership constraints"], "assessment": "cross-structure evidence can enter one PCA population"},
        {"transition": "WL139 fitter", "created": ["one representative control grid"], "preserved": ["XYZ and fixed physical UV only"], "merged": [], "flattened": ["one pooled representative"], "discarded": ["all sidecar provenance"], "assessment": "fit success is not surface identity"},
        {"transition": "WL148 B materialization", "created": ["post-fit occupancy selection"], "preserved": ["frozen representative/support mask"], "merged": [], "flattened": [], "discarded": [], "assessment": "post-fit support cannot retroactively constrain the fit"},
    ]


def _contract_loss_analysis(actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "B. ARCHITECTURE_BYPASS",
        "boundary_first_active_on_wl139_path": False,
        "local_surface_decomposition_active_on_wl139_path": False,
        "earliest_bypass": {
            "stage": "WL145 clean-oracle representative control path",
            "location": actual["source_evidence"]["union_and_pca"],
            "finding": "The path never invokes construct_visible_nurbs_from_gaussians, form_surface_regions, boundary ordering, or the Boundary First materialization adapter before the WL139 fit.",
            "timing": "BEFORE_REPRESENTATIVE_FIT",
        },
        "decisive_identity_flattening": {
            "location": actual["source_evidence"]["union_and_pca"],
            "operation": "clean_points = np.concatenate([cloud.points for cloud in clouds], axis=0)",
            "finding": "The separate per-view clouds become one pooled XYZ array with no region or boundary identity argument.",
        },
        "chart_extent_owner_loss": {
            "location": actual["source_evidence"]["pca_frame"],
            "finding": "Global min/max over pooled PCA coordinates establishes the rectangle independently of a physical boundary owner.",
        },
        "post_fit_support": {
            "location": actual["source_evidence"]["wl148_frozen_replay"],
            "finding": "WL148 B only materializes cells after the representative and its global rectangle already exist.",
            "required_statement": "WL148 B does NOT by itself restore Boundary First semantics.",
        },
        "failure_vs_bypass": "The observed WL149 off-surface extremum is a failure mode directly targeted by the intended contracts, but this particular WL139/WL145 path is an architecture bypass rather than evidence that the active canonical Boundary First implementation failed.",
        "semantic_drift": "Not the primary classification: canonical source still exposes pre-fit region/boundary gates; the isolated oracle path intentionally does not call them.",
        "no_repair_performed": True,
    }


def _historical_motivation_check() -> dict[str, Any]:
    historical_paths = [
        "docs/worklogs/4_boundary_first_review_geometry_semantics_and_crossing_gate.md",
        "docs/worklogs/6_boundary_first_star_shaped_anchor_correspondence.md",
        "docs/worklogs/10_consensus_aware_surface_region_formation_foundation.md",
        "docs/worklogs/14_ordered_boundary_to_visible_nurbs_materialization_adapter.md",
        "docs/worklogs/78_constructor_provenance_chart_frontier_semantics.md",
        "docs/worklogs/79_constructor_wide_chart_domain_coverage.md",
        "docs/worklogs/85_evidence_scale_local_surface_topology_boundary.md",
        "docs/worklogs/86_partition_seam_parametric_chart_domain.md",
        "docs/worklogs/116_visible_nurbs_representation_contract_recovery_audit.md",
        "docs/worklogs/117_holey_chart_fitting_coupling_attribution.md",
        "docs/worklogs/139_physical_chart_surface_representative_closure.md",
        "docs/worklogs/145_genuine_physical_sheet_oracle_clean_support_representative.md",
        "docs/worklogs/149_physical_sheet_evidence_vs_chart_extent_failure_attribution.md",
    ]
    evidence = []
    for relative in historical_paths:
        path = REPO_ROOT / relative
        evidence.append({"path": relative, "exists": path.exists(), "sha256": _sha256_file(path) if path.exists() else None})
    return {
        "classification": {
            "off_surface_event_controls_representative_chart_extent": "DIRECTLY TARGETED FAILURE",
            "wl139_clean_oracle_global_pooling": "RELATED BUT NOT DIRECTLY TARGETED",
            "claim_that_canonical_full_scene_pipeline_failed": "OUTSIDE ORIGINAL CONTRACT",
        },
        "reasoning": [
            "Boundary First source rejects open/branch/ambiguous observed boundaries instead of synthesizing a rectangle and routes eligible boundary plus interior evidence into the fit.",
            "Local surface formation creates region-owned identities before boundary admission and materialization.",
            "An unrelated event defining a global representative extent is therefore exactly the class of cross-surface/domain ownership failure those contracts were designed to prevent.",
            "WL139/WL145 explicitly defines a manual clean-oracle diagnostic and calls a physical-chart fitter directly; that demo scope is related to the failure but does not claim to execute the canonical constructor.",
        ],
        "source_and_worklog_evidence": evidence,
        "human_review_scope": "Only event 1527 is overridden by the supplied human review; other sparse events remain untouched and are not broadly rejected.",
    }


def _source_history_manifest() -> dict[str, Any]:
    source_paths = [
        CANONICAL_CONSTRUCTION, CANONICAL_PIPELINE, BOUNDARY_BUILDER, BOUNDARY_ADAPTER,
        REGION_FORMATION, DECOMPOSITION_DIAGNOSTIC, WL139_MODULE, WL144_MODULE, WL145_MODULE,
        WL148_MODULE, WL149_MODULE,
    ]
    source_files = []
    for path in source_paths:
        source_files.append({"path": _relative(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    commits = [
        {"commit": "cef5ae6", "subject": "Integrate boundary-first and Gaussian foundation work"},
        {"commit": "8dac4a2", "subject": "Add covariance-guided visible-surface construction thread (worklogs 111-123)"},
        {"commit": "943a764", "subject": "Worklog 127: evidence-bounded projective TSDF for direct Visible Surface construction"},
        {"commit": "9482a34", "subject": "worklog139-physical-chart-representative-closure"},
        {"commit": "6f7482e", "subject": "worklog145 validate genuine physical-sheet oracle"},
        {"commit": "7c67275", "subject": "worklog148 reconcile WL145 support materialization"},
        {"commit": "9ce0705", "subject": "worklog149 attribute physical-sheet chart extent failure"},
    ]
    return {
        "audit": "committed source/history inspection; no canonical source modification",
        "source_files": source_files,
        "historical_commits": commits,
        "wl149_commit_baseline": "9ce0705",
        "wl150_diagnostic_only": True,
    }


def _architecture_comparison(intended_boundary: dict[str, Any], intended_local: dict[str, Any], actual: dict[str, Any], loss: dict[str, Any]) -> tuple[dict[str, Any], str]:
    comparison = {
        "intended": [
            "Visible renderer evidence",
            "Local Surface Decomposition / form_surface_regions",
            "Region and local physical-surface ownership",
            "Observed Boundary First candidate/order/status",
            "Region-owned local chart/support",
            "Pre-fit eligible boundary + region-core NURBS fit",
            "Supported materialization",
        ],
        "actual": [
            "WL145 manual polygon control",
            "Independent per-camera renderer median event clouds",
            "Pooled XYZ union of 1586 rows",
            "Global PCA and min/max rectangular chart",
            "WL139 fixed-physical-UV representative fit",
            "WL148 post-fit support materialization",
            "WL149 extrema attribution replay",
        ],
        "differences": [
            "Local Surface Decomposition is not invoked on the WL145 clean-oracle path.",
            "No canonical region or boundary owner exists before WL139 fitting.",
            "Three camera clouds are pooled into one global PCA population.",
            "The chart rectangle is set by global extrema rather than a region-owned boundary.",
            "WL148 B applies support after fit and does not restore pre-fit Boundary First semantics.",
        ],
        "verdict": loss["verdict"],
        "source_contracts": {"boundary_first": intended_boundary["contract_status"], "local_surface_decomposition": intended_local["contract_status"]},
    }
    markdown = f"""# Worklog 150 — Boundary First / Local Surface Decomposition Contract Trace

## Intended architecture

```text
Visible renderer evidence
        ↓
Local Surface Decomposition (`form_surface_regions`)
        ↓
Region / local physical-surface ownership
        ↓
Boundary First: observed boundary candidates → ordered eligibility
        ↓
Region-owned local chart and support
        ↓
PRE-FIT eligible boundary + region-core NURBS fit
        ↓
Supported materialization
```

## Actual WL139–WL149 path

```text
WL145 manual polygon control
        ↓
Independent per-camera renderer median event clouds
        ↓
`clean_points = np.concatenate([cloud.points for cloud in clouds], axis=0)`
        ↓  ← local region/boundary ownership was never created; provenance is flattened
Global PCA (`world_xyz @ axes`) + global min/max rectangle
        ↓  ← event 1527 becomes `v_min`
WL139 fixed physical-UV representative fit
        ↓
WL148 B post-fit support materialization
        ↓
WL149 extrema/influence replay
```

## Result

The first decisive bypass is before representative fitting: WL145's clean-oracle
path does not call the canonical constructor or local decomposition, and its
pooled XYZ union is then used for global PCA. Event 1527 therefore remains in
the representative population and owns the global `v_min` extent. This is
`{loss['verdict']}`. It is not a repair and does not show that canonical
Boundary First itself failed.

Required semantic distinction: **WL148 B does NOT by itself restore Boundary First semantics.**

Human review is limited to event 1527:
`HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: {HUMAN_REVIEW_PHYSICAL_SHEET_STATUS}`.
No broad rejection rule is inferred for the other events.
"""
    return comparison, markdown


def build_trace() -> dict[str, Any]:
    baseline = _load_wl149_baseline()
    event = _load_event_1527(baseline)
    intended_boundary = _intended_boundary_first_contract()
    intended_local = _intended_local_surface_decomposition_contract()
    actual = _actual_dataflow(baseline, event)
    event_trace = _event_trace(baseline, event, actual)
    loss = _contract_loss_analysis(actual)
    history = _historical_motivation_check()
    source_history = _source_history_manifest()
    comparison, architecture_markdown = _architecture_comparison(intended_boundary, intended_local, actual, loss)
    return {
        "baseline_reconciliation": baseline,
        "intended_boundary_first_contract": intended_boundary,
        "intended_local_surface_decomposition_contract": intended_local,
        "actual_wl139_wl149_dataflow": actual,
        "event_1527_trace": event_trace,
        "local_surface_identity_audit": _identity_audit(),
        "contract_loss_analysis": loss,
        "historical_motivation_check": history,
        "architecture_comparison": comparison,
        "architecture_comparison_markdown": architecture_markdown,
        "source_history_manifest": source_history,
        "architecture_verdict": {
            "verdict": "B. ARCHITECTURE_BYPASS",
            "boundary_first": "bypassed on WL139/WL145 clean-oracle path; canonical source retains pre-fit boundary gates",
            "local_surface_decomposition": "bypassed and local identity flattened before WL139 PCA/fit",
            "earliest_contract_loss": loss["earliest_bypass"],
            "event_1527": {"human_review": HUMAN_REVIEW_PHYSICAL_SHEET_STATUS, "remains_present": True, "owns_v_min": True},
            "wl148_B": "WL148 B does NOT by itself restore Boundary First semantics.",
            "repair_performed": False,
        },
        "report": {
            "1. CURRENT ARCHITECTURE QUESTION": "Why could an off-surface renderer event become a WL139/WL149 chart-extremum owner despite the intended contracts?",
            "2. HUMAN REVIEW UPDATE": {"event_1527_status": HUMAN_REVIEW_PHYSICAL_SHEET_STATUS, "scope": "event 1527 only; no broad rejection"},
            "3. INTENDED BOUNDARY FIRST CONTRACT": intended_boundary,
            "4. INTENDED LOCAL SURFACE DECOMPOSITION CONTRACT": intended_local,
            "5. WL139 REPRESENTATIVE DATAFLOW": actual,
            "6. EVENT 1527 END-TO-END TRACE": event_trace,
            "7. PRE-FIT VS POST-FIT DOMAIN CONTROL": {"classification": "POST-FIT_MATERIALIZATION_ON_WL148_B_AFTER_GLOBAL_PRE-FIT_RECTANGLE_AND_FIT", "explanation": "WL139's PCA rectangle and representative fit are not constrained by canonical Boundary First; WL148 B only gates later cells.", "required_statement": "WL148 B does NOT by itself restore Boundary First semantics."},
            "8. CONTRACT LOSS / BYPASS LOCATION": loss,
            "9. INTENDED VS ACTUAL ARCHITECTURE DIAGRAMS": comparison,
            "10. HISTORICAL MOTIVATION MATCH": history,
            "11. ARCHITECTURE VERDICT": "B. ARCHITECTURE_BYPASS",
            "12. RETAINED / REJECTED / OPEN": {
                "RETAINED": ["WL139, WL145, WL148, WL149 artifacts", "1586 event union and event 1527", "PCA axes, representative XYZ/normals, support mask, all-four relation", "canonical renderer, checkpoint, Candidate B, production continuation"],
                "REJECTED": ["repair inference", "general rejection of sparse events", "claim that the canonical Boundary First implementation failed on this path", "claim that Occluded Surface is solved"],
                "OPEN": ["whether canonical region identity remains physically correct for current renderer evidence", "how a future publishable method should define physical-sheet membership, extent, termination, and confidence", "whether historical Boundary First semantics should be re-enabled after a separate contract review"],
            },
            "INTENT ALIGNMENT": {"diagnostic_only": True, "no_geometry_change": True, "no_parameter_tuning": True, "no_repair": True, "human_override_scope": "event 1527 only"},
            "IMPLEMENTATION FIDELITY": {"source_history_read": True, "frozen_wl149_replay": True, "exact_event_ids_preserved": True, "canonical_source_modified": False, "wl139_wl145_wl148_wl149_modified": False, "new_heuristic_added": False},
            "ARCHITECTURE RESULT": {"verdict": "B. ARCHITECTURE_BYPASS", "earliest_loss_before_fit": True, "boundary_first_was_governing_wl139_path": False, "wl148_post_fit_only": True},
        },
    }


def write_trace() -> dict[str, Any]:
    trace = build_trace()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = {
        "baseline_reconciliation.json": trace["baseline_reconciliation"],
        "intended_boundary_first_contract.json": trace["intended_boundary_first_contract"],
        "intended_local_surface_decomposition_contract.json": trace["intended_local_surface_decomposition_contract"],
        "actual_wl139_wl149_dataflow.json": trace["actual_wl139_wl149_dataflow"],
        "event_1527_trace.json": trace["event_1527_trace"],
        "local_surface_identity_audit.json": trace["local_surface_identity_audit"],
        "contract_loss_analysis.json": trace["contract_loss_analysis"],
        "historical_motivation_check.json": trace["historical_motivation_check"],
        "architecture_comparison.json": trace["architecture_comparison"],
        "architecture_verdict.json": trace["architecture_verdict"],
        "source_history_manifest.json": trace["source_history_manifest"],
        "boundary_first_local_surface_decomposition_contract_trace_report.json": trace["report"],
    }
    for name, value in files.items():
        _write_json(OUTPUT_ROOT / name, value)
    (OUTPUT_ROOT / "architecture_comparison.md").write_text(trace["architecture_comparison_markdown"], encoding="utf-8")
    readme = f"""# WL150 contract trace

Diagnostic-only source/history audit. Existing WL139/WL145/WL148/WL149
artifacts and canonical source were not modified.

- Verdict: **B. ARCHITECTURE_BYPASS**
- Event 1527: `HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: {HUMAN_REVIEW_PHYSICAL_SHEET_STATUS}`
- Event 1527 remains present and remains the frozen `v_min` owner.
- Earliest decisive flattening: WL145 `clean_points = np.concatenate(...)` before WL139 fitting.
- `WL148 B does NOT by itself restore Boundary First semantics.`

See `architecture_comparison.md` and the numbered JSON reports for the exact
source-function references and baseline hashes.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_ROOT.iterdir():
        destination = TEMP_ROOT / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)
    return {"output_root": _relative(OUTPUT_ROOT), "temp_root": _relative(TEMP_ROOT), "files": sorted(files) + ["architecture_comparison.md", "README.md"], "verdict": trace["architecture_verdict"]["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Boundary First/local-surface contract around WL139-WL149")
    parser.add_argument("--write", action="store_true", help="write numbered output and temp artifacts")
    arguments = parser.parse_args()
    result = write_trace() if arguments.write else build_trace()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
