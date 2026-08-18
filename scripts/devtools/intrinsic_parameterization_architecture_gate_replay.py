"""Worklog 100 -- bounded intrinsic parameterization architecture gate.

Worklog 99 found that 80.4% (37/46) of Worklog 98's orientation-coherent
components already fail a pre-fit parameter-domain check before any of its
own A/B/C candidates are even attempted -- but that validator had two
methodological confounds (UV-space kNN instead of source-graph adjacency;
an independently-signed PCA/support normal instead of the synchronized
Worklog 98 frame orientation). Both are fixed in
:mod:`~osn_gs.surface.torch_parametric_domain_validity` before this batch
runs (Worklog 100 section 1) -- there is no separate diagnostic worklog for
that correction.

This script re-evaluates Worklog 98's own coherent components (identical 3D
evidence, identical synchronized tangent field, identical supported edges,
identical held-out evidence -- no candidate is tuned from held-out/fit
results) against THREE parameter-domain construction candidates, no
fallback between them:

A. TREE_INTEGRATED_UV -- Worklog 98's own tree-integrated ``(u, v)``,
   completely unmodified, evaluated only with the corrected validator.
B. GLOBAL_DIFFERENTIAL_INTEGRATION -- a real global weighted least-squares
   solve over every continuously-supported edge simultaneously
   (:func:`~osn_gs.surface.torch_global_differential_uv_integration.integrate_global_differential_uv`).
C. ORIENTATION_PRESERVING_GLOBAL_INTEGRATION -- initialized strictly from
   B, plus a fixed local-injectivity refinement schedule
   (:func:`~osn_gs.surface.torch_orientation_preserving_uv_integration.integrate_orientation_preserving_uv`).

For every candidate whose resulting UV passes the corrected pre-fit domain
contract, the EXISTING, UNCHANGED Worklog 97/98 network-native NURBS fitter
(``fit_torch_visible_surface_from_uv``, degree=2, 6x6 control grid, current
regularized solver, current safety/extrapolation thresholds) is the only
patch representation evaluated in this batch -- no NURBS capacity tuning,
no Gordon/adaptive fitting, no curve-seeding changes, no raw Gaussian
connectivity, no PCA-UV.

Fixed and unmodified throughout: visible Gaussian training, ADC, region
ownership, Worklog 95 latent-surface estimator, continuous support
contract, Worklog 98 synchronized tangent-frame field/coherent components,
curve construction, held-out evaluation convention (Worklog 87's ``_holdout``),
and the existing extrapolative/unsafe/valid_supported definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _holdout, _median_nn
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from curve_network_native_fit_replay import classify_fitted_surface
from osn_gs.surface.torch_global_differential_uv_integration import integrate_global_differential_uv
from osn_gs.surface.torch_latent_surface_edge_differential import build_edge_differentials
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, build_seed_curves
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv
from osn_gs.surface.torch_orientation_preserving_uv_integration import integrate_orientation_preserving_uv
from osn_gs.surface.torch_parametric_domain_validity import (
    assess_parametric_domain_validity,
    cycle_position_drift_p95,
)

CATEGORY_DOMAIN_INVALID = "PARAMETER_DOMAIN_INVALID"
CATEGORY_FIT_FAILED = "FIT_FAILED"
CATEGORY_EXTRAPOLATIVE = "FIT_SUCCEEDED_BUT_EXTRAPOLATIVE"
CATEGORY_UNSAFE = "FIT_SUCCEEDED_BUT_UNSAFE"
CATEGORY_VALID = "VALID_SUPPORTED"

NURBS_RESOLUTION = 6  # fixed 6x6 control grid, unchanged from Worklog 98
NURBS_DEGREE = 2


def _pick_field_anchor(seeds) -> tuple[object, object] | tuple[None, None]:
    for seed in seeds:
        if seed.seed_type != SEED_INTERIOR_CONSTRUCTION and int(seed.points.shape[0]) >= 2:
            anchor_position = seed.points[0]
            anchor_hint = seed.points[1] - seed.points[0]
            if float(anchor_hint.norm().item()) > 1e-9:
                return anchor_position, anchor_hint
    return None, None


def _domain_record(report) -> dict:
    return {
        "valid": report.valid, "invalid_reasons": list(report.invalid_reasons),
        "u_extent": report.u_extent, "v_extent": report.v_extent,
        "duplicate_incompatible_count": report.duplicate_incompatible_count,
        "global_orientation_flip_applied": report.global_orientation_flip_applied,
        "fold_fraction": report.fold_fraction, "singular_fraction": report.singular_fraction,
        "mean_condition_number": report.mean_condition_number,
        "max_condition_number": report.max_condition_number,
        "stretch_ratio_p95": report.stretch_ratio_p95,
        "area_distortion_p95": report.area_distortion_p95,
        "shear_distortion_p95": report.shear_distortion_p95,
    }


def _fit_and_classify(points, uv, held_out_target, scale) -> dict:
    try:
        surface = fit_torch_visible_surface_from_uv(
            points, uv, resolution_u=NURBS_RESOLUTION, resolution_v=NURBS_RESOLUTION,
            degree_u=NURBS_DEGREE, degree_v=NURBS_DEGREE,
        )
    except Exception as exc:  # noqa: BLE001
        return {"category": CATEGORY_FIT_FAILED, "fit_error": f"{type(exc).__name__}: {exc}"}
    record = classify_fitted_surface(surface, points, held_out_target, scale)
    classification = record.get("classification")
    category = {
        "unsafe_geometry": CATEGORY_UNSAFE, "extrapolative": CATEGORY_EXTRAPOLATIVE,
        "valid_supported": CATEGORY_VALID,
    }.get(classification, CATEGORY_FIT_FAILED)
    record["category"] = category
    return record


def analyze_component(component, support, held_out_target: torch.Tensor, scale: float) -> dict:
    record: dict = {"component_size": len(component.node_indices), "coherent": component.coherent}

    # --- Candidate A: TREE_INTEGRATED_UV, unmodified, corrected validator only. ---
    uv_a = torch.stack([component.u, component.v], dim=1)
    report_a = assess_parametric_domain_validity(component, uv_a, support.median_spacing)
    cycle_drift = cycle_position_drift_p95(component, support.median_spacing)
    record["A_tree_integrated_uv"] = {"domain_report": _domain_record(report_a), "cycle_position_drift_p95_over_spacing": cycle_drift}
    if report_a.valid:
        record["A_tree_integrated_uv"]["fit"] = _fit_and_classify(component.positions, uv_a, held_out_target, scale)
    else:
        record["A_tree_integrated_uv"]["fit"] = {"category": CATEGORY_DOMAIN_INVALID}

    # --- Edge differential constraints (shared input for B and C). ---
    edges = build_edge_differentials(component, support.median_spacing)
    record["edge_count"] = len(edges)

    # --- Candidate B: GLOBAL_DIFFERENTIAL_INTEGRATION. ---
    result_b = integrate_global_differential_uv(component, edges)
    if result_b.valid:
        report_b = assess_parametric_domain_validity(component, result_b.uv, support.median_spacing)
        record["B_global_differential_integration"] = {
            "integration_valid": True,
            "overall_residual_rms": result_b.overall_residual_rms,
            "per_edge_residual_p50": result_b.per_edge_residual_p50,
            "per_edge_residual_p95": result_b.per_edge_residual_p95,
            "cycle_edge_residual_rms": result_b.cycle_edge_residual_rms,
            "domain_report": _domain_record(report_b),
        }
        if report_b.valid:
            record["B_global_differential_integration"]["fit"] = _fit_and_classify(
                component.positions, result_b.uv, held_out_target, scale,
            )
        else:
            record["B_global_differential_integration"]["fit"] = {"category": CATEGORY_DOMAIN_INVALID}
    else:
        record["B_global_differential_integration"] = {"integration_valid": False, "invalid_reason": result_b.invalid_reason}

    # --- Candidate C: ORIENTATION_PRESERVING_GLOBAL_INTEGRATION, initialized strictly from B. ---
    result_c = integrate_orientation_preserving_uv(component, edges, support.median_spacing)
    record["C_orientation_preserving_global_integration"] = {
        "integration_valid": result_c.valid,
        "invalid_reason": result_c.invalid_reason,
        "refinement_iterations_used": result_c.refinement_iterations_used,
    }
    if result_c.valid:
        report_c = assess_parametric_domain_validity(component, result_c.uv, support.median_spacing)
        record["C_orientation_preserving_global_integration"]["domain_report"] = _domain_record(report_c)
        if report_c.valid:
            record["C_orientation_preserving_global_integration"]["fit"] = _fit_and_classify(
                component.positions, result_c.uv, held_out_target, scale,
            )
        else:
            record["C_orientation_preserving_global_integration"]["fit"] = {"category": CATEGORY_DOMAIN_INVALID}
    else:
        record["C_orientation_preserving_global_integration"]["fit"] = {"category": CATEGORY_DOMAIN_INVALID}

    return record


def analyze_region(
    region_id: int, evidence: torch.Tensor, chart, representative_positions, representative_index,
) -> dict:
    train_evidence, held_evidence = _holdout(evidence)
    if int(train_evidence.shape[0]) < 4:
        return {"region": region_id, "skip_reason": "insufficient_train_split_evidence"}

    support = build_latent_surface_support(train_evidence)
    held_out_target = held_evidence if int(held_evidence.shape[0]) > 0 else evidence
    scale = _median_nn(held_out_target) if int(held_out_target.shape[0]) >= 2 else _median_nn(evidence)

    seeds = build_seed_curves(train_evidence, chart, representative_positions, representative_index, support)
    anchor_position, anchor_hint = _pick_field_anchor(seeds)
    field_result = build_tangent_frame_field(
        train_evidence, support, anchor_position=anchor_position, anchor_hint_direction=anchor_hint,
    )
    coherent_components = [component for component in field_result.components if component.coherent]

    component_records = [
        analyze_component(component, support, held_out_target, scale) for component in coherent_components
    ]
    return {
        "region": region_id,
        "held_out_evidence_size": int(held_out_target.shape[0]),
        "coherent_component_count": len(coherent_components),
        "components": component_records,
    }


def analyze(checkpoint: Path, cap: int, device: str) -> dict:
    model, stable_ids = _load_model(checkpoint, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    region_rows = []
    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        if len(full_indices) < 4:
            region_rows.append({"region": region_id, "skip_reason": "insufficient_owned_evidence"})
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        chart = chart_by_region.get(region_id)
        region_rows.append(
            analyze_region(region_id, evidence, chart, representative_positions, representative_index)
        )

    def _aggregate(candidate_key: str) -> dict:
        category_evidence = Counter()
        total_held_out = 0
        p95_values: list[float] = []
        domain_valid_count = 0
        integration_failed_count = 0
        fold_fraction_values: list[float] = []
        area_distortion_values: list[float] = []
        shear_distortion_values: list[float] = []
        condition_number_values: list[float] = []
        global_flip_count = 0
        refinement_iterations: list[int] = []
        for row in region_rows:
            components = row.get("components", [])
            if not components:
                continue
            held_size = row["held_out_evidence_size"]
            share = held_size / len(components)
            total_held_out += held_size
            for record in components:
                candidate = record.get(candidate_key, {})
                if candidate.get("integration_valid") is False:
                    integration_failed_count += 1
                    category_evidence[CATEGORY_DOMAIN_INVALID] += share
                    continue
                domain_report = candidate.get("domain_report")
                if domain_report is not None:
                    if domain_report["valid"]:
                        domain_valid_count += 1
                    fold_fraction_values.append(domain_report["fold_fraction"])
                    if domain_report["global_orientation_flip_applied"]:
                        global_flip_count += 1
                    if domain_report["area_distortion_p95"] is not None:
                        area_distortion_values.append(domain_report["area_distortion_p95"])
                    if domain_report["shear_distortion_p95"] is not None:
                        shear_distortion_values.append(domain_report["shear_distortion_p95"])
                    if domain_report["mean_condition_number"] is not None:
                        condition_number_values.append(domain_report["mean_condition_number"])
                if candidate_key == "C_orientation_preserving_global_integration":
                    refinement_iterations.append(candidate.get("refinement_iterations_used", 0))
                fit = candidate.get("fit", {})
                category = fit.get("category", CATEGORY_DOMAIN_INVALID)
                category_evidence[category] += share
                p95 = fit.get("extrapolation_p95")
                if p95 is not None:
                    p95_values.append(p95)
        total = total_held_out or 1
        total_components = sum(len(row.get("components", [])) for row in region_rows)
        return {
            "component_count": total_components,
            "domain_valid_component_count": domain_valid_count,
            "integration_failed_component_count": integration_failed_count,
            "global_orientation_flip_count": global_flip_count,
            "mean_fold_fraction": (sum(fold_fraction_values) / len(fold_fraction_values)) if fold_fraction_values else None,
            "mean_area_distortion": (sum(area_distortion_values) / len(area_distortion_values)) if area_distortion_values else None,
            "mean_shear_distortion": (sum(shear_distortion_values) / len(shear_distortion_values)) if shear_distortion_values else None,
            "mean_condition_number": (sum(condition_number_values) / len(condition_number_values)) if condition_number_values else None,
            "mean_refinement_iterations": (sum(refinement_iterations) / len(refinement_iterations)) if refinement_iterations else None,
            "evidence_weighted_fractions": {
                key: category_evidence.get(key, 0) / total
                for key in (CATEGORY_DOMAIN_INVALID, CATEGORY_FIT_FAILED, CATEGORY_EXTRAPOLATIVE, CATEGORY_UNSAFE, CATEGORY_VALID)
            },
            "held_out_p50_mean": (sorted(p95_values)[len(p95_values) // 2]) if p95_values else None,
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out,
        }

    total_coherent = sum(row.get("coherent_component_count", 0) for row in region_rows)

    return {
        "checkpoint": str(checkpoint), "cap": cap,
        "regions": region_rows,
        "summary": {
            "total_coherent_components": total_coherent,
            "A_tree_integrated_uv": _aggregate("A_tree_integrated_uv"),
            "B_global_differential_integration": _aggregate("B_global_differential_integration"),
            "C_orientation_preserving_global_integration": _aggregate("C_orientation_preserving_global_integration"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/extent_ab/val100/intrinsic_parameterization_architecture_gate_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()
    report = analyze(args.checkpoint, args.cap, args.device)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["summary"]["runtime_seconds"] = elapsed
    report["summary"]["peak_python_memory_bytes"] = peak

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
