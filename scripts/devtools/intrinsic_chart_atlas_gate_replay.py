"""Worklog 101 -- intrinsic-integrability-driven local chart atlas gate.

Worklog 100 found that forcing ONE global (u, v) chart over an entire
Worklog 98 coherent component fails a majority of the time regardless of
integration method: global differential integration (candidate B) only
modestly improved domain validity over tree integration (15->18/46), and a
fixed local-injectivity refinement (candidate C) rescued ZERO additional
components beyond B. This script tests the next architecture step directly:
does decomposing each coherent component into a deterministic ATLAS of
smaller, overlapping, individually-valid local charts
(:mod:`~osn_gs.surface.torch_intrinsic_chart_atlas`) turn most of that
previously-unparameterizable evidence into valid intrinsic charts?

A. SINGLE_COMPONENT_GLOBAL_UV -- Worklog 100 candidate B, unchanged: one
   global differential integration over the WHOLE coherent component.
B. LOCAL_INTRINSIC_CHART_ATLAS -- this batch: graph-ring chart growth,
   candidate-B integration per chart, corrected validator per chart, no
   chart creation/resize/merge/split informed by fit or held-out error.

For every domain-valid chart, the EXISTING, UNCHANGED Worklog 97/98
network-native NURBS fitter (``fit_torch_visible_surface_from_uv``,
degree=2, 6x6 control grid, current regularized solver, current
safety/extrapolation thresholds) is the only downstream probe -- no NURBS
capacity tuning. A chart below the fixed 6x6 grid's own control-point count
(36) is reported as ``CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT``,
never conflated with ``PARAMETER_DOMAIN_INVALID``.

Fixed and unmodified throughout: visible Gaussian training, ADC, region
ownership, Worklog 95 latent-surface estimator, continuous support
contract, Worklog 98 synchronized tangent-frame field/coherent components,
Worklog 100 symmetric edge differentials/global differential integration,
corrected source-graph parameter-domain validator, held-out evaluation
convention (Worklog 87's ``_holdout``), and the existing
extrapolative/unsafe/valid_supported definitions.
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
from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_edge_differential import build_edge_differentials
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, build_seed_curves
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv
from osn_gs.surface.torch_parametric_domain_validity import assess_parametric_domain_validity

CATEGORY_DOMAIN_INVALID = "PARAMETER_DOMAIN_INVALID"
CATEGORY_INSUFFICIENT_PATCH_SUPPORT = "CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT"
CATEGORY_FIT_FAILED = "FIT_FAILED"
CATEGORY_EXTRAPOLATIVE = "FIT_SUCCEEDED_BUT_EXTRAPOLATIVE"
CATEGORY_UNSAFE = "FIT_SUCCEEDED_BUT_UNSAFE"
CATEGORY_VALID = "VALID_SUPPORTED"

NURBS_RESOLUTION = 6  # fixed 6x6 control grid, unchanged from Worklog 97/98/100
NURBS_DEGREE = 2
MIN_PATCH_SUPPORT = NURBS_RESOLUTION * NURBS_RESOLUTION  # structural fact of the fixed grid, not replay-tuned


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
        "global_orientation_flip_applied": report.global_orientation_flip_applied,
        "fold_fraction": report.fold_fraction, "singular_fraction": report.singular_fraction,
        "mean_condition_number": report.mean_condition_number,
        "max_condition_number": report.max_condition_number,
        "stretch_ratio_p95": report.stretch_ratio_p95,
        "area_distortion_p95": report.area_distortion_p95,
        "shear_distortion_p95": report.shear_distortion_p95,
    }


def _fit_and_classify(points, uv, held_out_target, scale) -> dict:
    if int(points.shape[0]) < MIN_PATCH_SUPPORT:
        return {"category": CATEGORY_INSUFFICIENT_PATCH_SUPPORT, "support_count": int(points.shape[0])}
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


def analyze_single_component(component, support, held_out_target, scale, median_spacing) -> dict:
    """Candidate A: SINGLE_COMPONENT_GLOBAL_UV -- Worklog 100 candidate B, unchanged."""

    edges = build_edge_differentials(component, median_spacing)
    integration = integrate_global_differential_uv(component, edges)
    if not integration.valid:
        return {"integration_valid": False, "invalid_reason": integration.invalid_reason, "fit": {"category": CATEGORY_DOMAIN_INVALID}}
    report = assess_parametric_domain_validity(component, integration.uv, median_spacing)
    record = {"integration_valid": True, "domain_report": _domain_record(report)}
    if report.valid:
        record["fit"] = _fit_and_classify(component.positions, integration.uv, held_out_target, scale)
    else:
        record["fit"] = {"category": CATEGORY_DOMAIN_INVALID}
    return record


def analyze_chart_atlas(component, support, held_out_target, scale, median_spacing) -> dict:
    """Candidate B: LOCAL_INTRINSIC_CHART_ATLAS."""

    atlas = build_local_chart_atlas(component, median_spacing)
    chart_records = []
    for chart in atlas.charts:
        record = {
            "chart_id": chart.chart_id, "size": len(chart.node_indices), "ring_reached": chart.ring_reached,
            "domain_report": _domain_record(chart.domain_report),
            "overall_residual_rms": chart.integration.overall_residual_rms,
            "per_edge_residual_p95": chart.integration.per_edge_residual_p95,
            "cycle_edge_residual_rms": chart.integration.cycle_edge_residual_rms,
        }
        record["fit"] = _fit_and_classify(chart.component.positions, chart.integration.uv, held_out_target, scale)
        chart_records.append(record)

    total = int(component.positions.shape[0])
    return {
        "component_size": total,
        "chart_count": len(atlas.charts),
        "covered_count": len(atlas.covered_node_indices),
        "multiply_covered_count": len(atlas.multiply_covered_node_indices),
        "uncovered_count": len(atlas.uncovered_node_indices),
        "unchartable_seed_count": len(atlas.unchartable_seed_node_indices),
        "seam_edge_count": len(atlas.seam_edges),
        "chart_node_counts": [len(chart.node_indices) for chart in atlas.charts],
        "chart_ring_reached": [chart.ring_reached for chart in atlas.charts],
        "charts": chart_records,
    }


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

    component_records = []
    for component in coherent_components:
        component_records.append({
            "component_size": len(component.node_indices),
            "A_single_component_global_uv": analyze_single_component(
                component, support, held_out_target, scale, support.median_spacing,
            ),
            "B_local_intrinsic_chart_atlas": analyze_chart_atlas(
                component, support, held_out_target, scale, support.median_spacing,
            ),
        })
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
        region_chart = chart_by_region.get(region_id)
        region_rows.append(
            analyze_region(region_id, evidence, region_chart, representative_positions, representative_index)
        )

    def _aggregate_single() -> dict:
        category_evidence = Counter()
        total_held_out = 0
        p95_values: list[float] = []
        domain_valid_count = 0
        total_components = 0
        for row in region_rows:
            components = row.get("components", [])
            if not components:
                continue
            held_size = row["held_out_evidence_size"]
            share = held_size / len(components)
            total_held_out += held_size
            for record in components:
                total_components += 1
                candidate = record["A_single_component_global_uv"]
                domain_report = candidate.get("domain_report")
                if domain_report is not None and domain_report["valid"]:
                    domain_valid_count += 1
                fit = candidate.get("fit", {})
                category = fit.get("category", CATEGORY_DOMAIN_INVALID)
                category_evidence[category] += share
                p95 = fit.get("extrapolation_p95")
                if p95 is not None:
                    p95_values.append(p95)
        total = total_held_out or 1
        return {
            "component_count": total_components, "domain_valid_component_count": domain_valid_count,
            "evidence_weighted_fractions": {
                key: category_evidence.get(key, 0) / total
                for key in (CATEGORY_DOMAIN_INVALID, CATEGORY_FIT_FAILED, CATEGORY_EXTRAPOLATIVE, CATEGORY_UNSAFE, CATEGORY_VALID)
            },
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out,
        }

    def _aggregate_atlas() -> dict:
        category_evidence = Counter()
        chart_category_count = Counter()
        total_held_out = 0
        p95_values: list[float] = []
        total_charts = 0
        domain_valid_charts = 0
        components_with_one_chart = 0
        components_with_multiple_charts = 0
        components_with_zero_charts = 0
        total_covered = total_uncovered = total_unchartable = 0
        total_source_nodes = 0
        multiply_covered_total = 0
        seam_edge_total = 0
        chart_node_counts: list[int] = []
        for row in region_rows:
            components = row.get("components", [])
            if not components:
                continue
            held_size = row["held_out_evidence_size"]
            share_per_component = held_size / len(components)
            total_held_out += held_size
            for record in components:
                atlas = record["B_local_intrinsic_chart_atlas"]
                chart_count = atlas["chart_count"]
                total_charts += chart_count
                total_source_nodes += atlas["component_size"]
                total_covered += atlas["covered_count"]
                total_uncovered += atlas["uncovered_count"]
                total_unchartable += atlas["unchartable_seed_count"]
                multiply_covered_total += atlas["multiply_covered_count"]
                seam_edge_total += atlas["seam_edge_count"]
                chart_node_counts.extend(atlas["chart_node_counts"])
                if chart_count == 0:
                    components_with_zero_charts += 1
                elif chart_count == 1:
                    components_with_one_chart += 1
                else:
                    components_with_multiple_charts += 1
                if chart_count == 0:
                    continue
                share_per_chart = share_per_component / chart_count
                for chart_record in atlas["charts"]:
                    if chart_record["domain_report"]["valid"]:
                        domain_valid_charts += 1
                    fit = chart_record.get("fit", {})
                    category = fit.get("category", CATEGORY_DOMAIN_INVALID)
                    chart_category_count[category] += 1
                    category_evidence[category] += share_per_chart
                    p95 = fit.get("extrapolation_p95")
                    if p95 is not None:
                        p95_values.append(p95)
        total = total_held_out or 1
        node_counts_sorted = sorted(chart_node_counts)
        return {
            "total_charts": total_charts, "domain_valid_charts": domain_valid_charts,
            "components_with_one_chart": components_with_one_chart,
            "components_with_multiple_charts": components_with_multiple_charts,
            "components_with_zero_charts": components_with_zero_charts,
            "total_source_nodes": total_source_nodes,
            "covered_node_total": total_covered, "multiply_covered_node_total": multiply_covered_total,
            "uncovered_node_total": total_uncovered, "unchartable_seed_node_total": total_unchartable,
            "seam_edge_total": seam_edge_total,
            "median_chart_node_count": node_counts_sorted[len(node_counts_sorted) // 2] if node_counts_sorted else None,
            "p95_chart_node_count": (
                node_counts_sorted[int(0.95 * (len(node_counts_sorted) - 1))] if node_counts_sorted else None
            ),
            "chart_category_counts": dict(chart_category_count),
            "evidence_weighted_fractions": {
                key: category_evidence.get(key, 0) / total
                for key in (
                    CATEGORY_DOMAIN_INVALID, CATEGORY_INSUFFICIENT_PATCH_SUPPORT, CATEGORY_FIT_FAILED,
                    CATEGORY_EXTRAPOLATIVE, CATEGORY_UNSAFE, CATEGORY_VALID,
                )
            },
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out,
        }

    total_coherent = sum(row.get("coherent_component_count", 0) for row in region_rows)

    return {
        "checkpoint": str(checkpoint), "cap": cap,
        "regions": region_rows,
        "summary": {
            "total_coherent_components": total_coherent,
            "A_single_component_global_uv": _aggregate_single(),
            "B_local_intrinsic_chart_atlas": _aggregate_atlas(),
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
        default=Path("output/extent_ab/val101/intrinsic_chart_atlas_gate_replay.json"),
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
