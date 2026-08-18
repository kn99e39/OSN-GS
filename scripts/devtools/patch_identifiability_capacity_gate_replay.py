"""Worklog 102 -- pre-fit patch identifiability / adaptive capacity gate.

Worklog 101 found chart-level domain validity is high (materialized charts
are domain-valid by construction; the meaningful figure is source-evidence
coverage, 87.8% combined), but rejected any chart below the fixed 6x6
grid's 36-sample count as structurally insufficient. That count-based
cutoff is NOT a mathematical requirement of the existing regularized
tensor-product NURBS solver: a Tikhonov-anchored, second-difference-
penalized system is solvable even when the raw data design matrix is
underdetermined. This script replaces the count-based cutoff with the
explicit pre-fit algebraic identifiability contract in
:mod:`~osn_gs.surface.torch_patch_identifiability`, and compares THREE
capacity/degree strategies, no fallback between them:

A. FIXED_6x6_DEGREE2 -- the existing Worklog 101 downstream probe,
   unchanged. Identifiability reported separately from fitting outcome.
B. ADAPTIVE_QUADRATIC_NURBS -- degree fixed at 2, largest identifiable
   tensor-product control lattice from 3x3 up to 6x6
   (:func:`~osn_gs.surface.torch_adaptive_patch_capacity.select_adaptive_quadratic_capacity`).
C. SUPPORT_ADAPTIVE_LOCAL_NURBS -- degree AND capacity chosen solely from
   pre-fit identifiability, preferring the highest order/capacity that is
   actually identifiable, degree 1 (min 2x2) available as a fallback below
   degree 2 (min 3x3)
   (:func:`~osn_gs.surface.torch_adaptive_patch_capacity.select_support_adaptive_capacity`).

All three candidates use the EXACT SAME Worklog 101 chart membership,
intrinsic UV, source evidence, overlap, synchronized frame, and held-out
evidence -- nothing about chart construction is touched in this batch.

Fixed and unmodified throughout: visible Gaussian training, ADC, region
ownership, Worklog 95 latent-surface estimator, continuous support
contract, Worklog 98 synchronized tangent-frame field, Worklog 100
symmetric edge differentials/global differential integration/corrected
source-graph validator, Worklog 101 local intrinsic chart atlas, held-out
evaluation convention (Worklog 87's ``_holdout``), and the existing
regularized tensor-product NURBS solver.
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
from osn_gs.surface.torch_adaptive_patch_capacity import (
    select_adaptive_quadratic_capacity,
    select_support_adaptive_capacity,
)
from osn_gs.surface.torch_chart_overlap_consistency import evaluate_overlap_consistency
from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, build_seed_curves
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv
from osn_gs.surface.torch_patch_identifiability import assess_patch_identifiability

CATEGORY_NOT_IDENTIFIABLE = "PATCH_NOT_IDENTIFIABLE"
CATEGORY_FIT_FAILED = "FIT_FAILED"
CATEGORY_EXTRAPOLATIVE = "FIT_SUCCEEDED_BUT_EXTRAPOLATIVE"
CATEGORY_UNSAFE = "FIT_SUCCEEDED_BUT_UNSAFE"
CATEGORY_VALID = "VALID_SUPPORTED"

FIXED_RESOLUTION = 6
FIXED_DEGREE = 2


def _pick_field_anchor(seeds) -> tuple[object, object] | tuple[None, None]:
    for seed in seeds:
        if seed.seed_type != SEED_INTERIOR_CONSTRUCTION and int(seed.points.shape[0]) >= 2:
            anchor_position = seed.points[0]
            anchor_hint = seed.points[1] - seed.points[0]
            if float(anchor_hint.norm().item()) > 1e-9:
                return anchor_position, anchor_hint
    return None, None


def _identifiability_record(report) -> dict:
    return {
        "identifiable": report.identifiable, "invalid_reason": report.invalid_reason,
        "degree_u": report.degree_u, "degree_v": report.degree_v,
        "control_grid_u": report.control_grid_u, "control_grid_v": report.control_grid_v,
        "sample_count": report.sample_count, "control_variable_count": report.control_variable_count,
        "effective_rank": report.effective_rank, "achievable_rank": report.achievable_rank,
        "condition_number": report.condition_number,
        "u_extent": report.u_extent, "v_extent": report.v_extent,
        "u_constrained": report.u_constrained, "v_constrained": report.v_constrained,
    }


def _fit_and_classify(points, uv, degree_u, degree_v, resolution_u, resolution_v, held_out_target, scale) -> dict:
    try:
        surface = fit_torch_visible_surface_from_uv(
            points, uv, resolution_u=resolution_u, resolution_v=resolution_v,
            degree_u=degree_u, degree_v=degree_v,
        )
    except Exception as exc:  # noqa: BLE001
        return {"category": CATEGORY_FIT_FAILED, "fit_error": f"{type(exc).__name__}: {exc}"}, None
    record = classify_fitted_surface(surface, points, held_out_target, scale)
    classification = record.get("classification")
    category = {
        "unsafe_geometry": CATEGORY_UNSAFE, "extrapolative": CATEGORY_EXTRAPOLATIVE,
        "valid_supported": CATEGORY_VALID,
    }.get(classification, CATEGORY_FIT_FAILED)
    record["category"] = category
    return record, surface


def _candidate_a(points, uv, held_out_target, scale) -> tuple[dict, object | None]:
    report = assess_patch_identifiability(uv, FIXED_DEGREE, FIXED_DEGREE, FIXED_RESOLUTION, FIXED_RESOLUTION)
    record: dict = {"identifiability": _identifiability_record(report)}
    if not report.identifiable:
        record["fit"] = {"category": CATEGORY_NOT_IDENTIFIABLE}
        return record, None
    fit_record, surface = _fit_and_classify(
        points, uv, FIXED_DEGREE, FIXED_DEGREE, FIXED_RESOLUTION, FIXED_RESOLUTION, held_out_target, scale,
    )
    record["fit"] = fit_record
    return record, surface


def _candidate_b(points, uv, held_out_target, scale) -> tuple[dict, object | None]:
    selection = select_adaptive_quadratic_capacity(uv)
    record: dict = {
        "selected": selection.selected, "degree_u": selection.degree_u, "degree_v": selection.degree_v,
        "control_grid_u": selection.control_grid_u, "control_grid_v": selection.control_grid_v,
        "identifiability": _identifiability_record(selection.report) if selection.report else None,
    }
    if not selection.selected:
        record["fit"] = {"category": CATEGORY_NOT_IDENTIFIABLE}
        return record, None
    fit_record, surface = _fit_and_classify(
        points, uv, selection.degree_u, selection.degree_v,
        selection.control_grid_u, selection.control_grid_v, held_out_target, scale,
    )
    record["fit"] = fit_record
    return record, surface


def _candidate_c(points, uv, held_out_target, scale) -> tuple[dict, object | None]:
    selection = select_support_adaptive_capacity(uv)
    record: dict = {
        "selected": selection.selected, "degree_u": selection.degree_u, "degree_v": selection.degree_v,
        "control_grid_u": selection.control_grid_u, "control_grid_v": selection.control_grid_v,
        "identifiability": _identifiability_record(selection.report) if selection.report else None,
    }
    if not selection.selected:
        record["fit"] = {"category": CATEGORY_NOT_IDENTIFIABLE}
        return record, None
    fit_record, surface = _fit_and_classify(
        points, uv, selection.degree_u, selection.degree_v,
        selection.control_grid_u, selection.control_grid_v, held_out_target, scale,
    )
    record["fit"] = fit_record
    return record, surface


def analyze_chart_atlas(component, support, held_out_target, scale, median_spacing) -> dict:
    atlas = build_local_chart_atlas(component, median_spacing)
    chart_records = []
    surfaces_a: dict[int, object | None] = {}
    surfaces_b: dict[int, object | None] = {}
    surfaces_c: dict[int, object | None] = {}
    uv_by_chart: dict[int, object] = {}

    for chart in atlas.charts:
        points = chart.component.positions
        uv = chart.integration.uv
        uv_by_chart[chart.chart_id] = uv

        record_a, surface_a = _candidate_a(points, uv, held_out_target, scale)
        record_b, surface_b = _candidate_b(points, uv, held_out_target, scale)
        record_c, surface_c = _candidate_c(points, uv, held_out_target, scale)
        surfaces_a[chart.chart_id] = surface_a
        surfaces_b[chart.chart_id] = surface_b
        surfaces_c[chart.chart_id] = surface_c

        chart_records.append({
            "chart_id": chart.chart_id, "size": len(chart.node_indices),
            "A_fixed_6x6_degree2": record_a,
            "B_adaptive_quadratic_nurbs": record_b,
            "C_support_adaptive_local_nurbs": record_c,
        })

    overlap_a = evaluate_overlap_consistency(list(atlas.charts), surfaces_a, uv_by_chart, scale)
    overlap_b = evaluate_overlap_consistency(list(atlas.charts), surfaces_b, uv_by_chart, scale)
    overlap_c = evaluate_overlap_consistency(list(atlas.charts), surfaces_c, uv_by_chart, scale)

    def _overlap_summary(pairs) -> dict:
        both_fitted_pairs = [pair for pair in pairs if pair.both_fitted]
        position_p95 = [pair.position_disagreement_p95 for pair in both_fitted_pairs]
        normal_p95 = [pair.normal_disagreement_degrees_p95 for pair in both_fitted_pairs]
        return {
            "pair_count": len(pairs), "both_fitted_pair_count": len(both_fitted_pairs),
            "mean_position_disagreement_p95": (sum(position_p95) / len(position_p95)) if position_p95 else None,
            "mean_normal_disagreement_degrees_p95": (sum(normal_p95) / len(normal_p95)) if normal_p95 else None,
        }

    return {
        "component_size": int(component.positions.shape[0]),
        "chart_count": len(atlas.charts),
        "covered_count": len(atlas.covered_node_indices),
        "multiply_covered_count": len(atlas.multiply_covered_node_indices),
        "uncovered_count": len(atlas.uncovered_node_indices),
        "unchartable_seed_count": len(atlas.unchartable_seed_node_indices),
        "charts": chart_records,
        "overlap_consistency": {
            "A_fixed_6x6_degree2": _overlap_summary(overlap_a),
            "B_adaptive_quadratic_nurbs": _overlap_summary(overlap_b),
            "C_support_adaptive_local_nurbs": _overlap_summary(overlap_c),
        },
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

    component_records = [
        analyze_chart_atlas(component, support, held_out_target, scale, support.median_spacing)
        for component in coherent_components
    ]
    total_source_nodes = sum(record["component_size"] for record in component_records)
    return {
        "region": region_id,
        "held_out_evidence_size": int(held_out_target.shape[0]),
        "coherent_component_count": len(coherent_components),
        "total_source_nodes": total_source_nodes,
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

    def _aggregate(candidate_key: str) -> dict:
        category_counts = Counter()
        category_evidence = Counter()
        total_held_out = 0
        p95_values: list[float] = []
        capacity_counts = Counter()
        degree_counts = Counter()
        total_charts = 0
        identifiable_charts = 0
        for row in region_rows:
            components = row.get("components", [])
            if not components:
                continue
            held_size = row["held_out_evidence_size"]
            for record in components:
                charts = record["charts"]
                if not charts:
                    continue
                share_per_chart = held_size / len(charts)
                total_held_out += held_size
                for chart_record in charts:
                    total_charts += 1
                    candidate = chart_record[candidate_key]
                    identifiable = candidate.get("identifiability", {})
                    is_identifiable = (
                        identifiable.get("identifiable") if isinstance(identifiable, dict) else None
                    )
                    if is_identifiable:
                        identifiable_charts += 1
                        capacity_counts[
                            f"{candidate.get('control_grid_u', identifiable.get('control_grid_u'))}x"
                            f"{candidate.get('control_grid_v', identifiable.get('control_grid_v'))}"
                        ] += 1
                        degree_counts[candidate.get("degree_u", identifiable.get("degree_u"))] += 1
                    fit = candidate.get("fit", {})
                    category = fit.get("category", CATEGORY_NOT_IDENTIFIABLE)
                    category_counts[category] += 1
                    category_evidence[category] += share_per_chart
                    p95 = fit.get("extrapolation_p95")
                    if p95 is not None:
                        p95_values.append(p95)
        total = total_held_out or 1
        return {
            "total_charts": total_charts, "identifiable_charts": identifiable_charts,
            "capacity_distribution": dict(capacity_counts), "degree_distribution": dict(degree_counts),
            "category_counts": dict(category_counts),
            "evidence_weighted_fractions": {
                key: category_evidence.get(key, 0) / total
                for key in (
                    CATEGORY_NOT_IDENTIFIABLE, CATEGORY_FIT_FAILED, CATEGORY_EXTRAPOLATIVE, CATEGORY_UNSAFE, CATEGORY_VALID,
                )
            },
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out,
        }

    def _overlap_aggregate(candidate_key: str) -> dict:
        pair_counts = 0
        both_fitted = 0
        position_p95s: list[float] = []
        normal_p95s: list[float] = []
        for row in region_rows:
            for record in row.get("components", []):
                summary = record["overlap_consistency"][candidate_key]
                pair_counts += summary["pair_count"]
                both_fitted += summary["both_fitted_pair_count"]
                if summary["mean_position_disagreement_p95"] is not None:
                    position_p95s.append(summary["mean_position_disagreement_p95"])
                if summary["mean_normal_disagreement_degrees_p95"] is not None:
                    normal_p95s.append(summary["mean_normal_disagreement_degrees_p95"])
        return {
            "pair_count": pair_counts, "both_fitted_pair_count": both_fitted,
            "mean_position_disagreement_p95": (sum(position_p95s) / len(position_p95s)) if position_p95s else None,
            "mean_normal_disagreement_degrees_p95": (sum(normal_p95s) / len(normal_p95s)) if normal_p95s else None,
        }

    total_source_nodes = sum(row.get("total_source_nodes", 0) for row in region_rows)
    total_covered = sum(
        record["covered_count"] for row in region_rows for record in row.get("components", [])
    )
    total_unchartable = sum(
        record["unchartable_seed_count"] for row in region_rows for record in row.get("components", [])
    )
    total_uncovered = sum(
        record["uncovered_count"] for row in region_rows for record in row.get("components", [])
    )

    return {
        "checkpoint": str(checkpoint), "cap": cap,
        "regions": region_rows,
        "summary": {
            "total_source_nodes": total_source_nodes,
            "covered_node_total": total_covered, "uncovered_node_total": total_uncovered,
            "unchartable_seed_node_total": total_unchartable,
            "A_fixed_6x6_degree2": _aggregate("A_fixed_6x6_degree2"),
            "B_adaptive_quadratic_nurbs": _aggregate("B_adaptive_quadratic_nurbs"),
            "C_support_adaptive_local_nurbs": _aggregate("C_support_adaptive_local_nurbs"),
            "overlap_consistency": {
                "A_fixed_6x6_degree2": _overlap_aggregate("A_fixed_6x6_degree2"),
                "B_adaptive_quadratic_nurbs": _overlap_aggregate("B_adaptive_quadratic_nurbs"),
                "C_support_adaptive_local_nurbs": _overlap_aggregate("C_support_adaptive_local_nurbs"),
            },
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
        default=Path("output/extent_ab/val102/patch_identifiability_capacity_gate_replay.json"),
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
