"""Worklog 99 -- bounded parametric patch architecture gate.

For every Worklog 98 coherent, tangent-frame-synchronized component, first
characterizes the tree-integrated ``(u, v)`` parameter domain itself
(pre-fit, before any candidate is attempted) -- orientation coherence
(Worklog 98) does not by itself prove a well-conditioned parameter domain.
Only pre-fit domain-VALID components are handed to three fixed,
independently-evaluated (no fallback between them) parametric-patch
candidates on IDENTICAL input evidence:

A. FIXED_6x6_LSQ -- Worklog 98's own path, unchanged
   (``fit_curve_lattice_native``).
B. ADAPTIVE_REGULARIZED_NURBS -- same external-UV LSQ fitter
   (``fit_torch_visible_surface_from_uv``, unchanged, already regularized
   by its existing second-difference/Tikhonov terms), but with a
   deterministic, structural (never fit/held-out-error-derived) control-
   grid capacity.
C. GORDON_CURVE_NETWORK -- treats the component's U/V curve families as
   primary constraints via a Gordon-style transfinite-interpolation
   construction, never collapsing them into an unordered point-fit
   problem first.

Fixed and unmodified: visible Gaussian training, ADC, region ownership,
Worklog 95 latent-surface estimator, continuous support contract, Worklog
98 synchronized tangent-frame field/coherent components, held-out
evaluation, and the existing extrapolative/unsafe/valid_supported
definitions. No candidate is tuned using held-out results.
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
from curve_network_native_fit_replay import SAMPLE_RESOLUTION, classify_fitted_surface
from osn_gs.surface.torch_adaptive_nurbs_capacity import select_adaptive_control_grid_capacity
from osn_gs.surface.torch_curve_lattice_native_fit import fit_curve_lattice_native
from osn_gs.surface.torch_gordon_curve_network_surface import construct_gordon_surface
from osn_gs.surface.torch_latent_surface_curve_lattice import build_curve_lattice
from osn_gs.surface.torch_latent_surface_seed_curves import (
    SEED_INTERIOR_CONSTRUCTION,
    build_seed_curves,
)
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv
from osn_gs.surface.torch_parametric_domain_validity import (
    assess_parametric_domain_validity,
    cycle_position_drift_p95,
)

CATEGORY_DOMAIN_INVALID = "PARAMETER_DOMAIN_INVALID"
CATEGORY_FIT_FAILED = "FIT_FAILED"
CATEGORY_EXTRAPOLATIVE = "FIT_SUCCEEDED_BUT_EXTRAPOLATIVE"
CATEGORY_UNSAFE = "FIT_SUCCEEDED_BUT_UNSAFE"
CATEGORY_VALID = "VALID_SUPPORTED"


def _pick_field_anchor(seeds) -> tuple[object, object] | tuple[None, None]:
    for seed in seeds:
        if seed.seed_type != SEED_INTERIOR_CONSTRUCTION and int(seed.points.shape[0]) >= 2:
            anchor_position = seed.points[0]
            anchor_hint = seed.points[1] - seed.points[0]
            if float(anchor_hint.norm().item()) > 1e-9:
                return anchor_position, anchor_hint
    return None, None


def _bending_roughness(control_grid: torch.Tensor) -> float:
    """Discrete bending-energy-style roughness: sum of squared second
    differences of the control net along both axes -- a pure post-hoc
    diagnostic (reported, never used to select capacity or reject a fit)."""

    if control_grid.shape[0] < 3 or control_grid.shape[1] < 3:
        return 0.0
    second_u = control_grid[2:, :, :] - 2 * control_grid[1:-1, :, :] + control_grid[:-2, :, :]
    second_v = control_grid[:, 2:, :] - 2 * control_grid[:, 1:-1, :] + control_grid[:, :-2, :]
    return float((second_u.square().sum() + second_v.square().sum()).item())


def _classify(surface_or_none, curve_points, held_out_target, scale, fit_error: str | None) -> dict:
    if surface_or_none is None:
        return {"category": CATEGORY_FIT_FAILED, "fit_error": fit_error}
    record = classify_fitted_surface(surface_or_none, curve_points, held_out_target, scale)
    classification = record.get("classification")
    category = {
        "unsafe_geometry": CATEGORY_UNSAFE, "extrapolative": CATEGORY_EXTRAPOLATIVE,
        "valid_supported": CATEGORY_VALID,
    }.get(classification, CATEGORY_FIT_FAILED)
    record["category"] = category
    record["control_net_roughness"] = _bending_roughness(surface_or_none.control_grid)
    return record


def analyze_component(component, support, held_out_target: torch.Tensor, scale: float) -> dict:
    lattice = build_curve_lattice(component, support)
    record: dict = {"component_size": len(component.node_indices), "coherent": component.coherent}
    if not lattice.valid:
        record["domain_valid"] = False
        record["domain_invalid_reason"] = lattice.invalid_reason
        record["category"] = CATEGORY_DOMAIN_INVALID
        return record

    domain_report = assess_parametric_domain_validity(
        lattice.points, lattice.uv, component.normals, support.median_spacing,
    )
    cycle_drift = cycle_position_drift_p95(component, support.median_spacing)
    record["domain_report"] = {
        "valid": domain_report.valid, "invalid_reasons": list(domain_report.invalid_reasons),
        "u_extent": domain_report.u_extent, "v_extent": domain_report.v_extent,
        "duplicate_incompatible_count": domain_report.duplicate_incompatible_count,
        "fold_fraction": domain_report.fold_fraction, "singular_fraction": domain_report.singular_fraction,
        "mean_condition_number": domain_report.mean_condition_number,
        "max_condition_number": domain_report.max_condition_number,
        "stretch_ratio_p95": domain_report.stretch_ratio_p95,
        "cycle_position_drift_p95_over_spacing": cycle_drift,
    }
    record["domain_valid"] = domain_report.valid
    if not domain_report.valid:
        record["category"] = CATEGORY_DOMAIN_INVALID
        return record

    u_curve_count = len(lattice.u_curves)
    v_curve_count = len(lattice.v_curves)

    # A. FIXED_6x6_LSQ (Worklog 98's own unchanged path).
    native = fit_curve_lattice_native(lattice)
    record["A_fixed_6x6_lsq"] = _classify(
        native.surface, lattice.points, held_out_target, scale,
        native.invalid_reason if not native.valid_lattice else None,
    )
    if native.valid_lattice:
        record["A_fixed_6x6_lsq"]["u_curve_residual_mean"] = native.u_curve_residual.mean
        record["A_fixed_6x6_lsq"]["v_curve_residual_mean"] = native.v_curve_residual.mean
        record["A_fixed_6x6_lsq"]["overall_curve_residual_mean"] = native.overall_residual.mean

    # B. ADAPTIVE_REGULARIZED_NURBS -- capacity fixed structurally BEFORE fitting.
    capacity = select_adaptive_control_grid_capacity(
        u_curve_count, v_curve_count, len(component.node_indices),
        domain_report.u_extent, domain_report.v_extent,
    )
    try:
        b_surface = fit_torch_visible_surface_from_uv(
            lattice.points, lattice.uv, resolution_u=capacity.resolution_u, resolution_v=capacity.resolution_v,
            degree_u=2, degree_v=2,
        )
        b_error = None
    except Exception as exc:  # noqa: BLE001
        b_surface, b_error = None, f"{type(exc).__name__}: {exc}"
    record["B_adaptive_regularized_nurbs"] = _classify(b_surface, lattice.points, held_out_target, scale, b_error)
    record["B_adaptive_regularized_nurbs"]["resolution_u"] = capacity.resolution_u
    record["B_adaptive_regularized_nurbs"]["resolution_v"] = capacity.resolution_v

    # C. GORDON_CURVE_NETWORK.
    gordon = construct_gordon_surface(
        lattice.points, lattice.uv, resolution_u=capacity.resolution_u, resolution_v=capacity.resolution_v,
        u_curve_count=u_curve_count, v_curve_count=v_curve_count,
    )
    record["C_gordon_curve_network"] = _classify(
        gordon.surface, lattice.points, held_out_target, scale,
        gordon.invalid_reason if not gordon.valid else None,
    )
    record["C_gordon_curve_network"]["u_level_count"] = gordon.u_level_count
    record["C_gordon_curve_network"]["v_level_count"] = gordon.v_level_count
    record["C_gordon_curve_network"]["intersection_grid_residual"] = gordon.intersection_grid_residual
    if gordon.valid:
        record["C_gordon_curve_network"]["resolution_u"] = capacity.resolution_u
        record["C_gordon_curve_network"]["resolution_v"] = capacity.resolution_v

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

    def _aggregate(candidate_key: str | None) -> dict:
        category_evidence = Counter()
        total_held_out = 0
        p95_values: list[float] = []
        for row in region_rows:
            components = row.get("components", [])
            if not components:
                continue
            held_size = row["held_out_evidence_size"]
            share = held_size / len(components)
            total_held_out += held_size
            for record in components:
                if record.get("domain_valid") is not True:
                    category_evidence[CATEGORY_DOMAIN_INVALID] += share
                    continue
                candidate_record = record.get(candidate_key, {})
                category = candidate_record.get("category", CATEGORY_FIT_FAILED)
                category_evidence[category] += share
                p95 = candidate_record.get("extrapolation_p95")
                if p95 is not None:
                    p95_values.append(p95)
        total = total_held_out or 1
        return {
            "evidence_weighted_fractions": {
                key: category_evidence.get(key, 0) / total
                for key in (CATEGORY_DOMAIN_INVALID, CATEGORY_FIT_FAILED, CATEGORY_EXTRAPOLATIVE, CATEGORY_UNSAFE, CATEGORY_VALID)
            },
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out,
        }

    total_coherent = sum(row.get("coherent_component_count", 0) for row in region_rows)
    domain_valid = sum(
        1 for row in region_rows for record in row.get("components", [])
        if record.get("domain_valid") is True
    )

    return {
        "checkpoint": str(checkpoint), "cap": cap,
        "regions": region_rows,
        "summary": {
            "total_coherent_components": total_coherent,
            "domain_valid_components": domain_valid,
            "domain_invalid_components": total_coherent - domain_valid,
            "A_fixed_6x6_lsq": _aggregate("A_fixed_6x6_lsq"),
            "B_adaptive_regularized_nurbs": _aggregate("B_adaptive_regularized_nurbs"),
            "C_gordon_curve_network": _aggregate("C_gordon_curve_network"),
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
        default=Path("output/extent_ab/val99/parametric_patch_architecture_gate_replay.json"),
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
