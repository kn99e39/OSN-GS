"""Worklog 98 -- globally synchronized tangent-frame curve lattice replay.

Paired three-way architecture comparison on identical latent-surface
evidence per region:

A. Worklog 96 curve network + PCA-UV point fitting (unchanged).
B. Worklog 96 curve network (independent per-seed direction) + Worklog 97
   network-native fitting (unchanged).
C. NEW: one synchronized tangent-frame field per region (replacing
   Worklog 96's independent per-seed transversal direction selection) +
   curve lattice + Worklog 97 network-native fitting (same
   ``fit_torch_visible_surface_from_uv``, unchanged).

No fallback between any of the three paths. Identical NURBS capacity
(6x6/degree-2), identical held-out convention (Worklog 87 checkerboard
``_holdout``), identical safety classification thresholds for all three.

Fixed and unmodified: visible Gaussian training, ADC, region ownership,
Worklog 95 latent-surface estimator, supported-query semantics, continuous
segment-support requirements, existing seed provenance, NURBS degree (2),
6x6 control grid, Worklog 97 external-UV/native fitting, held-out/safety
criteria.
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
from curve_network_native_fit_replay import classify_fitted_surface, evaluate_block_pair
from osn_gs.surface.torch_curve_lattice_native_fit import fit_curve_lattice_native
from osn_gs.surface.torch_latent_surface_curve_families import build_curve_network_blocks
from osn_gs.surface.torch_latent_surface_curve_lattice import build_curve_lattice
from osn_gs.surface.torch_latent_surface_seed_curves import (
    SEED_INTERIOR_CONSTRUCTION,
    build_seed_curves,
)
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field


def _pick_field_anchor(seeds) -> tuple[object, object] | tuple[None, None]:
    """A representative typed (non-interior) seed's own start point and
    tangent, if any survived -- used only to optionally anchor path C's
    gauge to an observed boundary/feature direction (Worklog 98 section 3).
    Never required; an all-interior seed set leaves the field ungauged."""

    for seed in seeds:
        if seed.seed_type != SEED_INTERIOR_CONSTRUCTION and int(seed.points.shape[0]) >= 2:
            anchor_position = seed.points[0]
            anchor_hint = seed.points[1] - seed.points[0]
            if float(anchor_hint.norm().item()) > 1e-9:
                return anchor_position, anchor_hint
    return None, None


def analyze_region(
    region_id: int, evidence: torch.Tensor, chart, representative_positions, representative_index,
) -> dict:
    train_evidence, held_evidence = _holdout(evidence)
    if int(train_evidence.shape[0]) < 4:
        return {"region": region_id, "skip_reason": "insufficient_train_split_evidence"}

    support = build_latent_surface_support(train_evidence)
    held_out_target = held_evidence if int(held_evidence.shape[0]) > 0 else evidence
    scale = _median_nn(held_out_target) if int(held_out_target.shape[0]) >= 2 else _median_nn(evidence)

    # A/B: unchanged Worklog 96 per-seed blocks.
    seeds = build_seed_curves(train_evidence, chart, representative_positions, representative_index, support)
    blocks = build_curve_network_blocks(seeds, support)
    satisfying_blocks = [block for block in blocks if block.satisfies_contract]
    ab_records = [evaluate_block_pair(block, held_out_target, scale) for block in satisfying_blocks]

    # C: one synchronized field over the SAME train evidence, replacing
    # independent per-seed direction selection.
    anchor_position, anchor_hint = _pick_field_anchor(seeds)
    field_result = build_tangent_frame_field(
        train_evidence, support, anchor_position=anchor_position, anchor_hint_direction=anchor_hint,
        anchor_seed_type="anchored" if anchor_position is not None else None,
    )
    coherent_components = [component for component in field_result.components if component.coherent]
    incoherent_components = [component for component in field_result.components if not component.coherent]

    c_records = []
    for component in field_result.components:
        lattice = build_curve_lattice(component, support)
        if not lattice.valid:
            c_records.append({
                "component_size": len(component.node_indices), "coherent": component.coherent,
                "curve_network_native": {"classification": "parameterization_invalid", "invalid_reason": lattice.invalid_reason},
            })
            continue
        fit = fit_curve_lattice_native(lattice)
        if fit.surface is None:
            c_records.append({
                "component_size": len(component.node_indices), "coherent": component.coherent,
                "curve_network_native": {"classification": "fit_failed"},
            })
            continue
        classification = classify_fitted_surface(fit.surface, lattice.points, held_out_target, scale)
        classification["overall_curve_residual_mean"] = fit.overall_residual.mean
        classification["u_curve_residual_mean"] = fit.u_curve_residual.mean
        classification["v_curve_residual_mean"] = fit.v_curve_residual.mean
        c_records.append({
            "component_size": len(component.node_indices), "coherent": component.coherent,
            "curve_network_native": classification,
        })

    return {
        "region": region_id,
        "held_out_evidence_size": int(held_out_target.shape[0]),
        "ab_blocks": ab_records,
        "field_component_count": len(field_result.components),
        "coherent_field_component_count": len(coherent_components),
        "incoherent_field_component_count": len(incoherent_components),
        "field_unsupported_edge_count": field_result.unsupported_edge_count,
        "field_total_candidate_edges": field_result.total_candidate_edges,
        "c_components": c_records,
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

    def _aggregate_ab(path_key: str) -> dict:
        classification_evidence = Counter()
        total_held_out_evidence = 0
        blocks_attempted = 0
        blocks_fit_failed = 0
        p95_values: list[float] = []
        for row in region_rows:
            blocks = row.get("ab_blocks", [])
            if not blocks:
                continue
            held_size = row["held_out_evidence_size"]
            share = held_size / len(blocks)
            total_held_out_evidence += held_size
            for record in blocks:
                blocks_attempted += 1
                classification = record[path_key].get("classification", "unresolved")
                if classification in ("fit_failed", "parameterization_invalid"):
                    blocks_fit_failed += 1
                classification_evidence[classification] += share
                p95 = record[path_key].get("extrapolation_p95")
                if p95 is not None:
                    p95_values.append(p95)
        total = total_held_out_evidence or 1
        return {
            "blocks_attempted": blocks_attempted,
            "blocks_fit_failed_or_invalid": blocks_fit_failed,
            "evidence_weighted_fractions": {
                key: classification_evidence.get(key, 0) / total
                for key in ("valid_supported", "extrapolative", "unsafe_geometry", "parameterization_invalid", "fit_failed")
            },
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out_evidence,
        }

    def _aggregate_c() -> dict:
        classification_evidence = Counter()
        total_held_out_evidence = 0
        components_attempted = 0
        components_incoherent = 0
        components_fit_failed = 0
        p95_values: list[float] = []
        for row in region_rows:
            components = row.get("c_components", [])
            if not components:
                continue
            held_size = row["held_out_evidence_size"]
            share = held_size / len(components)
            total_held_out_evidence += held_size
            for record in components:
                components_attempted += 1
                if not record["coherent"]:
                    components_incoherent += 1
                classification = record["curve_network_native"].get("classification", "unresolved")
                if classification in ("fit_failed", "parameterization_invalid"):
                    components_fit_failed += 1
                classification_evidence[classification] += share
                p95 = record["curve_network_native"].get("extrapolation_p95")
                if p95 is not None:
                    p95_values.append(p95)
        total = total_held_out_evidence or 1
        return {
            "components_attempted": components_attempted,
            "components_incoherent": components_incoherent,
            "components_fit_failed_or_invalid": components_fit_failed,
            "evidence_weighted_fractions": {
                key: classification_evidence.get(key, 0) / total
                for key in ("valid_supported", "extrapolative", "unsafe_geometry", "parameterization_invalid", "fit_failed")
            },
            "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
            "total_held_out_evidence": total_held_out_evidence,
        }

    return {
        "checkpoint": str(checkpoint), "cap": cap,
        "regions": region_rows,
        "summary": {
            "pca_uv": _aggregate_ab("pca_uv"),
            "curve_network_native_independent": _aggregate_ab("curve_network_native"),
            "synchronized_frame_native": _aggregate_c(),
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
        default=Path("output/extent_ab/val98/synchronized_frame_lattice_replay.json"),
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
