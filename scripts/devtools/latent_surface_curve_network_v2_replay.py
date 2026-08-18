"""Worklog 96 -- completed latent-surface curve-network constructor replay.

Extends Worklog 95's prototype by removing the legacy
``eligible_parametric_chart_boundary`` entrance gate, adding typed
interior-construction seed fallback, parallel-transport curve tracing,
continuous (not just endpoint) segment-support validation, explicit U/V
curve-family correspondence, and pre-fit multi-block partitioning (one
region may now materialize zero, one, or several independent NURBS
patches instead of exactly one).

Fixed and unmodified: visible Gaussian training, ADC, region ownership,
the existing NURBS fitter (``fit_torch_visible_surface_lsq``), and Worklog
95's own ``evaluate_curve_network_fit`` classification convention
(EXTRAPOLATION_BOUND=4.0, BASE_GRID=6/degree=2) plus Worklog 87's
checkerboard ``_holdout`` split -- reused verbatim for direct
comparability with the Worklog 89/94/95 baselines.

Two evidence conditions, always reported separately:

- ``ALL_VISIBLE_EVIDENCE_CONSTRUCTION``: latent surface + seeds + blocks
  built from the FULL region evidence -- measures what the production
  constructor could actually materialize.
- ``HELD_OUT_VALIDATION``: latent surface + seeds + blocks built ONLY from
  the checkerboard train half; the held half never influences
  construction, used only to validate the fitted patches. Worklog 95's
  support thresholds are not loosened for this condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _holdout
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from latent_surface_curve_network_prototype_replay import evaluate_curve_network_fit
from osn_gs.surface.torch_latent_surface_curve_families import build_curve_network_blocks
from osn_gs.surface.torch_latent_surface_seed_curves import (
    SEED_CREASE_FEATURE,
    SEED_INTERIOR_CONSTRUCTION,
    SEED_OBSERVATION_FRONTIER,
    SEED_PHYSICAL_BOUNDARY,
    build_seed_curves,
)
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support

SEED_TYPES = (SEED_PHYSICAL_BOUNDARY, SEED_CREASE_FEATURE, SEED_OBSERVATION_FRONTIER, SEED_INTERIOR_CONSTRUCTION)
_ELIGIBLE_CHART_STATUS = "eligible_parametric_chart_boundary"


def _segment_support_stats(blocks) -> tuple[int, int]:
    """(attempted, accepted) rung segment counts recomputed from the
    already-public block fields -- attempted is the common-depth span
    between each pair of adjacent transversal traces; accepted is how many
    of those actually produced a continuously-supported rung."""

    attempted = 0
    accepted = 0
    for block in blocks:
        traces = block.transversal_traces
        for i in range(len(traces) - 1):
            attempted += min(int(traces[i].points.shape[0]), int(traces[i + 1].points.shape[0]))
        accepted += len(block.rungs)
    return attempted, accepted


def _construct_condition(
    evidence: torch.Tensor, chart, representative_positions, representative_index,
):
    """Shared construction step for both A and B: build support, seeds,
    blocks from whatever evidence subset the caller passes in."""

    support = build_latent_surface_support(evidence)
    seeds = build_seed_curves(evidence, chart, representative_positions, representative_index, support)
    blocks = build_curve_network_blocks(seeds, support)
    return support, seeds, blocks


def analyze_region_construction(
    region_id: int, evidence: torch.Tensor, chart, representative_positions, representative_index,
) -> dict:
    """Condition A: ALL_VISIBLE_EVIDENCE_CONSTRUCTION."""

    support, seeds, blocks = _construct_condition(evidence, chart, representative_positions, representative_index)
    support_check = support.query_batch(evidence)
    unsupported_fraction = float((~support_check.supported).float().mean().item())
    attempted, accepted = _segment_support_stats(blocks)
    unsupported_segment_fraction = 1.0 - (accepted / attempted) if attempted else None

    satisfying = [block for block in blocks if block.satisfies_contract]
    patch_records = []
    materialized = 0
    for block in satisfying:
        fit = evaluate_curve_network_fit(block.all_points, evidence, f"region{region_id}_{block.seed_id}")
        fit_class = fit.get("classification", "unresolved")
        if fit_class not in ("fit_failed",):
            materialized += 1
        patch_records.append({
            "seed_id": block.seed_id, "seed_type": block.seed_type,
            "curve_point_count": int(block.all_points.shape[0]),
            "fit_classification": fit_class,
        })

    seed_type_counts = Counter(seed.seed_type for seed in seeds)
    return {
        "region": region_id,
        "seed_count": len(seeds),
        "seed_type_counts": dict(seed_type_counts),
        "block_count": len(blocks),
        "satisfying_block_count": len(satisfying),
        "materialized_patch_count": materialized,
        "unsupported_latent_surface_fraction": unsupported_fraction,
        "unsupported_curve_segment_fraction": unsupported_segment_fraction,
        "patches": patch_records,
        "chart_was_eligible": bool(chart is not None and chart.status == _ELIGIBLE_CHART_STATUS and len(chart.ordered_node_ids) >= 3),
    }


def analyze_region_held_out(
    region_id: int, evidence: torch.Tensor, chart, representative_positions, representative_index,
) -> dict:
    """Condition B: HELD_OUT_VALIDATION."""

    train_evidence, held_evidence = _holdout(evidence)
    if int(train_evidence.shape[0]) < 4:
        return {"region": region_id, "skip_reason": "insufficient_train_split_evidence"}

    support, seeds, blocks = _construct_condition(train_evidence, chart, representative_positions, representative_index)
    support_check = support.query_batch(evidence)
    unsupported_fraction = float((~support_check.supported).float().mean().item())
    attempted, accepted = _segment_support_stats(blocks)
    unsupported_segment_fraction = 1.0 - (accepted / attempted) if attempted else None

    satisfying = [block for block in blocks if block.satisfies_contract]
    held_out_target = held_evidence if int(held_evidence.shape[0]) > 0 else evidence
    held_out_size = int(held_out_target.shape[0])
    classification = Counter()
    classification_evidence = Counter()  # evidence-weighted, same convention as Worklog 87/89/94/95
    p95_values: list[float] = []
    patch_records = []
    # Evidence not covered by ANY satisfying block is unresolved (mirrors
    # the Worklog 87/94/95 convention: a region/patch that never
    # materializes contributes its full evidence share to "unresolved").
    if not satisfying:
        classification_evidence["unresolved"] += held_out_size
    else:
        share = held_out_size / len(satisfying)
        for block in satisfying:
            fit = evaluate_curve_network_fit(block.all_points, held_out_target, f"region{region_id}_{block.seed_id}")
            fit_class = fit.get("classification", "unresolved")
            classification[fit_class] += 1
            classification_evidence[fit_class] += share
            p95 = fit.get("extrapolation_p95")
            if p95 is not None:
                p95_values.append(float(p95))
            patch_records.append({
                "seed_id": block.seed_id, "seed_type": block.seed_type,
                "curve_point_count": int(block.all_points.shape[0]),
                "fit_classification": fit_class, "p95": p95,
            })

    seed_type_counts = Counter(seed.seed_type for seed in seeds)
    return {
        "region": region_id,
        "seed_count": len(seeds),
        "seed_type_counts": dict(seed_type_counts),
        "block_count": len(blocks),
        "satisfying_block_count": len(satisfying),
        "held_out_evidence_size": held_out_size,
        "patch_classification_counts": dict(classification),
        "classification_evidence": dict(classification_evidence),
        "held_out_p95_mean": (sum(p95_values) / len(p95_values)) if p95_values else None,
        "unsupported_latent_surface_fraction": unsupported_fraction,
        "unsupported_curve_segment_fraction": unsupported_segment_fraction,
        "patches": patch_records,
        "chart_was_eligible": bool(chart is not None and chart.status == _ELIGIBLE_CHART_STATUS and len(chart.ordered_node_ids) >= 3),
    }


def analyze(checkpoint: Path, cap: int, device: str) -> dict:
    model, stable_ids = _load_model(checkpoint, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    construction_rows = []
    held_out_rows = []
    total_evidence = 0

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        region_size = len(full_indices)
        total_evidence += region_size
        if region_size < 4:
            construction_rows.append({"region": region_id, "skip_reason": "insufficient_owned_evidence"})
            held_out_rows.append({"region": region_id, "skip_reason": "insufficient_owned_evidence"})
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        chart = chart_by_region.get(region_id)

        construction_rows.append(
            analyze_region_construction(region_id, evidence, chart, representative_positions, representative_index)
        )
        held_out_rows.append(
            analyze_region_held_out(region_id, evidence, chart, representative_positions, representative_index)
        )

    def _aggregate_construction(rows: list[dict]) -> dict:
        usable_rows = [row for row in rows if "seed_count" in row]
        regions_with_any_seed = sum(1 for row in usable_rows if row["seed_count"] > 0)
        regions_with_valid_network = sum(1 for row in usable_rows if row["satisfying_block_count"] > 0)
        total_blocks = sum(row["satisfying_block_count"] for row in usable_rows)
        total_patches = sum(row["materialized_patch_count"] for row in usable_rows)
        seed_type_total = Counter()
        for row in usable_rows:
            seed_type_total.update(row["seed_type_counts"])
        previously_blocked_now_construct = sum(
            1 for row in usable_rows
            if not row["chart_was_eligible"] and row["satisfying_block_count"] > 0
        )
        multi_patch_regions = sum(1 for row in usable_rows if row["satisfying_block_count"] > 1)
        unsupported_latent = [row["unsupported_latent_surface_fraction"] for row in usable_rows]
        unsupported_segment = [row["unsupported_curve_segment_fraction"] for row in usable_rows if row["unsupported_curve_segment_fraction"] is not None]
        return {
            "regions_with_any_usable_seed": regions_with_any_seed,
            "seed_type_breakdown": dict(seed_type_total),
            "regions_with_valid_supported_curve_network": regions_with_valid_network,
            "coherent_curve_network_block_count": total_blocks,
            "materialized_nurbs_patch_count": total_patches,
            "regions_previously_blocked_by_legacy_boundary_gate_now_constructing": previously_blocked_now_construct,
            "multi_patch_region_count": multi_patch_regions,
            "unsupported_latent_surface_fraction_mean": (
                sum(unsupported_latent) / len(unsupported_latent) if unsupported_latent else None
            ),
            "unsupported_curve_segment_fraction_mean": (
                sum(unsupported_segment) / len(unsupported_segment) if unsupported_segment else None
            ),
        }

    def _aggregate_held_out(rows: list[dict]) -> dict:
        usable_rows = [row for row in rows if "seed_count" in row]
        patch_classification = Counter()
        evidence_classification = Counter()
        total_held_evidence = 0
        p95_values = []
        total_satisfying = sum(row["satisfying_block_count"] for row in usable_rows)
        regions_with_valid_network = sum(1 for row in usable_rows if row["satisfying_block_count"] > 0)
        for row in usable_rows:
            patch_classification.update(row.get("patch_classification_counts", {}))
            evidence_classification.update(row.get("classification_evidence", {}))
            total_held_evidence += row.get("held_out_evidence_size", 0)
            if row.get("held_out_p95_mean") is not None:
                p95_values.append(row["held_out_p95_mean"])
        total_patches = sum(patch_classification.values()) or 1
        total_evidence = total_held_evidence or 1
        return {
            "regions_with_valid_supported_curve_network": regions_with_valid_network,
            "coherent_curve_network_block_count": total_satisfying,
            "patch_count_fractions": {
                key: patch_classification.get(key, 0) / total_patches
                for key in ("valid_supported", "extrapolative", "unsafe_geometry", "unresolved", "fit_failed")
            },
            "evidence_weighted_fractions": {
                key: evidence_classification.get(key, 0) / total_evidence
                for key in ("valid_supported", "extrapolative", "unsafe_geometry", "unresolved", "fit_failed")
            },
            "total_held_out_evidence": total_held_evidence,
            "held_out_p95_mean_of_regions": (sum(p95_values) / len(p95_values)) if p95_values else None,
        }

    return {
        "checkpoint": str(checkpoint), "cap": cap, "total_evidence": total_evidence,
        "construction": {
            "summary": _aggregate_construction(construction_rows),
            "regions": construction_rows,
        },
        "held_out_validation": {
            "summary": _aggregate_held_out(held_out_rows),
            "regions": held_out_rows,
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
        default=Path("output/extent_ab/val96/latent_surface_curve_network_v2_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = analyze(args.checkpoint, args.cap, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "construction_summary": report["construction"]["summary"],
        "held_out_validation_summary": report["held_out_validation"]["summary"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
