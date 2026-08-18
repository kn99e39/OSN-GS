"""Worklog 97 -- curve-network-native NURBS fitting vs. existing PCA-UV
point fitting, paired per Worklog 96 coherent curve-network block.

For every Worklog 96 block (same real 7-region checkpoints as Worklogs 95
and 96), runs BOTH:

A. PCA_UV_POINT_FIT -- the existing Worklog 95/96 fitting path unchanged
   (``fit_torch_visible_surface_lsq``), same curve-network samples.
B. CURVE_NETWORK_NATIVE_FIT -- the SAME samples, but parameterized entirely
   from the curve network's own chord-length-derived, family-reconciled
   ``(u, v)`` (Worklog 97's ``build_curve_network_uv`` +
   ``fit_torch_visible_surface_from_uv``), never PCA.

No fallback between A and B. Identical 6x6/degree-2 NURBS capacity, held-out
evaluation convention (Worklog 87's checkerboard ``_holdout``, unchanged),
and safety classification thresholds (EXTRAPOLATION_BOUND=4.0,
near-degenerate Jacobian, local orientation fold) for both paths -- any
difference is attributable to preserving vs. discarding the curve network's
parametric structure, not to different downstream criteria.

Fixed and unmodified: visible Gaussian training, ADC, region ownership,
Worklog 95/96 latent-surface support/seeding/tracing/continuous-support/
family construction/block decomposition, NURBS degree (2), control grid
(6x6), held-out validation convention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _holdout, _median_nn
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from osn_gs.surface.torch_curve_network_native_fit import (
    DEGREE_U,
    DEGREE_V,
    RESOLUTION_U,
    RESOLUTION_V,
    fit_curve_network_native,
    fit_pca_uv,
)
from osn_gs.surface.torch_latent_surface_curve_families import build_curve_network_blocks
from osn_gs.surface.torch_latent_surface_seed_curves import build_seed_curves
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics

SAMPLE_RESOLUTION = 24
EXTRAPOLATION_BOUND = 4.0


def _pct(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p95": None, "max": None}
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _sample(surface, resolution: int = SAMPLE_RESOLUTION):
    device, dtype = surface.control_grid.device, surface.control_grid.dtype
    g = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(g, g, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    pts, du, dv = surface.evaluate_with_derivatives(uv)
    return pts, du, dv, torch.cross(du, dv, dim=1)


def classify_fitted_surface(surface, curve_points: torch.Tensor, held_out_evidence: torch.Tensor, scale: float) -> dict:
    """Same safety-classification tail as Worklog 87/95's ``evaluate_fit``
    (identical EXTRAPOLATION_BOUND/Jacobian/fold criteria), but accepting
    an ALREADY-FITTED surface -- reused for both the PCA-UV and
    curve-network-native paths so the downstream safety criteria are
    identical for both."""

    pts, du, dv, normals = _sample(surface)
    jac = compute_parametric_jacobian_metrics(du, dv, scale=scale)
    fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)

    def err(subset: torch.Tensor) -> dict:
        if int(subset.shape[0]) == 0:
            return _pct(np.array([]))
        return _pct(torch.cdist(pts, subset).min(dim=1).values.detach().cpu().numpy() / scale)

    held_err = err(held_out_evidence)
    curve_err = err(curve_points)
    p95 = (held_err["p95"] if held_err["p95"] is not None else curve_err["p95"]) or 0.0
    record = {
        "held_out_visible_evidence_error": held_err,
        "surface_to_curve_network_error": curve_err,
        "extrapolation_p95": p95,
        "jacobian_near_degenerate_count": jac["near_degenerate_count"],
        "local_fold_fraction": fold["local_fold_fraction"],
    }
    unsafe = jac["near_degenerate_count"] > 0 or fold["local_fold_fraction"] > 0.01
    if unsafe:
        record["classification"] = "unsafe_geometry"
    elif p95 <= EXTRAPOLATION_BOUND:
        record["classification"] = "valid_supported"
    else:
        record["classification"] = "extrapolative"
    return record


def evaluate_block_pair(block, held_out_evidence: torch.Tensor, scale: float) -> dict:
    record: dict = {"seed_id": block.seed_id, "seed_type": block.seed_type}

    # A. PCA_UV_POINT_FIT
    pca = fit_pca_uv(block)
    if pca.surface is None:
        record["pca_uv"] = {"classification": "fit_failed", "fit_error": pca.fit_error}
    else:
        pca_classification = classify_fitted_surface(pca.surface, block.all_points, held_out_evidence, scale)
        pca_classification["overall_curve_residual_mean"] = pca.overall_residual.mean
        record["pca_uv"] = pca_classification

    # B. CURVE_NETWORK_NATIVE_FIT
    native = fit_curve_network_native(block)
    if not native.valid_parameterization:
        record["curve_network_native"] = {
            "classification": "parameterization_invalid", "invalid_reason": native.invalid_reason,
        }
    elif native.surface is None:
        record["curve_network_native"] = {"classification": "fit_failed"}
    else:
        native_classification = classify_fitted_surface(
            native.surface, native.curve_network_uv.points, held_out_evidence, scale,
        )
        native_classification["overall_curve_residual_mean"] = native.overall_residual.mean
        native_classification["trace_family_residual_mean"] = native.trace_family_residual.mean
        native_classification["rung_family_residual_mean"] = native.rung_family_residual.mean
        record["curve_network_native"] = native_classification

    if record["pca_uv"].get("extrapolation_p95") is not None and record["curve_network_native"].get("extrapolation_p95") is not None:
        record["paired_p95_delta_native_minus_pca"] = (
            record["curve_network_native"]["extrapolation_p95"] - record["pca_uv"]["extrapolation_p95"]
        )
    return record


def analyze_region(
    region_id: int, evidence: torch.Tensor, chart, representative_positions, representative_index,
) -> dict:
    train_evidence, held_evidence = _holdout(evidence)
    if int(train_evidence.shape[0]) < 4:
        return {"region": region_id, "skip_reason": "insufficient_train_split_evidence"}

    support = build_latent_surface_support(train_evidence)
    seeds = build_seed_curves(train_evidence, chart, representative_positions, representative_index, support)
    blocks = build_curve_network_blocks(seeds, support)
    satisfying = [block for block in blocks if block.satisfies_contract]

    held_out_target = held_evidence if int(held_evidence.shape[0]) > 0 else evidence
    scale = _median_nn(held_out_target) if int(held_out_target.shape[0]) >= 2 else _median_nn(evidence)

    block_records = [evaluate_block_pair(block, held_out_target, scale) for block in satisfying]
    parameterization_invalid_count = sum(
        1 for record in block_records if record["curve_network_native"].get("classification") == "parameterization_invalid"
    )
    return {
        "region": region_id,
        "coherent_block_count": len(satisfying),
        "held_out_evidence_size": int(held_out_target.shape[0]),
        "parameterization_invalid_block_count": parameterization_invalid_count,
        "blocks": block_records,
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

    def _aggregate(path_key: str) -> dict:
        classification_evidence = Counter()
        total_held_out_evidence = 0
        blocks_attempted = 0
        blocks_fit_failed = 0
        p95_values: list[float] = []
        for row in region_rows:
            blocks = row.get("blocks", [])
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

    paired_deltas = [
        row_block["paired_p95_delta_native_minus_pca"]
        for row in region_rows for row_block in row.get("blocks", [])
        if "paired_p95_delta_native_minus_pca" in row_block
    ]

    return {
        "checkpoint": str(checkpoint), "cap": cap,
        "regions": region_rows,
        "summary": {
            "pca_uv": _aggregate("pca_uv"),
            "curve_network_native": _aggregate("curve_network_native"),
            "paired_p95_delta_native_minus_pca_mean": (
                sum(paired_deltas) / len(paired_deltas) if paired_deltas else None
            ),
            "paired_block_count": len(paired_deltas),
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
        default=Path("output/extent_ab/val97/curve_network_native_fit_replay.json"),
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
