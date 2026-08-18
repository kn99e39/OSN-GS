"""Worklog 95 -- end-to-end latent-surface curve-network constructor prototype.

Region-owned visible Gaussians -> latent surface support -> structural curve
network -> NURBS patch -> held-out visible-evidence validation.

This removes the Worklog 82/83/89 raw Gaussian-center connectivity/adjacency
dependency entirely for chart construction: no same-surface kNN graph, no
chart-unit assembly, no full-region face-incidence topology. The only inputs
are (1) each region's own owned Gaussian centers, used solely to build a
:class:`LatentSurfaceSupport` MLS estimator, and (2) the ALREADY EXISTING,
UNMODIFIED sparse parametric chart boundary (Worklog 79/80's
``construct_region_parametric_chart_boundaries``, produced by the fixed
canonical-construction pipeline) as curve seeds.

Fixed and unmodified: visible Gaussian training, ADC, region ownership/
formation, and the existing NURBS fitter (``fit_torch_visible_surface_lsq``).
No new threshold is tuned toward a favorable outcome; every constant reused
here is either an existing worklog default (BASE_GRID=6, degree=2,
EXTRAPOLATION_BOUND=4.0 -- identical to ``evaluate_fit`` in Worklog 87's own
module, reused unmodified for direct comparability) or a documented fixed
convention from the new curve-network/support modules.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from chart_unit_general_partition_seam_replay import _holdout
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from osn_gs.surface.torch_latent_surface_curve_network import (
    STATUS_CURVE_NETWORK,
    STATUS_NO_ELIGIBLE_SEED_CHART,
    build_latent_surface_curve_network,
)
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_single_chart_uv_validity import neighborhood_preservation, uv_duplicate_diagnostics

# Identical to Worklog 87's evaluate_fit convention -- reused, not retuned.
BASE_GRID = 6
SAMPLE_RESOLUTION = 24
EXTRAPOLATION_BOUND = 4.0


def _median_nn(points: torch.Tensor) -> float:
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    v = float(d.min(dim=1).values.median())
    return v if v > 0 else 1e-6


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


def evaluate_curve_network_fit(curve_points: torch.Tensor, held_out_evidence: torch.Tensor, label: str) -> dict:
    """Fit the existing NURBS implementation to curve-network samples
    (never raw region evidence, never every Gaussian center) and validate
    against held-out visible evidence that was not used to build the
    supporting curves -- same classification convention as Worklog 87's
    own ``evaluate_fit`` (identical EXTRAPOLATION_BOUND/BASE_GRID/degree)
    for direct comparability with the Worklog 89/94 baseline.
    """

    scale = _median_nn(held_out_evidence) if int(held_out_evidence.shape[0]) >= 2 else _median_nn(curve_points)
    record: dict = {"label": label, "curve_network_point_count": int(curve_points.shape[0])}

    if int(curve_points.shape[0]) >= 4:
        uv = pca_parameterize_points(curve_points)
        dup = uv_duplicate_diagnostics(uv)
        nb = neighborhood_preservation(curve_points, uv, k=8)
        record["uv_validity"] = {
            "uv_near_collision_count": dup["uv_near_collision_count"],
            "neighborhood_preservation_mean": nb["neighborhood_preservation_mean"],
        }

    try:
        surface, _ = fit_torch_visible_surface_lsq(
            curve_points, resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2,
        )
    except Exception as exc:  # noqa: BLE001
        record["classification"] = "fit_failed"
        record["fit_error"] = f"{type(exc).__name__}: {exc}"
        return record

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
    record.update({
        "held_out_visible_evidence_error": held_err,
        "curve_network_to_surface_error": curve_err,
        "extrapolation_p95": p95,
        "jacobian_near_degenerate_count": jac["near_degenerate_count"],
        "local_fold_fraction": fold["local_fold_fraction"],
    })
    unsafe = jac["near_degenerate_count"] > 0 or fold["local_fold_fraction"] > 0.01
    if unsafe:
        record["classification"] = "unsafe_geometry"
    elif p95 <= EXTRAPOLATION_BOUND:
        record["classification"] = "valid_supported"
    else:
        record["classification"] = "extrapolative"
    return record


def analyze(checkpoint: Path, cap: int, device: str) -> dict:
    model, stable_ids = _load_model(checkpoint, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    rows = []
    counts = Counter()
    classification = Counter()
    total_evidence = 0
    unsupported_evidence_fraction_values: list[tuple[float, int]] = []
    held_out_p95_values: list[tuple[float, int]] = []

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        region_size = len(full_indices)
        row: dict = {"region": region_id, "total_evidence": region_size}
        total_evidence += region_size
        if region_size < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue

        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        chart = chart_by_region.get(region_id)

        counts["regions_with_evidence"] += 1
        if chart is not None and chart.status == "eligible_parametric_chart_boundary" and len(chart.ordered_node_ids) >= 3:
            counts["regions_with_usable_seed_curves"] += 1

        # Strict train/held-out split (Worklog 87's own PCA-UV checkerboard
        # convention, unmodified): the latent surface estimator and the
        # entire curve network are built ONLY from the train half. The held
        # half never influences support construction, seed densification,
        # or transversal tracing -- only used afterward to validate the
        # fitted patch, matching the directive's "not directly used to
        # construct the supporting curves" requirement as strictly as
        # Worklog 87's own held-out convention already does.
        train_evidence, held_evidence = _holdout(evidence)
        if int(train_evidence.shape[0]) < 4:
            row["skip_reason"] = "insufficient_train_split_evidence"
            rows.append(row)
            continue

        support = build_latent_surface_support(train_evidence)
        support_check = support.query_batch(evidence)
        unsupported_fraction = float((~support_check.supported).float().mean().item())
        row["region_evidence_unsupported_by_own_latent_surface_fraction"] = unsupported_fraction
        unsupported_evidence_fraction_values.append((unsupported_fraction, region_size))

        network = build_latent_surface_curve_network(region_id, chart, representative_positions, representative_index, support)
        row["curve_network_status"] = network.status
        row["seed_segment_count"] = len(network.seed_segments)
        row["transversal_curve_count"] = len(network.transversal_curves)
        row["rung_curve_count"] = len(network.rung_curves)

        if not network.has_curve_network:
            classification["unresolved"] += region_size
            rows.append(row)
            continue

        counts["regions_producing_curve_network"] += 1
        held_out_target = held_evidence if int(held_evidence.shape[0]) > 0 else evidence
        fit = evaluate_curve_network_fit(network.all_points, held_out_target, f"region{region_id}")
        row["fit"] = fit
        fit_class = fit.get("classification", "unresolved")
        classification[fit_class] += region_size
        if fit_class not in ("fit_failed", "unresolved"):
            counts["materialized_patches"] += 1
            p95 = fit.get("extrapolation_p95")
            if p95 is not None:
                held_out_p95_values.append((float(p95), region_size))
        rows.append(row)

    total = total_evidence or 1
    unresolved_evidence = classification["unresolved"] + classification["fit_failed"]

    def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
        if not values:
            return None
        weight_sum = sum(weight for _v, weight in values)
        return sum(v * weight for v, weight in values) / weight_sum if weight_sum else None

    summary = {
        "total_evidence": total_evidence,
        "region_count": len(rows),
        "regions_with_usable_seed_curves": counts["regions_with_usable_seed_curves"],
        "regions_producing_curve_network": counts["regions_producing_curve_network"],
        "materialized_nurbs_patches": counts["materialized_patches"],
        "evidence_fractions": {
            "valid_supported": classification["valid_supported"] / total,
            "extrapolative": classification["extrapolative"] / total,
            "unsafe_geometry": classification["unsafe_geometry"] / total,
            "unresolved": unresolved_evidence / total,
        },
        "held_out_visible_evidence_p95_weighted": _weighted_mean(held_out_p95_values),
        "region_evidence_unsupported_by_own_latent_surface_fraction_weighted": _weighted_mean(
            unsupported_evidence_fraction_values
        ),
    }
    return {"checkpoint": str(checkpoint), "cap": cap, "summary": summary, "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/extent_ab/val95/latent_surface_curve_network_prototype_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = analyze(args.checkpoint, args.cap, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
