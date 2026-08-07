"""Worklog 70: Region-Owned Full-Evidence Boundary Materialization.

Worklog 69 found the dominant single-chart-invalidity signal (22/22
`partition_materialization_required`) is not true multi-sheet folding but a
scale mismatch: a chart boundary built from only 3-4 REPRESENTATIVE points
cannot geometrically contain the region's much larger region-owned FULL
evidence footprint (worklog 67). Per this round's explicit instruction, that
22/22 verdict is NOT adopted as canonical.

This script applies `materialize_dense_boundary`
(`osn_gs/surface/torch_region_owned_boundary_materialization.py`, worklog 70)
to each region's existing parametric chart boundary, using the SAME
region-owned full evidence worklog 67 already recovered, and re-evaluates
single-chart UV validity only AFTER the dense boundary is in place. Region
formation, representative topology, chart boundary construction, and
ownership gating are never touched -- the dense boundary is an ADDITIVE
downstream artifact built only from the already-accepted topology's own
boundary and the region's own owned evidence.

Per explicit instruction this round: `parallel_sheet_suspected` and raw-
evidence triangle folding (`torch_single_chart_uv_validity.uv_triangulation_
diagnostics`) are recorded as DIAGNOSTIC ONLY and excluded from the
admission gate -- gating uses only uv_near_collision, neighborhood_
preservation, accepted_edge_uv_crossing, interior_outside_boundary (the same
four checks and thresholds worklog 69 already established, unchanged).

Four typed outcomes (never a fifth ad hoc state):
  - `full_evidence_boundary_materialization_required`: the dense boundary
    itself could not be built (fails closed, per
    `torch_region_owned_boundary_materialization.materialize_dense_boundary`).
  - `partition_materialization_required`: the dense boundary WAS built, but
    reduced-gate UV validity still fails afterward, or the post-boundary fit
    has a degenerate Jacobian -- escalated only at this point, never before.
  - `extrapolative` / `valid_supported`: worklog 66's own existing borrowed
    thresholds (`UNDER_SUPPORTED_MIN_EVIDENCE`,
    `EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND`), applied to the held-out
    (deterministic spatial checkerboard split, worklog 68's convention)
    dense-NN normalized point-to-surface/surface-to-evidence p95.

`validate_simple_closed_loop` (used inside `materialize_dense_boundary`)
stays a BOUNDARY-LOOP simple-polygon check; `surface_self_intersection` is
reported literally as `"not_checked"` in every record, per the standing
worklog 68/69 convention -- nothing in this pipeline checks the fitted 3D
parametric surface for folding over itself away from its own boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_region_owned_boundary_materialization import materialize_dense_boundary
from osn_gs.surface.torch_single_chart_uv_validity import (
    accepted_edge_uv_crossings,
    interior_within_boundary,
    neighborhood_preservation,
    parallel_sheet_suspicion,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

BASE_GRID = 6  # worklog 68: do not adopt a higher resolution.
HOLDOUT_CHECKER_K = 4  # worklog 68's PCA-uv checkerboard convention, unchanged.
SAMPLE_RESOLUTION = 24

# Both borrowed unchanged from worklog 66 (themselves cited there as
# pre-existing production conventions, e.g.
# RegionFormationConfig.core_region_typical_min_size == 4) -- never tuned to
# this round's results.
UNDER_SUPPORTED_MIN_EVIDENCE = 4
EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND = 4.0

STATE_BOUNDARY_FAILED = "full_evidence_boundary_materialization_required"
STATE_PARTITION_REQUIRED = "partition_materialization_required"
STATE_EXTRAPOLATIVE = "extrapolative"
STATE_VALID_SUPPORTED = "valid_supported"


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def _median_nn_spacing(points: torch.Tensor) -> float:
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    value = float(d.min(dim=1).values.median())
    return value if value > 0 else 1e-6


def _deterministic_spatial_holdout_split(evidence: torch.Tensor, k: int = HOLDOUT_CHECKER_K) -> tuple[torch.Tensor, torch.Tensor]:
    """Same PCA-uv checkerboard split as worklog 68 -- deterministic,
    spatially interleaved, reproducible."""

    if int(evidence.shape[0]) < 8:
        return evidence, evidence[:0]
    uv = pca_parameterize_points(evidence)
    cell_u = (uv[:, 0] * k).clamp(0, k - 1e-6).floor().long()
    cell_v = (uv[:, 1] * k).clamp(0, k - 1e-6).floor().long()
    holdout_mask = ((cell_u + cell_v) % 2) == 0
    if int(holdout_mask.sum()) == 0 or int((~holdout_mask).sum()) == 0:
        return evidence, evidence[:0]
    return evidence[~holdout_mask], evidence[holdout_mask]


def _sample_surface_with_normals(surface, resolution: int = SAMPLE_RESOLUTION):
    device = surface.control_grid.device
    dtype = surface.control_grid.dtype
    grid = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(grid, grid, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    points, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    normals = torch.cross(deriv_u, deriv_v, dim=1)
    return points, deriv_u, deriv_v, normals


def _surface_area(sample_points: torch.Tensor, resolution: int) -> float:
    pts = sample_points.reshape(resolution, resolution, 3)
    p00, p01, p10, p11 = pts[:-1, :-1], pts[:-1, 1:], pts[1:, :-1], pts[1:, 1:]
    a1 = 0.5 * torch.cross(p01 - p00, p10 - p00, dim=-1).norm(dim=-1)
    a2 = 0.5 * torch.cross(p11 - p01, p10 - p01, dim=-1).norm(dim=-1)
    return float((a1 + a2).sum())


def _raw_and_dense_nn_error(evidence: torch.Tensor, surface) -> dict:
    sample_points = _sample_surface_with_normals(surface)[0]
    if int(evidence.shape[0]) == 0:
        empty = _percentiles(np.array([]))
        return {
            "point_to_surface_raw": empty, "surface_to_evidence_raw": empty,
            "point_to_surface_norm_dense_nn": empty, "surface_to_evidence_norm_dense_nn": empty,
            "dense_nn_scale": None, "evidence_count": 0,
        }
    fwd_raw = torch.cdist(evidence, sample_points).min(dim=1).values.detach().cpu().numpy()
    bwd_raw = torch.cdist(sample_points, evidence).min(dim=1).values.detach().cpu().numpy()
    scale = _median_nn_spacing(evidence)
    return {
        "point_to_surface_raw": _percentiles(fwd_raw),
        "surface_to_evidence_raw": _percentiles(bwd_raw),
        "point_to_surface_norm_dense_nn": _percentiles(fwd_raw / scale),
        "surface_to_evidence_norm_dense_nn": _percentiles(bwd_raw / scale),
        "dense_nn_scale": scale, "evidence_count": int(evidence.shape[0]),
    }


def _shared_uv_validity(
    representative_positions: torch.Tensor, representative_stable_ids: list,
    full_evidence_positions: torch.Tensor, full_evidence_stable_ids: list,
    boundary_ids_ordered: list, accepted_edges: list[tuple],
) -> dict:
    """Single shared PCA-UV frame over (region representatives + region-owned
    full evidence) -- ids for BOTH populations, so any ordered boundary loop
    drawn from either population (the original representative-only loop, or
    worklog 70's dense loop that also includes evidence-sourced vertices) can
    be looked up directly by id, with no nearest-position matching hack."""

    all_points = torch.cat((representative_positions, full_evidence_positions), dim=0)
    all_ids = list(representative_stable_ids) + list(full_evidence_stable_ids)
    uv_all = pca_parameterize_points(all_points)
    uv_by_id = {sid: uv_all[i] for i, sid in enumerate(all_ids)}

    rep_count = int(representative_positions.shape[0])
    uv_evidence = uv_all[rep_count:]

    dup = uv_duplicate_diagnostics(uv_evidence)
    neighbor = neighborhood_preservation(full_evidence_positions, uv_evidence, k=8)
    uv_by_id_float = {sid: tuple(v.tolist()) for sid, v in uv_by_id.items()}
    crossing = accepted_edge_uv_crossings(uv_by_id_float, accepted_edges)
    triangulation = uv_triangulation_diagnostics(full_evidence_positions, uv_evidence)

    boundary_id_set = set(boundary_ids_ordered)
    boundary_uv_ordered = (
        torch.stack([uv_by_id[sid] for sid in boundary_ids_ordered], dim=0)
        if boundary_ids_ordered else uv_all[:0]
    )
    interior_mask = torch.tensor([sid not in boundary_id_set for sid in full_evidence_stable_ids], dtype=torch.bool)
    interior_uv = uv_evidence[interior_mask] if int(interior_mask.numel()) else uv_evidence[:0]
    containment = interior_within_boundary(interior_uv, boundary_uv_ordered)

    centered = full_evidence_positions - full_evidence_positions.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal_axis = vh[-1] if vh.shape[0] >= 3 else vh[0]
    parallel = parallel_sheet_suspicion(full_evidence_positions, normal_axis)

    reduced_reasons = []
    if dup["uv_near_collision_count"] > 0:
        reduced_reasons.append(f"uv_near_collision_count={dup['uv_near_collision_count']}")
    if neighbor["neighborhood_preservation_mean"] is not None and neighbor["neighborhood_preservation_mean"] < 0.5:
        reduced_reasons.append(f"neighborhood_preservation_mean={neighbor['neighborhood_preservation_mean']:.2f}<0.5")
    if crossing["accepted_edge_uv_crossing_count"] > 0:
        reduced_reasons.append(f"accepted_edge_uv_crossing_count={crossing['accepted_edge_uv_crossing_count']}")
    if containment["interior_total_count"] > 0 and containment["interior_outside_boundary_count"] / containment["interior_total_count"] > 0.1:
        reduced_reasons.append(f"interior_outside_boundary={containment['interior_outside_boundary_count']}/{containment['interior_total_count']}>10%")

    return {
        "uv_duplicate": dup, "neighborhood_preservation": neighbor, "accepted_edge_crossing": crossing,
        "boundary_containment": containment,
        "triangulation_diagnostic_only": triangulation, "parallel_sheet_diagnostic_only": parallel,
        "reduced_gate_valid": len(reduced_reasons) == 0, "reduced_gate_invalid_reasons": reduced_reasons,
    }


def _fit_and_error(boundary_positions: torch.Tensor, interior_evidence: torch.Tensor, full_evidence_positions: torch.Tensor) -> dict:
    """Unconditional diagnostic fit at BASE_GRID -- always computed for both
    the ORIGINAL and the dense boundary, regardless of whether either passes
    the reduced UV-validity gate, so the round's requested before/after
    error comparison has real numbers on both sides even where admission is
    denied. Classification itself still only USES the `after` result when
    the gate actually passes (see `analyze_patch`)."""

    train_interior, holdout_interior = _deterministic_spatial_holdout_split(interior_evidence)
    fit_observed = torch.cat((boundary_positions, train_interior), dim=0)
    try:
        surface, _ = fit_torch_visible_surface_lsq(fit_observed, resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2)
    except Exception as exc:  # noqa: BLE001
        return {"fit_failed": f"{type(exc).__name__}: {exc}"}

    train_error = _raw_and_dense_nn_error(train_interior, surface)
    holdout_error = _raw_and_dense_nn_error(holdout_interior, surface)
    full_error = _raw_and_dense_nn_error(full_evidence_positions, surface)
    sample_points, deriv_u, deriv_v, normals = _sample_surface_with_normals(surface)
    dense_nn_scale = _median_nn_spacing(full_evidence_positions)
    jacobian = compute_parametric_jacobian_metrics(deriv_u, deriv_v, scale=dense_nn_scale)
    local_fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)
    area = _surface_area(sample_points, SAMPLE_RESOLUTION)
    return {
        "train_evidence_count": int(train_interior.shape[0]), "holdout_evidence_count": int(holdout_interior.shape[0]),
        "fit_train_error": train_error, "fit_holdout_error": holdout_error, "fit_full_error": full_error,
        "jacobian_near_degenerate_count": jacobian["near_degenerate_count"],
        "local_fold_fraction_diagnostic_only": local_fold["local_fold_fraction"],
        "patch_area": area,
    }


def analyze_patch(
    chart_type: str, region_id: int,
    boundary_points: torch.Tensor, boundary_ids: list, segment_kinds: list,
    full_evidence_positions: torch.Tensor, full_evidence_stable_ids: list,
    representative_positions: torch.Tensor, representative_stable_ids: list,
    accepted_edges: list[tuple], local_evidence_scale: float,
) -> dict:
    before_validity = _shared_uv_validity(
        representative_positions, representative_stable_ids,
        full_evidence_positions, full_evidence_stable_ids, boundary_ids, accepted_edges,
    )

    boundary_id_set = set(boundary_ids)
    ext_mask = torch.tensor([sid not in boundary_id_set for sid in full_evidence_stable_ids], dtype=torch.bool)
    ext_positions = full_evidence_positions[ext_mask] if int(ext_mask.numel()) else full_evidence_positions[:0]
    ext_ids = [sid for sid, keep in zip(full_evidence_stable_ids, ext_mask.tolist()) if keep]

    dense = materialize_dense_boundary(
        boundary_points, list(boundary_ids), list(segment_kinds),
        ext_positions, ext_ids, local_evidence_scale=local_evidence_scale,
    )

    before_interior_mask = torch.tensor([sid not in boundary_id_set for sid in full_evidence_stable_ids], dtype=torch.bool)
    before_interior = full_evidence_positions[before_interior_mask] if int(before_interior_mask.numel()) else full_evidence_positions[:0]
    before_fit = _fit_and_error(boundary_points, before_interior, full_evidence_positions)

    record = {
        "chart_type": chart_type, "source_region_id": region_id,
        "before_boundary_vertex_count": len(boundary_ids),
        "full_evidence_support_count": int(full_evidence_positions.shape[0]),
        "before_uv_validity": before_validity,
        "before_fit_diagnostic": before_fit,
        "dense_boundary_state": dense.state, "dense_boundary_reasons": list(dense.reasons),
        "dense_boundary_extension_count": dense.extension_count,
        "dense_boundary_vertex_count": len(dense.ordered_ids),
        "local_evidence_scale": local_evidence_scale,
        "surface_self_intersection": "not_checked",
    }

    if dense.state != "materialized":
        record["status"] = STATE_BOUNDARY_FAILED
        record["status_reasons"] = list(dense.reasons)
        return record

    dense_boundary_ids = list(dense.ordered_ids)
    after_validity = _shared_uv_validity(
        representative_positions, representative_stable_ids,
        full_evidence_positions, full_evidence_stable_ids, dense_boundary_ids, accepted_edges,
    )
    record["after_uv_validity"] = after_validity

    dense_boundary_id_set = set(dense_boundary_ids)
    interior_mask = torch.tensor([sid not in dense_boundary_id_set for sid in full_evidence_stable_ids], dtype=torch.bool)
    interior_evidence = full_evidence_positions[interior_mask] if int(interior_mask.numel()) else full_evidence_positions[:0]
    after_fit = _fit_and_error(dense.ordered_positions, interior_evidence, full_evidence_positions)
    record["after_fit_diagnostic"] = after_fit

    if not after_validity["reduced_gate_valid"]:
        record["status"] = STATE_PARTITION_REQUIRED
        record["status_reasons"] = list(after_validity["reduced_gate_invalid_reasons"])
        return record

    if "fit_failed" in after_fit:
        record["status"] = STATE_PARTITION_REQUIRED
        record["status_reasons"] = [f"post_boundary_fit_failed:{after_fit['fit_failed']}"]
        return record

    holdout_error = after_fit["fit_holdout_error"]
    full_error = after_fit["fit_full_error"]
    eval_error = holdout_error if holdout_error["evidence_count"] > 0 else full_error
    bwd_p95 = eval_error["surface_to_evidence_norm_dense_nn"]["p95"] or 0.0
    fwd_p95 = eval_error["point_to_surface_norm_dense_nn"]["p95"] or 0.0
    total_support = int(dense.ordered_positions.shape[0]) + int(interior_evidence.shape[0])

    status_reasons: list[str] = []
    if after_fit["jacobian_near_degenerate_count"] > 0:
        status = STATE_PARTITION_REQUIRED
        status_reasons.append(f"jacobian_near_degenerate_count={after_fit['jacobian_near_degenerate_count']}")
    elif total_support < UNDER_SUPPORTED_MIN_EVIDENCE:
        status = STATE_PARTITION_REQUIRED
        status_reasons.append(f"post_boundary_support_count={total_support}<{UNDER_SUPPORTED_MIN_EVIDENCE}")
    elif bwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND or fwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND:
        status = STATE_EXTRAPOLATIVE
        status_reasons.append(f"held_out_normalized_p95 fwd={fwd_p95:.2f} bwd={bwd_p95:.2f}>{EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND}")
    else:
        status = STATE_VALID_SUPPORTED

    record["status"] = status
    record["status_reasons"] = status_reasons
    return record


def analyze_condition(model, cap: int, device: str, label: str) -> dict:
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction
    region_by_id = {r.region_id: r for r in construction.surface_regions.regions}
    chart_boundary_by_region = {c.region_id: c for c in construction.region_parametric_chart_boundaries}
    rep_stable_ids_all = bundle.representative_stable_ids
    rep_positions_all = model.get_xyz.detach()[bundle.representative_indices]
    rep_index_by_id = {sid: i for i, sid in enumerate(rep_stable_ids_all)}
    mean_spacing = bundle.evidence.mean_spacing if bundle.evidence is not None else None

    items = [("physical", item) for item in construction.materialized_visible_nurbs_surfaces if item.surface is not None]
    items += [("parametric", item) for item in construction.materialized_parametric_chart_surfaces if item.surface is not None]

    patches = []
    for chart_type, item in items:
        region_id = item.input.source_region_id
        key = (chart_type, region_id)
        fit = bundle.region_owned_full_evidence_fits.get(key)
        if fit is None or fit.state != "materialized":
            continue
        boundary_points = item.input.ordered_boundary_points
        boundary_ids = list(item.input.ordered_boundary_point_ids)
        if fit.full_evidence_stable_ids:
            evidence_index = torch.tensor(list(fit.full_evidence_stable_ids), dtype=torch.long, device=boundary_points.device)
            full_evidence = model.get_xyz.detach()[evidence_index]
        else:
            full_evidence = boundary_points[:0]
        full_evidence_stable_ids = list(fit.full_evidence_stable_ids)

        region = region_by_id.get(region_id)
        member_ids = [sid for sid in (region.member_ids if region else []) if sid in rep_index_by_id]
        member_local = [rep_index_by_id[sid] for sid in member_ids]
        representative_positions = rep_positions_all[member_local] if member_local else boundary_points[:0]
        accepted_edges = list(region.internal_accepted_edge_ids) if region else []

        if chart_type != "parametric":
            # No real data has ever produced a "physical" chart_type through
            # this analysis path (worklog 69's own run: 22/22 parametric).
            # No per-edge typed segment-kind provenance exists for the
            # physical eligible_closed_boundary path (only an aggregate
            # `boundary_reason_distribution`, worklog 39/54) -- fail closed
            # rather than invent/guess a segment kind per edge.
            patches.append({
                "chart_type": chart_type, "source_region_id": region_id,
                "status": STATE_BOUNDARY_FAILED,
                "status_reasons": ["no_per_edge_segment_kind_provenance_for_physical_chart_boundary"],
                "surface_self_intersection": "not_checked",
            })
            continue

        chart = chart_boundary_by_region.get(region_id)
        if chart is None:
            patches.append({
                "chart_type": chart_type, "source_region_id": region_id,
                "status": STATE_BOUNDARY_FAILED,
                "status_reasons": ["region_parametric_chart_boundary_missing"],
                "surface_self_intersection": "not_checked",
            })
            continue
        edge_kind_lookup = {frozenset({seg.node_a, seg.node_b}): seg.segment_kind for seg in chart.segments}
        n = len(boundary_ids)
        try:
            segment_kinds = [
                edge_kind_lookup[frozenset({boundary_ids[i], boundary_ids[(i + 1) % n]})]
                for i in range(n)
            ]
        except KeyError:
            patches.append({
                "chart_type": chart_type, "source_region_id": region_id,
                "status": STATE_BOUNDARY_FAILED,
                "status_reasons": ["missing_edge_provenance_for_one_or_more_boundary_edges"],
                "surface_self_intersection": "not_checked",
            })
            continue

        if mean_spacing is not None and member_local:
            local_indices = torch.tensor(member_local, dtype=torch.long, device=mean_spacing.device)
            local_scale = float(mean_spacing[local_indices].median())
        else:
            local_scale = _median_nn_spacing(full_evidence) if int(full_evidence.shape[0]) >= 2 else 1e-6

        print(
            f"  patch {chart_type}/{region_id}: boundary={n} evidence={int(full_evidence.shape[0])} "
            f"representatives={len(member_ids)} local_scale={local_scale:.6f} ...", flush=True,
        )
        patches.append(analyze_patch(
            chart_type, region_id, boundary_points, boundary_ids, segment_kinds,
            full_evidence, full_evidence_stable_ids,
            representative_positions, member_ids, accepted_edges, local_scale,
        ))

    return {"label": label, "patch_count": len(patches), "patches": patches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val70/dense_boundary_materialization_report.json"))
    args = parser.parse_args()
    device = "cuda"

    from osn_gs.gaussian.torch_model import TorchGaussianModel

    def load_ckpt(ckpt_dir: Path):
        payload = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
        raw = payload["model_raw"]
        rest_dim = int(raw["features_rest"].shape[-2])
        degree = 0
        while (degree + 1) ** 2 - 1 < rest_dim:
            degree += 1
        model = TorchGaussianModel(sh_degree=degree, device=device)
        model.replace_tensors(
            xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
            opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
            uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
            surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
            surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
            stable_gaussian_ids=raw.get("stable_gaussian_ids"),
        )
        return model

    report: dict = {"baseline_compatible": {}, "baseline": {}}
    for it in args.iterations:
        ckpt = args.baseline_compatible_run_dir / str(it)
        if (ckpt / "checkpoint.pt").exists():
            print(f"analyzing baseline_compatible@{it} ...", flush=True)
            model = load_ckpt(ckpt)
            report["baseline_compatible"][str(it)] = analyze_condition(model, args.cap, device, f"baseline_compatible@{it}")
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            print(f"analyzing baseline@{it} ...", flush=True)
            model = baseline_ply_analysis.load_baseline_ply_as_model(ply, device)
            report["baseline"][str(it)] = analyze_condition(model, args.cap, device, f"baseline@{it}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
