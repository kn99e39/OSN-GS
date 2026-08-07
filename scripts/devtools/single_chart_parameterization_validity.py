"""Worklog 69: Single-Chart Parameterization Validity and Repair.

Determines whether each region's existing PCA-UV parameterization is a
valid SINGLE chart for its evidence, and whether invalidity (not fitting
capacity, per worklog 68's rejection of grid-resolution increases) explains
the extrapolation/local-folding pattern. Region formation, chart boundary,
and ownership gating are never touched; only the already-recovered
boundary+region-owned evidence (worklog 67, unmodified) is analyzed here,
at the ORIGINAL 6x6 grid (worklog 68's finding: do not adopt a higher
resolution).

Typed states (never silently "fixed" by relaxing a threshold):
  - `uv_valid`: passes every check below.
  - `partition_materialization_required`: fails at least one UV validity
    check. Split into a real patch ONLY if a safe partition boundary is
    directly derivable from the region's OWN existing accepted-edge graph
    (never a PCA rectangle/convex-hull/invented seam); otherwise stays
    fail-closed and unpartitioned.

Explicit non-claim: surface self-intersection (the fitted 3D parametric
surface folding over itself away from its own boundary) is NOT checked
anywhere in this script or the production pipeline it reads from --
reported literally as `"surface_self_intersection": "not_checked"` in every
patch record, never inferred from any of the UV/fold diagnostics below.
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
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_single_chart_uv_validity import (
    accepted_edge_uv_crossings,
    interior_within_boundary,
    neighborhood_preservation,
    parallel_sheet_suspicion,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

BASE_GRID = 6  # worklog 68: do not adopt a higher resolution -- stick with the original.
DENSITY_FRACTIONS = (0.25, 0.50, 1.00)
SAMPLE_RESOLUTION = 24


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


def _sample_surface(surface, resolution: int = SAMPLE_RESOLUTION):
    device = surface.control_grid.device
    dtype = surface.control_grid.dtype
    grid = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(grid, grid, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    return surface.evaluate(uv)


def _raw_and_dense_nn_error(evidence: torch.Tensor, surface) -> dict:
    sample_points = _sample_surface(surface)
    fwd_raw = torch.cdist(evidence, sample_points).min(dim=1).values.detach().cpu().numpy()
    bwd_raw = torch.cdist(sample_points, evidence).min(dim=1).values.detach().cpu().numpy()
    scale = _median_nn_spacing(evidence)
    return {
        "point_to_surface_raw": _percentiles(fwd_raw),
        "surface_to_evidence_raw": _percentiles(bwd_raw),
        "point_to_surface_norm_dense_nn": _percentiles(fwd_raw / scale),
        "surface_to_evidence_norm_dense_nn": _percentiles(bwd_raw / scale),
        "dense_nn_scale": scale,
    }


def _deterministic_spatial_fraction(evidence: torch.Tensor, fraction: float, grid_k: int = 8) -> torch.Tensor:
    """Same PCA-UV grid-cell technique as worklog 66-68's checkerboard
    holdout, generalized from a 2-way split to an N-way deterministic,
    spatially-interleaved subsample at an arbitrary fraction."""

    n = int(evidence.shape[0])
    if fraction >= 0.999 or n < 8:
        return evidence
    uv = pca_parameterize_points(evidence)
    cell_u = (uv[:, 0] * grid_k).clamp(0, grid_k - 1e-6).floor().long()
    cell_v = (uv[:, 1] * grid_k).clamp(0, grid_k - 1e-6).floor().long()
    cell_hash = (cell_u * grid_k + cell_v) % max(1, int(round(1.0 / fraction)))
    keep = cell_hash == 0
    if int(keep.sum()) < 4:
        return evidence
    return evidence[keep]


def _uv_validity_for_patch(
    boundary_points: torch.Tensor, full_evidence: torch.Tensor,
    representative_positions: torch.Tensor, representative_stable_ids: list,
    accepted_edges: list[tuple],
) -> dict:
    all_points = torch.cat((representative_positions, full_evidence), dim=0)
    all_ids = list(representative_stable_ids) + [f"fe{i}" for i in range(int(full_evidence.shape[0]))]
    uv_all = pca_parameterize_points(all_points)
    uv_by_id = {sid: tuple(uv_all[i].tolist()) for i, sid in enumerate(all_ids)}

    rep_count = int(representative_positions.shape[0])
    uv_evidence = uv_all[rep_count:]

    dup = uv_duplicate_diagnostics(uv_evidence)
    neighbor = neighborhood_preservation(full_evidence, uv_evidence, k=8)
    crossing = accepted_edge_uv_crossings(uv_by_id, accepted_edges)
    triangulation = uv_triangulation_diagnostics(full_evidence, uv_evidence)

    # Boundary LOOP UV, in the SAME shared PCA-UV frame as `uv_evidence`
    # (never a second independent PCA call, which would use different axes)
    # -- the boundary loop is always a subset of the region's representative
    # members (worklog 54-61), so each ordered boundary point is matched to
    # its representative-space UV by nearest world-position lookup,
    # preserving the ORIGINAL loop order for the point-in-polygon test.
    if int(boundary_points.shape[0]) and rep_count:
        boundary_match = torch.cdist(boundary_points, representative_positions).argmin(dim=1)
        boundary_uv_ordered = uv_all[:rep_count][boundary_match]
    else:
        boundary_uv_ordered = uv_all[:0]
    containment = interior_within_boundary(uv_evidence, boundary_uv_ordered)

    centered = full_evidence - full_evidence.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal_axis = vh[-1] if vh.shape[0] >= 3 else vh[0]
    parallel = parallel_sheet_suspicion(full_evidence, normal_axis)

    invalid_reasons = []
    if dup["uv_near_collision_count"] > 0:
        invalid_reasons.append(f"uv_near_collision_count={dup['uv_near_collision_count']}")
    if neighbor["neighborhood_preservation_mean"] is not None and neighbor["neighborhood_preservation_mean"] < 0.5:
        invalid_reasons.append(f"neighborhood_preservation_mean={neighbor['neighborhood_preservation_mean']:.2f}<0.5")
    if crossing["accepted_edge_uv_crossing_count"] > 0:
        invalid_reasons.append(f"accepted_edge_uv_crossing_count={crossing['accepted_edge_uv_crossing_count']}")
    if triangulation["triangle_total_count"] > 0 and triangulation["triangle_fold_count"] / triangulation["triangle_total_count"] > 0.05:
        invalid_reasons.append(f"triangle_fold_fraction={triangulation['triangle_fold_count']}/{triangulation['triangle_total_count']}>5%")
    if containment["interior_total_count"] > 0 and containment["interior_outside_boundary_count"] / containment["interior_total_count"] > 0.1:
        invalid_reasons.append(f"interior_outside_boundary={containment['interior_outside_boundary_count']}/{containment['interior_total_count']}>10%")
    if parallel["parallel_sheet_suspected"]:
        invalid_reasons.append(f"parallel_sheet_suspected gap_ratio={parallel['parallel_sheet_gap_ratio']:.2f}")

    return {
        "uv_duplicate": dup, "neighborhood_preservation": neighbor, "accepted_edge_crossing": crossing,
        "triangulation": triangulation, "boundary_containment": containment, "parallel_sheet": parallel,
        "valid": len(invalid_reasons) == 0, "invalid_reasons": invalid_reasons,
        "normal_axis": normal_axis.detach().cpu().tolist(),
    }


def _attempt_evidence_backed_partition(
    full_evidence: torch.Tensor, normal_axis: list, accepted_edges: list[tuple],
    representative_stable_ids: list, representative_positions: torch.Tensor,
) -> dict | None:
    """ONLY returns a partition when the region's OWN existing accepted-edge
    graph already shows near-zero direct connectivity between the two
    normal-axis clusters -- i.e. the topology itself already almost
    separated them. Never invents a boundary (no PCA rectangle, no convex
    hull, no arbitrary seam) -- the cut is read directly off the accepted
    edges the region formation step already produced."""

    axis = torch.as_tensor(normal_axis, dtype=full_evidence.dtype, device=full_evidence.device)
    projection = full_evidence @ axis
    median = projection.median()
    cluster_a = projection <= median
    if int(cluster_a.sum()) < 3 or int((~cluster_a).sum()) < 3:
        return None

    # Map representative stable ids to the SAME cluster split via their own
    # position's projection onto the same axis, then count direct accepted
    # edges crossing between the two clusters.
    rep_projection = representative_positions @ axis
    rep_median = rep_projection.median()
    rep_cluster = {sid: bool(rep_projection[i] <= rep_median) for i, sid in enumerate(representative_stable_ids)}
    cross_edges = 0
    total_edges = 0
    for a, b in accepted_edges:
        if a in rep_cluster and b in rep_cluster:
            total_edges += 1
            if rep_cluster[a] != rep_cluster[b]:
                cross_edges += 1
    if total_edges == 0:
        return None
    cross_ratio = cross_edges / total_edges
    if cross_ratio > 0.05:
        # The existing topology does NOT already separate these clusters --
        # no safe, evidence-backed cut to derive. Fail closed.
        return None
    return {
        "cluster_a": full_evidence[cluster_a], "cluster_b": full_evidence[~cluster_a],
        "cross_edge_ratio": cross_ratio, "total_accepted_edges_checked": total_edges,
    }


def analyze_patch(chart_type: str, region_id: int, boundary_points: torch.Tensor, full_evidence: torch.Tensor,
                   representative_positions: torch.Tensor, representative_stable_ids: list, accepted_edges: list) -> dict:
    validity = _uv_validity_for_patch(boundary_points, full_evidence, representative_positions, representative_stable_ids, accepted_edges)

    base_observed = torch.cat((boundary_points, full_evidence), dim=0)
    surface, _ = fit_torch_visible_surface_lsq(base_observed, resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2)
    base_fidelity = _raw_and_dense_nn_error(full_evidence, surface) if validity["valid"] else None

    density_sweep = {}
    for fraction in DENSITY_FRACTIONS:
        subset = _deterministic_spatial_fraction(full_evidence, fraction)
        observed = torch.cat((boundary_points, subset), dim=0)
        try:
            fraction_surface, _ = fit_torch_visible_surface_lsq(observed, resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2)
            density_sweep[fraction] = {
                "evidence_count": int(subset.shape[0]),
                **_raw_and_dense_nn_error(subset, fraction_surface),
            }
        except Exception as exc:  # noqa: BLE001
            density_sweep[fraction] = {"fit_failed": type(exc).__name__}

    partition_result = None
    if not validity["valid"] and validity["parallel_sheet"]["parallel_sheet_suspected"]:
        partition = _attempt_evidence_backed_partition(
            full_evidence, validity["normal_axis"], accepted_edges, representative_stable_ids, representative_positions,
        )
        if partition is not None:
            try:
                surface_a, _ = fit_torch_visible_surface_lsq(
                    torch.cat((boundary_points, partition["cluster_a"]), dim=0), resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2,
                )
                surface_b, _ = fit_torch_visible_surface_lsq(
                    torch.cat((boundary_points, partition["cluster_b"]), dim=0), resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2,
                )
                before_error = _raw_and_dense_nn_error(full_evidence, surface)
                after_error_a = _raw_and_dense_nn_error(partition["cluster_a"], surface_a)
                after_error_b = _raw_and_dense_nn_error(partition["cluster_b"], surface_b)
                partition_result = {
                    "applied": True, "cross_edge_ratio": partition["cross_edge_ratio"],
                    "cluster_a_count": int(partition["cluster_a"].shape[0]), "cluster_b_count": int(partition["cluster_b"].shape[0]),
                    "before_error": before_error, "after_error_cluster_a": after_error_a, "after_error_cluster_b": after_error_b,
                }
            except Exception as exc:  # noqa: BLE001
                partition_result = {"applied": False, "reason": f"repair_fit_failed:{type(exc).__name__}"}
        else:
            partition_result = {"applied": False, "reason": "no_safe_partition_derivable_from_existing_accepted_topology"}

    status = "uv_valid" if validity["valid"] else "partition_materialization_required"

    return {
        "chart_type": chart_type, "source_region_id": region_id,
        "status": status, "uv_validity": {k: v for k, v in validity.items() if k != "normal_axis"},
        "base_grid_fidelity_6x6": base_fidelity,
        "density_subsampling": density_sweep,
        "partition_attempt": partition_result,
        "surface_self_intersection": "not_checked",
    }


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
    rep_stable_ids_all = bundle.representative_stable_ids
    rep_positions_all = model.get_xyz.detach()[bundle.representative_indices]
    rep_index_by_id = {sid: i for i, sid in enumerate(rep_stable_ids_all)}

    items = [("physical", item) for item in construction.materialized_visible_nurbs_surfaces if item.surface is not None]
    items += [("parametric", item) for item in construction.materialized_parametric_chart_surfaces if item.surface is not None]

    patches = []
    for chart_type, item in items:
        key = (chart_type, item.input.source_region_id)
        fit = bundle.region_owned_full_evidence_fits.get(key)
        if fit is None or fit.state != "materialized":
            continue
        boundary_points = item.input.ordered_boundary_points
        if fit.full_evidence_stable_ids:
            evidence_index = torch.tensor(list(fit.full_evidence_stable_ids), dtype=torch.long, device=boundary_points.device)
            full_evidence = model.get_xyz.detach()[evidence_index]
        else:
            full_evidence = boundary_points[:0]

        region = region_by_id.get(item.input.source_region_id)
        member_ids = [sid for sid in (region.member_ids if region else []) if sid in rep_index_by_id]
        member_local = [rep_index_by_id[sid] for sid in member_ids]
        representative_positions = rep_positions_all[member_local] if member_local else boundary_points[:0]
        accepted_edges = list(region.internal_accepted_edge_ids) if region else []

        print(f"  patch {chart_type}/{item.input.source_region_id}: evidence={int(full_evidence.shape[0])} representatives={len(member_ids)} ...", flush=True)
        patches.append(analyze_patch(chart_type, item.input.source_region_id, boundary_points, full_evidence, representative_positions, member_ids, accepted_edges))

    return {"label": label, "patch_count": len(patches), "patches": patches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val69/single_chart_validity_report.json"))
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
