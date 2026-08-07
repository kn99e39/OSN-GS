"""Worklog 71: Region-Owned Full-Evidence Boundary Topology Reconstruction.

worklog 70's per-edge densification of the 3-4-point REPRESENTATIVE boundary
is retired (its `interior_outside_boundary` stayed >10% in every patch that
even materialized a simple loop -- the original edge/wedge topology itself
cannot trace real evidence shape no matter how densely each wedge is
filled). Region formation, representative membership, and ownership gating
are never touched here; the representative boundary is used only as typed
PROVENANCE seed positions, never as the forced final topology.

This script:
  1. Recomputes the region's typed boundary half-edge evidence directly via
     the existing, unmodified `extract_support_termination_candidates`
     (physical termination / reliability frontier / sampling gap) AND
     `extract_world_space_boundary_halfedge_candidates` (crease / parallel-
     sheet conflict / ambiguous continuation) -- the latter is computed by
     production but never merged into `construction.boundary_halfedge_
     candidates` (a dead-code gap found this round, not fixed in production,
     only worked around here by calling the same function ourselves).
  2. Collects REGION-owned (not patch-owned) full-cloud evidence via the
     same normal/residual compatibility gate as worklog 67
     (`TorchOSNGSPipeline._propagate_with_evidence_gating`, unmodified),
     clustered by REGION id instead of by materialized-patch id.
  3. Reconstructs ordered boundary topology per region
     (`osn_gs.surface.torch_region_owned_full_evidence_boundary_topology`,
     worklog 71, new) -- typed closed_loop / open_fragment / branch /
     ambiguous / insufficient-evidence outcomes; multiple independent closed
     loops in one region are kept and reported separately.
  4. Only for `boundary_topology_closed_loop_recovered` regions: re-fits a
     6x6 NURBS patch (worklog 68: do not raise resolution) from the
     recovered boundary + remaining interior evidence, re-runs worklog 69's
     single-chart UV validity with the SAME reduced gate worklog 70 used
     (uv_near_collision / neighborhood_preservation / accepted_edge_uv_
     crossing / interior_outside_boundary only), and only escalates to
     `partition_materialization_required` if that STILL fails.

`surface_self_intersection` stays `"not_checked"` everywhere, as in every
prior worklog since 66.
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
from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import (
    STATE_CLOSED_LOOP_RECOVERED,
    reconstruct_region_boundary_topology,
)
from osn_gs.surface.torch_single_chart_uv_validity import (
    accepted_edge_uv_crossings,
    interior_within_boundary,
    neighborhood_preservation,
    parallel_sheet_suspicion,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)
from osn_gs.surface.torch_termination_neighborhood_scale import resolve_termination_neighborhood_scale
from osn_gs.surface.torch_visible_surface_construction import _orient_normals_along_accepted_topology
from osn_gs.surface.torch_world_space_boundary_halfedges import extract_world_space_boundary_halfedge_candidates

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

BASE_GRID = 6  # worklog 68: do not adopt a higher resolution.
HOLDOUT_CHECKER_K = 4  # worklog 68's PCA-uv checkerboard convention, unchanged.
SAMPLE_RESOLUTION = 24
UNDER_SUPPORTED_MIN_EVIDENCE = 4  # worklog 66, borrowed unchanged.
EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND = 4.0  # worklog 66, borrowed unchanged.

STATUS_BOUNDARY_UNRESOLVED = "boundary_topology_unresolved"
STATUS_PARTITION_REQUIRED = "partition_materialization_required"
STATUS_EXTRAPOLATIVE = "extrapolative"
STATUS_VALID_SUPPORTED = "valid_supported"


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
    if int(evidence.shape[0]) == 0:
        empty = _percentiles(np.array([]))
        return {
            "point_to_surface_raw": empty, "surface_to_evidence_raw": empty,
            "point_to_surface_norm_dense_nn": empty, "surface_to_evidence_norm_dense_nn": empty,
            "dense_nn_scale": None, "evidence_count": 0,
        }
    sample_points = _sample_surface_with_normals(surface)[0]
    fwd_raw = torch.cdist(evidence, sample_points).min(dim=1).values.detach().cpu().numpy()
    bwd_raw = torch.cdist(sample_points, evidence).min(dim=1).values.detach().cpu().numpy()
    scale = _median_nn_spacing(evidence)
    return {
        "point_to_surface_raw": _percentiles(fwd_raw), "surface_to_evidence_raw": _percentiles(bwd_raw),
        "point_to_surface_norm_dense_nn": _percentiles(fwd_raw / scale),
        "surface_to_evidence_norm_dense_nn": _percentiles(bwd_raw / scale),
        "dense_nn_scale": scale, "evidence_count": int(evidence.shape[0]),
    }


def _shared_uv_validity(
    representative_positions: torch.Tensor, representative_stable_ids: list,
    full_evidence_positions: torch.Tensor, full_evidence_stable_ids: list,
    boundary_ids_ordered: list, accepted_edges: list[tuple],
) -> dict:
    """Same reduced-gate contract worklog 70 used: uv_near_collision /
    neighborhood_preservation / accepted_edge_uv_crossing / interior_
    outside_boundary only. parallel_sheet_suspected and raw-evidence
    triangle_fold_fraction stay diagnostic-only per this round's own
    instruction (same as worklog 70's)."""

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
        torch.stack([uv_by_id[sid] for sid in boundary_ids_ordered if sid in uv_by_id], dim=0)
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


def analyze_recovered_loop(
    region_id: int, loop_index: int, ordered_ids: tuple, ordered_positions_3d: torch.Tensor,
    full_evidence_positions: torch.Tensor, full_evidence_stable_ids: list,
    representative_positions: torch.Tensor, representative_stable_ids: list, accepted_edges: list,
) -> dict:
    validity = _shared_uv_validity(
        representative_positions, representative_stable_ids,
        full_evidence_positions, full_evidence_stable_ids, list(ordered_ids), accepted_edges,
    )
    record = {
        "region_id": region_id, "loop_index": loop_index,
        "boundary_vertex_count": len(ordered_ids),
        "full_evidence_support_count": int(full_evidence_positions.shape[0]),
        "uv_validity": validity,
        "surface_self_intersection": "not_checked",
    }
    if not validity["reduced_gate_valid"]:
        record["status"] = STATUS_PARTITION_REQUIRED
        record["status_reasons"] = list(validity["reduced_gate_invalid_reasons"])
        return record

    boundary_id_set = set(ordered_ids)
    interior_mask = torch.tensor([sid not in boundary_id_set for sid in full_evidence_stable_ids], dtype=torch.bool)
    interior_evidence = full_evidence_positions[interior_mask] if int(interior_mask.numel()) else full_evidence_positions[:0]
    train_interior, holdout_interior = _deterministic_spatial_holdout_split(interior_evidence)
    fit_observed = torch.cat((ordered_positions_3d, train_interior), dim=0)

    try:
        surface, _ = fit_torch_visible_surface_lsq(fit_observed, resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2)
    except Exception as exc:  # noqa: BLE001
        record["status"] = STATUS_PARTITION_REQUIRED
        record["status_reasons"] = [f"post_topology_fit_failed:{type(exc).__name__}"]
        return record

    train_error = _raw_and_dense_nn_error(train_interior, surface)
    holdout_error = _raw_and_dense_nn_error(holdout_interior, surface)
    full_error = _raw_and_dense_nn_error(full_evidence_positions, surface)
    sample_points, deriv_u, deriv_v, normals = _sample_surface_with_normals(surface)
    dense_nn_scale = _median_nn_spacing(full_evidence_positions)
    jacobian = compute_parametric_jacobian_metrics(deriv_u, deriv_v, scale=dense_nn_scale)
    local_fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)
    area = _surface_area(sample_points, SAMPLE_RESOLUTION)

    eval_error = holdout_error if holdout_error["evidence_count"] > 0 else full_error
    bwd_p95 = eval_error["surface_to_evidence_norm_dense_nn"]["p95"] or 0.0
    fwd_p95 = eval_error["point_to_surface_norm_dense_nn"]["p95"] or 0.0
    total_support = int(ordered_positions_3d.shape[0]) + int(interior_evidence.shape[0])

    status_reasons: list[str] = []
    if jacobian["near_degenerate_count"] > 0:
        status = STATUS_PARTITION_REQUIRED
        status_reasons.append(f"jacobian_near_degenerate_count={jacobian['near_degenerate_count']}")
    elif total_support < UNDER_SUPPORTED_MIN_EVIDENCE:
        status = STATUS_PARTITION_REQUIRED
        status_reasons.append(f"post_topology_support_count={total_support}<{UNDER_SUPPORTED_MIN_EVIDENCE}")
    elif bwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND or fwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND:
        status = STATUS_EXTRAPOLATIVE
        status_reasons.append(f"held_out_normalized_p95 fwd={fwd_p95:.2f} bwd={bwd_p95:.2f}>{EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND}")
    else:
        status = STATUS_VALID_SUPPORTED

    record.update({
        "status": status, "status_reasons": status_reasons,
        "train_evidence_count": int(train_interior.shape[0]), "holdout_evidence_count": int(holdout_interior.shape[0]),
        "fit_train_error": train_error, "fit_holdout_error": holdout_error, "fit_full_error": full_error,
        "jacobian_near_degenerate_count": jacobian["near_degenerate_count"],
        "local_fold_fraction_diagnostic_only": local_fold["local_fold_fraction"],
        "patch_area": area,
    })
    return record


def analyze_condition(model, cap: int, device: str, label: str) -> dict:
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    points = model.get_xyz.detach()
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            points, covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction
    regions = construction.surface_regions
    rep_indices = bundle.representative_indices
    rep_stable_ids = bundle.representative_stable_ids
    rep_positions = points[rep_indices]

    if len(rep_stable_ids) != len(regions.node_region_id):
        return {"label": label, "region_count": 0, "regions": [], "note": "representative_set_not_downsampled_or_mismatched"}

    # Step 1: recompute BOTH typed candidate families ourselves (worklog 71
    # finding: `extract_world_space_boundary_halfedge_candidates`'s output is
    # computed by production but never merged into
    # `construction.boundary_halfedge_candidates` -- recovered here
    # additively, production untouched).
    frame = construction.covariance_frame
    reliability = construction.reliability
    graph = construction.manifold_affinity
    accepted = construction.accepted_local_topology
    oriented_normals = _orient_normals_along_accepted_topology(frame.normal_candidate, accepted, rep_stable_ids)
    canonical_frames = construct_canonical_region_tangent_frames(rep_positions, frame, reliability, regions, ids=rep_stable_ids)
    resolved_scale = resolve_termination_neighborhood_scale(candidate_scale=None, tangent_major_scale=frame.tangent_major_scale)
    termination_halfedges = extract_support_termination_candidates(
        rep_positions, oriented_normals, resolved_scale, regions, ids=rep_stable_ids,
        sectors=8, canonical_frames=canonical_frames, continuation=None, affinity_graph=graph,
    )
    relation_halfedges = extract_world_space_boundary_halfedge_candidates(rep_positions, oriented_normals, regions, graph, ids=rep_stable_ids)
    all_seed_candidates = tuple(sorted(tuple(termination_halfedges) + tuple(relation_halfedges), key=lambda c: c.half_edge_id))

    # Step 2: region-owned (not patch-owned) full evidence via the SAME
    # normal/residual gate as worklog 67, clustered by REGION id directly.
    cluster_ids_by_representative = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _diag = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster_ids_by_representative)
    propagated_cpu = propagated.detach().cpu().tolist()
    evidence_by_region: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated_cpu):
        if region_id >= 0:
            evidence_by_region.setdefault(region_id, []).append(full_index)

    mean_spacing = bundle.evidence.mean_spacing

    region_records = []
    for region in regions.regions:
        region_id = region.region_id
        evidence_indices = evidence_by_region.get(region_id, [])
        member_ids = [sid for sid in region.member_ids if sid in rep_stable_ids]
        member_local = [rep_stable_ids.index(sid) for sid in member_ids]  # small M, fine
        representative_positions = rep_positions[member_local] if member_local else rep_positions[:0]
        accepted_edges = list(region.internal_accepted_edge_ids)

        if not evidence_indices:
            region_records.append({
                "region_id": region_id, "loops": [{"status": STATUS_BOUNDARY_UNRESOLVED, "topology_status": "boundary_topology_insufficient_evidence", "reasons": ["no_region_owned_full_evidence"]}],
            })
            continue
        evidence_ids = [stable_ids[i] for i in evidence_indices]
        evidence_positions = points[torch.tensor(evidence_indices, dtype=torch.long, device=points.device)]

        # A single already-established scale per region (worklog 32's own
        # mean_spacing, median over this region's own representatives) --
        # not a new invented constant, and deliberately not the broader 6x
        # `local_radius_tangent_scale_multiplier` (worklog 135, a different
        # "is this evidence at all" question) since this round needs a TIGHT
        # per-edge chaining/binning scale, not a broad membership radius.
        if member_local:
            local_scale = float(mean_spacing[torch.tensor(member_local, dtype=torch.long, device=mean_spacing.device)].median())
        else:
            local_scale = _median_nn_spacing(evidence_positions)

        topology_results = reconstruct_region_boundary_topology(
            region_id, all_seed_candidates, evidence_ids, evidence_positions, local_scale,
        )
        print(f"  region {region_id}: evidence={len(evidence_ids)} seed_candidates={sum(1 for c in all_seed_candidates if c.source_region_id == region_id)} local_scale={local_scale:.6f} -> {[r.status for r in topology_results]}", flush=True)

        loops = []
        for loop_index, result in enumerate(topology_results):
            if result.status != STATE_CLOSED_LOOP_RECOVERED:
                loops.append({
                    "topology_status": result.status, "reasons": list(result.reasons),
                    "seed_component_state": result.seed_component.ordering_state if result.seed_component else None,
                    "seed_component_member_count": len(result.seed_component.ordered_half_edge_ids) if result.seed_component else 0,
                    "status": STATUS_BOUNDARY_UNRESOLVED,
                })
                continue
            densified = result.densified
            ordered_positions_3d = densified.ordered_positions.to(dtype=points.dtype, device=points.device)
            fit_record = analyze_recovered_loop(
                region_id, loop_index, densified.ordered_ids, ordered_positions_3d,
                evidence_positions, evidence_ids, representative_positions, member_ids, accepted_edges,
            )
            fit_record["topology_status"] = result.status
            fit_record["geometry_planarity_class"] = result.geometry.planarity.planarity_class if result.geometry and result.geometry.planarity else None
            fit_record["geometry_crossing_check"] = result.geometry.crossing_check if result.geometry else None
            fit_record["geometry_proper_crossing_count"] = result.geometry.proper_crossing_count if result.geometry else None
            fit_record["densified_extension_count"] = densified.extension_count
            fit_record["seed_vertex_count"] = densified.seed_vertex_count
            loops.append(fit_record)
        region_records.append({"region_id": region_id, "loops": loops})

    return {"label": label, "region_count": len(region_records), "regions": region_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val71/full_evidence_boundary_topology_report.json"))
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
