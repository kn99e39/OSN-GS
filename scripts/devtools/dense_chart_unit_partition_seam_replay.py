"""Worklog 86: partition-seam parametric chart-domain contract replay.

Worklog 85 closed strict physical-perimeter reconstruction: requiring a
chart's ENTIRE boundary to be a closed loop of observed boundary-support
Gaussians left 83% of coherent, chart-scale-assembled evidence with no valid
manifold topology, not because the recovery mechanism was unsound (verified
directly on synthetic data) but because real admitted boundary candidates
fragment into many small closed loops and open paths rather than one
dominant perimeter.

This script replays the corrected contract: physical surface termination and
parametric chart boundary are NOT the same thing. A chart domain may close
via a physical loop alone (unchanged from Worklog 85), OR via one physical
fragment plus an evidence-backed PARTITION SEAM through the unit's own
coherent interior (Worklog 86, `torch_chart_unit_partition_seam.py`) when
physical evidence alone does not close. Two or more disjoint physical
fragments are disclosed unresolved, never guessed at.

    region-owned evidence
    -> worklog 82 micro-components (UNCHANGED)
    -> worklog 83 assembly (UNCHANGED)
    -> worklog 84 coherence audit (UNCHANGED, applied inside worklog 86)
    -> worklog 86 partition-seam chart-domain recovery (NEW)
    -> worklog 79 coverage contract (UNCHANGED, applied inside worklog 86)
    -> PCA-UV (UNCHANGED)
    -> 6x6 NURBS fit (UNCHANGED)
    -> held-out evaluation (UNCHANGED)

Reports evidence-weighted: coherent chart-unit coverage, partitioned
parametric-domain coverage, physical vs partition-seam boundary length/count,
chart-domain coverage, valid_supported/extrapolative/unsafe/unresolved, and
held-out p95, across all 7 real baseline_compatible@2900 regions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_chart_unit_partition_seam import (
    PHYSICAL_ONLY,
    PHYSICAL_PLUS_PARTITION_SEAM,
)
from osn_gs.surface.torch_chart_unit_partition_seam import (
    STATE_MATERIALIZED as DOMAIN_STATE_MATERIALIZED,
)
from osn_gs.surface.torch_chart_unit_partition_seam import materialize_chart_unit_domain
from osn_gs.surface.torch_dense_chart_unit_assembly import build_chart_unit_assembly
from osn_gs.surface.torch_dense_surface_consistency_components import (
    build_dense_surface_consistency_components,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_single_chart_uv_validity import (
    neighborhood_preservation,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)

BASE_GRID = 6
SAMPLE_RESOLUTION = 24
HOLDOUT_K = 4
EXTRAPOLATION_BOUND = 4.0


def _pct(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p95": None, "max": None}
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _median_nn(points: torch.Tensor) -> float:
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    v = float(d.min(dim=1).values.median())
    return v if v > 0 else 1e-6


def _holdout(evidence: torch.Tensor):
    if int(evidence.shape[0]) < 8:
        return evidence, evidence[:0]
    uv = pca_parameterize_points(evidence)
    cu = (uv[:, 0] * HOLDOUT_K).clamp(0, HOLDOUT_K - 1e-6).floor().long()
    cv = (uv[:, 1] * HOLDOUT_K).clamp(0, HOLDOUT_K - 1e-6).floor().long()
    mask = ((cu + cv) % 2) == 0
    if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
        return evidence, evidence[:0]
    return evidence[~mask], evidence[mask]


def _sample(surface, resolution=SAMPLE_RESOLUTION):
    device, dtype = surface.control_grid.device, surface.control_grid.dtype
    g = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(g, g, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    pts, du, dv = surface.evaluate_with_derivatives(uv)
    return pts, du, dv, torch.cross(du, dv, dim=1)


def evaluate_fit(boundary: torch.Tensor, evidence: torch.Tensor, label: str) -> dict:
    """Identical downstream fit chain worklogs 80-85 already used
    (unmodified): UV validity -> 6x6 NURBS -> held-out. Coverage is already
    checked upstream by `materialize_chart_unit_domain`."""
    scale = _median_nn(evidence)
    record: dict = {"label": label}

    if int(evidence.shape[0]) >= 4:
        uv = pca_parameterize_points(evidence)
        dup = uv_duplicate_diagnostics(uv)
        nb = neighborhood_preservation(evidence, uv, k=8)
        tri = uv_triangulation_diagnostics(evidence, uv)
        record["uv_validity"] = {
            "uv_near_collision_count": dup["uv_near_collision_count"],
            "neighborhood_preservation_mean": nb["neighborhood_preservation_mean"],
            "triangle_fold_fraction": (tri["triangle_fold_count"] / tri["triangle_total_count"]) if tri["triangle_total_count"] else None,
        }

    train, held = _holdout(evidence)
    try:
        surface, _ = fit_torch_visible_surface_lsq(
            torch.cat((boundary, train), dim=0), resolution_u=BASE_GRID, resolution_v=BASE_GRID,
            degree_u=2, degree_v=2,
        )
    except Exception as exc:  # noqa: BLE001
        record["classification"] = "fit_failed"
        record["fit_error"] = f"{type(exc).__name__}: {exc}"
        return record

    pts, du, dv, normals = _sample(surface)
    jac = compute_parametric_jacobian_metrics(du, dv, scale=scale)
    fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)

    def err(subset):
        if int(subset.shape[0]) == 0:
            return _pct(np.array([]))
        return _pct(torch.cdist(pts, subset).min(dim=1).values.detach().cpu().numpy() / scale)

    held_err = err(held)
    full_err = err(evidence)
    p95 = (held_err["p95"] if held_err["p95"] is not None else full_err["p95"]) or 0.0
    record.update({
        "heldout_surface_to_evidence": held_err,
        "full_surface_to_evidence": full_err,
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
    from osn_gs.gaussian.torch_model import TorchGaussianModel

    payload = torch.load(checkpoint / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    rest = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest:
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
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    points = model.get_xyz.detach()
    stable_ids = list(range(int(points.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            points, covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction
    regions = construction.surface_regions
    rep_ids = bundle.representative_stable_ids
    rep_pos = points[bundle.representative_indices]
    rep_index = {s: i for i, s in enumerate(rep_ids)}

    frames = construct_canonical_region_tangent_frames(
        rep_pos, construction.covariance_frame, construction.reliability, regions, ids=rep_ids,
    )
    frame_by_region = {}
    for frame in frames:
        if frame is not None and frame.region_id not in frame_by_region:
            frame_by_region[frame.region_id] = frame
    chart_by_region = {c.region_id: c for c in construction.region_parametric_chart_boundaries}

    cluster = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _ = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)

    rows = []
    for region in regions.regions:
        region_id = region.region_id
        indices = owned.get(region_id, [])
        total_evidence = len(indices)
        row = {"region": region_id, "total_evidence": total_evidence}
        if total_evidence < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        evidence_covariance = covariance[selector]
        chart = chart_by_region.get(region_id)
        frame = frame_by_region.get(region_id)

        arc_starts = arc_ends = None
        arc_kinds: list[str] = []
        if chart is not None and chart.ordered_node_ids and frame is not None:
            nodes_with_pos = [n for n in chart.ordered_node_ids if n in rep_index]
            if len(nodes_with_pos) >= 2:
                sparse_positions = torch.stack([rep_pos[rep_index[n]] for n in nodes_with_pos], dim=0)
                kinds = [s.segment_kind for s in chart.segments][: len(nodes_with_pos)]
                n_arc = int(sparse_positions.shape[0])
                arc_starts = sparse_positions
                arc_ends = torch.stack([sparse_positions[(i + 1) % n_arc] for i in range(n_arc)], dim=0)
                arc_kinds = kinds

        # ---- worklog 82 micro-components (UNCHANGED) ------------------------
        consistency = build_dense_surface_consistency_components(
            region_id, evidence, covariance=evidence_covariance,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        if not consistency.components:
            row["assembled_units"] = []
            rows.append(row)
            continue

        micro_components = tuple(c.member_indices for c in consistency.components)
        non_manifold_flags = tuple(c.non_manifold_suspected for c in consistency.components)
        full_evidence_scale = _median_nn(evidence)

        # ---- worklog 83 assembly (UNCHANGED, preserved as-is) ----------------
        assembly = build_chart_unit_assembly(
            region_id, evidence, covariance=evidence_covariance,
            micro_components=micro_components, non_manifold_flags=non_manifold_flags,
            full_evidence_spacing=full_evidence_scale,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )

        units = []
        evidence_coherent = 0
        evidence_domain_recovered = 0  # any materialized domain (physical or seam)
        classification_evidence = {"valid_supported": 0, "extrapolative": 0, "unsafe_geometry": 0, "unresolved": 0}
        boundary_state_counts: dict[str, int] = {}
        physical_segment_total = 0
        partition_seam_segment_total = 0
        physical_only_count = 0
        physical_plus_seam_count = 0

        for unit_idx, unit in enumerate(assembly.chart_units):
            member_local = list(unit.member_indices)
            unit_size = len(member_local)
            unit_evidence = evidence[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            unit_covariance = evidence_covariance[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            unit_stable_ids = [stable_ids[indices[i]] for i in member_local]

            unit_record = {"unit_index": unit_idx, "member_count": unit_size}

            domain = materialize_chart_unit_domain(
                unit_evidence, unit_covariance, unit_stable_ids, unit_evidence,
                arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
            )
            unit_record["coherent"] = domain.coherence.coherent if domain.coherence else None
            unit_record["domain_state"] = domain.state
            unit_record["boundary_composition"] = domain.boundary_composition
            unit_record["physical_segment_count"] = domain.physical_segment_count
            unit_record["partition_seam_segment_count"] = domain.partition_seam_segment_count
            boundary_state_counts[domain.state] = boundary_state_counts.get(domain.state, 0) + 1

            if domain.coherence and domain.coherence.coherent:
                evidence_coherent += unit_size

            if domain.state != DOMAIN_STATE_MATERIALIZED:
                unit_record["classification"] = "unresolved"
                classification_evidence["unresolved"] += unit_size
                units.append(unit_record)
                continue

            evidence_domain_recovered += unit_size
            physical_segment_total += domain.physical_segment_count
            partition_seam_segment_total += domain.partition_seam_segment_count
            if domain.boundary_composition == PHYSICAL_ONLY:
                physical_only_count += 1
            elif domain.boundary_composition == PHYSICAL_PLUS_PARTITION_SEAM:
                physical_plus_seam_count += 1

            fit = evaluate_fit(domain.ordered_positions, unit_evidence, f"region{region_id}_unit{unit_idx}")
            unit_record.update(fit)
            classification = fit.get("classification", "unresolved")
            classification_evidence[classification] = classification_evidence.get(classification, 0) + unit_size
            units.append(unit_record)

        row["assembly"] = {
            "micro_component_count": len(micro_components),
            "chart_unit_count": len(assembly.chart_units),
        }
        row["evidence_fractions"] = {
            "coherent_chart_unit_coverage": evidence_coherent / total_evidence,
            "partitioned_parametric_domain_coverage": evidence_domain_recovered / total_evidence,
            "valid_supported": classification_evidence["valid_supported"] / total_evidence,
            "extrapolative": classification_evidence["extrapolative"] / total_evidence,
            "unsafe_geometry": classification_evidence["unsafe_geometry"] / total_evidence,
            "unresolved": classification_evidence["unresolved"] / total_evidence,
        }
        row["boundary_composition_unit_counts"] = {
            "physical_only": physical_only_count,
            "physical_plus_partition_seam": physical_plus_seam_count,
        }
        row["boundary_segment_totals"] = {
            "physical_segment_count": physical_segment_total,
            "partition_seam_segment_count": partition_seam_segment_total,
        }
        row["domain_state_counts"] = boundary_state_counts
        row["assembled_units"] = units
        rows.append(row)
    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val86/dense_chart_unit_partition_seam_replay.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}\n", flush=True)

    print(f"{'reg':>3} {'evid':>5} {'units':>5} | evid% coherent/domain | evid% valid/extrap/unsafe/unresolved | phys_only/phys+seam units | phys/seam segs")
    for row in report["regions"]:
        if "skip_reason" in row:
            print(f"{row['region']:>3} {row['total_evidence']:>5} skipped: {row['skip_reason']}")
            continue
        asm = row.get("assembly", {})
        ef = row.get("evidence_fractions", {})
        bc = row.get("boundary_composition_unit_counts", {})
        bs = row.get("boundary_segment_totals", {})
        print(
            f"{row['region']:>3} {row['total_evidence']:>5} {asm.get('chart_unit_count', 0):>5} | "
            f"{ef.get('coherent_chart_unit_coverage', 0)*100:>5.1f}/{ef.get('partitioned_parametric_domain_coverage', 0)*100:>5.1f} | "
            f"{ef.get('valid_supported', 0)*100:>5.1f}/{ef.get('extrapolative', 0)*100:>5.1f}/{ef.get('unsafe_geometry', 0)*100:>5.1f}/{ef.get('unresolved', 0)*100:>5.1f} | "
            f"{bc.get('physical_only', 0)}/{bc.get('physical_plus_partition_seam', 0)} | "
            f"{bs.get('physical_segment_count', 0)}/{bs.get('partition_seam_segment_count', 0)}"
        )

    all_units = [u for row in report["regions"] for u in row.get("assembled_units", [])]
    counts: dict[str, int] = {}
    for u in all_units:
        key = u.get("classification", "?")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nALL-UNIT totals across 7 regions: {counts}")

    total_evid = sum(r.get("total_evidence", 0) for r in report["regions"])
    tot = {"coherent": 0.0, "domain": 0.0, "valid": 0.0, "extrap": 0.0, "unsafe": 0.0, "unresolved": 0.0}
    for row in report["regions"]:
        ef = row.get("evidence_fractions")
        if not ef:
            continue
        te = row["total_evidence"]
        tot["coherent"] += ef["coherent_chart_unit_coverage"] * te
        tot["domain"] += ef["partitioned_parametric_domain_coverage"] * te
        tot["valid"] += ef["valid_supported"] * te
        tot["extrap"] += ef["extrapolative"] * te
        tot["unsafe"] += ef["unsafe_geometry"] * te
        tot["unresolved"] += ef["unresolved"] * te
    print(
        f"\nEvidence-weighted totals across all regions (total evidence {total_evid}): "
        f"coherent_chart_unit={tot['coherent']/total_evid*100:.1f}% "
        f"partitioned_domain={tot['domain']/total_evid*100:.1f}% "
        f"valid_supported={tot['valid']/total_evid*100:.1f}% "
        f"extrapolative={tot['extrap']/total_evid*100:.1f}% "
        f"unsafe_geometry={tot['unsafe']/total_evid*100:.1f}% "
        f"unresolved={tot['unresolved']/total_evid*100:.1f}%"
    )
    total_phys_units = sum(r.get("boundary_composition_unit_counts", {}).get("physical_only", 0) for r in report["regions"])
    total_seam_units = sum(r.get("boundary_composition_unit_counts", {}).get("physical_plus_partition_seam", 0) for r in report["regions"])
    total_phys_segs = sum(r.get("boundary_segment_totals", {}).get("physical_segment_count", 0) for r in report["regions"])
    total_seam_segs = sum(r.get("boundary_segment_totals", {}).get("partition_seam_segment_count", 0) for r in report["regions"])
    print(f"boundary composition: physical_only units={total_phys_units} physical_plus_seam units={total_seam_units}")
    print(f"boundary segments: physical={total_phys_segs} partition_seam={total_seam_segs}")


if __name__ == "__main__":
    main()
