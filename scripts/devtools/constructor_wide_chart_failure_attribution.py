"""Worklog 79: one constructor-wide pass over all seven real regions.

Worklog 78 restored the physical-termination / parametric-chart-frontier
distinction and left baseline_compatible@2900 at 5 materialized charts from 7
regions, 1 `valid_supported` and 4 strongly `extrapolative`. This script does
NOT run another one-factor ablation: it evaluates chart topology, chart
domain, UV parameterization, and NURBS fitting TOGETHER for every region and
attributes each outcome to a single dominant constructor-level cause.

Attribution vocabulary (one per materialized chart):
  * `boundary_chart_extent_mismatch` -- the chart domain does not cover the
    evidence it is fit to (fails BEFORE parameterization).
  * `single_chart_domain_invalid`    -- domain covers the evidence but the
    region is not one regular chart.
  * `uv_parameterization_distortion` -- the UV map itself collapses/folds.
  * `nurbs_fit_model_mismatch`       -- domain and UV are sound; the fitted
    surface still misses held-out evidence.
  * `mixed`, `valid_supported`.

Regions with no eligible chart are separately classified as genuinely
unsupported/ambiguous versus requiring more than one legitimate chart, by
reading the region's OWN accepted-edge graph (pendant/bridge structure vs
multiple disjoint cycles). No closure is forced and no partition is invented.

Everything downstream of region formation is read-only here; covariance_normal,
full_evidence_spacing, the worklog 77 correction, the dense connectivity
certificate, region formation, ownership, and photometric training are all
untouched.
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
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_single_chart_uv_validity import (
    neighborhood_preservation,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)

BASE_GRID = 6          # worklog 68: not tuned here
SAMPLE_RESOLUTION = 24
HOLDOUT_K = 4          # worklog 68 checkerboard convention
EXTRAPOLATION_BOUND = 4.0  # worklog 66 convention, unchanged


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


def _errors(evidence: torch.Tensor, surface, scale: float) -> dict:
    if int(evidence.shape[0]) == 0:
        return {"surface_to_evidence": _pct(np.array([])), "point_to_surface": _pct(np.array([])), "count": 0}
    pts = _sample(surface)[0]
    fwd = torch.cdist(evidence, pts).min(dim=1).values.detach().cpu().numpy() / scale
    bwd = torch.cdist(pts, evidence).min(dim=1).values.detach().cpu().numpy() / scale
    return {"point_to_surface": _pct(fwd), "surface_to_evidence": _pct(bwd), "count": int(evidence.shape[0])}


def _in_polygon(points_uv, polygon) -> np.ndarray:
    poly = list(polygon)
    n = len(poly)
    out = []
    for x, y in points_uv:
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        out.append(inside)
    return np.array(out, dtype=bool)


def _extent(points: torch.Tensor) -> float:
    """Diameter proxy: max pairwise distance via bounding sphere of the PCA
    extremes -- cheap and orientation-free."""
    if int(points.shape[0]) < 2:
        return 0.0
    centered = points - points.mean(dim=0, keepdim=True)
    return float(centered.norm(dim=1).max() * 2.0)


def _cycle_structure(nodes, adjacency) -> dict:
    """Read the region's OWN accepted-edge graph: pendant/bridge structure
    (genuinely open) vs multiple disjoint cycles (needs >1 chart)."""
    degrees = {n: len(adjacency[n]) for n in nodes}
    pendants = [n for n, d in degrees.items() if d <= 1]
    # 2-core: repeatedly strip degree<=1 nodes; what remains carries all cycles.
    core = dict(adjacency)
    changed = True
    core_nodes = set(nodes)
    while changed:
        changed = False
        for n in list(core_nodes):
            if len({x for x in core[n] if x in core_nodes}) <= 1:
                core_nodes.discard(n)
                changed = True
    # connected components of the 2-core = independent cycle carriers
    seen = set()
    components = []
    for start in sorted(core_nodes, key=str):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend([y for y in core[x] if y in core_nodes and y not in seen])
        components.append(sorted(comp, key=str))
    edge_count = sum(len({y for y in core[x] if y in core_nodes}) for x in core_nodes) // 2
    return {
        "node_count": len(nodes),
        "degree_min": min(degrees.values()) if degrees else 0,
        "degree_max": max(degrees.values()) if degrees else 0,
        "pendant_nodes": len(pendants),
        "two_core_size": len(core_nodes),
        "two_core_components": len(components),
        "two_core_edges": edge_count,
        "cyclomatic_number": edge_count - len(core_nodes) + len(components) if core_nodes else 0,
    }


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
    materialized = {i.input.source_region_id: i for i in construction.materialized_parametric_chart_surfaces}

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
        evidence = points[torch.tensor(indices, dtype=torch.long, device=points.device)] if indices else points[:0]
        chart = chart_by_region.get(region_id)
        members = [s for s in region.member_ids if s in rep_index]
        adjacency = {s: set() for s in members}
        for a, b in region.internal_accepted_edge_ids:
            if a in adjacency and b in adjacency:
                adjacency[a].add(b)
                adjacency[b].add(a)

        row = {
            "region": region_id,
            "representative_members": len(members),
            "owned_full_evidence": len(indices),
            "evidence_to_member_ratio": (len(indices) / max(len(members), 1)),
            "chart_status": chart.status if chart else "no_chart_record",
            "chart_reason": chart.reason if chart else "",
            "chart_node_count": len(chart.ordered_node_ids) if chart else 0,
            "chart_segment_kinds": chart.segment_kind_counts() if chart else {},
            "accepted_topology_structure": _cycle_structure(members, adjacency),
        }

        item = materialized.get(region_id)
        if item is None or item.surface is None:
            structure = row["accepted_topology_structure"]
            if structure["two_core_components"] >= 2:
                row["no_chart_classification"] = "requires_more_than_one_chart"
            elif structure["pendant_nodes"] > 0 or structure["two_core_size"] < 3:
                row["no_chart_classification"] = "genuinely_open_or_unsupported_topology"
            else:
                row["no_chart_classification"] = "ambiguous_branching_topology"
            rows.append(row)
            continue

        boundary = item.input.ordered_boundary_points
        scale = _median_nn(evidence) if int(evidence.shape[0]) >= 2 else 1e-6

        # ---- chart domain vs owned evidence (BEFORE parameterization) ----
        boundary_extent = _extent(boundary)
        evidence_extent = _extent(evidence)
        frame = frame_by_region.get(region_id)
        inside_fraction = None
        if frame is not None and int(evidence.shape[0]) and int(boundary.shape[0]) >= 3:
            origin = rep_pos[rep_index[frame.gaussian_id]]
            au, av = frame.tangent_axis_0, frame.tangent_axis_1
            b_off = boundary - origin
            e_off = evidence - origin
            poly = [(float(p @ au), float(p @ av)) for p in b_off]
            ev_uv = [(float(p @ au), float(p @ av)) for p in e_off]
            inside = _in_polygon(ev_uv, poly)
            inside_fraction = float(inside.mean())
        row["chart_domain"] = {
            "boundary_extent": boundary_extent,
            "owned_evidence_extent": evidence_extent,
            "boundary_over_evidence_extent": (boundary_extent / evidence_extent) if evidence_extent > 0 else None,
            "evidence_inside_chart_domain_fraction": inside_fraction,
            "evidence_outside_chart_domain_fraction": (1.0 - inside_fraction) if inside_fraction is not None else None,
        }

        # ---- UV parameterization (DURING parameterization) ----
        uv_ev = pca_parameterize_points(evidence) if int(evidence.shape[0]) >= 4 else None
        uv_block = {}
        if uv_ev is not None:
            dup = uv_duplicate_diagnostics(uv_ev)
            nb = neighborhood_preservation(evidence, uv_ev, k=8)
            tri = uv_triangulation_diagnostics(evidence, uv_ev)
            uv_block = {
                "uv_near_collision_count": dup["uv_near_collision_count"],
                "neighborhood_preservation_mean": nb["neighborhood_preservation_mean"],
                "triangle_fold_fraction": (tri["triangle_fold_count"] / tri["triangle_total_count"]) if tri["triangle_total_count"] else None,
            }
        row["uv_parameterization"] = uv_block

        # ---- fitting (DURING fitting) ----
        train, held = _holdout(evidence)
        fit_input = torch.cat((boundary, train), dim=0)
        try:
            surface, _ = fit_torch_visible_surface_lsq(
                fit_input, resolution_u=BASE_GRID, resolution_v=BASE_GRID, degree_u=2, degree_v=2,
            )
        except Exception as exc:  # noqa: BLE001
            row["fit"] = {"failed": type(exc).__name__}
            row["attribution"] = "nurbs_fit_model_mismatch"
            rows.append(row)
            continue
        train_err = _errors(train, surface, scale)
        held_err = _errors(held, surface, scale)
        full_err = _errors(evidence, surface, scale)
        _, du, dv, normals = _sample(surface)
        jac = compute_parametric_jacobian_metrics(du, dv, scale=scale)
        fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)
        eval_err = held_err if held_err["count"] else full_err
        p95 = eval_err["surface_to_evidence"]["p95"] or 0.0
        row["fit"] = {
            "train_surface_to_evidence": train_err["surface_to_evidence"],
            "heldout_surface_to_evidence": held_err["surface_to_evidence"],
            "full_surface_to_evidence": full_err["surface_to_evidence"],
            "heldout_point_to_surface": held_err["point_to_surface"],
            "extrapolation_p95": p95,
            "jacobian_near_degenerate_count": jac["near_degenerate_count"],
            "local_fold_fraction": fold["local_fold_fraction"],
            "train_count": train_err["count"], "heldout_count": held_err["count"],
        }

        # ---- single dominant attribution ----
        outside = row["chart_domain"]["evidence_outside_chart_domain_fraction"]
        extent_ratio = row["chart_domain"]["boundary_over_evidence_extent"]
        uv_bad = (
            (uv_block.get("neighborhood_preservation_mean") is not None and uv_block["neighborhood_preservation_mean"] < 0.5)
            or (uv_block.get("uv_near_collision_count") or 0) > 0
        )
        geometry_unsafe = jac["near_degenerate_count"] > 0 or fold["local_fold_fraction"] > 0.01
        if p95 <= EXTRAPOLATION_BOUND and not geometry_unsafe:
            attribution = "valid_supported"
            stage = "none"
        elif outside is not None and outside > 0.5:
            attribution = "boundary_chart_extent_mismatch"
            stage = "before_parameterization"
        elif uv_bad and geometry_unsafe:
            attribution = "mixed"
            stage = "during_parameterization"
        elif uv_bad:
            attribution = "uv_parameterization_distortion"
            stage = "during_parameterization"
        elif extent_ratio is not None and extent_ratio < 0.5:
            attribution = "boundary_chart_extent_mismatch"
            stage = "before_parameterization"
        else:
            attribution = "nurbs_fit_model_mismatch"
            stage = "during_fitting"
        row["attribution"] = attribution
        row["dominant_stage"] = stage
        row["classification"] = "valid_supported" if attribution == "valid_supported" else "extrapolative"
        rows.append(row)

    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val79/constructor_attribution.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)

    print("\n=== SEVEN-REGION CONSTRUCTOR FAILURE MATRIX ===")
    header = f"{'reg':>3} {'mem':>4} {'evid':>5} {'ratio':>7} {'chart':>6} {'out%':>6} {'ext':>6} {'nbp':>5} {'p95':>7} {'attribution':<34} {'stage'}"
    print(header)
    for row in report["regions"]:
        if "attribution" not in row:
            print(f"{row['region']:>3} {row['representative_members']:>4} {row['owned_full_evidence']:>5} "
                  f"{row['evidence_to_member_ratio']:>7.1f} {'none':>6} {'-':>6} {'-':>6} {'-':>5} {'-':>7} "
                  f"{row['no_chart_classification']:<34} no_chart")
            continue
        cd = row["chart_domain"]
        uvb = row["uv_parameterization"]
        print(f"{row['region']:>3} {row['representative_members']:>4} {row['owned_full_evidence']:>5} "
              f"{row['evidence_to_member_ratio']:>7.1f} {row['chart_node_count']:>6} "
              f"{(cd['evidence_outside_chart_domain_fraction'] or 0)*100:>5.1f}% "
              f"{(cd['boundary_over_evidence_extent'] or 0):>6.3f} "
              f"{(uvb.get('neighborhood_preservation_mean') or 0):>5.2f} "
              f"{row['fit']['extrapolation_p95']:>7.2f} {row['attribution']:<34} {row['dominant_stage']}")


if __name__ == "__main__":
    main()
