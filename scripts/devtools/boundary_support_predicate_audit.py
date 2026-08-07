"""Worklog 77: dense boundary-support PREDICATE audit (admission, before connectivity).

Worklog 75 removed the normal source and worklog 76 removed the connectivity
scale as the bottleneck. This script audits what is left: does
`observed_support_termination` actually materialize the observed perimeter?

Two halves:

1. SYNTHETIC, with ground truth derived from the fixture construction itself
   (not from any predicate output):
     * `box_face` -- `_flat_grid(9, .12)`, a 9x9 grid reshaped row-major, so
       the true geometric boundary is exactly {row in {0,8}} u {col in {0,8}}.
     * `cylinder` side -- a (24 angular x 9 height) grid, CLOSED
       circumferentially and open axially, so the true boundary is exactly
       {height index in {0,8}}.
     * `cylinder` caps -- `_flat_grid(7,...)` masked to a disc; a cap point is
       a true boundary point when it lacks a full 4-neighbour ring inside the
       mask.
     * `sphere` -- a closed manifold with NO boundary anywhere: the negative
       control, where the correct answer is zero candidates.
   Reports precision/recall, inspects false negatives / false positives, and
   reports the SPATIAL RUN LENGTH of consecutive missed boundary support along
   the true perimeter (not just aggregate counts).

2. REAL `baseline_compatible@2900` region-owned evidence: every point gets
   exactly one terminal admission outcome, separated into
   `insufficient_local_evidence` / `degenerate_tangent_frame` /
   `insufficient_angular_gap` / `admitted`, with the angular-gap rejections
   further attributed to local-neighbourhood contamination (near-normal
   neighbours whose tangent-plane projection carries no usable azimuth) vs a
   genuinely surrounded neighbourhood. Around accepted candidates it then asks
   whether missing continuity is due to REJECTED observed points lying along
   the local boundary path or a genuine absence of observed evidence.

This is a measurement script: it never creates an edge across a gap, never
relaxes a threshold, and never introduces a geometric fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_boundary_support_spacing import measure_edge_support_occupancy
from osn_gs.surface.torch_dense_boundary_connectivity_diagnostics import diagnose_dense_boundary_connectivity
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_nurbs import pca_parameterize_points
from osn_gs.surface.torch_region_owned_dense_boundary_support import (
    estimate_full_evidence_sampling_scale,
    extract_dense_boundary_support,
)
from osn_gs.surface.torch_single_chart_uv_validity import interior_within_boundary

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

_EPS = 1e-8
NEIGHBORS = 12
MISSING_SECTOR = math.pi


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "max": None, "mean": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "max": float(values.max()), "mean": float(values.mean()),
    }


def admission_trace(points: torch.Tensor, normals: torch.Tensor) -> dict:
    """Replay the admission predicate read-only, assigning EVERY point exactly
    one terminal outcome plus the quantities behind it. Mirrors
    `extract_dense_boundary_support` stage for stage (including worklog 77's
    discretization-bias correction); it never re-decides anything."""

    n = int(points.shape[0])
    scale = estimate_full_evidence_sampling_scale(points)
    if n < 4 or scale <= 0:
        return {"outcomes": {"insufficient_local_evidence": n}, "per_point": [], "full_evidence_scale": scale}

    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    k = min(NEIGHBORS, n - 1)
    near = distances.topk(k, largest=False).indices

    outcomes: dict[str, int] = {}
    per_point = []
    for i in range(n):
        normal = normals[i] / normals[i].norm().clamp_min(_EPS)
        reference = points[near[i, 0]] - points[i]
        reference = reference - normal * (reference @ normal)
        if float(reference.norm()) <= _EPS:
            outcomes["degenerate_tangent_frame"] = outcomes.get("degenerate_tangent_frame", 0) + 1
            per_point.append({"index": i, "outcome": "degenerate_tangent_frame"})
            continue
        reference = reference / reference.norm()
        axis = torch.linalg.cross(normal, reference)
        axis = axis / axis.norm().clamp_min(_EPS)
        delta = points[near[i]] - points[i]
        tangential = delta - normal[None, :] * (delta @ normal)[:, None]

        # Local-neighbourhood contamination: a neighbour displaced mostly ALONG
        # the normal projects to a near-zero tangent vector, whose azimuth is
        # not meaningfully defined; `atan2` still returns a definite angle for
        # it, so it can fabricate directional support.
        tangential_ratio = tangential.norm(dim=1) / delta.norm(dim=1).clamp_min(_EPS)
        contaminating = int((tangential_ratio < 0.5).sum())

        angles = torch.atan2(tangential @ axis, tangential @ reference).remainder(2 * math.pi).sort().values
        gaps = torch.diff(torch.cat((angles, angles[:1] + 2 * math.pi)))
        gap, gap_index = gaps.max(dim=0)
        rest = torch.cat((gaps[:gap_index], gaps[gap_index + 1 :]))
        resolution = float(rest.median()) if int(rest.numel()) else 0.0

        admitted = float(gap) >= MISSING_SECTOR - resolution
        # What the pre-worklog-77 (uncorrected) predicate would have said.
        admitted_uncorrected = float(gap) >= MISSING_SECTOR
        outcome = "admitted" if admitted else "insufficient_angular_gap"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        per_point.append({
            "index": i, "outcome": outcome,
            "gap_over_pi": float(gap) / math.pi,
            "resolution_over_pi": resolution / math.pi,
            "contaminating_neighbors": contaminating,
            "admitted_uncorrected": bool(admitted_uncorrected),
        })
    return {"outcomes": outcomes, "per_point": per_point, "full_evidence_scale": scale}


# --------------------------------------------------------------------------
# Synthetic ground truth (derived from fixture construction, never predicted)
# --------------------------------------------------------------------------


def _grid_boundary_mask(side: int) -> torch.Tensor:
    return torch.tensor([(i // side in (0, side - 1)) or (i % side in (0, side - 1)) for i in range(side * side)])


def _perimeter_run_lengths(boundary_indices: list[int], missed: set[int], positions: torch.Tensor) -> dict:
    """Order the true-boundary points along the perimeter (nearest-neighbour
    walk restricted to true-boundary points) and report the run lengths of
    CONSECUTIVE missed points -- the spatial distribution of missing support,
    not just its count."""

    if not boundary_indices:
        return {"runs": [], "max_run": 0, "run_count": 0}
    subset = positions[torch.tensor(boundary_indices)]
    m = len(boundary_indices)
    d = torch.cdist(subset, subset)
    d.fill_diagonal_(float("inf"))
    order = [0]
    seen = {0}
    while len(order) < m:
        current = order[-1]
        candidates = [(float(d[current, j]), j) for j in range(m) if j not in seen]
        if not candidates:
            break
        seen.add(min(candidates)[1])
        order.append(min(candidates)[1])
    flags = [boundary_indices[j] in missed for j in order]
    runs = []
    run = 0
    for flag in flags + [False]:
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    return {"runs": runs, "max_run": max(runs) if runs else 0, "run_count": len(runs),
            "total_missed": sum(runs), "perimeter_length": m}


def synthetic_report(scene_name: str) -> dict:
    from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene

    scene = make_gaussian_reliability_scene(scene_name, seed=0)
    points = scene.positions.to(torch.float32)
    covariance = scene.covariances.to(torch.float32)
    normals = extract_covariance_frame(covariance).normal_candidate
    labels = scene.group_labels or ("region",) * int(points.shape[0])

    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)

    region_reports = []
    for label, indices in sorted(groups.items()):
        if len(indices) < 4:
            continue
        selector = torch.tensor(indices, dtype=torch.long)
        region_points = points[selector]
        region_normals = normals[selector]
        count = len(indices)

        truth = None
        if scene_name == "box_face":
            truth = _grid_boundary_mask(9)
        elif scene_name == "cylinder" and label == "side":
            truth = torch.tensor([(i % 9 in (0, 8)) for i in range(count)])
        elif scene_name == "sphere":
            truth = torch.zeros(count, dtype=torch.bool)  # closed manifold: no boundary anywhere
        elif scene_name == "cylinder" and label.endswith("_cap"):
            # disc-masked grid: a point is boundary when it has fewer than 4
            # in-plane neighbours at ~grid spacing in the 4 axis directions.
            xy = region_points[:, :2]
            d = torch.cdist(xy, xy)
            d.fill_diagonal_(float("inf"))
            spacing = float(d.min(dim=1).values.median())
            neighbour_count = (d <= spacing * 1.3).sum(dim=1)
            truth = neighbour_count < 4

        trace = admission_trace(region_points, region_normals)
        admitted = {p["index"] for p in trace["per_point"] if p["outcome"] == "admitted"}
        admitted_uncorrected = {p["index"] for p in trace["per_point"] if p.get("admitted_uncorrected")}

        entry = {
            "region": f"{scene_name}:{label}", "point_count": count,
            "outcomes": trace["outcomes"], "admitted": len(admitted),
            "admitted_uncorrected": len(admitted_uncorrected),
        }
        if truth is not None:
            truth_indices = [i for i in range(count) if bool(truth[i])]
            for name, admitted_set in (("corrected", admitted), ("uncorrected", admitted_uncorrected)):
                tp = len([i for i in truth_indices if i in admitted_set])
                fp = len([i for i in admitted_set if not bool(truth[i])])
                fn = len([i for i in truth_indices if i not in admitted_set])
                entry[name] = {
                    "true_boundary": len(truth_indices), "tp": tp, "fp": fp, "fn": fn,
                    "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                    "missed_runs": _perimeter_run_lengths(
                        truth_indices, {i for i in truth_indices if i not in admitted_set}, region_points,
                    ),
                }
            false_negatives = [p for p in trace["per_point"] if p["index"] in set(truth_indices) and p["outcome"] != "admitted"]
            false_positives = [p for p in trace["per_point"] if p["index"] in admitted and not bool(truth[p["index"]])]
            entry["false_negative_inspection"] = {
                "count": len(false_negatives),
                "gap_over_pi": _percentiles(np.array([p["gap_over_pi"] for p in false_negatives])) if false_negatives else None,
                "contaminating_neighbors": _percentiles(np.array([p["contaminating_neighbors"] for p in false_negatives])) if false_negatives else None,
            }
            entry["false_positive_inspection"] = {
                "count": len(false_positives),
                "gap_over_pi": _percentiles(np.array([p["gap_over_pi"] for p in false_positives])) if false_positives else None,
            }
        region_reports.append(entry)
    return {"scene": scene_name, "regions": region_reports}


# --------------------------------------------------------------------------
# Real regions
# --------------------------------------------------------------------------


def _rejected_points_along_paths(points, admitted_indices, rejected_indices, scale) -> dict:
    """For each pair of accepted candidates that are near each other but have
    NO accepted candidate between them, ask whether REJECTED observed points
    lie along the connecting path. Measurement only -- no edge is created."""

    if len(admitted_indices) < 2:
        return {"pairs_examined": 0}
    admitted_positions = points[torch.tensor(admitted_indices, dtype=torch.long)]
    rejected_positions = points[torch.tensor(rejected_indices, dtype=torch.long)] if rejected_indices else points[:0]
    d = torch.cdist(admitted_positions, admitted_positions)
    d.fill_diagonal_(float("inf"))
    nearest = d.min(dim=1).values

    pairs = 0
    with_rejected_on_path = 0
    genuinely_empty = 0
    for i in range(len(admitted_indices)):
        j = int(d[i].argmin())
        a, b = admitted_positions[i], admitted_positions[j]
        length = float((b - a).norm())
        if length <= _EPS:
            continue
        pairs += 1
        if int(rejected_positions.shape[0]) == 0:
            genuinely_empty += 1
            continue
        unit = (b - a) / length
        offset = rejected_positions - a[None, :]
        projection = offset @ unit
        perpendicular = (offset - projection[:, None] * unit[None, :]).norm(dim=-1)
        on_path = ((projection > 0.2 * length) & (projection < 0.8 * length) & (perpendicular <= scale)).sum()
        if int(on_path) > 0:
            with_rejected_on_path += 1
        else:
            genuinely_empty += 1
    return {
        "pairs_examined": pairs,
        "nearest_accepted_gap_over_scale": _percentiles((nearest / max(scale, _EPS)).detach().cpu().numpy()),
        "pairs_with_rejected_points_on_path": with_rejected_on_path,
        "pairs_with_genuinely_no_observed_evidence": genuinely_empty,
        "measurement_only": True,
    }


def connectivity_report(points, normals, stable_ids, representative_scale) -> dict:
    """Run the EXISTING, unchanged dense-boundary connectivity path."""

    result = extract_dense_boundary_support(points, normals, stable_ids, representative_scale=representative_scale)
    candidates = result.candidates
    diagnostics = diagnose_dense_boundary_connectivity(candidates)
    status: dict[str, int] = {}
    for component in result.components:
        status[component.status] = status.get(component.status, 0) + 1
    closed = [c for c in result.components if c.closed]

    positions = torch.tensor([c.position for c in candidates], dtype=torch.float64, device=points.device) if candidates else torch.zeros((0, 3), dtype=torch.float64, device=points.device)
    stages = diagnostics.get("stages", {})
    mutual = stages.get("mutuality", {})
    edges = []
    index_by_id = {c.stable_id: i for i, c in enumerate(candidates)}
    for component in result.components:
        members = [index_by_id[s] for s in component.stable_ids if s in index_by_id]
        for a in members:
            for b in members:
                if a < b:
                    edges.append((a, b))
    occupancy = measure_edge_support_occupancy(
        edges[:2000], positions, points.to(torch.float64), full_evidence_spacing=result.full_evidence_scale,
    )

    crossings = sum(c.geometry.proper_crossing_count for c in closed if c.geometry is not None)
    containment = []
    for component in closed:
        loop = torch.tensor([candidates[index_by_id[s]].position for s in component.stable_ids if s in index_by_id],
                            dtype=points.dtype, device=points.device)
        if int(loop.shape[0]) < 3:
            continue
        combined = torch.cat((loop, points), dim=0)
        uv = pca_parameterize_points(combined)
        loop_ids = set(component.stable_ids)
        interior_mask = torch.tensor([s not in loop_ids for s in stable_ids], dtype=torch.bool)
        interior_uv = uv[int(loop.shape[0]):][interior_mask]
        report = interior_within_boundary(interior_uv, uv[: int(loop.shape[0])])
        if report["interior_total_count"]:
            containment.append(report["interior_outside_boundary_count"] / report["interior_total_count"])

    return {
        "candidate_count": len(candidates),
        "no_candidate_within_local_scale": diagnostics.get("terminal_outcomes", {}).get("no_candidate_within_local_scale", 0),
        "half_line_denominator": diagnostics.get("half_line_denominator", 0),
        "directional_coverage": diagnostics.get("directional_coverage", {}),
        "surviving_edges": {k: stages.get(k, {}).get("surviving_directional_proposals") for k in ("distance_local_scale", "normal", "tangent", "mutuality")},
        "component_status_counts": status,
        "closed_loop_count": len(closed),
        "proper_crossings": int(crossings),
        "edge_support_occupancy": occupancy,
        "interior_outside_boundary": _percentiles(np.array(containment)) if containment else None,
    }


def real_report(checkpoint: Path, cap: int, device: str, max_regions: int) -> dict:
    from osn_gs.gaussian.torch_model import TorchGaussianModel

    payload = torch.load(checkpoint / "checkpoint.pt", map_location=device, weights_only=False)
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
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    points = model.get_xyz.detach()
    stable_ids = list(range(int(points.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            points, covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    regions = bundle.construction.surface_regions
    rep_stable_ids = bundle.representative_stable_ids
    cluster_ids = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _ = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster_ids)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)
    mean_spacing = bundle.evidence.mean_spacing

    reports = []
    for region in regions.regions:
        indices = owned.get(region.region_id, [])
        if len(indices) < 4:
            continue
        member_local = [rep_stable_ids.index(s) for s in region.member_ids if s in rep_stable_ids]
        representative_scale = (
            float(mean_spacing[torch.tensor(member_local, dtype=torch.long, device=mean_spacing.device)].median())
            if member_local else None
        )
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        region_points = points[selector]
        region_normals = extract_covariance_frame(covariance[selector]).normal_candidate
        print(f"  region {region.region_id}: evidence={len(indices)} ...", flush=True)

        trace = admission_trace(region_points, region_normals)
        admitted = [p["index"] for p in trace["per_point"] if p["outcome"] == "admitted"]
        rejected = [p["index"] for p in trace["per_point"] if p["outcome"] != "admitted"]
        gap_rejected = [p for p in trace["per_point"] if p["outcome"] == "insufficient_angular_gap"]
        contaminated = [p for p in gap_rejected if p["contaminating_neighbors"] > 0]

        reports.append({
            "region": f"real:region{region.region_id}",
            "point_count": len(indices),
            "outcomes": trace["outcomes"],
            "admitted_uncorrected": sum(1 for p in trace["per_point"] if p.get("admitted_uncorrected")),
            "angular_gap_rejection_attribution": {
                "total": len(gap_rejected),
                "with_contaminating_neighbors": len(contaminated),
                "contaminating_neighbor_count": _percentiles(np.array([p["contaminating_neighbors"] for p in gap_rejected])) if gap_rejected else None,
                "gap_over_pi": _percentiles(np.array([p["gap_over_pi"] for p in gap_rejected])) if gap_rejected else None,
            },
            "continuity_attribution": _rejected_points_along_paths(
                region_points, admitted, rejected, trace["full_evidence_scale"],
            ),
            "connectivity": connectivity_report(
                region_points, region_normals, [stable_ids[i] for i in indices], representative_scale,
            ),
        })
        if len(reports) >= max_regions:
            break
    return {"scene": f"real:{checkpoint.name}", "regions": reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--max_real_regions", type=int, default=7)
    parser.add_argument("--synthetic", nargs="+", default=["box_face", "cylinder", "sphere"])
    parser.add_argument("--skip_real", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val77/predicate_audit.json"))
    args = parser.parse_args()

    scenes = []
    for name in args.synthetic:
        print(f"synthetic {name} ...", flush=True)
        scenes.append(synthetic_report(name))
    if not args.skip_real and (args.checkpoint / "checkpoint.pt").exists():
        print(f"real {args.checkpoint} ...", flush=True)
        scenes.append(real_report(args.checkpoint, args.cap, "cuda", args.max_real_regions))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"scenes": scenes}, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
