"""Worklog 76: boundary_support_spacing scale-domain comparison and decision.

Worklog 72-74 attributed 68% of dense boundary-support half-line failures to
"no continuation candidate within the local scale", where that scale is the
FULL-EVIDENCE sampling spacing while the objects being connected are the much
sparser BOUNDARY-SUPPORT CANDIDATES (measured at 2.08-3.72x full-evidence
spacing). This script decides whether that is a units error worth correcting
in production, comparing exactly three defensible estimators:

  A `full_evidence_spacing`           -- current production, unchanged baseline
  B `region_boundary_support_spacing` -- robust region-level candidate spacing
  C `local_boundary_support_spacing`  -- robust per-candidate local spacing

The 2.5x distance multiplier and the 0.1x ambiguity tolerance are FIXED across
all three modes and are never tuned per mode -- the question is which scale
DOMAIN is correct, not which constant produces loops. The connectivity
certificate itself (stage order, predicates, mutuality) is untouched.

A scale is acceptable only if it restores local continuity WITHOUT branch
explosion, unsupported gap bridging, proper crossings, or worse containment;
every one of those is measured here, and `measure_edge_support_occupancy`
discloses edges that span observed empty space rather than assuming they do
not exist.

Region formation, ownership gating, representative membership, candidate
admission, and the covariance_normal path are all unchanged.
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

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_boundary_support_spacing import (
    SPACING_MODE_FULL_EVIDENCE,
    SPACING_MODE_LOCAL_BOUNDARY_SUPPORT,
    SPACING_MODE_REGION_BOUNDARY_SUPPORT,
    measure_edge_support_occupancy,
    resolve_boundary_support_spacing,
)
from osn_gs.surface.torch_dense_boundary_connectivity_diagnostics import diagnose_dense_boundary_connectivity
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_region_owned_dense_boundary_support import _connect, extract_dense_boundary_support
from osn_gs.surface.torch_single_chart_uv_validity import interior_within_boundary
from osn_gs.surface.torch_nurbs import pca_parameterize_points

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

MODES = (SPACING_MODE_FULL_EVIDENCE, SPACING_MODE_REGION_BOUNDARY_SUPPORT, SPACING_MODE_LOCAL_BOUNDARY_SUPPORT)


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _mutual_edges_from_diagnostics(candidates, connectivity_scale) -> list[tuple[int, int]]:
    """Recover the accepted (mutual) edge list using the SAME staged rule the
    certificate uses, by replaying it read-only -- no new acceptance logic."""

    n = len(candidates)
    if not n:
        return []
    p = torch.tensor([c.position for c in candidates], dtype=torch.float64)
    t = torch.tensor([c.tangent for c in candidates], dtype=torch.float64)
    z = torch.tensor([c.normal for c in candidates], dtype=torch.float64)
    scales = [float(s) for s in connectivity_scale]
    chosen: dict[tuple[int, int], int] = {}
    for i in range(n):
        local_scale = scales[i]
        for sign in (-1, 1):
            valid = []
            for j in range(n):
                if i == j:
                    continue
                delta = p[j] - p[i]
                dist = float(delta.norm())
                if dist > 2.5 * local_scale:
                    continue
                if candidates[i].boundary_reason != candidates[j].boundary_reason:
                    continue
                if abs(float(t[i] @ t[j])) < 0.5:
                    continue
                if abs(float(z[i] @ z[j])) < 0.8:
                    continue
                if sign * float(delta @ t[i]) <= 0:
                    continue
                valid.append((dist, j))
            valid.sort()
            if len(valid) > 1 and abs(valid[1][0] - valid[0][0]) <= 0.1 * local_scale:
                continue
            if valid:
                chosen[(i, sign)] = valid[0][1]
    edges = set()
    for (i, _sign), j in chosen.items():
        if any(v == i for (k, _s), v in chosen.items() if k == j):
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def _loop_containment(component, candidates, evidence_positions, evidence_ids) -> dict | None:
    by_id = {c.stable_id: c for c in candidates}
    loop_positions = torch.tensor(
        [by_id[sid].position for sid in component.stable_ids if sid in by_id],
        dtype=evidence_positions.dtype, device=evidence_positions.device,
    )
    if int(loop_positions.shape[0]) < 3:
        return None
    combined = torch.cat((loop_positions, evidence_positions), dim=0)
    uv = pca_parameterize_points(combined)
    loop_uv = uv[: int(loop_positions.shape[0])]
    evidence_uv = uv[int(loop_positions.shape[0]) :]
    loop_ids = set(component.stable_ids)
    interior_mask = torch.tensor([sid not in loop_ids for sid in evidence_ids], dtype=torch.bool)
    interior_uv = evidence_uv[interior_mask] if int(interior_mask.numel()) else evidence_uv
    return interior_within_boundary(interior_uv, loop_uv)


def run_mode_on_region(
    mode: str, candidates, evidence_positions, evidence_ids,
    full_evidence_spacing: float, representative_spacing: float | None,
) -> dict:
    device = evidence_positions.device
    positions = (
        torch.tensor([c.position for c in candidates], dtype=torch.float64, device=device)
        if candidates else torch.zeros((0, 3), dtype=torch.float64, device=device)
    )
    resolved = resolve_boundary_support_spacing(
        mode, positions, full_evidence_spacing=full_evidence_spacing, representative_spacing=representative_spacing,
    )
    scale = resolved.per_candidate_scale
    result = _connect(candidates, representative_spacing, scale)
    diagnostics = diagnose_dense_boundary_connectivity(candidates, scale)

    status_counts: dict[str, int] = {}
    for component in result.components:
        status_counts[component.status] = status_counts.get(component.status, 0) + 1
    closed = [c for c in result.components if c.closed]

    edges = _mutual_edges_from_diagnostics(candidates, scale)
    occupancy = measure_edge_support_occupancy(
        edges, positions, evidence_positions.to(torch.float64), full_evidence_spacing=full_evidence_spacing,
    )

    crossings = []
    planarity = []
    containment = []
    for component in closed:
        if component.geometry is not None:
            crossings.append(component.geometry.proper_crossing_count)
            planarity.append(
                component.geometry.planarity.planarity_class if component.geometry.planarity else None
            )
        result_containment = _loop_containment(component, candidates, evidence_positions, evidence_ids)
        if result_containment and result_containment["interior_total_count"]:
            containment.append(
                result_containment["interior_outside_boundary_count"] / result_containment["interior_total_count"]
            )

    stages = diagnostics.get("stages", {})
    outcomes = diagnostics.get("terminal_outcomes", {})
    degrees = {k: stages.get("mutuality", {}).get(k) for k in ("degree_0", "degree_1", "degree_2")}
    return {
        "mode": mode,
        "candidate_count": len(candidates),
        "full_evidence_spacing": resolved.full_evidence_spacing,
        "representative_spacing": resolved.representative_spacing,
        "boundary_support_spacing": resolved.boundary_support_spacing,
        "resolved_scale_stats": {
            "median": float(np.median(scale)) if scale else None,
            "max": float(np.max(scale)) if scale else None,
            "ratio_to_full_evidence": (
                float(np.median(scale) / full_evidence_spacing) if scale and full_evidence_spacing > 0 else None
            ),
        },
        "spacing_diagnostics": resolved.diagnostics,
        "no_candidate_within_local_scale": outcomes.get("no_candidate_within_local_scale", 0),
        "half_line_denominator": diagnostics.get("half_line_denominator", 0),
        "directional_coverage": diagnostics.get("directional_coverage", {}),
        "surviving_edges": {
            stage: stages.get(stage, {}).get("surviving_directional_proposals")
            for stage in ("distance_local_scale", "normal", "tangent", "mutuality")
        },
        "mutual_degree_distribution": degrees,
        "component_status_counts": status_counts,
        "closed_loop_count": len(closed),
        "proper_crossings_total": int(sum(crossings)) if crossings else 0,
        "planarity_classes": planarity,
        "edge_support_occupancy": occupancy,
        "interior_outside_boundary": _percentiles(np.array(containment)) if containment else None,
        "_result": result,
        "_scale": scale,
    }


def compare_region(label: str, points: torch.Tensor, covariance: torch.Tensor, stable_ids: list,
                   representative_spacing: float | None) -> dict:
    normals = extract_covariance_frame(covariance).normal_candidate  # covariance_normal, unchanged
    baseline = extract_dense_boundary_support(points, normals, stable_ids, representative_scale=representative_spacing)
    candidates = baseline.candidates
    if not candidates:
        return {"region": label, "point_count": int(points.shape[0]), "modes": {}, "note": "no_boundary_support_candidates"}

    modes = {}
    for mode in MODES:
        modes[mode] = run_mode_on_region(
            mode, candidates, points, stable_ids, baseline.full_evidence_scale, representative_spacing,
        )
    return {
        "region": label, "point_count": int(points.shape[0]), "candidate_count": len(candidates),
        "modes": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in modes.items()},
        "_modes": modes,
    }


def run_synthetic(scene_name: str) -> dict:
    from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene

    scene = make_gaussian_reliability_scene(scene_name, seed=0)
    points = scene.positions.to(torch.float32)
    covariance = scene.covariances.to(torch.float32)
    labels = scene.group_labels or ("region",) * int(points.shape[0])
    regions: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        regions.setdefault(label, []).append(index)
    reports = []
    for label, indices in sorted(regions.items()):
        if len(indices) < 4:
            continue
        selector = torch.tensor(indices, dtype=torch.long)
        reports.append(compare_region(f"{scene_name}:{label}", points[selector], covariance[selector], indices, None))
    return {"scene": scene_name, "regions": reports}


def run_real(checkpoint: Path, cap: int, device: str, max_regions: int) -> dict:
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
        member_local = [rep_stable_ids.index(sid) for sid in region.member_ids if sid in rep_stable_ids]
        representative_spacing = (
            float(mean_spacing[torch.tensor(member_local, dtype=torch.long, device=mean_spacing.device)].median())
            if member_local else None
        )
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        print(f"  region {region.region_id}: evidence={len(indices)} ...", flush=True)
        reports.append(compare_region(
            f"real:region{region.region_id}", points[selector], covariance[selector],
            [stable_ids[i] for i in indices], representative_spacing,
        ))
        if len(reports) >= max_regions:
            break
    return {"scene": f"real:{checkpoint.name}", "regions": reports}


def aggregate(scenes: list) -> dict:
    totals = {mode: {
        "regions": 0, "candidates": 0, "no_candidate": 0, "half_lines": 0,
        "both": 0, "one": 0, "neither": 0,
        "distance_edges": 0, "normal_edges": 0, "tangent_edges": 0, "mutual_edges": 0,
        "closed": 0, "branch": 0, "open_or_ambiguous": 0,
        "proper_crossings": 0, "unsupported_edges": 0, "total_edges": 0,
        "degree_gt2": 0,
    } for mode in MODES}
    containment = {mode: [] for mode in MODES}
    for scene in scenes:
        for region in scene["regions"]:
            for mode in MODES:
                block = region.get("modes", {}).get(mode)
                if not block:
                    continue
                bucket = totals[mode]
                bucket["regions"] += 1
                bucket["candidates"] += block["candidate_count"]
                bucket["no_candidate"] += block["no_candidate_within_local_scale"]
                bucket["half_lines"] += block["half_line_denominator"]
                coverage = block["directional_coverage"]
                bucket["both"] += coverage.get("both_directions_valid", 0)
                bucket["one"] += coverage.get("one_direction_valid", 0)
                bucket["neither"] += coverage.get("neither_direction_valid", 0)
                edges = block["surviving_edges"]
                bucket["distance_edges"] += edges.get("distance_local_scale") or 0
                bucket["normal_edges"] += edges.get("normal") or 0
                bucket["tangent_edges"] += edges.get("tangent") or 0
                bucket["mutual_edges"] += edges.get("mutuality") or 0
                status = block["component_status_counts"]
                bucket["closed"] += status.get("closed_loop", 0)
                bucket["branch"] += status.get("branch_detected", 0)
                bucket["open_or_ambiguous"] += status.get("open_or_ambiguous", 0)
                bucket["proper_crossings"] += block["proper_crossings_total"]
                occupancy = block["edge_support_occupancy"]
                bucket["unsupported_edges"] += occupancy.get("edges_with_empty_interior_bin", 0)
                bucket["total_edges"] += occupancy.get("edge_count", 0)
                if block["interior_outside_boundary"]:
                    containment[mode].append(block["interior_outside_boundary"]["median"])
    for mode in MODES:
        values = containment[mode]
        totals[mode]["interior_outside_boundary_median"] = float(np.median(values)) if values else None
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--max_real_regions", type=int, default=7)
    parser.add_argument("--synthetic", nargs="+", default=["box_face", "cylinder"])
    parser.add_argument("--skip_real", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val76/boundary_support_spacing.json"))
    args = parser.parse_args()

    scenes = []
    for name in args.synthetic:
        print(f"synthetic {name} ...", flush=True)
        scenes.append(run_synthetic(name))
    if not args.skip_real and (args.checkpoint / "checkpoint.pt").exists():
        print(f"real {args.checkpoint} ...", flush=True)
        scenes.append(run_real(args.checkpoint, args.cap, "cuda", args.max_real_regions))

    clean = [{"scene": s["scene"], "regions": [{k: v for k, v in r.items() if not k.startswith("_")} for r in s["regions"]]} for s in scenes]
    report = {"scenes": clean, "aggregate": aggregate(clean)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
