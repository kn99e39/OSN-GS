"""Worklog 39 (task section 8/9/10): audit the sector histogram against the
exact geometric angular gap, per node, across every analytic fixture.

Neither gate is modified here. This measures, for every potential boundary
representative, what the smeared fixed-sector histogram decides versus what
the exact sorted circular-angle gap decides, and classifies every
disagreement -- so the choice between them rests on measured behaviour
rather than on either gate's stated intent.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_boundary_support_termination import _missing_sector_runs
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def _unit(vector):
    return vector / vector.norm().clamp_min(1e-12)


def audit_scene(scene_name: str, seed: int = 0, sectors: int = 8) -> dict:
    scene = make_gaussian_reliability_scene(scene_name, seed=seed)
    ids = tuple(range(scene.positions.shape[0]))
    construction = construct_visible_nurbs_from_gaussians(
        scene.positions, covariance=scene.covariances, stable_ids=ids,
    )
    canonical_frames = construct_canonical_region_tangent_frames(
        scene.positions, construction.covariance_frame, construction.reliability,
        construction.surface_regions, ids=ids,
    )
    region_result = construction.surface_regions
    count = len(region_result.node_region_id)
    index = {item: node for node, item in enumerate(ids)}
    adjacency = {node: [] for node in range(count)}
    for region in region_result.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                adjacency[index[left]].append(index[right])
                adjacency[index[right]].append(index[left])

    tangent_scales = construction.covariance_frame.equivalent_tangent_scale
    width = 2 * math.pi / sectors
    disagreement = Counter()
    rows = []

    for source in range(count):
        region_id = region_result.node_region_id[source]
        if region_id < 0 or region_result.node_membership_state[source] not in ("core_member", "consensus_attached"):
            continue
        canonical = canonical_frames[source] if canonical_frames is not None else None
        if canonical is None:
            continue
        normal = canonical.oriented_normal
        axis_u, axis_v = canonical.tangent_axis_0, canonical.tangent_axis_1
        local = []
        for target in adjacency[source]:
            delta = scene.positions[target] - scene.positions[source]
            tangent = delta - normal * (delta @ normal)
            distance = float(tangent.norm())
            if 1e-8 < distance <= float(tangent_scales[source]) * 4.0:
                local.append(_unit(tangent))
        if len(local) < 2:
            continue

        # --- Gate A: smeared fixed-sector histogram (current production) ---
        occupied_raw: set[int] = set()
        occupied_smeared: set[int] = set()
        for vector in local:
            angle = math.atan2(float(vector @ axis_v), float(vector @ axis_u))
            value = (angle + math.pi) / width
            primary = int(math.floor(value)) % sectors
            occupied_raw.add(primary)
            occupied_smeared.add(primary)
            fraction = value - math.floor(value)
            if fraction < 0.15:
                occupied_smeared.add((primary - 1) % sectors)
            if fraction > 0.85:
                occupied_smeared.add((primary + 1) % sectors)
        runs_smeared = _missing_sector_runs(occupied_smeared, sectors)
        runs_raw = _missing_sector_runs(occupied_raw, sectors)

        # --- Gate B: exact sorted circular-angle largest gap ---
        angles = sorted(math.atan2(float(v @ axis_v), float(v @ axis_u)) for v in local)
        gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) for i in range(len(angles))]
        gap = max(gaps)

        histogram_accepts = bool(runs_smeared)
        geometric_accepts = gap >= width * 1.5
        production_accepts = histogram_accepts and geometric_accepts

        if histogram_accepts and geometric_accepts:
            verdict = "both_accept"
        elif not histogram_accepts and not geometric_accepts:
            verdict = "both_reject"
        elif geometric_accepts and not histogram_accepts:
            verdict = "geometric_accepts_histogram_vetoes"
        else:
            verdict = "histogram_accepts_geometric_vetoes"
        disagreement[verdict] += 1

        rows.append({
            "stable_id": ids[source],
            "region_id": int(region_id),
            "neighbor_count": len(local),
            "largest_gap_rad": gap,
            "gap_threshold_rad": width * 1.5,
            "gap_margin_rad": gap - width * 1.5,
            "occupied_raw": len(occupied_raw),
            "occupied_smeared": len(occupied_smeared),
            "runs_raw": len(runs_raw),
            "runs_smeared": len(runs_smeared),
            "verdict": verdict,
            "production_accepts": production_accepts,
        })

    conflicts = [r for r in rows if r["verdict"] == "geometric_accepts_histogram_vetoes"]
    conflicts.sort(key=lambda r: -r["gap_margin_rad"])
    return {
        "scene": scene_name,
        "evaluated_nodes": len(rows),
        "verdict_counts": dict(disagreement),
        "geometric_accepts_histogram_vetoes_examples": conflicts[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=[
        "box_face", "box", "cylinder", "sphere", "thin_slab",
        "box_with_bridge", "box_isolated_floater", "box_isotropic_contamination",
    ])
    args = parser.parse_args()
    for scene_name in args.scenes:
        print(json.dumps(audit_scene(scene_name), indent=2, default=str))


if __name__ == "__main__":
    main()
