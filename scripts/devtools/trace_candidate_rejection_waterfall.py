"""Worklog 38 (task section 10): full `extract_support_termination_candidates`
rejection waterfall.

Instruments every gate in the candidate-extraction path, per potential
boundary representative, with measured value / threshold / signed margin and
a first-failure + all-failure classification. Offline diagnostic only --
reproduces the production gate order exactly without modifying it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from osn_gs.surface.torch_full_cloud_continuation_shell import STATE_NO_GAP, STATE_OBSERVED_TERMINATION

# Rejection reason taxonomy (task section 10).
NOT_REGION_MEMBER = "not_region_member"
NO_CANONICAL_FRAME = "no_canonical_frame"
NO_EXPOSED_DIRECTION = "no_exposed_direction"
CONTINUATION_SUPPORTED = "continuation_supported"
INSUFFICIENT_TERMINATION_EVIDENCE = "insufficient_termination_evidence"
INSUFFICIENT_ANGULAR_GAP = "insufficient_angular_gap"
DEGENERATE_BOUNDARY_TANGENT = "degenerate_boundary_tangent"
RELIABILITY_FRONTIER = "reliability_frontier"
SAMPLING_GAP = "sampling_gap"
GENERATED_GENUINE = "generated_genuine_termination"
GENERATED_NONPHYSICAL = "generated_nonphysical_state"


def _unit(vector):
    return vector / vector.norm().clamp_min(1e-12)


def waterfall(construction, positions, ids, canonical_frames, continuation, tangent_scales, sectors: int = 8) -> dict:
    """Replay the exact gate sequence of `extract_support_termination_candidates`."""
    region_result = construction.surface_regions
    count = len(region_result.node_region_id)
    index = {item: node for node, item in enumerate(ids)}
    adjacency = {node: [] for node in range(count)}
    for region in region_result.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                adjacency[index[left]].append(index[right])
                adjacency[index[right]].append(index[left])

    width = 2 * math.pi / sectors
    rows = []
    first_failure = Counter()

    for source in range(count):
        node_id = ids[source]
        region_id = region_result.node_region_id[source]
        membership = region_result.node_membership_state[source]
        row = {
            "stable_id": node_id,
            "source_region_id": int(region_id),
            "region_membership_state": membership,
            "region_internal_degree": len(adjacency[source]),
        }

        # Gate 1: region membership
        if region_id < 0 or membership not in ("core_member", "consensus_attached"):
            row["first_failure"] = NOT_REGION_MEMBER
            first_failure[NOT_REGION_MEMBER] += 1
            rows.append(row)
            continue

        # Gate 2: canonical frame availability
        canonical = canonical_frames[source] if canonical_frames is not None else None
        if canonical is None:
            row["first_failure"] = NO_CANONICAL_FRAME
            first_failure[NO_CANONICAL_FRAME] += 1
            rows.append(row)
            continue

        # Gate 3: full-cloud continuation query path (production prefers this)
        query = continuation.get(node_id) if continuation is not None else None
        if query is not None:
            row["path"] = "continuation"
            row["continuation_state"] = query.state
            row["gap_width_degrees"] = query.gap_width_degrees
            row["same_mode_support_count"] = query.same_mode_support_count
            if query.state == STATE_NO_GAP or query.outward_direction is None:
                row["first_failure"] = CONTINUATION_SUPPORTED
                first_failure[CONTINUATION_SUPPORTED] += 1
                rows.append(row)
                continue
            normal = canonical.oriented_normal
            boundary_tangent = normal.cross(query.outward_direction, dim=0)
            if float(boundary_tangent.norm()) <= 1e-8:
                row["first_failure"] = DEGENERATE_BOUNDARY_TANGENT
                first_failure[DEGENERATE_BOUNDARY_TANGENT] += 1
                rows.append(row)
                continue
            if query.state == STATE_OBSERVED_TERMINATION:
                row["first_failure"] = GENERATED_GENUINE
                first_failure[GENERATED_GENUINE] += 1
            elif query.state == "reliability_frontier":
                row["first_failure"] = RELIABILITY_FRONTIER
                first_failure[RELIABILITY_FRONTIER] += 1
            elif query.state == "unresolved_sampling_gap":
                row["first_failure"] = SAMPLING_GAP
                first_failure[SAMPLING_GAP] += 1
            else:
                row["first_failure"] = GENERATED_NONPHYSICAL
                first_failure[GENERATED_NONPHYSICAL] += 1
            rows.append(row)
            continue

        # Gate 4+: representative-only sector histogram path
        row["path"] = "sector_histogram"
        normal = canonical.oriented_normal
        axis_u, axis_v = canonical.tangent_axis_0, canonical.tangent_axis_1
        local = []
        for target in adjacency[source]:
            delta = positions[target] - positions[source]
            tangent = delta - normal * (delta @ normal)
            distance = float(tangent.norm())
            if 1e-8 < distance <= float(tangent_scales[source]) * 4.0:
                local.append(_unit(tangent))
        row["local_neighbor_count"] = len(local)
        row["local_neighbor_threshold"] = 2
        row["local_neighbor_margin"] = len(local) - 2
        if len(local) < 2:
            row["first_failure"] = INSUFFICIENT_TERMINATION_EVIDENCE
            first_failure[INSUFFICIENT_TERMINATION_EVIDENCE] += 1
            rows.append(row)
            continue

        occupied = set()
        for vector in local:
            angle = math.atan2(float(vector @ axis_v), float(vector @ axis_u))
            value = (angle + math.pi) / width
            primary = int(math.floor(value)) % sectors
            occupied.add(primary)
            fraction = value - math.floor(value)
            if fraction < 0.15:
                occupied.add((primary - 1) % sectors)
            if fraction > 0.85:
                occupied.add((primary + 1) % sectors)

        # largest geometric gap
        angles = sorted(math.atan2(float(v @ axis_v), float(v @ axis_u)) for v in local)
        gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) for i in range(len(angles))]
        gap = max(gaps) if gaps else 0.0
        row["largest_gap_radians"] = gap
        row["gap_threshold_radians"] = width * 1.5
        row["gap_margin_radians"] = gap - width * 1.5
        row["occupied_sector_count"] = len(occupied)

        has_runs = len(occupied) != sectors
        if not has_runs or gap < width * 1.5:
            row["first_failure"] = INSUFFICIENT_ANGULAR_GAP
            first_failure[INSUFFICIENT_ANGULAR_GAP] += 1
            rows.append(row)
            continue

        if len(local) >= 3:
            row["first_failure"] = GENERATED_GENUINE
            first_failure[GENERATED_GENUINE] += 1
        else:
            row["first_failure"] = SAMPLING_GAP
            first_failure[SAMPLING_GAP] += 1
        rows.append(row)

    return {"first_failure_counts": dict(first_failure), "rows": rows}


def _build_for_scene(scene_name: str, seed: int = 0):
    from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
    from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
    from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

    scene = make_gaussian_reliability_scene(scene_name, seed=seed)
    ids = tuple(range(scene.positions.shape[0]))
    construction = construct_visible_nurbs_from_gaussians(
        scene.positions, covariance=scene.covariances, stable_ids=ids,
    )
    canonical_frames = construct_canonical_region_tangent_frames(
        scene.positions, construction.covariance_frame, construction.reliability,
        construction.surface_regions, ids=ids,
    )
    return scene, construction, ids, canonical_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="cylinder")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene, construction, ids, canonical_frames = _build_for_scene(args.scene, args.seed)
    result = waterfall(
        construction, scene.positions, ids, canonical_frames, None,
        construction.covariance_frame.equivalent_tangent_scale,
    )
    per_region = {}
    for row in result["rows"]:
        rid = row["source_region_id"]
        per_region.setdefault(rid, Counter())[row["first_failure"]] += 1
    print(json.dumps({
        "scene": args.scene,
        "first_failure_counts": result["first_failure_counts"],
        "per_region_first_failure": {str(k): dict(v) for k, v in sorted(per_region.items())},
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
