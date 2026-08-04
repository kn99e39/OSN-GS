"""Worklog 41 (task section 9/10): gate-level physical-termination extraction
waterfall on real snapshots, with a rigorous R1 vs R2 separation.

Worklog 40 reported R1 and R2 summed together ("insufficient candidates").
This reproduces `extract_support_termination_candidates`'s gate sequence node
by node so the two can be told apart:

  R1 = the region never reaches a physical perimeter proxy at all
  R2 = the region DOES reach a perimeter proxy, but the extraction gates
       reject the nodes sitting on it

The perimeter proxy is evidence-based (no ground truth): a region member is
"perimeter-like" when its own accepted-neighbour directions leave a genuine
angular gap, i.e. it is not surrounded by its own region.

Offline diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch


def _unit(vector):
    return vector / vector.norm().clamp_min(1e-12)


def trace(construction, positions, ids, canonical_frames, resolved_candidate_scale, sectors: int = 8) -> dict:
    from osn_gs.surface.torch_boundary_support_termination import _missing_sector_runs

    regions = construction.surface_regions
    count = len(regions.node_region_id)
    index = {item: node for node, item in enumerate(ids)}
    adjacency = {node: [] for node in range(count)}
    for region in regions.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                adjacency[index[left]].append(index[right])
                adjacency[index[right]].append(index[left])

    # Worklog 41 (task section 13, Case C fix): must match production's
    # resolved representative-graph scale, not the raw per-Gaussian
    # footprint -- see torch_visible_surface_construction.py.
    tangent_scales = resolved_candidate_scale
    width = 2 * math.pi / sectors
    generated = {
        h.source_gaussian_id for h in construction.boundary_halfedge_candidates
        if h.boundary_reason == "observed_support_termination"
    }
    reclassified = {
        h.source_gaussian_id: h.boundary_reason
        for h in construction.boundary_halfedge_candidates
        if h.boundary_reason != "observed_support_termination"
    }

    first_failure = Counter()
    perimeter_like_by_region: Counter = Counter()
    generated_by_region: Counter = Counter()

    for source in range(count):
        region_id = regions.node_region_id[source]
        node_id = ids[source]
        if region_id < 0 or regions.node_membership_state[source] not in ("core_member", "consensus_attached"):
            first_failure["not_region_member"] += 1
            continue
        frame = canonical_frames[source] if canonical_frames is not None else None
        if frame is None:
            first_failure["no_canonical_frame"] += 1
            continue

        # Production may generate this node through the full-cloud continuation
        # path before any representative-sector local-neighbor replay is
        # meaningful. Count the authoritative typed production candidate first;
        # otherwise this diagnostic misclassifies continuation-backed physical
        # candidates as sector no-neighbor failures and reports a smaller count
        # than the production boundary waterfall.
        if node_id in generated:
            first_failure["generated_physical_candidate"] += 1
            generated_by_region[region_id] += 1
            continue
        if node_id in reclassified:
            first_failure[reclassified[node_id]] += 1
            continue

        normal = frame.oriented_normal
        axis_u, axis_v = frame.tangent_axis_0, frame.tangent_axis_1
        local = []
        for target in adjacency[source]:
            delta = positions[target] - positions[source]
            tangent = delta - normal * (delta @ normal)
            distance = float(tangent.norm())
            if 1e-8 < distance <= float(tangent_scales[source]) * 4.0:
                local.append(_unit(tangent))

        if len(local) < 2:
            first_failure["no_neighbor_support"] += 1
            continue

        angles = sorted(math.atan2(float(v @ axis_v), float(v @ axis_u)) for v in local)
        gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) for i in range(len(angles))]
        gap = max(gaps)

        # Perimeter proxy: a genuine angular hole in this node's OWN region
        # support means the region stops here -- independent of whether the
        # extraction gates go on to accept it.
        is_perimeter_like = gap >= width * 1.5
        if is_perimeter_like:
            perimeter_like_by_region[region_id] += 1

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
        runs = _missing_sector_runs(occupied, sectors)

        if not is_perimeter_like:
            first_failure["insufficient_geometric_exposure"] += 1
        elif not runs:
            first_failure["histogram_veto_no_missing_sector_run"] += 1
        elif len(local) < 3:
            first_failure["insufficient_termination_evidence"] += 1
        else:
            first_failure["other"] += 1

    # Region-level R1 vs R2.
    region_verdicts = Counter()
    region_rows = []
    closed_regions = {
        c.region_id for c in construction.ordered_boundary_components
        if c.ordering_state == "ordered_closed_loop"
    }
    for region in regions.regions:
        perimeter_like = perimeter_like_by_region.get(region.region_id, 0)
        produced = generated_by_region.get(region.region_id, 0)
        if region.region_id in closed_regions:
            verdict, confidence = "closed", "high"
        elif perimeter_like == 0:
            verdict, confidence = "R1_region_never_reaches_perimeter_proxy", "medium"
        elif produced < 3:
            verdict, confidence = "R2_perimeter_reached_but_extraction_rejected", "medium"
        else:
            verdict, confidence = "R3_or_later", "medium"
        region_verdicts[verdict] += 1
        region_rows.append({
            "region_id": region.region_id,
            "member_count": len(region.member_ids),
            "perimeter_like_nodes": perimeter_like,
            "physical_candidates": produced,
            "verdict": verdict,
            "confidence": confidence,
        })

    region_rows.sort(key=lambda r: -r["member_count"])
    return {
        "first_failure_counts": dict(first_failure),
        "region_verdicts": dict(region_verdicts),
        "top_regions": region_rows[:10],
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state
    from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
    from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput
    from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()

    state = build_frozen_state(args.checkpoint, args.cap)
    continuation_input = ContinuationShellInput(
        full_positions=state.full_points, full_frame=state.full_frame,
        full_intrinsic=state.full_intrinsic, full_opacity=state.full_opacity,
        full_stable_ids=state.full_stable_ids,
        nearest_representative_index=state.nearest_representative_index,
        representative_mean_spacing=state.representative_mean_spacing,
    )
    construction = construct_visible_nurbs_from_gaussians(
        state.rep_points, covariance=state.rep_covariance, stable_ids=state.rep_stable_ids,
        reliability=state.reliability, continuation_input=continuation_input,
        candidate_scale=state.candidate_scale, residual_scale=state.residual_scale,
    )
    canonical = construct_canonical_region_tangent_frames(
        state.rep_points, construction.covariance_frame, construction.reliability,
        construction.surface_regions, ids=state.rep_stable_ids,
    )
    resolved_candidate_scale = (
        state.candidate_scale if state.candidate_scale is not None
        else construction.covariance_frame.tangent_major_scale
    )
    report = trace(construction, state.rep_points, state.rep_stable_ids, canonical, resolved_candidate_scale)
    report["checkpoint"] = str(args.checkpoint)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

