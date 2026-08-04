"""Worklog 40 (task section 12/13): real 3k/5k/10k boundary waterfall and
per-region R1-R6 classification.

Runs the production path on a frozen real checkpoint and reports, per region:
where candidate generation stopped, how many compatible directed edges the
region's candidates actually formed, and which of R1-R6 explains the missing
closed loop. Offline diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from osn_gs.surface.torch_directed_boundary_ordering import (
    _build_accepted_adjacency,
    _compatible_directed_edges,
)


def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _norm(a): return max(sum(x * x for x in a) ** .5, 1e-12)


def analyze(construction, rep_points, rep_stable_ids) -> dict:
    regions = construction.surface_regions
    genuine = [
        h for h in construction.boundary_halfedge_candidates
        if h.boundary_reason == "observed_support_termination"
    ]
    nonphysical = Counter(
        h.boundary_reason for h in construction.boundary_halfedge_candidates
        if h.boundary_reason != "observed_support_termination"
    )
    accepted_pairs = {frozenset(p) for p in construction.accepted_local_topology}
    accepted_adjacency = _build_accepted_adjacency(construction.accepted_local_topology)

    by_region: dict[int, list] = {}
    for h in genuine:
        by_region.setdefault(h.source_region_id, []).append(h)

    closed_regions = {
        c.region_id for c in construction.ordered_boundary_components
        if c.ordering_state == "ordered_closed_loop"
    }

    index = {sid: i for i, sid in enumerate(rep_stable_ids)}
    classification = Counter()
    region_rows = []

    for region in regions.regions:
        members = region.member_ids
        candidates = by_region.get(region.region_id, [])
        member_count = len(members)

        if member_count >= 2:
            idx = torch.tensor([index[s] for s in members if s in index])
            pts = rep_points[idx]
            diameter = float(torch.cdist(pts, pts).max())
        else:
            diameter = 0.0

        compat_edges = 0
        zero_out = 0
        if len(candidates) >= 2:
            nearest = [
                _norm(_sub(s.world_position, t.world_position))
                for s in candidates for t in candidates
                if s.half_edge_id != t.half_edge_id
            ]
            local_spacing = sorted(nearest)[len(nearest) // 2]
            candidate_ids = frozenset(c.source_gaussian_id for c in candidates)
            edges = _compatible_directed_edges(
                candidates, accepted_pairs, local_spacing, accepted_adjacency, candidate_ids,
            )
            compat_edges = len(edges)
            out_degree = Counter(k[0] for k in edges)
            zero_out = sum(1 for c in candidates if out_degree.get(c.half_edge_id, 0) == 0)

        n = len(candidates)
        if region.region_id in closed_regions:
            verdict = "closed"
        elif n < 3:
            verdict = "R1_or_R2_insufficient_candidates"
        elif compat_edges < n:
            verdict = "R3_compatibility_insufficient"
        elif zero_out > 0:
            verdict = "R3_compatibility_insufficient"
        else:
            verdict = "R4_ordering_failed"
        classification[verdict] += 1

        region_rows.append({
            "region_id": region.region_id,
            "member_count": member_count,
            "spatial_diameter": round(diameter, 4),
            "genuine_candidates": n,
            "compatible_directed_edges": compat_edges,
            "zero_out_degree_candidates": zero_out,
            "verdict": verdict,
        })

    region_rows.sort(key=lambda r: -r["member_count"])
    return {
        "region_count": len(regions.regions),
        "genuine_candidate_total": len(genuine),
        "nonphysical_candidate_counts": dict(nonphysical),
        "closed_component_count": len(closed_regions),
        "verdict_counts": dict(classification),
        "top_regions": region_rows[:12],
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state
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
    report = analyze(construction, state.rep_points, state.rep_stable_ids)
    report["checkpoint"] = str(args.checkpoint)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
