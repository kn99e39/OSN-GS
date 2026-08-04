"""Worklog 51: does the RAW full cloud (before representative reduction)
contain a coherent physical-boundary chain across a representative-level
open-chain gap, or does the gap correspond to genuinely sparse/interior raw
evidence too?

For a given region and a straight-line path between two representative-level
chain endpoints, samples points along the path, anchors each sample to the
nearest RAW full-cloud Gaussian, and runs the SAME same-mode + largest-
circular-gap measurement `torch_full_cloud_continuation_shell.py` uses
production-side -- but centered on that raw Gaussian's own frame, using only
OTHER raw full-cloud Gaussians in the same representative-assigned region
(``nearest_representative_index``) as support. No representative reduction
is involved at all; this is the ground-truth raw-cloud measurement the
representative-level candidate is supposed to approximate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from osn_gs.surface.torch_full_cloud_continuation_shell import _largest_circular_gap_from_bins
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions


def _raw_gap_at(region_full_points, region_full_normal, point, *, anchor_radius=0.1, same_mode_radius=0.3, alignment_min=0.75):
    distance_to_point = (region_full_points - point).norm(dim=-1)
    nearest = torch.nonzero(distance_to_point <= anchor_radius, as_tuple=False).reshape(-1)
    if nearest.numel() == 0:
        return {"anchor_found": False}
    anchor = nearest[distance_to_point[nearest].argmin()]
    center = region_full_points[anchor]
    normal = region_full_normal[anchor]
    reference = torch.tensor([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    axis_u = torch.linalg.cross(normal, reference)
    axis_u = axis_u / axis_u.norm().clamp_min(1e-12)
    axis_v = torch.linalg.cross(normal, axis_u)
    within = (region_full_points - center).norm(dim=-1)
    neighbors = torch.nonzero((within > 1e-6) & (within <= same_mode_radius), as_tuple=False).reshape(-1)
    if neighbors.numel() == 0:
        return {"anchor_found": True, "same_mode_count": 0, "gap_degrees": 360.0}
    displacement = region_full_points[neighbors] - center
    alignment = (region_full_normal[neighbors] * normal).sum(dim=-1).abs()
    same_mode = alignment >= alignment_min
    same_mode_displacement = displacement[same_mode]
    if same_mode_displacement.shape[0] == 0:
        return {"anchor_found": True, "same_mode_count": 0, "gap_degrees": 360.0}
    tangent_offset = (same_mode_displacement * normal).sum(dim=-1, keepdim=True)
    tangent = same_mode_displacement - normal * tangent_offset
    angle = torch.atan2((tangent * axis_v).sum(dim=-1), (tangent * axis_u).sum(dim=-1))
    bins = 180
    bin_width = 2.0 * math.pi / bins
    occupied = torch.zeros((bins,), dtype=torch.bool)
    for a in angle.tolist():
        b = int((a + math.pi) / bin_width) % bins
        occupied[max(0, b - 1):b + 2] = True
    _start, length = _largest_circular_gap_from_bins(occupied, bins)
    return {
        "anchor_found": True,
        "same_mode_count": int(same_mode.sum()),
        "gap_degrees": length * math.degrees(bin_width),
    }


def trace(checkpoint: Path, cap: int, gaps: list[tuple[int, int, int]]) -> dict:
    """``gaps``: list of (region_id, source_stable_id_a, source_stable_id_b)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state

    state = build_frozen_state(checkpoint, cap)
    regions = form_surface_regions(
        state.rep_points, state.rep_frame, state.reliability, state.graph, ids=state.rep_stable_ids,
    )
    id_to_index = {sid: i for i, sid in enumerate(state.rep_stable_ids)}
    region_id_by_rep = torch.tensor(list(regions.node_region_id))
    full_region = region_id_by_rep[state.nearest_representative_index]

    rows = []
    for region_id, sid_a, sid_b in gaps:
        if sid_a not in id_to_index or sid_b not in id_to_index:
            rows.append({"region_id": region_id, "a": sid_a, "b": sid_b, "error": "stable id not found (region ids may have shifted)"})
            continue
        point_a = state.rep_points[id_to_index[sid_a]]
        point_b = state.rep_points[id_to_index[sid_b]]
        mask = full_region == region_id
        region_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        region_full_points = state.full_points[region_indices]
        region_full_normal = state.full_frame.normal_candidate[region_indices]
        samples = []
        for t in (0.15, 0.35, 0.5, 0.65, 0.85):
            point = point_a * (1 - t) + point_b * t
            result = _raw_gap_at(region_full_points, region_full_normal, point)
            result["t"] = t
            samples.append(result)
        rows.append({
            "region_id": region_id, "a": sid_a, "b": sid_b,
            "endpoint_distance": float((point_a - point_b).norm()),
            "region_full_cloud_count": int(region_full_points.shape[0]),
            "samples": samples,
        })
    return {"checkpoint": str(checkpoint), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--gap", action="append", nargs=3, type=int, metavar=("REGION_ID", "STABLE_ID_A", "STABLE_ID_B"), required=True)
    args = parser.parse_args()
    result = trace(args.checkpoint, args.cap, [tuple(item) for item in args.gap])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
