"""Worklog 48: does the same-mode support that produces `no_gap` sit on a
candidate-local, bounded, accepted same-surface path back to the query
representative, or is it merely within the same spatial radius (which can
include a physically disconnected fold/gap-crossing patch of the same
region)?

Reuses build_frozen_state / form_surface_regions / construct_canonical_region_tangent_frames
exactly as production does. Re-derives the SAME per-member same_mode mask
`torch_full_cloud_continuation_shell.build_continuation_shells` computes
(identical thresholds, identical config) but, unlike production, also keeps
each same-mode member's own nearest representative id and computes its
accepted-topology hop distance (BFS over region.internal_accepted_edge_ids)
back to the query representative. No production code path is changed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

import torch

from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_full_cloud_continuation_shell import (
    ContinuationShellConfig,
    _group_full_indices_by_representative,
    _largest_circular_gap_from_bins,
)
from osn_gs.surface.torch_gaussian_structural_reliability import INTRINSIC_REJECTED
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions

# Any depth is "local" as long as it is the SAME accepted-edge connected
# component -- region formation can union two components under one region id
# via a non-adjacency merge criterion (worklog 35's parallel-shortcut
# aggregate merge, or a bridge veto exemption), and a member reachable only
# through that heuristic union is not evidence of a continuous local surface
# path. Unbounded BFS restricted to the region's OWN accepted-edge graph is
# the correct locality certificate; a fixed hop cap would flag legitimately
# long, but genuinely continuous, thin regions as nonlocal.
def _build_hop_distances(accepted_pairs, region_node_ids, source_id):
    adjacency = {node: [] for node in region_node_ids}
    for left, right in accepted_pairs:
        if left in adjacency and right in adjacency:
            adjacency[left].append(right)
            adjacency[right].append(left)
    distance = {source_id: 0}
    queue = deque([source_id])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def trace(checkpoint: Path, cap: int) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state

    state = build_frozen_state(checkpoint, cap)
    regions = form_surface_regions(
        state.rep_points, state.rep_frame, state.reliability, state.graph, ids=state.rep_stable_ids,
    )
    canonical_frames = construct_canonical_region_tangent_frames(
        state.rep_points, state.rep_frame, state.reliability, regions, ids=state.rep_stable_ids,
    )
    config = ContinuationShellConfig()
    m = int(state.rep_points.shape[0])
    device = state.rep_points.device
    representative_distance = torch.cdist(state.rep_points, state.rep_points)
    radius = torch.maximum(
        config.radius_tangent_scale_multiplier * state.rep_frame.tangent_major_scale,
        config.radius_spacing_multiplier * state.representative_mean_spacing.clamp_min(1e-9),
    )
    members_by_representative = _group_full_indices_by_representative(state.nearest_representative_index, m)
    region_id_by_rep = list(regions.node_region_id)
    membership_by_rep = list(regions.node_membership_state)
    region_id_tensor = torch.tensor(region_id_by_rep, dtype=torch.long, device=device)

    accepted_by_region = {
        region.region_id: {frozenset(pair) for pair in region.internal_accepted_edge_ids}
        for region in regions.regions
    }
    region_node_ids_by_region = {
        region.region_id: set(region.member_ids) for region in regions.regions
    }

    id_index = {stable_id: index for index, stable_id in enumerate(state.rep_stable_ids)}
    fullcloud_stable_by_index = state.full_stable_ids

    rows = []
    summary = Counter()
    major_region_ids = sorted(
        region_id_by_rep, key=lambda region_id: -sum(1 for r in region_id_by_rep if r == region_id),
    )
    major_region_set = set(sorted(set(region_id_by_rep), key=lambda region_id: -region_id_by_rep.count(region_id))[:12])

    for source in range(m):
        region_id = region_id_by_rep[source]
        if region_id < 0 or membership_by_rep[source] not in ("core_member", "consensus_attached"):
            continue
        if region_id not in major_region_set:
            continue
        canonical = canonical_frames[source] if source < len(canonical_frames) else None
        if canonical is None:
            continue

        same_region_mask = region_id_tensor == region_id
        within_radius_mask = representative_distance[source] <= radius[source]
        neighbor_reps = torch.nonzero(same_region_mask & within_radius_mask, as_tuple=False).reshape(-1).tolist()
        if source not in neighbor_reps:
            neighbor_reps.append(source)
        member_index_parts = [members_by_representative[j] for j in neighbor_reps if members_by_representative[j].numel() > 0]
        if not member_index_parts:
            continue
        members = torch.cat(member_index_parts, dim=0)

        node_position = state.rep_points[source]
        node_normal = canonical.oriented_normal
        axis_u = canonical.tangent_axis_0
        axis_v = canonical.tangent_axis_1
        node_tangent_scale = state.rep_frame.tangent_major_scale[source].clamp_min(1e-12)

        displacement = state.full_points[members] - node_position
        radial_distance = torch.linalg.norm(displacement, dim=-1)
        within_radius = radial_distance <= radius[source]
        nonzero_radius = radial_distance > 1e-8
        keep = within_radius & nonzero_radius
        if not bool(keep.any()):
            continue
        members = members[keep]
        displacement = displacement[keep]
        radial_distance = radial_distance[keep]

        member_normal = state.full_frame.normal_candidate[members]
        sign = torch.where((member_normal * node_normal).sum(dim=-1) < 0.0, -1.0, 1.0).unsqueeze(-1)
        corrected_normal = member_normal * sign
        alignment = (corrected_normal * node_normal).sum(dim=-1).clamp(-1.0, 1.0)
        tangent_offset = (displacement * node_normal).sum(dim=-1)
        residual_ratio = tangent_offset.abs() / node_tangent_scale
        footprint_ratio = state.full_frame.tangent_major_scale[members] / node_tangent_scale
        rejected = torch.tensor(
            [state.full_intrinsic.intrinsic_class[int(i)] == INTRINSIC_REJECTED for i in members.tolist()],
            device=device,
        )
        same_mode = (
            (~rejected)
            & (alignment >= config.same_mode_normal_alignment_min)
            & (residual_ratio <= config.same_mode_residual_max_ratio)
            & (footprint_ratio >= config.footprint_ratio_min)
            & (footprint_ratio <= config.footprint_ratio_max)
        )
        if not bool(same_mode.any()):
            continue

        tangent_vector = displacement - node_normal * tangent_offset.unsqueeze(-1)
        angle = torch.atan2((tangent_vector * axis_v).sum(dim=-1), (tangent_vector * axis_u).sum(dim=-1))
        angular_halfwidth = torch.atan2(
            state.full_frame.equivalent_tangent_scale[members], radius[source]
        ).clamp(max=math.radians(config.max_footprint_halfwidth_degrees))
        bins = config.angular_bins
        bin_width = 2.0 * math.pi / bins
        same_mode_occupied = torch.zeros((bins,), dtype=torch.bool)
        member_angle = angle[same_mode]
        member_halfwidth = angular_halfwidth[same_mode]
        for a, hw in zip(member_angle.tolist(), member_halfwidth.tolist()):
            lo = int(math.floor((a - hw + math.pi) / bin_width))
            hi = int(math.floor((a + hw + math.pi) / bin_width))
            for offset in range(lo, hi + 1):
                same_mode_occupied[offset % bins] = True
        _, same_mode_length = _largest_circular_gap_from_bins(same_mode_occupied, bins)
        gap_threshold_bins = max(1, int(math.ceil(math.radians(config.min_gap_degrees) / bin_width)))
        if same_mode_length >= gap_threshold_bins:
            continue  # not a no_gap node

        source_stable_id = state.rep_stable_ids[source]
        accepted_pairs = accepted_by_region.get(region_id, set())
        region_node_ids = region_node_ids_by_region.get(region_id, set())
        hop_distances = _build_hop_distances(accepted_pairs, region_node_ids, source_stable_id)

        same_mode_members = members[same_mode].tolist()
        same_mode_straight_line = radial_distance[same_mode].tolist()
        residual_vals = residual_ratio[same_mode].tolist()
        alignment_vals = alignment[same_mode].tolist()
        local_mass, nonlocal_mass, unresolved_mass = 0, 0, 0
        member_rows = []
        opacity = state.full_opacity[members][same_mode].tolist()
        node_spacing = float(state.representative_mean_spacing[source].clamp_min(1e-9))
        for member_full_index, straight_line, residual, align, op in zip(
            same_mode_members, same_mode_straight_line, residual_vals, alignment_vals, opacity,
        ):
            nearest_rep_index = int(state.nearest_representative_index[member_full_index])
            nearest_rep_stable_id = state.rep_stable_ids[nearest_rep_index]
            hop = hop_distances.get(nearest_rep_stable_id)
            fold_ratio = None
            if hop is None:
                classification = "nonlocal_different_component"
                nonlocal_mass += 1
            else:
                # A same-component member reached only via a graph path many
                # times longer than its straight-line embedding distance is
                # the fold/gap-crossing signature: pointwise-local in space,
                # but not locally continuous along the accepted surface path.
                path_length_estimate = hop * node_spacing
                fold_ratio = round(path_length_estimate / max(straight_line, 1e-9), 3)
                if hop >= 2 and fold_ratio >= 3.0:
                    classification = "nonlocal_fold_signature"
                    nonlocal_mass += 1
                else:
                    classification = "local_connected"
                    local_mass += 1
            member_rows.append({
                "full_cloud_stable_id": fullcloud_stable_by_index[member_full_index],
                "nearest_representative_stable_id": nearest_rep_stable_id,
                "hop_distance": hop,
                "straight_line_distance_over_spacing": round(straight_line / node_spacing, 3),
                "fold_ratio": fold_ratio,
                "residual_ratio": round(residual, 4),
                "normal_alignment": round(align, 4),
                "opacity": round(op, 4),
                "classification": classification,
            })

        node_classification = (
            "local_connected_smooth" if nonlocal_mass == 0 else
            "nonlocal_leakage" if local_mass == 0 else
            "mixed_local_and_nonlocal"
        )
        summary[node_classification] += 1
        rows.append({
            "region_id": region_id,
            "source_stable_id": source_stable_id,
            "same_mode_support_count": int(same_mode.sum().item()),
            "local_support_count": local_mass,
            "nonlocal_support_count": nonlocal_mass,
            "node_classification": node_classification,
            "member_sample": (
                sorted((row for row in member_rows if row["classification"] != "local_connected"), key=lambda row: row["classification"])[:12]
                + sorted((row for row in member_rows if row["classification"] == "local_connected"), key=lambda row: row["hop_distance"])[:4]
            ),
        })

    return {
        "checkpoint": str(checkpoint),
        "no_gap_node_count": len(rows),
        "classification_summary": dict(summary),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps({path.stem + "_" + path.parent.name: trace(path, args.cap) for path in args.checkpoints}, indent=2))


if __name__ == "__main__":
    main()
