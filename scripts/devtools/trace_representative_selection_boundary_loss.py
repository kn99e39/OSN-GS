"""Worklog 49: does representative selection drop genuine boundary evidence?

`select_density_preserving_representatives` already separates a voxel cell's
members into locally-consistent normal/offset MODES
(`_split_cell_into_modes`) before farthest-point sampling picks one
representative per selected mode under the global cap. A cell with two modes
is, by that split's own compatibility gate (normal alignment < 0.6 OR offset
beyond 3x thickness), evidence of two genuinely different local surface
orientations sharing one voxel -- exactly what a real crease/boundary looks
like at cell granularity.

This script reruns candidate construction (reusing the production
`_voxel_cells`/`_split_cell_into_modes` helpers, no new geometry) and the
SAME weighted-farthest-point selection, but keeps the full candidate list
instead of discarding it after budget selection. It then asks: how many
mode-candidates are SIBLINGS of a mode that WAS selected in the same cell,
but were themselves dropped by the budget-competition FPS step? That is an
unambiguous evidence-loss signal (the split algorithm found genuine surface-
orientation divergence right there; the drop is purely a budget artifact),
not a heuristic guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from osn_gs.surface.torch_density_preserving_representative_selection import (
    RepresentativeSelectionConfig,
    _split_cell_into_modes,
    _voxel_cells,
)


def _build_candidates(points, frame, opacity, stable_ids, *, budget, config):
    torch_ = torch
    keys, _resolution = _voxel_cells(points, budget)
    order = torch_.argsort(keys)
    sorted_keys = keys[order]
    boundaries = torch_.ones_like(sorted_keys, dtype=torch_.bool)
    boundaries[1:] = sorted_keys[1:] != sorted_keys[:-1]
    cell_starts = torch_.nonzero(boundaries, as_tuple=False).reshape(-1).tolist()
    cell_starts.append(int(order.numel()))

    normals_np = frame.normal_candidate.detach().cpu().numpy()
    thickness_np = frame.normal_thickness.detach().cpu().numpy()
    positions_np = points.detach().cpu().numpy()
    stable_id_keys = [str(item) for item in stable_ids]

    candidates = []
    for cell_index in range(len(cell_starts) - 1):
        start, end = cell_starts[cell_index], cell_starts[cell_index + 1]
        member_local_indices = order[start:end].tolist()
        cell_id = int(sorted_keys[start])
        modes = _split_cell_into_modes(
            member_local_indices, normals_np, thickness_np, positions_np, stable_id_keys, config=config
        )
        for mode_id, member_indices in enumerate(modes):
            member_tensor = torch.tensor(member_indices, dtype=torch.long, device=points.device)
            member_opacity = opacity[member_tensor]
            opacity_sum = float(member_opacity.sum())
            centroid = (
                (points[member_tensor] * member_opacity.unsqueeze(-1)).sum(dim=0) / opacity_sum
                if opacity_sum > 1e-9 else points[member_tensor].mean(dim=0)
            )
            distances = torch.linalg.norm(points[member_tensor] - centroid, dim=-1).tolist()
            order_key = [(distances[i], str(stable_ids[member_indices[i]])) for i in range(len(member_indices))]
            best_local = min(range(len(member_indices)), key=lambda i: order_key[i])
            representative_index = member_indices[best_local]
            candidates.append({
                "cell_id": cell_id, "mode_id": mode_id, "representative_index": representative_index,
                "source_count": len(member_indices), "source_opacity_mass": opacity_sum,
                "centroid": centroid, "mode_normal": normals_np[representative_index],
                "cell_mode_count": len(modes),
            })
    return candidates


def _weighted_fps_select(points, candidates, budget, stable_ids):
    positions_tensor = torch.stack([points[c["representative_index"]] for c in candidates], dim=0)
    support = torch.tensor([c["source_count"] for c in candidates], dtype=torch.float32)
    opacity_mass = torch.tensor([c["source_opacity_mass"] for c in candidates], dtype=torch.float32)
    weight = 0.5 * (support / support.max().clamp_min(1e-9)) + 0.5 * (opacity_mass / opacity_mass.max().clamp_min(1e-9))
    stable_key = [str(stable_ids[c["representative_index"]]) for c in candidates]
    total_candidates = len(candidates)
    pairwise_distance = torch.cdist(positions_tensor, positions_tensor)
    stable_rank = torch.empty((total_candidates,), dtype=torch.long)
    for rank, index in enumerate(sorted(range(total_candidates), key=lambda i: stable_key[i])):
        stable_rank[index] = rank
    support_best = support.max()
    seed_pool = torch.nonzero(support == support_best, as_tuple=False).reshape(-1)
    opacity_best = opacity_mass[seed_pool].max()
    seed_candidates = seed_pool[opacity_mass[seed_pool] == opacity_best]
    seed = int(seed_candidates[stable_rank[seed_candidates].argmin()].item())
    selected_local = [seed]
    selected_mask = torch.zeros((total_candidates,), dtype=torch.bool)
    selected_mask[seed] = True
    min_distance = pairwise_distance[seed].masked_fill(selected_mask, -1.0)
    for _ in range(budget - 1):
        score = min_distance.clamp_min(0.0) * weight
        score = score.masked_fill(selected_mask, -1.0)
        best_score = score.max()
        tied = torch.nonzero(score >= best_score - 1e-9, as_tuple=False).reshape(-1)
        next_pick = int(tied[stable_rank[tied].argmin()].item())
        selected_local.append(next_pick)
        selected_mask[next_pick] = True
        min_distance = torch.minimum(min_distance, pairwise_distance[next_pick]).masked_fill(selected_mask, -1.0)
    return set(selected_local)


def trace(checkpoint: Path, cap: int) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state

    state = build_frozen_state(checkpoint, cap)
    config = RepresentativeSelectionConfig()
    opacity = state.full_opacity if hasattr(state, "full_opacity") else None
    candidates = _build_candidates(state.full_points, state.full_frame, state.full_opacity, state.full_stable_ids, budget=cap, config=config)
    if len(candidates) <= cap:
        return {"checkpoint": str(checkpoint), "total_candidates": len(candidates), "budget": cap, "note": "full_coverage, no FPS drop possible"}
    selected_local = _weighted_fps_select(state.full_points, candidates, cap, state.full_stable_ids)

    by_cell: dict[int, list[int]] = {}
    for local_index, candidate in enumerate(candidates):
        by_cell.setdefault(candidate["cell_id"], []).append(local_index)

    import numpy as np

    dropped_sibling_of_selected = []
    for cell_id, local_indices in by_cell.items():
        if len(local_indices) < 2:
            continue
        selected_here = [i for i in local_indices if i in selected_local]
        dropped_here = [i for i in local_indices if i not in selected_local]
        if not selected_here or not dropped_here:
            continue
        for dropped_index in dropped_here:
            candidate = candidates[dropped_index]
            best_alignment = max(
                float(abs(np.dot(candidate["mode_normal"], candidates[i]["mode_normal"])))
                for i in selected_here
            )
            dropped_sibling_of_selected.append({
                "cell_id": cell_id,
                "dropped_stable_id": state.full_stable_ids[candidate["representative_index"]],
                "source_count": candidate["source_count"],
                "cell_mode_count": candidate["cell_mode_count"],
                "best_sibling_normal_alignment": round(best_alignment, 4),
                "sibling_selected_stable_ids": [
                    state.full_stable_ids[candidates[i]["representative_index"]] for i in selected_here
                ],
            })

    cell_mode_count_histogram = Counter(item["cell_mode_count"] for item in dropped_sibling_of_selected)
    source_counts = sorted(item["source_count"] for item in dropped_sibling_of_selected)
    alignments = sorted(item["best_sibling_normal_alignment"] for item in dropped_sibling_of_selected)

    return {
        "checkpoint": str(checkpoint),
        "total_candidates": len(candidates),
        "budget": cap,
        "selected_count": len(selected_local),
        "dropped_count": len(candidates) - len(selected_local),
        "multi_mode_cell_with_partial_drop_count": len(dropped_sibling_of_selected),
        "cell_mode_count_histogram": dict(cell_mode_count_histogram),
        "dropped_source_count_percentiles": {
            "min": source_counts[0] if source_counts else None,
            "p50": source_counts[len(source_counts) // 2] if source_counts else None,
            "p90": source_counts[int(len(source_counts) * 0.9)] if source_counts else None,
            "max": source_counts[-1] if source_counts else None,
        },
        "best_sibling_normal_alignment_percentiles": {
            "min": alignments[0] if alignments else None,
            "p10": alignments[int(len(alignments) * 0.1)] if alignments else None,
            "p50": alignments[len(alignments) // 2] if alignments else None,
            "p90": alignments[int(len(alignments) * 0.9)] if alignments else None,
            "max": alignments[-1] if alignments else None,
        },
        "low_alignment_lt_0.3_count": sum(1 for a in alignments if a < 0.3),
        "low_alignment_lt_0.0_count": sum(1 for a in alignments if a < 0.0),
        "low_alignment_examples": sorted(dropped_sibling_of_selected, key=lambda item: item["best_sibling_normal_alignment"])[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps({path.stem + "_" + path.parent.name: trace(path, args.cap) for path in args.checkpoints}, indent=2))


if __name__ == "__main__":
    main()
