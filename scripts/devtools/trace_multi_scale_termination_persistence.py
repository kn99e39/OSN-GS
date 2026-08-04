"""Worklog 50: does the continuation shell's single `radius_spacing_multiplier
* representative_mean_spacing` (aka "4x candidate_scale") term reach far
enough to let same-surface support from BEYOND a genuine close-range gap
suppress a real termination?

Reruns production `build_continuation_shells_from_input` at
radius_spacing_multiplier in {1, 2, 3, 4} (1x/2x/3x/4x today's default),
keeping every other config field (radius_tangent_scale_multiplier, same-mode
thresholds, angular-bin gap logic, worklog 48's fold-signature locality gate)
at production defaults. No new geometry is introduced -- this just asks the
existing continuation-shell query at four different radii and records the
state sequence and the members that filled the gap at each radius.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellConfig, ContinuationShellInput, build_continuation_shells_from_input
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions

SCALES = (1.0, 2.0, 3.0, 4.0)
FOCUS_REGIONS = ""


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
    accepted_edges = tuple(edge for region in regions.regions for edge in region.internal_accepted_edge_ids)

    per_scale = {}
    for scale in SCALES:
        config = ContinuationShellConfig(radius_spacing_multiplier=scale)
        continuation_input = ContinuationShellInput(
            full_positions=state.full_points, full_frame=state.full_frame, full_intrinsic=state.full_intrinsic,
            full_opacity=state.full_opacity, full_stable_ids=state.full_stable_ids,
            nearest_representative_index=state.nearest_representative_index,
            representative_mean_spacing=state.representative_mean_spacing, config=config,
        )
        queries = build_continuation_shells_from_input(
            continuation_input, state.rep_points, state.rep_frame, state.rep_stable_ids, regions, canonical_frames,
        )
        per_scale[scale] = queries

    region_id_by_node = dict(zip(state.rep_stable_ids, regions.node_region_id))
    member_count_by_region = {region.region_id: len(region.member_ids) for region in regions.regions}
    major_region_ids = sorted(member_count_by_region, key=lambda r: -member_count_by_region[r])[:12]

    focus_region_ids = {int(item) for item in (FOCUS_REGIONS or "").split(",") if item.strip()}
    rows = []
    for node_id in state.rep_stable_ids:
        region_id = region_id_by_node.get(node_id, -1)
        if region_id not in major_region_ids and region_id not in focus_region_ids:
            continue
        state_sequence = [per_scale[scale].get(node_id).state if per_scale[scale].get(node_id) else None for scale in SCALES]
        if state_sequence[0] is None:
            continue
        # A "scale-inconsistent" node: NOT no_gap at a smaller scale but
        # no_gap once the radius reaches production default (4x).
        became_no_gap_at = next((scale for scale, s in zip(SCALES, state_sequence) if s == "no_gap"), None)
        was_physical_before = any(
            s == "observed_support_termination" for s, scale in zip(state_sequence, SCALES) if became_no_gap_at is None or scale < became_no_gap_at
        )
        flips_to_no_gap_after_physical = became_no_gap_at is not None and became_no_gap_at > 1.0 and was_physical_before
        row = {
            "region_id": region_id, "node_id": node_id, "state_sequence": dict(zip(SCALES, state_sequence)),
            "flips_to_no_gap_after_physical": flips_to_no_gap_after_physical,
        }
        if flips_to_no_gap_after_physical:
            # Record what filled the gap at the scale where it flipped, and
            # what the gap looked like one scale step before.
            before_scale = SCALES[SCALES.index(became_no_gap_at) - 1]
            before_query = per_scale[before_scale].get(node_id)
            after_query = per_scale[became_no_gap_at].get(node_id)
            row["became_no_gap_at_scale"] = became_no_gap_at
            row["before_scale"] = before_scale
            row["before_gap_width_degrees"] = before_query.gap_width_degrees if before_query else None
            row["before_same_mode_support_count"] = before_query.same_mode_support_count if before_query else None
            row["after_same_mode_support_count"] = after_query.same_mode_support_count if after_query else None
            row["after_support_radius"] = after_query.support_radius if after_query else None
            row["after_fingerprint_sample"] = list(after_query.source_full_cloud_fingerprint[:10]) if after_query else None
        rows.append(row)

    summary_counts = {}
    for scale in SCALES:
        from collections import Counter
        summary_counts[scale] = dict(Counter(q.state for q in per_scale[scale].values()))

    return {
        "checkpoint": str(checkpoint),
        "state_counts_by_scale": summary_counts,
        "major_region_node_count": len(rows),
        "flip_count": sum(1 for row in rows if row["flips_to_no_gap_after_physical"]),
        "flip_rows": [row for row in rows if row["flips_to_no_gap_after_physical"]][:30],
        "focus_region_rows": [row for row in rows if row["region_id"] in focus_region_ids],
    }


def main():
    global FOCUS_REGIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--focus-regions", type=str, default="")
    args = parser.parse_args()
    FOCUS_REGIONS = args.focus_regions
    print(json.dumps({path.stem + "_" + path.parent.name: trace(path, args.cap) for path in args.checkpoints}, indent=2))


if __name__ == "__main__":
    main()
