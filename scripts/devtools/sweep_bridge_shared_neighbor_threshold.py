"""Worklog 37 (task section 7): diagnostic-only sweep of
`bridge_min_shared_neighbor_for_well_supported` on the frozen replay.
NEVER applied to production -- diagnostic sweep only, per explicit
instruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationConfig

from frozen_core_seeding_replay import build_frozen_state, replay_region_formation


def sweep(checkpoint: Path, cap: int) -> dict:
    state = build_frozen_state(checkpoint, cap)
    results = {}
    for threshold in [0, 1, 2, 3]:
        config = RegionFormationConfig(bridge_min_shared_neighbor_for_well_supported=threshold)
        result = replay_region_formation(state, config)
        core_member = sum(1 for s in result.node_membership_state if s == "core_member")
        consensus_attached = sum(1 for s in result.node_membership_state if s == "consensus_attached")
        ambiguous = sum(1 for s in result.node_membership_state if s == "ambiguous_unassigned")
        member_counts = [len(r.member_ids) for r in result.regions]
        micro = sum(1 for c in member_counts if c <= 3)
        major = sum(1 for c in member_counts if c > 10)
        results[f"threshold_{threshold}"] = {
            "core_member": core_member,
            "consensus_attached": consensus_attached,
            "ambiguous_unassigned": ambiguous,
            "region_count": len(result.regions),
            "region_member_max": max(member_counts) if member_counts else 0,
            "micro_region_count": micro,
            "major_region_count": major,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    result = sweep(args.checkpoint, args.cap)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
