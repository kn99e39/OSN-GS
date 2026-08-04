"""Worklog 54: report the region-boundary-status contract distribution
(eligible_closed_boundary / open_observed_fragment / insufficient_observation
/ ambiguous_boundary / rejected_unsafe) on a frozen real checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def trace(checkpoint: Path, cap: int) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state, replay_boundary_candidates, replay_region_formation

    from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
    from osn_gs.surface.torch_visible_boundary_region_status import classify_all_region_boundary_statuses

    state = build_frozen_state(checkpoint, cap)
    regions = replay_region_formation(state, None)
    candidates = replay_boundary_candidates(state, regions)
    accepted = tuple(edge for region in regions.regions for edge in region.internal_accepted_edge_ids)
    _, components = recover_directed_boundary_components(candidates, accepted)
    statuses = classify_all_region_boundary_statuses(
        tuple(region.region_id for region in regions.regions), components, candidates,
    )
    return {
        "checkpoint": str(checkpoint),
        "region_count": len(statuses),
        "status_distribution": dict(Counter(status.status for status in statuses)),
        "eligible_regions": [status.payload() for status in statuses if status.status == "eligible_closed_boundary"],
        "rejected_unsafe_regions": [status.payload() for status in statuses if status.status == "rejected_unsafe"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps({path.stem + "_" + path.parent.name: trace(path, args.cap) for path in args.checkpoints}, indent=2))


if __name__ == "__main__":
    main()
