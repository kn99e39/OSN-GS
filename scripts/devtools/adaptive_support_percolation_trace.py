"""Worklog 101 section 12 -- percolation regression check for
SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL.

The real-scene A/B showed largest_subset_surfel_fraction jumping from
22.91% (Worklog 100 / FIXED_MASKED_KNN) to 42.13% (ADAPTIVE_SAME_REGION_LOCAL)
-- a substantial increase that must not be hidden behind the aggregate
number. This traces whether the patio-side and hedge-side seeds from
Worklog 100's own lineage trace end up connected under adaptive support, and
if so, finds the merge chain responsible.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import load_primitive_model, checkpoint_primitive, PRIMITIVE_SURFEL_2D  # noqa: E402
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig
from osn_gs.surface.torch_region_adaptive_support_merge import (
    AdaptiveSupportConfig,
    SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL,
    partition_surfels_region_adaptive_support,
    region_adaptive_support_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel


def _progress(message: str) -> None:
    print(f"[percolation trace] {message}", flush=True)


PATIO_SEED = 117922
HEDGE_SEED = 711179


def main() -> None:
    checkpoint = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000"
    _progress(f"loading checkpoint {checkpoint}")
    model, payload = load_primitive_model(checkpoint, device="cuda")
    primitive = checkpoint_primitive(payload)
    assert primitive == PRIMITIVE_SURFEL_2D and int(getattr(model, "scale_dim", 3)) == 2

    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)

    with torch.no_grad():
        from dataclasses import replace as _dc_replace

        full_orientation = derive_surface_orientation_from_surfel(model)
        orientation = _dc_replace(
            full_orientation,
            gaussian_ids=full_orientation.gaussian_ids[visible_selector],
            positions=full_orientation.positions[visible_selector],
            tangent_axis_u=full_orientation.tangent_axis_u[visible_selector],
            tangent_axis_v=full_orientation.tangent_axis_v[visible_selector],
            surface_normal=full_orientation.surface_normal[visible_selector],
            tangent_scale_u=full_orientation.tangent_scale_u[visible_selector],
            tangent_scale_v=full_orientation.tangent_scale_v[visible_selector],
        )
        local_config = CoverageFirstPartitionConfig()
        region_config = RegionCoherenceConfig(local=local_config, require_positional_continuity=True)
        config = AdaptiveSupportConfig(local=local_config, region=region_config, support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)

        partition = partition_surfels_region_adaptive_support(orientation, config, progress=_progress)
        accounting = region_adaptive_support_accounting(partition)

        patio_subset = int(partition.subset_ids[PATIO_SEED].item())
        hedge_subset = int(partition.subset_ids[HEDGE_SEED].item())
        connected = patio_subset == hedge_subset
        _progress(f"patio subset={patio_subset} hedge subset={hedge_subset} connected={connected}")

        chain_records = []
        if connected:
            patio_region = int(partition.initial_region_ids[PATIO_SEED].item())
            hedge_region = int(partition.initial_region_ids[HEDGE_SEED].item())
            adjacency: dict[int, list[tuple[int, int]]] = {}
            for index, record in enumerate(partition.merge_provenance):
                a, b = record["region_a"], record["region_b"]
                adjacency.setdefault(a, []).append((b, index))
                adjacency.setdefault(b, []).append((a, index))
            visited = {patio_region: None}
            queue = deque([patio_region])
            found = patio_region == hedge_region
            while queue and not found:
                current = queue.popleft()
                for neighbor, prov_index in adjacency.get(current, []):
                    if neighbor in visited:
                        continue
                    visited[neighbor] = (current, prov_index)
                    if neighbor == hedge_region:
                        found = True
                        break
                    queue.append(neighbor)
            if found:
                chain = []
                node = hedge_region
                while node != patio_region:
                    parent, prov_index = visited[node]
                    chain.append(prov_index)
                    node = parent
                chain.reverse()
                chain_records = [partition.merge_provenance[index] for index in chain]
                _progress(f"lineage chain length: {len(chain_records)}")
                for record in chain_records:
                    _progress(f"  {record}")

    output_path = REPO_ROOT / "output/osn_gs_region_support_attribution/adaptive_percolation_trace.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "largest_subset_surfel_fraction": accounting["largest_subset_surfel_fraction"],
                "patio_seed": PATIO_SEED, "hedge_seed": HEDGE_SEED,
                "patio_subset": patio_subset, "hedge_subset": hedge_subset,
                "patio_and_hedge_connected": connected,
                "lineage_chain_length": len(chain_records),
                "lineage_chain": chain_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _progress(f"report -> {output_path}")


if __name__ == "__main__":
    main()
