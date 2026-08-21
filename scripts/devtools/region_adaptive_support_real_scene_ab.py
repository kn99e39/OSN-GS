"""Worklog 101 -- real-scene A/B: FIXED_MASKED_KNN vs ADAPTIVE_SAME_REGION_LOCAL
support acquisition, with every merge threshold identical (Worklog 100
values, unchanged).
"""

from __future__ import annotations

import json
import sys
import time
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
    print(f"[adaptive support real-scene] {message}", flush=True)


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
        config = AdaptiveSupportConfig(
            local=local_config, region=region_config, support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL,
        )

        started = time.time()
        partition = partition_surfels_region_adaptive_support(orientation, config, progress=_progress)
        accounting = region_adaptive_support_accounting(partition)
        elapsed = time.time() - started
        _progress(f"done in {elapsed:.1f}s")

    output_path = REPO_ROOT / "output/osn_gs_region_support_attribution/adaptive_same_region_local_accounting.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(accounting, indent=2), encoding="utf-8")
    _progress(f"report -> {output_path}")
    print(json.dumps({
        "initial_region_count": accounting["initial_region_count"],
        "final_region_count": accounting["final_region_count"],
        "largest_subset_surfel_fraction": accounting["largest_subset_surfel_fraction"],
        "merges_applied": accounting["merges_applied"],
        "interfaces_accepted": accounting["interfaces_accepted"],
        "interfaces_rejected": accounting["interfaces_rejected"],
        "rejection_reason_counts": accounting["rejection_reason_counts"],
        "runtime_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
