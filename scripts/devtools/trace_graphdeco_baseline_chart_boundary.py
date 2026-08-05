"""Ad hoc: run the real visible-surface / parametric-chart-boundary
construction path (worklog 54-61) against the GRAPHDECO 3DGS baseline's own
trained point clouds (gaussian-splatting/output/scene/point_cloud/*), not
OSN-GS's own checkpoints -- to check whether the physical/parametric chart
counts look meaningfully different on an independently-trained scene.

Fairness caveat (see project_baseline_comparison memory): the baseline has
no `is_uncertain`/`surface_owner_kind` concept at all -- every Gaussian here
is "visible", unlike OSN-GS checkpoints where those masks matter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def load_baseline_ply(path: Path):
    ply = PlyData.read(str(path))
    v = ply.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacity_raw = np.asarray(v["opacity"])[..., None]
    scale_names = sorted(
        (p.name for p in v.properties if p.name.startswith("scale_")), key=lambda x: int(x.split("_")[-1]),
    )
    scale_raw = np.stack([np.asarray(v[name]) for name in scale_names], axis=1)
    rot_names = sorted(
        (p.name for p in v.properties if p.name.startswith("rot")), key=lambda x: int(x.split("_")[-1]),
    )
    rot_raw = np.stack([np.asarray(v[name]) for name in rot_names], axis=1)

    xyz = torch.as_tensor(xyz, dtype=torch.float32)
    opacity = torch.sigmoid(torch.as_tensor(opacity_raw, dtype=torch.float32)).reshape(-1)
    activated_scale = torch.exp(torch.as_tensor(scale_raw, dtype=torch.float32))
    rotation = torch.nn.functional.normalize(torch.as_tensor(rot_raw, dtype=torch.float32), dim=1)
    covariance = covariance_from_scale_rotation(activated_scale, rotation)
    return xyz, covariance, opacity


def trace(path: Path, cap: int, device: str = "cpu") -> dict:
    xyz, covariance, opacity = load_baseline_ply(path)
    stable_ids = list(range(int(xyz.shape[0])))
    config = TorchPipelineConfig(canonical_construction_max_points=int(cap))
    pipeline = TorchOSNGSPipeline(config, device=device)
    bundle = pipeline._construct_canonical_with_full_evidence(
        xyz.to(device), covariance.to(device), opacity.to(device), stable_ids,
    )
    s = bundle.construction.diagnostic_summary
    return {
        "checkpoint": str(path),
        "input_gaussian_count": s["input_gaussian_count"],
        "region_count": s["region_count"],
        "reliable_count": s["reliable_count"],
        "before_physical_eligible_closed_count": s["region_boundary_eligible_closed_count"],
        "before_physical_materialized_surface_count": s["materialized_surface_count"],
        "after_parametric_chart_eligible_count": s["parametric_chart_eligible_count"],
        "after_parametric_chart_materialized_surface_count": s["parametric_chart_materialized_surface_count"],
        "combined_materialized_surface_count": (
            s["materialized_surface_count"] + s["parametric_chart_materialized_surface_count"]
        ),
        "parametric_chart_insufficient_topology_count": s["parametric_chart_insufficient_topology_count"],
        "parametric_chart_open_or_branching_count": s["parametric_chart_open_or_branching_count"],
        "parametric_chart_self_intersecting_count": s["parametric_chart_self_intersecting_count"],
        "parametric_chart_partition_seam_segment_count": s["parametric_chart_partition_seam_segment_count"],
        "parametric_chart_physical_termination_segment_count": s["parametric_chart_physical_termination_segment_count"],
        "boundary_failure_stage": s["boundary_failure_stage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plys", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps({path.parent.name: trace(path, args.cap) for path in args.plys}, indent=2))


if __name__ == "__main__":
    main()
