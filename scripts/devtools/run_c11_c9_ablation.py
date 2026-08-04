"""Worklog 35: run the A/B/C/D ablation matrix (baseline, C11-only, C9-only,
combined) across real 3k/5k/10k checkpoints plus box_face, by swapping the two
changed production files in and out of place before each subprocess run.
Offline-only; restores the working tree to the combined (D) state on exit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("c:/Projects/OSN-GS")
ORDERING = REPO / "osn_gs/surface/torch_directed_boundary_ordering.py"
REGION = REPO / "osn_gs/surface/torch_gaussian_surface_region_formation.py"
BACKUP = Path("C:/Users/dna10/AppData/Local/Temp/claude/c--Projects-OSN-GS/3697c6bf-838e-4135-bfc1-38e17fb7cfc0/scratchpad/ablation_backup")

CONFIGS = {
    "A_baseline": (BACKUP / "torch_directed_boundary_ordering_OLD.py", BACKUP / "torch_gaussian_surface_region_formation_OLD.py"),
    "B_c11_only": (BACKUP / "torch_directed_boundary_ordering_NEW.py", BACKUP / "torch_gaussian_surface_region_formation_OLD.py"),
    "C_c9_only": (BACKUP / "torch_directed_boundary_ordering_OLD.py", BACKUP / "torch_gaussian_surface_region_formation_NEW.py"),
    "D_combined": (BACKUP / "torch_directed_boundary_ordering_NEW.py", BACKUP / "torch_gaussian_surface_region_formation_NEW.py"),
}

SNAPSHOT_SCRIPT = """
import json, torch
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation

def sh_degree(raw):
    rest_dim = int(raw['features_rest'].shape[-2]); degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim: degree += 1
    return degree

payload = torch.load(r"{checkpoint}", map_location='cpu', weights_only=False)
raw = payload['model_raw']
model = TorchGaussianModel(sh_degree=sh_degree(raw), device='cpu')
model.replace_tensors(xyz=raw['xyz'], features_dc=raw['features_dc'], features_rest=raw['features_rest'],
    opacity=raw['opacity'], scaling=raw['scaling'], rotation=raw['rotation'],
    uncertain_confidence=raw['uncertain_confidence'], uncertain_mask=raw['is_uncertain'],
    surface_uv=raw['surface_uv'], cluster_ids=raw['cluster_ids'],
    surface_owner_kind=raw.get('surface_owner_kind'), surface_owner_id=raw.get('surface_owner_id'),
    stable_gaussian_ids=raw.get('stable_gaussian_ids'))
config = TorchPipelineConfig(canonical_construction_max_points=2048)
pipeline = TorchOSNGSPipeline(config, device='cpu')
import time
t0 = time.time()
with torch.no_grad():
    eligible_mask = (~model.is_uncertain) & (model.surface_owner_kind != 2)
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
    points = model.get_xyz.detach()[eligible_indices]
    activated_scale = model.get_scaling.detach()[eligible_indices]
    normalized_rotation = model.get_rotation.detach()[eligible_indices]
    covariance = covariance_from_scale_rotation(activated_scale, normalized_rotation)
    opacity = model.get_opacity.detach()[eligible_indices, 0]
    stable_ids = tuple(int(v) for v in model.stable_gaussian_ids[eligible_indices].detach().cpu().tolist())
    bundle = pipeline._construct_canonical_with_full_evidence(points, covariance, opacity, stable_ids)
elapsed = time.time() - t0
construction = bundle.construction
regions = construction.surface_regions
member_counts = [len(r.member_ids) for r in regions.regions]
core_member = sum(1 for s in regions.node_membership_state if s == 'core_member')
consensus_attached = sum(1 for s in regions.node_membership_state if s == 'consensus_attached')
ambiguous_unassigned = sum(1 for s in regions.node_membership_state if s == 'ambiguous_unassigned')
summary = dict(construction.diagnostic_summary)
summary['core_member'] = core_member
summary['consensus_attached'] = consensus_attached
summary['ambiguous_unassigned'] = ambiguous_unassigned
summary['region_member_median'] = sorted(member_counts)[len(member_counts)//2] if member_counts else 0
summary['region_member_max'] = max(member_counts) if member_counts else 0
summary['micro_region_le3'] = sum(1 for c in member_counts if c <= 3)
summary['runtime_seconds'] = elapsed
print(json.dumps(summary))
"""

SCENE_SCRIPT = """
import json
from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
import torch, time

scene = make_gaussian_reliability_scene("{scene}", seed=0)
stable_ids = tuple(range(scene.positions.shape[0]))
opacity = torch.ones(scene.positions.shape[0])
config = TorchPipelineConfig(canonical_construction_max_points={cap})
pipeline = TorchOSNGSPipeline(config, device="cpu")
t0 = time.time()
bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
elapsed = time.time() - t0
construction = bundle.construction
summary = dict(construction.diagnostic_summary)
summary["runtime_seconds"] = elapsed
print(json.dumps(summary))
"""


def install(config_name: str) -> None:
    ordering_src, region_src = CONFIGS[config_name]
    shutil.copy(ordering_src, ORDERING)
    shutil.copy(region_src, REGION)


def run_snippet(code: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"error": proc.stderr[-3000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    results = {}
    try:
        for config_name in CONFIGS:
            install(config_name)
            results[config_name] = {}
            for cp in ["3000", "5000", "10000"]:
                code = SNAPSHOT_SCRIPT.format(checkpoint=f"output/osn_gs_scene/{cp}/checkpoint.pt")
                results[config_name][f"real_{cp}"] = run_snippet(code)
            code = SCENE_SCRIPT.format(scene="box_face", cap=27)
            results[config_name]["box_face_downsampled_27"] = run_snippet(code)
    finally:
        install("D_combined")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
