"""Capture a detached actual-scene mode-selection replay input and public oracle.
This tool is intentionally separate from the production selection hot path.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# This import order matches the working training runner and avoids standalone
# package circular-import ordering on this Windows environment.
from scripts.experiments.run_long_horizon_adc_visible_surface import MaturationTrainer
from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.surface.torch_density_preserving_representative_selection import select_density_preserving_representatives
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.utils.torch_ops import default_device
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig

def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "DATASET"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--image-downscale", type=int, default=8)
    parser.add_argument("--train-resolution-scale", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2048)
    args = parser.parse_args()
    import torch
    device = default_device(prefer_cuda=True)
    scene = load_colmap_scene(args.source, device=device, image_device="cpu", image_downscale=args.image_downscale, max_images=args.max_images)
    trainer = MaturationTrainer(
        pipeline_config=TorchPipelineConfig(canonical_construction_max_points=args.budget),
        training_config=TorchTrainingConfig(
            iterations=1, progress_log_interval=0, timing_log_interval=0,
            prefer_cuda=device == "cuda", write_output_files=False,
            visible_nurbs_update_schedule="disabled", surface_loss_patch_budget=0,
            density_control=TorchDensityControlConfig(densify_until_iter=0),
        ),
        rasterizer_config=GaussianRasterizerConfig(prefer_cuda=True), device=device,
        snapshot_iterations=set(),
    )
    result = trainer.train(scene, Path(args.output).parent / "capture_training")
    model = result.state.model
    mask = (~model.is_uncertain) & (model.surface_owner_kind != 2)
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    positions = model.get_xyz.detach()[indices]
    scales = model.get_scaling.detach()[indices]
    rotations = model.get_rotation.detach()[indices]
    opacity = model.get_opacity.detach()[indices, 0]
    stable_ids = model.stable_gaussian_ids.detach()[indices]
    covariance = covariance_from_scale_rotation(scales, rotations)
    frame = extract_covariance_frame(covariance)
    selection = select_density_preserving_representatives(positions, frame, opacity, stable_ids.cpu().tolist(), max_points=args.budget)
    payload = {
        "schema_version": "mode_aware_selection_replay_v1_public_oracle",
        "source_commit": _git("rev-parse", "HEAD"),
        "working_tree_porcelain": _git("status", "--porcelain"),
        "selection_config": {"max_points": args.budget, "max_modes_per_cell": 4, "mode_normal_alignment_min": 0.6, "mode_offset_max_thickness_ratio": 3.0},
        "input": {
            "positions": positions.cpu(), "scales": scales.cpu(), "rotations": rotations.cpu(),
            "opacity": opacity.cpu(), "stable_ids": stable_ids.cpu(), "observed_mask": mask.detach().cpu(),
            "dtype": str(positions.dtype),
        },
        "public_oracle": {
            "representative_indices": selection.representative_indices.cpu(),
            "representative_stable_ids": stable_ids[selection.representative_indices].cpu(),
            "cell_ids": selection.cell_ids.cpu(), "mode_ids": selection.mode_ids.cpu(),
            "diagnostics": selection.diagnostics.__dict__,
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "gaussians": int(positions.shape[0]), "representatives": int(selection.representative_indices.numel()), "candidate_modes": selection.diagnostics.total_candidate_mode_count}, indent=2))
if __name__ == "__main__": main()
