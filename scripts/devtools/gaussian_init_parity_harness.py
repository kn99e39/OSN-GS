"""Worklog 64: three-way Gaussian initialization parity harness.

Compares, from the SAME COLMAP point cloud and the SAME baseline-style
camera sequence (train.py's own seed(0) -> Scene shuffle -> pop-based
per-iteration selection):

  - Graphdeco baseline (unmodified `gaussian_renderer.render` + loss + ADC)
  - OSN-GS with `gaussian_initialization_mode="covariance_knn"` (the
    pre-worklog-64 local-PCA planar-surfel init, now an explicit
    experimental option)
  - OSN-GS with `gaussian_initialization_mode="baseline_compatible"` (the
    new default -- Graphdeco-equivalent isotropic init)

Each OSN-GS condition builds its own model through the REAL, unmodified
`TorchOSNGSPipeline.initialize()` (no tensor transplant -- this round is
about each side's OWN initialization behavior, not about isolating
render/loss code-path differences). All three run through their own real
render/loss/optimizer/ADC code. Checkpoints are recorded at steps
0/1/100/600 (0 = immediately after init, before any optimizer step) plus
whichever step the first real ADC event fires on (density_control_interval
defaults match on both sides: densify_from_iter=500, interval=100).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.gaussian.torch_density_control import (
    TorchDensityControlConfig,
    add_densification_stats,
    apply_adaptive_density_control,
    should_run_adc,
    update_max_radii,
)
from osn_gs.gaussian.torch_model import GaussianParameterGroups
from osn_gs.losses.torch_losses import image_reconstruction_loss
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "gaussian-splatting"
CHECKPOINT_STEPS = {0, 1, 100, 600}


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _population_stats(scaling: torch.Tensor, opacity: torch.Tensor) -> dict:
    with torch.no_grad():
        scale = scaling.detach().cpu().numpy()
        sorted_scale = np.sort(scale, axis=1)
        s_min, s_mid, s_max = sorted_scale[:, 0], sorted_scale[:, 1], sorted_scale[:, 2]
        anisotropy = s_max / np.clip(s_min, 1e-12, None)
        opa = opacity.detach().reshape(-1).cpu().numpy()
    return {
        "count": int(scale.shape[0]),
        "s_min": _percentiles(s_min), "s_mid": _percentiles(s_mid), "s_max": _percentiles(s_max),
        "anisotropy": _percentiles(anisotropy),
        "opacity_mean": float(opa.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="DATASET")
    parser.add_argument("--baseline_model_path", type=str, default="output/extent_ab/init_parity_baseline")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--out", type=str, default="output/extent_ab/gaussian_init_parity.json")
    args = parser.parse_args()

    report: dict = {"baseline": {}, "covariance_knn": {}, "baseline_compatible": {}}

    # ---------------------------------------------------------------- baseline
    sys.path.insert(0, str(BASELINE_ROOT))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render as baseline_render
    from utils.loss_utils import l1_loss as baseline_l1_loss, ssim as baseline_ssim
    from utils.general_utils import safe_state
    from argparse import ArgumentParser as BaselineArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    baseline_parser = BaselineArgumentParser()
    lp = ModelParams(baseline_parser)
    op = OptimizationParams(baseline_parser)
    pp = PipelineParams(baseline_parser)
    baseline_parser.add_argument('--debug_from', type=int, default=-1)
    baseline_parser.add_argument('--detect_anomaly', action='store_true', default=False)
    baseline_parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--quiet", action="store_true")
    baseline_parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--start_checkpoint", type=str, default=None)
    argv = ["-s", args.source_path, "-m", args.baseline_model_path, "--eval", "--iterations", str(args.steps)]
    parsed = baseline_parser.parse_args(argv)

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)
    dataset = lp.extract(parsed)
    opt = op.extract(parsed)
    pipe = pp.extract(parsed)

    baseline_model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, baseline_model)
    baseline_model.training_setup(opt)
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    train_cameras = scene.getTrainCameras()
    print(f"INIT_PARITY baseline_init n={baseline_model.get_xyz.shape[0]} cameras_extent={scene.cameras_extent}", flush=True)

    stats0 = _population_stats(baseline_model.get_scaling, baseline_model.get_opacity)
    report["baseline"]["0"] = stats0
    print(f"INIT_PARITY baseline step=0 anisotropy={stats0['anisotropy']}", flush=True)

    viewpoint_stack = list(train_cameras)
    viewpoint_indices = list(range(len(viewpoint_stack)))
    camera_sequence = []
    for _ in range(args.steps):
        if not viewpoint_stack:
            viewpoint_stack = list(train_cameras)
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)
        camera_sequence.append(cam)

    baseline_adc_events = []
    for step in range(1, args.steps + 1):
        cam = camera_sequence[step - 1]
        baseline_model.update_learning_rate(step)
        if step % 1000 == 0:
            baseline_model.oneupSHdegree()
        render_pkg = baseline_render(cam, baseline_model, pipe, background)
        image = render_pkg["render"]
        gt_image = cam.original_image.cuda()
        Ll1 = baseline_l1_loss(image, gt_image)
        ssim_value = baseline_ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        baseline_model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            radii = render_pkg["radii"]
            visible = render_pkg["visibility_filter"]
            baseline_model.max_radii2D[visible] = torch.max(baseline_model.max_radii2D[visible], radii[visible])
            baseline_model.add_densification_stats(render_pkg["viewspace_points"], visible)
            if step < opt.densify_until_iter and step > opt.densify_from_iter and step % opt.densification_interval == 0:
                before = int(baseline_model.get_xyz.shape[0])
                size_threshold = 20 if step > opt.opacity_reset_interval else None
                baseline_model.densify_and_prune(
                    opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii,
                )
                after = int(baseline_model.get_xyz.shape[0])
                baseline_adc_events.append({"step": step, "before": before, "after": after})
                print(f"INIT_PARITY baseline ADC step={step} before={before} after={after}", flush=True)
        baseline_model.optimizer.step()
        baseline_model.optimizer.zero_grad(set_to_none=True)
        if step in CHECKPOINT_STEPS:
            stats = _population_stats(baseline_model.get_scaling, baseline_model.get_opacity)
            report["baseline"][str(step)] = stats
            print(f"INIT_PARITY baseline step={step} count={stats['count']} anisotropy={stats['anisotropy']}", flush=True)
    report["baseline"]["adc_events"] = baseline_adc_events
    report["baseline"]["cameras_extent"] = float(scene.cameras_extent)

    # ---------------------------------------------------------------- OSN-GS common inputs
    eval_split = load_colmap_scene_with_eval_split(args.source_path, device="cuda", llffhold=8)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))

    def run_osn(mode: str) -> None:
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(gaussian_initialization_mode=mode), device="cuda")
        state = pipeline.initialize(eval_split.train_scene.initial_points, eval_split.train_scene.initial_colors)
        model = state.model
        model.spatial_lr_scale = float(scene.cameras_extent)
        model.training_setup(GaussianParameterGroups())

        stats0 = _population_stats(model.get_scaling, model.get_opacity)
        report[mode]["0"] = stats0
        print(f"INIT_PARITY osn[{mode}] step=0 count={stats0['count']} anisotropy={stats0['anisotropy']}", flush=True)

        density_config = TorchDensityControlConfig(
            densify_until_iter=15000, densification_interval=100,
        )
        adc_events = []
        for step in range(1, args.steps + 1):
            baseline_cam = camera_sequence[step - 1]
            match = [
                (cam, img) for cam, img in zip(eval_split.train_scene.cameras, eval_split.train_scene.images)
                if cam.image_name == baseline_cam.image_name
            ]
            osn_cam, osn_target = match[0]
            model.update_learning_rate(step)
            render_pkg = rasterizer.render(osn_cam, model, background)
            image = render_pkg["render"]
            target = osn_target.to(device="cuda", dtype=torch.float32)
            loss, _ = image_reconstruction_loss(image, target, opt.lambda_dssim)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                visibility = render_pkg.get("visibility_filter")
                radii = render_pkg.get("radii")
                update_max_radii(model, radii, visibility)
                add_densification_stats(model, render_pkg.get("viewspace_points"), visibility)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            if should_run_adc(step, density_config):
                before = len(model)
                report_adc = apply_adaptive_density_control(
                    model, density_config, scene_extent=float(scene.cameras_extent), iteration=step,
                )
                after = len(model)
                adc_events.append({"step": step, "before": before, "after": after, "cloned": report_adc.cloned, "split": report_adc.split, "pruned": report_adc.pruned})
                print(f"INIT_PARITY osn[{mode}] ADC step={step} before={before} after={after} cloned={report_adc.cloned} split={report_adc.split} pruned={report_adc.pruned}", flush=True)
            if step in CHECKPOINT_STEPS:
                stats = _population_stats(model.get_scaling, model.get_opacity)
                report[mode][str(step)] = stats
                print(f"INIT_PARITY osn[{mode}] step={step} count={stats['count']} anisotropy={stats['anisotropy']}", flush=True)
        report[mode]["adc_events"] = adc_events

    run_osn("covariance_knn")
    run_osn("baseline_compatible")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"INIT_PARITY wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
