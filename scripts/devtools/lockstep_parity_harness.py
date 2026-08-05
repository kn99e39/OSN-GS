"""Worklog 62: true lockstep parity harness between the Graphdeco baseline's
own training core and OSN-GS's, from IDENTICAL initial Gaussian tensors and
an IDENTICAL camera sequence, with ADC disabled on both sides.

Since ADC is off, population size never changes -- Gaussian index i means
the SAME Gaussian in both models for the whole run (tensors are transplanted
directly from the baseline's own post-init state into OSN-GS's model), so
per-step comparisons are exact, not aggregate-only.

Neither side's render()/loss()/optimizer.step() is reimplemented -- both run
through their own real, unmodified production code paths (baseline's own
`gaussian_renderer.render` + `utils.loss_utils`; OSN-GS's own
`OSNGaussianRasterizer.render` + `image_reconstruction_loss`). Surface
reconstruction and reliability code is never touched or invoked.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Import every OSN-GS module needed BEFORE gaussian-splatting/ is added to
# sys.path below -- otherwise its same-named submodules (e.g. `scene`,
# `utils`) shadow/collide with osn_gs's own imports (circular-import errors).
import osn_gs.core.torch_pipeline  # noqa: F401 -- resolves osn_gs's own internal circular-import order first
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.gaussian.torch_model import GaussianParameterGroups, TorchGaussianModel
from osn_gs.losses.torch_losses import image_reconstruction_loss
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "gaussian-splatting"
CHECKPOINT_STEPS = {1, 2, 10, 50, 100, 300, 600}


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _population_stats(scaling: torch.Tensor, opacity: torch.Tensor, xyz: torch.Tensor, rotation: torch.Tensor) -> dict:
    with torch.no_grad():
        scale = scaling.detach().cpu().numpy()
        sorted_scale = np.sort(scale, axis=1)
        s_min, s_mid, s_max = sorted_scale[:, 0], sorted_scale[:, 1], sorted_scale[:, 2]
        anisotropy = s_max / np.clip(s_min, 1e-12, None)
        opa = opacity.detach().reshape(-1).cpu().numpy()
    return {
        "s_min": _percentiles(s_min), "s_mid": _percentiles(s_mid), "s_max": _percentiles(s_max),
        "anisotropy": _percentiles(anisotropy),
        "opacity_mean": float(opa.mean()),
        "xyz_norm_mean": float(xyz.detach().norm(dim=1).mean().cpu()),
        "rotation_norm_mean": float(rotation.detach().norm(dim=1).mean().cpu()),
    }


def _grad_stats(tensor: torch.Tensor | None) -> dict:
    if tensor is None or tensor.grad is None:
        return {"norm": None, "mean_abs": None, "nonzero_rows": None}
    grad = tensor.grad.detach()
    row_norm = grad.norm(dim=tuple(range(1, grad.dim()))) if grad.dim() > 1 else grad.abs()
    return {
        "norm": float(grad.norm().cpu()),
        "mean_abs": float(grad.abs().mean().cpu()),
        "nonzero_rows": int((row_norm > 0).sum().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="DATASET")
    parser.add_argument("--baseline_model_path", type=str, default="output/extent_ab/lockstep_baseline")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    # ---------------------------------------------------------------- baseline setup
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
    argv = [
        "-s", args.source_path, "-m", args.baseline_model_path, "--eval",
        "--iterations", str(args.steps), "--densify_until_iter", "0",
    ]
    parsed = baseline_parser.parse_args(argv)

    # Exact same call order as the real train.py::training(): safe_state (seeds
    # random/np/torch to 0) -> GaussianModel -> Scene (which shuffles train/test
    # cameras using the just-seeded `random` global state) -> training_setup.
    # Preserving this order is what makes the later camera-sequence replay valid.
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    dataset = lp.extract(parsed)
    opt = op.extract(parsed)
    pipe = pp.extract(parsed)

    baseline_model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, baseline_model)
    baseline_model.training_setup(opt)

    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    train_cameras = scene.getTrainCameras()  # already shuffled by Scene.__init__
    print(f"LOCKSTEP baseline_init n={baseline_model.get_xyz.shape[0]} cameras_extent={scene.cameras_extent} "
          f"train_cameras={len(train_cameras)}", flush=True)

    # ---------------------------------------------------------------- OSN-GS setup
    eval_split = load_colmap_scene_with_eval_split(args.source_path, device="cuda", llffhold=8)
    osn_by_name = {
        cam.image_name: (cam, img)
        for cam, img in zip(eval_split.train_scene.cameras, eval_split.train_scene.images)
    }

    n = int(baseline_model.get_xyz.shape[0])
    osn_model = TorchGaussianModel(sh_degree=dataset.sh_degree, device="cuda")
    osn_model.replace_tensors(
        xyz=baseline_model._xyz.detach().clone(),
        features_dc=baseline_model._features_dc.detach().clone(),
        features_rest=baseline_model._features_rest.detach().clone(),
        opacity=baseline_model._opacity.detach().clone(),
        scaling=baseline_model._scaling.detach().clone(),
        rotation=baseline_model._rotation.detach().clone(),
        uncertain_confidence=torch.full((n, 1), 12.0, dtype=torch.float32, device="cuda"),
        uncertain_mask=torch.zeros((n,), dtype=torch.bool, device="cuda"),
        surface_uv=torch.zeros((n, 2), dtype=torch.float32, device="cuda"),
        cluster_ids=torch.full((n,), -1, dtype=torch.long, device="cuda"),
        stable_gaussian_ids=torch.arange(n, dtype=torch.long, device="cuda"),
    )
    osn_model.spatial_lr_scale = float(scene.cameras_extent)  # calibration-extent basis, matches baseline exactly
    osn_model.training_setup(GaussianParameterGroups())

    # Sanity: transplanted tensors are byte-identical at init.
    for name, b, o in (
        ("xyz", baseline_model._xyz, osn_model._xyz),
        ("scaling", baseline_model._scaling, osn_model._scaling),
        ("rotation", baseline_model._rotation, osn_model._rotation),
        ("opacity", baseline_model._opacity, osn_model._opacity),
    ):
        same = bool(torch.equal(b.detach(), o.detach()))
        print(f"LOCKSTEP init_tensor_identical {name}={same}", flush=True)

    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))

    # ---------------------------------------------------------------- camera sequence
    # Exact replica of train.py's own pop-based per-iteration camera selection,
    # continuing to consume the SAME `random` global state Scene.__init__ already
    # advanced (shuffle) -- this is not a re-seeded independent draw.
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

    missing = [c.image_name for c in camera_sequence if c.image_name not in osn_by_name]
    print(f"LOCKSTEP camera_sequence_built n={len(camera_sequence)} missing_in_osn_side={len(missing)}", flush=True)

    # One-time camera-parameter parity check on the first camera actually used.
    first_b_cam = camera_sequence[0]
    first_o_cam, first_o_img = osn_by_name[first_b_cam.image_name]
    print(
        f"LOCKSTEP CAMERA_CHECK name={first_b_cam.image_name} "
        f"b_FoVx={first_b_cam.FoVx:.10f} o_FoVx={first_o_cam.FoVx:.10f} "
        f"b_FoVy={first_b_cam.FoVy:.10f} o_FoVy={first_o_cam.FoVy:.10f} "
        f"b_H={first_b_cam.image_height} o_H={first_o_cam.image_height} "
        f"b_W={first_b_cam.image_width} o_W={first_o_cam.image_width}",
        flush=True,
    )
    b_wvt = first_b_cam.world_view_transform.detach().cpu()
    o_wvt = first_o_cam.world_view_transform.detach().cpu()
    b_fpt = first_b_cam.full_proj_transform.detach().cpu()
    o_fpt = first_o_cam.full_proj_transform.detach().cpu()
    b_cc = first_b_cam.camera_center.detach().cpu()
    o_cc = first_o_cam.camera_center.detach().cpu()
    print(
        f"LOCKSTEP CAMERA_CHECK world_view_transform_max_abs_diff={float((b_wvt - o_wvt).abs().max()):.8g} "
        f"full_proj_transform_max_abs_diff={float((b_fpt - o_fpt).abs().max()):.8g} "
        f"camera_center_max_abs_diff={float((b_cc - o_cc).abs().max()):.8g}",
        flush=True,
    )
    b_gt = first_b_cam.original_image.detach().cpu()
    o_gt = first_o_img.detach().cpu()
    same_shape = tuple(b_gt.shape) == tuple(o_gt.shape)
    print(
        f"LOCKSTEP CAMERA_CHECK gt_image_shape_match={same_shape} b_shape={tuple(b_gt.shape)} o_shape={tuple(o_gt.shape)} "
        f"gt_mean_abs_diff={float((b_gt - o_gt).abs().mean()) if same_shape else 'N/A'}",
        flush=True,
    )

    # ---------------------------------------------------------------- lockstep loop
    for step in range(1, args.steps + 1):
        baseline_cam = camera_sequence[step - 1]
        osn_cam, osn_target = osn_by_name[baseline_cam.image_name]

        baseline_model.update_learning_rate(step)
        osn_model.update_learning_rate(step)

        # --- baseline step ---
        render_pkg = baseline_render(baseline_cam, baseline_model, pipe, background)
        image = render_pkg["render"]
        gt_image = baseline_cam.original_image.cuda()
        Ll1 = baseline_l1_loss(image, gt_image)
        ssim_value = baseline_ssim(image, gt_image)
        b_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        baseline_model.optimizer.zero_grad(set_to_none=True)
        b_loss.backward()
        b_xyz_grad = _grad_stats(baseline_model._xyz)
        b_scaling_grad = _grad_stats(baseline_model._scaling)
        b_rotation_grad = _grad_stats(baseline_model._rotation)
        b_opacity_grad = _grad_stats(baseline_model._opacity)
        b_radii = render_pkg["radii"].detach()
        b_visible = render_pkg["visibility_filter"]
        baseline_model.max_radii2D[b_visible] = torch.max(baseline_model.max_radii2D[b_visible], b_radii[b_visible])
        baseline_model.optimizer.step()

        # --- OSN-GS step ---
        osn_pkg = rasterizer.render(osn_cam, osn_model, background)
        osn_image = osn_pkg["render"]
        osn_target_gpu = osn_target.to(device="cuda", dtype=torch.float32)
        osn_loss, osn_mse = image_reconstruction_loss(osn_image, osn_target_gpu, opt.lambda_dssim)
        osn_model.optimizer.zero_grad(set_to_none=True)
        osn_loss.backward()
        o_xyz_grad = _grad_stats(osn_model._xyz)
        o_scaling_grad = _grad_stats(osn_model._scaling)
        o_rotation_grad = _grad_stats(osn_model._rotation)
        o_opacity_grad = _grad_stats(osn_model._opacity)
        o_radii = osn_pkg["radii"].detach()
        osn_model.max_radii2D = torch.maximum(osn_model.max_radii2D, o_radii.float())
        osn_model.optimizer.step()

        if step in CHECKPOINT_STEPS:
            b_stats = _population_stats(baseline_model.get_scaling, baseline_model.get_opacity, baseline_model.get_xyz, baseline_model.get_rotation)
            o_stats = _population_stats(osn_model.get_scaling, osn_model.get_opacity, osn_model.get_xyz, osn_model.get_rotation)
            with torch.no_grad():
                xyz_diff = float((baseline_model._xyz.detach() - osn_model._xyz.detach()).norm(dim=1).mean().cpu())
                scaling_diff = float((baseline_model._scaling.detach() - osn_model._scaling.detach()).norm(dim=1).mean().cpu())
                rotation_diff = float((baseline_model._rotation.detach() - osn_model._rotation.detach()).norm(dim=1).mean().cpu())
                opacity_diff = float((baseline_model._opacity.detach() - osn_model._opacity.detach()).abs().mean().cpu())
                image_diff = float((image.detach() - osn_image.detach()).abs().mean().cpu())
            print(
                f"LOCKSTEP step={step} camera={baseline_cam.image_name} "
                f"b_loss={float(b_loss.detach().cpu()):.8f} o_loss={float(osn_loss.detach().cpu()):.8f} "
                f"image_mean_abs_diff={image_diff:.8g}",
                flush=True,
            )
            print(
                f"LOCKSTEP step={step} GRAD xyz b_norm={b_xyz_grad['norm']} o_norm={o_xyz_grad['norm']} "
                f"scaling b_norm={b_scaling_grad['norm']} o_norm={o_scaling_grad['norm']} "
                f"rotation b_norm={b_rotation_grad['norm']} o_norm={o_rotation_grad['norm']} "
                f"opacity b_norm={b_opacity_grad['norm']} o_norm={o_opacity_grad['norm']}",
                flush=True,
            )
            print(
                f"LOCKSTEP step={step} MEAN_TENSOR_DIFF xyz={xyz_diff:.8g} scaling={scaling_diff:.8g} "
                f"rotation={rotation_diff:.8g} opacity={opacity_diff:.8g}",
                flush=True,
            )
            print(
                f"LOCKSTEP step={step} POP_STATS baseline s_min={b_stats['s_min']} anisotropy={b_stats['anisotropy']} "
                f"opacity_mean={b_stats['opacity_mean']:.6g}",
                flush=True,
            )
            print(
                f"LOCKSTEP step={step} POP_STATS osn_gs   s_min={o_stats['s_min']} anisotropy={o_stats['anisotropy']} "
                f"opacity_mean={o_stats['opacity_mean']:.6g}",
                flush=True,
            )
            b_screen = baseline_model.max_radii2D.detach().cpu().numpy()
            o_screen = osn_model.max_radii2D.detach().cpu().numpy()
            print(
                f"LOCKSTEP step={step} SCREEN_RADII baseline={_percentiles(b_screen[b_screen > 0])} "
                f"osn_gs={_percentiles(o_screen[o_screen > 0])}",
                flush=True,
            )
            b_accum = (baseline_model.xyz_gradient_accum / baseline_model.denom.clamp(min=1)).detach().cpu().numpy().reshape(-1)
            o_accum = (osn_model.xyz_gradient_accum / osn_model.denom.clamp(min=1)).detach().cpu().numpy().reshape(-1)
            print(
                f"LOCKSTEP step={step} DENSIF_GRAD_ACCUM baseline={_percentiles(b_accum)} osn_gs={_percentiles(o_accum)}",
                flush=True,
            )

        # densification gradient accumulator (both sides, every step, matching each
        # framework's own real accumulation call -- not reimplemented here).
        baseline_model.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"])
        osn_model.xyz_gradient_accum = getattr(osn_model, "xyz_gradient_accum", torch.zeros((n, 1), device="cuda"))
        osn_model.denom = getattr(osn_model, "denom", torch.zeros((n, 1), device="cuda"))
        osn_visible = torch.nonzero(o_radii > 0, as_tuple=False).reshape(-1)
        if osn_pkg["viewspace_points"].grad is not None:
            osn_model.xyz_gradient_accum[osn_visible] += torch.norm(
                osn_pkg["viewspace_points"].grad[osn_visible, :2], dim=-1, keepdim=True,
            )
            osn_model.denom[osn_visible] += 1


if __name__ == "__main__":
    main()
