from __future__ import annotations

"""OSN-GS Torch training CLI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene, load_colmap_scene_with_eval_split
from osn_gs.eval.held_out_metrics import (
    evaluate_held_out_cameras,
    final_iteration_opacity_reset_applies,
)
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.interop.colab_args import (
    add_surface_fit_arguments,
    surface_fit_config_kwargs,
)
from osn_gs.render.diff_gaussian_loader import validate_diff_gaussian_build_environment
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.utils.torch_ops import default_device, enable_timestamped_stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the OSN-GS torch framework.")
    parser.add_argument("-s", "--source_path", type=str, default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
    parser.add_argument(
        "--eval", action="store_true",
        help="Hold out a Graphdeco-identical test-camera split (see osn_gs/data/vendor/graphdeco_scene_split.py) "
             "and report held-out PSNR/SSIM after training, for a same-condition A/B against the baseline "
             "gaussian-splatting/train.py --eval run. Held-out cameras are never sampled during training.",
    )
    parser.add_argument("--llffhold", type=int, default=8, help="[--eval only] Hold out every Nth sorted image, matching baseline's own default.")
    parser.add_argument(
        "--resolution", type=int, default=-1,
        help="[--eval only] Matches baseline's -r/--resolution: -1 auto-downscales to <=1.6K width, "
             "1/2/4/8 divide by that exact factor, any other value is treated as a target width.",
    )
    parser.add_argument("--resolution_scale", type=float, default=1.0, help="[--eval only] Additional multiplicative resolution factor, matching baseline's own resolution_scale.")
    parser.add_argument("--output", type=str, default="outputs/osn_gs_torch", help="Output directory.")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument(
        "--image_device",
        type=str,
        default="",
        help="Device for storing training images. Images stay CPU-staged and only sampled views are transferred to the training device.",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--train_resolution_scale", type=int, default=1, help="Additional training-time render downscale.")
    parser.add_argument(
        "--position_lr_extent_mode",
        type=str,
        default="scene",
        choices=("scene", "calibration"),
        help="Position-LR scale: robust point-cloud scene extent (default) or Graphdeco camera calibration extent for A/B.",
    )
    parser.add_argument(
        "--gaussian_initialization_mode",
        type=str,
        default="baseline_compatible",
        choices=("baseline_compatible", "covariance_knn"),
        help="Trainable scale/rotation init for new visible Gaussians; see train.py for details.",
    )
    parser.add_argument("--canonical_covariance_knn", type=int, default=8)
    parser.add_argument("--canonical_construction_max_points", type=int, default=2048)
    parser.add_argument("--covariance_knn_chunk_size", type=int, default=0)
    parser.add_argument("--covariance_min_scale", type=float, default=1e-4)
    parser.add_argument("--covariance_max_scale_ratio", type=float, default=0.05)
    parser.add_argument("--covariance_scale_multiplier", type=float, default=1.0)
    add_surface_fit_arguments(parser)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    # Defaults reproduce the notebook's VRAM-safe recipe (original 3DGS ADC schedule)
    # so this CLI matches train.py / colab_train_3dgs.ipynb. Pass 0 to disable ADC.
    parser.add_argument("--densify_until_iter", type=int, default=15000, help="Run 3DGS-style ADC until this iteration. 0 disables ADC.")
    parser.add_argument("--densification_interval", type=int, default=100, help="Run 3DGS-style ADC every N iterations. 0 disables ADC.")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002, help="Screen-space gradient threshold for ADC clone/split.")
    parser.add_argument("--adc_max_gaussians", type=int, default=0, help="Optional hard cap for Gaussian count during ADC. 0 means uncapped.")
    parser.add_argument("--adc_percent_dense", type=float, default=0.01)
    parser.add_argument("--adc_prune_opacity_threshold", type=float, default=0.005)
    parser.add_argument("--adc_split_samples", type=int, default=2)
    parser.add_argument("--adc_max_screen_size", type=float, default=20.0)
    parser.add_argument("--adc_max_scale_ratio", type=float, default=0.1)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--adc_drop_survivor_gradients", action="store_true", help="A/B only: discard survivor gradients after ADC shape replacement to match Graphdeco lifecycle.")
    parser.add_argument("--screen_size_prune_from_iter", type=int, default=3000)
    parser.add_argument(
        "--surface_update_interval",
        "--surface_rebuild_interval",
        dest="surface_update_interval",
        type=int,
        default=1000,
        help="Inspect persistent NURBS patch quality every N iterations; this does not globally rebuild voxels.",
    )
    parser.add_argument("--surface_loss_patch_budget", type=int, default=16, help="NURBS patches evaluated per iteration. 0 evaluates all patches.")
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--progress_log_interval", type=int, default=100, help="Print training progress every N iterations. 0 disables periodic progress logs.")
    parser.add_argument("--timing_log_interval", type=int, default=10, help="Print per-stage training timing every N iterations. 0 disables periodic timing logs.")
    parser.add_argument("--skip_cuda_build_preflight", action="store_true", help="Skip the early MSVC/CUDA/Ninja readiness check before CUDA rasterizer loading.")
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    parser.add_argument("--stream_url", type=str, default="")
    parser.add_argument("--stream_every", type=int, default=1)
    parser.add_argument("--stream_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--stream_max_gaussians", type=int, default=0)
    parser.add_argument("--stream_cache_dir", type=str, default="", help="Directory for cached stream snapshot JSON files.")
    parser.add_argument("--stream_queue_size", type=int, default=2, help="Maximum pinned-memory snapshots awaiting serialization/I/O.")
    parser.add_argument("--disable_stream_nurbs", action="store_true")
    parser.add_argument("--disable_output_files", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument(
        "--low_vram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conservative preset for 16GB-class GPUs (on by default): keep images on CPU and halve train resolution. Pass --no-low_vram for a full-resolution run.",
    )
    return parser


def main() -> None:
    enable_timestamped_stdout()
    args = build_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    image_device = args.image_device or ("auto" if device == "cuda" else device)
    if args.low_vram and not args.image_device:
        image_device = "cpu"
    print(f"OSN-GS device: train={device}, images={image_device}", flush=True)
    if not args.disable_cuda_rasterizer and not args.skip_cuda_build_preflight:
        preflight = validate_diff_gaussian_build_environment()
        print(
            "OSN-GS CUDA build preflight: "
            f"cl={preflight['compiler']} nvcc={preflight['nvcc']}",
            flush=True,
        )
    train_resolution_scale = max(1, int(args.train_resolution_scale))
    if args.eval and train_resolution_scale > 1:
        print(
            f"OSN-GS --eval: ignoring train_resolution_scale={train_resolution_scale} "
            "(--low_vram or --train_resolution_scale would otherwise apply an EXTRA "
            "training-time downscale on top of the loader's own baseline-matched "
            "resolution, breaking the same-resolution A/B).",
            flush=True,
        )
        train_resolution_scale = 1

    densify_grad_threshold = float(args.densify_grad_threshold)
    if densify_grad_threshold <= 0.0:
        densify_grad_threshold = TorchDensityControlConfig().densify_grad_threshold
    density_control_config = TorchDensityControlConfig(
        densify_from_iter=max(0, int(args.densify_from_iter)),
        densify_until_iter=max(0, int(args.densify_until_iter)),
        densification_interval=max(0, int(args.densification_interval)),
        densify_grad_threshold=densify_grad_threshold,
        max_gaussians=max(0, int(getattr(args, "adc_max_gaussians", 0))),
        percent_dense=max(0.0, float(args.adc_percent_dense)),
        prune_opacity_threshold=max(0.0, float(args.adc_prune_opacity_threshold)),
        split_samples=max(1, int(args.adc_split_samples)),
        max_screen_size=max(0.0, float(args.adc_max_screen_size)),
        max_scale_ratio=max(0.0, float(args.adc_max_scale_ratio)),
        opacity_reset_interval=max(0, int(args.opacity_reset_interval)),
        screen_size_prune_from_iter=max(0, int(args.screen_size_prune_from_iter)),
        preserve_adc_gradients=not bool(args.adc_drop_survivor_gradients),
    )

    pipeline_config = TorchPipelineConfig(
        gaussian_initialization_mode=str(args.gaussian_initialization_mode),
        canonical_covariance_knn=max(3, int(args.canonical_covariance_knn)),
        canonical_construction_max_points=max(
            16, int(args.canonical_construction_max_points)
        ),
        covariance_knn_chunk_size=max(0, int(args.covariance_knn_chunk_size)),
        covariance_min_scale=max(0.0, float(args.covariance_min_scale)),
        covariance_max_scale_ratio=max(
            0.0, float(args.covariance_max_scale_ratio)
        ),
        covariance_scale_multiplier=max(
            0.0, float(args.covariance_scale_multiplier)
        ),
        **surface_fit_config_kwargs(args),
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=max(0, int(args.surface_update_interval)),
        surface_loss_patch_budget=max(0, int(args.surface_loss_patch_budget)),
        density_control_interval=args.density_control_interval,
        save_interval=args.save_interval,
        save_iterations=(),
        progress_log_interval=args.progress_log_interval,
        timing_log_interval=args.timing_log_interval,
        stream_url=args.stream_url,
        stream_every=max(0, int(args.stream_every)),
        stream_iterations=tuple(sorted({int(value) for value in args.stream_iterations if int(value) > 0})),
        stream_max_gaussians=max(0, int(args.stream_max_gaussians)),
        stream_cache_dir=args.stream_cache_dir,
        stream_queue_size=max(1, int(args.stream_queue_size)),
        stream_nurbs=not args.disable_stream_nurbs,
        write_output_files=not args.disable_output_files,
        resume_checkpoint=args.resume_checkpoint,
        prefer_cuda=device == "cuda",
        train_resolution_scale=train_resolution_scale,
        position_lr_extent_mode=args.position_lr_extent_mode,
        density_control=density_control_config,
    )

    print(
        "OSN-GS surface loss: "
        f"patch_budget={training_config.surface_loss_patch_budget} (0=all patches)",
        flush=True,
    )
    rasterizer_config = GaussianRasterizerConfig(prefer_cuda=not args.disable_cuda_rasterizer)
    trainer = TorchOSNGSTrainer(
        pipeline_config=pipeline_config,
        training_config=training_config,
        rasterizer_config=rasterizer_config,
        device=device,
    )

    if not args.source_path:
        raise ValueError("OSN-GS requires --source_path/-s pointing to a COLMAP dataset root.")

    eval_split = None
    if args.eval:
        eval_split = load_colmap_scene_with_eval_split(
            args.source_path,
            device=device,
            image_dir_name=args.images,
            sparse_dir_name=args.sparse_dir,
            resolution=args.resolution,
            resolution_scale=args.resolution_scale,
            eval=True,
            llffhold=args.llffhold,
        )
        scene = eval_split.train_scene
    else:
        scene = load_colmap_scene(
            args.source_path,
            device=device,
            image_device=image_device,
            image_dir_name=args.images,
            sparse_dir_name=args.sparse_dir,
            image_downscale=args.image_downscale,
            max_images=args.max_images,
        )

    result = trainer.train(scene, args.output)
    print(
        "OSN-GS torch training complete: "
        f"iteration={result.state.iteration}, "
        f"loss={result.state.last_loss:.6f}, "
        f"psnr={result.state.last_psnr:.3f}, "
        f"gaussians={len(result.state.model)}, "
        f"uncertain={int(result.state.model.is_uncertain.sum().item())}, "
        f"output={result.output_dir}"
    )

    if eval_split is not None:
        post_opacity_reset = final_iteration_opacity_reset_applies(
            result.state.iteration,
            trainer.training_config.density_control.opacity_reset_interval,
            trainer.training_config.density_control.densify_until_iter,
        )
        if post_opacity_reset:
            print(
                "OSN-GS held-out eval warning: final iteration coincides with an opacity reset; "
                "the reported metrics describe the reset model and are not comparable to a pre-reset checkpoint.",
                flush=True,
            )
        held_out = evaluate_held_out_cameras(
            trainer.rasterizer, result.state.model, eval_split.test_cameras, eval_split.test_images, device=device,
        )
        print(
            "OSN-GS held-out eval "
            f"(cameras={held_out['camera_count']}, resolution={eval_split.resolution}, "
            f"llffhold={args.llffhold}): "
            f"psnr_mean={held_out['psnr_mean']:.3f} ssim_mean={held_out['ssim_mean']:.4f}",
            flush=True,
        )
        if not args.disable_output_files:
            import json

            report_path = Path(result.output_dir) / "held_out_eval.json"
            report_path.write_text(
                json.dumps(
                    {
                        "iteration": result.state.iteration,
                        "resolution": list(eval_split.resolution),
                        "downscale_factor": eval_split.downscale_factor,
                        "llffhold": args.llffhold,
                        "post_opacity_reset": post_opacity_reset,
                        **held_out,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"OSN-GS held-out eval report: {report_path}", flush=True)


if __name__ == "__main__":
    main()




