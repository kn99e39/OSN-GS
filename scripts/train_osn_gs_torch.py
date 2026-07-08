from __future__ import annotations

"""OSN-GS Torch training CLI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.utils.torch_ops import default_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the OSN-GS torch framework.")
    parser.add_argument("-s", "--source_path", type=str, default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
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
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_u", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_v", type=int, default=4)
    parser.add_argument("--visible_surface_resolution_scale", type=float, default=1.0)
    parser.add_argument("--covariance_init", type=str, default="knn", choices=("knn", "constant"))
    parser.add_argument("--covariance_knn_chunk_size", type=int, default=0)
    parser.add_argument("--covariance_min_scale", type=float, default=1e-4)
    parser.add_argument("--covariance_max_scale_ratio", type=float, default=0.05)
    parser.add_argument("--covariance_scale_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--visible_surface_fit_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Workspace device for visible NURBS fitting. cpu lowers VRAM use; cuda can be faster.",
    )
    parser.add_argument(
        "--visible_surface_fit_chunk_size",
        type=int,
        default=0,
        help="NURBS grid samples per fitting chunk. 0 auto-selects once from available VRAM at startup.",
    )
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    parser.add_argument("--max_uncertain_gaussians", type=int, default=0, help="Cap the number of uncertain Gaussians.")
    parser.add_argument("--densify_until_iter", type=int, default=0, help="Run 3DGS-style ADC until this iteration. 0 disables ADC.")
    parser.add_argument("--densification_interval", type=int, default=0, help="Run 3DGS-style ADC every N iterations. 0 disables ADC.")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002, help="Screen-space gradient threshold for ADC clone/split.")
    parser.add_argument("--adc_max_gaussians", type=int, default=0, help="Optional hard cap for Gaussian count during ADC. 0 means uncapped.")
    parser.add_argument("--surface_rebuild_interval", type=int, default=1000)
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--progress_log_interval", type=int, default=100, help="Print training progress every N iterations. 0 disables periodic progress logs.")
    parser.add_argument("--timing_log_interval", type=int, default=100, help="Print per-stage training timing every N iterations. 0 disables periodic timing logs.")
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    parser.add_argument("--stream_cache_dir", type=str, default="", help="Directory for cached stream snapshot JSON files.")
    parser.add_argument("--low_vram", action="store_true", help="Apply a conservative 16GB VRAM preset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    image_device = args.image_device or ("auto" if device == "cuda" else device)
    if args.low_vram and not args.image_device:
        image_device = "cpu"
    print(f"OSN-GS device: train={device}, images={image_device}", flush=True)
    train_resolution_scale = max(1, int(args.train_resolution_scale))
    uncertain_samples_u = args.uncertain_samples_u
    uncertain_samples_v = args.uncertain_samples_v
    max_uncertain_gaussians = max(0, int(args.max_uncertain_gaussians))
    if args.low_vram:
        train_resolution_scale = max(train_resolution_scale, 2)
        uncertain_samples_u = min(uncertain_samples_u, 8)
        uncertain_samples_v = min(uncertain_samples_v, 2)
        if max_uncertain_gaussians == 0:
            max_uncertain_gaussians = 128

    densify_grad_threshold = float(args.densify_grad_threshold)
    if densify_grad_threshold <= 0.0:
        densify_grad_threshold = TorchDensityControlConfig().densify_grad_threshold
    density_control_config = TorchDensityControlConfig(
        densify_until_iter=max(0, int(args.densify_until_iter)),
        densification_interval=max(0, int(args.densification_interval)),
        densify_grad_threshold=densify_grad_threshold,
        max_gaussians=max(0, int(getattr(args, "adc_max_gaussians", 0))),
    )

    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        visible_surface_resolution_u=args.visible_surface_resolution_u,
        visible_surface_resolution_v=args.visible_surface_resolution_v,
        visible_surface_resolution_scale=args.visible_surface_resolution_scale,
        covariance_init=args.covariance_init,
        covariance_knn_chunk_size=args.covariance_knn_chunk_size,
        covariance_min_scale=args.covariance_min_scale,
        covariance_max_scale_ratio=args.covariance_max_scale_ratio,
        covariance_scale_multiplier=args.covariance_scale_multiplier,
        visible_surface_fit_device=args.visible_surface_fit_device,
        visible_surface_fit_chunk_size=args.visible_surface_fit_chunk_size,
        uncertain_samples_u=uncertain_samples_u,
        uncertain_samples_v=uncertain_samples_v,
        max_uncertain_gaussians=max_uncertain_gaussians,
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=args.surface_rebuild_interval,
        density_control_interval=args.density_control_interval,
        save_interval=args.save_interval,
        save_iterations=(),
        progress_log_interval=args.progress_log_interval,
        timing_log_interval=args.timing_log_interval,
        stream_cache_dir=args.stream_cache_dir,
        prefer_cuda=device == "cuda",
        train_resolution_scale=train_resolution_scale,
        density_control=density_control_config,
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


if __name__ == "__main__":
    main()


