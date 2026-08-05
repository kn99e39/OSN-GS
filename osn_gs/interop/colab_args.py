from __future__ import annotations

"""Argument helpers for Colab/notebook entrypoints.

The `3DGS_Renderer/colab_train_3dgs.ipynb` notebook was originally written
around Graphdeco-style `train.py` arguments. OSN-GS has its own Torch trainer,
so this module translates the shared subset of notebook arguments into OSN-GS
configuration without forcing the notebook to know every internal class.
"""

import argparse
from pathlib import Path


def add_surface_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the projection knob used after canonical materialization."""

    parser.add_argument(
        "--surface_projection_iterations",
        type=int,
        default=4,
        help="Gauss-Newton refinement steps for canonical NURBS foot-point UV projection.",
    )


def surface_fit_config_kwargs(args: argparse.Namespace) -> dict:
    """Map the remaining canonical projection argument onto pipeline config."""

    return {
        "surface_projection_iterations": max(
            0, int(args.surface_projection_iterations)
        )
    }

def build_osn_gs_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OSN-GS from notebook-compatible arguments.")

    parser.add_argument("-s", "--source_path", default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("-m", "--model_path", default="outputs/osn_gs", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--densify_from_iter", type=int, default=500)
    # Defaults reproduce the notebook's VRAM-safe recipe so a bare CLI run matches
    # colab_train_3dgs.ipynb exactly. ADC is on by default (original 3DGS schedule);
    # pass 0 to disable. See docs/README.md "Notebook/CLI Training Parity".
    parser.add_argument("--densify_until_iter", type=int, default=15000)
    parser.add_argument("--densification_interval", type=int, default=100)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--adc_max_gaussians", type=int, default=0, help="Optional hard cap for Gaussian count during ADC. 0 means uncapped.")
    parser.add_argument("--adc_percent_dense", type=float, default=0.01)
    parser.add_argument("--adc_prune_opacity_threshold", type=float, default=0.005)
    parser.add_argument("--adc_split_samples", type=int, default=2)
    parser.add_argument("--adc_max_screen_size", type=float, default=20.0)
    parser.add_argument("--adc_max_scale_ratio", type=float, default=0.1)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--adc_drop_survivor_gradients", action="store_true", help="A/B only: discard survivor gradients after ADC shape replacement to match Graphdeco lifecycle.")
    parser.add_argument("--screen_size_prune_from_iter", type=int, default=3000)

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
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument(
        "--image_device",
        type=str,
        default="",
        help="Device that stores training images. Images stay CPU-staged and only sampled views are transferred to the training device.",
    )
    parser.add_argument(
        "--train_resolution_scale",
        type=int,
        default=1,
        help="Additional training-time render downscale. 2 means half resolution in each axis.",
    )
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
        help=(
            "Trainable scale/rotation init for newly created visible Gaussians. "
            "baseline_compatible (default) matches Graphdeco's isotropic "
            "distCUDA2-based create_from_pcd exactly. covariance_knn is the "
            "experimental local-PCA planar-surfel init (worklog 64); it does "
            "not affect the separate covariance always used for visible "
            "surface construction/reliability."
        ),
    )
    parser.add_argument("--canonical_covariance_knn", type=int, default=8, help="Neighbor count for canonical local-PCA planar covariance initialization.")
    parser.add_argument("--canonical_construction_max_points", type=int, default=2048, help="Maximum deterministic voxel-center samples used by canonical O(N^2) topology construction.")
    parser.add_argument("--covariance_knn_chunk_size", type=int, default=0, help="KNN chunk for canonical covariance initialization. 0 auto-selects from VRAM.")
    parser.add_argument("--covariance_min_scale", type=float, default=1e-4)
    parser.add_argument("--covariance_max_scale_ratio", type=float, default=0.05)
    parser.add_argument("--covariance_scale_multiplier", type=float, default=1.0)
    add_surface_fit_arguments(parser)
    parser.add_argument(
        "--surface_update_interval",
        "--surface_rebuild_interval",
        dest="surface_update_interval",
        type=int,
        default=1000,
        help="Rebuild visible NURBS through the canonical Gaussian pipeline every N iterations. 0 disables rebuild.",
    )
    parser.add_argument(
        "--visible_nurbs_update_schedule",
        type=str,
        default="initialize",
        choices=("initialize", "adc_post_commit", "disabled"),
        help="Visible NURBS schedule: current initialization behavior, detached rebuild after structural ADC commits, or Gaussian-only control.",
    )
    parser.add_argument("--surface_loss_patch_budget", type=int, default=16, help="NURBS patches evaluated per iteration. 0 evaluates all patches.")
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--progress_log_interval", type=int, default=100, help="Print training progress every N iterations. 0 disables periodic progress logs.")
    parser.add_argument("--timing_log_interval", type=int, default=10, help="Print per-stage training timing every N iterations. 0 disables periodic timing logs.")
    parser.add_argument("--stream_url", type=str, default="", help="Optional WebSocket URL for live renderer snapshots.")
    parser.add_argument("--stream_every", type=int, default=1, help="Broadcast every N iterations; default 1 broadcasts each iteration.")
    parser.add_argument("--stream_iterations", nargs="*", type=int, default=[], help="Exact iterations to stream.")
    parser.add_argument("--stream_max_gaussians", type=int, default=0, help="Cap streamed Gaussians. 0 streams all Gaussians.")
    parser.add_argument("--stream_cache_dir", type=str, default="", help="Directory for cached stream snapshot JSON files.")
    parser.add_argument("--stream_queue_size", type=int, default=2, help="Maximum pinned-memory snapshots awaiting serialization/I/O.")
    parser.add_argument("--disable_stream_nurbs", action="store_true", help="Do not include NURBS payloads in streamed snapshots.")
    parser.add_argument("--disable_output_files", action="store_true", help="Skip PLY/NURBS/checkpoint file output; useful when streaming.")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Resume a v2 OSN-GS checkpoint.")
    parser.add_argument("--skip_cuda_build_preflight", action="store_true", help="Skip the early MSVC/CUDA/Ninja readiness check before CUDA rasterizer loading.")
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    parser.add_argument(
        "--low_vram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conservative preset for 16GB-class GPUs (on by default): keep images on CPU and halve train resolution. Pass --no-low_vram for a full-resolution run.",
    )
    return parser


def save_iterations_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in args.save_iterations if int(value) > 0}))


def save_interval_from_args(args: argparse.Namespace) -> int:
    if sorted({int(value) for value in args.save_iterations if int(value) > 0}):
        return 0
    return max(1, int(args.iterations))


def output_dir_from_args(args: argparse.Namespace) -> Path:
    return Path(args.model_path)
