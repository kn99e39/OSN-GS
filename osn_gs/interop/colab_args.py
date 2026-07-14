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
    """Register parametric surface-fitting arguments shared by both entrypoints."""

    parser.add_argument(
        "--surface_fit_mode",
        type=str,
        default="lsq",
        choices=("lsq", "idw"),
        help="Visible NURBS fitting: regularized least-squares with parameter correction, or the legacy inverse-distance seed only.",
    )
    parser.add_argument("--surface_degree_u", type=int, default=2, help="NURBS degree along u. Auto-clamped when a patch has few control points.")
    parser.add_argument("--surface_degree_v", type=int, default=2, help="NURBS degree along v. Auto-clamped when a patch has few control points.")
    parser.add_argument("--surface_fit_smoothness", type=float, default=1e-4, help="Second-difference regularization weight in the least-squares fit.")
    parser.add_argument("--surface_fit_tikhonov", type=float, default=1e-4, help="Tikhonov anchor weight toward the seed grid for sparsely covered control points.")
    parser.add_argument("--surface_fit_rounds", type=int, default=2, help="Least-squares fit / foot-point reprojection alternation rounds.")
    parser.add_argument("--surface_projection_iterations", type=int, default=4, help="Gauss-Newton refinement steps for foot-point UV projection.")


def surface_fit_config_kwargs(args: argparse.Namespace) -> dict:
    """Map surface-fitting CLI arguments onto TorchPipelineConfig fields."""

    return {
        "surface_fit_mode": str(args.surface_fit_mode),
        "surface_degree_u": max(1, int(args.surface_degree_u)),
        "surface_degree_v": max(1, int(args.surface_degree_v)),
        "surface_fit_smoothness": max(0.0, float(args.surface_fit_smoothness)),
        "surface_fit_tikhonov": max(0.0, float(args.surface_fit_tikhonov)),
        "surface_fit_rounds": max(1, int(args.surface_fit_rounds)),
        "surface_projection_iterations": max(0, int(args.surface_projection_iterations)),
    }


def build_osn_gs_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OSN-GS from notebook-compatible arguments.")

    parser.add_argument("-s", "--source_path", default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("-m", "--model_path", default="outputs/osn_gs", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=0)
    parser.add_argument("--densification_interval", type=int, default=0)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--adc_max_gaussians", type=int, default=0, help="Optional hard cap for Gaussian count during ADC. 0 means uncapped.")
    parser.add_argument("--adc_percent_dense", type=float, default=0.01)
    parser.add_argument("--adc_prune_opacity_threshold", type=float, default=0.005)
    parser.add_argument("--adc_split_samples", type=int, default=2)
    parser.add_argument("--adc_max_screen_size", type=float, default=20.0)
    parser.add_argument("--adc_max_scale_ratio", type=float, default=0.1)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--screen_size_prune_from_iter", type=int, default=3000)

    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
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
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_u", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_v", type=int, default=4)
    parser.add_argument("--visible_surface_resolution_scale", type=float, default=1.0)
    parser.add_argument("--max_surface_control_points", type=int, default=65536)
    parser.add_argument("--covariance_init", type=str, default="knn", choices=("knn", "constant"), help="Initialize Gaussian covariance scales from KNN spacing or a constant fallback.")
    parser.add_argument("--covariance_knn_chunk_size", type=int, default=0, help="KNN chunk for covariance initialization. 0 auto-selects from VRAM.")
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
    add_surface_fit_arguments(parser)
    parser.add_argument("--disable_voxel_surface_regions", action="store_true", help="Bypass pre-NURBS voxel surface-region placement.")
    parser.add_argument("--voxel_grid_resolution", type=int, default=16, help="Coarse voxel grid resolution per axis before NURBS fitting.")
    parser.add_argument("--disable_adaptive_voxel_density", action="store_true", help="Keep all occupied voxel cells at the coarse resolution.")
    parser.add_argument("--voxel_max_subdivision_depth", type=int, default=1, help="Density-driven voxel subdivision depth.")
    parser.add_argument("--voxel_density_quantile", type=float, default=0.75, help="Occupied-cell density quantile selected for subdivision.")
    parser.add_argument("--voxel_density_covariance_weight_cap", type=float, default=10.0, help="Maximum inverse-covariance contribution to voxel density.")
    parser.add_argument("--voxel_normal_knn", type=int, default=16, help="Neighbor count for local PCA normal estimation.")
    parser.add_argument("--voxel_boundary_angle_degrees", type=float, default=35.0, help="Normal-angle change used to mark voxel boundaries.")
    parser.add_argument("--voxel_min_points_per_region", type=int, default=1, help="Minimum Gaussians required to keep a voxel region.")
    parser.add_argument("--voxel_normal_chunk_size", type=int, default=4096, help="Regions processed per batched SVD/cdist chunk during voxel normal estimation.")
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    parser.add_argument(
        "--max_uncertain_gaussians",
        type=int,
        default=0,
        help="Cap the number of uncertain Gaussians. 0 keeps the full surface sample set.",
    )
    parser.add_argument(
        "--surface_update_interval",
        "--surface_rebuild_interval",
        dest="surface_update_interval",
        type=int,
        default=1000,
        help="Inspect persistent NURBS patch quality every N iterations; this does not globally rebuild voxels.",
    )
    parser.add_argument("--surface_residual_ratio_threshold", type=float, default=0.03)
    parser.add_argument("--surface_residual_patience", type=int, default=3)
    parser.add_argument("--surface_local_min_gaussians", type=int, default=64)
    parser.add_argument("--surface_local_min_component", type=int, default=16)
    parser.add_argument("--disable_local_surface_correction", action="store_true")
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--progress_log_interval", type=int, default=100, help="Print training progress every N iterations. 0 disables periodic progress logs.")
    parser.add_argument("--timing_log_interval", type=int, default=100, help="Print per-stage training timing every N iterations. 0 disables periodic timing logs.")
    parser.add_argument("--stream_url", type=str, default="", help="Optional WebSocket URL for live renderer snapshots.")
    parser.add_argument("--stream_server_host", type=str, default="127.0.0.1", help="Host for the trainer-owned local WebSocket server.")
    parser.add_argument("--stream_server_port", type=int, default=0, help="Trainer WebSocket server port. 0 disables server mode.")
    parser.add_argument("--stream_every", type=int, default=0, help="Stream every N iterations. 0 disables interval streaming.")
    parser.add_argument("--stream_iterations", nargs="*", type=int, default=[], help="Exact iterations to stream.")
    parser.add_argument("--stream_max_gaussians", type=int, default=0, help="Cap streamed Gaussians. 0 streams all Gaussians.")
    parser.add_argument("--stream_cache_dir", type=str, default="", help="Directory for cached stream snapshot JSON files.")
    parser.add_argument("--disable_stream_nurbs", action="store_true", help="Do not include NURBS payloads in streamed snapshots.")
    parser.add_argument("--disable_output_files", action="store_true", help="Skip PLY/NURBS/checkpoint file output; useful when streaming.")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Resume a v2 OSN-GS checkpoint.")
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Apply a conservative preset for 16GB-class GPUs: keep images on CPU, halve train resolution, and cap uncertain Gaussians.",
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

