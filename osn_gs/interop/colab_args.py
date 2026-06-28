from __future__ import annotations

"""Argument helpers for Colab/notebook entrypoints.

The `3DGS_Renderer/colab_train_3dgs.ipynb` notebook was originally written
around Graphdeco-style `train.py` arguments. OSN-GS has its own Torch trainer,
so this module translates the shared subset of notebook arguments into OSN-GS
configuration without forcing the notebook to know every internal class.
"""

import argparse
from pathlib import Path


def build_osn_gs_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OSN-GS from notebook-compatible arguments.")

    # Graphdeco-compatible aliases. `source_path` points to a COLMAP/3DGS scene
    # when OSN-GS should train from real images instead of the synthetic scene.
    parser.add_argument("-s", "--source_path", default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("-m", "--model_path", default="outputs/osn_gs", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--densify_until_iter", type=int, default=0)
    parser.add_argument("--densification_interval", type=int, default=0)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0)

    # Native OSN-GS options.
    parser.add_argument("--scene_npz", type=str, default="", help="Optional NPZ scene with points/colors/images.")
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--point_count", type=int, default=48)
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    parser.add_argument("--surface_rebuild_interval", type=int, default=1000)
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    return parser


def save_interval_from_args(args: argparse.Namespace) -> int:
    """Choose a single periodic save interval from Graphdeco-style save iterations."""

    save_iterations = sorted({int(value) for value in args.save_iterations if int(value) > 0})
    if save_iterations:
        return max(1, save_iterations[0])
    return max(1, int(args.iterations))


def output_dir_from_args(args: argparse.Namespace) -> Path:
    return Path(args.model_path)
