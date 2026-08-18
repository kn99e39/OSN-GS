#!/bin/bash
# exp/2dgs-nurbs-surface-evidence -- the branch's real training run.
#
# Two arms, identical in everything the 2DGS methodology does not force to
# differ (dataset, calibrated cameras, train/test split, scene normalization,
# resolution, SH degree, background, LR schedule, densification window,
# opacity-reset schedule, evaluation views, output cadence):
#
#   vanilla  : volumetric 3D Gaussian, `baseline_compatible` init, 3DGS
#              affine-covariance rasterizer, photometric loss only.
#   2dgs     : 2D surfel, official `create_from_pcd` init, official
#              perspective-correct diff-surfel rasterizer, photometric loss
#              plus depth distortion and normal consistency.
#
# The differences the 2DGS method requires and that must stay intact
# (arXiv:2403.17888v3 / hbb1/2d-gaussian-splatting @ 335ad61):
#   * planar primitive             --primitive surfel_2d
#   * perspective-correct raster   (implied by --primitive)
#   * depth distortion             --lambda_dist 100   (paper alpha, unbounded)
#   * normal consistency           --lambda_normal 0.05
#   * 2DGS density control         --adc_prune_opacity_threshold 0.05
#                                  (paper sec. 6.1 "remove splats with opacity
#                                   lower than 0.05"; 3DGS/OSN-GS use 0.005)
#
# Budget: the OFFICIAL 30,000-iteration schedule, with the official activation
# milestones left exactly where upstream puts them (depth distortion after
# 3000, normal consistency after 7000) and the official densification window
# (500..15000, interval 100, opacity reset every 3000). Nothing is rescaled.
#
# OSN_GS_ADAPTATION -- primitive cap. `--adc_max_gaussians ${MAX_PRIMITIVES}`
# is applied IDENTICALLY to both arms. Neither the official 3DGS nor the
# official 2DGS implementation caps primitive count; both were run on 24 GB
# GPUs (the 2DGS paper reports an RTX 3090). This host has a 12 GB RTX 3080 Ti
# with ~10.8 GB free, and a measured ~2.1 GB per million primitives at this
# resolution, so an uncapped run reaches OOM around iteration 9-10k -- before
# normal consistency (active from 7000) has had time to act. The cap therefore
# exists to let the OFFICIAL SCHEDULE run to completion, not to shape either
# arm's geometry. It has two disclosed consequences: (1) once it binds, both
# arms stop densifying at the same budget, which makes the comparison an
# equal-primitive-count one; (2) OSN-GS's `_limited_indices` truncates
# densification candidates by row index rather than by gradient rank, so the
# subset admitted on a capped step is deterministic but not the highest-
# gradient one. Both effects apply equally to both arms.
#
# Usage (inside the osn-gs-2dgs:cuda11.8 container):
#     bash scripts/experiments/run_2dgs_vs_vanilla_30k.sh <arm> <output_dir>
# with <arm> in {vanilla, 2dgs}.
set -euo pipefail

ARM="${1:?usage: run_2dgs_vs_vanilla_30k.sh <vanilla|2dgs> <output_dir>}"
OUT="${2:?usage: run_2dgs_vs_vanilla_30k.sh <vanilla|2dgs> <output_dir>}"
SOURCE="${SOURCE_PATH:-/data}"
ITERATIONS="${ITERATIONS:-30000}"
MAX_PRIMITIVES="${MAX_PRIMITIVES:-3000000}"

COMMON=(
  -s "${SOURCE}" --sparse_dir sparse/0 --eval --llffhold 8
  -m "${OUT}"
  --iterations "${ITERATIONS}"
  --save_iterations 7000 15000 "${ITERATIONS}"
  --densify_from_iter 500 --densify_until_iter 15000 --densification_interval 100
  --densify_grad_threshold 0.0002 --adc_percent_dense 0.01
  --adc_max_screen_size 20.0 --adc_max_scale_ratio 0.1 --adc_split_samples 2
  --opacity_reset_interval 3000 --screen_size_prune_from_iter 3000
  --adc_max_gaussians "${MAX_PRIMITIVES}"
  --position_lr_extent_mode scene --surface_update_interval 0
  --skip_cuda_build_preflight
  --progress_log_interval 500 --timing_log_interval 500
)

case "${ARM}" in
  vanilla)
    exec python3.10 train.py "${COMMON[@]}" \
      --primitive gaussian_3d \
      --gaussian_initialization_mode baseline_compatible \
      --adc_prune_opacity_threshold 0.005
    ;;
  2dgs)
    exec python3.10 train.py "${COMMON[@]}" \
      --primitive surfel_2d \
      --lambda_dist 100 --lambda_normal 0.05 --depth_ratio 0 \
      --dist_from_iter 3000 --normal_from_iter 7000 \
      --adc_prune_opacity_threshold 0.05
    ;;
  *)
    echo "unknown arm: ${ARM}" >&2
    exit 2
    ;;
esac
