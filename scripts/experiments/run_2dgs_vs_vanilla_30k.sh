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
# RESOLUTION / PRIMITIVE BUDGET -- both set by what the VANILLA arm can
# complete on this GPU, then applied IDENTICALLY to both arms.
#
# Three attempts established the constraint empirically on this host
# (RTX 3080 Ti, 12 GB, ~10.8 GB free, ~2.05 GB per million primitives):
#
#   1600x1036, cap 3.0M : vanilla bound the cap at iteration 5000. The old
#                         cap enforcement spent the remaining budget on clones
#                         first, so `split_parents` was 0 on EVERY later step.
#                         Vanilla's train PSNR at 30k (21.01) fell below its
#                         own 2.9k reference (23.29). Discarded.
#   1297x840  (official 2DGS m360 `-i images_4`), cap 4.0M
#                       : vanilla reached 3.84M by iteration 6300 and died
#                         with CUDA OOM, 8,700 iterations short of the end of
#                         the densification window. 2DGS completed it (peak
#                         2.55M, final 2.13M). Discarded for vanilla.
#   648x420   (`images_8`), cap 4.0M
#                       : vanilla reached 2.99M by iteration 6900 and was
#                         still climbing ~18k/100 iterations, i.e. heading for
#                         the cap again. Lowering resolution does not stop it.
#
# Vanilla 3DGS on this scene simply does not fit an uncapped official 30k
# schedule in 12 GB at any resolution tried. The experiment therefore uses an
# EQUAL PRIMITIVE BUDGET: both arms train at 648x420 with the same
# `--adc_max_gaussians`, so the constraint is symmetric and the comparison is
# "same scene, same schedule, same primitive budget, different primitive".
#
# The cap's ENFORCEMENT was fixed first (osn_gs/gaussian/torch_density_control.py):
# a bound budget is now divided between clone and split in proportion to their
# demand, and the retained candidates are the highest-gradient ones rather
# than the lowest row indices. Without that fix a bound cap silently deletes
# splitting, which is not "less densification" but different geometry.
# `scripts/devtools/check_adc_cap.sh` still reports whether the cap bound.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for both arms: the vanilla
# OOM report showed 1.88 GiB reserved-but-unallocated, i.e. fragmentation.
#
set -euo pipefail

ARM="${1:?usage: run_2dgs_vs_vanilla_30k.sh <vanilla|2dgs> <output_dir>}"
OUT="${2:?usage: run_2dgs_vs_vanilla_30k.sh <vanilla|2dgs> <output_dir>}"
SOURCE="${SOURCE_PATH:-/data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ITERATIONS="${ITERATIONS:-30000}"
MAX_PRIMITIVES="${MAX_PRIMITIVES:-2000000}"

COMMON=(
  -s "${SOURCE}" --sparse_dir sparse/0 --images "${IMAGES:-images_8}" --eval --llffhold 8
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
