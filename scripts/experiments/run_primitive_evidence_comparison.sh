#!/bin/bash
# exp/2dgs-nurbs-surface-evidence -- two-arm structural evidence comparison.
#
# Both arms enter the SAME unmodified downstream constructor chain through the
# SAME adapter (`osn_gs.gaussian.torch_primitive_evidence_adapter`), which
# dispatches on the primitive each checkpoint itself records. Same cap, same
# device, same representation (RAW_CENTER_BASELINE, i.e. the evidence is used
# as-is with no representation swap), same thresholds -- the 2DGS branch gets
# no easier constructor.
#
# The surfel arm is measured twice on purpose:
#   exact_rank2         -- the true 2DGS geometry, smallest covariance
#                          eigenvalue exactly zero. PRIMARY.
#   epsilon_regularized -- a disclosed surrogate that manufactures a
#                          normal-direction thickness, only so that legacy
#                          OSN-GS metrics which divide by that thickness are
#                          not silently reporting a saturated ratio. SECONDARY;
#                          its numbers describe the surrogate, not 2DGS.
set -euo pipefail

CAP="${CAP:-2048}"
DEVICE="${DEVICE:-cuda}"
VANILLA="${VANILLA:-output/vanilla_30k/30000}"
SURFEL="${SURFEL:-output/2dgs_30k/30000}"
OUT_DIR="${OUT_DIR:-reports}"

mkdir -p "${OUT_DIR}"

python3.10 scripts/devtools/primitive_structural_evidence_comparison.py \
  --arm "vanilla_3dgs=${VANILLA}" \
  --arm "2dgs_surfel=${SURFEL}" \
  --cap "${CAP}" --device "${DEVICE}" \
  --surfel_covariance_mode exact_rank2 \
  --output "${OUT_DIR}/primitive_evidence_comparison_exact_rank2.json"

python3.10 scripts/devtools/primitive_structural_evidence_comparison.py \
  --arm "2dgs_surfel=${SURFEL}" \
  --cap "${CAP}" --device "${DEVICE}" \
  --surfel_covariance_mode epsilon_regularized \
  --output "${OUT_DIR}/primitive_evidence_comparison_epsilon_regularized.json"
