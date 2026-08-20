---
name: project-2dgs-coverage-first-surfel-partition
description: "OSN-GS pivoted to 2DGS surfels as canonical surface evidence; branch arch/2dgs-coverage-first-surface; retrained on RTX 5080, real partition measured, mixed result, no architecture decision yet"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-20T04:43:50.034Z
---

Worklog 96 (2026-08-19/20, on the new branch, not `voxel-surface-regions`): the user adopted the already-implemented **2DGS surfel** branch (`exp/2dgs-nurbs-surface-evidence`) as the canonical surface-evidence direction, superseding the plan to keep repairing volumetric-3DGS covariance normals (the Worklog 105/106 line on `voxel-surface-regions`).

New direction: trained 2DGS surfel scene -> intrinsic surfel tangent plane/normal (`t_w = t_u x t_v`, read directly off the trained rotation quaternion, NO eigen-decomposition) -> Coverage-first Surfel Subset partition -> (later) subset-local trust -> latent surface -> 1 subset : 1 NURBS patch.

**Branch**: `arch/2dgs-coverage-first-surface`, based on `origin/exp/2dgs-nurbs-surface-evidence` @ `54b72c2`. Explicit, user-directed exception to [[project_branch_rule]] — work continues here, NOT on `voxel-surface-regions`, never merged back. `exp/2dgs-nurbs-surface-evidence` itself stays untouched as historical evidence.

Implementation: `osn_gs/surface/torch_surfel_surface_orientation.py` (reads `model.get_tangent_u/get_tangent_v/get_normal` directly; AST-tested, no eigh/covariance token); `osn_gs/surface/torch_coverage_first_subset_partition.py` ported byte-for-byte from `voxel-surface-regions` (orientation type hint loosened to a structural `Protocol`); `scripts/devtools/coverage_first_surfel_partition_export.py` (fails closed on non-`surfel_2d` checkpoint). 38 focused tests pass.

**Checkpoint**: Worklog 95's original 30k checkpoint (RTX 3080 Ti/CUDA 11.8 Docker host) was not present on this local machine (RTX 5080/CUDA 13). Asked the user; they chose to retrain here. Built the vendored `diff_surfel_rasterization` CUDA extension for this machine via new `scripts/build_surfel_extension.bat` (VS2022 + `TORCH_CUDA_ARCH_LIST=12.0`, same pattern as `scripts/build_baseline_extensions.bat`) — all 21 previously-skipped CUDA surfel tests now pass. Retrained with Worklog 95's exact 2dgs-arm config (`scripts/experiments/run_2dgs_vs_vanilla_30k.sh`'s parameters, unchanged): 30k iterations, ~9 min on RTX 5080 (vs. 19:34 on the 3080 Ti). Result closely reproduced the original: 1,197,331 final surfels (orig 1,193,268), held-out PSNR 28.256/SSIM 0.8997 (orig 28.24/0.899). Checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt`.

**Real partition measurement done** (`output/osn_gs_2dgs_coverage_first_subset_partition/`, report `surfel_partition_report.json`). Compared against the CORRECTED 3DGS baseline (Worklog 106, `output/osn_gs_scene/3000` — not the flawed Worklog 105 checkpoint):

- Coverage: 1,197,331 surfels all assigned, 0 unassigned/multiply-owned, 0 disconnected subsets.
- 58,646 subsets, max subset 894,378 = **74.70%** (3DGS: 82.94% — improved).
- Normal-incompatible cut edges 22.12% of spatial edges (3DGS: 25.13% — improved).
- Singleton/fallback ownership 3.38% (3DGS: 2.09% — WORSE, not better).
- Local unsigned normal agreement (spatial-neighborhood) distribution nearly identical to 3DGS (mean 0.877 vs 0.869, p05 0.356 vs 0.389 — 2DGS's worst 5% is actually slightly worse).
- Visual check: the giant subset still chains across the flat patio ground AND the higher-curvature hedge background (two visually distinct orientation regions) — same failure mode as the volumetric partition, just at a slightly lower percentage.

**No architecture decision made.** The result is mixed, not a clean win. Next step is the user visually reviewing the 4 review-export views before any decision on subset-local trust estimation or further partition work. See [[project_coverage_first_subset_partition]]/[[project_coverage_first_partition_measurements]]/[[feedback_correct_replay_checkpoint]] for the superseded volumetric-3DGS line this replaces as the primary direction.
