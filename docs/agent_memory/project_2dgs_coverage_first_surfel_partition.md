---
name: project-2dgs-coverage-first-surfel-partition
description: OSN-GS pivoted to 2DGS surfels as canonical surface evidence; new branch arch/2dgs-coverage-first-surface; checkpoint not yet located
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-20T04:15:47.139Z
---

Worklog 96 (2026-08-19, on the new branch, not `voxel-surface-regions`): the user adopted the already-implemented **2DGS surfel** branch (`exp/2dgs-nurbs-surface-evidence`) as the canonical surface-evidence direction, superseding the plan to keep repairing volumetric-3DGS covariance normals (the Worklog 105/106 line on `voxel-surface-regions`, and the "Worklog 106 `partition_normal_reliability`" idea that was explicitly forbidden going forward).

New direction: trained 2DGS surfel scene -> intrinsic surfel tangent plane/normal (`t_w = t_u x t_v`, read directly off the trained rotation quaternion, NO eigen-decomposition) -> Coverage-first Surfel Subset partition -> (later) subset-local trust -> latent surface -> 1 subset : 1 NURBS patch.

**Branch**: `arch/2dgs-coverage-first-surface`, based on `origin/exp/2dgs-nurbs-surface-evidence` @ `54b72c2`. This is an explicit, user-directed exception to [[project_branch_rule]] ("never create other branches unless the user explicitly says so") — work continues here, NOT on `voxel-surface-regions`, and this branch is never merged back. `exp/2dgs-nurbs-surface-evidence` itself stays untouched as historical evidence.

Implementation done: `osn_gs/surface/torch_surfel_surface_orientation.py` (reads `model.get_tangent_u/get_tangent_v/get_normal` directly; AST-tested to contain no eigh/covariance token); `osn_gs/surface/torch_coverage_first_subset_partition.py` ported byte-for-byte from `voxel-surface-regions` (only the orientation type hint loosened to a structural `Protocol`); `scripts/devtools/coverage_first_surfel_partition_export.py` (fails closed on a non-`surfel_2d` checkpoint, verified). 38 new/ported focused tests pass.

**Blocker, unresolved**: the actual trained 2DGS checkpoint (Worklog 95's 30k-iteration run, RTX 3080 Ti / CUDA 11.8 Docker host, held-out PSNR 28.24) is not present on this local machine (RTX 5080 / CUDA 13) — searched `output/` and the whole filesystem, nothing found. No real partition measurement exists yet. Two paths forward, neither started: (a) retrain here (the vendored `diff_surfel_rasterization` CUDA extension's buildability on this CUDA/GPU combo is confirmed UNbuilt — the full regression's 21 `test_surfel_rasterization_cuda.py`/`test_surfel_regularization_cuda.py` tests all skip with "CUDA and the vendored diff_surfel_rasterization extension are required"), or (b) recover the original checkpoint from wherever it was trained. This is a decision for the user, not something to resolve unilaterally by launching a long training run.

See [[project_branch_rule]] for the branch policy this overrides, and [[project_coverage_first_subset_partition]]/[[project_coverage_first_partition_measurements]]/[[feedback_correct_replay_checkpoint]] for the superseded volumetric-3DGS line this replaces as the primary direction (those worklogs and their numbers stay valid history on `voxel-surface-regions`, just not the path forward).
