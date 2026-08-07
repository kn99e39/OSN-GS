---
name: project_visible_patch_fidelity_validation
description: worklog 66 - per-patch geometric fidelity validation of materialized visible NURBS patches; confirms over-segmentation fix (worklog 65) comes with real quality gain
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T03:52:41.442Z
---

worklog 66: validated whether materialized visible NURBS patches (from [[project_baseline_compatible_3k_production_validation]]'s normalized checkpoints) actually reconstruct observed surface accurately, not just "fewer patches". New per-patch metrics (`scripts/devtools/visible_patch_fidelity_validation.py`): point-to-surface / surface-to-evidence distance (normalized by local evidence scale), coverage, overlap, Jacobian/orientation/self-intersection, boundary provenance. 5-way classification (valid_supported/under_supported/extrapolative/unsafe_geometry/duplicate_or_overlapping) with thresholds borrowed from existing RegionFormationConfig conventions (core_region_typical_min_size=4, local_backbone_max_normalized_distance=4.0), never tuned to results.

Result: valid_supported ratio baseline_compatible 0%→36% (2900→3100), covariance_knn 27%→33%, real-baseline-PLY reference 50%. covariance_knn has far more patches (84-90 vs 5-11) but WORSE gap fraction (55-59% vs 41-43%) — confirms the worklog 65 patch-count reduction is genuine over-segmentation mitigation, not lost coverage. Most patches across ALL conditions are `under_supported` (3-4 evidence points per tiny region) — a separate, common root limitation (region size itself, likely same family as the §2.2 "candidate evidence density" bottleneck), not specific to baseline_compatible.

Found+fixed an implementation defect in the NEW analysis script itself (not production): tiny 3-member regions pass identical Gaussians as both boundary_points and interior_points, so concatenating them gave exact-duplicate evidence rows, collapsing the local-scale nearest-neighbor estimate to ~0 and exploding normalized distances into 5-6 digit noise. Fixed via `torch.unique` dedup before any distance computation.

Separately, profiling this validation's own construction pass (in response to a "high VRAM, low GPU util, slow" complaint) found and fixed a REAL production perf bug: `_boundary_evidence_swap_in` in `osn_gs/surface/torch_density_preserving_representative_selection.py` spent 316 of 358s doing O(pool^2) individual GPU syncs (`float(cuda_tensor[a,b])` inside a Python double loop building an adjacency graph). Vectorized into one batched boolean comparison; verified bit-identical output via git-stash A/B (region_count/patch_count/classification/area to the last decimal, region IDs) before committing — 338.22s→31.31s, 10.8x. Focused pytest (86 tests across selection/reliability/region-formation/covariance-frame/parametric-chart) passed; full suite NOT re-run (per task instruction — only focused tests when production code changes but the round's own instructions say skip full pytest).

Artifact with representative success/failure patch visualizations: https://claude.ai/code/artifact/5831b65d-a3c8-4a87-8d93-6858a600c2cd

**Why:** answers the natural follow-up question after worklog 65's over-segmentation fix — "is the surviving output actually good, or just less bad."

**How to apply:** the under_supported bottleneck (most patches have only 3-4 evidence points) is now an isolated, named open problem separate from Gaussian init mode or the screen-prune storm — likely the next thing worth tracing if visible-surface quality work resumes. Full pytest should be re-run before any further production-code changes since worklog 66 only ran focused tests.
