---
name: project_dense_evidence_fit_capacity_calibration
description: "worklog 68 - determined worklog 67's 20/21 extrapolative patches are caused by overfitting (grid resolution increase worsens geometry), not capacity/metric-density/weighting; raising NURBS resolution is not a safe fix"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T05:02:07.447Z
---

worklog 68: diagnosed WHY [[project_region_owned_full_evidence_patch_support]]'s (worklog 67) region-owned full evidence reclassified 20/21 patches `extrapolative`. Tested 3 hypotheses (fitting capacity insufficiency, dense-NN normalization metric's density dependence, non-uniform evidence weighting) via a grid-resolution sweep (6x6/8x8/10x10 NURBS control grid, same boundary+evidence from worklog 67, unmodified) with deterministic spatial holdout (checkerboard split via existing `pca_parameterize_points`), uniform vs density-compensated point_weights, and 4 different error normalizations (dense-NN spacing, representative-only spacing, robust normal-noise MAD, patch diameter).

Result: 20/21 patches (95%) classified `overfitting` -- raising grid resolution increases LOCAL orientation folding (adjacent-sample normal sign disagreement, new isolated metric in `osn_gs/surface/torch_local_orientation_folding.py`, distinct from the existing GLOBAL single-reference `compute_orientation_consistency`) and/or improves only training error while held-out error doesn't keep pace. Only 1 patch was genuinely `capacity_insufficient` (train AND held-out error both dropped meaningfully with geometry staying safe) and 1 was `inconclusive`. Zero patches showed `metric_density_dependent` or `weighting_problem` -- both of those hypotheses were NOT supported.

**Why:** answers whether bumping the existing 6x6/degree-2 NURBS fitting resolution would be a safe response to worklog 67's extrapolative finding -- it would NOT be; it trades "extrapolative due to coarse fit" for "geometrically unsafe due to overfitting/local folding."

**How to apply:** don't raise NURBS grid resolution as a fix for extrapolative patches without a different mechanism (e.g. adaptive local regularization) -- this was tested directly and found to make geometry worse, not better, on 95% of real patches.
