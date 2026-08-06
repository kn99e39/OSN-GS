---
name: project_baseline_compatible_3k_production_validation
description: worklog 65 - real 3k replay confirms baseline_compatible init recovers anisotropy/min-scale-collapse/over-segmentation but NOT screen-size prune storm at opacity reset
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-06T05:29:36.028Z
---

worklog 65: validated [[project_gaussian_initialization_parity]]'s `gaussian_initialization_mode=baseline_compatible` (production default) with a real 3k training replay (iteration 0/600/2900/3000/3100), 3-way vs covariance_knn and Graphdeco baseline.

Confirmed recovered: anisotropy median 35.77(covariance_knn)→3.66(baseline_compatible), baseline=5.49 — same order of magnitude now. min-scale collapse 1.49%→0.027% (below baseline's 0.448%). Chart materialization count drop (90→11 @2900) verified as genuine over-segmentation mitigation, NOT lost coverage: ran the SAME reliability pipeline directly on baseline's own PLY as a reference (3-8 regions) — baseline_compatible's region_count (7-19) is the same order of magnitude as that reference, while covariance_knn's (145-184) is 20-60x it.

NOT recovered / new open finding: iteration-3100 screen-size prune count is essentially unchanged (224,164 covariance_knn vs 233,178 baseline_compatible) despite the anisotropy fix — suggests the screen-size prune storm right after opacity-reset is a SEPARATE mechanism from anisotropy, not explained by worklog 64's fix. Render quality (PSNR/SSIM/LPIPS) improves consistently toward baseline at every checkpoint but does not fully close the gap (~1.1-1.4 PSNR remains @2900).

**Why:** this is the natural production-scale follow-up to worklog 64's step-600 harness result: confirms the fix generalizes to real 3k training, and cleanly separates "anisotropy is fixed" from "screen-prune storm is NOT fixed" — two previously-conflated phenomena.

**How to apply:** if a future round investigates the opacity-reset screen-prune storm, this worklog is the starting point — it's now isolated as a standalone open problem, unrelated to Gaussian init mode. No code was changed this round (pure replay/analysis); no pytest re-run needed.
