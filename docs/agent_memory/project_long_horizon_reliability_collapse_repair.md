---
name: project_long_horizon_reliability_collapse_repair
description: "Worklog 30 (docs/worklogs/30_long_horizon_reliability_collapse_and_nan_repair.md) — real long-horizon (1.6M-3M Gaussian) constructor collapse root-caused to two bugs (cuSOLVER batch-size ceiling, NOT NaN; unbounded Voronoi evidence aggregation), both narrowly fixed; reliable_count still low, unsolved"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T07:16:43.557Z
---

Follows [[project_full_cloud_continuation_boundary_recovery]] but supersedes its bottleneck framing for real long-horizon training: worklog 130's `B_candidate_linking_failed` was measured on a 6-iteration/1.6M-Gaussian-early snapshot and is NOT the actual bottleneck once training runs longer (3k/5k/10k iterations, 1.6M-3M Gaussians) — the constructor collapses much earlier, before boundary recovery is ever reached.

**Root cause 1 (10k iteration crash)**: the crash message says "input matrix contains NaN" but this was **false**. Directly reproduced with a pure synthetic finite identity-matrix batch: `torch.linalg.eigh` on CUDA (`cusolverDnXsyevBatched`) hard-crashes with `CUSOLVER_STATUS_INVALID_VALUE` once batch size exceeds ~2,064,888 (measured on RTX 5080/CUDA 13.0/PyTorch 2.12.1+cu130 — not a documented constant, GPU/driver-dependent). `extract_covariance_frame` (`osn_gs/surface/torch_gaussian_covariance_frame.py`) called eigh on the ENTIRE eligible cloud unchunked. Fixed by adding `_batched_eigh()` chunking at a conservative 1,000,000/batch — purely a batch split, mathematically identical result (eigh over independent 3x3 blocks has no cross-row interaction).

**Root cause 2 (3k/5k/10k `reliable_count≈0`)**: NOT intrinsic covariance quality — full-cloud AND representative-level intrinsic reliability were consistently 92.9-97.9% reliable across all three snapshots. The collapse happens entirely in CONTEXTUAL evidence: `assign_nearest_representative` (`osn_gs/surface/torch_full_neighborhood_evidence.py`) does an unbounded global nearest-representative Voronoi partition of the full cloud onto the 2048-cap representative set — as Gaussian count grows with a fixed cap, each representative's cell absorbs up to 23,881 spatially non-local members, inflating `tangent_residual_mean` to 13-16x the `consistent_max_mutual_tangent_residual=0.35` gate. Fixed by bounding evidence aggregation (not the returned `nearest`/`spacing` assignment itself, which other callers like the continuation shell still use unchanged) to a local radius = `6 x representative's own tangent_major_scale` (reusing worklog 130's continuation-shell convention, not inventing a new constant). Result: `tangent_residual_mean` median 4.6-5.7 -> 1.6-1.7, `reliable_count` 4/2/crash -> 7/7/9 out of 2048.

**Why:** User explicitly demanded this be one execution batch (diagnose + fix + verify, no splitting into follow-up rounds) and forbade guessing multiple policy changes at once, loosening reliability thresholds, or touching representative cap/selection/boundary-linking policy this round.

**How to apply:** `reliable_count` at long horizon is STILL only 7-9/2048 after both fixes — meaningfully better but not healthy. Cap sweep (2048/4096/8192) showed `tangent_residual_mean` barely responds to cap size post-fix, ruling out cap as the remaining driver. Leading unverified hypothesis for the REMAINING gap: `tangent_residual` normalizes by a single representative Gaussian's own tiny individual `tangent_major_scale` (real baseline-trained Gaussians are often much smaller than local point spacing — see [[project_benchmark_surface_aligned_covariance]]'s anisotropy stats), which may be too small a denominator for real data. NOT fixed — flagged as the next bottleneck, explicitly not touched to avoid speculative multi-policy changes in one pass.

New diagnostic-only field: `diagnostic_summary["reliability_failure_stage"]` (`intrinsic_reliability_collapse`/`contextual_reliability_collapse`/`partial_contextual_reliability_collapse`/`not_failed`) plus `intrinsic_reliable_count`/`intrinsic_ambiguous_count`/`intrinsic_rejected_count`, added purely additively to `torch_visible_surface_construction.py` — `construction_state`/public API untouched.

New scripts (offline, not production): `scripts/devtools/replay_long_horizon_snapshot.py`, `scripts/devtools/diagnose_long_horizon_reliability_collapse.py` — load a checkpoint's raw tensors directly and replay the exact `reconstruct_visible_after_adc` code path without retraining.

New tests: `tests/test_long_horizon_reliability_collapse_repair.py` (5 tests: chunked-vs-unchunked eigh equivalence, local-radius exclusion, no-op when all-local). Repo-wide pytest: 604 passed, 1 skipped, 0 failed at completion (the "2 unrelated topology failures" noted in [[project_boundary_conditioned_occlusion]] were already gone by this session, unrelated/not investigated).

Full detail: `docs/worklogs/30_long_horizon_reliability_collapse_and_nan_repair.md`.
