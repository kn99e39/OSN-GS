---
name: project-full-cloud-continuation-boundary-recovery
description: "Worklog 130 (now docs/worklogs/25_full_cloud_continuation_boundary_recovery.md) - decomposed worklog 129's boundary_recovery_failed into A/B/C stages via full-cloud continuation shells; real DATASET bottleneck is now precisely B_candidate_linking_failed (only 1-2 genuine termination candidates found)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2efdbcb1-f6ae-4abc-8cd4-dfbd07ed7f20
  modified: 2026-07-31T06:35:38.919Z
---

Worklog 130 (now `docs/worklogs/25_full_cloud_continuation_boundary_recovery.md` after the 2026-07-31 renumbering, 2026-07-30) followed [[project_density_preserving_representative_evidence]] (worklog 129, now `docs/worklogs/24_...md`). Worklog 129 fixed the reliability-admission collapse but left `construction_state=boundary_recovery_failed` unexplained on the real DATASET (region_count>0 but boundary_component_count=0 always). Worklog 130 added stage-by-stage boundary-pipeline diagnostics and a "full-cloud continuation shell" so the support-termination stage can see real density instead of just sparse representative-to-representative spacing.

**New module**: `osn_gs/surface/torch_full_cloud_continuation_shell.py` — per representative-node, gathers full-cloud Gaussians assigned (via worklog 129's existing Voronoi partition, reused not recomputed) to same-region representatives within an adaptive radius, classifies them same_mode/parallel_conflict/crease/ambiguous, and runs a continuous 180-bin circular gap query (replacing the old 8-sector histogram) to classify each node's own gap direction into `observed_support_termination` / `reliability_frontier` / `unresolved_sampling_gap` / `crease_discontinuity` / `parallel_sheet_conflict` / `ambiguous_continuation` / `no_gap`. Only `observed_support_termination` ever reaches directed ordering (existing filter in `torch_directed_boundary_ordering.py`, unchanged) — frontier/sampling-gap states stay diagnostic-only, never force a closed loop.

**Real bug found via testing** (density sweep on synthetic box, not the real DATASET): using each member's own (near-zero at high density) radial distance as the angular-footprint reference caused a handful of very-close points to subtend near-180° each, blanketing the whole circle and making genuine candidates disappear AS DENSITY INCREASED (paradoxical). Fixed by referencing the QUERY's own fixed search radius instead of the noisy per-member distance.

**Second bug found on the real DATASET (CUDA)**: two `torch.tensor(...)` calls in the new module lacked an explicit `device=`, defaulting to CPU while everything else was on CUDA — `RuntimeError: cuda:0 and cpu`. Fixed by passing `device=representative_positions.device` explicitly. Always test new torch code against the real CUDA DATASET run, not just CPU synthetic fixtures — this class of bug won't surface on CPU-only test suites.

**Real DATASET result** (same worklog 126(now `20_...md`)/129(now `24_...md`) repro config, cap=2048): `construction_state` stays `boundary_recovery_failed`/`boundary_component_count=0`, but the NEW `boundary_failure_stage` diagnostic pinpoints it as `B_candidate_linking_failed` — only 1-2 genuine termination candidates exist among 22-27 reliable representatives (most reliable nodes read as `no_gap`, i.e. fully interior). Runtime at worklog-130 time was ~40s/event since everything is reused, not recomputed — **but see worklog 131/132 below, this number is now stale.**

**Worklog 131** (now `docs/worklogs/26_canonical_reconstruction_gpu_synchronization_optimization.md`, perf-only, boundary logic untouched): found the real cost driver wasn't boundary/continuation logic at all — `evaluate_intrinsic_reliability` was doing 138,766 individual CUDA scalar host syncs. Replaced with a GPU boolean mask. Detached reconstruction 38.1s → 5.675s (6.7x). `construction_state` unchanged (`boundary_recovery_failed`), purely a speed fix.

**Worklog 132** (now `docs/worklogs/27_mode_aware_selection_phase2_exact_optimization.md`, perf-only): optimized the remaining mode-aware FPS/medoid representative-selection bottleneck — same class of bug, 138,766 individual CUDA scalar reads replaced with one bulk transfer. Selection median 4.278s → 2.324s, verified exact match against the v3 replay artifact.

**Worklog 133** (now `docs/worklogs/28_native_exact_cell_splitter_gate.md`): tried a native C++ CPU splitter to further cut cost — **rejected**. Couldn't reproduce np.dot's floating-point summation order, so discrete mode assignment diverged from the existing Python/NumPy backend. Backend stays Python/NumPy.

**⚠️ Runtime note for future reference**: worklog 130's ~40s/event figure is PRE-131/132 and no longer representative — current pipeline is ~5.675s(131)→faster still(132). Any future re-measurement (e.g. the next-step recommendation below) must benchmark against current HEAD, not cite the 40s number.

**Next step recommendation** (not yet done): re-measure candidate sparsity on a snapshot with more training iterations/images, since this scene is only 6 iterations / 1 image deep, using CURRENT (post-131/132) pipeline timing. Density-sweep synthetic evidence (region_count stays stable as density increases) suggests candidate sparsity may ease with more training, but this is an untested hypothesis, explicitly out of scope for worklog 130. Do not jump straight to topology-aware chart materialization or core-to-shell expansion.

Full detail: `docs/worklogs/25_full_cloud_continuation_boundary_recovery.md` through `28_native_exact_cell_splitter_gate.md` (originally numbered `130`-`133`, renumbered 2026-07-31 same day — a repo-wide worklogs renumbering/pruning pass renamed the whole surviving kept set, see [[project_boundary_first_isolated_topology_rebuild]] for the fuller mapping note).

As of 2026-07-31, HEAD=d359c5e and this whole 129-133 thread (files now 24-28) is fully committed. The working tree separately has 17 unrelated uncommitted files from a concurrent session (see [[project_uncertain_confidence_rename]], [[project_renderer_depth_sort_perf_regression]]) — unrelated to this thread, do not touch as part of this work.
