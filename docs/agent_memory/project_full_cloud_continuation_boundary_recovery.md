---
name: project-full-cloud-continuation-boundary-recovery
description: "Worklog 130 - decomposed worklog 129's boundary_recovery_failed into A/B/C stages via full-cloud continuation shells; real DATASET bottleneck is now precisely B_candidate_linking_failed (only 1-2 genuine termination candidates found)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2efdbcb1-f6ae-4abc-8cd4-dfbd07ed7f20
  modified: 2026-07-30T06:01:37.206Z
---

Worklog 130 (2026-07-30) followed [[project_density_preserving_representative_evidence]] (worklog 129). Worklog 129 fixed the reliability-admission collapse but left `construction_state=boundary_recovery_failed` unexplained on the real DATASET (region_count>0 but boundary_component_count=0 always). Worklog 130 added stage-by-stage boundary-pipeline diagnostics and a "full-cloud continuation shell" so the support-termination stage can see real density instead of just sparse representative-to-representative spacing.

**New module**: `osn_gs/surface/torch_full_cloud_continuation_shell.py` — per representative-node, gathers full-cloud Gaussians assigned (via worklog 129's existing Voronoi partition, reused not recomputed) to same-region representatives within an adaptive radius, classifies them same_mode/parallel_conflict/crease/ambiguous, and runs a continuous 180-bin circular gap query (replacing the old 8-sector histogram) to classify each node's own gap direction into `observed_support_termination` / `reliability_frontier` / `unresolved_sampling_gap` / `crease_discontinuity` / `parallel_sheet_conflict` / `ambiguous_continuation` / `no_gap`. Only `observed_support_termination` ever reaches directed ordering (existing filter in `torch_directed_boundary_ordering.py`, unchanged) — frontier/sampling-gap states stay diagnostic-only, never force a closed loop.

**Real bug found via testing** (density sweep on synthetic box, not the real DATASET): using each member's own (near-zero at high density) radial distance as the angular-footprint reference caused a handful of very-close points to subtend near-180° each, blanketing the whole circle and making genuine candidates disappear AS DENSITY INCREASED (paradoxical). Fixed by referencing the QUERY's own fixed search radius instead of the noisy per-member distance.

**Second bug found on the real DATASET (CUDA)**: two `torch.tensor(...)` calls in the new module lacked an explicit `device=`, defaulting to CPU while everything else was on CUDA — `RuntimeError: cuda:0 and cpu`. Fixed by passing `device=representative_positions.device` explicitly. Always test new torch code against the real CUDA DATASET run, not just CPU synthetic fixtures — this class of bug won't surface on CPU-only test suites.

**Real DATASET result** (same worklog 126/129 repro config, cap=2048): `construction_state` stays `boundary_recovery_failed`/`boundary_component_count=0`, but the NEW `boundary_failure_stage` diagnostic pinpoints it as `B_candidate_linking_failed` — only 1-2 genuine termination candidates exist among 22-27 reliable representatives (most reliable nodes read as `no_gap`, i.e. fully interior). Runtime unchanged from worklog 129 (~40s/event) since everything is reused, not recomputed.

**Next step recommendation** (not yet done): re-measure candidate sparsity on a snapshot with more training iterations/images, since this scene is only 6 iterations / 1 image deep. Density-sweep synthetic evidence (region_count stays stable as density increases) suggests candidate sparsity may ease with more training, but this is an untested hypothesis, explicitly out of scope for worklog 130.

Full detail: `docs/worklogs/130_full_cloud_continuation_boundary_recovery.md`.
