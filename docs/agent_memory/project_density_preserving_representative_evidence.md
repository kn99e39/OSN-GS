---
name: project-density-preserving-representative-evidence
description: "Worklog 129 (now docs/worklogs/24_density_preserving_canonical_representative_evidence_and_reliability_repair.md) - fixed worklog 126 (now docs/worklogs/20_adc_synchronized_canonical_visible_nurbs_experiment.md)'s reliable_count=0-on-all-caps by replacing representative-only kNN contextual reliability with full-observed-cloud evidence + mode-aware representative selection"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2efdbcb1-f6ae-4abc-8cd4-dfbd07ed7f20
  modified: 2026-07-31T06:35:00.771Z
---

Worklog 129 (now `docs/worklogs/24_density_preserving_canonical_representative_evidence_and_reliability_repair.md` after the 2026-07-31 renumbering, 2026-07-30) fixed the root cause behind [[project_boundary_conditioned_occlusion]]-adjacent worklog 126 (now `docs/worklogs/20_adc_synchronized_canonical_visible_nurbs_experiment.md`)'s finding that `reconstruct_visible_after_adc` on the real DATASET (~139k Gaussians) produced `reliable_count=0`/`region_count=0`/`no_admissible_region` at every representative cap tried (512/1024/2048).

**Root cause**: the representative sampler (`TorchOSNGSPipeline._canonical_construction_indices`) kept one Gaussian per occupied voxel cell, then contextual reliability recomputed 8-nearest-neighbor evidence *among that same sparse representative set* — representative spacing bore no relation to real local density, so almost every representative read as isolated/disagreeing.

**Fix** (new modules, all under `osn_gs/surface/`): `torch_full_neighborhood_evidence.py` (per-representative aggregate evidence over its full-cloud Voronoi cell — support count, opacity mass, normal consensus, tangent residual, etc., computed via chunked nearest-representative assignment, never O(N²) on the full cloud) + `torch_density_preserving_representative_selection.py` (mode-aware candidate generation per voxel cell — splits a cell into multiple representatives when it contains structurally distinct normal/offset clusters — plus weighted farthest-point selection up to the existing `canonical_construction_max_points` budget). `construct_visible_nurbs_from_gaussians` gained one optional `reliability` override param as the sole injection point; no new CLI selector.

**Scope**: only wired into `reconstruct_visible_after_adc` (the path worklog 126 actually broke on) — `_initialize_canonical`/`maintain_surface_from_certain` intentionally left untouched (no learned per-Gaussian covariance exists at raw-point-cloud init time, so full-neighborhood evidence has nothing to aggregate there).

**Real DATASET result after the fix** (same worklog 126 repro config, cap=2048): `reliable_count` 0→22-24, `region_count` 0→1-3, but state moved from `no_admissible_region` to `boundary_recovery_failed` — i.e. the bottleneck is now demonstrably boundary/topology recovery (`extract_support_termination_candidates`/`recover_directed_boundary_components` never finding a closed loop on this scene), not reliability admission collapse. That is the recommended next investigation thread, not general topology/atlas work.

**Real cost disclosed**: per-ADC-event runtime went from ~0.9s (worklog 126) to ~37-42s (full-cloud O(N) eigen-decomposition + chunked Voronoi assignment + mode-aware cell splitting over the real cloud). Relevant to [[project_deferred_followups]] (training speed, parked until NURBS representation is complete) — not fixed here, only disclosed.

**Non-goal**: did not touch region formation (`torch_gaussian_surface_region_formation.py`), boundary/materialization stages, or ADC scheduling/transaction semantics (worklog 126) — those modules just now receive better-evidenced representative input.

Full detail: `docs/worklogs/24_density_preserving_canonical_representative_evidence_and_reliability_repair.md` (originally `129_...md`, renumbered 2026-07-31).
