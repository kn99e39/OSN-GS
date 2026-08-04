---
name: representative-selection-boundary-evidence-recovery
description: "Worklog 49 — representative selection was dropping real orthogonal boundary evidence to budget competition; fixed with a safe deterministic swap-in, but it wasn't the closed-loop bottleneck"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-03T11:04:11.563Z
---

Worklog 49 (docs/worklogs/49_representative_selection_boundary_evidence_recovery.md) audited whether full-cloud physical termination evidence is lost at representative-selection or region-assignment/candidate-admission stages, per user directive to fix in the same round if selection loss is confirmed.

**Finding:** `select_density_preserving_representatives`'s own `_split_cell_into_modes` already separates a voxel cell's members into locally-consistent normal/offset modes — a multi-mode cell IS evidence of a real orientation split. But global weighted-FPS budget selection can still drop one mode to competition. Filtering same-cell sibling drops to alignment≤0.3 (well below the 0.6 mode-split gate) + source_count≥3 found real, well-supported (up to 1456/3006 Gaussians), near-orthogonal (alignment down to 0.0016) evidence being dropped: 26/1452 candidates on 3k, 100/2128 on 10k.

**Fix:** `osn_gs/surface/torch_density_preserving_representative_selection.py` — new `_boundary_evidence_swap_in()`, cap unchanged, one eviction per swap-in. Took **5 iterations** to make safe (each earlier version regressed box's `region_count` 6→7/8/9 by disconnecting a face's own kNN graph): (1) eviction must target the SIBLING's own orientation neighborhood, never the swap-in's own position; (2) eviction candidate must not be an articulation point of an explicit connected-component check on the sibling's orientation pool (radius=4x pool's own median spacing, degree≥4 required) — pure nearest-neighbor redundancy was not a safe proxy; (3) pool must have ≥15 members before any swap is attempted; (4) accepted swap-ins must be mutually ≥3x the original selection's median spacing apart (root cause of the last remaining box regression: many voxel cells along ONE edge each independently proposing a swap, cumulatively starving that face's kNN graph regardless of which representative got evicted).

**Result:** negative-control fixtures (Box/Cylinder/Sphere/Thin-slab at cap 64, and Box density-sweep mult 1/2/4/8 × cap 128/256) — region_count always preserved, swap_in=0 at cap 64 (matches [[project_candidate_local_smooth_continuation_repair]]'s worklog 48 baseline exactly), swap_in 9-57 at larger box configs with region_count still 6. Real 3k/5k/10k: swap-in fired 22/51/76 times (genuine evidence, not noise), physical candidate counts barely moved (±1-5, no consistent direction), **closed/materialized unchanged on all three (0/0, 2/2, 0/0)**. Full pytest 720 passed / 1 skipped, same as worklog 47/48.

**Conclusion:** representative-selection evidence loss was real and is now fixed, but it is NOT the dominant cause of the real-checkpoint closed-loop absence — confirms [[project_full_cloud_continuation_boundary_recovery]]/worklog 47/48's standing conclusion that the bottleneck is real observed-termination evidence *density* on large region perimeters, not a pipeline transfer defect. Region-assignment/candidate-admission stages were not separately found to have a defect this round.

**How to apply:** if a future round touches `_boundary_evidence_swap_in` or its safety gates, re-run the box density-sweep A/B (`scripts/devtools/trace_representative_selection_boundary_loss.py` + manual region-formation replay) before trusting any change — the safety margin here was empirically hard-won across 5 failed attempts, not a first-try design. Do not loosen the connectivity/dedup gates to increase swap_in count without re-verifying box region_count stays at 6.
