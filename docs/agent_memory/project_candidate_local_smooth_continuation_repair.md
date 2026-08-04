---
name: candidate-local-smooth-continuation-repair
description: "Worklog 48 — no_gap same-mode support now requires a candidate-local bounded accepted-topology path, not just radius+pointwise alignment"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-03T09:36:08.191Z
---

Worklog 48 (docs/worklogs/48_candidate_local_smooth_continuation_repair.md) audited whether the same-mode support that makes `build_continuation_shells()` return `no_gap` is actually candidate-locally connected to the query representative via the region's own `internal_accepted_edge_ids` graph, or merely within the same spatial radius with matching pointwise geometry (normal alignment / tangent residual / footprint ratio).

**Finding:** on real 3k/5k/10k checkpoints (cap 2048), 0 no_gap nodes had same-mode support that was *entirely* nonlocal, but 8/89 (3k), 3/53 (5k), 1/62 (10k) had a partial fold/gap-crossing signature: same-mode member reachable only via an accepted-edge path (hop ≥ 2) several times longer (≥3x) than its straight-line distance. 5k's two genuinely-closed regions (133/143) had no no_gap nodes at all — they succeeded because they're small enough that gaps stay observable, not because they lacked this contamination.

**Fix:** `osn_gs/surface/torch_full_cloud_continuation_shell.py` — same-mode members are split into `same_mode_local` (closes the gap) vs a new `nonlocal_same_mode` category (counts as general occupancy but never closes a same-mode gap; if it borders the remaining gap, state fails closed to `STATE_AMBIGUOUS`, never force-promoted to `observed_support_termination`). Locality = unbounded BFS within the region's own accepted-edge connected component (a fixed hop cap wrongly flagged long-but-genuinely-continuous thin regions), gated further by path-length-vs-straight-line ratio ≥3x at hop≥2. `region_internal_accepted_edges` now threads through `build_continuation_shells`/`_from_input`; production call site (`torch_visible_surface_construction.py`) needed no changes since `region_result.regions` was already available there.

**Result:** 3k/5k each reclassified exactly 1 no_gap node → `parallel_sheet_conflict`; 10k unchanged (the one flagged node's remaining local support still covered the full circle, so it correctly stayed `no_gap` — nothing was forced). physical candidate and closed/materialized counts unchanged on all three checkpoints (153/181/121, closed 0/2/0 — same as worklog 47). Negative-control fixtures (Box/Cylinder/Sphere/Thin-slab) verified via `scripts/devtools/compare_fold_signature_toggle.py` A/B (gate on vs. effectively disabled): **zero difference** — this fix doesn't touch synthetic fixtures at all, only real-checkpoint fold patterns. Full pytest 720 passed / 1 skipped, same as worklog 47 baseline (no regression).

**Why it doesn't move the real-checkpoint bottleneck:** the contamination found was real but small-share (2-13% of no_gap nodes, never total). The dominant remaining blocker, unchanged, is [[project_full_cloud_continuation_boundary_recovery]]-adjacent: large-region-perimeter true smooth `no_gap` genuinely dominates because independent observed-termination evidence density itself is too sparse to chain — a real-data floor, not an algorithm defect. Do not force-close it (same principle as [[project_stage1_voxel_patch]]).

**How to apply:** if a future round revisits `no_gap`/continuation-shell logic, this locality gate is now load-bearing — don't bypass it by reverting to pure pointwise same-mode matching. New devtools scripts `scripts/devtools/trace_no_gap_local_connectivity.py` (per-node local/nonlocal/mixed classification) and `scripts/devtools/compare_fold_signature_toggle.py` (A/B toggle harness) are reusable for the next audit round.
