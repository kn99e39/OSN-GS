---
name: project_directed_cycle_solver_contract_hardening
description: "Worklog 36 (docs/worklogs/36_directed_cycle_solver_contract_hardening_and_region_coverage_recovery.md) — root cause of worklog 34/35 baseline discrepancy found and fixed (worklog 35's ablation captured pre-worklog-34 HEAD via git show, not worklog 34's real baseline); box_face candidate 100% accounting bug fixed (isolated nodes were silently dropped); lexicographic max-coverage objective tried and REJECTED (regressed cylinder); branch-ambiguity gating tried and REJECTED (false-positived on regular grids); real snapshot closed-loop absence precisely split into 90% region-coverage-too-small vs 10% candidate-recall/compatibility, zero pure ordering-solver-bug cases found on real data"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T14:09:12.797Z
---

Follows [[project_core_region_consolidation_and_boundary_cycle_recovery]] (worklog 35). This round was explicitly a hardening/audit pass on worklog 35's own C11/C9 work, not new feature scope.

**Baseline discrepancy root cause (solved)**: worklog 35's 4-way ablation used `git show HEAD:<file>` to capture "baseline" state — but this repo has never committed worklogs 1-35 (everything lives in the uncommitted working tree). HEAD (d359c5e) predates BOTH worklog 34's growth fix and worklog 35's own C9 fix, so worklog 35's "A_baseline" (region_count=70, core_member=362, consensus_attached=1) was actually a **pre-worklog-34** measurement, not worklog 34's own reported baseline (75/392) — which cannot be exactly reconstructed since it was never committed at that intermediate point. Fixed going forward: added two diagnostic-only config flags to `RegionFormationConfig` (`enable_worklog34_growth_weak_bridge_exemption`, `enable_worklog35_parallel_veto_nearby_evidence_gate`, both default True = current production), enabling in-process 4-way ablation with zero file-swap risk. New authoritative baseline (both flags True, reproducible via `scripts/devtools/authoritative_replay_fingerprint.py`): 3k region_count=77/core_member=414/consensus_attached=12.

**C11 hardening (`osn_gs/surface/torch_directed_boundary_ordering.py`)**:
- **Fixed**: `_decompose_into_paths_and_cycles` silently dropped nodes that were neither a matched source nor matched target (zero compatibility survived matching in both directions) — box_face(cap=27)'s 19 candidates only summed to 18 in output. Now explicit `isolated_boundary_candidate` state, 100% accounting verified.
- **Fixed**: candidate>150-per-region silently fell back to a lower-confidence greedy heuristic (worklog 35's own defensive fallback) — replaced with explicit `ordering_capacity_exceeded` fail-closed state (Case C), never silently degrading correctness contract.
- **Added**: `osn_gs/surface/torch_boundary_self_intersection.py` — explicit segment-crossing validation (not just turning-angle/z-stddev) wired into `torch_visible_boundary_materialization_adapter.py` before NURBS fitting. Winding-number-via-centroid was found numerically unstable near-degenerate cases; replaced with turning-angle-sum (only compares adjacent edges, no centroid dependency).
- **Tried and REJECTED**: cardinality-first lexicographic matching objective (maximize matched-node count first, score second) — recovered box_face's stranded node (15→16-node loop) but FRAGMENTED cylinder's clean 88-candidate side wall into 3 pieces + isolated nodes. Reverted to max-score-only; the box_face residual is honestly disclosed as unresolved, not force-fixed.
- **Tried and REJECTED**: pre-admission branch/Y-junction diagnostic (flag nodes with compatibility degree>2 and ambiguous score margin, demote their cycle to review_required) — false-positived on EVERY node of box_face's regular 9x9 grid (symmetric spacing makes near-tied scores the norm, not evidence of a real branch). Diagnostic function (`_diagnose_branch_ambiguity`) kept in the module but not wired into admission.

**C9 verification (no new production defect found)**: `contradicting_parallel_neighbor_count`'s "nearby" semantics confirmed to be genuinely graph-neighborhood-bounded (reuses the affinity graph's own bounded-kNN candidate edges, no new scene-specific radius) — audited 10 real-3k override candidates, mix of vetoed/passed confirms real discrimination, not a degenerate always-true/false condition.

**Real snapshot diagnosis, precisely split (new this round)**:
- Ambiguous-unassigned waterfall (R1-R6): R2 ("same_surface neighbors exist but NONE of them are in any region yet" — a chicken-and-egg gap, since growth only attaches to already-formed regions) dominates at 53-57% across 3k/5k/10k, ahead of R1 (zero same_surface degree, 31-38%). This means **core-seeding coverage itself**, not growth-threshold strictness, is the structural bottleneck — confirms worklog 35's `bridge_min_shared_neighbor_for_well_supported=2` finding from the opposite direction.
- Physical boundary candidate coverage: of 77 real-3k regions, 69 (90%) fail purely on candidate count <3 (structurally can't close regardless of ordering quality); only 8 (10%) have >=3 candidates and still fail — and those 8 are all small (3-8 members) with candidates mostly `isolated_boundary_candidate` (zero compatibility), i.e. spatial-sparsity/candidate-recall failures, NOT the box_face-style "candidates plentiful but ordering fragments them" pattern. **Zero real-snapshot regions exhibit a pure ordering-solver defect.**

**Why:** User's own framing: don't force lexicographic/branch-gating improvements just because they help one fixture — verify against ALL positive controls first, and revert on any regression. Both rejected changes are documented with the exact regression that caused rejection, not silently dropped.

**How to apply:** Materialized NURBS on real snapshots is still 0 and this round did NOT change that (not required — task explicitly said materialization isn't a success criterion). The next real lever for real-snapshot materialization is `bridge_min_shared_neighbor_for_well_supported` (core-seeding threshold) — explicitly forbidden to touch without separate authorization in this and the prior round. Full pytest: 647 passed, 1 skipped, 0 failed, 188.68s.

Full detail: `docs/worklogs/36_directed_cycle_solver_contract_hardening_and_region_coverage_recovery.md`.
