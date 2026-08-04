---
name: project_region_quality_boundary_admission_repair
description: "Worklog 34 (docs/worklogs/34_region_quality_and_boundary_component_admission_repair.md) — region membership breakdown shows core_member is only 19-22% (growth barely worked, 80% stayed ambiguous_unassigned); Stage C boundary failure has TWO distinct root causes depending on scenario (C9 region-too-small for real checkpoints, C11 directed-ordering mutual-matching fragmentation for well-formed synthetic regions); fixed one real growth-loop bug (reused core-merge veto for single-node attach), materialization still 0"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T09:39:35.171Z
---

Follows [[project_representative_graph_scale_region_formation_repair]] (worklog 33), which recovered region_count (0->64-85) and advanced `boundary_failure_stage` to C (component admission) but left two open questions: are these regions coherent, and what exactly blocks Stage C. This round answered both with real numbers.

**Region quality**: of 2048 representatives, only 19-22% became `core_member` (392/447/357 across 3k/5k/10k); ~80% stayed `ambiguous_unassigned`; `consensus_attached` (region GROWTH — attaching loose nodes to existing regions) was essentially 0-1. Region size: median 4-5 members, only 1 singleton, 17-22% "micro" (<=3 members), 3-5 "major" (>10). Not fully fragmented micro-regions, but growth structurally wasn't working.

**Root cause of near-zero growth (found and fixed)**: `_seed_core_components` builds `boundary_conflict_edges` from 4 distinct veto reasons — CONSENSUS_CONTRADICTED, PATH_PHASE_ALIAS, oversized-footprint-parallel-veto (all genuine per-EDGE quality signals), and **weak bridge** (a component-MERGE-specific veto: "should two ALREADY-DISTINCT core clusters merge through this edge without enough independent cross-support"). 73% of all conflict edges (1039/1429 on the 3k snapshot) were weak-bridge-only. The region-GROWTH loop (attaching a single unassigned node to an EXISTING region — NOT a component merge) was reusing this SAME flat set, silently blocking 93/93 checked real-snapshot growth candidates that otherwise had sufficient same-region core support. Fixed narrowly in `torch_gaussian_surface_region_formation.py`'s growth loop: still exclude edges vetoed for edge-intrinsic reasons (contradicted/phase-alias/oversized-footprint), but no longer exclude purely-weak-bridge-vetoed edges from growth. Result: `consensus_attached` 0-1 -> 7-10 per snapshot. Did NOT touch core-to-core bridge-veto/merge-threshold itself (explicitly forbidden, and it's legitimately conservative for merging pre-existing distinct clusters).

**Stage C has two genuinely different dominant causes, not one**:
1. **Real long-horizon checkpoints (3k/5k/10k)**: `C9` — most regions have a median of only 1-2 genuine boundary termination candidates (a closed loop needs >=3), so no ordering algorithm could close a loop regardless of quality. This is a region-size/candidate-scarcity problem, not an ordering bug.
2. **Post-ADC synthetic single big region (`box_face`, 27 members, 1 region, 19 genuine candidates)**: candidates are plentiful (15/19 individually find a valid forward successor) but the directed-ordering algorithm's strict mutual-agreement matching + greedy augmentation (`torch_directed_boundary_ordering.py`) still fragments them into 6 disconnected open chains (max 7 nodes) instead of one 19-node closed loop — a genuine `C11` (ordering implementation limitation), untouched this round (explicitly out of scope, matches the "don't redesign G1" spirit for this equally sensitive algorithm).

**Boundary-local scale audit**: `local_spacing` inside `_recover_directed_boundary_components` is ALREADY correctly derived from same-region candidates' own measured spacing (median nearest-neighbor distance) — NOT a misused Gaussian-footprint/representative scale like the worklog 30-33 pattern. This stage was already well-designed; no scale-mismatch bug found here.

**Why:** User explicitly said "diagnose region quality first, then find Stage C's true first-failure point, only narrowly fix proven defects — don't force loops closed or loosen boundary thresholds by feel." Materialization (closed loop count) is STILL 0 after this round — both C9 and C11 are real, substantial, and were correctly left unresolved rather than forced.

**How to apply:** Do not claim boundary linking is "fixed" — `materialized_surface_count=0` on all three real snapshots AND on box_face/box/cylinder/sphere post-ADC positive controls. Next round must pick ONE of: (a) investigate why CORE cluster bridge-veto is so conservative that clusters stay small (touches region-merge policy, needs separate authorization), or (b) redesign `torch_directed_boundary_ordering.py`'s mutual-matching heuristic (a sensitive, previously production-adopted algorithm — needs its own careful, separately-scoped round, same caution as G1). Full pytest run only once at the very end of the session per explicit instruction: 612 passed, 1 skipped, 0 failed, 152.18s.

Full detail: `docs/worklogs/34_region_quality_and_boundary_component_admission_repair.md`.
