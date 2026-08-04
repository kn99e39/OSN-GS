---
name: project_seed_merge_semantics_correction
description: "Worklog 38 (docs/worklogs/38_seed_merge_semantics_correction_and_candidate_recall_audit.md) — worklog 37's exempt_intra_raw_component_unions_from_bridge_veto PROVEN a mathematical tautology (raw components computed from the core_eligible edge set itself, so 100% of edges exempt, bridge veto evaluated 0 edges, 47 articulation bridges unioned); reverted to False and replaced with an explicit two-phase seed/merge DSU; candidate rejection waterfall built; box confirmed C4 (compat edges < candidates); cylinder cap regression root-caused to sector-histogram smearing but left unpatched because every fix required a scene-tuned constant"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T15:32:43.194Z
---

Corrects [[project_core_seeding_coverage_candidate_recall_separation]] (worklog 37).

**Worklog 37's fix was a tautology — do not reintroduce it.** It computed "raw same_surface connected components" FROM the `core_eligible` edge set and then skipped the bridge veto whenever an edge's endpoints shared a raw component. By the definition of connected components, both endpoints of every edge in that set always share a component. Measured on the real 3k checkpoint: 2092/2092 core-eligible edges exempt, 0 inter-component edges, bridge veto evaluated **0** edges (vs 1244 with the flag off, 862 of them `weak_bridge_candidate`), and **47 articulation bridges** (single edge whose removal disconnects the graph, >=3 nodes on each side) were unioned anyway — including one fusing a 7-node and a 23-node cluster through one edge. Same pattern on 5k (0 vs 1180) and 10k (0 vs 1072). The flag is now `False` by default and kept only for ablation replay.

**Replacement: explicit two-phase DSU** (`separate_seed_and_merge_phases=True`, canonical), in `_seed_core_components_two_phase`:
- Phase 1 unions ONLY `seed_strong_edge`s — edges that pass every edge-intrinsic veto AND have `CONSENSUS_WELL_SUPPORTED` (which already means real shared same_surface neighbour support, so this is not "same_surface therefore strong"). Each resulting component is an independently valid seed.
- Phase 2 groups the remaining `weak_bridge_edge`s by phase-1 component PAIR and requires aggregate `merge_min_distinct_endpoint_support=2` distinct endpoints per side, on top of the unchanged per-edge bridge veto. A single fragile bridge gives 1/1 and can never merge; a refused merge leaves BOTH seeds intact.
- Typed edge categories added: `EDGE_SEED_STRONG`, `EDGE_WEAK_BRIDGE`, `EDGE_MERGE_SUPPORTED`, `EDGE_CONSENSUS_CONTRADICTED`, `EDGE_PHASE_ALIAS`, `EDGE_OVERSIZED_FOOTPRINT`.

Result (3k, frozen replay): articulation bridges unioned **47 -> 0**; core_member 414 -> 799 (worklog 37's inflated 908 came from the disabled veto); major-region coverage 57 -> 163; worst-case region normal dispersion 0.1333 (worklog 37) -> 0.0995 (two-phase) vs 0.0538 baseline. `bridge_min_shared_neighbor_for_well_supported` stayed at 2 — no threshold was changed.

**Candidate recall audit (worklog 37 left this undone)**:
- `scripts/devtools/trace_candidate_rejection_waterfall.py` replays every gate of `extract_support_termination_candidates` with measured value / threshold / signed margin and a first-failure classification.
- **box_face candidate precision = 1.000, recall = 1.000** (all 32 ground-truth boundary nodes generated, zero false positives) — the generator itself is correct.
- **sphere**: 22 genuine candidates, 0 closed loops — no invented outer boundary or seam.
- **box**: each face has 16-20 candidates but only 10-17 compatible directed edges. An N-node ring needs N edges, so every face is structurally edge-starved — this is **C4 (compatibility), not an ordering-solver defect**. Binding constraint is `accepted_core_pairs` (only 9-11% of possible pairs are in accepted topology). Directed ordering solver was NOT modified.

**Cylinder cap regression: root-caused but NOT fixed (honest disclosure).** `extract_support_termination_candidates`'s sector histogram smears each neighbour's occupancy into adjacent bins (+-0.15) to resist bin-boundary jitter. With enough neighbours this marks all 8 sectors occupied even when the measured geometric gap clearly clears its threshold, so `_missing_sector_runs` returns `()` and the node is dropped. Measured: cap nodes 257/266 have gap 1.568/1.590 rad against a 1.178 rad threshold yet occupied == all 8 sectors. One extra accepted neighbour (5->6) flips a node into this state, which is why *any* core-seeding improvement (worklog 37's B or worklog 38's C alike — confirmed both give closed=2 vs baseline closed=3) surfaces it as a "cap regression". Every attempted repair (independent gap-dominates-histogram threshold; deriving the missing run from the measured gap span) reduced to picking a constant just above this fixture's 1.568 rad — scene tuning, explicitly forbidden — so it was reverted and left as a documented diagnosis in the code.

**Why:** User demanded the worklog 37 fix be re-examined for logical validity before anything else, with explicit disqualifying conditions for keeping it, and required the candidate waterfall that worklog 37 skipped.

**How to apply:** Never reintroduce a "raw component" exemption computed from the same edge set the veto iterates — it is definitionally vacuous. The top next bottleneck is reconciling the smeared sector histogram with the geometric gap measurement without a scene-tuned constant (blocks cylinder cap closure); second is box's `accepted_core_pairs` sparsity (9-11%), newly localized to the compatibility/accepted-topology stage rather than candidate generation or ordering. Real 3k/5k/10k materialization is still 0 and was never a required criterion. Full pytest: 672 passed, 1 skipped, 0 failed, 218.67s.

Full detail: `docs/worklogs/38_seed_merge_semantics_correction_and_candidate_recall_audit.md`.
