---
name: project_per_representative_reliability_gate_trace
description: "Worklog 31 (docs/worklogs/31_per_representative_reliability_gate_trace.md) — diagnostic-only per-representative reliability trace revealed region_seed_core=0 is caused by manifold-affinity-graph candidate scarcity (representative spacing 12-16x their own tangent scale), NOT by low final reliable_count as worklog 30 assumed; zero production code changed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T07:39:54.096Z
---

Follows [[project_long_horizon_reliability_collapse_repair]] (worklog 30). That worklog's framing — "final reliable_count is low (7-9/2048), so region_count=0" — turned out to be an incomplete causal story. This diagnostic-only round (explicitly no threshold/normalization/radius/region-admission changes, zero production code modified) traced every representative through the real production code path and found:

**`region_seed_core = 0` at all three snapshots (3k/5k/10k), independent of final reliable_count.** Read directly from the code (`torch_gaussian_surface_region_formation.py::_seed_core_components`): region-seed edge eligibility checks ONLY `intrinsic_class == INTRINSIC_RELIABLE` on both endpoints — the combined/contextual `reliability_class` (worklog 30's "final reliable") is NOT an input to region seeding at all. So even though final_reliable=7-9 is real, it's not what's blocking regions.

**The actual blocker is one layer further upstream**: `build_manifold_affinity_graph`'s candidate generation (`torch_gaussian_manifold_affinity.py`) requires representative-pair distance to be within `scale_radius_multiplier=6.0 x` each representative's own `tangent_major_scale` to even become a "candidate" edge. Measured directly: representative nearest-neighbor spacing is 12.25x (3k) to 15.93x (10k) larger than their own tangent_major_scale on average — so 94.7-97.0% of kNN-considered pairs never become candidates at all (`outside_candidate_support`). Of the tiny fraction that DO become candidates, only ~2% classify as `same_surface` (rest: `ambiguous`/`parallel_but_separate`) — same root pattern as worklog 30's full-neighborhood `tangent_residual` finding, but now manifesting in the PAIRWISE representative-to-representative affinity graph. Result: 2027/2048 (99%) representatives have same_surface degree 0, so zero core seeds are possible regardless of reliability class.

Contextual-gate-level finding (matches/refines worklog 30): among intrinsic-reliable-but-not-contextual-consistent representatives, the `tangent_residual` gate (threshold 0.35) is the dominant single failure cause (93-98% of that population), confirmed per-representative with signed margins, not just via worklog 30's median statistic. 28-38% of representatives fail MULTIPLE gates simultaneously — `first_failed_gate` alone would undercount this.

**Why:** User explicitly required a single diagnostic-only batch (no threshold/normalization/radius/region-admission edits, no new reliability states) to precisely trace where each representative fails before any further repair round.

**How to apply:** Next repair round (not yet authorized/done) should NOT just target the contextual `tangent_residual` gate in isolation — the manifold-affinity `scale_radius_multiplier`/candidate-generation stage is an equally or more fundamental bottleneck for region formation specifically (though it's a separate code path/config from worklog 30's `local_radius_tangent_scale_multiplier` fix, which only affected `compute_full_neighborhood_evidence`, not `build_manifold_affinity_graph`). Both point at the same underlying pattern: a single representative Gaussian's own learned covariance scale is a poor proxy for true local surface/spacing scale in real long-horizon-trained data. Don't fix speculatively — this needs its own dedicated, evidence-gated round.

New diagnostic tool (offline only, not production): `scripts/devtools/trace_representative_reliability_gates.py` — reads already-computed production result objects only (no reliability logic reimplemented), emits per-representative JSONL + gate waterfall + dominant-failure histogram.

Full detail: `docs/worklogs/31_per_representative_reliability_gate_trace.md`.
