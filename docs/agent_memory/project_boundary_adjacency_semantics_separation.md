---
name: project_boundary_adjacency_semantics_separation
description: "Worklog 39 (docs/worklogs/39_boundary_adjacency_semantics_and_angular_coverage_audit.md) — worklog 38's 'sector histogram smearing is the top bottleneck' was DISPROVEN by node-level measurement (the histogram is load-bearing: without it a closed sphere emits 154 false candidates instead of 22); the real defect was accepted_core_pair (region-topology evidence) being used as boundary perimeter adjacency, fixed by allowing 2-hop paths via a NON-candidate interior node — cylinder closed 2->3, box 0->5, sphere still 0; sphere's 22 false candidates root-caused to region fragmentation at the hemisphere seam, guard attempted and reverted"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T16:46:27.813Z
---

Follows [[project_seed_merge_semantics_correction]] (worklog 38).

**Worklog 38's stated top bottleneck was an misdiagnosis — do not act on it.** It concluded that the sector histogram's +-0.15 bin smearing "vetoes a valid geometric gap" and needed a principled replacement by continuous angular coverage. Node-level comparison of the two gates across every analytic fixture shows the opposite: the histogram is doing real work. On a closed sphere (no physical boundary anywhere) the exact geometric gap alone accepts all 154 evaluated nodes; the histogram vetoes 108 of them, leaving 22. Replacing the histogram with continuous geometric coverage (the task's own candidate B/E) would make sphere false positives **7x worse**, not better. The histogram is retained unchanged; bin count and smearing multiplier were not touched.

**The real defect (fixed): `accepted_core_pair` misused as boundary adjacency.** `internal_accepted_edge_ids` is built from the bounded-kNN affinity graph and answers "are these two Gaussians linked in the region's connectivity" — not "are these two boundary candidates consecutive on the perimeter". Requiring a DIRECT accepted edge between boundary candidates rejected genuinely perimeter-adjacent pairs whose direct affinity edge was dropped by bounded-k. Measured on box: every face loses 5-11 perimeter-adjacent pairs to this gate (leaving 10-17 compatible edges where an N-node ring needs N, so no face closes), and **all 45 such rejected pairs across the six faces are reachable by a 2-hop path in the region's own accepted graph** (0 needed 3 hops, 0 had no path). box_face, which does close, loses exactly 0.

Fix in `torch_directed_boundary_ordering.py`: `_has_region_topology_support()` accepts a direct accepted edge **or** a 2-hop path through a shared neighbour that is itself NOT a boundary candidate. All geometric gates unchanged, no threshold touched, no cross-region connection, Hungarian objective and cycle decomposition untouched, NURBS fitting untouched.

Results: **cylinder closed 2 -> 3** (the section-16 required regression fix), **box closed 0 -> 5** of 6 faces, box_face/thin_slab/floater/contamination all unchanged, **sphere still 0**. All 11 recovered loops pass `validate_simple_closed_loop` and stay on a single planar face (min bbox axis extent 0.0023-0.0059).

**The non-candidate restriction is load-bearing, not cosmetic.** Allowing any shared neighbour broke the Y-junction negative control: an interior stub (radius 0.6 vs the ring's 1.0) with a single accepted edge to ring node 0 became "adjacent" to node 1 via that node and got spliced into a 13-node cycle — a fabricated boundary through a non-perimeter node. Perimeter-consecutive candidates are separated by interior surface so their bridging evidence is an interior node; a branch stub's only route is through another candidate. That asymmetry is the discriminator.

**Sphere's 22 false candidates: root-caused, NOT fixed.** The sphere fragments into two ~hemispheres (99/90 members) and all 22 `observed_support_termination` candidates sit exactly on the seam (z in [0.084,0.293] and [-0.292,-0.084]). Each has ~25 same-region AND ~26 other-region observed neighbours inside its own support radius — the "gap" direction is full of real Gaussians that merely carry a different region id. This is region frontier being promoted to physical boundary. A guard demoting such candidates to `reliability_frontier` when out-of-region support occupies the outward arc was implemented and measured: it fixes sphere (22->0) but destroys genuine candidates on box (110->0), cylinder (74->0, closed 2->0) and thin_slab (48->3), because on a polyhedral solid a real patch boundary legitimately abuts another region across a real crease. Out-of-region support alone cannot separate "this surface continues" from "a different surface meets here" — that needs the affinity graph's crease/parallel relation evidence wired through to the termination stage. Reverted, reported.

**Two-phase seeding remains PROVISIONAL.** It controls false bridge merges (articulation unions still 0, no thin_slab/crease/floater false merge) but micro-region ratio stays at 36-39% vs the worklog-36 baseline's 7-12%. The section-3/4 component-level seed-admission audit and S2-S7 candidate comparison were not completed this round, so no criterion was adopted to suppress it. Do not promote two-phase to canonical on core_member count alone.

**Why:** User required verifying worklog 38's conclusions before extending them, forbade scene-tuned thresholds and directed-solver changes, and required separating region-seeding topology from boundary-ordering adjacency.

**How to apply:** Next bottleneck is sphere region fragmentation (fix the region split, or wire crease/parallel evidence into termination) — not the histogram. Real 3k/5k/10k still closed=0; a transient closed=1 (3-node triangle) appeared mid-round via a candidate-to-candidate 2-hop route and correctly disappeared once the non-candidate restriction landed. Full pytest: 686 passed, 1 skipped, 0 failed, 172.52s.

Full detail: `docs/worklogs/39_boundary_adjacency_semantics_and_angular_coverage_audit.md`.
