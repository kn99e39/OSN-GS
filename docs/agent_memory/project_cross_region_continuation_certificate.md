---
name: project_cross_region_continuation_certificate
description: "Worklog 40 (docs/worklogs/40_cross_region_continuation_certificate_and_boundary_topology_safety.md) — sphere's 22 false observed_support_termination candidates fixed by classifying cross-region relations at REGION-PAIR level (sphere same_surface=12/crease=0 vs box crease=32-33, cylinder crease=88-90, thin_slab parallel=57); box/cylinder/thin_slab boundaries fully preserved; worklog 39's 2-hop certificate proven topology-safe on U-shape/hole/narrow-neck/near-touching; real 3k/5k/10k waterfall CONFIRMS bottleneck is candidate extraction (R1/R2 84-94%) with ordering failures at exactly ZERO"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-08-03T03:32:48.353Z
---

Follows [[project_boundary_adjacency_semantics_separation]] (worklog 39).

**Sphere false-boundary fix — the key insight is the level of aggregation.** Sphere fragments into two ~hemispheres and emitted 22 `observed_support_termination` candidates on the seam. Per-Gaussian-pair lookup makes all 22 look like "cross-region support with no relation evidence", because bounded-kNN frequently omits the direct edge between a candidate and the specific Gaussian filling its gap. Aggregating at REGION-PAIR level instead, the affinity graph's verdict is unambiguous:

| fixture | region pair evidence | verdict |
|---|---|---|
| sphere | same_surface=12, **crease=0** | smooth_continuation |
| box | crease=32-33 on every face pair | crease_adjacent |
| cylinder | crease=88-90 (side/cap) | crease_adjacent |
| thin_slab | parallel_but_separate=57 | parallel_separate |

Sphere is the ONLY fixture whose touching regions are crease-free and same_surface-bearing. New `classify_cross_region_pairs()` in `torch_boundary_support_termination.py` aggregates relations the affinity graph already computed (no new geometry, no new threshold, bounded to existing candidate edges). Only a `smooth_continuation` verdict reclassifies a candidate, and only to the existing non-physical `reliability_frontier` state with provenance kept. Result: sphere genuine 22 -> 0, while box (110 genuine, closed=5), cylinder (74, closed=3), thin_slab (48, closed=2), box_face (32, closed=1) are all unchanged. Folded-sheet angle sweep confirms 90°/120° real folds get zero reclassification.

This is the correct version of what worklog 39 attempted and reverted: the crude rule (suppress whenever ANY out-of-region support occupies the arc) destroyed box 110->0, cylinder 74->0, thin_slab 48->3, because those are exactly the crease/parallel cases. Consulting the relation class is what separates them.

**Worklog 39's direct-or-2-hop certificate is now proven topology-safe** (it was unverified before). New adversarial fixtures: U-shape concavity (1 loop of 56, zero nodes inside the notch, loop traces the notch walls at x=±0.36 with no |x|<0.34 entry), sheet with hole (outer 48 + inner 4 kept separate), narrow neck (27+27, lobes never fused), near-touching patches at gap 0.30/0.45/0.60 (always 24+24, zero loops spanning both). All recovered loops pass `validate_simple_closed_loop`. Y-junction/interior-stub controls from worklog 39 still hold.

**Real 3k/5k/10k waterfall — bottleneck definitively located.** New `scripts/devtools/trace_real_snapshot_boundary_waterfall.py` classifies every region R1-R6:

| snapshot | R1/R2 candidate-starved | R3 compatibility | **R4 ordering** |
|---|---|---|---|
| 3k | 138/157 (88%) | 19 (12%) | **0** |
| 5k | 124/148 (84%) | 24 (16%) | **0** |
| 10k | 132/141 (94%) | 9 (6%) | **0** |

Zero ordering failures on every snapshot. Even the largest regions (21-28 members, spatial diameter 6-9) produce only 0-4 genuine candidates. Cross-region misclassification is NOT a real-snapshot factor either (reliability_frontier reclassifications: 3k=3, 5k=1, 10k=0). **The single remaining bottleneck is candidate extraction.**

**Histogram audit (diagnostic only, production unchanged):** exact rotation invariance measured at angles 0.0/0.37/0.91/1.57 rad — box_face 32 genuine/closed=1, cylinder 74/closed=3, sphere 0/closed=0 at every angle. Histogram (local angular exposure) and the cross-region certificate (is there real continuation in that direction) are separate gates at separate code points and do not override each other.

**Box 6th face: root-caused, deliberately not forced.** Face 4 has 2 candidates with zero out-degree; tracing them shows the problem is not compatibility or topology support but an irregular candidate set — only 2 of 4 corner candidates are generated, plus 4 spurious interior candidates. Evidence is genuinely insufficient for a square perimeter, so per the task's own instruction it was reported with numbers rather than forced.

**Why:** User required distinguishing genuine crease/parallel boundaries from nonphysical region frontiers before suppressing anything, required proving the 2-hop rule safe on adversarial topology, and required finishing the real waterfall.

**How to apply:** Next round must target candidate extraction on real data (R1/R2 = 84-94%) — not ordering (0 failures) and not compatibility (6-16%). Box face 4's missing corner candidates are the analytic miniature of the same problem and are worth fixing together. Two-phase seeding was deliberately untouched this round and remains provisional (micro-region 36-39%). Full pytest: 707 passed, 1 skipped, 0 failed, 184.67s.

Full detail: `docs/worklogs/40_cross_region_continuation_certificate_and_boundary_topology_safety.md`.
