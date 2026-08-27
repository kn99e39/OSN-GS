---
name: project_observed_occluded_value_space_supplement
description: "WL121 — value-space audit of WL120; ranking survives, five provenance corrections, B advances to frontier validation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 24cb901d-62f4-4192-8bb7-bb0b66edd28f
  modified: 2026-08-27T05:20:09.498Z
---

Worklog 121 (branch `arch/2dgs-coverage-first-surface`) is a diagnostic supplement
to [[project_observed_occluded_volumetric_operationalization]], not a new
classifier batch. WL120's four decision functions and `shared.aggregate_global`
were left completely unmodified; the baseline replay gate reproduced the original
4,712-query bank and all A/B/C/D per-view/global state arrays **bit-identically**
before any value was interpreted (plus representative union 785,937 and WL107/109
topology 559,989 components / 535,910 singletons).

**Five provenance corrections, reporting layer only:**
1. WL120's `C_nearest_blocker_t` (median 0.99935) was MAX(t) = the **query-nearest**
   blocker. Camera-nearest (MIN t) is 0.9692; the blocker region's real world
   thickness is median 0.165 (0.083 on source-view anchors).
2. Candidate C's primitive is the `rho3d geometric footprint derived from the
   canonical alpha cutoff` — a strict SUBSET of what the renderer composites,
   because acceptance is `min(rho3d, rho2d)`. Never "complete renderer support".
3. Candidate D is `canonical traversal-order reachability`, not physical-depth
   prefix visibility.
4. Of 97,676 termination events, **`T_pre < 1e-4` in ZERO** (its minimum is exactly
   1.000e-4). The quantity compared against 1e-4 is `test_T = T_pre*(1-alpha)`
   (median 7.11e-5, alpha median 0.603). WL120's phrasing implied otherwise.
5. WL120's R5 same-region-midpoint approximation was replaced (for the supplemental
   bank only) by ACTUAL frozen fragmentation: 5,713,235 observed cross-component
   raster adjacencies → 300 contexts → endpoint A/B + midpoint.

**New full-population D fidelity audit**: 99.998% of relevant pairs sit on a pixel
with ≥1 accepted-event depth inversion (median 31); **81.2%** of REACHED
resolutions still see accepted events physically IN FRONT of the query afterwards
(median 6). D's OCCLUDED side (termination) is order-invariant and robust; its
OBSERVED-by-reaching side is materially order-dependent.

**Verdicts — WL120's ranking survives, with stronger evidence:**
- **A NOT VIABLE** — its 135,544 UNRESOLVED pairs have signed depth delta ≤ 0
  *without exception* and are **all B=OBSERVED**; 300/300 UNRESOLVED at real
  fragmentation midpoints. Its blind region is exactly "observed free space".
- **B ADVANCE TO DEDICATED FRONTIER VALIDATION** — only candidate passing coverage
  (0 UNRESOLVED of 383,322) and numerical coherence on real fragmentation queries;
  sole defect is float32 round-off (relative ≤ 2.15e-7, 1,653/2,640 exactly zero).
  This is NOT a claim that median depth is a physical first hit — WL120's S6 stands.
- **C NOT VIABLE** — median **95% of an anchor's blockers are in the SAME frozen
  visible component**; self-occlusion is same-surface overlapping footprints,
  confirmed against topology.
- **D NOT VIABLE AS STATED** — global OCCLUDED still 0, but D says per-view
  OCCLUDED on **35.2%** of the B-vs-D disagreement pairs; the frozen global
  aggregation is what erases them.

**B vs D**: strict containment with ZERO exceptions across 383,322 pairs, yet the
evidence favours "two different renderer-level questions" — D's OBSERVED bucket
mixes REACHED with CONTRIBUTOR_LIST_EXHAUSTED, which has no counterpart in B's
frontier notion. **No claim that a true boundary lies between them**; no threshold
was searched or swept.

Disclosed inability: the corrected physical-depth prefix transmittance for
late-front events was NOT computed (would need per-event alpha accumulation). Only
its direction is established — OCCLUDED can only increase, never decrease.

See [[project_worklog121_value_diagnostic_layout]] for where the code lives.
