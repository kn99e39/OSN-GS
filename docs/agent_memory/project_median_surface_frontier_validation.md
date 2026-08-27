---
name: project_median_surface_frontier_validation
description: WL122 — Candidate B median frontier is numerically coherent but semantic validity INCONCLUSIVE; arbitrary-3D boundary policy unapproved
metadata: 
  node_type: memory
  type: project
  originSessionId: 24cb901d-62f4-4192-8bb7-bb0b66edd28f
  modified: 2026-08-27T06:06:04.815Z
---

Worklog 122 validated **Candidate B only** (A/C retired, D historical context only).
B's decision function and `shared.aggregate_global` were left file-unmodified; no
epsilon, tolerance, threshold sweep or hybrid anywhere.

**Verdict: B — MEDIAN FRONTIER NUMERICALLY COHERENT BUT SEMANTIC VALIDITY
REMAINS INCONCLUSIVE.** (Not A, not C, not D.)

Established, on exhaustive measurement:
- **Frontier self-closure**: all 43,817,760 median events across 161 views,
  reconstructed (G2) and reprojected into their OWN camera → 18.62% classified
  OCCLUDED. But **raster-pixel reassignment = 0** (100% preserved), >8 ULP = **5
  events**, 96.93% of contradictions are exactly **1 ULP**, and rho3d/rho2d rates
  are 18.53%/18.95%. Closure loss is entirely a **world-coordinate float32
  round-trip**, established without tuning any epsilon.
- **An exact identity contract EXISTS**: representing a query as
  `(source camera, source pixel, stored median depth)` gives **0 contradictions
  out of 43,817,760**. Diagnostic only — B unchanged.
- **Structural property**: per-pixel post-median mass fraction maxes at 0.49993 —
  the median rule guarantees the frontier always keeps >half the ray's total
  contribution mass in front of it.
- **Post-median mass is 39.06% of total**, but **27.65% of it is physically IN
  FRONT** of the median (centre-depth traversal-order artifact) → genuinely
  behind = **28.26%**. Of post-median mass, **94.28% comes from surfels the
  renderer itself designates a median representative somewhere** (65.40% this
  view + 28.88% another view); never-designated is 5.72% → **independent
  behind-frontier evidence ≤ 2.24% of total contribution** (marginal upper bound).
- Cross-component share (20.44%) is NOT readable as second-surface evidence:
  WL107/109 topology splits 1.19M surfels into 559,989 components, 535,910 of
  them singletons.
- **Cross-view disocclusion**: 3,400 median-event anchors, 99.71% hidden in ≥1
  other view, **global OBSERVED retention 99.44%**; residual 19 (0.56%) end
  global OCCLUDED.
- **S1/S2/S3 PASS**; S4 (one opaque surface as 12 splats) keeps the frontier
  inside the physical surface with 100% same-component post-median; S5 places the
  frontier on the near visible layer. Translucent fixture kept OUT-OF-SCOPE.
- WL121's 300 fragmentation contexts replay **bit-identically** (endpoint A
  290/10, B 296/4, midpoint 300/0, gating 288/12/0). Midpoint OBSERVED was NOT
  used as surface-continuity or merge evidence.

Blocking promotion to VIABLE: OSN-GS's decomposition operates on **arbitrary 3D
points**, and there the frontier's own defining events flip 18.62% of the time;
fixing that needs an **unapproved numerical boundary policy** (an exact identity
existing is not the same as a policy being approved). Plus 0.56% residual global
contradictions, and the joint distribution was never measured.

**Remaining question**: should OSN-GS query the frontier on an exact identity
representation `(camera, pixel, stored median depth)`, or approve an explicit
numerical boundary policy for arbitrary-3D classification — and how does that
choice interact with the 28.26% of contribution mass genuinely behind the
frontier?

Explicitly NOT carried forward from WL121: the claim that physical-depth
reordering can only increase D's OCCLUDED count. The symmetric early-behind case
was never measured, so the direction is unestablished.

See [[project_worklog122_frontier_code_layout]]; predecessors
[[project_observed_occluded_volumetric_operationalization]] and
[[project_observed_occluded_value_space_supplement]].
