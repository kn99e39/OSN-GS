---
name: project_observed_occluded_volumetric_operationalization
description: WL120 — four competing Observed/Occluded operationalizations (Surface-Hit/Median-Depth/Geometric-Visibility/Renderer-Reachability) all fail; no winner declared
metadata: 
  node_type: memory
  type: project
  originSessionId: 24cb901d-62f4-4192-8bb7-bb0b66edd28f
  modified: 2026-08-26T09:53:54.717Z
---

Worklog 120 (2026-08-26, branch `arch/2dgs-coverage-first-surface`) compared FOUR
independent architecture hypotheses for splitting the camera-supported 3D domain
into OBSERVED / OCCLUDED, on one deterministic 4,712-query bank across all 161
training views (758,632 query-view pairs), under a frozen global aggregation
(any relevant view OBSERVED ⇒ GLOBAL OBSERVED; all relevant views OCCLUDED ⇒
GLOBAL OCCLUDED; else UNRESOLVED — no vote, no multiplicity threshold).

**Verdict: all four fail, no winner.**

- **A Surface-Hit** — NOT VIABLE, EVIDENCE-STARVED. per-view OBSERVED 0.70%;
  each anchor is OBSERVED in exactly one view (min=median=max=1); 98.0% of
  front-of-surface probes UNRESOLVED. Surface-hit is a measure-zero test, so it
  cannot represent observed free space at all (synthetic S2 shows the same).
- **B Median-Depth** — INCONCLUSIVE. Only candidate passing both coverage
  (UNRESOLVED 0.00%) and contradiction (6 cases, all float32 round-off at
  relative 8.6e-8; the 507 source-view "contradictions" are the same round-off,
  quantified via the `query_depth - median_depth` distribution). But synthetic S6
  (first contributor 4.00 / median crossing 4.05 / canonical termination 5.25 at
  distinct depths) shows the median plane is NOT a first-surface boundary.
- **C Geometric Visibility** — NOT VIABLE, FATAL contradiction. 98.67% of R1
  anchors are OCCLUDED **in their own generating view**. Cause: blocker count
  median 16, nearest_blocker_t median 0.99935 — a trained 2DGS surface is a thick
  soup of overlapping discs, so a ray to a surface point always crosses other
  discs of the same surface first. Support came from the canonical alpha cutoff
  (`rho_max = 2 ln(255·opacity)`), so no k-sigma was invented.
- **D Renderer Reachability** — NOT VIABLE AS STATED. per-view OCCLUDED 25.48%
  (real signal) but **GLOBAL OCCLUDED exactly 0**. Canonical termination needs
  T < 1e-4, so a single semi-transparent primitive behind is still OBSERVED
  (S3b). Zero new thresholds — it observes only the canonical `test_T < 0.0001f`.

**Structural finding**: B and D are a containment pair (`OCCLUDED→OBSERVED: 654`,
reverse 0) — not competing boundaries but the conservative/aggressive ends of one
axis. Choosing between them would require inventing exactly the threshold the
batch forbids.

**Remaining architecture question**: does a binary boundary naturally exist on
this representation at all, or is Observed/Occluded intrinsically a continuous
reachability quantity that cannot be dichotomized without a new threshold?

Frozen-state fingerprint (median surface representative union over all views) =
785,937, matching WL119 exactly — model/cameras/renderer untouched. Zero tracked
files modified; everything new. See [[project_worklog120_code_layout]] for where
the code lives, and [[project_visible_nurbs_geometry_uv_control_correction]] for
the WL119 G2 provenance this batch reused.
