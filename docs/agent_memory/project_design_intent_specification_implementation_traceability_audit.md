---
name: project_design_intent_specification_implementation_traceability_audit
description: "OSN-GS worklog 115 -- pure audit (no implementation) of WL107-113's intent->specification->implementation->result causal chain; no intent-level or implementation-deviation failures found, failures are predominantly specification-induced"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-25T09:03:18.777Z
---

Worklog 115 (branch `arch/2dgs-coverage-first-surface`): a pure design-intent/specification/implementation traceability audit of Worklogs 107-113, requested explicitly as NOT another ablation experiment -- no code touched, no new mechanism implemented.

**Verdict**: no pure INTENT-level failure (I) and no IMPLEMENTATION DEVIATION (III) were found anywhere in the audited scope. Observed failures are predominantly **SPECIFICATION-INDUCED (II)**: the chart unit definition (one-blob-one-chart, which WL111's own text already frames as "the simplest first test," not a canonical commitment), the rectangular [0,1]^2 UV domain (a mechanical consequence of combining "image coords as UV" with per-blob bbox normalization, never independently justified -- **key concrete finding: `osn_gs/surface/torch_nurbs.py::TorchNURBSSurface` already has an unused `uv_support_mask` trimming field built for exactly this, never exercised in the WL111-114 lineage**), and per-view non-merged chart identity (the direct, necessary cause of every batch's persistent overlap-disagreement metric). Secondary causes: **CONTROL-EXPERIMENT LIMITATION (IV)** for the frozen 8x4/degree-2 NURBS capacity (WL111 explicitly reused an unrelated pre-existing function default, never validated for this architecture), and **DATA/RENDERER PHENOMENON (V)** for WL113's D-signature (renderer median-depth genuinely jumps at specific pixels, e.g. depth 8.76->1723 within one small same-component chart -- not an implementation artifact).

**Pre-registered concern about WL114** (the rank-complete local-chart proposal, launched the same session): "full column rank" is an algebraic identifiability criterion, not a geometric chart-validity criterion -- nothing guarantees a rank-complete pixel region is compact/disk-like/depth-continuous. Flagged this explicitly BEFORE WL114's results landed. WL114's own real-scene measurement then directly confirmed the gap: one hedge chart reached full rank (32) while having depth_std=32.5 and dominating both the residual-max and overlap-max top-1 slot simultaneously.

**Design debt / canonical core split**: proven canonical contracts (renderer T>0.5 median representative as observation primitive; image-space 4-neighbor + multi-view positive union topology, GATE PASS; representative vs non-representative evidence always tracked separately; AMBIGUOUS/LAYERED evidence never forced into ownership; image coords as UV, not invented 3D parameterization; visible-topology-evidence vs NURBS-materialized-evidence always reported separately) vs unresolved representation choices (the chart unit itself; rectangular vs trimmed UV domain; whether 8x4 is final capacity; per-view vs merged chart identity; whether full rank is an adequate geometric-validity proxy; how to eventually represent non-representative ambiguous evidence).

No code changes, no full regression (static/semantic audit only, per directive).
