---
name: project_local_rank_complete_chart_network
description: "OSN-GS worklog 114 -- replaced WL112's one-blob-one-chart unit with deterministic local rank-complete NURBS chart domains; fit quality/domain shape improved but coverage dropped and overlap-normal consistency got worse -- NOT VIABLE"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-25T09:03:02.134Z
---

Worklog 114 (branch `arch/2dgs-coverage-first-surface`): froze WL107/109 topology, WL112 renderer-native pixel-surface geometry, and the fixed 8x4/degree-2 NURBS config, and changed ONLY the chart UNIT -- from WL112's "one camera-connected blob == one NURBS chart" to "blob -> deterministic pole-of-inaccessibility-seeded BFS growth -> stop at the first pixel count where the fixed 8x4 design matrix reaches full column rank -> multiple local NURBS patches." New module `osn_gs/surface/torch_local_rank_complete_chart_growth.py`.

**Critical performance bug found and fixed before real measurement**: the first implementation's BFS (`_bfs_order`) eagerly enumerated the ENTIRE remaining blob for every single chart extracted, making cost O(blob_size^2 / chart_size) per blob -- a real 272,160-pixel synthetic blob stalled indefinitely. Fixed via a lazy `_bfs_levels` generator that only traverses as far as each chart's own closure actually requires; also moved all rank-check tensor ops to CPU (the original per-candidate CUDA round-trip dominated cost for these tiny 32-column matrices). After the fix, the same worst-case blob closed in 235s (1,784 charts).

**Controlled real-scene comparison** (topology/representative sweep on the FULL 161 views as always; the expensive chart-growth+fit stage bounded to 8 stride-20 views for BOTH arm A [WL112 baseline, recomputed on the same subset] and arm B [new method] -- disclosed scope reduction, not full-scene): chart count 889->14,137 (15.9x, under the 20x auto-explosion threshold), residual median/p95 improved ~9x/~8x, fraction of charts with holes 46.1%->15.7%, aspect ratio p95 3.36->1.22 (much more compact/square domains) -- **but representative coverage DROPPED 11.7% (consistently across all 5 regions) and overlap normal-disagreement got materially WORSE (median 5.8deg->18.2deg, p95 59.3deg->96.5deg, near-saturated)**. WL113's D-outlier persisted identically in the new method (one hedge chart, rank=32/full, depth_std=32.5, dominates BOTH the residual-max and overlap-max top-1 slot simultaneously) -- confirming [[project_design_intent_specification_implementation_traceability_audit]]'s (WL115) pre-registered concern that full column rank (algebraic identifiability) does not entail geometric chart validity.

**Architecture verdict: LOCAL_CHART_UNIT_NOT_VIABLE.** Directive's own multi-factor judgment criteria (fit quality AND domain shape AND coverage-not-dropped AND overlap-not-worsened AND reasonable patch count) were only partially met -- coverage and overlap-normal both failed materially. Per directive: stop and reassess NURBS representation itself before adding further mechanisms; do not implement adaptive capacity/merging/attachment/Trust/occluded surface next.

9 new focused tests (deterministic rank-closed growth, full-rank verification, component isolation, hole/gap non-crossing, insufficient-support preservation, runtime safety valve), all passing. Real-scene script `scripts/devtools/local_rank_complete_chart_network.py`, report at `output/114_osn_gs_local_chart_network/local_rank_complete_chart_network_report.json`, 11 named review exports (combined `preview_png/` folder per the corrected convention).
