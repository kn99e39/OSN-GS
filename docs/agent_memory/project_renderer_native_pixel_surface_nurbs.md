---
name: project_renderer_native_pixel_surface_nurbs
description: "OSN-GS worklog 112 -- controlled A/B swapping WL111's representative-center 3D fitting target for renderer-native per-pixel median-depth unprojection; NO material improvement"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-24T08:55:03.656Z
---

Worklog 112 (branch `arch/2dgs-coverage-first-surface`): preserved [[project_representative_only_visible_nurbs]] (WL111) exactly -- same chart connectivity, same image-space UV, same fixed 8x4/degree-2 NURBS config -- and changed ONLY the 3D fitting target from the representative surfel's own center to the renderer's own per-pixel `median_depth` (`out_others[MIDDEPTH_OFFSET]`, captured at the exact same T>0.5 crossing as WL107's `median_surfel_id`), unprojected via the already-established OFFICIAL_CODE_FAITHFUL `osn_gs/render/surfel_geometry.py::depths_to_points` (no new mechanism, no CUDA rebuild needed -- `out_others` already exposed the channel). Minimum chart validity switched from representative-count to PIXEL-sample-count (still 32, derived from the 8x4 control grid, not component size -- directive explicitly forbade re-treating small components as a topology failure).

**Real-scene result (all 161 views, topology replay again exactly matched 36.77%/45.02%)**: valid charts jumped 3,963 -> 14,900 (pixel-count threshold much more lenient), representative membership coverage 68.4% -> 71.5%, pixel coverage 93.1% -> 94.7% -- modest gains. **But per-component coverage distribution stayed EXACTLY median=0%/p95=0%, unchanged from WL111** -- coverage is still concentrated in the same handful of giant components. Fitting residual median/p95 essentially unchanged but **max exploded 7.9 -> 1517.2**; overlap position discrepancy median got WORSE (0.030 -> 0.055) and max exploded 8.1 -> 1514.4 (dense per-pixel sampling exposed renderer depth noise that WL111's per-representative averaging had inadvertently smoothed). Region coverage improved modestly (curved table rim 57.3% -> 62.2%, hedge 49.0% -> 54.8%) without any corresponding fit-quality improvement.

**Architecture verdict: NO** -- representative-center/pixel-surface mismatch was NOT the primary cause of WL111's failure (directive's "materially improve coverage/residual/overlap" bar not met; residual and overlap actually got worse in several respects). The remaining bottleneck is still WL111's identified NURBS chart capacity/granularity problem (giant undivided blobs forced into one 8x4 control grid) -- this batch made that problem worse with denser data, not better. Per directive: reassess the representation contract (chart subdivision/granularity) before adding complexity, do not chase representative-center vs pixel-surface further.

9 new focused tests (including real-CUDA proof that the same representative surfel produces distinct per-pixel 3D positions), all passing. No CUDA/production code touched. Real-scene script `scripts/devtools/renderer_native_pixel_surface_nurbs.py`, report at `output/osn_gs_pixel_surface_nurbs/renderer_native_pixel_surface_nurbs_report.json` (loads and directly compares against WL111's own report), 11 named review exports including a copy of WL111's own `VISIBLE_NURBS_PATCHES` baseline for side-by-side comparison.
