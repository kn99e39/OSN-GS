---
name: project_representative_only_visible_nurbs
description: OSN-GS worklog 111 -- camera-observed chart domains + NURBS fit over frozen WL107/109 representative topology only; NOT VIABLE result
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-24T08:16:53.132Z
---

Worklog 111 (branch `arch/2dgs-coverage-first-surface`): froze [[project_renderer_native_topology_gate_closure]]'s canonical topology and [[project_nonrepresentative_evidence_attribution]]'s AMBIGUOUS/LAYERED SUPPORT verdict (non-representative evidence stays out of fitting), and tested whether the 785,937 MEDIAN_SURFACE_REPRESENTATIVE primitives alone can support a scene-covering continuous visible NURBS scaffold.

New pure-logic module `osn_gs/surface/torch_camera_observed_chart_domains.py`: builds NURBS chart candidates from each training view's own representative map, using the camera's own pixel coordinates as chart UV (not a new 3D PCA/kNN parameterization). Uses exact scipy connected-components (4-connectivity, edges only between same-canonical-component-id neighbors) so two different canonical components can structurally never share a chart. Reused `torch_nurbs.fit_torch_visible_surface_lsq`'s own established defaults (8x4 control grid, degree 2) unmodified; minimum chart membership (32) is mathematically derived from that control-point count, not tuned.

**Real-scene result (all 161 views, topology replay exactly matched WL107/109's known 36.77%/45.02%)**: raw chart candidates 1,163,380, but only 3,963 meet the 32-member validity threshold. Representative membership coverage 68.4%, pixel coverage 93.1% look good, but **per-component coverage distribution has median AND p95 = 0%** -- coverage is concentrated almost entirely in a handful of giant components (inherited from WL107/109's known fragmentation, 45.02% singleton), while the vast majority of the 155,457 representative-bearing components get zero valid charts. Even the covered minority fit poorly: residual median 0.032 but max 7.9 (scene units), overlap normal disagreement median 5.04 degrees but p95 57.9 degrees / max near 180 -- exactly the "one global UV domain forced on a huge/non-disk component" failure the directive warned against, because a giant component's single-view screen footprint often IS one connected blob.

Region breakdown: table_legs 88.9% (best), table_side_curved 57.3% (worst, curved rim), patio 77.8%, hedge 49.0% (worst region, matches known fragmentation).

**Architecture verdict: NOT VIABLE** with the current per-view-blob chart construction, attributed to two causes: (1) primary -- CANONICAL_TOPOLOGY_ISSUE, the frozen topology's own fragmentation structurally prevents most components from ever reaching 32 members in any view; (2) secondary -- CHART_PARAMETERIZATION_FAILURE, the few giant components that DO chart get fit as one undivided huge blob per view rather than subdivided, producing the long fitting/overlap tails. Chart subdivision was explicitly NOT implemented this batch (directive forbade new arbitrary thresholds/sweeps) -- left for a future architecture decision.

15 new focused tests (9 chart-domain construction, 6 synthetic contracts A-F from the directive), all passing. No CUDA/production/training code touched -- full regression not rerun. Real-scene script `scripts/devtools/representative_only_visible_nurbs.py`, report at `output/osn_gs_rep_only_nurbs/representative_only_visible_nurbs_report.json`, 11 named PLY/PPM/PNG review exports with Korean READMEs.
