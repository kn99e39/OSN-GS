---
name: project_dense_chart_unit_assembly
description: "worklog 83 -- chart-scale assembly layer over worklog 82 micro-components using aggregate (never pointwise) multi-signal evidence, verdict B refined"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-10T04:25:07.942Z
---

Worklog 83 (docs/worklogs/83_dense_chart_unit_assembly.md): built `osn_gs/surface/torch_dense_chart_unit_assembly.py` on top of [[project_dense_surface_consistency_chart_unit_decomposition]]'s micro-components (91% no_chart, median 3-6 pts). Component-pair adjacency is decided via AGGREGATE evidence, never a single pointwise edge: proximity gate (2.5x, reused from worklog 72's `_connect`), typed crease veto (reused from worklog 82), then requires >=2 of 3 independent signals (mean normal alignment>=0.85, count of individually-passing same_surface point-pairs>=3, `measure_edge_support_occupancy` (worklog 76, first use as an ACCEPTANCE signal rather than disclosure-only) showing no empty interior bin). All thresholds reused unchanged, no ablation. Fit quality is never a merge input (verified via signature test).

Real result: 364 micro-components -> 178 chart units (52% fewer) across 7 real regions -- fragmentation genuinely resolved: 75-93% of region evidence recovered into size>=4 chart-unit candidates, up to a 239-point single merged unit. Audited all 219 ACCEPTED edges: 0 relied on occupancy-bypass (normal+correspondence only) -- no unsupported gap bridging occurred. BUT no_chart fraction stayed ~91% (90.9%->91.0%, essentially unchanged) -- the bottleneck MOVED, not resolved: failures are now dominated by worklog-80's sparse macro-topology arc-typing lacking coverage for larger assembled shapes (38 cases) and dense chart boundary construction itself failing to close safely at the larger scale (coverage-failed 27 + self-intersecting 18 = 45 cases). valid_supported actually dropped 16->4 because merged/larger charts are harder to fit within the frozen PCA-UV+6x6 NURBS held-out bound (though one 74-point merged unit did stay valid_supported, p95=3.58, proving large valid charts are possible).

Verdict: (B), refined -- local surface units are real and CAN be safely assembled at chart scale via redundant aggregate evidence (rejects C again), but existing macro-topology resolution and chart-boundary/materialization capacity (not the assembly mechanism itself) still can't safely finish most of them into final charts (rejects A: no_chart rate unchanged). Region 4/5 remain fully unsupported, consistent with worklog 80.

Related: [[project_dense_surface_consistency_chart_unit_decomposition]], [[project_dense_parametric_chart_representation_redesign]]
