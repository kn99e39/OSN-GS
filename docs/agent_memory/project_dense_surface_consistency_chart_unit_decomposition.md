---
name: project_dense_surface_consistency_chart_unit_decomposition
description: "worklog 82 -- evidence-scale surface-consistency components inside a region (Region<->Chart no longer 1:1), verdict B"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-08T06:05:27.158Z
---

Worklog 82 (docs/worklogs/82_dense_surface_consistency_chart_unit_decomposition.md): built `osn_gs/surface/torch_dense_surface_consistency_components.py` to break the implicit Region==Chart 1:1 assumption every prior round (61-81) made. Bounded-degree kNN candidate adjacency (never a radius graph/clique) over region-owned dense evidence; an edge becomes `same_surface` only when normal alignment>=0.85 AND mutual tangent residual<=0.35 (same thresholds `torch_gaussian_manifold_affinity.py` already uses at representative scale, reused unchanged, not retuned). An edge whose endpoints fall on opposite sides of an already-typed worklog-80 crease/frontier arc is vetoed regardless of geometry. Components = connected components of same_surface edges only; unresolved points and internally-inconsistent (`non_manifold_suspected`, >15% internal normal disagreement) components fail closed rather than materializing.

Wired per-component into worklog 80's dense chart support + worklog 79 coverage contract + PCA-UV (worklog 81 confirmed no better alternative) + 6x6 NURBS + held-out eval, replayed across all 7 real regions (`scripts/devtools/dense_surface_consistency_replay.py`).

**Result: 364 components total -> valid_supported 16 (regions 1/2/6), extrapolative 8, unsafe_geometry 9, no_chart 331 (91%)**. First time any real chart reached valid_supported across worklogs 79-81 (always 0 before). But severe fragmentation: median component size 3-6 points vs 92-1035 region evidence; even the largest per-region component (24-157 pts, passes non_manifold check) mostly ends extrapolative/no_chart. Verified fragmentation is NOT a normalization bug: swapping residual normalizer from kNN-spacing (chosen) to the reference `tangent_major_scale` default makes residual stricter (0.34->0.63 median), confirming the chosen normalizer was already the more permissive option -- real evidence noise, independently consistent with worklog 81's local-normal-disagreement finding.

**Verdict: (B)** -- evidence genuinely supports multiple chart units (16 valid_supported charts prove it, rejecting C), but current evidence-scale kNN/degree-cap topology resolution cannot assemble that signal into safe chart-scale units for most regions (91% no_chart, rejecting A's "adopt as canonical"). Regions 4/5 still produce zero valid charts, consistent with worklog 80's ambiguous-branching/open-topology classification.

Related: [[project_dense_parametric_chart_representation_redesign]], [[project_intrinsic_boundary_parameterization_decision]]
