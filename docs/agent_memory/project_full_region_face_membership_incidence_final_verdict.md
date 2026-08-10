---
name: project_full_region_face_membership_incidence_final_verdict
description: worklog 89 -- corrected full-region local-frame faces followed by chart-unit membership incidence; actual final NO-GO
metadata:
  node_type: memory
  type: project
  modified: 2026-08-10
---

Worklog 89 (`docs/worklogs/89_full_region_face_membership_incidence_final_go_no_go.md`) is the actual final boundary-first visible-constructor verdict. It supersedes Worklog 88's conclusion because Worklog 88 used global PCA-UV for the rotation system and chose a largest outer face from the chart-unit induced graph.

Corrected experimental modules:

- `torch_full_region_surface_face_topology.py`: use unchanged Worklog 82 full-region same-surface adjacency and each evidence vertex's existing covariance normal/tangent frame. Project actual neighbor directions locally, order by local tangent angle, and recover full-region observed half-edge face orbits before membership.
- `torch_chart_unit_face_incidence_partition_boundary.py`: mark faces fully supported by chart-unit membership; two unit-face incidences are chart interior, one is boundary. Preserve physical/crease/frontier provenance. Only a membership cut through a two-sided continuous full-region face incidence becomes `partition_seam`. Trace every independent boundary loop and preserve `outer_boundary`/`interior_boundary`; open, branching, non-manifold, untyped physical exterior, or unrepresentable holes fail closed.

No threshold/kNN/normal/residual/NURBS/UV ablation or new boundary heuristic was used. Worklog 79 coverage -> existing PCA-UV -> 6x6 NURBS -> held-out evaluation remained unchanged. The old topology module path is now a compatibility import to the corrected implementation; the rejected global-PCA/largest-face implementation was removed.

Real 7-region replay (`output/extent_ab/val89/chart_unit_face_incidence_partition_boundary_replay.json`, 3526 evidence): coherent 3108/3526 (88.15%); cut recoverable 6/3526 (0.170%, 0.193% of coherent); physical-only/mixed/seam-only 6/0/0; valid/extrapolative/unsafe/unresolved 0/0/6/3102; evidence-weighted held-out p95 2.675. Ten loops were proven, all outer; neither a safe domain nor a mixed/seam-only domain was recovered.

Candidate-zero audit: under the corrected full-region provenance lookup, 38 units/124 evidence are zero-candidate; only one 3-evidence physical-only unsafe unit recovers. The Worklog 87 prose's 64 is not reproducible: its saved artifact contains 71 zero-candidate units/164 evidence. All 71 were mapped by region/unit index into Worklog 89: 61 open/non-manifold, 4 coverage failures, 3 occupancy failures, 1 untyped physical exterior, 2 physical-only unsafe recoveries. Thus the purported 64 were not omitted.

Actual final verdict: **NO-GO**. 99.807% of coherent evidence cannot form a closed coverage-valid domain under the exact full-region-face -> membership-incidence contract, and the remainder is unsafe. Do not integrate Region->Charts canonically and do not follow this result with another boundary experiment. Canonical visible Gaussian training, region ownership, PCA-UV, and 6x6 NURBS remain unchanged.
