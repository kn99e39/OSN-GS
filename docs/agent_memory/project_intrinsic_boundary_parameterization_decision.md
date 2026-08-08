---
name: project_intrinsic_boundary_parameterization_decision
description: "worklog 81 -- PCA-UV vs Tutte-embedding intrinsic parameterization decision on worklog 80's 4 passing real charts"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-08T04:28:50.030Z
---

Worklog 81 (docs/worklogs/81_intrinsic_boundary_conditioned_parameterization_decision.md): built `osn_gs/surface/torch_intrinsic_boundary_parameterization.py` -- a boundary-conditioned discrete harmonic (Tutte) embedding fixing worklog 80's dense chart boundary to a convex unit-circle domain and solving interior UV via a 3D kNN graph, guaranteeing injectivity onto the disk. Compared against production `pca_parameterize_points` through the identical `fit_torch_visible_surface_lsq` call (only `initial_uv` differs) on the 4 real regions (0/1/2/3) that pass [[project_dense_parametric_chart_representation_redesign]]'s coverage contract.

**Result: intrinsic is worse than PCA-UV in every region** -- neighborhood preservation 0.53-0.70->0.39-0.63, fold% 21-36%->47-70%, large new UV near-collisions in regions 1/2 (2->79, 2->100). Confirmed not a graph-sparsity artifact (`knn_k` 8->20, same result). Root cause measured directly: owned evidence's own local PCA normals (k=10 kNN) disagree with the region canonical normal by >60 degrees for 16.3-37.4% of points, and normal-direction thickness is 17-55% of tangent extent -- the evidence itself is not locally flat, independent of any parameterization choice.

**Verdict: A (replace PCA-UV) clearly rejected** -- an injectivity-guaranteed alternative is measurably worse, so the failure is not a parameterization-choice problem. **Between B and C**: evidence doesn't support a single chart (now has direct local-normal/thickness evidence), but `independent_chart_components` (worklog 80) is 1 for all 4 regions, so existing topology cannot justify a split. No multi-chart split applied; production PCA-UV NOT replaced (verdict isn't A, so no full regression run per instruction, only focused: new 12 + existing 48, all pass).

Forward direction flagged in Master doc: combine the local-normal-disagreement evidence with topology-resolution-vs-evidence-density to justify a future chart-decomposition contract.

Related: [[project_dense_parametric_chart_representation_redesign]], [[project_constructor_chart_domain_coverage_verdict]]
