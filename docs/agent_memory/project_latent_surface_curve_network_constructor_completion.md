---
name: project_latent_surface_curve_network_constructor_completion
description: worklog 96 -- legacy boundary gate removed, multi-patch curve-network construction works, but safe NURBS did not improve over worklog 95; bottleneck moved to downstream patch fitting
metadata:
  node_type: memory
  type: project
  modified: 2026-08-18
---

Worklog 96 completes the Worklog 95 architecture by removing the legacy `eligible_parametric_chart_boundary` entrance gate and solving 4 structural limitations together. Visible Gaussian training, ADC, region ownership, and the existing NURBS fitter remain fixed and unmodified.

New modules:
- `osn_gs/surface/torch_latent_surface_seed_curves.py`: boundary/crease/frontier seeds (existing chart `segment_kind` mapped unchanged) are preferred and preserved whenever they exist; interior-construction seeds (deterministic farthest-point-sampled anchors, fixed count 6, each individually verified supported) are used ONLY as fallback when no boundary seed survives. Anchors are starting locations only -- no raw Gaussian-center adjacency is built between them.
- `osn_gs/surface/torch_latent_surface_curve_tracer.py`: `propagate_tangent_onto_plane` projects the previous walking direction onto each new local tangent plane and re-grounds it in the realized displacement (parallel-transport style) -- never re-selects an arbitrary PCA-axis sign per step. `sample_segment_continuous_support` densifies a chord and requires EVERY intermediate sample supported, not just the two endpoints.
- `osn_gs/surface/torch_latent_surface_curve_families.py`: per seed, family V = transversal traces (one per seed sample), family U = continuously-supported rungs connecting adjacent transversal traces at shared depths. Fixed provisional contract (not tuned from replay): >=2 curves per family, >=2 depths connected by a validated rung between the same adjacent-trace pair (the 2x2 correspondence quad). Block membership is decided entirely pre-fit (the module never imports the NURBS fitter, verified by an AST test) and is never re-split based on fit/held-out error. One block per seed -> a region can now materialize multiple independent NURBS patches (one-region-one-patch assumption removed).

Two reported conditions via `scripts/devtools/latent_surface_curve_network_v2_replay.py`: A. ALL_VISIBLE_EVIDENCE_CONSTRUCTION (full region evidence, measures production capability) and B. HELD_OUT_VALIDATION (Worklog 87's checkerboard `_holdout` split, Worklog 95's support thresholds NOT loosened for sparse train).

Real 7-region replay (baseline_compatible checkpoints 2900, final):
- Construction: regions with usable seed 5/7->7/7 and 6/7 (up from Worklog 95); regions with valid curve network 4/7->5/7 and 6/7; 1 region per checkpoint that was blocked purely by the legacy boundary gate now constructs via interior fallback; 3 and 2 multi-patch regions respectively.
- Held-out (evidence-weighted, same convention as Worklog 87/89/94/95): unresolved dropped substantially at 2900 (42.60%->20.31%; final ~flat 32.51%->33.67%). **valid_supported did NOT improve over Worklog 95** (2900: 2.64%->2.72%, essentially flat; final: 11.95%->9.26%, slightly worse). **extrapolative+unsafe combined INCREASED at both checkpoints** (2900: 54.8%->77.0%; final: 55.6%->57.1%) -- newly-constructing regions mostly land as extrapolative, not valid_supported.

Decision: conditions 1-3 (gate-independence, continuous support, multi-patch structure) are met. Condition 4 (valid_supported materially above raw-center baseline) holds vs. the raw baseline (54-180x) but not vs. Worklog 95. **Condition 5 (extrapolative/unsafe must not merely increase from more aggressive construction) FAILS at both checkpoints.** Per the directive's own decision tree: curve-network construction is now broadly available, but safe NURBS did not improve -- STOP adding new curve-seeding heuristics. The bottleneck has moved to the downstream parametric fitting/patch representation (PCA-UV 6x6 NURBS cannot safely digest latent-surface curve-network samples), not curve construction. Do not nominate this constructor as the production architecture; do not follow with another isolated seed-rule/threshold diagnostic. See [[project_latent_surface_curve_network_constructor_prototype]] for the Worklog 95 baseline this extends.
