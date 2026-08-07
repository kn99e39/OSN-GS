---
name: project_region_owned_full_evidence_boundary_materialization
description: "worklog 70 - built dense per-edge boundary materialization from region-owned full evidence (world-3D, binned along edge tangent, never a hull); measurably improved containment (91.1%->54.8% interior_outside_boundary) but 0/22 patches reach valid_supported/extrapolative - 11/22 fail-closed at loop construction, 11/22 still fail UV validity gate"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T06:11:15.964Z
---

worklog 70: worklog 69's 22/22 `partition_materialization_required` was approved as a diagnosis but NOT adopted as canonical. Built dense boundary materialization from region-owned full evidence (worklog 67, unchanged) on top of the existing representative boundary (worklog 61's parametric chart boundary, unchanged) -- [[project_single_chart_parameterization_validity]].

New `osn_gs/surface/torch_region_owned_boundary_materialization.py`: per boundary EDGE (never a global hull/rectangle), assign evidence to nearest edge via 3D point-to-segment distance, keep only points farther from the loop's world centroid than both endpoints by more than `local_evidence_scale` (worklog 32's per-representative `mean_spacing`, reused not invented), bin the edge's world tangent into `local_evidence_scale`-wide bins and keep only the farthest-from-centroid point per bin (guarantees monotonic non-backtracking insertion order), splice in with inherited edge provenance (crease/observation_frontier/partition_seam/physical_termination -- never mixed), fail-closed via unmodified `validate_simple_closed_loop` if the result isn't a simple loop.

Design history within this round (measured then rejected, not tuned to a preferred outcome): (1) single-point-per-edge -- too sparse, `interior_outside_boundary` stayed >10% in all 12 that materialized; (2) all-qualifying-points-per-edge inserted raw in tangent order (no binning) -- real evidence noise makes 18/22 self-intersect. Also found+fixed a design flaw in an early PCA-UV-frame version: projecting boundary+evidence into a shared `pca_parameterize_points` frame is unstable because adding outlier evidence rotates the frame and silently changes which edge a point is nearest to (caught via test-fixture debugging) -- final version does everything in world 3D space, no PCA in this module at all.

Real 22-patch result (baseline_compatible@2900/3100 + baseline reference): `full_evidence_boundary_materialization_required` 11/22 (dense extension itself fails `validate_simple_closed_loop` -- 7 proper_self_intersection, 4 nonplanar-not-checked, 6 orientation_inconsistency), `partition_materialization_required` 11/22 (dense boundary materializes but `interior_outside_boundary` still >10% in 11/11 -- 100%), `valid_supported`/`extrapolative` 0/22. Before/after: boundary vertex count 3.27->8.73, `interior_outside_boundary` 91.1%->54.8% (real improvement, structurally confirms worklog 69), but held-out fitting error did NOT improve (23.87->28.86 dense-NN normalized p95). Sharper diagnosis than worklog 69: the limiting factor isn't just "boundary too small" but that the original topology has only 3-4 edges/wedges, so no amount of per-edge densification traces the evidence's true outer shape -- the wedge partition itself is too coarse.

`parallel_sheet_suspected`/raw-evidence `triangle_fold_fraction` recorded diagnostic-only this round (not gates), per instruction. `surface_self_intersection` still `"not_checked"` everywhere.

New `scripts/devtools/region_owned_full_evidence_boundary_materialization.py`. New `tests/test_region_owned_boundary_materialization.py` (7 tests, all pass). Focused pytest only, no full pytest per standing instruction.

**Why:** answers whether densifying the existing boundary (rather than repartitioning) can fix worklog 69's containment failure -- partially yes (measurable improvement) but not sufficient alone; the real limiting factor is the coarse original edge/wedge topology itself, a new, more specific open problem than worklog 69's.

**How to apply:** if a future round redesigns the wedge/edge topology itself (not just densifying within existing wedges) to trace evidence shape, it must still keep region formation/representative topology invariant per the standing constraint reaffirmed every round since worklog 67.
