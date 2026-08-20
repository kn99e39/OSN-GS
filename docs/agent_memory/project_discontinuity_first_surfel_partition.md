---
name: project-discontinuity-first-surfel-partition
description: "Worklog 98 replaced WL97's normal-concentration gate with a local shape-operator smooth-surface-residual test; works on synthetic fixtures but FAILED on the real scene (94.51% giant subset, worse than WL96/97); no architecture decision"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-20T06:58:54.853Z
---

Worklog 98 (2026-08-20, `arch/2dgs-coverage-first-surface`): the user pointed out a real problem with [[project_region_coherent_surfel_partition]] (Worklog 97) visible on the real scene render -- the curved side of a table fragments into several subsets purely because its normal rotates, even though a smoothly curved surface is exactly what a valid NURBS patch is supposed to represent. Normal concentration alone can't distinguish "smooth curvature" from "a real discontinuity".

Fix attempted: replace the union rule again with `osn_gs/surface/torch_discontinuity_first_surfel_partition.py`. For each surfel, fit a local 2x2 shape operator `S_i` (`Delta n ≈ -S Delta x_T`) via batched least squares in the surfel's own tangent-plane basis (intrinsic `tangent_axis_u/v`, no eigendecomposition). Cut a candidate edge if EITHER (A) the observed normal change disagrees with BOTH directions' shape-operator prediction (residual > median + 3*MAD, robust data-derived fence) OR (B) the displacement's normal-direction component exceeds its own tangential component (ratio > 1.0, self-normalizing parity test, no spacing reference). Final subsets = plain connected components of the surviving graph (structurally simpler than WL97 -- no region-growth/ownership-propagation machinery needed, since cutting an edge never removes a node).

Two real bugs found and fixed during implementation (both caught by fixture tests, not by inspection):
1. First tried MAX of the two directional residuals -- caused severe over-fragmentation near a synthetic crease (53 subsets instead of ~2) because a node whose kNN neighbourhood straddles the crease gets a contaminated shape-operator fit, which then makes ALL its edges look suspect. Switched to MIN (cut only if BOTH directions agree) -- reduced same-side false positives from 111/281 to 20/281, and the two dominant final subsets recovered 92% of the fixture.
2. The parallel-sheet criterion originally reused the SAME `spatial_connect_spacing_multiplier` (2.0) already used to build the candidate graph -- discovered via a fixture test to be structurally degenerate (the normal-offset component can never exceed the total displacement, which the candidate gate already bounds at that same 2.0x). Replaced with a self-normalizing normal-vs-tangential-offset ratio (threshold 1.0, no external scale reference).

**Synthetic fixture results: all correct.** Flat sheet stays one subset; a 180-degree-rotating cylindrical band stays ONE subset with zero cuts (the central design goal -- large normal gradient with small residual is not a boundary); a 90-degree crease is cut cleanly (two dominant subsets, 92% coverage); two nearby parallel sheets with identical normals correctly separate.

**Real-scene measurement: NEGATIVE result.** On the Worklog 96/97 checkpoint (retrained after the original went missing from disk -- see below -- reproduced closely: 1,190,469 surfels, held-out PSNR 28.226 vs original 28.256), only 18.15% of spatial edges were cut (81.85% survived), and that was nowhere near enough to break percolation in the dense k=8 kNN graph: the largest subset is **94.51%** of the scene -- WORSE than both Worklog 96 (74.70%) and Worklog 97 (20.84% on this same retrained checkpoint). Per-edge local classification, however individually correct, does not guarantee the graph is actually topologically disconnected -- WL97's region-LEVEL global-state re-check at every merge is apparently what actually prevented percolation, at the cost of over-fragmenting curved surfaces. Visually: the table no longer fragments by leg (the original motivating problem is gone), but it also no longer separates from the floor/hedge (it's part of the same giant blob).

**Checkpoint note**: the original Worklog 96/97 30k checkpoint (`2dgs_run1/30000`) went missing from disk (likely lost during the user's `output/confirmed/` reorganization) partway through this session. Asked the user; they chose to retrain with the identical config -- confirmed reproducible.

Review export: `output/osn_gs_discontinuity_first_surfel_partition/` (7 views: ORIGINAL_2DGS_SCENE, RAW_INTRINSIC_NORMAL, NORMAL_GRADIENT_MAGNITUDE, SMOOTH_SURFACE_MODEL_RESIDUAL, DETECTED_DISCONTINUITY_BOUNDARY, WL97_REGION_CONCENTRATION_PARTITION, DISCONTINUITY_FIRST_PARTITION).

16 new focused tests, all pass. **No architecture decision made — this result is reported honestly as negative.** Likely next direction (not decided): combine WL97's region-level global-state anti-chaining with this batch's curvature-aware residual (replace WL97's raw-concentration test with the shape-operator residual) rather than relying on per-edge cuts alone. See [[project_region_coherent_surfel_partition]] and [[project_2dgs_coverage_first_surfel_partition]] for the preceding stages.
