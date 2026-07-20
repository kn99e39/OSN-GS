# 30. Phase 3 Trimmed Component Correctness Baseline

Date: 2026-07-20

## Work performed

- Validated the in-progress Phase 3 component-level trimmed NURBS baseline against the Final Boundary governing plan.
- Reused the existing IDW-seeded, regularized LSQ, foot-point-correction fitter without changing legacy or voxel_patch_stage1 construction paths.
- Fixed support-domain evaluation to use the same extent-adaptive trim-aware rasterization as patch-union evaluation. A one-sample-per-output-cell raster fragmented trimmed surfaces and falsely reported thousands of holes.
- Classified support holes consistently with Phase 2: loops of 20 cells or fewer remain visible as tiny diagnostic artifacts but are not reported as significant topology holes.

## Result

At points=600, seed=0:

- plane: one component/patch, support IoU 0.981, significant holes 0, no Jacobian degeneracy.
- sine: one component/patch, support IoU 0.981, significant holes 0, no Jacobian degeneracy.
- planar_hole: one component and one geometry patch, significant hole count 1, zero uncovered support, support IoU 0.916, and no active-active seam.
- crease and close_parallel_sheets retain two components/patches with ARI 1.0.
- All Phase 3 fitter tests and the ground-truth NURBS tests passed (10 tests).

## Evaluation

This is the required correctness baseline: its control grid may span a hole, while the trim mask prevents the rendered/evaluated support from covering it. The implementation is benchmark-only; it does not alter trainer or ADC behavior.

## Remaining risks

- density_gradient still reports one significant support gap, with support IoU 0.772 and uncovered fraction 0.093. This comes from sparse-support calibration and inactive input leaves, not from the planar-hole trim contract.
- The 20-cell significant-hole threshold is calibrated for the default 128x128 support raster and must be revalidated if resolution changes.
- Phase 4 boundary-conforming charts remain a separate gated phase.
