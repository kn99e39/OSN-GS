# 29. Phase 2 Component-Level Boundary Extraction

Date: 2026-07-20

## Work performed

- Audited the Phase 1 component builder and the in-progress component-level boundary extractor against the Final Boundary governing plan.
- Ran the required six-scene Phase 1 benchmark and the Phase 2 boundary benchmark.
- Calibrated the default significant-loop threshold at 64x64 support resolution from 4 to 20 cells. This does not fill or delete a mask region: it only classifies a loop as a tiny diagnostic artifact rather than a topological hole.

## Result

- Phase 1 recovers the expected component count and assignment ARI of 1.0 on plane, sine, planar_hole, crease, and close_parallel_sheets. density_gradient retains one component with 3.7% inactive/unassigned samples.
- Phase 2 preserves the planar_hole significant hole (262 cells) while classifying the density_gradient 17-cell gap as one tiny artifact, leaving its significant hole count at zero.
- Added a regression test for that distinction.

## Evaluation

The Phase 2 result is benchmark-only and does not change legacy or voxel_patch_stage1 constructor behavior, trainer behavior, or ADC. The artifact threshold remains visible in diagnostics and configurable through the component-report CLI; it is not a morphology operation or a forced hole-count rule.

## Remaining risks

- The 20-cell default is calibrated for the 64x64 benchmark mask. It must be scaled or revalidated when the support resolution changes materially.
- density_gradient still has 3.7% samples in inactive leaves, so its support coverage is not a complete real-data eligibility solution.
- This report does not approve Phase 3. The governing plan requires user approval after Phase 2 results are reviewed.
