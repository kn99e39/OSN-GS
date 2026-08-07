---
name: project_region_owned_full_evidence_patch_support
description: "worklog 67 - recovered full-cloud evidence per region for NURBS fitting (additive-only); eliminated under_supported entirely, revealed a new fitting-resolution finding instead"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T05:07:40.627Z
---

worklog 67: without changing representative topology or chart boundary, recovered each approved region's real full-cloud Gaussian evidence for NURBS fitting/fidelity validation only (new `osn_gs/surface/torch_region_owned_full_evidence.py` + additive wiring into `TorchOSNGSPipeline._construct_canonical_with_full_evidence` via a new `region_owned_full_evidence_fits` field). Reuses the EXISTING production gate `_propagate_with_evidence_gating` (worklog 129, normal-alignment/residual thresholds) as the single source of truth for evidence ownership -- never reimplemented, so it structurally can't merge cross-region evidence or bleed across creases/parallel-sheet-conflicts (same gate already used for exactly that).

Result on baseline_compatible@2900/3100 + Graphdeco baseline reference: full-evidence support was 55-333x larger than representative-only support, and `under_supported`/`unsafe_geometry` disappeared completely (22/22 patches reached full_evidence_state="materialized" -- corrected count, worklog 69 fixed an earlier "21" miscount). BUT 95% (21/22) got reclassified `extrapolative` instead — not a fitting failure, but a measurement-scale effect: `local_evidence_scale` now comes from hundreds-to-thousands of dense real Gaussians instead of 3-8 representatives, so the SAME 6x6/degree-2 NURBS fit is judged against a much stricter (denser) yardstick. Graphdeco baseline reference shows the identical pattern (4/4, 2/2 extrapolative) -- confirms this is a fitting-RESOLUTION limitation, not an OSN-GS training-quality problem. Per task instruction, resolution was NOT bumped this round -- pure finding, no reactive fix.

Also corrected worklog 66 terminology: `validate_simple_closed_loop` checks the 2D BOUNDARY LOOP only (renamed field to `boundary_loop_simple_polygon_violation`), NOT full 3D surface self-intersection (never checked anywhere in this pipeline). Documented explicit 5-way classification priority order (unsafe_geometry > duplicate_or_overlapping post-pass > under_supported > extrapolative > valid_supported). Clarified worklog 66's gap-fraction numbers are within-condition only, never a cross-condition ground-truth-coverage comparison.

232 focused tests passed (9 new + 223 related), full pytest NOT run (explicit instruction, applies to worklog 66's prior production change too).

**Why:** answers whether [[project_visible_patch_fidelity_validation]]'s dominant `under_supported` finding was a real evidence-scarcity problem or a representative-only-fitting design limitation -- it was the latter.

**How to apply:** the NEW open problem is fitting resolution (6x6 control grid, degree 2) being too coarse for real dense evidence -- this is the natural next thing to investigate if visible-surface fidelity work continues, separate from the still-open screen-prune-storm problem ([[project_baseline_compatible_3k_production_validation]]).
