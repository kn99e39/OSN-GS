---
name: project_holey_chart_fitting_coupling_attribution
description: "OSN-GS worklog 117 -- attribution batch testing whether WL113's hole/residual correlation is real fitting-coupling (B2) or a scale/complexity proxy; MIXED result, giant charts show the opposite direction"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-26T04:01:18.491Z
---

Worklog 117 (branch `arch/2dgs-coverage-first-surface`): bounded attribution batch (no new chart mechanism) testing whether WL113's "holed charts have 2.6x worse residual" finding is genuine fitting-coupling failure (B2) or a scale/complexity confound. Froze WL107/109 topology, WL112 one-blob-one-chart baseline, fixed 8x4 (control only).

**Corrected [[project_visible_nurbs_representation_contract_recovery_audit]] (WL116)**: verified via git history that `TorchOSNGSPipeline._fit_surface_patches`/`_target_resolution` (density/boundary-fraction-driven adaptive capacity allocation) DID exist historically (commit before `13d9f61`), but was removed by that exact integration commit and does not exist in the current working tree at all -- corrected classification: historical precedent existed (now deleted), current renderer-native capacity policy remains unresolved.

**B1 (materialization) proven empirically**: applied `TorchOSNGSPipeline._uv_occupancy_mask` (existing config defaults, resolution=24, dilation=1) to all 14,900 WL112-identical fitted charts; verified via `torch.equal` that control_grid and evaluate() output are bitwise unchanged in every single case (0/14,900 violations) -- not just inferred from code reading, actually measured.

**B2 (fitting coupling) -- MIXED result, not the crude auto-decision**: within-chart residual-vs-distance-to-unsupported correlation is weak but real (median -0.055, 66% of charts negative; hole-specific distance -0.098, 68% negative). BUT scale-stratified (pixel-count and representative-count quantile) matched analysis shows the raw ~2.6-3x ratio is NOT stable: middle strata show artificially inflated ratios (14x, 2874x, 489x) driven by near-zero-residual unholed charts in the denominator, while the LARGEST-scale strata show the ratio shrink to 1.3x or invert below 1.0 (unholed actually worse) for the largest representative-count stratum.

**Key finding**: the top-10 giant patio charts (component 0, ~211k-217k pixels each, all with exactly 1 hole, the same charts that drove WL112/113's residual_max=1517 story) show residual HIGHER far from the unsupported/hole boundary in 8 of 10 cases (up to 5.4x worse far vs near) -- the OPPOSITE of what B2 predicts. This is strong evidence that the highest-impact real-scene failure is NOT dominated by hole-proximity/fitting-coupling, but by something scale/capacity/parameterization-related instead.

**Synthetic full-vs-hole controls** (unmodified fitter, planar and curved fixtures): planar shows no effect at all (uninformative negative control, zero curvature is trivially representable regardless of missing data). Curved shows no median effect but a real tail-quality degradation (p95 +39%, max +50%) when the hole is present.

**Final decision: MIXED/INCONCLUSIVE**, deliberately overriding the script's own crude binary auto-decision (`B2_NOT_SUPPORTED`) after inspecting signal magnitudes directly -- weak-but-real B2 signal exists in the general/typical chart population and in curved-surface tail quality, but does NOT hold (and often reverses) for the highest-impact giant charts, where scale/capacity is the better explanation. Coupled multi-patch fitting was NOT implemented this batch (attribution only).

15 new focused tests (mask hole/edge decomposition, distance accounting, within-chart statistics, synthetic geometry contracts, and a bitwise proof that support-mask assignment never alters an already-computed fit), all passing. No production code touched. Real-scene script `scripts/devtools/holey_chart_fitting_coupling_attribution.py`, report at `output/117_osn_gs_holey_chart_attribution/holey_chart_fitting_coupling_attribution_report.json` (all 161 views, 14,900 charts matching WL112/113 exactly), 7 named review exports.
