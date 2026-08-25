---
name: project_visible_nurbs_representation_contract_recovery_audit
description: "OSN-GS worklog 116 -- pure audit finding the existing NURBS fitter already handles rank-deficient data via Tikhonov anchoring, and that unused uv_support_mask (trimming) and fit_coupled_patch_graph_lsq (multi-patch) mechanisms already exist; reclassifies WL113/114 accordingly"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-25T09:32:12.646Z
---

Worklog 116 (branch `arch/2dgs-coverage-first-surface`): pure diagnostic audit (no code changes) recovering the actual `osn_gs/surface/torch_nurbs.py` contract, following [[project_local_rank_complete_chart_network]] (WL114) and [[project_design_intent_specification_implementation_traceability_audit]] (WL115). Explicitly retires WL114's closure rule ("full column rank == valid chart boundary") without generalizing WL114's negative result to "local NURBS is not viable."

**Key finding 1**: `_solve_control_grid_lsq` (`torch_nurbs.py:730-769`) always adds `tikhonov_lambda * eye(n)` (anchored to the IDW seed, not zero) to the normal-equations system -- this makes the regularized system strictly positive-definite and solvable **regardless of the data matrix's rank**, including the degenerate zero-rank case. Full column rank was never a mathematical requirement of the fitter; WL111-114 imposed it as an external gate before even calling the fitter.

**Key finding 2**: `TorchNURBSSurface.uv_support_mask` (a UV-trimming field) already exists and is live in the OLDER boundary-first pipeline (`osn_gs/core/torch_pipeline.py::_assign_uv_support_masks`, `osn_gs/surface/torch_trimmed_component_fitter.py`), but has never been used anywhere in the WL107-114 renderer-native lineage. Precisely distinguishes materialization (A/B1, solved -- mask is assigned strictly AFTER fitting, purely restricts what's drawn/measured) from fitting coupling (B2, NOT solved -- the fit itself always uses the full [0,1]^2 tensor-product basis; a control point near a mask boundary is still influenced by data on both sides of a hole). `torch_trimmed_component_fitter.py`'s own docstring already states this exact gap ("topology is carried entirely by the trim mask, not by control-grid structure... a correctness baseline, not the final architecture") -- the codebase's own authors flagged this years before WL113 independently rediscovered it.

**Key finding 3**: genuine multi-patch coupled fitting already exists (`fit_coupled_patch_graph_lsq`, `torch_nurbs.py:877-1044`) -- arbitrary patch graphs with `SharedBoundaryConstraint`s, shared control points union-find-merged into one joint unknown BEFORE the first solve (not independently fit then averaged). `fit_coupled_wedge_ring_lsq`'s docstring explicitly distinguishes "representation seam" from "physical boundary" and notes full tangent continuity was deliberately deferred as a separate, never-implemented step. Built for the old annulus/boundary-first lineage, never wired to WL107-114.

**Key finding 4**: fixed 8x4 was never an architectural law -- six different resolution defaults (6, 6x6, 7, 8, 8x4, 12x12) already coexist across the codebase's various NURBS call sites, none data-derived. WL111 froze 8x4 only because it's `fit_torch_visible_surface_lsq`'s bare function-signature default.

**Reclassified WL113 A/B/C/D**: A's "32-sample requirement" and B's "rectangular domain failure" are now understood as artifacts of WL111's own external gating/parameterization choices, not fitter requirements -- B specifically traces to the unused `uv_support_mask` (B1) vs the still-unresolved B2 coupling gap. C remains a narrow, deliberately-frozen control-experiment limitation. D unchanged (renderer/data phenomenon).

**Reinterpreted WL114**: separated "LOCALITY IS USEFUL" (residual ~9x/domain-shape materially better, evidence-supported) from "RANK-CLOSED DISJOINT EXTRACTION IS NOT VIABLE" (coverage/patch-count/overlap costs, evidence-supported) -- these are different claims and only both together were previously conflated into one verdict. Critically: WL114's overlap-normal degradation is NOT attributed to "more seams" (the directive explicitly forbade that unproven causal leap) but to the narrower, better-supported claim that 15.9x more charts were fit completely independently, with zero use of the already-existing `fit_coupled_patch_graph_lsq` shared-boundary mechanism.

**The one unresolved question flagged to drive the next batch**: does applying the existing (unused) `uv_support_mask` to WL112's current one-blob-one-chart baseline already resolve enough of failure B to avoid a new chart-unit redesign, or does the B2 fitting-coupling gap mean coupled multi-patch fitting (`fit_coupled_patch_graph_lsq`) is the necessary next mechanism instead? No implementation, no new mechanism, no tests needed (code was unambiguous) this batch.
