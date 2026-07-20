# Boundary-First Refactor Pass + Unified Benchmark CLI

Date: 2026-07-20
Status: complete.
Governing document: `OSN_GS_Final_Boundary_First_NURBS_Direction.md`.

Two follow-up requests after Phase 4 landed (`docs/worklogs/31_phase4_boundary_conforming_chart.md`):
1. a refactoring pass across the Phase 1-4 modules to remove any forced-overwrite or temporary/ad-hoc code;
2. fold the Phase 1-4 pipeline into the actual `osn-gs benchmark` CLI instead of leaving it in separate per-phase scripts.

## 1. Refactoring pass

Re-audited every Phase 1-4 module (`torch_surface_components.py`, `torch_component_boundary.py`, `torch_trimmed_component_fitter.py`, `torch_chart_topology.py`, `torch_annulus_chart.py`) plus the benchmark report scripts for forced overwrites, dead code, and stale comments. `torch_annulus_chart.py`'s hard-C0 remnants were already fully removed during the Phase 4 review (see worklog 31); no other module had any.

The one real smell found: `phase4_report.py` imported private (`_`-prefixed) symbols (`_PseudoState`, `_PseudoModel`, `_uv_support_payload`) directly out of `phase3_report.py`. Extracted these into a new shared `nurbs_constructor_benchmark/benchmark_common.py` (public names `PseudoState`/`PseudoModel`/`uv_support_payload`) and updated both scripts to import from there instead of reaching into each other's internals.

Also reviewed the two files flagged as modified-but-unreviewed at session start (`OSN_GS_Final_Boundary_First_NURBS_Direction.md`, `docs/README.md`): both changes are purely additive documentation (a governing-adaptive-voxel-contract section and a legacy-retirement end-state section), not code.

Verified behavior-preserving: full test suite 86/86 passing, and `phase3_report`/`phase4_report` re-runs on `planar_hole` reproduced byte-identical numbers to the ones recorded in worklog 31 (chamfer=0.0058, false_fill=0.167, seam gap=0.012/0.064).

## 2. Unified benchmark CLI

Discovered (via user report, screenshot of a stale rectangle+trim NURBS render) that `osn-gs benchmark` — the actual command the user runs — never reflected any of Phase 1-4: `nurbs_constructor_benchmark/runner.py`'s `--constructor` only ever supported `legacy`/`voxel_patch_stage1` via `TorchOSNGSPipeline`. Phase 1-4 only existed as three separate scripts (`component_report.py`, `phase3_report.py`, `phase4_report.py`) invoked directly by module path, invisible to `osn-gs benchmark`.

Fixed by adding `boundary_first` as a third `--constructor` choice on `runner.py` itself:

- New `nurbs_constructor_benchmark/boundary_first.py`: `construct_boundary_first()` runs Stage 1 hierarchy -> Phase 1 components -> Phase 2 boundary extraction -> Phase 4 topology-routed chart generation (which already falls back to Phase 3's trimmed-rectangle baseline for every non-annulus topology, so there is no separate Phase 3 codepath to run alongside it) and returns a duck-typed `BoundaryFirstState` (`model.get_xyz`/`.cluster_ids`/`.surface_uv`, `surface_patches`, no single combined `surface`).
- `runner.py`'s `evaluate_scene` was split into construction (`TorchOSNGSPipeline.initialize()`) and a generalized `score_state(scene, state, construction_seconds, export_dir)` scoring body. `evaluate_scene_boundary_first()` calls `construct_boundary_first()` then the SAME `score_state`, so all three constructors are scored identically and land in one `report.json` with directly comparable fields (plus an extra `"boundary_first"` key: `component_count`/`per_component` topology+seam diagnostics).
- `AnnulusChartSlice` gained a `uv` field (the per-Gaussian post-fit UV each slice already computed internally but didn't expose) so `score_state`'s per-point residual scoring works for annulus patches the same way it does for trimmed-rectangle patches.
- `state.surface` (the single coarse fallback surface legacy/Stage 1 use to score unassigned points) doesn't exist for boundary-first; `score_state` now scores any unassigned point as zero residual instead of crashing, documented inline as a deliberate scoring convention (unassigned fraction is already reported separately).
- New `write_point_cloud_ply()` in `boundary_first.py`: a minimal renderer-compatible Gaussian PLY (positions + colors from the scene, fixed placeholder opacity/scale/rotation) since boundary-first doesn't run the trainer's covariance/opacity init step. Renderer export lands in the same `<output>/NURBS_output/<scene>/{point_cloud.ply,nurbs_surface.json}` convention as `legacy`/`voxel_patch_stage1`.
- New `--bf-*` CLI flags on `runner.py`: `--bf-normal-threshold-degrees`, `--bf-offset-threshold-ratio`, `--bf-boundary-resolution`, `--bf-density-threshold` (default 3.0, per the Phase 3 sweep), `--bf-coarse-gap-closing-cells`, `--bf-annulus-segments`.
- The three old per-phase scripts were deleted.

Deliberately does **not** touch `osn_gs/core/torch_pipeline.py` or the trainer: boundary-first construction logic stays entirely inside `osn_gs/surface/*` plus this benchmark-side orchestration module. Only the benchmark CLI *surface* was unified (one command, one report schema); the trainer's own default constructor and lifecycle are untouched, consistent with the plan's §10.1 gate (wiring into the trainer requires explicit approval after the final phase).

### Verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole --output <dir>
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor legacy --scenes plane --output <dir>
```

- Full suite: 86 passed.
- `boundary_first` on `planar_hole`: chamfer_rms=0.005800, false_fill=0.167, mean/max seam gap=0.01227/0.06433, jacobian_fold=0 — identical to the pre-consolidation Phase 4 numbers (worklog 31).
- Renderer export verified on disk (`point_cloud.ply` with 600 vertices, valid PLY header; `nurbs_surface.json` with 8 patches).
- `legacy` on `plane` re-run unaffected (chamfer_rms=0.028743, matching prior baseline runs) — confirms the `evaluate_scene`/`TorchOSNGSPipeline` path through the refactored `score_state` split is unchanged.

## Lesson

Any new construction phase must be wired into the existing `osn-gs benchmark` CLI (`nurbs_constructor_benchmark/runner.py`, a new `--constructor` choice) as part of that phase's own completion, not left as a separate standalone script. A script only reachable by someone who already knows its exact module path is effectively invisible to the user and misleads them into thinking the latest work isn't reflected in "the benchmark." Recorded as a standing rule in project memory (`feedback_benchmark_cli_unification`).
