# Phase 4 — Boundary-Conforming Chart Generator

Date: 2026-07-20
Status: implemented and verified. Do not start Phase 5 without explicit user approval.
Governing document: `OSN_GS_Final_Boundary_First_NURBS_Direction.md`, Phase 4.

## Scope completed

- Topology routing (`disk_like`, `annulus`, `multi_hole`, `complex`, `non_chartable`) from Phase 2's own outer/hole loop counts, `osn_gs/surface/torch_chart_topology.py`.
- A hole-area significance filter (`MIN_HOLE_AREA_FRACTION = 0.02`): a "hole" under 2% of the outer loop's own area is treated as `disk_like`, not `annulus`. Without this, `density_gradient`'s sparse-sampling region (a 17-cell density-threshold artifact against a 2468-cell outer loop, 0.7%) was misrouted into an 8-slice O-grid and collapsed (mean seam gap 0.267, vs. planar_hole's real hole at 0.005).
- An annulus O-grid generator (`osn_gs/surface/torch_annulus_chart.py`): one component with one outer loop and one significant hole is represented by N (default 8) radial NURBS wedges. Each wedge reuses the existing IDW/LSQ/foot-point fitter UNMODIFIED, seeded with a Coons-style polar-local UV (`local_s`/`local_t`) whose radius bounds are shared at each slice-boundary angle — this shared seed is what actually drives continuity (see "Findings" below). Non-annulus topologies fall back to Phase 3's trimmed-component path unchanged (the plan's own "safe fallback").
- JSON chart provenance: per-slice bounds, control grids, seam diagnostics, topology checks, `boundary_anchor_max_error`, and U/V iso-line polylines evaluated from the actual fitted NURBS patches (`v=0`/`v=1` = inner/outer boundary families, `u=0`/`u=1` = radial connectors).
- `nurbs_constructor_benchmark/metrics.py`: fixed a pre-existing (not Phase-4-specific) artifact in `support_domain_metrics` — it sampled the generated surface at one UV point per output cell (`sample_generated_surface`/`_rasterize_xy`), which fragments any non-trivially-shaped trimmed chart into thousands of spurious "holes" (verified: legacy's own `plane` scene showed `support_generated_hole_count` in the thousands before this fix, an artifact unrelated to real topology). Now reuses `patch_union_metrics`'s adaptive-density rasterization (`_patch_xy_mask`) instead.

## Finding: a "hard C0" attempt regressed accuracy and was reverted

An intermediate version of `build_annulus_chart` forced exact C0 continuity by overwriting each wedge's boundary control-point columns (`control_grid[:, 0]`/`[:, -1]`, and the `u=0`/`u=1` radial edges) with values shared identically between adjacent slices, computed after the free LSQ fit. This measurably worked as intended for continuity (seam gap ~1e-7, i.e. machine precision) but regressed the primary accuracy criteria on `planar_hole` versus the free-fit baseline:

| variant | chamfer_rms | GT-compared false-fill | mean seam gap |
|---|---|---|---|
| free fit (Coons-seeded only, no hard constraint) | **0.0058** | **0.167–0.180** | 0.005–0.012 |
| hard constraint, 2-point chord boundary | 0.0061 | 0.200 | ~0 |
| hard constraint, loop-sampled (raster-noisy) boundary | 0.0095 | 0.311 | ~0 |
| Phase 3 baseline (rectangle + trim) | 0.0080 | 0.200 | n/a (1 patch) |

Forcing the boundary pulls each wedge's fit away from what the data itself supports; a version that additionally tried to source the forced boundary from Phase 2's raster loop points (rather than a 2-point chord) made this *worse*, not better, because Phase 2's loop points are raster-cell centers with their own staircase quantization noise. Neither hard-constrained variant beat the free fit on the metrics that matter for this gate. Per the plan's own §4.5 ("초기: shared boundary C0... 후속: G1/C1"), the shipped implementation stays a **free fit with a shared Coons-style UV seed** and *measures* (does not force) the resulting seam; the dead `_loop_points_at_angles` helper and the now-unused `outer_boundary_world_points` parameter from the hard-constraint attempt were removed.

## Verification (current, re-run after the above correction)

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --output C:\tmp\osn_gs_phase4_final
```

- Full suite: 86 passed.
- `planar_hole`: classified `annulus`; 8 wedges; Jacobian fold count 0; mean seam gap 0.012, max 0.064 (small relative to the 0.32–0.9 domain scale); support conformality 1.000 (untrimmed by construction — nothing to trim, unlike Phase 3's rectangle).
- vs. Phase 3 baseline: chamfer RMS 0.0058 vs 0.0080; GT-compared false-fill (`patch_union_metrics`, the reliable metric in this project — see `support_domain_metrics`' fix above) 0.167 vs 0.200; both **improved**, not merely maintained.
- `union_hole_count=2` (gt 1): the 2nd is a 1-cell seam-speckle artifact (`union_tiny_false_hole_count=1`), already correctly separated by the existing tiny-hole diagnostics from the real 1113-cell hole that matches GT.
- Inner/outer iso-lines (`v=0`/`v=1`) sampled directly from the fitted NURBS trace radius 0.292–0.306 (true hole radius 0.32) and 0.880–0.924 (true outer radius 0.9) — boundary-conforming, verified numerically not just architecturally.
- `plane`, `sine`, `crease`, `close_parallel_sheets`, `density_gradient` all route to the Phase 3 trimmed fallback with numbers identical to Phase 3's own report (confirms the fallback path is untouched).
- Legacy/Stage 1/Stage 1-F production files (`torch_pipeline.py`, `torch_voxel_hierarchy.py`, `torch_boundary_refinement.py`) show zero diff against the committed baseline (`71a4ae0`).

## Guardrails and remaining work

- Benchmark-only. Does not change the trainer default or remove legacy/Stage 1 comparison paths.
- Deterministic seam placement uses angle zero (no distinguished low-curvature location on these rotationally-uniform synthetic scenes). Curvature/confidence-directed seam placement and true G1/C1 continuity remain future refinement work.
- The hole-area significance filter's 2% threshold is set from one labeled example (planar_hole's real hole at 8.2% vs. density_gradient's artifact at 0.7%); revisit if a third labeled case narrows the margin.

## Gate

Phase 4 meets its O-grid, topology routing (with hole-significance filtering), Jacobian, seam-measurement, iso-line, and trimmed-baseline-comparison checks, with the free-fit-vs-hard-constraint trade-off resolved in favor of the metrics that matter for this gate. Per the governing plan, do not start Phase 5 until the user explicitly approves it.

## Update, 2026-07-20: consolidated into the unified `osn-gs benchmark`

The three separate per-phase scripts (`component_report.py`, `phase3_report.py`,
`phase4_report.py`) have been removed. The boundary-first pipeline (Phase 1-4,
Phase 4's topology router already falling back to Phase 3 automatically) is now
a third `--constructor` choice on the SAME unified benchmark entry point used
by `legacy`/`voxel_patch_stage1`:

```powershell
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole --output <dir>
```

New module `nurbs_constructor_benchmark/boundary_first.py` builds a duck-typed
`BoundaryFirstState` and is scored by the exact same `score_state` body
(`nurbs_constructor_benchmark/runner.py`) as every other constructor, so all
three land in one `report.json` with directly comparable fields, plus an
extra `"boundary_first"` key with per-component topology/seam diagnostics.
Renderer export uses the same `<output>/NURBS_output/<scene>/` convention.
Verified byte-for-byte same numbers as before consolidation (planar_hole:
chamfer=0.0058, false_fill=0.167, seam gap=0.012/0.064). Does not touch
`osn_gs/core/torch_pipeline.py` or the trainer -- boundary-first construction
stays fully inside `osn_gs/surface/*` + this benchmark-side orchestration.
