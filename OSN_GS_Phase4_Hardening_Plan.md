# Phase 4 Hardening Plan: seam continuity, iso-line quality, Jacobian gating, boundary conformance

Status: **Steps 1-3 complete. Step 4 in progress: first candidate (arc-length-outer segment placement) tested and REJECTED (see below) — trying the next candidate.** 99/99 tests passing. See `docs/worklogs/39_phase4_hardening_step1_3.md` for the Step 1-3 writeup.
Governing document: `OSN_GS_Final_Boundary_First_NURBS_Direction.md` (this is a hardening pass on the already-approved Phase 4, not a new phase number; Phase 5 stays gated on explicit user approval per that document's §14, independent of this plan).

## Context

The rendered Phase-4 annulus O-grid output (screenshot, 2026-07-21) showed the expected 8-wedge structure but with a visible crack/seam on one side and uneven radial iso-line spacing. Working from a detailed external review, the user identified this as squarely Phase 4's own scope (not Phase 5's — Phase 5 only *uses* the UV field, it doesn't fix a broken one). A first draft of this plan was corrected by the user on several metric definitions and its sequencing before being approved; this document reflects the corrected, approved version.

Phase 4 was previously marked complete/approved (worklog 31/36) against accuracy/false-fill/seam-*position* gates only. This is a hardening pass on that same phase, reopened with explicit user go-ahead, following the project's established discipline: **measure first with real numbers on multiple scenes, validate that new metrics actually detect injected failures, only change behavior where a fix is proven not to regress chamfer/false-fill, never assume a good seed alone guarantees a downstream property, and never force a hard constraint that looks clean but hurts accuracy** (the hard-C0 attempt was reverted for exactly this reason — `osn_gs/surface/torch_annulus_chart.py`, see the "NOT hard-enforced" comment block in `build_annulus_chart`).

Grounding from code (verified, not assumed, as of Step 1):
- Seam metric before Step 1 was position-only; the seam loop already wraps `(k+1) % n` so periodic (last-to-first) closure was already sampled.
- Jacobian check before Step 1 was `‖Su×Sv‖`-based only (a norm, so it can never go negative — the old "fold count" was actually measuring near-degeneracy, not orientation reversal).
- `local_s` (tangential) increases with global angle for every slice by construction, so `Su = ∂S/∂local_s` points in the same physical rotational direction on both sides of every seam — an invariant of this specific O-grid construction, documented in the module docstring and exercised (not just asserted) by the existing test suite.
- Phase 2 (`osn_gs/surface/torch_component_boundary.py`) exposes real boundary geometry beyond Phase 4's own coarse radius bins: `LoopDescriptor.boundary_world_points` per loop (explicit outer/hole loop identity) and a finer marching-squares `contour_world`. This is itself an *estimated* support boundary (density threshold + marching squares), not ground truth — never call it "true boundary" in code or docs; it is the Phase-2 observed-support boundary.
- `fit_torch_visible_surface_lsq` exposes only generic `smoothness_lambda`/`tikhonov_lambda`; nothing seam-aware exists.
- `segments`/`angle_step` in `build_annulus_chart` is always perfectly uniform; no adaptive spacing exists.
- Only one annulus-topology scene existed before Step 3 (`planar_hole`, centered circular hole, flat plane, `nurbs_constructor_benchmark/scenes.py`). No off-center/elliptical/nonuniform-density/curved variant exists yet — a Step 3 task.
- Gate machinery in `nurbs_constructor_benchmark/runner.py` is a flat list of opt-in `--max-*`/`--min-*` CLI thresholds checked against `result["ground_truth"][...]`. New architecture-diagnostic metrics must NOT be stuffed into that `ground_truth` namespace — they get their own (`chart_quality`, added in Step 1).

## Corrected metric definitions (as implemented in Step 1)

### Jacobian / fold-over
Per sample, from `J = [Su Sv] ∈ R^{3x2}`, compute the two singular values `sigma_min <= sigma_max` via the closed-form eigenvalues of `J^T J` (2x2), not just `‖Su×Sv‖`:
- `area_jacobian = ‖Su × Sv‖` = `sigma_min * sigma_max` exactly (local area scale — kept as `min_area_jacobian`, this is what the old `jacobian_min` measured).
- `condition_number = sigma_max / max(sigma_min, eps)` (anisotropic distortion; `sigma_min -> 0` is local collapse/compression).
- `orientation_dot = dot(normalize(Su×Sv), n_ref)` where `n_ref` is a **per-slice, self-propagated** reference normal (seeded from that slice's own central sample, majority-sign-aligned across that slice's own sample grid) — NOT one global fixed vector for the whole component, so a genuinely curved annulus doesn't get spurious flips just from legitimate curvature. This catches orientation reversal/fold-over WITHIN one patch; reversal BETWEEN adjacent patches is a distinct condition, covered separately by the seam normal-angle metric below.
- Aggregated per slice and rolled up per component: `min_area_jacobian`, `min_jacobian_singular_value`, `max_jacobian_condition` (mean/p95/max reported, not one scalar), `orientation_flip_count`, `near_degenerate_count`.
- Implemented in `_jacobian_diagnostics()`, `osn_gs/surface/torch_annulus_chart.py`.

### Seam continuity — two separate metric families, explicit coordinate convention
Convention (documented in the module docstring, `u`=tangential/angular, `v`=radial for every slice): a shared seam is a constant-`u` boundary. For adjacent slices A (`local_s=1` edge) and B (`local_s=0` edge):
- **Along-seam continuity**: position gap (existing `mean_gap`/`max_gap`) + seam-tangent angle between `Sv_A` and `Sv_B` — no sign flip needed, both slices parameterize `v` inner(0)->outer(1) identically.
- **Across-seam continuity**: angle between `Su_A` (at `local_s=1`) and `Su_B` (at `local_s=0`) — both physically oriented the same rotational direction by the stated invariant, no sign flip needed — plus normal angle (`cross(Su,Sv)` both sides) and derivative magnitude ratio.
- Kept as separate fields (`seam_tangent_angle_deg_*`, `seam_cross_derivative_angle_deg_*`, `seam_normal_angle_deg_*`, `seam_derivative_ratio_*`), not collapsed into one number.
- Implemented in `_measure_seams()`, same file.

### Iso-line / parameter quality
Per slice: within-line spacing CV (`cv_v_along_u_line_mean`, `cv_u_along_v_line_mean`), directional stretch (`‖Su‖`,`‖Sv‖` distributions), anisotropy (`min/max` stretch ratio), orthogonality (`|Su·Sv| / (‖Su‖‖Sv‖)`), area distortion (reuses `min_area_jacobian`, not recomputed). Raw CV is diagnostic-only for now (Step 1-3) — a polar O-grid has an *expected* radial contraction near the inner edge that raw CV cannot distinguish from genuine crowding; a detrended version is a candidate refinement, deferred until Step 3 shows it's actually needed. Implemented in `_parameter_quality()`.

### Boundary conformance vs. Phase 2's observed-support boundary
`seed_boundary_anchor_error` (renamed from `boundary_anchor_max_error`) stays as Phase 4's own self-consistency check (fit vs. its own Coons-derived radius bins). NEW, separate `phase2_boundary_conformance` compares against Phase 2's actual loop boundary points, using a **symmetric** distance (`edge_to_reference_*`, `reference_to_edge_*`, `symmetric_chamfer`, `hausdorff`, `boundary_coverage_ratio`) so a chart edge collapsed onto a sub-arc of the true boundary can't hide behind a one-directional metric. Loop association (inner=hole loop, outer=outer loop) is explicit by Phase 2's own loop-kind labeling, never nearest-loop heuristics. Implemented in `_boundary_conformance()`.

### Gate structure
New diagnostics live under `result["boundary_first"]["per_component"][i]["chart_quality"]` (new namespace) — never in `result["ground_truth"]`. `topology_checks` keeps its existing (renamed) coarse fields for the printed summary; `chart_quality` carries the full mean/p95/max detail. No new CLI gates wired yet (deferred to Step 6, after Step 3's multi-scene distribution exists — gate thresholds must not be fit to one scene).

## Sequencing

**Step 1 — Diagnostics only, no behavior change. COMPLETE.** All metrics above implemented in `osn_gs/surface/torch_annulus_chart.py`, threaded through `nurbs_constructor_benchmark/boundary_first.py`'s per-component payload as `chart_quality`, `runner.py`'s printed summary updated for the renamed fields. Full suite 86/86 passing; `planar_hole` chamfer/false_fill unchanged (0.005800/0.167 — confirms no behavior change).

**Step 1 finding, root-caused:** on `planar_hole` (8 segments, seed 0), slices 4 and 5 were measurable outliers versus the other 6 (healthy: `min_jacobian_singular_value` 0.17-0.20, `max_jacobian_condition` 2.9-7.7, 0 orientation flips; slices 4/5: singular value 0.010-0.012, condition 46-66, 5 total orientation flips), showing up as `normal_angle_deg_max=180°` specifically on seams `3->4` and `4->5` only — likely the visible crack in the original screenshot.

Root cause: mapping the in-plane Jacobian determinant sign (`Su_x*Sv_y - Su_y*Sv_x`, meaningful here since `planar_hole` is exactly flat so every control point has `z=0`) across each slice's full UV grid showed the sign flip is confined EXACTLY to the `(u=0, v=0)` corner cell(s) — the corner nearest the hole (`v=0` = inner boundary) AND nearest the shared seam with the previous slice (`u=0`). This is the O-grid's inherent inner-pole degeneracy: physical tangential arc length at radius `r` is `r * angle_step`, so near the inner boundary the true circumference available to a wedge shrinks toward zero while the radial extent does not -- `Su` legitimately becomes very small there regardless of data, making the local parameterization's sign highly sensitive to any noise in that corner's sparse point support. With only `resolution_v=4`/`degree_v=1` radial control freedom and a handful of real (not exact-circle) points feeding a free LSQ fit, that near-singular corner occasionally buckles into an actual local self-intersection.

Confirmed structural, not seed-0-specific bad luck: reran seeds 0-5 -- 3 of 6 seeds produced at least one flipped slice (seed 0: slices 4/5; seed 2: slice 5; seed 5: slices 0/1, with slice 1 alone hitting 8/144 flipped samples, worse than seed 0's worst case). Every flip in every seed was confined to a `(u≈0, v≈0)` inner-corner cell. This is the same underlying phenomenon as the plan's issue #2 (radial iso-line crowding near the inner boundary) taken to its extreme, not an independent bug -- it is currently unaddressed by any of Steps 1-3's diagnostics-only work and should be treated as a Step 4 priority candidate (arc-length reparameterization, already planned, is the most likely lever since it's the mechanism that would rebalance parameter density away from the physically-tiny inner corner; if that alone isn't enough, inner-corner-specific handling, e.g. widening the corner's own point-selection window or increasing local radial control density, is a candidate follow-up within Step 4's low-risk-seed-change scope).

A second, separate finding -- `phase2_boundary_conformance.outer` (symmetric_chamfer=0.095, hausdorff=0.696, coverage=0.605) much worse than `.inner` (0.023/0.061/0.965) -- is noted but deferred to Step 3/4 (likely a per-angle-bin circular-radius seed mismatch against a non-circular outer support region; needs the broader scene set to confirm before acting).

**Step 2 — Metric validation. COMPLETE.** `tests/test_annulus_chart.py` extended with 13 new tests (99/99 total passing):
- `JacobianDiagnosticsUnitTest`: hand-built healthy flat grid (area/singular-value/condition all exactly 1.0, no flips/degeneracy), a collapsed radial-extent grid (`near_degenerate_count > 0`, area `< 1e-4`), and a "bowtie" twisted grid (`orientation_flip_count > 0` with `near_degenerate_count == 0` and healthy area — proving degeneracy and orientation-reversal are correctly distinguished, unlike the old `jacobian_min <= 0` check).
- `SeamMetricsUnitTest`: perfect match (all metrics ~0), pure translation (gap moves, angles stay 0 — proves independence), tangent reversal (`seam_tangent_angle_deg_mean ≈ 180`), mirrored slice (normal AND tangent both ≈ 180 — documented as coupled by construction for any flat/z=0 surface, not independently injectable), and periodic wrap-around (`n`-th seam is `(n-1, 0)`, confirming the last-to-first closure is actually measured).
- `BoundaryConformanceUnitTest`: perfect match, uniform offset (both directions grow equally, coverage drops to 0 past tolerance), and — directly validating the plan review's point #7 — a **collapsed edge** (every chart sample = the same single reference point): one-directional distance alone reports 0.0 (looks perfect), while the symmetric direction correctly reports `reference_to_edge_mean > 0.1` and `boundary_coverage_ratio < 0.2`, proving the symmetric metric catches what a one-directional metric would miss.
- Also added to `AnnulusOGridChartTest`: a "known bad seed" regression/detection guard (`_annulus(seed=14)`, the test fixture's own generator) reproducing 8 orientation-flipped samples confined to slices with `min_jacobian_singular_value < 0.05` — proves the new metrics detect a REAL failure mode found during Step 1, not just synthetic constructions. This is a detection guard, not a target; Step 4 should revisit the expected count once a fix is applied.

**Step 3 — Baseline capture across more than one scene. COMPLETE.** Added 4 new scenes to `nurbs_constructor_benchmark/scenes.py`/`support_domains.py`: `planar_hole_offcenter` (hole shifted off-origin), `planar_hole_elliptical` (elliptical inner/outer boundary), `planar_hole_density_gradient` (radially inner-biased point density on the same circular annulus), `curved_annulus` (sine height on the annulus support).

Baseline (`points=600, seed=0`), across the 4 scenes that actually route to the O-grid (`curved_annulus` does not -- see limitation below):

| scene | orientation_flips | max_jacobian_condition | seam_normal_deg_mean | false_fill | outer conformance (chamfer/coverage) | inner conformance (chamfer/coverage) |
|---|---|---|---|---|---|---|
| planar_hole | 5 | 66.1 | 10.00 | 0.167 | 0.095 / 0.605 | 0.023 / 0.965 |
| planar_hole_offcenter | **20 (worst)** | **190.3 (worst)** | **17.50 (worst)** | **0.333 (worst)** | 0.060 / 0.795 | 0.019 / 1.000 |
| planar_hole_elliptical | 2 (best) | 22.9 | 5.00 | 0.112 (best) | 0.068 / 0.704 | 0.022 / 0.985 |
| planar_hole_density_gradient | 0 (best) | 19.7 | 0.00 (best) | 0.166 | 0.098 / 0.534 | 0.026 / 0.964 |

Two conclusions this distribution supports that a single scene could not:
1. **Off-center hole is the worst-case stressor** for the inner-corner Jacobian degeneracy (4x the flip count, 3x the condition number, 2x the false-fill of the centered case) -- asymmetric radial extent per angle means some slices get a much thinner inner corner than others. Any Step 4 fix must be validated against this scene, not just `planar_hole`.
2. **Outer boundary conformance is systematically poor across every scene** (chamfer 0.06-0.10, coverage 0.53-0.80), while inner is consistently good (chamfer 0.02-0.03, coverage 0.96-1.0) -- not scene-specific noise. Confirms the per-angle-bin circular-radius Coons seed is a poor fit for the outer boundary specifically (which is not the small, roughly-circular hole loop the seed was implicitly tuned against) regardless of hole shape/position/density. This is now a confirmed Step 4 target, not just a hypothesis.
3. `planar_hole_density_gradient` has the best Jacobian/seam numbers of all 4 (0 flips) but by far the worst `support_extrapolation_fraction_local`/`global` (not shown above, see raw report) -- a different, expected failure mode (sparse outer coverage) unrelated to this plan's scope.

**Known limitation, not fixed in this plan:** `curved_annulus` does not route to the annulus O-grid at all -- Phase 1's component builder splits it into 2 `disk_like` components (falling back to Phase 3's trimmed-rectangle path for both), regardless of `--bf-normal-threshold-degrees`/`--bf-offset-threshold-ratio` (tried 60°/2.0, no change -- the split happens somewhere in Stage 1's voxel hierarchy construction, not Phase 1's leaf-merge thresholds). Root-causing and fixing this is Phase 1/Stage-1-hierarchy scope, not this Phase 4 annulus-chart hardening plan's scope -- noted here as an open item for whoever next touches curved multi-loop component construction. Practical effect: this hardening plan's curvature-related design points (e.g. the per-slice self-propagated Jacobian reference normal, specifically designed so a genuinely curved annulus wouldn't misfire) remain architecturally justified but currently UNTESTED against a real curved O-grid case; only validated on flat scenes so far.

**Step 4 — Low-risk seed changes only, ablated one at a time. IN PROGRESS.**

**Candidate 1: equal-arc-length segment placement along the outer boundary. TRIED, REJECTED.** Implemented `_equal_arc_length_boundary_angles()` in `osn_gs/surface/torch_annulus_chart.py` -- a new opt-in `segment_placement` parameter (`"uniform_angle"` default, byte-identical to before; `"arc_length_outer"` places the 8 seam angles at equal arc length along the OUTER boundary instead of equal angle from the hole centroid). Motivation: `planar_hole_offcenter`'s equal-angle wedges are wildly unequal in physical size around an off-center hole. Verified `"uniform_angle"` stays exactly byte-identical (same 99/99 tests, same `planar_hole` numbers: seam_gap=0.01227, flips=5, cond_p95=3.38). A/B tested `"arc_length_outer"` against all 4 baseline scenes -- REJECTED, made things worse on 3 of 4:

| scene | flips (uniform -> arc_length_outer) | max_jacobian_condition_p95 | seam_normal_deg_mean | false_fill |
|---|---|---|---|---|
| planar_hole | 5 -> 9 | 3.38 -> 4.72 | 10.0 -> 5.0 | 0.167 -> 0.154 |
| planar_hole_offcenter (worst case, the one this was meant to fix) | 20 -> 20 | 8.64 -> 8.35 | 17.5 -> 25.0 | 0.333 -> 0.341 |
| planar_hole_elliptical | 2 -> 3 | 3.56 -> 3.70 | 5.0 -> 5.0 | 0.112 -> 0.112 |
| planar_hole_density_gradient | **0 -> 20** | 19.7 -> 6.67 | **0.0 -> 25.0** | 0.166 -> 0.199 |

Did not fix `planar_hole_offcenter` (the target case) at all, and catastrophically regressed `planar_hole_density_gradient` (0 flips -> 20). Root cause of the regression: using ONLY the outer radius to drive arc length ignores the inner boundary's own geometry entirely, so it doesn't address the actual inner-corner mechanism found in Step 1, and for `density_gradient` the outer-radius-per-angle-bin estimate is itself noisy (sparse outer coverage), producing an unstable arc-length CDF and badly-placed segments. Per the plan's own discipline ("never force a hard constraint that looks clean but hurts accuracy" -- the same principle applies to a seed choice, not just a hard constraint), `segment_placement` stays defaulted to `"uniform_angle"`; `"arc_length_outer"` is kept in the code as a documented, tested, working ablation tool (not deleted -- consistent with how the hard-C0 comparison was kept as a documented rejected alternative in this same file's history), but is not adopted.

**Remaining Step 4 candidates, not yet tried:** a seam-offset sweep, and a Hermite/derivative-aware Coons seed (blending a first-derivative estimate at the shared boundary, not just position). A candidate not in the original plan but suggested by this negative result: arc-length placement driven by BOTH inner and outer radius jointly (e.g. average, or whichever loop has the tighter local curvature) rather than outer alone, since the outer-only version provably ignores the actual inner-corner failure mechanism it was meant to fix.

**Step 5 — Post-fit continuity check, not assumed (not yet started).** Verify with Step 1/2 metrics whether each Step 4 seed change's continuity survives the independent per-slice LSQ fit. Only if seam metrics remain bad does this proceed to a **soft** seam regularization penalty (position+tangent+normal, small weight sweep) — never the previously-rejected hard shared-control-point constraint. Flagged as the largest architectural change in the plan; confirm with the user before implementing if Step 4 alone doesn't close the gap.

**Step 6 — Regression gate (not yet started).** `chamfer_rms`/`false_fill` must not regress beyond a small tolerance from baseline on every Step-3 scene. Add: no orientation flips, `min_jacobian_singular_value` above threshold, seam-quality and boundary-conformance thresholds derived from Step 3's actual multi-scene distribution. Update `docs/worklogs/` with full before/after numbers. Phase 5 remains gated on explicit user approval afterward.

## Verification
- `python -m unittest discover -s tests -p "test_*.py"` after each step (86/86 as of Step 1).
- `python -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole,<new annulus variants> --output <dir>` before/after each Step 4/5 change, comparing full metric distributions against the Step 3 baseline.
- Step 2's injected-failure tests must fail loudly before any Step 4 changes are trusted to make things better.

## User's Instruction
- if User confirms the plan is accomplished successfully, delete this hardening plan.