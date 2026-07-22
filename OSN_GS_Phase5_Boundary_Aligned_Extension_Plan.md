# Phase 5 Plan: Boundary-Aligned Extension Charts (+ coupled patch-boundary prerequisite)

> **Living document** -- kept in sync with progress, same convention as the now-retired `OSN_GS_Phase4_Hardening_Plan.md`. Updated after each step, not rewritten from scratch.

Status: **Step 5-A (coupled patch-boundary fitting) implemented, evaluated, and PRODUCTION ADOPTED, 2026-07-22 (`docs/worklogs/55` implementation+evaluation, `docs/worklogs/56` adoption).** Result: exceptionally clean 4-scene x 5-seed win -- orientation flips drop to exactly 0 in every scene/seed (including the previously-worst `planar_hole_offcenter` seeds 1/3 and `planar_hole_density_gradient`'s condition regression), region-segmented flip counts (seam-adjacent/inner/outer/patch-interior) are ALL zero (the fold is genuinely removed, not relocated to the interior), chamfer_rms flat-or-improved on all 4 scenes, false_fill roughly flat (small, non-alarming upticks on 2 scenes). Jacobian condition max/p99 improve dramatically everywhere. Seam tangent/normal mismatch also drop near-zero as an emergent (not designed-for) side effect, without Step 5-B. `build_annulus_chart`'s `coupled_boundary_fit` now defaults to `True` (user-approved production adoption); the pre-Step-5-A independent per-wedge fit stays available via `coupled_boundary_fit=False` / CLI `--bf-disable-coupled-boundary-fit` as a tested fallback, not deleted. **Step 5-B (soft G1) is ON HOLD -- explicitly not to be started until the user gives separate go-ahead.**

**New prerequisite opened, 2026-07-22 ("Prerequisite 2" below), before Phase 5 proper starts:** while spot-checking the just-adopted pipeline, found (a) a curved surface with a hole (`curved_annulus`) still does not route to annulus topology at all (a known, previously-documented Phase 1 gap, worklog 39/43 -- just never carried into a tracked TODO item before now), and, newly, (b) a hole-FREE curved surface (`mild_curved_sheet`) is misrouted INTO annulus topology (a spurious hole). A worklog-wide audit for other similarly "found but never actually fixed" issues also surfaced that **outer-boundary conformance has been structurally poor on every annulus scene since Step 3 of the hardening pass (worklog 39) and was never once directly targeted** by any Step 4 or Step 5-A candidate -- confirmed still true under the current coupled-fit production default (see `TODO.md`'s new subsection under Priority 3 for the re-measured numbers). Both are relevant to Phase 5 proper, which consumes this same boundary estimate for its extension-chart seed. Agreed sequencing (user + external GPT review, 2026-07-22): reopen Phase 1's curved-component segmentation -> verify `curved_annulus` routes to ONE annulus component -> re-confirm coupled boundary fit behaves correctly on it -> only then proceed to Phase 5 proper. Not yet started.

Governing document: `OSN_GS_Final_Boundary_First_NURBS_Direction.md` §"Phase 5 -- Boundary-Aligned Extension Charts" (목표/5.1-5.4/완료조건) and §14 승인 게이트 ("Phase 5 보고": extension chart, boundary frame, confidence, observed/extension separation -- 사용자 승인 없이 다음 Phase로 자동 진행하지 않는다).

## How Phase 5 was opened

Phase 4 (Boundary-Conforming Chart Generator) was previously completed/approved (worklog 31/36), then hardened (worklogs 39-51, `OSN_GS_Phase4_Hardening_Plan.md`). The hardening pass's Step 4 (seam-angle placement) tried three fixed rules (`outer_radius_weighted_segment_placement`, `seam_phase_offset`, `hermite_boundary_seed`) and two adaptive local optimizers (`worst_wedge_optimized` -- worklog 43/52, `profile_constrained` -- worklog 53) against the same 4-scene x 5-seed protocol (`planar_hole`, `planar_hole_offcenter`, `planar_hole_elliptical`, `planar_hole_density_gradient`). **None produced a clean multi-scene win.** Per the user's decision (2026-07-22):

- Step 4-D-style seam-angle-repositioning search is **closed as an avenue** -- no further proxy refinement. `worst_wedge_optimized`/`profile_constrained` remain opt-in ablation tools only; production stays `uniform_angle`.
- `planar_hole_offcenter`/`planar_hole_density_gradient` are **NOT** accepted as documented exceptions (unlike `planar_hole`, which independently met that bar in worklog 51). The user's stated reason: there is no evidence the current failures are an underfitting problem, and increasing per-wedge `resolution_v`/degree or adding knot refinement would only risk giving more expressive power to an already-invalid parameterization, not fix it.
- `OSN_GS_Phase4_Hardening_Plan.md` is deleted (2026-07-22) per its own closing instruction ("사용자가 계획이 성공적으로 달성되었다고 확인하면, 이 하드닝 계획 문서를 삭제한다") -- its full detail lives on in `docs/worklogs/39-43`, `45-51`, and this session's `52`/`53`.
- The next work is **not** further seam-angle search, but a structurally different mechanism: **coupled patch-boundary fitting** (below, "Step 5-A/B"), followed by the actual Phase 5 scope once that prerequisite is resolved and reported.

## Prerequisite: Step 5-A / Step 5-B (coupled patch-boundary fitting)

Scoped and specified by the user (2026-07-22, relaying an external GPT review the user then edited/tightened). Framing, verbatim intent: **not** "seal the seam so it looks good," but verify whether moving from independently-fit wedges to a shared-boundary-constrained system also relieves the patch-interior fold identified back in Step 1 of the hardening pass (the inner-pole degeneracy, `Su -> 0` as tangential arc length shrinks toward the hole) -- a per-wedge parameterization failure that shared-boundary coupling is not guaranteed to touch, and could conceivably worsen if a wedge's interior is forced to agree with a neighbor's already-poor boundary estimate. The evaluation below exists specifically to catch that gap, not just to confirm seams look clean.

**Architectural distinction from the already-rejected hard-C0 attempt** (`torch_annulus_chart.py`'s "NOT hard-enforced" docstring block, from Phase 4 itself): that attempt fit each wedge independently to its own local optimum, then overwrote the shared boundary control points post-hoc -- forcing the interior to honor a boundary it was never fit against, which is exactly why it regressed accuracy (`planar_hole` chamfer 0.0058->0.0061, false-fill 0.167->0.200). Step 5-A instead makes the shared boundary control points **joint fit variables from the start**, so both wedges' interiors are optimized consistently against the one boundary they'll actually share -- a fundamentally different (and, in this codebase, previously untested) approach.

**Scope discipline (explicit user constraints):** no selector, retry, fallback, or scene-specific branch. No hard C1 up front. Reuse the exact same 4-scene x 5-seed x count=600 protocol as every prior Step 4 candidate -- report only, do not change the production default without a separate approval.

**Implementation-scope note (my proposal, approved by the user):** couple ONLY the shared seam's own boundary curve control points (the radial edge at each shared seam angle) as joint variables -- not the full control grid -- to avoid recreating the previously-rejected "force everything" pattern under new packaging.

### Step 5-A -- shared-boundary joint fit. **IMPLEMENTED AND EVALUATED, 2026-07-22 (`docs/worklogs/55`).**

- Represent each adjacent wedge pair's seam as one shared physical boundary curve.
- Both patches reference the SAME shared boundary control points (joint variables in a single least-squares system), not two independently-fit curves later averaged.
- With the shared edge fixed (once solved), refit each patch's own interior control points against it.
- Explicitly forbidden: plain post-hoc edge averaging of two independently-fit patches.
- **Implementation**: `fit_coupled_wedge_ring_lsq` (`osn_gs/surface/torch_nurbs.py`) assembles one global linear system across all wedges, scatter-accumulating each wedge's own local normal-equations system (data term + wedge-private smoothness/Tikhonov, unchanged) into shared vs. private variable slots, solving once, then gathering back. `build_annulus_chart`'s `coupled_boundary_fit` **now defaults to `True` (production adopted, `docs/worklogs/56`)**; `coupled_boundary_fit=False` / CLI `--bf-disable-coupled-boundary-fit` recovers the pre-Step-5-A independent fit. Couples ONLY the shared boundary columns, per this doc's own scoping note above -- no cross-seam smoothness term (that would be 5-B).
- **Result: all 4 required questions below answered cleanly positive** -- see worklog 55's full table. Orientation flips go to exactly 0 in every scene/seed tested, region-segmented (not just aggregated), chamfer_rms flat-or-improved everywhere, false_fill roughly flat. **Adopted as production default, user-approved 2026-07-22.**

### Step 5-B -- soft continuity (only if 5-A is insufficient). **ON HOLD -- do not start without separate, explicit user go-ahead.**

- Soft tangent-plane / G1 continuity penalty, small weight, added only if Step 5-A's own evaluation (below) doesn't clear the bar.
- No hard C1 constraint from the outset.
- Worklog 55 found seam tangent/normal mismatch already near-zero as an emergent (not designed-for) side effect of Step 5-A alone, on all 4 tested scenes -- suggestive that 5-B may be unnecessary, but not proof for untested scenes/conditions. User has explicitly deferred this decision -- do not implement until instructed.

### Required evaluation (per scene, per seed, all three placements the seam mechanism is orthogonal to: at minimum `uniform_angle`; extend to the two rejected optimizers only if useful for comparison)

- seam positional gap
- seam tangent/normal mismatch
- orientation flip count and fraction, segmented into: seam-adjacent / inner / outer / patch-interior (not aggregated into one number -- this is the specific check for whether coupling only moves the fold rather than removing it)
- outer-boundary flips specifically
- Jacobian condition p95/p99/max
- normalized minimum singular value
- near-degenerate count
- chamfer_rms (GT)
- false_fill (GT-compared union metric)
- inner/outer Phase-2 boundary conformance (symmetric chamfer + coverage, both loops)

### Specific questions the report must answer

1. Does `planar_hole_offcenter` seed 1's wedge-wide fold (the worst documented case, worklog 51) actually resolve, or only relocate?
2. Does `planar_hole_offcenter` seed 3's recurring regression (seen across worklog 51/52/53) disappear?
3. Does `planar_hole_density_gradient`'s Jacobian-condition regression (worklog 52: 14.22->56.28 mean cond_p95 under `worst_wedge_optimized`; worklog 53: ->95.63 under `profile_constrained`) NOT reappear under coupled fitting?
4. After continuity is enforced, does a patch-interior fold still remain (i.e. did coupling only move the problem to the interior, leaving total defect count unchanged)?

**Stop condition:** report results only; do not adopt as production default without a separate, explicit user go-ahead -- same discipline as every prior Step 4 candidate.

## Prerequisite 2: Phase 1 curved-component segmentation reopening + outer-boundary conformance (opened 2026-07-22, NOT STARTED)

Discovered while spot-checking the just-adopted coupled-fit pipeline (not a regression it caused -- both issues predate Step 5-A and were already latent in Phase 1/2).

### Curved + hole component misrouting (both directions)

- `curved_annulus` (sine-curved surface with one hole, GT topology = annulus): Phase 1's `build_surface_components` still splits it into 2 `disk_like`/`complex` components instead of routing it to the annulus O-grid path -- reproduced independent of `--bf-normal-threshold-degrees`/`--bf-offset-threshold-ratio` (originally found in worklog 39/43, never carried into a tracked TODO item until now).
- **Newly found, 2026-07-22**: the OPPOSITE failure also exists -- `mild_curved_sheet` (a single curved surface with NO hole, GT topology = disk_like) is misrouted INTO annulus topology, producing a spurious 8-wedge O-grid split (`patches=8` vs `gt=1`, `topology_label_ari=0.000`). Same underlying instability (Phase 1/2 loop/topology classification is unreliable once real curvature is present), manifesting in both directions (false negative and false positive for "has a hole").
- **Risk assessment (agent judgment, given to the user directly)**: curvature + an occlusion-created hole is plausibly a common combination in real captured data (more common than the synthetic flat-plane-with-hole case), and this failure mode degrades quality silently (no crash, only visible on inspection of the render or the topology/ARI numbers) -- so likely to matter non-trivially on real complex datasets. This is why the user and an external GPT review agreed to reopen Phase 1 now rather than deferring it further.
- **Agreed sequencing**: reopen Phase 1 (`osn_gs/surface/torch_surface_components.py`, `torch_voxel_hierarchy.py`) specifically for curved multi-loop component segmentation -> verify `curved_annulus` routes to exactly ONE annulus component (not 2 disk_like/complex) -> re-confirm `coupled_boundary_fit` (Step 5-A) behaves correctly once it actually reaches the O-grid path on this scene -> only then start Phase 5 proper. NOT YET STARTED -- no Phase 1 code has been touched.

### Outer-boundary conformance never targeted, still bad under the adopted pipeline

- `phase2_boundary_conformance`'s outer (outer-loop) symmetric chamfer/coverage has been measurably worse than inner (hole-loop) on every annulus scene since Step 3 of the hardening pass (worklog 39), reconfirmed at every later checkpoint (worklogs 50-53), and **still true after Step 5-A's adoption** -- re-measured 2026-07-22 on the current production default: outer chamfer 0.069-0.099 / coverage 0.52-0.69 vs. inner chamfer 0.020-0.026 / coverage 0.93-1.0 across all 4 scenes. Full table in `TODO.md`'s Priority 3 subsection.
- Root cause candidate (Phase 1's per-leaf plane-AABB polygon union over-extending outward, worklog 45/46) was only partially addressed by the eligibility classifier (worklog 47-49), which fixed overall false_fill/coverage but not this specific outer-conformance metric.
- Relevant to Phase 5 because the extension chart's own boundary tangent/local frame (§5.2) will be seeded from this same Phase 2 boundary estimate -- an unresolved outer-conformance gap risks propagating directly into extension chart quality. Not yet investigated or fixed; flagged here so it isn't silently carried into Phase 5 proper.

## Phase 5 proper (starts only after Step 5-A/B AND Prerequisite 2 are resolved and reported)

Per the governing document, once the observed-surface chart itself (including its boundary/seam quality) is trustworthy:

- **5.1 boundary segment selection**: classify boundary samples into observed outer support boundary / inner hole boundary / crease boundary / invalid-untrusted boundary / potential occlusion boundary. Not every boundary is an extension target.
- **5.2 local frame**: per boundary sample -- boundary tangent, surface normal, in-surface outward normal, confidence.
- **5.3 extension chart**: `S_ext(s, t)`, `s` = boundary tangential parameter, `t` = outward extension parameter.
- **5.4 controlled extrapolation**: `C_ext(s, t)` kept as a separate field (not merged into observed-surface confidence), seeded from boundary confidence, local curvature continuation, normal consistency, support density decay, and a maximum extension distance.

**완료 조건 (governing doc, verbatim scope):**
- observed geometry와 extension geometry가 명시적으로 분리됨
- extension이 support mask 바깥에서만 생성됨
- boundary tangent 방향이 안정적임
- extension confidence export 및 visualization 가능

**§14 Phase 5 보고 요구사항** (this doc's eventual completion report must cover): extension chart, boundary frame, confidence, observed/extension separation. 사용자 승인 없이 다음 Phase(6)로 자동 진행하지 않는다.

## Verification

- `python -m unittest discover -s tests -p "test_*.py"` after each step.
- Same 4-scene x 5-seed x count=600 protocol via `construct_boundary_first`/`score_state` (or the CLI `osn-gs benchmark --constructor boundary_first`) before/after each Step 5-A/B change.
- Worklogs in Korean, per the project's standing convention, linked from this doc as they land.

## User's Instruction

- Update this document as work proceeds (do not let it go stale); do not auto-advance past Step 5-A/B into Phase 5 proper, or past Phase 5 into Phase 6, without explicit user approval at each gate.
- If the user confirms this plan's scope is fully accomplished, retire this document the same way `OSN_GS_Phase4_Hardening_Plan.md` was retired.
