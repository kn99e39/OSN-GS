# Phase 5 Plan: Boundary-Aligned Extension Charts

> **Living document.** 현재 blocker와 재개 조건만 유지하며, 완료된 실험의 상세 내용은 `docs/worklogs/`를 기준으로 한다.

Status: **Boundary-First는 Phase 5에 도달했고 Step 5-A coupled shared-boundary fit을 기본값으로 채택했다(worklog 55/56). 현재는 Phase 5에서 발견된 component/topology blocker 때문에 Phase 1 remediation으로 되돌아간 상태다. Step 5-B soft G1은 별도 승인 전까지 보류한다.**

**Current blocker:** Phase 5에서 `curved_annulus`가 두 component로 갈리고, hole이 없는 `mild_curved_sheet`가 annulus로 오분류되는 Phase 1/2 문제가 확인되어 Phase 1 remediation으로 되돌아갔다. Worklog 60-B–65의 pairwise quadratic-proxy 및 Gaussian-native continuity 조사는 production 변경 없이 완료됐지만 broad feasibility가 기각됐다. Phase 5는 폐기된 것이 아니라 새로운 neighborhood/manifold-level connectivity 방법론이 승인·검증될 때까지 block된 상태다.

Governing document: `OSN_GS_Final_Boundary_First_NURBS_Direction.md` §"Phase 5 -- Boundary-Aligned Extension Charts" (목표/5.1-5.4/완료조건) and §14 승인 게이트 ("Phase 5 보고": extension chart, boundary frame, confidence, observed/extension separation -- 사용자 승인 없이 다음 Phase로 자동 진행하지 않는다).

## 완료된 Phase 5 선행 작업

- Step 5-A coupled shared-boundary fit은 구현·평가 후 기본값으로 채택했다. 상세 근거는 `docs/worklogs/55_step5a_coupled_boundary_fit_evaluation.md`와 `docs/worklogs/56_step5a_coupled_boundary_fit_production_adoption.md`에 있다.
- Step 5-B soft G1 continuity는 별도 승인 전까지 보류한다. Step 5-A만으로 seam tangent/normal mismatch가 충분히 낮아 우선 필요성이 입증되지 않았다.
- Phase 4 hardening 및 Step 5-A의 과거 후보·정량표는 worklog 39–56에 보존한다. 이 living plan에서 완료된 실험 절차를 반복하지 않는다.

## Prerequisite 2: Phase 1 curved-component connectivity remediation + outer-boundary conformance (진행 중, 미해결)

Discovered while spot-checking the just-adopted coupled-fit pipeline (not a regression it caused -- both issues predate Step 5-A and were already latent in Phase 1/2).

### Curved + hole component misrouting (both directions)

- `curved_annulus` (sine-curved surface with one hole, GT topology = annulus): Phase 1's `build_surface_components` still splits it into 2 `disk_like`/`complex` components instead of routing it to the annulus O-grid path -- reproduced independent of `--bf-normal-threshold-degrees`/`--bf-offset-threshold-ratio` (originally found in worklog 39/43, never carried into a tracked TODO item until now).
- **Newly found, 2026-07-22**: the OPPOSITE failure also exists -- `mild_curved_sheet` (a single curved surface with NO hole, GT topology = disk_like) is misrouted INTO annulus topology, producing a spurious 8-wedge O-grid split (`patches=8` vs `gt=1`, `topology_label_ari=0.000`). Same underlying instability (Phase 1/2 loop/topology classification is unreliable once real curvature is present), manifesting in both directions (false negative and false positive for "has a hole").
- **Risk assessment (agent judgment, given to the user directly)**: curvature + an occlusion-created hole is plausibly a common combination in real captured data (more common than the synthetic flat-plane-with-hole case), and this failure mode degrades quality silently (no crash, only visible on inspection of the render or the topology/ARI numbers) -- so likely to matter non-trivially on real complex datasets. This is why the user and an external GPT review agreed to reopen Phase 1 now rather than deferring it further.
- **현재 sequencing**: Phase 1 component connectivity를 다시 열어 `curved_annulus`를 정확히 ONE annulus component로 복원하고 `mild_curved_sheet`의 spurious hole을 제거한다. Worklog 60-B–65의 pairwise 방법은 기각됐으므로 반복하지 않는다. 새 neighborhood/manifold-level 방법론이 broad negative control까지 통과한 뒤 coupled boundary fit을 재검증하고 Phase 5 extension 본편으로 복귀한다.

### Outer-boundary conformance never targeted, still bad under the adopted pipeline

- `phase2_boundary_conformance`'s outer (outer-loop) symmetric chamfer/coverage has been measurably worse than inner (hole-loop) on every annulus scene since Step 3 of the hardening pass (worklog 39), reconfirmed at every later checkpoint (worklogs 50-53), and **still true after Step 5-A's adoption** -- re-measured 2026-07-22 on the current production default: outer chamfer 0.069-0.099 / coverage 0.52-0.69 vs. inner chamfer 0.020-0.026 / coverage 0.93-1.0 across all 4 scenes. Full table in `TODO.md`'s Priority 3 subsection.
- Root cause candidate (Phase 1's per-leaf plane-AABB polygon union over-extending outward, worklog 45/46) was only partially addressed by the eligibility classifier (worklog 47-49), which fixed overall false_fill/coverage but not this specific outer-conformance metric.
- Relevant to Phase 5 because the extension chart's own boundary tangent/local frame (§5.2) will be seeded from this same Phase 2 boundary estimate -- an unresolved outer-conformance gap risks propagating directly into extension chart quality. Not yet investigated or fixed; flagged here so it isn't silently carried into Phase 5 proper.

## Phase 5 extension 본편 (Phase 1 remediation과 boundary blocker 해결 후 재개)

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

- 전체 pytest를 각 implementation area 종료 시 실행한다.
- Phase 1 remediation은 worklog 64/65의 scene·seed·distance negative-control 범위보다 좁아지면 안 된다.
- Remediation gate 통과 후 `construct_boundary_first`/`score_state`와 `osn-gs benchmark --constructor boundary_first`로 coupled boundary fit 및 Phase 5 입력을 재검증한다.
- Worklog는 한국어로 작성하고 이 문서와 `docs/README.md`에서 연결한다.

## User's Instruction

- Phase 1 remediation의 broad feasibility와 사용자 승인이 확인되기 전 Phase 5 extension 본편을 재개하지 않는다.
- Step 5-B soft G1과 Phase 6은 각각 별도 사용자 승인 전 시작하지 않는다.
- Phase 5 범위가 완료되면 정량 근거를 worklog에 남긴 뒤 이 living plan을 폐기한다.
