# Phase 4 하드닝 Step 6 — 회귀 게이트 실행 (플랜에 정의만 되고 미실행 상태였음)

작성일: 2026-07-22
상태: **실행 완료, 결과 혼재(mixed) — 사용자 판단 필요.**
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/39`(Step 3 baseline), `49`(production 채택)

## 배경

사용자가 "Phase 4는 끝난 것으로 간주하고, 완료 시 수행하기로 했던 작업이 있으면 그대로 따라가봐"라고 지시했다. `OSN_GS_Phase4_Hardening_Plan.md`를 확인해보니 **Step 6(회귀 게이트)이 플랜에 명시적으로 정의만 되어 있고 "not yet started" 상태로 남아있었다** — 정확히 "완료 시 수행하기로 했던 작업"에 해당한다:

> Step 6 — Regression gate (not yet started). `chamfer_rms`/`false_fill` must not regress beyond a small tolerance from baseline on every Step-3 scene. Add: no orientation flips, `min_jacobian_singular_value` above threshold, seam-quality and boundary-conformance thresholds derived from Step 3's actual multi-scene distribution. Update `docs/worklogs/` with full before/after numbers. Phase 5 remains gated on explicit user approval afterward.

이 pass에서 실제로 실행했다.

## 방법

`docs/worklogs/39`(Step 3 baseline, `OSN_GS_Phase4_Hardening_Plan.md` 하드닝 시작 전 원본 numbers)와, worklog 45-49를 전부 반영한 **현재 production 기본값**(`osn-gs benchmark --constructor boundary_first`, 추가 플래그 없음 — `filter_boundary_leaf_eligibility=True`, `eligibility_gap_closing_cells=1`, `segment_placement=uniform_angle` 등 전부 기본값)을 Step 3와 동일한 4개 씬(`planar_hole`, `planar_hole_offcenter`, `planar_hole_elliptical`, `planar_hole_density_gradient`)에 대해 직접 비교했다.

## 결과

| 씬 | orientation_flips (S3→현재) | max_jacobian_condition (S3→현재) | false_fill (S3→현재) |
|---|---|---|---|
| planar_hole | 5 → **10 (악화)** | 66.1 → 33.9 (개선) | 0.167 → 0.128 (개선) |
| planar_hole_offcenter | 20 → 6 (개선) | 190.3 → **324.3 (악화, +70%)** | 0.333 → 0.158 (개선) |
| planar_hole_elliptical | 2 → 0 (개선) | 22.9 → 4.4 (개선) | 0.112 → 0.111 (동일) |
| planar_hole_density_gradient | 0 → 0 (동일) | 19.7 → 12.6 (개선) | 0.166 → 0.167 (동일) |

- **false_fill**: 4개 씬 전부 개선 또는 동일 — Step 6 게이트의 핵심 기준(정확도/coverage 회귀 없음)은 명확히 통과.
- **orientation_flips**: 3/4는 개선 또는 동일이지만, **`planar_hole`은 5→10으로 2배 악화**됐다.
- **max_jacobian_condition**: 3/4는 크게 개선됐지만, **`planar_hole_offcenter`(Step 3에서 이미 최악 케이스로 지목된 씬)가 190.3→324.3으로 70% 악화**됐다.
- 4개 씬 전부 `holonomy_consistent=True`(방향 일관성 위반 없음), `total_near_degenerate_samples=0`(퇴화 없음) — 이 두 항목은 완전히 통과.
- outer boundary conformance(Step 3에서 확인된, 이번 pass의 범위 밖인 기존 gap): chamfer 0.07-0.10 / coverage 0.52-0.69 — Step 3(chamfer 0.06-0.10 / coverage 0.53-0.80)와 사실상 동일, 개선도 악화도 아님(예상대로, 이번 pass가 건드린 지점이 아님).
- `min_jacobian_singular_value`에 대한 명시적 threshold는 플랜에 한 번도 정의된 적이 없다 — "above threshold"라는 게이트 문구는 있지만 숫자가 없어서 pass/fail 판정을 내릴 수 없다(수치만 기록: 0.0014~0.13, `offcenter`가 가장 낮음).

## 해석

Step 6가 요구하는 "`chamfer_rms`/`false_fill`이 tolerance 이내에서 회귀하지 않아야 한다"는 기준은 **명확히 통과**한다 — 정확도/coverage 관점에서는 모든 씬이 개선되거나 동일하다.

하지만 "no orientation flips"라는 문구를 엄격하게 읽으면 `planar_hole`의 flip 증가(5→10)와 `offcenter`의 max_jacobian_condition 악화(+70%)는 **문자 그대로는 게이트를 깨끗하게 통과하지 못한다.** 두 지표 모두 eligibility 필터링이 어떤 leaf가 어떤 wedge에 점을 공급하는지를 바꾸면서 특정 wedge의 inner-corner 상황이 달라진 결과로 보이지만(worklog 39/41이 이미 문서화한, inner-corner degeneracy가 데이터 분포에 민감하다는 기존 패턴과 일치), 별도로 근본 원인을 분해하지는 않았다.

## 권고

이 결과를 **"조건부 통과"** 로 본다: 이번 하드닝 pass의 원래 동기(seam continuity, outer-boundary bias, false-fill)는 명확히 개선됐고 회귀 없음이 확인됐지만, Jacobian/flip 지표는 씬에 따라 트레이드오프가 있다. 정밀 threshold가 애초에 정의된 적이 없어서 엄격한 자동 pass/fail 판정은 불가능하고, 이건 **사용자의 판단이 필요한 지점**이라고 본다 — 이 정도 트레이드오프를 감수하고 Phase 4를 공식적으로 닫을지, 아니면 `planar_hole`/`offcenter`의 Jacobian 회귀를 추가로 조사할지.

## Phase 5 게이트

플랜 문서와 `OSN_GS_Final_Boundary_First_NURBS_Direction.md` §14 모두 "Phase 5는 사용자의 명시적 승인 없이 시작하지 않는다"를 반복해서 명시한다. Step 6를 실행했다고 해서 이 게이트가 자동으로 풀리는 게 아니다 — Phase 4를 닫기로 결정하더라도 Phase 5 착수는 **별도의 명시적 승인**이 필요하다.

## 검증

```powershell
.venv\Scripts\python.exe -m osn-gs benchmark --constructor boundary_first --scenes planar_hole planar_hole_offcenter planar_hole_elliptical planar_hole_density_gradient --points 600
```

코드 변경 없음 — 이번 pass는 순수 검증/보고이며, 기존 4개 실제 씬에 대해 이미 존재하는 production 기본값을 그대로 실행하고 `report.json`의 실제 수치를 Step 3 baseline과 비교한 것뿐이다.
