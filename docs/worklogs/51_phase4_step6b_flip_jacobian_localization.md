# Phase 4 하드닝 Step 6-B — flip/Jacobian 회귀 국소화 진단 (production 미변경)

작성일: 2026-07-22
상태: **진단 완료, 판정 두 갈래(씬별로 다름). Phase 4 완료 선언은 아직 하지 않음. Production 코드 변경 없음.**
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/50`(Step 6)

## 배경

worklog 50(Step 6 회귀 게이트)에서 `planar_hole`의 orientation flip 증가(5→10)와 `planar_hole_offcenter`의 max_jacobian_condition 악화(190.3→324.3, +70%)가 발견됐다. 사용자가 이걸 근거로 Phase 4를 완료 처리하지 말고, **production 변경 없이** 이 이상이 (a) inner-corner의 작은 국소 outlier인지 (b) 구조적 회귀인지 판별하는 제한적 진단(Step 6-B)만 하라고 지시했다.

## 방법

기존 `_jacobian_diagnostics`의 `collect_samples=True`(기존 opt-in 파라미터, 기본 동작 무변경)를 통해 각 wedge의 12×12 raw per-sample grid(`u, v, sigma_min, condition, orientation_dot`)를 수집했다. `planar_hole`/`planar_hole_offcenter` 각각 5개 seed(0-4)에 대해:
- 전체 sample 통계: condition mean/p95/p99/max, sigma_min min/p01(및 `characteristic_length`로 정규화한 값), condition>50/100/200 비율, flip 비율.
- flip·high-condition(>200) sample의 wedge id와 UV 위치(`u`=tangential, `v`=radial; `v<0.18`→inner, `v>0.82`→outer, `u` 양 끝 1-cell→seam 근접)를 분류.
- wedge별 12×12 flag grid에 4-connectivity flood-fill을 적용해 고립 sample인지 연속 영역인지 판별.
- outer boundary(`v≥0.82`, 2행)에 flip/high-condition sample이 있는지 별도 카운트.
- flip이 발생한 wedge에 인접한 seam(`SeamDiagnostic`)의 gap/normal-angle을 대조.
- `phase2_boundary_conformance["outer"]`(Phase 2 outer loop 대비 chart edge conformance)를 매 seed마다 함께 기록.

## 결과: `planar_hole` (5 seed)

| seed | n | cond mean/p95/p99/max | flip_ratio | flip 위치 | outer flip/highcond | outer conformance chamfer/coverage |
|---|---|---|---|---|---|---|
| 0 | 1152 | 1.70/2.83/7.16/33.86 | 0.87%(10) | inner·inner+seam·mid+seam, wedge 4/6/7 | 0/0 | 0.099/0.610 |
| 1 | 1152 | 1.52/2.56/3.02/4.20 | 0.00%(0) | — | 0/0 | 0.109/0.530 |
| 2 | 1152 | 1.64/2.77/4.36/72.30 | 0.61%(7) | inner·mid+seam, wedge 5만 | 0/0 | 0.106/0.571 |
| 3 | 1152 | 1.52/2.64/3.05/3.67 | 0.00%(0) | — | 0/0 | 0.102/0.624 |
| 4 | 1152 | 1.47/2.34/2.74/16.45 | 0.17%(2) | inner+seam, wedge 3만 | 0/0 | 0.110/0.617 |

- **5개 seed 전부 `condition>200` 비율 0.00%.** 개별 seed의 `condition_max`는 가끔 크게 튀지만(seed2=72.3) `p95`(2.3~2.8)·`p99`(2.7~7.2)는 항상 매우 작다 — max는 고립된 outlier 1개 샘플이지, 분포 전체가 나빠진 게 아니다.
- **flip은 항상 wedge 1개, 2~4개 sample의 작은 연속 클러스터**로만 나타나고, 매번 `inner`/`inner+seam` 근처다. `outer` 영역 flip/highcond는 **5개 seed 전부 0**.
- outer conformance(chamfer 0.10~0.11, coverage 0.53~0.62)는 seed 상관없이 거의 일정 — Step 3에서 이미 확인된, 이번 pass 범위 밖의 기존 gap과 정확히 일치(새로 나빠진 게 아님).

**판정 기준 적용**: 이상이 inner-corner의 작은 국소 영역에만 있고(✓), p95/p99가 건강하고(✓), outer boundary가 건강하고(✓, 5/5 seed에서 flip 0), 여러 seed에서 catastrophic하게 반복되지 않는다(✓, 최악의 경우도 flip 10/1152=0.87%, condition>200 비율 0%). **→ `planar_hole`은 documented exception으로 완료 처리 기준을 만족한다.**

## 결과: `planar_hole_offcenter` (5 seed)

| seed | n | cond mean/p95/p99/max | flip_ratio | flip 위치 | outer flip/highcond | outer conformance chamfer/coverage |
|---|---|---|---|---|---|---|
| 0 | 1152 | 2.44/3.52/7.32/324.26 | 0.52%(6) | inner·inner+seam, wedge 4/5 | 0/0 | 0.069/0.693 |
| **1** | 1152 | **8.12/13.99/57.57/3109.37** | **19.01%(219)** | inner·mid·**outer**(28)·seam 전영역, wedge 0/1/2/3/4/**5(136)**/6 | **28/0** | 0.084/0.558 |
| 2 | 1152 | 1.77/3.58/4.64/9.68 | 0.00%(0) | — | 0/0 | 0.077/0.669 |
| 3 | 1152 | 3.49/3.91/**13.19**/1556.79 | 1.56%(18) | inner·inner+seam·mid+seam, wedge 2/3/4/7 | 0/0 | 0.067/0.748 |
| 4 | 1152 | 1.96/4.00/5.24/51.79 | 0.26%(3) | inner·inner+seam, wedge 4만 | 0/0 | 0.070/0.720 |

- **5개 seed 중 2개(seed 1, 3)에서 뚜렷한 악화가 반복된다** — 1회성 outlier가 아니다.
- **seed 1이 특히 심각하다**: flip이 1152개 sample 중 219개(19%), wedge 5 하나에서만 136개(그 wedge의 144 sample 중 94%) — inner corner만이 아니라 **wedge 전체(inner·mid·outer 전 영역)에 걸친 연속 클러스터**다. `condition_max=3109`, `p99=57.6`(다른 seed 대비 5~10배)까지 나빠졌다. **outer 영역에서도 flip 28개 발생** — "outer extension candidate boundary에 영향을 주면 안 된다"는 기준을 직접 위반한다.
- seed 3도 `p99=13.19`(healthy seed 대비 2~3배), `condition_max=1557`로 명확히 나쁜 쪽에 속한다(flip은 outer까지는 안 갔지만 18개로 seed 0/2/4보다 3~35배 많음).
- seed 1의 wedge 5를 직접 조사: `inner_radius=0.033`(도메인 스케일 대비 거의 0에 가까움), `point_count=145`(정상), `holonomy_consistent=True`이지만 `holonomy_local_disagreement_count=2`(정상은 0) — 즉 인접 seam 경계 두 곳에서 참조 법선 부호가 실제로 뒤집혀 있다(우연히 짝수라서 전역 parity 체크만으로는 "consistent"로 읽히지만, 이 wedge의 참조 법선 자체가 이웃과 어긋나 있다는 뜻).

**해석**: wedge 5의 `inner_radius≈0.033`은 이 wedge가 그 seed의 무작위 점 배치에서 hole 경계에 거의 닿을 만큼 얇은 쐐기라는 뜻이다 — Step 1/worklog 39/41에서 이미 문서화된 "inner-corner collapse"(`Su -> 0`) 메커니즘과 정확히 같은 기제이지만, 이 씬(중심에서 벗어난 hole)에서는 특정 무작위 시드가 그 collapse를 wedge 하나 전체를 접히게 할 만큼 극단적으로 만들 수 있다는 것이 이번에 새로 확인됐다. `planar_hole_offcenter`가 Step 3에서부터 이미 "inner-corner degeneracy 최악 케이스"로 지목됐던 것과 일치하는, 같은 근본 원인의 심화된 발현이다 — eligibility 필터링(worklog 45-49)이 이 메커니즘을 새로 만든 것은 아니지만, 어떤 leaf가 어떤 wedge에 점을 공급하는지를 바꾸면서 이 특정 seed에서 그 collapse를 더 심하게 유발했을 가능성이 있다(직접 검증하지 않음, 가설임을 명시).

**판정 기준 적용**: outer boundary에 영향을 준다(✗, seed 1에서 outer flip 28개), p99까지 악화된다(✗, seed 1·3 모두), 여러 seed에서 반복된다(✗, 5개 중 2개). **→ `planar_hole_offcenter`는 documented exception 기준을 만족하지 못한다. 추가 조사 및 targeted fix 제안이 필요하다.**

## 제안 (구현하지 않음, 방향만 제시)

1. **근본 원인은 이미 알려진 inner-corner collapse 메커니즘**(Step 1/worklog 39/41)의 심화 사례로 보인다 — 새로운 별도 결함이 아니라, 기존에 "보류(HOLD)" 처리된 Step 4-D(`worst_wedge_optimized`, `kappa_k = max(d_k, w_k_inner)/(min(d_k, w_k_inner)+eps)` 최소화)가 정확히 이 문제(얇은 inner corner wedge)를 겨냥해서 만들어졌던 후보다. `planar_hole_offcenter`(우선순위 타깃)에서 Step 4-D가 flip을 20→12로 줄였다는 기존 결과(worklog43)와 이번 발견이 같은 메커니즘을 가리킨다.
2. 가능한 targeted fix 방향(구현 안 함, 제안만):
   - Step 4-D(`worst_wedge_optimized`)를 `planar_hole_offcenter`류의 off-center-hole 씬에 한정해 재평가 — 이미 존재하는 옵션이라 재구현 불필요, A/B 재검증만 필요.
   - wedge별 "health gate": `inner_radius`가 도메인 스케일 대비 임계값 이하로 떨어지는 wedge를 감지해 seam 배치를 조정하거나(Step 4-D와 유사) 해당 wedge만 trimmed-rect fallback으로 격하하는 방안 — 새로운 메커니즘이라 별도 설계·승인 필요.
   - 이번 진단이 발견한 `holonomy_local_disagreement_count>0`(현재는 로깅만 되고 gate에 안 쓰임)을 향후 회귀 게이트의 정식 지표로 승격하는 것도 고려할 만하다(현재는 우연한 짝수 상쇄로 안 걸러짐).

## 검증

```powershell
# 코드 변경 없음 -- 기존 collect_diagnostic_samples=True(byte-identical opt-in, 기존 파라미터)만 사용.
# scratch 진단 스크립트(저장소에 커밋 안 함): step6b_diagnostic.py
```

`osn_gs/surface/*`, `nurbs_constructor_benchmark/*` 전부 이번 pass에서 미변경. 테스트 스위트 재실행 불필요(변경 없음).
