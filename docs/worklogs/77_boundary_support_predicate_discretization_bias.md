# Worklog 77: boundary-support predicate의 discretization bias 규명·수정

## 목적

Worklog 75가 normal source를, worklog 76이 connectivity scale을 병목에서 제거했다. 남은 후보인 **dense boundary-support predicate 자체**를 감사해, `observed_support_termination`이 관측된 perimeter를 실제로 조밀하게 materialize하지 못하는지 판정하고, 국소적으로 명확히 귀속되는 결함이 있으면 같은 배치에서 고친다.

covariance_normal, production connectivity scale, connectivity certificate, region formation, ownership, visible Gaussian 학습은 전부 미변경이다.

## 규명한 결함: 각도 gap 추정기의 discretization bias

`extract_dense_boundary_support`는 kNN을 tangent plane에 투영해 최대 각도 gap을 재고 `gap >= missing_sector_radians(=pi)`일 때만 candidate로 인정한다. 그런데 gap은 **양옆 이웃 "광선" 사이**로 측정되는 반면 이웃은 연속 표면의 점 표본이므로, 측정값은 국소 각도 표본 해상도만큼 **체계적으로 과소평가**된다.

직선 boundary의 참 empty sector는 **정확히 pi**다. 따라서 raw 측정값은 pi에 **아래에서** 수렴하고, 표본이 조밀해질수록 `gap >= pi`는 오히려 확실히 실패한다 — 즉 threshold가 엄격한 것이 아니라 **추정기가 점근적으로 편향**돼 있다.

`box_face`(9×9 격자, position noise 0.001) 실측이 이를 그대로 보여준다.

| 집합 | gap/pi min | median | max |
|---|---:|---:|---:|
| 참 boundary(32점) | 0.9868 | **0.9989** | 1.5018 |
| interior(49점) | 0.2505 | 0.2526 | 0.2591 |

참 boundary 32점 중 **25점이 pi의 1% 이내**에 몰려 있어 admission이 사실상 표본 잡음으로 결정되고 있었다. 반면 interior는 0.25pi 대로 **완전히 분리**돼 있어 판별력 자체는 충분하다.

### 수정

threshold는 pi 그대로 두고, **그 점 자신의 비최대 gap 중앙값(= 자기 국소 각도 표본 해상도)만큼 편향을 보정**한다.

```
resolution = median(비최대 gap)
admit  ⟺  gap >= missing_sector_radians - resolution
```

새 상수를 도입하지 않고 그 점 자신의 증거만 쓰며, **표본 밀도가 올라가면 보정항이 0으로 수렴해 정확히 `gap >= pi`로 되돌아간다** — 편향 보정이 갖춰야 할 성질이다. interior 점의 최대 gap(~2pi/k)은 `pi - resolution`에 한참 못 미치므로 precision은 영향받지 않는다.

**threshold 완화가 아니라는 증거: precision이 전 fixture에서 1.000으로 불변이고, 닫힌 manifold(sphere)는 여전히 candidate 0이다.** threshold를 낮췄다면 recall과 precision이 맞교환됐을 것이다.

## 합성 fixture 측정 (ground truth는 fixture 생성식에서 직접 유도, predicate 출력 무관)

`box_face`는 9×9 격자의 {행∈{0,8}}∪{열∈{0,8}}, `cylinder` side는 (24각×9높이) 격자에서 원주 방향은 닫혀 있고 축 방향만 열려 있으므로 {높이 index∈{0,8}}, cap은 disc mask 후 4-이웃 고리가 불완전한 점, `sphere`는 boundary가 없는 닫힌 manifold(정답은 candidate 0).

| region | 참 boundary | 수정 전 P/R | 수정 후 P/R | 연속 누락 run 최대 |
|---|---:|---|---|---:|
| box_face | 32 | 1.000 / 0.438 | **1.000 / 1.000** | 4 → **0** |
| cylinder side | 48 | 1.000 / 0.250 | **1.000 / 1.000** | 5 → **0** |
| cylinder bottom_cap | 16 | 1.000 / 0.500 | 1.000 / 0.750 | 2 → 1 |
| cylinder top_cap | 16 | 1.000 / 0.438 | 1.000 / 0.750 | 3 → 1 |
| sphere(닫힌 면) | 0 | candidate 0 | **candidate 0** | — |

**연속 누락 run**이 핵심이다 — 총량이 아니라 분포다. 수정 전에는 perimeter를 따라 **연속 4~5점이 통째로 비는 구간**이 있었고, 국소 certificate로는 그런 구멍을 결코 이을 수 없다. 수정 후 box_face/side는 누락 0이다. cap의 0.75는 cap ground-truth 자체가 근사(disc mask된 격자에서 4-이웃 미만)라 그대로 정직하게 남긴다.

## 합성 fixture: 수정 후 기존(미변경) connectivity 경로

| region | candidate | perimeter coverage | no_candidate | components | closed | crossing | containment |
|---|---:|---|---|---|---:|---:|---|
| box_face | 32 | **32/32 (100%)** | **0/64** | closed_loop 1 | **1** | 0 | **0.000** |
| cylinder side | 48 | 48/48 (100%) | 0/96 | closed_loop 2 | **2** | 0 | 0.516* |
| cylinder bottom_cap | 12 | — | 0/24 | closed_loop 1 | **1** | 0 | **0.000** |
| sphere | 0 | — | 0/0 | — | 0 | 0 | — |

`box_face`는 **worklog 69~77 통틀어 처음으로 interior evidence를 100% 포함하는(interior_outside_boundary 0.000) 유효 closed loop**을 만들었다. cylinder side가 loop 2개인 것도 정답이다 — 튜브는 위/아래 두 개의 독립 boundary ring을 가지며, 병합되지 않고 각각 보존됐다. (*side의 containment 0.516은 튜브 evidence를 하나의 ring에 대해 단일 UV 평면으로 평가한 데서 오는 위상적 산물이며 결함이 아니다.)

## Real baseline_compatible@2900 (7개 region, 3,526점)

모든 점에 정확히 하나의 terminal admission 결과를 부여했다. `degenerate_tangent_frame`은 **전 region 0건**, `insufficient_local_evidence`도 0건이므로 실질 분기는 admitted / insufficient_angular_gap 둘뿐이다.

| region | 점수 | admitted(수정 전) | gap 거부 | 경로상 거부점 존재 | **관측 증거 자체 부재** |
|---|---:|---|---:|---:|---:|
| r0 | 93 | 26 (20) | 67 | 4 | 22 |
| r1 | 519 | 152 (126) | 367 | 44 | 108 |
| r2 | 510 | 154 (134) | 356 | 48 | 106 |
| r3 | 92 | 29 (23) | 63 | 5 | 24 |
| r4 | 1035 | 266 (224) | 769 | 91 | 175 |
| r5 | 375 | 95 (77) | 280 | 20 | 75 |
| r6 | 902 | 218 (181) | 684 | 54 | 164 |
| **합계** | **3526** | **940 (785)** | **2586** | **266 (28.3%)** | **674 (71.7%)** |

수정으로 candidate가 **785 → 940 (+19.7%)** 늘었다. 그러나 **accepted candidate와 가장 가까운 accepted candidate 사이 경로 중 71.7%에는 관측된 점이 아예 존재하지 않는다** — predicate가 거부한 boundary 점이 놓여 있는 경우는 28.3%뿐이다. (이 진단에서 gap을 가로지르는 edge는 만들지 않았다.)

### 수정 후 기존(미변경) connectivity 경로 — real

| 지표 | 값 |
|---|---|
| candidate | 940 (수정 전 785) |
| `no_candidate` | 1161/1880 (61.8%) |
| components | 전 region `open_or_ambiguous`만 (branch 0) |
| **closed loop** | **0** (7개 region 전부) |
| proper crossing | **0** |
| 미지지 edge occupancy | 39/354 (11%)* |
| containment | 유효 loop 부재로 측정 대상 없음 |

(*이 occupancy는 component 내부 전체 쌍을 센 **상한**이며 accepted adjacency 집합이 아니다.)

## 함께 측정했으나 이번 배치에서 조치하지 않은 것

gap 거부 점의 **40~66%가 near-normal 이웃(‖tangential‖/‖delta‖ < 0.5, tangent 평면 투영이 방위를 담지 못하는 이웃)을 최소 1개 포함**한다(r4는 505/769). 합성 fixture에서는 이 수치가 **0**이다 — real 학습 Gaussian cloud가 국소적으로 volumetric·noisy하다는 worklog 75의 관찰과 일치한다. `atan2(0,0)=0`이 실재하지 않는 방향 지지를 만들어낼 수 있으므로 잠재적 2차 결함이다.

**그러나 이번 배치에서 고치지 않았다.** 합성 fixture에서는 이 현상이 0건이라 **ground truth로 검증할 수단이 전혀 없고**, real에는 정답 라벨이 없다. 검증 불가능한 2차 변경을 얹는 것은 이 과제가 금지한 "결과를 향한 조정"과 구분되지 않는다. 측정값만 정직하게 남기고, 검증 가능한 fixture가 확보될 때 별도로 다룬다.

## 아키텍처 질문에 대한 답

> 현재 visible-surface constructor는 관측된 boundary 점이 존재하는데 predicate가 거부해서 실패하는가, 아니면 관측 region에 연속적 boundary 증거가 실제로 없어서 실패하는가?

**둘 다이지만 지배적인 것은 후자다.**

- predicate에는 실재하고 국소적으로 귀속되는 결함이 있었다(discretization bias). 이를 고치자 깨끗한 기하에서 recall이 0.44→1.00, 0.25→1.00으로 올라가고 real candidate가 +19.7% 늘었다. **이 부분은 실제 결함이었고 실제로 고쳐졌다.**
- 그러나 수정 후에도 real의 perimeter 불연속 중 **71.7%는 관측 증거 자체의 부재**이며 predicate 거부가 아니다. real closed loop은 여전히 0이다.

**가장 결정적인 증거는 합성/real 대비다: 동일한 수정된 predicate가, 관측이 실제로 연속적인 경우(box_face)에는 완전한 perimeter(32/32) + 유효 closed loop + containment 0.000을 만들어낸다.** 즉 predicate는 이제 연속 관측이 주어지면 조밀한 perimeter를 materialize할 능력이 있음이 입증됐으므로, real에서의 실패는 predicate가 아니라 **증거**에 귀속된다.

증거가 실제로 없는 곳에서는 fail-closed를 유지했다 — sphere는 여전히 candidate 0이고, real은 hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·cross-region merge 없이 0 loop으로 남는다.

## 검증

신규 `tests/test_boundary_support_predicate_bias.py` 7개(직선 boundary 전량 admission, interior 미admission, 정확히 degenerate한 edge 중점, 잡음 하 안정성, 닫힌 manifold 0건, 빈 evidence 0건, 밀도 증가 시 interior 미유입) — **수정 전에는 직선 boundary 관련 2개가 실제로 실패하고 fail-closed 계열은 전부 통과**함을 stash 대조로 확인했다(완화가 아니라 편향 보정이라는 직접 증거). 변경 모듈을 소비하는 기존 테스트 포함 **49 passed**. 지시대로 focused tests만 실행했고 full pytest는 수행하지 않았다.
