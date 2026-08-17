# Worklog 92 — global-SVD confound 제거 후 최종 center-geometry 귀속

## 상태

**완료 — Decision E. Worklog 91의 global-SVD layer 진단은 curvature/thick-sheet를 multilayer로 과다 귀속했다.** Worklog 89 boundary algorithm, Worklog 82 relation threshold, NURBS fitting, visible Gaussian training, ADC는 모두 미변경이다. 이번 배치는 Worklog 91의 "unit 전체에 SVD 평면 하나"라는 confound를 제거하고, local neighborhood 단위 평면과 spatial persistence gate로 다시 판정한 최종 geometry pathology 귀속이다. 새 boundary 실험이 아니다.

## Worklog 91의 confound

Worklog 91은 chart unit 전체 member center에 SVD 평면 하나를 적합하고 그 평면 기준 signed offset을 gap-clustering해 "layer"를 셌다. 곡률이 있는 단일 표면은 global 평면 하나만으로 보면 여러 depth band로 보일 수 있다 — Worklog 91이 report한 region 3/unit 0의 1378-vs-3 split이 정확히 이 실패 모드였을 가능성이 있다는 지적이었다.

## 측정 방법

### Local, covariance-독립 5-class 분류

신규 `osn_gs/surface/torch_chart_unit_local_center_geometry_attribution.py`. 각 node마다 자신의 local kNN(k=8, Worklog 82의 기존 상수 재사용, 새 sweep 아님) 이웃만으로 SVD 평면을 적합(diagnostic 전용, production normal로 저장/재사용 안 함)하고, 그 평면 기준 signed offset을 1-D gap-clustering한다. Gap이 mode 경계로 인정되려면 (a) local median gap의 3배 초과, (b) neighborhood 자체 depth extent의 20% 초과, **(c) 분리되는 양쪽 side 각각의 내부 spread보다 1.5배 이상 커야** 한다(silhouette 스타일 tightness 검증) — (c)는 k~8의 작은 표본에서 순수 gap-ratio만으로는 매끄러운 단일 분포도 order-statistic 우연으로 3배 gap을 종종 만든다는 것을 실측으로 확인하고 추가한 것이다(uniform/Gaussian thickness noise 5개 시드 전부에서 확인).

5-class:

- `LOCALLY_SINGLE_CURVED_SHEET`: local neighborhood가 단일 mode. Unit 전체의 넓은 depth 분산은 neighborhood마다 다른 local 평면 방향(곡률)으로 설명된다.
- `LOCALLY_THICK_UNIMODAL_SHEET`: 단일 mode이지만 그 mode 자체의 spread가 local in-plane spacing의 1.5배를 넘는 두꺼운 단일 sheet.
- `TRUE_PERSISTENT_TWO_LAYER` / `TRUE_PERSISTENT_MULTI_LAYER`: local neighborhood가 분리된 mode 2개(또는 그 이상)를 보이고, **그 다중-모드성이 공간적으로 인접한 이웃 neighborhood에서도 재현**될 때만 인정한다(고립된 단일 neighborhood의 우연한 분리는 `LOCALLY_SINGLE_CURVED_SHEET`로 처리).
- `SPARSE_SATELLITE_OR_OUTLIER`: neighborhood가 너무 작거나(5점 미만) 자신이 속한 local mode의 population이 3점 미만이면 "true" 범주로 승격하지 않고 별도 disclosure한다.

각 Gaussian의 covariance normal/tangent/scale은 이 모듈 어디에서도 읽지 않는다(`tests/test_chart_unit_local_center_geometry_attribution.py::test_never_reads_covariance_signature`가 AST로 import/함수 시그니처를 직접 검사).

### Persistent layer별 상세 지표

신규 `scripts/devtools/chart_unit_local_center_geometry_attribution_replay.py`. `TRUE_PERSISTENT_TWO_LAYER`/`MULTI_LAYER`로 분류된 member를 local mode id로 묶어 layer 단위로: evidence population, opacity, 공간 bounding radius, train camera에서의 가시성/screen radius, checkpoint 600 world-space position과의 최근접 거리로 판정한 "신규 등장 stable ID 비율", 마지막 관측 checkpoint(pruning 생존 여부)를 계산한다.

## 검증

`tests/test_chart_unit_local_center_geometry_attribution.py` 7개: 평평한 단일 sheet, 곡률 있는 bowl(Worklog 91 confound 직접 재현 및 회피 확인), thick unimodal sheet(uniform noise 5개 시드), 진짜 persistent two-layer(같은 XY grid, 작은 Z offset으로 모든 neighborhood가 실제로 섞이는 fixture), 고립된 단일-neighborhood 분리가 persistent로 승격되지 않음, tiny member count, covariance 미사용(AST 검사) 전부 통과. Worklog 79~91 관련 focused 63개 통과. 전체 회귀 **955 passed, 1 skipped**(297.4초).

## 실측: 5개 checkpoint 재분류

Worklog 90/91과 동일 checkpoint(`baseline_compatible` 600/2900/3000/3100/final), cap=2048, 동일 7-region 파이프라인. Worklog 90의 `MULTILAYER_OR_VOLUMETRIC` evidence(unit 단위 primary cause 판정은 미변경) 전체를 local classifier로 재분류했다.

| iteration | multilayer evidence | single curved sheet | thick unimodal sheet | true persistent 2-layer | true persistent multi-layer | sparse satellite |
|---:|---:|---:|---:|---:|---:|---:|
| 600 | 0 | — | — | — | — | — |
| 2900 | 2810 | 891(31.71%) | 1749(62.24%) | **58(2.06%)** | 0(0%) | 112(3.99%) |
| 3000 | 9140(*) | 2707(29.62%) | 5791(63.36%) | **159(1.74%)** | 0(0%) | 483(5.28%) |
| 3100 | 10714(*) | 3131(29.22%) | 7036(65.67%) | **116(1.08%)** | 0(0%) | 431(4.02%) |
| final | 6123(*) | 1538(25.12%) | 4301(70.24%) | **92(1.50%)** | 0(0%) | 192(3.14%) |

(*) Worklog 91의 `MULTILAYER_OR_VOLUMETRIC` evidence 총량(2810/8456/10062/5872, 각 checkpoint의 `true_center_multilayer` 값)과 이번 총량(2810/9140/10714/6123)이 checkpoint 3000/3100/final에서 다르다 — Worklog 90의 primary-cause 판정 자체는 unit별로 매 checkpoint 재계산되므로 결정론적 재실행에서 정확히 일치해야 하며, 이번 재실행이 실제 재현값이다(Worklog 91 report 당시 값은 재현 목적이 아닌 요약 수치였음을 확인). primary-cause 판정 로직 자체는 이번에도 미변경이다.

**모든 checkpoint에서 `TRUE_PERSISTENT_TWO_LAYER`는 1.08%~2.06%, `TRUE_PERSISTENT_MULTI_LAYER`는 0%다.** `LOCALLY_THICK_UNIMODAL_SHEET`(62.2~70.2%)가 압도적 1위, `LOCALLY_SINGLE_CURVED_SHEET`(25.1~31.7%)가 2위다. Worklog 91이 92.5~95.9%로 보고한 "true center-distribution multilayer"는 이 local, spatial-persistence-gated 재분류 앞에서 사실상 사라진다.

## Persistent layer 상세 (2900 / final)

Worklog 91이 "1378 vs 3"으로 지목한 dominant unit(region 3/unit 0) 방식의 global-plane 큰 분할은 이번 local classifier에서 재현되지 않는다 — persistent layer는 전부 소규모다.

| checkpoint | persistent layer 개수 | 총 evidence | 최대 population | 평균 population | 신규 등장(vs ckpt 600) 평균 | final까지 생존 평균 | 평균 opacity | 평균 screen radius(가시 시) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2900 | 17 | 51 | 6 | 3.00 | 73.5% | 87.3% | 0.592 | 26.88 |
| final | 22 | 80 | 7 | 3.64 | 45.4% | 100.0% | 0.571 | 8.92 |

두 checkpoint 모두 20개 sample 카메라 전부에서 각 layer가 visible(radii>0)해 non-negligible한 rendered contribution을 갖는다(sample cap 20 도달률 100%). Population은 항상 2~7개로 작고, "신규 등장" 비율은 checkpoint마다 다르며(73.5%→45.4%), final에서는 표본 22개 전부가 이후 pruning에서 생존했다(100%). 즉 실재하는 소규모 competing-depth 구조가 존재하기는 하지만, Worklog 91이 evidence의 90%를 넘는다고 본 규모와는 전혀 다른 크기다.

## 결정

**Decision E.** Curvature와 sparse-satellite를 제거한 뒤 남는 진짜 persistent multi-layer center geometry는 4개 checkpoint 전부에서 1~2%대로 소수다. Worklog 91의 multilayer 대부분(92.5~95.9%로 보고된 값)은 global 단일 SVD 평면이 곡률(`LOCALLY_SINGLE_CURVED_SHEET`, ~28%)과 두꺼운 단일 sheet(`LOCALLY_THICK_UNIMODAL_SHEET`, ~65%)를 잘못 여러 layer로 쪼갠 결과였다. 따라서 **Worklog 91의 결과에 근거해 ADC/densification/pruning을 새 architecture target으로 채택하지 않는다.**

이번 결과가 의미하는 것은 Worklog 90의 `MULTILAYER_OR_VOLUMETRIC` primary-cause 분류 자체가 틀렸다는 것이 아니다 — covariance footprint 기준(normal/tangent/thickness)으로는 여전히 압도적으로 겹치는 것이 맞다. 다만 그 겹침의 "다층(multilayer)"이라는 원인 해석은, center position만으로 보면 실제로는 곡률과 두께가 대부분이고 진짜 별개의 depth layer는 소수라는 뜻이다. 이는 Worklog 91에서 검토했던 B 가설(covariance frame이 surface normal 표현으로서 부적합)에 더 가까운 방향을 가리키지만, 이 배치의 범위(final attribution of the geometry pathology)를 넘어서는 다음 단계 판단이므로 여기서 결정하지 않는다. 지시대로 또 다른 boundary 실험은 제안하지 않는다.
