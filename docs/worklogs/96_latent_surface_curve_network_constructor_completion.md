# Worklog 96 — latent-surface curve-network constructor 아키텍처 완성

## 상태

**완료 — construction 측면(legacy gate 제거, 다중 patch, continuous support, family correspondence)은 성공. 그러나 safe NURBS(valid_supported)는 Worklog 95 대비 개선되지 않았다.** Curve-network 구성이 널리 가능해졌지만 safe NURBS는 여전히 부족하므로, 지시된 decision tree에 따라 **더 이상 curve-seeding heuristic을 추가하지 않고 downstream parametric fitting/patch representation이 현재의 병목 architecture임을 보고한다.** Visible Gaussian training, ADC, region ownership, 기존 NURBS fitter는 모두 미변경이다.

## 구현

Worklog 95의 legacy `eligible_parametric_chart_boundary` entrance gate를 제거하고 4개 architecture 한계를 함께 해결했다.

### 1. Legacy entrance gate 제거 — `torch_latent_surface_seed_curves.py`

Boundary/typed seed(physical_boundary/crease_feature/observation_frontier, 기존 chart segment_kind 그대로 매핑)를 존재하면 우선·보존하고, 존재하지 않을 때만 interior fallback을 쓴다. Interior seed는 region evidence에서 결정론적 farthest-point sampling으로 뽑은 소수 anchor(6개, 고정)를 latent surface에 개별 검증(지지 확인)한 뒤 시작점으로만 쓴다 — anchor 간에는 raw Gaussian-center connectivity를 전혀 만들지 않는다. Convex hull/PCA rectangle/bounding box/alpha shape/forced closure는 없다.

### 2. Latent surface 상의 curve tracing — `torch_latent_surface_curve_tracer.py`

`propagate_tangent_onto_plane`으로 이전 방향을 새 local tangent plane에 투영해 이어가고(parallel-transport 스타일), 실제 이동량으로 다음 방향을 재접지(ground)한다 — 매 step마다 PCA axis 부호를 다시 고르지 않는다. 모든 step은 latent surface 추정기로 검증되며 미지원 즉시 종료한다.

### 3. 연속 지지 요구 — 동일 모듈의 `sample_segment_continuous_support`

Rung(연결 segment)은 양 끝점이 지지된다고 자동 승인되지 않는다 — 중간 sample을 densify해 전부 지지될 때만 승인한다.

### 4. Curve family/correspondence — `torch_latent_surface_curve_families.py`

Seed마다 family V(transversal curve, seed curve의 sample들에서 표면을 따라 안쪽으로 추적)와 family U(연속 지지 검증된 rung, 인접한 transversal curve 쌍을 depth별로 연결)를 만든다. 고정 계약(이번 배치에서 실측으로 튜닝하지 않음): family당 curve 2개 이상, 서로 다른 depth 2개 이상으로 연결된 2×2 상호 정합 correspondence 1개 이상. Depth/sample index는 생성 순서상 항상 단조 증가이므로 순서 요건은 구성 자체로 만족한다.

### 5. Region당 다중 patch — 동일 모듈

Seed 하나당 block 하나이므로 한 region은 만족하는 seed 수만큼 독립 block을 만들 수 있다. Block 판정은 NURBS fit 이전에 순수 구조(연속 지지 rung 연결성)로만 결정되며, fit 오차나 held-out 오차로 block을 나누거나 재시도하지 않는다(테스트로 확인: 모듈이 NURBS를 import하지 않음).

### 6. Curve 타입 구분

`SeedCurve.seed_type`이 physical_boundary/crease_feature/observation_frontier/interior_construction을 끝까지 보존한다. Interior curve는 physical termination의 증거로 취급되지 않는다.

### 7/8. NURBS + validation

기존 `fit_torch_visible_surface_lsq`(BASE_GRID=6/degree=2)와 Worklog 95의 `evaluate_curve_network_fit`(EXTRAPOLATION_BOUND=4.0)을 그대로 재사용한다. 두 조건을 분리 보고한다: **A. ALL_VISIBLE_EVIDENCE_CONSTRUCTION**(전체 region evidence로 support/seed/block 구성, production 능력 측정)과 **B. HELD_OUT_VALIDATION**(Worklog 87 checkerboard `_holdout`으로 train 절반만 구성에 사용, held 절반으로만 평가, Worklog 95의 support threshold를 그대로 유지 — sparse train 때문에 완화하지 않음).

## 검증

신규 focused 테스트 19개: `test_latent_surface_curve_tracer.py` 7개(tangent 부호 유지, 이탈 시 종료, 양 끝만 지지된 segment의 중간 미지원 거부, 완전 지지 segment 승인), `test_latent_surface_seed_curves.py` 6개(boundary 보존, interior fallback 2 경로, boundary 우선 시 interior 미혼입, interior anchor 독립성, 입력 불변), `test_latent_surface_curve_families.py` 6개(boundary/interior 모두 다중 block 생성, 계약 충족 검증, 모든 rung 개별 재검증 통과, 모듈이 NURBS를 import하지 않음으로 fit-driven split 부재 확인, 고립 seed의 계약 미충족). Worklog 79~95 관련 focused 131개 통과. 전체 회귀 **999 passed, 1 skipped**(255.0초).

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(3526 evidence), final(7774 evidence), cap=2048. 산출물: `output/extent_ab/val96/latent_surface_curve_network_v2_replay.json`, `..._final.json`.

### A. ALL_VISIBLE_EVIDENCE_CONSTRUCTION

| checkpoint | usable seed | valid curve network | coherent block | 다중 patch region | 이전 gate로 막혔다가 이번에 구성됨 | seed 구성 |
|---|---:|---:|---:|---:|---:|---|
| 2900 | 7/7 | 5/7 | 9 | 3 | 1 | crease 6, boundary 2, interior 4 |
| final | 6/7 | 6/7 | 9 | 2 | 1 | crease 6, boundary 4, frontier 1, interior 1 |

Worklog 95 대비 usable seed(5/7→7/7, 6/7), valid curve network(4/7→5/7, 6/7)가 늘고, 두 checkpoint 모두 legacy gate로 막혔던 region 1개가 interior fallback으로 새로 구성에 성공한다.

### B. HELD_OUT_VALIDATION (evidence-weighted, Worklog 87/89/94/95와 동일 규약)

| checkpoint | valid_supported | extrapolative | unsafe_geometry | unresolved | held-out p95 |
|---|---:|---:|---:|---:|---:|
| 2900 | 2.72% | 75.20% | 1.78% | 20.31% | 7.22 |
| final | 9.26% | 42.90% | 14.17% | 33.67% | 8.24 |

### Worklog 89/94/95 대비

| checkpoint | 지표 | Worklog 89/94(raw baseline) | Worklog 95(single-patch) | Worklog 96(multi-patch, 이번) |
|---|---|---:|---:|---:|
| 2900 | valid_supported | 0.000% | 2.64% | **2.72%** |
| 2900 | unresolved | 87.98% | 42.60% | **20.31%** |
| 2900 | extrapolative+unsafe | 0.17% | 54.77% | **76.98%** |
| final | valid_supported | 0.051% | 11.95% | **9.26%** |
| final | unresolved | 84.73% | 32.51% | **33.67%** |
| final | extrapolative+unsafe | 0.24% | 55.55% | **57.07%** |

Raw-center baseline 대비 valid_supported는 두 checkpoint 모두 여전히 압도적으로 높다(54~180배). Unresolved는 2900에서 크게 줄었다(42.6%→20.3%). 그러나 **valid_supported 자체는 Worklog 95의 single-patch 결과를 넘지 못했고**(2900은 거의 그대로, final은 오히려 소폭 하락), **extrapolative+unsafe 합계는 두 checkpoint 모두 Worklog 95보다 늘었다**(2900: 54.8%→77.0%, final: 55.6%→57.1%). 새로 구성에 성공한 region(예: 이전에 no_eligible_seed_chart였던 region)의 patch가 대부분 extrapolative로 분류되면서, unresolved 감소분이 valid_supported가 아니라 extrapolative로 흡수된다.

## 결정

**Condition 1~3(legacy gate 미의존, 연속 지지, multi-patch 구조)은 충족된다.** Condition 4(valid_supported가 raw baseline보다 material하게 높음)도 raw baseline 대비로는 충족되지만 Worklog 95 대비로는 개선이 아니다. **Condition 5(extrapolative/unsafe가 더 공격적인 구성만으로 단순 증가하지 않아야 함)는 충족되지 않는다** — 두 checkpoint 모두 extrapolative+unsafe 합계가 Worklog 95보다 늘었다.

지시된 decision tree에 따라: **curve-network 구성은 이제 널리 가능해졌지만(legacy gate 제거·multi-patch·interior fallback 모두 실측으로 동작 확인), safe NURBS(valid_supported)는 개선되지 않았다.** 따라서 **더 이상 새 curve-seeding heuristic을 추가하지 않는다.** 병목은 curve construction이 아니라 **downstream parametric fitting/patch representation**(PCA-UV 6×6 NURBS가 latent-surface curve-network 표본을 안전하게 소화하지 못하는 것)으로 이동했다고 보고한다. 새 constructor를 production architecture로 지명하지 않으며, 또 다른 고립된 seed-rule/threshold 진단으로 잇지 않는다.
