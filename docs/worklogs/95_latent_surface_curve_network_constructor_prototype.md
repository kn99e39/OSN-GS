# Worklog 95 — latent-surface curve-network constructor 프로토타입

## 상태

**완료 — 첫 end-to-end 구현. valid_supported가 0%(Worklog 89/94 baseline)에서 checkpoint 2900 2.6%, final 12.0%로 nonzero가 되고, unresolved가 84.7~88.0%에서 32.5~42.6%로 크게 줄었다. 다만 대다수 evidence는 여전히 extrapolative/unsafe다.** Visible Gaussian training, ADC, region ownership, 기존 NURBS fitter는 모두 미변경이다. 이번 배치는 새 diagnostic이 아니라 constructor 구현이며, Worklog 82/83/89의 raw Gaussian-center adjacency graph/chart-unit assembly/face-incidence topology를 이번 파이프라인에서 전혀 사용하지 않는다.

## 구현

Region-owned visible Gaussians → latent surface support → structural curve network → NURBS patch → held-out visible-evidence validation.

### 1. Latent surface support — `osn_gs/surface/torch_latent_surface_support.py`

Region-owned center를 표면 정점이 아니라 잡음 있는 관측으로 취급하는 position-based robust local estimator(moving-least-squares 스타일). Query 시 k=16 최근접 support point에 Gaussian-kernel weighted PCA를 적용해 local tangent frame(normal/tangent_u/tangent_v)을 얻고, query를 그 평면에 투영한 뒤 다시 이웃을 찾아 refit하는 2-iteration MLS 보정을 수행한다. Covariance는 어디에서도 읽지 않는다. `supported` 판정은 (a) 최근접 support point 거리가 local spacing의 3배 이내, (b) effective weighted neighbor 수가 3 이상, (c) local scatter의 planarity(중간/최소 고유값 비)가 1.5 이상 — 세 조건을 모두 만족해야 하며, 하나라도 실패하면 미지원으로 명시 보고한다(평면을 지어내지 않음). Gaussian position은 수정하지 않는다(입력 tensor 불변을 테스트로 확인).

### 2. Curve construction — `osn_gs/surface/torch_latent_surface_curve_network.py`

Dense point manifold 재구성이나 닫힌 관측 boundary를 요구하지 않는다. Seed curve는 이미 존재하는 Worklog 79/80의 sparse parametric chart boundary(`construct_region_parametric_chart_boundaries`, canonical construction pipeline이 이미 산출하는 미변경 결과)를 그대로 재사용한다 — 새 boundary reconstruction이 아니다.

- **Seed curve**: chart의 각 edge(대표점 a→b)를 고정 step 수(6)로 직선 보간하며 각 step을 latent surface에 투영한다. 어느 step이라도 미지원이면 그 edge를 그 지점에서 즉시 절단한다(gap을 잇지 않음).
- **Transversal curve**: 살아남은 seed 지점마다, seed 방향에 가장 직교하는 local tangent 축을 골라(centroid 방향으로 부호 고정) surface를 따라 고정 step 수(12)만큼 안쪽으로 걷는다. 매 step MLS 투영으로 보정하고, 미지원이면 그 지점에서 절단한다.
- **Interior rung curve**: 여러 transversal curve의 같은 step 깊이의 점들을 연결한다 — 이미 개별 검증된 점들만 묶으므로 새 fabrication이 없다.
- Convex hull, bounding box, PCA-rectangle, alpha-shape 등 어떤 gap-closing fallback도 없다. Chart가 `eligible_parametric_chart_boundary` 상태가 아니거나 seed walk가 즉시 실패하면 그 region은 curve network 없이 fail-closed된다.

### 3. NURBS

기존 `fit_torch_visible_surface_lsq`(Worklog 87의 `evaluate_fit`과 동일한 BASE_GRID=6, degree=2 규약)를 그대로 재사용한다. 입력은 curve-network sample point뿐이며, 모든 Gaussian center를 보간하도록 요구하지 않는다.

### 4. Validation

Worklog 87의 기존 PCA-UV checkerboard 분할(`_holdout`, HOLDOUT_K=4, 미변경)로 region evidence를 train/held로 나눈다. Latent surface estimator와 curve network 전체는 **train 절반만으로** 구성하고, held 절반은 어디에도 쓰이지 않다가 최종 NURBS fit 검증에만 쓰인다 — curve 구성에 직접 쓰이지 않은 evidence로 평가한다는 지시를 Worklog 87 자신의 held-out 규약만큼 엄격하게 satisfy한다. Classification(`valid_supported`/`extrapolative`/`unsafe_geometry`)은 Worklog 87과 동일한 EXTRAPOLATION_BOUND=4.0 등 기존 기준을 그대로 재사용해 baseline과 직접 비교 가능하다.

## 검증

`tests/test_latent_surface_support.py` 6개(surface 근접 query 지원·투영 정확도, 먼 query 미지원, normal-direction 이탈 query 미지원, 평평한 sheet의 normal 방향, 입력 불변, batch/single query 일치)와 `tests/test_latent_surface_curve_network.py` 5개(eligible chart의 정상 curve network 생성, ineligible chart의 fail-closed, 지지대역 밖 seed의 fail-closed, curve network의 모든 점이 개별적으로 재검증 통과, representative position 불변) 전부 통과. Worklog 79~94 관련 focused 112개 통과. 전체 회귀 **980 passed, 1 skipped**(256.0초).

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(3526 evidence), final(7774 evidence), cap=2048. 산출물: `output/extent_ab/val95/latent_surface_curve_network_prototype_replay.json`, `..._final.json`.

| checkpoint | 실측값 |
|---|---|
| 2900 | usable seed curves 5/7, curve network 4/7, materialized patches 4, **valid_supported 2.64%**, extrapolative 40.05%, unsafe 14.72%, unresolved 42.60%, held-out p95 7.96, region evidence unsupported by own latent surface(train half 기준) 59.53% |
| final | usable seed curves 5/7, curve network 4/7, materialized patches 4, **valid_supported 11.95%**, extrapolative 31.12%, unsafe 24.43%, unresolved 32.51%, held-out p95 7.76, region evidence unsupported by own latent surface(train half 기준) 55.98% |

### Worklog 89/94 baseline 대비

Worklog 94의 RAW_CENTER_BASELINE(Worklog 89와 동일 constructor)은 2900에서 valid_supported 0.000%/unresolved 87.98%, final에서 valid_supported 0.051%/unresolved 84.73%였다(Worklog 94 실측 재인용, 미변경). 이번 latent-surface curve-network 프로토타입은 **valid_supported를 checkpoint 전 구간에서 nonzero로 만들고**(2900: 0%→2.64%, final: 0.051%→11.95%) **unresolved를 절반 이하로 줄인다**(2900: 87.98%→42.60%, final: 84.73%→32.51%). 다만 unresolved 감소분의 대부분은 extrapolative(31.1~40.0%)와 unsafe(14.7~24.4%)로 흡수되고, valid_supported 자체는 여전히 대다수 evidence를 커버하지 못한다. 7개 region 중 3개(region 3, 4, 5)는 chart가 ineligible하거나 seed walk가 즉시 실패해 curve network를 만들지 못한다.

## 한계 (관측 사실, 후속 지시 아님)

- Train-only 절반으로 지은 latent surface에 대해 전체 evidence의 55.98~59.53%가 미지원으로 보고된다 — support_radius(local spacing의 3배)가 checkerboard-분할된 sparse train set 기준으로는 다소 타이트하다는 것을 실측이 보여준다.
- 7개 region 중 4개만 curve network를 만든다; region 3은 seed curve가 즉시 지지를 잃고, region 4/5는 애초에 eligible chart가 없다(Worklog 79/80 이후 그대로).
- Region 단위로 하나의 NURBS patch를 fit하므로 Worklog 89의 chart-unit 단위보다 훨씬 성긴 granularity다 — 그 자체가 valid_supported 증가에 기여했을 가능성이 있다(더 적은 patch 수, 더 큰 patch당 evidence).
