# Phase 4 Step 4-D: 현재 production 경계 추정기에서의 다중 씬·다중 seed 재평가

작성일: 2026-07-22

상태: **평가 완료; production 변경 없음; 현재 구현된 `worst_wedge_optimized`는 canonical 기본값 채택 대상이 아니다.** selector, 재시도 경로, trimmed fallback은 추가하지 않았다.

## 범위

현재 production Phase-2 추정기(`filter_boundary_leaf_eligibility=True`, `eligibility_gap_closing_cells=1`)를 고정했다. 네 annulus 씬에서 seed 0--4, 각 600 points 조건으로 `uniform_angle`과 기존 결정론적 `worst_wedge_optimized` layout을 비교했다. 모든 실행은 동일한 `Phase-2 boundary -> annulus-layout -> O-grid LSQ` 경로를 사용했다.

보고서는 wedge별 raw 12x12 Jacobian sample을 수집했다. 따라서 orientation-flip fraction과 outer-boundary flip은 aggregate 지표만으로 추정한 값이 아니다.

## 집계 결과 (5개 seed 평균, uniform -> optimized)

| 씬 | flips | outer flips | cond p95 / p99 | chamfer RMS | false-fill | 판정 |
|---|---:|---:|---:|---:|---:|---|
| planar_hole | 4.2 -> 5.8 | 0 -> 0 | 2.78/5.28 -> 2.79/4.63 | 0.00611 -> 0.00601 | 0.1766 -> 0.1852 | 혼재; outer flip은 없지만 flip/false-fill 증가 |
| planar_hole_offcenter | 56.6 -> 41.4 | 5.6 -> 4.0 | 6.33/21.21 -> 8.85/30.30 | 0.00675 -> 0.00646 | 0.4375 -> 0.4346 | 목표 씬의 총 flip/GT 지표는 개선됐지만 conditioning과 observed-boundary conformance는 회귀 |
| planar_hole_elliptical | 2.4 -> 3.6 | 0 -> 0 | 3.29/4.52 -> 3.33/4.53 | 0.00587 -> 0.00592 | 0.1806 -> 0.1909 | 회귀; 5개 seed 중 3개에서 false-fill 악화 |
| planar_hole_density_gradient | 32.6 -> 85.4 | 8.2 -> 19.0 | 6.64/33.33 -> 19.78/110.46 | 0.01440 -> 0.02249 | 0.4137 -> 0.4110 | 명백한 회귀 |

near-degenerate count는 모든 실행에서 0으로 유지됐다. 그러나 이것만으로 후보를 구제할 수는 없다. dense-gradient 씬은 flip fraction, p99, outer flip, geometry 모두에서 회귀했다.

## 필수 스트레스 사례 확인

### `planar_hole_offcenter`, seed 1

Uniform은 wedge 5의 outer-boundary flip 24개를 포함해 flip 218개를 보인다(해당 wedge Gaussian 145개). Optimized는 총 flip을 159개, outer flip을 20개로 줄이지만 wedge-wide failure를 **제거하지 못한다**. 폭 0.236 및 0.306 rad의 좁은 wedge가 새로 생긴다(명목 폭 0.785 rad). wedge 2는 Gaussian 5개뿐이지만 flip 72개와 outer flip 14개를, wedge 4는 Gaussian 11개와 flip 29개 및 outer flip 6개를 낸다. 두 넓은 wedge는 각각 1.588 rad이며 Gaussian 수는 271/209개다. 실패가 이동했을 뿐 해결되지 않았다.

### `planar_hole_offcenter`, seed 3

Optimizer는 uniform layout을 정확히 그대로 반환한다. 모든 폭은 0.785 rad이고, wedge별 Gaussian 수와 flip 18개도 모두 변하지 않는다. 따라서 현재 proxy는 이미 알려진 이 실패 seed에서 개선 방향을 찾지 못한다.

### production 추정기 변경 이후의 elliptical false-fill

기존 trade-off가 남아 있다. seed별 false-fill delta는 +0.0349, +0.0000, +0.0151, +0.0104, -0.0085이며, chamfer도 첫 네 seed에서 회귀하고 seed 4에서만 개선됐다. 단일 seed의 우연이 아니다.

## 레이아웃 / 지지 영역 진단

| 씬 | optimized angular-width 범위 (rad) | optimized wedge Gaussian-count 범위 |
|---|---:|---:|
| planar_hole | 0.589--0.942 | 46--110 |
| planar_hole_offcenter | 0.236--1.588 | 5--271 |
| planar_hole_elliptical | 0.672--1.002 | 58--109 |
| planar_hole_density_gradient | 0.236--3.604 | 4--559 |

현재 optimizer는 하한 폭 제약(`0.3 * nominal`)만 가지고 upper-width 또는 support-confidence 제약이 없다. density-gradient 씬에서는 하나의 wedge가 ring 대부분을 차지할 수 있으므로, worst-wedge aspect proxy가 낮아져도 post-fit Jacobian이 건강하다는 뜻이 되지 않는다.

Observed Phase-2 outer-boundary conformance도 일관되게 개선되지 않는다. 평균 outer symmetric chamfer는 네 씬 모두에서 각각 +0.0014, +0.0013, +0.0029, +0.0164 악화됐고 coverage도 네 씬 모두 감소했다. 현재 objective에 outer-loop 항이 없으므로 예상된 결과다.

## Phase-2 boundary conformance (5개 seed 평균, uniform -> optimized)

| 씬 | inner symmetric chamfer | inner coverage | outer symmetric chamfer | outer coverage |
|---|---:|---:|---:|---:|
| planar_hole | 0.0325 -> 0.0311 | 0.818 -> 0.836 | 0.1051 -> 0.1065 | 0.607 -> 0.595 |
| planar_hole_offcenter | 0.0193 -> 0.0224 | 0.970 -> 0.954 | 0.0725 -> 0.0738 | 0.677 -> 0.643 |
| planar_hole_elliptical | 0.0252 -> 0.0253 | 0.942 -> 0.901 | 0.0734 -> 0.0763 | 0.659 -> 0.623 |
| planar_hole_density_gradient | 0.0299 -> 0.0308 | 0.920 -> 1.000 | 0.1024 -> 0.1188 | 0.424 -> 0.395 |

현재 4-D objective는 outer-loop conformance objective가 아니다. Observed outer loop는 모든 집계 행에서 회귀하므로, 간헐적인 inner-loop 개선만으로 canonical 채택을 정당화할 수 없다.

## 현재 objective의 원인

`_optimize_worst_wedge_seam_angles()`는 인접한 두 wedge의 proxy 최대값만 최소화한다.

`kappa = max(radial_width, inner_radius * angular_width) / min(...)`.

`inner_radius`는 local raster/point window의 raw minimum으로 추정하고, endpoint window만 사용하며, cyclic partition의 나머지 폭까지 허용한다. 따라서 sparse/noisy minimum 하나가 proxy를 지배할 수 있고, 큰 angular 영역과 support-mass 불균형은 벌점이 없다.

## 정준 objective 수정 제안 (미구현)

경로는 하나만 유지한다.

`Phase-2 boundary -> deterministic constrained annulus-layout optimization -> O-grid NURBS fitting`.

다음 A/B pass 전에 현재 endpoint/min-only proxy를 deterministic observed-boundary profile objective로 교체한다.

1. 단일 support-cell minimum 대신 robust local quantile/interpolation을 사용해 명시적 Phase-2 hole/outer loop에서 inner/outer radial profile을 얻는다.
2. 두 seam window뿐 아니라 endpoint와 interior sample을 포함해 wedge 전체의 physical inner tangential width와 radial aspect를 평가한다.
3. component scale에 대한 최소 physical inner tangential width와 함께 cyclic feasibility constraint의 lower **및 upper** angular-width bound를 추가한다.
4. wedge Gaussian count/profile coverage는 scene selector나 별도 fitting path가 아니라 profile estimate의 confidence로만 사용한다.
5. Phase-2 geometry에서만 유도한 observed outer-loop conformance term을 추가한다. Runtime objective에는 GT chamfer나 GT false-fill을 쓰지 않으며, 이들은 benchmark acceptance metric으로 남긴다.

수정된 optimizer는 production 기본값을 바꾸기 전에 같은 4 scenes x 5 seeds protocol로 평가해 seed별 material regression이 없음을 보여야 한다.

## 검증

- 40회 construction 완료: 4 scenes x 5 seeds x 2 layouts.
- 이 pass에서 production source와 default는 변경하지 않았다.
- 평가 후 전체 unit suite: 133 passed, 1 skipped.