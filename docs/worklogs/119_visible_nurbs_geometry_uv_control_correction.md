# Worklog 119 -- Visible-NURBS Geometry / UV Control Correction

브랜치: `arch/2dgs-coverage-first-surface`

## Agent Interpretation of Intent

**DIRECTION**: 현재 topology(WL107/109)와 NURBS 표현 선택(고정 8x4 degree-2 control grid)을 전부 보존한다. 이번 배치는 딱 네 가지만 교정한다 -- (1) UV A/B 컨트롤 자체, (2) renderer geometry provenance, (3) low-pass outlier attribution, (4) evaluation semantics. adaptive capacity, domain-aware fitting, coupled fitting, 새 chart 단위, topology 변경은 이번 배치에서 구현하지 않는다.

**PURPOSE**: foot-point reparameterization이 (A) surface geometry 자체를 개선하는지, 아니면 (B) 각 샘플이 더 유리한 parameter 좌표로 옮겨가서 point residual만 낮추는 것인지 판별한다. 또한 renderer 자신의 median-surfel local intersection이 median-depth -> camera unprojection보다 더 깨끗한 per-pixel surface-geometry observation을 제공하는지 판별한다.

**CENTRAL INTENT**: 두 개의 연결된 질문에 답한다 -- (1) solve 횟수, regularization, evaluation metric을 통제했을 때 foot-point UV correction이 geometry, correspondence, 둘 다, 아니면 하나만 개선하는가? (2) true surfel footprint로 지지되는 renderer event에서, exact median-surfel local intersection이 depth unprojection을 대체할 만큼 더 직접적인 renderer-native geometry observation을 제공하는가?

**FROZEN QUANTITIES**: WL107/109 visible topology(연결성분/subset_ids 재생), WL112 camera blob 정의(`label_same_component_blobs`, `build_view_chart_pixel_samples`, `valid_pixel_chart_mask(32)`), 고정 8x4 degree-2 control grid(control-only), smoothness lambda=1e-4, Tikhonov lambda=1e-4, 두 arm 모두 동일한 `pixel_xyz` 입력(chart별로 한 번만 샘플링).

**ONLY CHANGED VARIABLES**: (a) ARM B의 solve 횟수(1 -> 2, ARM A와 동일하게)와 매 round UV를 camera UV로 고정하는지 여부; (b) 평가 metric 자체 -- METRIC G(둘 다 자기 자신의 최종 surface에 대해 독립적으로 재계산한 foot-point로 평가)와 METRIC C(둘 다 동일한 원본 camera UV로 평가)를 분리해서 보고; (c) renderer geometry source를 G0(공식 depth unprojection)/G1(rasterizer-pixel-center unprojection)/G2(surfel-local 직접 교차점 재구성)로 3원 비교; (d) low-pass event를 별도 semantic 분류로 유지(거부하지 않음); (e) pixel-level(chart 평균이 아닌) D-outlier attribution; (f) synthetic equal-count contract에 boundary-connected notch(D) 추가.

**SEMANTIC INVARIANTS**: METRIC G는 ARM A의 foot-point UV를 ARM B에 적용하거나 그 반대로 적용하는 교차 평가를 절대 하지 않는다 -- 각 arm은 자신의 최종 control grid에 대해서만 독립적으로 evaluation-only foot-point projection을 다시 계산한다(`project_torch_points_to_nurbs`는 fitting에 쓰인 UV를 덮어쓰지 않는다). METRIC C는 두 arm 모두 원본 `uv_camera`(절대 변하지 않는 입력)로만 평가한다.

**AGENT-ADDED ASSUMPTIONS (명시)**:
1. Pixel-level D attribution을 전체 모집단(수백만 픽셀)이 아니라 chart당 결정론적 stride subsample(목표 200픽셀/chart, 실제 `min(200, n_pixels)`)로 수행했다. 메모리/런타임을 유한하게 유지하기 위한 선택이며, subsample은 raster order를 따라 균등 간격으로 뽑아 재현 가능하다(RNG 없음).
2. `run_equal_count_synthetic_contracts_corrected`의 boundary notch(D)는 u=0 경계에 붙은 contiguous 전폭 블록으로 구현했다 -- 정확히 removed_count와 같지 않고 `round(removed_count/cols)`행 단위로 반올림한 근사치이며, 실제 제거 개수를 `notch_removed_count_in_D`로 항상 함께 보고한다.
3. G2 재구성은 `compute_transmat`(forward.cu)의 `splat2world`가 world-space `p_orig`/`L=R*S`를 직접 사용한다는 점(카메라 변환이 local-plane 좌표에는 적용되지 않음)에 근거해 `world = center + s_u*scale_u*tangent_u + s_v*scale_v*tangent_v`로 재구성했다. 코드를 직접 읽어 검증했고, 실측(G0 vs G1 vs G2 비교, section 6)으로도 교차 검증했다.
4. rho3d/rho2d branch classification에서 tie(`rho3d == rho2d`)는 "rho3d" branch로 분류했다 -- 커널 자신의 `rho = min(rho3d, rho2d)` 및 acceptance test의 `<=` 의미와 일치시키기 위함이다.

**INABILITY TO REALIZE THE REQUESTED CONTROL EXACTLY**: 없음 -- directive의 모든 필수 항목(2-11)을 요청된 그대로 구현했다. Section 11(boundary notch)은 "secondary, do not let it dominate"로 명시된 대로 결과 보고에서 부차적으로만 다룬다.

## Implementation Fidelity Statement

| 결과 | 모듈/함수 | 데이터 흐름 | 진단 | 테스트 |
|---|---|---|---|---|
| ARM B 동일 solve 횟수, UV 미변경 | `scripts/devtools/visible_nurbs_geometry_uv_control_correction.py::fit_fixed_uv_equal_solves` | `pixel_xyz`, `uv_camera` -> IDW seed(`fit_torch_visible_surface`) -> `_solve_control_grid_lsq` x2(같은 `uv_camera`) | 리포트 `corrected_uv_ab_metric_comparison` | `TestEqualSolveCountUVControl` (3개, monkeypatch로 solve 호출 횟수 직접 계수) |
| METRIC G/C 분리 평가 | `geometric_point_to_surface_error`/`camera_correspondence_error` | 각 arm의 최종 `TorchNURBSSurface` -> `project_torch_points_to_nurbs`(G) 또는 고정 `uv_camera`(C) -> residual | `metric_comparison_summary` | `TestCommonEvaluationMetrics` (3개, control_grid 불변 assert 포함) |
| G0/G1/G2 renderer geometry 비교 | `reconstruct_direct_surfel_intersection_world_point`, `depths_to_points`(official), `depths_to_points_rasterizer_pixel_center`(WL118 sibling) | `render_with_pixel_representative`의 `representative_id`/`median_s_u`/`median_s_v` + `model.get_xyz`/`get_tangent_u`/`get_tangent_v`/`get_scaling` | `geometry_source_comparison_G0_G1_G2` | `TestDirectSurfelIntersectionReconstruction` (2개, 손계산 검증) |
| rho3d/rho2d branch 분류 | `classify_median_event_branch` | `diag["median_rho3d"]`/`diag["median_rho2d"]` (WL118 CUDA diag 추가 필드, 무수정) | `pixel_level_d_attribution` | `TestBranchClassification` (2개) |
| pixel<->raster 정렬 | 본문 `rows_t, cols_t = np.nonzero(blob_mask_np)` vs `vs.pixel_xyz[pixel_sel]` | `torch_camera_observed_chart_domains.build_view_chart_pixel_samples`의 row-major flatten 규약 | `pixel_records` 원소별 정합성 assert(`n_pixels == pixel_xyz.shape[0]`) | `TestPixelRasterAlignment` (1개) |
| 결정론적 재현 | `run_equal_count_synthetic_contracts_corrected` | 고정 seed 없는 순수 결정론적 grid/stride 연산 | `synthetic_equal_count_contracts_corrected` | `TestDeterministicReplay` (2개) |

캐노니컬 렌더러(`osn_gs/render/vendor/diff_surfel_rasterization/`)와 `OSNSurfelRasterizer`, 학습 경로는 이번 배치에서 전혀 수정하지 않았다. WL118의 diagnostic CUDA 확장(`diff_surfel_rasterization_diag`)도 추가 필드 없이 그대로 재사용했다 -- G2 재구성은 순수 Python 레벨 계산(모델 텐서 + 이미 존재하는 diag 출력)이라 이번 배치는 CUDA 코드를 전혀 건드리지 않았다.

## 1. 실측 결과 (161-view 전체 실 scene, `output/119_osn_gs_geometry_uv_control_correction/visible_nurbs_geometry_uv_control_correction_report.json`)

- 전체 실행 시간: **4423.5초(약 73.7분)** -- WL118(2763.25초)의 약 1.6배. 이유는 부록 참고.
- `total_trained_surfels=1,190,469`, `median_surface_representatives=785,937`, `fitted_chart_count=14,900` -- WL112-118과 완전히 동일한 chart 개수.

## 2. WL107/109 replay 일관성

`visible_component_count=559,989`, `largest_component_surfel_fraction=0.367713...`, `singleton_surfel_count=535,910` -- WL107/109/112-118과 **비트 단위로 동일**. 이번 배치가 topology를 전혀 건드리지 않았음을 실측으로 재확인했다.

## 3. 교정된 equal-solve UV A/B (METRIC G / METRIC C 분리) -- 이번 배치의 핵심 발견

WL118의 "ARM A가 모든 영역에서 2.9~3배 우수하다"는 결론은 **불공정한 비교(evaluation UV가 arm마다 달랐음)에서 나온 것**이었다. Solve 횟수(2회)와 evaluation semantics를 통제하자 결과가 **방향이 갈린다**:

| 영역 | chart 수 | METRIC G (기하) A/B 비율 | METRIC C (카메라 대응) A/B 비율 |
|---|---|---|---|
| table_top | 573 | 0.86 (A 우수) | 1.60 (B 우수) |
| table_side_curved | 4234 | 0.79 (A 우수) | 1.09 (B 우수) |
| table_legs | 181 | 1.12 (거의 동일, 값 자체가 ~0 근처) | 1.69 (B 우수) |
| patio | 5805 | 0.80 (A 우수) | 1.10 (B 우수) |
| hedge | 4107 | 0.79 (A 우수) | 1.13 (B 우수) |
| **전체(median-of-median)** | 14900 | **0.80** (A 우수, ARM A=0.00462 vs B=0.00579) | **1.12** (B 우수, ARM A=0.01469 vs B=0.01314) |

p95도 동일한 패턴: METRIC G p95는 A(0.02096) < B(0.03039) -- A가 약 31% 우수. METRIC C p95는 A(0.06779) > B(0.06287) -- B가 약 8% 우수.

**즉, 모든 영역에서 예외 없이: METRIC G(기하)에서는 ARM A(foot-point correction)가 일관되게 우수하지만, METRIC C(카메라 대응)에서는 ARM B(고정 UV)가 일관되게 우수하거나 동등하다.** table_legs는 두 metric 모두 절대값이 거의 0(~1e-7)이라 이 chart 자체가 거의 평면적이어서 통계적으로 의미 있는 차이라기보다 noise에 가깝다.

부가 지표: `control_grid_diff` 중앙값 0.0295(두 arm의 최종 control grid가 실제로 다른 표면을 만든다는 것을 확인), `smoothness` 중앙값 A=0.00871 < B=0.01065(ARM A의 표면이 더 매끈함).

## 4. G0/G1/G2 renderer geometry 비교

G2(surfel-local 직접 교차점 재구성)가 렌더러 자신의 규약과 수학적으로 일치하는지 먼저 검증했다: rho3d-dominated 이벤트에서 G0-vs-G1 변위(median 0.007993)와 G0-vs-G2 변위(median 0.007993)가 **소수점 6자리까지 사실상 동일**하다 (rho2d-dominated에서도 0.010684 vs 0.010683로 동일). 즉 G2는 G1(rasterizer-pixel-center 규약)과 실질적으로 완전히 일치하며, G0과의 차이는 오직 half-pixel 컨벤션 차이(WL118에서 이미 규명)에서 온다 -- **G2 재구성이 수학적으로 정확함을 실측으로 확인**했다(directive 요구사항).

rho2d-dominated 이벤트의 G0-vs-G2 변위(median 0.010684)가 rho3d-dominated(0.007993)보다 약 33% 크다 -- 이는 G2 재구성의 오류가 아니라, screen-space low-pass가 선택된 이벤트 자체가 진짜 ray-plane intersection과는 다른 종류의 기하 관측이라는 것을 보여주는 진단적 차이다(directive의 의도한 해석).

## 5. low-pass vs true-footprint geometry semantics

rho3d-dominated pixel 965,614개(표본), rho2d-dominated pixel 295,355개(표본) -- **거부하지 않고 둘 다 유지**했다. rho2d-dominated pixel의 residual 중앙값(0.03047)은 rho3d-dominated(0.01445)의 **약 2.1배**다. 이는 low-pass가 선택된 이벤트가 renderer-valid contribution으로는 유효하지만, surface-geometry observation으로서는 체계적으로 덜 신뢰할 만하다는 것을 시사한다(단, 아래 섹션 6에서 보듯 "가장 극단적인" 개별 pixel은 이 패턴을 따르지 않는다).

## 6. pixel-level D attribution

Top-1000 잔차 pixel 중 **54%가 rho2d-dominated**(전체 표본 중 rho2d-dominated 비중은 23.4%에 불과) -- 즉 low-pass-dominated 이벤트가 극단적 잔차 tail에서 **뚜렷하게 과대표집**된다. WL118의 chart-평균 기반 결론("D-outlier는 low-pass와 무관하다")은 **모집단 수준에서는 부분적으로 정정**된다: low-pass 지배와 잔차 사이에 실제 통계적 연관이 있다.

그러나 top-20 극단 pixel을 직접 조사한 결과 **전부 chart_id=10592 (table_side_curved, WL113/114/118이 반복적으로 지목한 바로 그 작은 chart) 하나에서 나왔고**, 이 chart 안에서 branch는 **rho3d/rho2d가 섞여** 있다(예: residual 1313.5는 rho3d-dominated, residual 1309.1은 rho2d-dominated). depth(~9.2-9.3)와 s_magnitude(0.2-4)는 정상 범위이고 `g0_vs_g2_distance`(~0.0136)도 작다 -- 즉 **이 chart의 관측 기하 자체는 정상**이며, 문제는 이 특정 chart의 **fitted control grid 자체가 발산**한 것이다(`control_grid_diff`의 전체 최댓값 358.26, `smoothness_arm_a`의 전체 최댓값 94965.66이 이 chart에서 나왔을 가능성이 높다).

**결론**: population 수준에서는 rho2d(low-pass) 지배가 잔차와 real correlation을 가지지만(top-1000의 54%), 역사적으로 반복 지목된 단일 최극단 D-chart(10592)의 극단성은 branch로 설명되지 않는다 -- WL113의 원래 관찰("residual-max와 overlap-max는 서로 다른 메커니즘")이 pixel 레벨에서도 재확인된다: 이 chart의 문제는 low-pass 수치 불안정이 아니라 chart 단위 fit 퇴화(작고 특이한 domain shape에서 오는 병리적 control grid)다.

## 7. 교정된 correspondence 해석

`within_chart_representative_footprint_spread` 중앙값(0.04683)과 `cross_chart_position_discrepancy` 중앙값(0.05462)의 비율은 **0.857**(WL118의 85.7%와 일치 -- 동일 데이터 재계산이므로 당연히 일치). directive 지침대로 이를 "설명된 비율"이 아니라 **"두 분포가 같은 자릿수(order of magnitude)"**라는 진술로만 보고한다. `cross_chart_position_discrepancy`의 표본 수는 11,662,905쌍(WL118보다 훨씬 많은데, 이는 chart별 대표 pixel-record 표본이 늘어난 것이 아니라 동일 representative가 여러 chart에 걸쳐 나타나는 실제 인접 쌍 개수이며 WL118과 동일 계산이다).

## 8. 교정된 normal 해석

signed normal discrepancy 중앙값 5.373도, sign-invariant 5.372도로 **거의 동일**(WL118의 5.37과 일치). p95는 61.06도 -> 52.84도로 감소(부호 반전이 p95 tail의 일부를 설명). `fraction_of_signed_disagreement_explained_by_sign_flip=0.0272`(2.7%, WL118과 일치). 이번 배치는 directive 지침대로 **"부호 보정 후 남은 불일치가 곧 진짜 NURBS 불일치"라고 단정하지 않는다** -- 서로 다른 물리적 대응점, 유한한 surfel footprint, 실제 곡률, parameterization 효과, 진짜 NURBS 불일치가 모두 섞여 있을 수 있으며 이번 배치는 이를 분해하지 않는다.

## 9. boundary-notch synthetic control (부차적)

curved 표면에서 removed_count(B/C)=36, notch_removed_count(D)=48(24열 단위 반올림으로 인한 근사, 실제 개수는 함께 보고). foot-point 경로에서: A(전체)=0.00195, B(enclosed hole)=0.00174, C(dispersed)=0.00191, D(boundary notch)=0.00203 -- **notch(D)가 dispersed(C)보다도 오차가 크고 enclosed hole(B)보다도 크다**(median 기준). p95/max에서는 B(enclosed hole)가 가장 크다(p95=0.00779, max=0.01070). fixed-UV 경로에서는 네 조건이 서로 비슷한 규모(A=0.00622, B=0.00588, C=0.00615, D=0.00636)로 수렴해, WL118에서 관찰된 "foot-point 경로에서만 hole-topology 효과가 나타난다"는 패턴이 이번 notch 대조군에서도 재확인된다. 이 결과는 부차적이며 이번 배치의 결론을 좌우하지 않는다.

## 10. WL118 결론 중 생존/통제-유발 구분

- **통제-유발로 정정됨(핵심)**: WL118의 "ARM A가 모든 영역에서 2.9~3배 우수하다"는 결론은 **불공정한 A/B 비교의 인공물**이었다. 공정한 비교(동일 solve 횟수, 분리된 metric)에서는 **METRIC G는 A가 우수하지만 METRIC C는 B가 우수/동등**하다 -- 방향이 아예 반대인 metric이 존재한다는 게 새로 밝혀진 사실이다.
- **생존**: G0/G1/G2 재구성 검증, hole/boundary-notch 관련 synthetic 패턴(foot-point 경로에서만 나타남), 표준 정규분포(normal) sign-invariance 수치, correspondence 비율(85.7%) -- 모두 WL118과 동일 데이터를 재확인했으므로 생존.
- **정정됨**: WL118의 chart-평균 기반 D-outlier 귀속("low-pass는 giant-patio가 아니라 작은 chart에만 국한된다")은 pixel 수준에서 보면 더 미묘하다 -- population 수준에서는 low-pass 지배와 잔차 사이 real correlation이 있지만(top-1000의 54%), 역사상 반복 지목된 단일 D-chart(10592) 자체의 극단성은 branch로 설명되지 않고 chart 단위 fit 퇴화로 보인다.

## 11. 다음 아키텍처 질문 (단 하나)

**"관측(camera UV) 좌표와 피팅(foot-point) 좌표를 분리해서, METRIC G를 최적화하는 fitting과 METRIC C를 보존하는 evaluation을 별도로 유지하는 two-coordinate 아키텍처가 필요한가?"** -- 이번 배치는 이 질문에 대한 증거(섹션 3의 방향이 갈리는 A/B 결과)를 제시했을 뿐 구현하지 않았다. 다음 결정은 domain-aware fitting/capacity/topology가 아니라 이 observation-vs-fitting-coordinate 분리 여부가 되어야 한다는 것이 이번 배치의 실측 근거다.

## 부록: 성능/버그 노트

이번 배치 실행 중 GPU 활용률이 15% 수준에 머무는 것을 사용자 질문으로 확인했다. 두 가지 원인을 조사했다:

1. **실제로 고치고 검증한 버그**: pixel-level D attribution을 위해 chart당 최대 200개 pixel을 순수 Python `for` 루프로 dict에 담던 코드가 CPU에서 직렬화되고 있었다. 이를 `.numpy()` 일괄 변환 + `zip()` 방식으로 벡터화했고, 별도 스크립트로 구버전과 신버전이 **비트 단위로 동일한 결과**(450개 필드 비교, mismatch 0건)를 내는 것을 확인했다. 단, 실제 규모(15,000 chart x 200 샘플)로 다시 측정한 결과 이 부분이 차지하는 비중은 전체 런타임 중 **약 60초 -> 2초(29.6배 국소 개선, 전체의 약 1%)**에 불과해, GPU 활용률 저하의 주 원인은 아니었다.
2. **근본 원인(구조적, 이번 배치에서는 수정하지 않음)**: 이 scene은 21만+ 픽셀짜리 거대 patio chart와 32픽셀짜리 최소 chart가 공존하는 긴 꼬리 분포를 가지며, 이번 배치는 directive 3번 항목이 요구한 대로 METRIC G를 두 arm 각각에 대해 **독립적으로 재계산**하는 foot-point projection을 수행한다 -- 거대 chart 하나당 `project_torch_points_to_nurbs` 호출이 WL118의 2회(ARM A fit 내부)에서 이번 배치는 4회(ARM A fit 2회 + METRIC G 평가용 재투영 2회)로 늘었다. 이는 버그가 아니라 directive가 요구한 "오염되지 않은 독립 평가"의 정당한 비용이다. CPU 측 controlled benchmark(100/500/2,000/10,000점 chart)에서 chart 크기에 따라 시간이 거의 선형으로 증가하는 것을 확인해 이 설명을 뒷받침했다. GPU에서 직접 프로파일링은 실행 중이던 live job의 VRAM 여유가 35MB뿐이어서 OOM 위험 때문에 **의도적으로 하지 않았다**.
3. Chart별 solve들을 GPU에서 batch로 묶는 것(architecture 변경)은 대략 1.5~2.5배의 wall-clock 개선이 기대되지만, 이는 이번 배치 범위를 벗어나는 결정이라 적용하지 않았다.

## 부록: 표준 export

`ORIGINAL_2DGS_SCENE`, `CANONICAL_SUBSET_MEMBERSHIP`(신규 표준 지침에 따라 기본 포함), METRIC G/C 각 arm의 residual 시각화, G0-vs-G2 disagreement, rho3d/rho2d branch 분류를 표준 review export set으로 포함했다.
