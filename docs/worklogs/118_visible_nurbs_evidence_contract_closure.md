# Worklog 118 — Visible-NURBS Evidence Contract Closure

## 상태

**완료 — 실측 있음(전체 161개 뷰). 새 CUDA 진단 필드 추가(vendored canonical 무수정, sibling 진단 빌드만 확장).** Worklog 117을 조건부로 받아들인다 — "hole/미지지-경계 근접성이 현재 WL112 fitting 경로에서 거대 patio residual 실패를 설명하지 못한다"는 결과는 유효하지만, "B2가 존재하지 않는다"거나 "capacity가 이미 지배적 원인으로 증명됐다"고 결론짓지 않는다. 새 감사가 지적한 대로, WL117의 hole-거리 분석은 **post-fit foot-point UV occupancy**로 수행됐지만 WL113의 원래 "holey chart" 관측은 **camera-raster chart domain**에서 정의됐다 — 이 불일치를 이번 배치에서 명시적으로 닫았다. **핵심 발견**: (1) camera-domain과 fitted-domain의 hole 판정은 같은 24×24 해상도에서 봐도 **전체 chart의 51.1%가 서로 불일치**한다(camera 15.8% vs fitted 64.4% — 4배 차이) — WL117의 B2 분석은 WL113이 정의한 것과 사실상 다른 "holey" 모집단을 측정했다. (2) 고정 camera UV(ARM B) 대비 현재의 foot-point 보정(ARM A)은 모든 통계·모든 영역에서 일관되게 residual을 2.9~3배(때로 1500배) 개선한다 — foot-point 보정은 진짜로 가치 있다. (3) sign-invariant normal 비교 결과 부호 반전은 신호 불일치의 **2.7%만 설명**한다 — 대부분의 normal 불일치는 진짜다. (4) representative의 chart 내부 유한-지지 spread 중앙값(0.0468)이 cross-chart 평균 변위 중앙값(0.0546)의 **85.7%**에 달한다 — 역사적 overlap-position 지표의 상당 부분이 순수한 측정 모호성일 수 있다. (5) low-pass 지배 이벤트 비율은 WL113/114의 작은-chart D-outlier(chart 10592, 44.3%)에서는 모집단 중앙값(21.7%)보다 뚜렷이 높지만, 거대 patio chart(20-29%)에서는 그렇지 않다 — D는 작은-chart 사례에 국한된 것으로 재확인됐다.

## Agent Interpretation of Intent

1. **DIRECTION 이해**: renderer observation geometry, camera chart UV, post-fit NURBS UV, support-domain identity, cross-chart comparison이라는 서로 다른 다섯 가지 양(quantity)의 시맨틱을 명확히 구분하라는 지시로 이해했다. WL107/109 위상은 그대로, 새 NURBS architecture 메커니즘은 도입하지 않는다.
2. **PURPOSE 이해**: domain decomposition·adaptive capacity·coupled fitting 중 하나를 고르기 전에, WL112-117의 측정 자체가 신뢰할 만한지부터 확인하라는 것으로 이해했다. camera-raster UV, post-fit foot-point UV, median-depth 유래 XYZ, representative identity, NURBS chart 평균 위치, signed chart normal — 이 여섯 가지를 서로 교환 가능한 것으로 취급하지 않는다.
3. **CENTRAL INTENT 이해**: "renderer 기하·camera-domain UV·fitted UV·실제 cross-chart correspondence를 분리한 뒤, WL112-117의 실패 신호 중 몇 개가 진짜 NURBS-representation 실패로 남는가"에 답하는 것으로 이해했다. 이번 배치에서 NURBS를 최적화하지 않는다.
4. **동결 유지 사항**: WL107/109 위상, WL112 camera-blob 멤버십, 고정 8×4 degree-2(통제용), 현재 정규화(smoothness/Tikhonov), 현재 renderer representative identity.
5. **의도적으로 변경한 양**: (a) 3D fitting target의 UV 파라미터화 — camera UV(고정, ARM B) vs foot-point 보정 UV(현재, ARM A)를 명시적으로 병행 측정, (b) unprojection의 pixel-center 컨벤션(sibling 진단 함수로만, `depths_to_points` 자체는 무수정), (c) normal 비교 지표(signed 유지 + sign-invariant 병행 보고).
6. **진단 전용 수정**: 이번 배치의 유일한 CUDA 변경은 sibling 진단 확장(`diff_surfel_rasterization_diag`)에 `median_rho3d`/`median_rho2d`/`median_s_u`/`median_s_v` 4개 (H,W) 출력을 T>0.5 median 이벤트의 정확히 같은 지점에 추가한 것뿐이다 — canonical vendored 커널(`diff_surfel_rasterization/`)은 한 글자도 건드리지 않았다.
7. **도입한 조작적 가정**: (a) ARM B(고정 UV)는 기존 `_solve_control_grid_lsq`(private이지만 이미 `torch_nurbs.py` 내부에서 coupled-fitting에 재사용되는 함수)를 직접 호출해 "이미 존재하는 fixed-UV fitter 능력"으로 구현했다 — directive가 "새 fitter를 만들지 말고 기존 능력을 써라"고 명시했기 때문. (b) camera-domain hole 판정을 raster-native(WL113 방식)와 uv-binned(24×24, fitted-domain과 같은 해상도) 두 가지로 **모두** 계산하고 절대 직접 비교하지 않았다 — IoU/일치도는 반드시 같은 해상도 쌍끼리만 계산했다. (c) representative 유한-지지 spread는 처음에 chart마다 서로 다른 representative 수만큼 도는 순수 Python for-loop로 구현했는데, 실행 중(101/161 뷰 시점) 사용자가 속도 저하를 지적해 `index_add_`/`scatter_reduce_` 기반 완전 벡터화로 교체했다 — 두 구현이 그룹별로 정확히(부동소수점 오차 수준까지) 같은 값을 내는지 실측으로 검증한 뒤 교체했다. 이미 진행 중이던 실행(구버전 코드로 완주)에는 영향 없음을 확인했다. (d) equal-count 합성 대조군의 "dispersed removal"(C)은 raster 순서에서 매 N번째 샘플을 제거하는 결정론적 규칙을 썼다(directive는 "deterministic dispersed or boundary removal" 중 하나를 요구, 임의 선택이지만 scene-tuned는 아님).
8. **prompt의 모호함**: §8이 "cross-chart mean displacement"와 "within-chart finite-support spread"를 비교하라고 요구하는데, 기존 overlap-position-discrepancy 지표(WL111-114) 자체가 이미 "대표별 평균 fitted point"의 연속-쌍 비교이므로 "cross-chart displacement"의 구체적 정의를 그 기존 지표와 동일하게 재사용했다 — 새 지표를 발명하지 않기 위한 선택이며, 이 재사용이 directive가 원한 정확한 비교인지는 §10에서 명시적으로 논한다.

## Implementation Fidelity Statement

**CUDA(diff_surfel_rasterization_diag만, canonical 무수정)**: `forward.h`(시그니처 4개 필드 추가), `forward.cu`(median T>0.5 캡처 지점에 `median_rho3d`/`median_rho2d`/`median_s_u`/`median_s_v` 로컬 변수 선언+캡처+출력 버퍼 기록), `rasterizer.h`/`rasterizer_impl.cu`(시그니처 전달), `rasterize_points.cu`/`.h`(텐서 할당 4개 추가, 튜플 반환 12→16), `ext.cpp`는 무수정(pybind11이 튜플을 제네릭하게 처리). Python 측 `osn_gs/render/torch_surfel_representative_diagnostics.py::render_with_pixel_representative`가 4개 새 필드를 unpack해 dict에 추가. 재빌드는 `scripts/build_surfel_extension_diag.bat`로 clean rebuild(증분 빌드 캐시가 오래된 12-튜플 시그니처를 남겨 첫 시도는 링크 에러 발생 → `build/` 삭제 후 재빌드로 해결). 렌더링 불변성 재검증(`tests/test_surfel_representative_diagnostics.py`, 17개 전부 통과, 신규 3개 포함) — canonical 렌더 출력은 한 비트도 바뀌지 않았다.

**Python(신규 devtools만, production 코드 무수정)**: `scripts/devtools/visible_nurbs_evidence_contract_closure.py`(메인 실측 스크립트), `scripts/devtools/_wl118_subset_visualization_addendum.py`(사용자 피드백에 따른 subset-membership 시각화 보충 export). 읽기전용 재사용: `torch_camera_induced_visible_adjacency.py`(WL107/109, 위상), `torch_camera_observed_chart_domains.py`(WL112, chart 구성), `torch_nurbs.py`(fitter 전체 — `fit_torch_visible_surface_lsq`, `fit_torch_visible_surface`, `_solve_control_grid_lsq`), `osn_gs/render/surfel_geometry.py::depths_to_points`(공식 unprojection, 무수정), `osn_gs/core/torch_pipeline.py::TorchOSNGSPipeline._uv_occupancy_mask`(기존 trim 시맨틱), WL113의 `blob_domain_shape`/`_distribution`/`_bin_by_quantile`, WL117의 `hole_and_edge_masks`. 새 sibling 함수(`depths_to_points_rasterizer_pixel_center`)는 `depths_to_points`를 절대 수정하지 않고 devtools 스크립트 안에 별도로 구현했다.

## 1. 히스토리 보존

WL112/113/114/117은 그대로 보존했고 재작성하지 않았다 — 이번 배치는 새 증거로 그 결과를 **재해석**만 한다(§9-13 참조). WL107/109 위상 재생(`visible_component_count=559,989`, 최대 컴포넌트 36.77%, singleton 45.02%)과 `fitted_chart_count=14,900`이 WL112/113/117과 정확히 일치함을 재확인해, 이번 배치의 새 진단이 기존 파이프라인을 변형하지 않았음을 검증했다.

## 2. Camera UV vs Foot-Point UV

각 chart에서 `uv_camera`(WL112가 원래 공급한 카메라-래스터 픽셀 좌표)와 `uv_footpoint`(현재 foot-point 보정 후 최종 UV)를 절대 서로 덮어쓰지 않고 분리 보존했다.

| | 값(정규화 [0,1]² 단위) |
|---|---|
| chart별 중앙값 변위(중앙값/p95/최대) | 0.0269 / 0.0713 / 0.708 |
| chart별 최대 변위(중앙값/p95/최대) | 0.323 / 0.782 / 1.281 |
| 거대 chart(pixel_count 상위 사분위) 중앙값 변위 | 0.0253 |
| 구멍 있는 chart 중앙값 변위 | 0.0299 |
| 구멍 없는 chart 중앙값 변위 | 0.0233 |
| UV 변위 vs residual_A 상관계수 | 0.262 |

UV drift는 작지 않다(정규화 단위 기준 chart별 중앙값 변위가 0.027, 최대 변위는 여러 chart에서 0.3~1.28에 달함 — 도메인의 상당 부분을 가로지르는 재파라미터화가 실제로 일어난다). 거대 patio chart에서 유별나게 크지는 않고(오히려 전체 중앙값보다 약간 낮음), 구멍 있는 chart에서 다소 더 크다(0.030 vs 0.023). residual과의 상관은 약-중간 정도(0.262)로, drift가 클수록 fit이 나쁜 경향이 있지만 지배적이지는 않다.

## 3. Worklog 117 Support-Domain 해석 정정

같은 24×24 해상도에서 두 도메인을 **독립적으로** 계산했다: (A) camera-domain — `uv_camera`를 24×24에 bin; (B) fitted-domain — `uv_footpoint`에 기존 `TorchOSNGSPipeline._uv_occupancy_mask` 시맨틱 적용(WL117과 동일).

| | 값 |
|---|---|
| camera-domain raster-native(WL113 정의) hole 보유 비율 | 43.5% (WL113과 정확히 일치) |
| camera-domain uv-binned(24×24) hole 보유 비율 | **15.8%** |
| fitted-domain uv-binned(24×24) hole 보유 비율 | **64.4%** |
| 같은 해상도 IoU(중앙값/p95) | 0.735 / 0.954 |
| **같은 해상도에서 hole 유무 자체가 불일치하는 chart 비율** | **51.1%** |

**같은 해상도에서 봐도 fitted-domain은 camera-domain보다 4배 이상 더 자주 "구멍 있음"으로 판정한다.** IoU 중앙값 0.735는 두 도메인이 완전히 무관하지는 않지만 상당히 다르다는 것을 보여준다. 절반 이상(51.1%)의 chart에서 두 도메인은 "이 chart에 구멍이 있는가" 자체에 동의하지 않는다. **이것이 이번 배치의 핵심 correction이다**: WL117의 B2 within-chart 분석은 fitted-domain(post-foot-point) 정의로 "구멍"을 판정했지만, WL113이 원래 관측한 "holey chart"는 camera-raster 정의였다 — 두 분석은 부분적으로 다른 모집단을 측정했다. raster-native와 uv-binned 카메라-도메인 수치를 절대 직접 비교하지 않았다(서로 다른 해상도).

## 4. Fixed-UV A/B

동일한 chart 멤버십·XYZ·8×4 control grid·degree·smoothness·Tikhonov로 ARM A(현재: camera UV 초기화 → LSQ → foot-point 보정)와 ARM B(고정 camera UV: 같은 IDW seed → 기존 `_solve_control_grid_lsq`로 단 한 번의 정규화 solve, foot-point 재파라미터화 없음)를 비교했다.

| | ARM A(현재) 중앙값 | ARM B(고정 UV) 중앙값 |
|---|---|---|
| residual 중앙값의 분포 중앙값 | 0.00462 | **0.01328** (2.9배 나쁨) |
| residual p95의 분포 중앙값 | 0.0210 | **0.0635** (3.0배 나쁨) |
| residual 최대의 분포 중앙값 | 0.0377 | **0.108** (2.9배 나쁨) |
| residual 최대의 분포 최대값(전체 씬 최악) | 1517.2 | **1824.8** (더 나쁨) |

**모든 영역(table_top, table_side_curved, table_legs, patio, hedge)에서 예외 없이 ARM A가 ARM B보다 나았다** — table_legs는 1500배(0.00000053 vs 0.00082), patio/hedge/curved는 2.8~3.1배. **승자를 자동으로 선언하지 않되, 증거는 명확하다: foot-point 보정은 현재 architecture에서 진짜로, 일관되게, 모든 영역에서 fit 품질을 개선한다.**

## 5. Median-Depth Low-Pass Provenance

새 진단 CUDA 필드로 실측한 결과, 전형적 chart는 픽셀의 21.7%(중앙값)가 `rho2d < rho3d`(low-pass 분기가 acceptance를 지배)인 이벤트다. 영역별로 크게 다르다: table_top/table_legs 0.0%(평평하고 정면), patio 17.1%, hedge 23.3%, **table_side_curved 32.0%(다섯 영역 중 최고)** — 곡면·grazing 각도 표면일수록 low-pass 의존도가 높다는 예상과 정확히 일치한다. 독립 fixture 실측(`_single_surfel`)에서 `s`(surfel-local 교차 좌표)가 rho2d가 지배할 때 극단적으로 커질 수 있음(관측값 최대 8.6×10²²)을 직접 확인했다 — `depth`는 항상 `s`에서 유도되므로(rho3d/rho2d 선택과 무관), low-pass 지배 이벤트에서 `s`가 병적으로 크면 `depth`도 병적일 수 있다.

## 6. D-Outlier Low-Pass 귀속

residual_max 상위 10개 chart의 low-pass 지배 비율을 확인했다: 9개(전부 patio 거대 chart)는 20.7~28.8%로 모집단 중앙값(21.7%)보다 그리 높지 않다. **그러나 1개(chart 10592, table_side_curved, 271픽셀, residual_max=408.5 — WL113/114가 반복 지목한 바로 그 작은-chart D-outlier)는 44.3%로 뚜렷이 높다.** **결론: low-pass 지배는 WL113/114의 작은-chart D-outlier 현상을 부분적으로 설명하지만, 거대 patio chart의 residual 극단값은 low-pass가 아니라 다른 원인(WL117이 지목한 scale/capacity)에서 온다.** D를 chart-architecture 결론에 섞지 않는다.

## 7. Half-Pixel Unprojection 통제

`depths_to_points`(공식, 무수정)와 sibling 진단 함수(rasterizer 자신의 `(W-1)/2, (H-1)/2` 컨벤션) 사이의 XYZ 변위: 중앙값 0.00857, p95 0.0213, 최대 0.246(scene 단위, 32만 표본). 작지만 무시할 정도는 아니다 — UV 변위(§2)의 약 1/3 크기다. 어느 경로도 자동으로 우월하다고 선언하지 않는다; 이는 renderer 기하 자체의 관례 차이가 만드는 하한선 규모의 불확실성으로 기록한다.

## 8. Signed vs Sign-Invariant Normal 비교

| | signed(기존) | sign-invariant(`acos(abs(dot))`) |
|---|---|---|
| 중앙값 | 5.373°(WL113과 정확히 일치) | 5.372° |
| p95 | 61.06° | **52.84%(약 13% 감소)** |
| 최대 | 179.88° | 90.00°(지표 자체의 수학적 상한, 정보 없음) |
| 신호 불일치가 90°를 넘는 pair 비율 | — | **2.7%만** |

**부호 반전은 WL112/114가 보고한 normal 불일치의 작은 일부(약 2.7%의 pair, p95 기준 약 13% 감소)만 설명한다 — 대부분의 신호 불일치는 진짜다.** signed 지표를 topology/NURBS 실패 게이트로 계속 쓰지 않되(directive 지시), 부호 반전이 문제의 핵심이라는 가설은 이번 실측으로 기각한다.

## 9. Representative Position Correspondence 교정

같은 representative가 한 chart 안에서 여러 픽셀에 걸쳐 관측될 때(surfel 자신의 유한 footprint), 그 관측들의 spread(각 대표의 자기 그룹 평균으로부터의 최대 편차)를 계산했다.

| | 중앙값 |
|---|---|
| chart 내부 representative footprint spread | **0.0468** |
| cross-chart 평균 변위(기존 overlap-position-discrepancy) | 0.0546 |
| 비율(footprint spread / cross-chart 변위) | **0.857** |

**chart 내부의 자연스러운 유한-지지 spread가 이미 "cross-chart 표면 불일치"로 보고돼 온 값의 85.7%에 달한다.** 즉 WL111-114가 반복 보고한 overlap-position-discrepancy 지표의 상당 부분은 진짜 표면 간 불일치가 아니라, 하나의 surfel이 여러 픽셀에서 관측될 때 그 자체로 생기는 측정 모호성일 가능성이 높다. `same representative id = same physical point`라는 가정은 명확히 기각됐다 — 대표 하나의 위치도 이미 상당한 내재적 spread를 가진다.

## 10. Equal-Count 합성 대조군 (Worklog 117 교정)

평면/곡면 24×24 격자에서 A(전체) / B(중심 hole, 36개 제거) / C(같은 36개를 raster 순서 매 N번째로 분산 제거, **B와 정확히 같은 유지 개수**) 를 무수정 fitter로 비교했다.

| | A(전체) | B(둘러싸인 hole) | C(분산 제거, 같은 개수) |
|---|---|---|---|
| 평면, foot-point, p95 | 1.33e-7 | 1.69e-7 | 1.33e-7 |
| **곡면, foot-point, p95** | 0.00666 | **0.00779** | **0.00640** |
| **곡면, foot-point, 최대** | 0.00729 | **0.0107** | **0.00690** |
| 곡면, 고정-UV, 중앙값 | 0.00652 | 0.00619 | 0.00653 |

**평면은 무정보(둘 다 기계 정밀도 수준)** — 곡률이 0이면 hole이든 분산 제거든 차이가 없다. **곡면에서는 foot-point 보정 경로에 한해 B(둘러싸인 hole)가 C(같은 개수, 분산 제거)보다 꼬리(p95 +22%, 최대 +55%)에서 뚜렷이 나쁘다** — 이는 **단순 샘플 수 감소를 넘어서는, 둘러싸인-hole topology 자체의 효과**다(directive의 핵심 질문에 대한 명확한 "예"). 그러나 고정-UV 경로에서는 이 효과가 사라진다(B가 오히려 C보다 살짝 낫다) — hole의 해로운 효과는 foot-point 재파라미터화가 hole 주변에서 다르게 거동하는 것과 관련이 있어 보인다. **Synthetic PASS가 real-scene viability를 증명하지는 않는다** — 이는 통제된 최소 반례일 뿐이다.

## 11-13. WL112-117 결론 재평가

**생존(그대로 유지)**:
- WL109 GATE PASS(canonical topology) — 이번 배치에서 완전히 재확인, 무변경.
- WL112의 핵심 판정(대표-중심/픽셀-표면 불일치는 WL111 실패의 주 원인이 아님) — 이번 배치는 이를 건드리지 않았다.
- WL117의 "hole 근접성이 거대 patio 실패를 설명하지 않는다" — §6(D-outlier low-pass 귀속)이 독립적으로 재확인: 거대 chart의 low-pass 비율은 모집단 수준이지 상승하지 않았다.
- WL114의 "locality가 유용하다"는 부분 판정 — 건드리지 않음.

**약화됨**:
- WL117의 B2 within-chart 상관관계(중앙값 -0.05~-0.10) — §3의 domain-mismatch(51.1% 불일치) 때문에, 그 분석이 정확히 WL113이 정의한 것과 같은 "구멍" 모집단을 측정했다고 더 이상 확신할 수 없다. 신호 방향 자체는 부정되지 않았지만, 그 신호가 정확히 무엇을 측정했는지는 이번 배치 전까지 불명확했다.
- WL111-114가 반복 보고한 overlap-position-discrepancy를 "표면 불일치"로 해석하는 것 — §9(85.7% 비율)로 상당 부분 약화. 대부분이 아니라 "상당 부분"이라고 표현하는 것은, 15%가량은 여전히 순수한 within-chart spread로 설명되지 않는 진짜 cross-chart 성분일 수 있기 때문이다.
- WL112/114의 normal 불일치를 architecture 실패로 해석하는 것 — §8로 일부 약화(p95 기준 13% 감소)이지만, 나머지 87%는 진짜다(§8 결론 참조).

**무효화됨**: 이번 배치 범위에서 완전히 무효화된 이전 결론은 없다. 각 조정은 방향을 뒤집지 않고 크기와 해석을 교정했다.

## 14. 다음 architecture 질문

**"WL117의 within-chart hole-근접 상관관계를 camera-domain 정의(WL113과 일치)로 재계산하면 여전히 실재하는가, 아니면 §3에서 드러난 domain-mismatch 대부분에 기인한 것인가?"** 이 질문에 답하기 전까지는 domain-aware fitting이나 coupled-patch 연결 중 하나를 선택할 근거가 충분하지 않다 — WL117의 B2 신호 자체가 어느 도메인 정의를 쓰느냐에 따라 달라질 수 있음을 이번 배치가 직접 보였기 때문이다.

## 검토용 export 경로

`output/118_osn_gs_nurbs_evidence_contract_closure/` 아래 8개 뷰 폴더(`iteration_0000001/point_cloud.ply`, `render.ppm`, `README.md`): `ORIGINAL_2DGS_SCENE`, `UV_CAMERA_VS_FOOTPOINT_DISPLACEMENT`, `CAMERA_VS_FITTED_DOMAIN_DISAGREEMENT`, `FIXED_UV_RESIDUAL`, `CORRECTED_UV_RESIDUAL`, `NORMAL_SIGN_INVARIANCE_EFFECT`, `LOW_PASS_DOMINATED_EVENTS`, `HALF_PIXEL_UNPROJECTION_DISPLACEMENT`. 사용자 피드백(2026-08-26, "앞으로 subset 시각화를 항상 포함")에 따라 `CANONICAL_SUBSET_MEMBERSHIP`(위상 재생만으로 만든 보충 export, 이번 배치는 실행 중간에 요청이 와서 추가 스크립트로 부착 — WL119부터는 메인 스크립트에 기본 내장)도 추가했다. 미리보기 PNG는 `preview_png/<뷰이름>.png` 한 폴더에 통합. 전체 JSON 리포트: `output/118_osn_gs_nurbs_evidence_contract_closure/visible_nurbs_evidence_contract_closure_report.json`.

## 테스트

Canonical 벤더 CUDA는 무수정. Sibling 진단 확장만 새 필드 4개 추가 후 clean rebuild.

- `tests/test_surfel_representative_diagnostics.py`(기존 14개 + 신규 3개 = 17개): 렌더링 불변성 재검증, `median_rho3d`/`median_rho2d`가 representative와 정확히 같은 -1/유효 패턴을 따름, `median_s_u/v`가 같은 이벤트의 `rho3d = s_u²+s_v²`와 수치적으로 일치.
- `tests/test_visible_nurbs_evidence_contract_closure.py`(신규 12개): camera UV가 fit 호출 후에도 변경되지 않음, foot-point UV가 곡면에서 실제로 드리프트함, 고정-UV arm이 foot-point 재파라미터화를 전혀 호출하지 않고도 유한한 residual을 냄, sign-invariant 지표가 반대 법선/직교 법선에서 각각 0°/90°를 정확히 냄, half-pixel 변형이 depth=0에서 공식 경로와 일치하고 유한한 변위를 냄, equal-count 대조군에서 B/C가 정확히 같은 유지 개수를 가지며 곡면에서 B가 C보다 꼬리에서 나쁘고 평면에서는 무시할 만한 차이임.
- `.venv/Scripts/python.exe -m pytest tests/test_surfel_representative_diagnostics.py tests/test_visible_nurbs_evidence_contract_closure.py -q` → **29 passed**.
- 실측 스크립트는 `--max-views 4` smoke test로 전체 파이프라인(합성 대조군 → sweep → 위상 재생 → ARM A/B 이중 fit → 6개 섹션 진단 → export/렌더)이 오류 없이 끝까지 도는 것을 먼저 확인한 뒤, 전체 161개 뷰로 재실행했다(런타임 2763.2초). 실행 중(101/161 뷰, 63% 시점) representative 유한-지지 spread 계산이 chart마다 순수 Python for-loop를 도는 성능 문제를 사용자가 지적해 발견했고, `index_add_`/`scatter_reduce_` 기반 벡터화로 즉시 교체·검증했다(이미 진행 중이던 실행에는 영향 없음, 다음 배치부터 반영). 위상 재생 수치와 `fitted_chart_count=14,900`이 WL112/113/117과 완전히 일치함을 재확인했다. 전체 pytest는 재실행하지 않았다(canonical/production 코드 무수정 — sibling 진단 확장만 확장).
