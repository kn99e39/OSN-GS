# Worklog 113 — Chart Representation Contract Diagnostic

## 상태

**완료 — 진단 전용 배치, 건축 판정 없음(directive 지시).** Worklog 112를 튜닝하지 않고 chart 세분화도 구현하지 않았다. Worklog 107/109 위상, Worklog 111 카메라-관측 blob 구성, Worklog 112 렌더러-네이티브 픽셀-표면 기하, 고정 8×4/degree-2 NURBS 설정을 전부 그대로 동결한 채, "카메라-관측 연결 blob 하나 = 사각형 tensor-product NURBS chart 하나"라는 표현 단위 자체가 왜 실패하는지 정확한 원인을 물었다. **실측 결과: A(관측 지지 부족), B(사각형 도메인/구멍 불일치), C(고정 8×4 용량 한계), D(렌더러 median-depth 수치 불안정) 네 가지 원인이 전부 실제로 존재하지만, 서로 다른 실패 증상에 각각 다르게 책임이 있다** — zero-coverage 컴포넌트(153,600개)는 100% A, 전형적 fit 품질 저하는 주로 B, WL112가 남긴 residual 극단값(최대 1517)은 거대 컴포넌트에서의 B+C 결합, overlap 극단값(최대 1514)은 작은 컴포넌트에서의 D. 단일 원인으로 강제하지 않는다(directive 11절 지시).

## 1. 냉동한 것 / 바꾸지 않은 것

Worklog 107/109 캐노니컬 위상(`torch_camera_induced_visible_adjacency.py`), Worklog 111 blob 라벨링(`label_same_component_blobs`), Worklog 112 픽셀-표면 샘플 구성(`build_view_chart_pixel_samples`, `valid_pixel_chart_mask`), 고정 8×4/degree-2 NURBS 설정(`fit_torch_visible_surface_lsq`) 전부 **무수정, 읽기전용 재사용**만 했다. 새로 만든 것은 진단 전용 devtools 스크립트(`scripts/devtools/chart_representation_contract_diagnostic.py`)뿐이며, 어떤 canonical/production 코드도 건드리지 않았다. 위상 재생 수치(`visible_component_count=559989`, `largest_component_surfel_fraction=0.3677`, `singleton_surfel_count=535910`)가 WL107/109/111/112와 정확히 일치함을 다시 한번 확인했다.

## 2. 교정된 컴포넌트 지지 회계

WL111/112는 "representative_count < 32"를 곧 "chart를 만들 수 없다"로 취급하지 않았지만, 이번 배치는 그 구분을 명시적으로 측정했다. 대표를 가진 155,457개 컴포넌트에 대해:

- representative_count 분포: 중앙값 1, p95 4, 최대 437,751 (매우 롱테일).
- **total_valid_pixel_observations 분포**: 중앙값 3, p95 **57**, 평균 281.9, 최대 32,062,240 — representative 수보다 훨씬 관대한 지표인데도 p95가 57에 불과하다.
- max_single_blob_pixel_count 분포: 중앙값 1, p95 9 — 즉 대부분 컴포넌트는 어떤 단일 뷰에서도 10픽셀조차 못 모은다.
- **32픽셀 이상 블롭이 하나라도 있는 컴포넌트: 155,457개 중 1,857개(1.2%)뿐** — 그리고 이 1,857개는 전부(100%) 실제로 커버됐다(`components_with_ge1_blob_ge32_pixels_AND_covered = 1857`).
- **directive 2절이 요구한 핵심 구분**: representative 수가 이미 32개 이상인데도(즉 표본이 충분해 보이는데도) 32픽셀 블롭을 한 번도 만들지 못한 컴포넌트가 **11개** 있다(`components_with_zero_blob_ge32_pixels_but_LARGE_rep_count_ge32`) — 관측이 여러 소규모 blob으로 항상 흩어지는, "작은 인구"와는 다른 실패 양상이지만 전체 zero-coverage 중 극소수(0.007%)에 불과하다.

## 3. Zero-coverage 컴포넌트 정확한 귀속

zero-coverage 컴포넌트 153,600개 **전부**(100%)가 `NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES`다. `HAS_GE32_BLOB_BUT_STILL_UNCOVERED`는 0건이다 — 현재 fit 루프는 32픽셀 문턱을 넘은 블롭에 대해 실패/스킵 분기가 아예 없으므로(directive가 예측한 대로), "지지는 충분한데 fitting이 실패"하는 경로는 이 실측에서 관측되지 않았다. **§11 분류 A(SUPPORT_LIMITED)가 zero-coverage에 대해서는 예외 없이 유일한 원인이다.**

## 4. Blob 도메인 모양 분포 (14,900개 fitted chart)

- pixel_count: 중앙값 59, p95 567, 최대 216,783.
- **bbox_occupancy_ratio: 중앙값 0.5, p95 0.71** — 전형적 chart조차 자기 bounding rectangle의 절반이 빈 공간이다.
- aspect_ratio(width/height): 중앙값 1.2, p95 3.4, 최대 15.0 — 극단적으로 가늘고 긴 chart도 존재.
- hole_count: 중앙값 0, 평균 36.1, p95 9, 최대 3265. **fraction_of_charts_with_ge1_hole = 43.5%.**
- `unsupported_rectangular_domain_fraction`(= 1 − occupancy_ratio): 중앙값 0.5, p95 0.69 — 사각형 tensor-product 도메인이 요구하는 면적의 절반가량이 실제 관측과 무관하다.

## 5. 도메인 모양 ↔ fit 품질 관계

occupancy quantile 4구간 비교: residual 중앙값 자체는 구간별로 큰 차이가 없지만(0.0041~0.0056), **overlap_position_p95는 가장 조밀한 구간(occupancy 0.58~0.96)에서 6.82로 가장 낮은 구간(2.85~3.16)보다 2배 이상 높다** — 조밀하게 채워진 chart일수록 오히려 overlap 꼬리가 나쁘다(대개 복잡한 형상=거대 컴포넌트이기 때문으로 해석). **구멍 유무 비교가 더 뚜렷하다**: 구멍 있는 chart의 residual 중앙값 0.0078 vs 구멍 없는 chart 0.0030(약 2.6배), overlap 위치 중앙값 0.311 vs 0.178(약 1.7배) — **구멍은 극단값뿐 아니라 전형적 fit 품질에도 실질적으로 나쁜 영향을 준다.**

## 6. 고정 8×4 용량 진단

- design-matrix rank 분포: 중앙값 32(full), p95 32. **fraction_full_rank = 51.4%.**
- condition number: 중앙값 6492, p95 25억, 최대 404억 — 롱테일이 매우 크다(수치적으로 불안정한 chart가 존재).
- **residual_by_rank_deficiency**: full-rank chart의 residual 중앙값(0.0067)이 rank-deficient chart(0.0031)보다 오히려 **높다.** 이는 직관과 반대로 보이지만, rank-deficient chart는 대개 32에 가까스로 걸친 아주 작은 chart(표본이 적어 사실상 보간에 가까워 residual이 낮게 나옴)이고, full-rank chart는 표본이 충분한 만큼 실제 형상 복잡도도 더 크기 때문으로 해석된다. full-rank chart 내에서 sample count와 residual의 상관계수는 **0.545**(중간 정도 양의 상관) — **"표본이 부족해서"가 아니라 "표본은 충분한데 32개 control point로는 그 복잡도를 다 못 담는" 좁고 구체적인 신호**이며, 아래 §7에서 보듯 실제로는 소수의 거대 컴포넌트에 집중된다.

## 7. 극단값 정확한 출처 추적 (핵심 발견)

residual_max와 overlap_position_discrepancy 상위 15개 chart를 각각 view/camera/pixel/depth/hole/rank까지 추적한 결과, **두 극단값은 서로 다른, 겹치지 않는 메커니즘에서 나온다**:

- **residual_max 상위 15개 중 12개**가 patio의 최대 컴포넌트(`component_id=0`, 전체 표면의 36.77%를 차지하는 그 컴포넌트, WL107 이래 알려진 거대 컴포넌트)이고, 나머지 2개도 table_top의 큰 컴포넌트(`component_id=1`)다. 전부 픽셀 수 4만~21만, 구멍 수 780~3029개, **rank는 항상 32(full, 데이터는 충분)** — 즉 이 극단값은 관측 부족이 아니라 **거대하고 구멍투성이인 blob을 8×4 rectangular NURBS 하나로 욱여넣은 결과**(§11 분류 B+C 결합, 거대 컴포넌트에 한정). 유일한 예외 1건(chart 10592, `component_id=19`, table_side_curved, 271픽셀)은 rank=13(rank-deficient)이고 depth가 8.76~1723까지 튀는 이상 샘플을 포함한다 — 아래 D 사례와 동일한 chart다.
- **overlap_position_discrepancy 상위 15개는 정반대로 전부 작은 컴포넌트**(table_side_curved의 19/40/26/38, patio의 180/324)에서 나오며 픽셀 수 38~276으로 작다. 그 중 여러 건이 depth_std 10~104(같은 작은 chart 안에서 depth가 극단적으로 튐)를 보인다 — 예: chart 10592는 271픽셀짜리 chart 안에 depth 8.76~1723.4가 섞여 있다. hole_count는 대부분 0~7로 낮다. **이는 §11 분류 D(렌더러 median-depth의 국소 수치 불안정)의 명확하고 고립된 증거다** — 배경/차폐 경계 근처의 이상 depth 샘플 몇 개가 작은 chart의 fitting과 인접 view 간 일관성을 크게 흔든다.

## 8. table_top 결과

representative 62,608개 중 47,291개(**75.5%**) 커버, 나머지 15,317개(24.5%) 전부 `NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES`. fitted chart 573개의 hole 수는 중앙값 0이지만 **평균 244.1, p95 1241** — table_top을 커버하는 소수의 큰 chart(중앙 화분/장식물이 만드는 차폐 구멍)가 매우 holey하다. **table_top의 낮지 않은 커버리지에도 남은 결손은 A(파편화)이고, 커버된 부분의 fit 품질 위험은 B(구멍)다.**

## 9. table_side_curved 결과

representative 155,930개 중 97,007개(**62.2%**) 커버, 나머지 58,923개(37.8%) 전부 A. fitted chart 4,234개는 pixel_count 중앙값 58로 작고(patio/table_top보다 훨씬 작음), hole은 드물다(중앙값 0, 평균 1.2). **curved rim의 낮은 커버리지는 홀/도메인 모양(B) 때문이 아니라 순수하게 A(관측이 작은 조각으로 나뉘어 32픽셀 문턱을 못 넘음) 때문이다.**

## 10. patio 결과

representative 332,845개 중 265,977개(**79.9%**) 커버. fitted chart 5,805개는 hole 수 평균 65.2, p95 13 — 거대 patio 컴포넌트(§7)를 포함하는 소수의 chart가 hole 분포를 강하게 왜곡한다. **patio는 가장 높은 커버리지를 보이지만 §7의 극단적 residual/overlap 사례 대부분이 여기서 나온다 — 커버리지와 fit 안정성은 별개 축이다.**

## 11. hedge/배경 결과

representative 167,397개 중 91,805개(**54.8%**, 다섯 영역 중 최저) 커버. fitted chart 4,107개는 중앙값 픽셀 58, hole 중앙값 0 — curved rim과 마찬가지로 **A(관측 파편화)가 주된 원인이며 B의 기여는 제한적**이다.

## 12. 지배적 실패 분류

directive 11절 지시대로 단일 원인으로 강제하지 않는다:

- **A. SUPPORT_LIMITED — 적용됨.** zero-coverage 컴포넌트의 100.0%가 `NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES`. curved rim/hedge 저커버리지의 지배적 원인.
- **B. CHART_UNIT/RECTANGULAR_DOMAIN_FAILURE — 적용됨.** fitted chart의 p95 unsupported-domain-fraction 0.69, 43.5%가 구멍 보유, 구멍 있는 chart의 residual 중앙값이 없는 chart보다 2.6배 높음. residual 극단값(최대 1517)의 절반 원인.
- **C. FIXED_NURBS_CAPACITY_FAILURE — 좁게 적용됨.** full-rank(데이터 충분) chart의 residual이 rank-deficient chart보다 오히려 높고, full-rank 내 sample-count-residual 상관 0.545 — 그러나 이는 소수의 거대(patio component 0, table_top component 1) chart에 국한된 신호이며, 일반적 chart에서 "용량 부족"의 증거는 약하다. residual 극단값의 나머지 절반 원인(B와 결합).
- **D. NUMERICAL/GRAZING_SURFACE_FAILURE — 적용됨, 그러나 국소적.** overlap_position_discrepancy 극단값(최대 1514) 전부가 작은 컴포넌트의 소수 chart에서 나오며, 렌더러 median-depth가 국소적으로 8~1700대까지 튀는 사례가 확인됐다. residual 극단값에는 1건만 관여.

**요약**: zero-coverage(컴포넌트 개수의 대다수)는 거의 순수하게 A. 일반적 fit 품질은 B(구멍)의 영향을 실질적으로 받는다. residual의 극단적 꼬리는 소수의 거대·holey 컴포넌트에서 B와 C가 결합해 나타난다. overlap의 극단적 꼬리는 소수의 작은 컴포넌트에서 D가 고립적으로 나타난다. 네 원인이 서로 다른 통계량·서로 다른 컴포넌트 규모대에서 독립적으로 관측됐다.

## 13. 조건부 다음 표현 변경(구현하지 않음, directive 12절)

- A가 지배하는 영역(대다수 zero-coverage 컴포넌트, curved rim/hedge)에는 NURBS 튜닝이 무의미하다 — 이 컴포넌트들은 애초에 32픽셀짜리 관측조차 만들지 못하므로, 어떤 chart 표현으로 바꿔도 해결되지 않는다.
- B가 지배하는 영역(전형적 chart의 구멍/비사각형 도메인)에는 chart **단위** 자체를 바꿔야 한다 — 사각형 tensor-product 도메인이 아니라 실제 관측된 지지 영역에 맞는 도메인(trimmed/non-rectangular 또는 subdivided)이 필요하다.
- C는 소수의 거대 컴포넌트에 한정되므로, 단순히 control-grid 해상도를 올리는 것보다 **그 컴포넌트들만** 세분화하는 것이 더 정확한 다음 단계일 것이다.
- D는 chart 표현과 무관한, 렌더러 depth 자체의 국소 이상값 문제이므로 chart 설계를 바꾸기 전에 별도로 다뤄야 한다.

구체적 구현은 이번 배치에서 하지 않는다(directive 지시). 다음 단계 제안은 Master 문서 addendum에서만 다룬다([[feedback_no_next_step_suggestions_in_worklogs]]).

## 14. 검토용 export 경로

`output/113_osn_gs_chart_contract_diagnostic/`(WL113 종료 후 폴더 번호 규약 적용, [[feedback_output_folder_numbering]]) 아래 10개 뷰(`iteration_0000001/point_cloud.ply`, `render.ppm`, `preview_png/render.png`, `README.md`): `ORIGINAL_2DGS_SCENE`, `COMPONENT_SUPPORT_ATTRIBUTION`, `ZERO_COVERAGE_CAUSE`, `CHART_DOMAIN_OCCUPANCY`, `CHART_DOMAIN_HOLES`, `NURBS_CAPACITY_RANK_DEFICIT`, `EXTREME_RESIDUAL_PROVENANCE`, `TABLE_CONTRACT_DIAGNOSTIC`, `CURVED_CONTRACT_DIAGNOSTIC`, `HEDGE_CONTRACT_DIAGNOSTIC`. 전체 JSON 리포트: `output/113_osn_gs_chart_contract_diagnostic/chart_representation_contract_diagnostic_report.json`.

## 15. 테스트

canonical/production 코드는 무수정. 새 devtools 헬퍼 함수(`blob_domain_shape`, `design_matrix_rank_diagnostics`, `_bin_by_quantile`, `_distribution`)를 순수 numpy/scipy 로직으로 추출해 단위 테스트 가능하게 만들었다.

- `tests/test_chart_representation_contract_diagnostic.py`(신규, 12개): 사각형/링 모양/L자형/2-hole 라스터 마스크의 bbox·occupancy·hole 회계 정확성(5), 평면/퇴화 chart의 design-matrix rank/conditioning 및 결정론적 서브샘플링(3), 분포·quantile-binning 유틸리티 정확성(4).
- `.venv/Scripts/python.exe -m pytest tests/test_chart_representation_contract_diagnostic.py -q` → **12 passed**.
- 실측 스크립트는 `--max-views 6` smoke test로 전체 파이프라인이 오류 없이 끝까지 도는 것을 먼저 확인했다(1차 시도에서 `bbox_area` 미정의 버그 발견 → 즉시 수정 후 재실행 성공). 이후 전체 161개 뷰로 재실행(런타임 1039.9초). 161개 뷰 위상 재생 결과가 WL107/109/111/112의 알려진 수치(36.77%/45.02%)와 다시 한번 정확히 일치함을 확인했고, `valid_chart_count=14900`이 WL112와 정확히 일치해(동일 blob 구성 재사용을 확인) 이번 배치의 새 진단 코드가 기존 파이프라인을 변형하지 않았음을 검증했다.
- 전체 pytest는 재실행하지 않았다(directive 지시: canonical/production 코드 무수정, 순수 진단 경로로 유지).
