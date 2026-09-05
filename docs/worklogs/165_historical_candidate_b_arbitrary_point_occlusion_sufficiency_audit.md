# Worklog 165 — Historical Candidate-B Arbitrary-Point Occlusion Sufficiency Audit Against Analytic Blocker Ground Truth

## 1. 의도 정렬

이번 배치는 frozen Historical Candidate-B의 단일 명제만 감사했다.

`query_depth > renderer_median_depth`가 임의의 3D query가 같은 camera ray의 알려진 geometric blocker 뒤에 있음을 보장하는지 확인하는 것이 질문이다. Candidate-B, W160 aggregation, W161 paused 상태, W162–W164, canonical renderer 및 downstream geometry 경로는 변경하지 않았다.

## 2. 구현 충실도

새 diagnostic은 `devtools/demo/worklog_165_historical_candidate_b_arbitrary_point_occlusion_sufficiency_audit.py`에 격리했다. fronto-parallel rectangle, 28° oblique rectangle, sphere를 exact ray/surface intersection으로 계산하고, 고정된 64×64 camera와 fixed 2DGS surfel realization을 canonical qdepth forward path에 통과시켰다. coarse와 dense realization은 source에 선행 고정했으며 opacity, scale, FOV, background, resolution sweep은 하지 않았다.

Candidate-B 판정은 `observed_occluded.candidate_b_median_depth.classify_view`를 직접 호출했다. `h`, `mu`, epsilon, first-hit production path, transmittance rule은 사용하지 않았다.

## 3. Historical Candidate-B 보존

`candidate_b_median_depth.py`의 `median > 0`, `query_depth <= median`/`query_depth > median` ordering을 source hash와 focused test로 확인했다. W165의 실제 query 재판정도 같은 function을 사용했다. Candidate-B source, W160 global aggregation, canonical renderer kernel, TSDF/topology/Boundary First/NURBS/continuation은 수정하지 않았다.

## 4. Analytic ground-truth contract

- `GT_DIRECT_ACCESS`: camera→query segment가 analytic surface와 교차하지 않거나, first hit보다 query가 앞이다.
- `GT_BLOCKED`: analytic first intersection `z*`가 query보다 엄밀히 앞이다.
- `GT_BOUNDARY`: query가 analytic surface에 구성상 정확히 놓인 진단 case다.
- no-hit ray는 `GT_DIRECT_ACCESS_NO_BLOCKER`로 별도 보존했다.

`z*`와 `m`의 strict ordering은 raw 값으로 계산했고 exact equality는 별도 집계했다. counterexample을 margin 때문에 버리지 않았으며, float32 변환으로 query가 median event에 붙은 non-confirmed attempt도 report에 남겼다.

## 5. Fronto-parallel plane 결과

coarse/dense 모두 analytic hit ray는 2,304/4,096, no-hit ray는 1,792/4,096이었다. coarse는 valid median 4,096개 중 hit ray에서 `m < z* = 0`, exact equality 2,304, `m >= z*` 2,304였다. dense는 valid median 3,576개이며 동일하게 hit ray equality 2,304, `m < z* = 0`이었다.

이 fixture는 camera-depth convention과 equality/positive control을 통과했다. 동시에 coarse 1,792개, dense 1,272개의 no-hit ray에 valid median이 생겼다. coarse 대표 C query는 pixel `(row=32,col=0)`, `m=4.0`, `z(q)=4.5`, Candidate-B=`OCCLUDED`, GT=`GT_DIRECT_ACCESS_NO_BLOCKER`, `z(q)-m=0.5`였다. 이는 finite analytic surface 밖의 renderer support spill을 숨기지 않은 결과다.

## 6. Oblique plane 결과

coarse는 analytic hit 2,104, no-hit 1,992, valid median 3,714였다. hit-valid 중 `m < z*`는 748개(35.55%), `m >= z*`는 1,356개였고, raw `z*−m` distribution은 min `7.382e-09`, median `5.436e-08`, mean `1.066e-07`, p95/max `3.609e-07`이었다. strict-front 748개 중 interior 606개, 2-pixel diagnostic silhouette/support band 142개였다.

dense는 analytic hit/no-hit `2,104/1,992`, valid median 3,290이었다. `m < z*`는 1,010개(48.00%), `m >= z*`는 1,094개였고, raw `z*−m`은 min `1.402e-08`, median `7.966e-08`, mean `1.192e-07`, p95 `3.609e-07`, max `4.842e-07`이었다. interior strict-front는 848개였다.

oblique fixture의 일부 representative midpoint는 float32 query-depth 변환에서 `z(q)−m=0`으로 붙었고 Candidate-B=`OBSERVED`가 되었다. 이를 반례로 세지 않고 `nonconfirmed_counterexample_attempts`에 raw margin과 함께 보존했다. 이 수치적 경계와 별개로 sphere에서 큰 strict executable B 반례가 확인되어 결론에는 영향이 없다.

## 7. Curved-surface 결과

sphere는 coarse/dense 모두 analytic hit 1,356개였다. coarse는 no-hit 2,740, valid median 2,828, hit-valid 전부 `m < z*`였다. raw `z*−m`은 min `0.0004094`, median `0.1286535`, mean `0.1595260`, p95 `0.4162123`, max `0.6726638`이며 strict-front 중 interior 1,124, silhouette band 232였다.

dense는 no-hit 2,740, valid median 1,696, hit-valid 전부 `m < z*`였다. raw `z*−m`은 min `0.0004094`, median `0.0488567`, mean `0.0588175`, p95 `0.1478313`, max `0.3938419`이며 interior strict-front는 1,124였다. 따라서 현상은 coarse singular artifact가 아니며 dense realization에서도 유지된다.

대표 coarse executable B case는 `row=38,col=51`, `m=3.3358161`, `z(q)=3.6721480`, `z*=4.0084799`였다. Candidate-B=`OCCLUDED`, GT=`GT_DIRECT_ACCESS`, raw margins는 `z*−m=0.6726638`, `z*−z(q)=0.3363320`, `z(q)−m=0.3363318`이다.

대표 dense executable B case는 `row=34,col=52`, `m=3.6581535`, `z(q)=3.8550744`, `z*=4.0519955`였다. Candidate-B=`OCCLUDED`, GT=`GT_DIRECT_ACCESS`, raw margins는 `z*−m=0.3938419`, `z*−z(q)=0.1969210`, `z(q)−m=0.1969209`이다.

## 8. `m` versus `z*` accounting

전체 fixture별 총 ray/pixel, analytic hit/no-hit, valid median, `m<z*`, equality, `m>=z*`, no-hit valid median 및 distribution은 `output/165_historical_candidate_b_arbitrary_point_occlusion_sufficiency_audit/worklog_165_report.json`에 raw JSON으로 보존했다. `m<z*`와 no-hit valid median은 어느 것도 제거하지 않았다.

## 9. Strict counterexamples

coarse와 dense 각각 executable counterexample 4건을 확인했다. coarse/dense 모두 fronto-parallel C, oblique C, curved B, curved C가 Candidate-B=`OCCLUDED`로 재현됐다. 특히 curved B는 strict `m < z(q) < z*`이고 GT direct access인 독립 geometric 반례다. 이는 equality noise가 아니다.

## 10. Silhouette / support-spill attribution

silhouette은 analytic surface에서 projected 2-pixel 이내인 진단 구간으로만 분리했고 실패 case를 제외하는 필터로 쓰지 않았다. oblique와 sphere 모두 interior strict-front ray가 남았다. no-hit valid median은 finite analytic surface 밖 renderer support spill로 별도 `NO_BLOCKER_WITH_VALID_MEDIAN`에 계수했다.

## 11. Fixed denser replay

coarse grid 9×9/dense grid 17×17, sphere 9×18/dense 17×36을 사전에 고정했다. dense oblique는 interior `m<z*` 848개, dense sphere는 1,124개이며 dense sphere executable B margin도 위에 기록한 대로 strict하다. 따라서 architecture-changing counterexample은 dense replay에서도 유지된다.

## 12. Real-scene sanity replay

W160의 `DSC07960.JPG`, `DSC08003.JPG`, `DSC08043.JPG`와 기존 `tabletop`, `table_side_lower_geometry`, `vase_foreground_structure` ROI centroid만 사용했다. 각 ROI에서 offset `(-0.50, 0.0, 0.50, 1.00)`의 before/at/behind median ladder를 생성해 총 36 query를 Candidate-B로 재판정하고 world XYZ와 projection을 저장했다.

이 단계는 `NON-ORACLE REVIEW REFERENCE`다. real-scene physical blocker truth, TSDF sign, Gaussian Region, renderer contributor를 oracle로 사용하지 않았다.

## 13. Qualitative review exports

합성 fixture마다 `original_scene`, `observed_occluded`, `analytic_geometry`, `analytic_blocker_depth`, `renderer_median_depth`, `signed_depth_difference`, `categorical_ray_map`, `counterexample_rays`, `cross_section`을 PNG로 만들었고 각 visualization type에 UTF-8 README를 두었다. real-scene export는 camera가 아니라 `median_ladder_projection`과 `median_ladder_world` visualization type으로 구성했다. PPM은 생성하지 않았다.

## 14. Architecture result

**`BEHIND_MEDIAN_NOT_SUFFICIENT`**

strict analytic case에서 `m < z_query < z*`이고 frozen Candidate-B가 `OCCLUDED`를 반환했다. dense oblique/sphere에서도 interior strict ordering이 유지되므로 `SUFFICIENCY_FAILS_AT_RENDERER_SUPPORT_BOUNDARY`로 축소할 수 없다. 이 verdict는 Candidate-B를 패치하거나 대체하라는 제안이 아니며, 이번 batch에서는 실패를 보존하고 architecture review에서 멈춘다.

## 15. Retained / rejected / open

유지: Historical Candidate-B, POINT_QUERY_STATE, W160 per-view/global aggregation, W161 gap/paused status, W162–W164, canonical renderer behavior, `t_w`, Gaussian Region, TSDF, topology, Boundary First, NURBS, continuation, Eligibility.

거부: median replacement, expected/first-hit depth replacement, transmittance occlusion, contributor-aware arbitrary-query classifier, primitive observation을 point-query truth로 승격, threshold/epsilon tuning, TSDF sign을 occlusion oracle로 사용.

열린 항목: human architecture review, real-scene qualitative interpretation, synthetic analytic control 밖의 universal physical claim. W161 spatial Occlusion Domain construction은 재시작하지 않는다.

## 평가 및 검증

- focused W165 tests: `8 passed`.
- canonical synthetic replay: 3 fixture × coarse/dense 완료.
- real-scene sanity replay: 3 camera × 3 fixed ROI × 4 ladder query 완료.
- output: `output/165_historical_candidate_b_arbitrary_point_occlusion_sufficiency_audit/`.
- `output/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache/`는 대형 중간 cache이므로 temp mirror에 복사하지 않았고 report에도 제외 사실을 기록했다.
