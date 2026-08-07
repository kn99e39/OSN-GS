# Worklog 69: Single-Chart Parameterization Validity and Repair

## 정정: worklog 67/68 합계 오류

worklog 67/68는 patch 총계를 "21개"로 잘못 표기했다. 실제 조건별 patch 수(5+11+4+2)는 **총 22개**다. worklog 67의 "20/21(95%) extrapolative"는 **21/22(95%)**로, worklog 68의 "20/21(95%) overfitting"은 **20/22(90.9%)**로 정정한다. 조건별 세부 구성:

| 조건 | patch 수 | worklog 68 분류 |
|---|---:|---|
| baseline_compatible@2900 | 5 | overfitting 5 |
| baseline_compatible@3100 | 11 | overfitting 9, capacity_insufficient 1, inconclusive 1 |
| baseline@2900(참조) | 4 | overfitting 4 |
| baseline@3100(참조) | 2 | overfitting 2 |
| **합계** | **22** | overfitting 20, capacity_insufficient 1, inconclusive 1 |

worklog 67/68 본문과 Master doc, 메모리도 이 수치로 정정했다.

## 목적

각 region이 실제로 하나의 regular NURBS chart로 표현 가능한지(single-chart parameterization validity)를 검증하고, invalid single-chart가 worklog 68의 extrapolation/local folding 원인인지 확인한다. Region formation, chart boundary, ownership gating은 전혀 변경하지 않는다.

## 방법

신규 `osn_gs/surface/torch_single_chart_uv_validity.py`. 각 patch의 boundary+region-owned full evidence(worklog 67, 미변경)에 대해 기존 `pca_parameterize_points`(미변경)로 UV를 계산하고:

- **UV duplicate/near-collision**: UV 최근접 거리가 median spacing의 5% 미만인 쌍
- **3D-kNN vs UV-kNN neighborhood preservation**: k=8 최근접 이웃 집합의 Jaccard overlap
- **accepted-edge UV crossing**: region의 기존 `internal_accepted_edge_ids`를 UV에 투영해 `torch_boundary_self_intersection._segments_intersect`(기존 함수 재사용, 재구현 없음)로 교차 검사
- **local triangle orientation/fold**: scipy Delaunay로 UV 위에 삼각분할(UV 자체는 정의상 항상 winding-consistent라 이 자체는 fold를 만들 수 없음) 후, **UV상 인접한 삼각형 쌍의 3D normal 부호 일치**를 검사(worklog 68의 local-fold 관례를 raw evidence mesh에 그대로 적용)
- **interior evidence가 boundary 내부에 있는지**: UV 공간에서 ray-casting point-in-polygon
- **UV area distortion**: 삼각형별 UV 면적/3D 면적 비율
- **parallel sheet/multi-mode 의심**: evidence를 region의 dominant normal axis에 투영해 정렬 후 최대 gap을 탐지(양쪽에 전체의 10% 이상씩 있는 gap만 후보로 인정 — 아래 버그 참고)

**5개 조건(uv_near_collision/neighborhood_preservation<0.5/edge_crossing/fold_fraction>5%/interior_outside>10%/parallel_sheet_suspected) 중 하나라도 위반하면 `partition_materialization_required`, 전부 통과하면 `uv_valid`.**

동일 full evidence를 기존 checkerboard 기법을 N-way로 확장해 25%/50%/100% 결정론적 spatial subsampling하고, 6×6 grid로 raw error와 dense-NN 정규화 error를 비교했다(그리드 해상도는 worklog 68 지시대로 그대로 6×6 유지).

Partition 시도는 parallel-sheet 의심 patch에 대해서만, region의 기존 `internal_accepted_edge_ids`가 normal-axis 클러스터 사이를 건너는 edge 비율이 5% 이하일 때만(즉 기존 topology가 이미 두 클러스터를 거의 분리해 놓은 경우만) 허용했다 — PCA rectangle, convex hull, 임의 seam은 사용하지 않았다.

### 발견·수정한 결함(분석 스크립트, production 아님)

첫 실행에서 `parallel_sheet_suspected`가 22/22 patch에서 gap_ratio 39~2088라는 비현실적인 값으로 발동했다. 원인은 "최대 gap"을 양쪽 클러스터 크기와 무관하게 찾고 있었기 때문 — 실사용 ADC 학습 데이터에는 항상 극단적 outlier Gaussian이 소수 존재하므로(이번 세션 전체에서 반복 확인된 사실), outlier 1개가 나머지 수백 개와의 사이에 거대한 "gap"을 만들어 거짓 신호를 냈다. 양쪽에 최소 10%씩 있는 gap만 후보로 인정하도록 수정 — 재실행 결과 gap_ratio가 7~25로 낮아졌다(여전히 threshold 3.0 이상이라 발동은 유지되지만, 훨씬 그럴듯한 규모). 신규 테스트로 "outlier 1개는 감지되지 않아야 한다"를 직접 검증.

## 결과

**22개 patch 전부(100%) `partition_materialization_required`, `uv_valid`는 0개다.**

기준별 위반 비율(22개 중):

| 기준 | 위반 수 |
|---|---:|
| triangle_fold_fraction > 5% | 22/22 |
| interior_outside_boundary > 10% | 22/22 |
| parallel_sheet_suspected | 22/22 |
| uv_near_collision_count > 0 | 17/22 |
| accepted_edge_uv_crossing_count > 0 | 4/22 |
| neighborhood_preservation_mean < 0.5 | 3/22 |

**neighborhood_preservation은 대부분 건강하다**(3/22만 0.5 미만, 나머지는 0.39~0.94 분포, 대다수 0.5~0.8) — 이는 PCA-UV가 3D 국소 구조를 심각하게 훼손하는 "진짜 다중 시트/접힘" 문제가 지배적 원인이 **아님**을 시사한다.

**interior_outside_boundary는 예외 없이 극단적이다**(대부분 90~100%, 최소도 52%). 원인은 boundary loop가 대부분 3~4개 representative 점으로만 구성된 극소 다각형인 반면(worklog 66/67의 반복 관찰), region-owned full evidence는 18~2560개로 훨씬 넓은 실제 영역에 퍼져 있기 때문이다 — 3~4점짜리 다각형은 어떤 2D 투영에서도 수백~수천 개 점의 실제 공간 범위를 물리적으로 포함할 수 없다. **이것은 다중 시트/접힘의 증거라기보다, boundary loop 자체가 region의 실제 evidence 범위에 비해 지나치게 작다는 별개의 구조적 발견이다.**

triangle_fold_fraction도 대부분 15~50%로 높고(2/26=8%인 예외 1건 제외), UV area distortion p95도 2.8~217배로 넓게 분포한다 — worklog 68이 찾은 fitted-surface 24×24 grid 기준 local fold(1~3%)보다 raw evidence mesh 기준으로는 국소 요철이 훨씬 두드러진다는 뜻이다.

parallel_sheet_suspected는 22/22에서 발동(gap_ratio 7.2~24.6)했지만, 버그 수정 후에도 100% 발동률이라는 점에서 이 특정 임계값(3.0)이 실사용 밀집 evidence에 다소 민감할 수 있다는 점을 정직하게 남겨둔다 — threshold를 결과에 맞춰 조정하지 않았으므로 사전에 고정한 값 그대로 보고한다.

**6×6 fidelity of UV-valid patch: 해당 없음(uv_valid가 0개이므로 보고할 대상이 없다).**

**density subsampling(25%→100%) 결과**: raw p95 error 비율(25%/100%) median=1.15, mean=1.21; dense-NN 정규화 error 비율 median=1.20, mean=1.20 — **raw와 정규화 오차가 거의 같은 비율로 함께 움직인다.** 정규화 척도만 독립적으로 density에 흔들리는 패턴은 없다 — worklog 68의 `metric_density_dependent` 0건 결론과 일관된다.

**partition 적용 결과: 22/22 전부 `no_safe_partition_derivable_from_existing_accepted_topology`로 fail-closed됐다. 실제로 partition이 적용된 patch는 0건이다.** 모든 parallel-sheet 의심 patch에서, region의 기존 accepted-edge graph를 normal-axis 클러스터로 나눠 봤을 때 cross-edge 비율이 5%를 초과했다 — 즉 기존 topology가 두 클러스터를 이미 분리해 놓은 안전한 근거가 어디에도 없었다. 지시대로 자동 분할하지 않고 fail-closed 상태로 남겼다.

**surface self-intersection은 이번에도 어디에서도 검사되지 않았다** — 모든 patch record에 `"surface_self_intersection": "not_checked"`로 명시했다.

## 테스트

신규 `tests/test_single_chart_uv_validity.py`(13개, global reversal과 local fold 구분·outlier가 parallel-sheet로 오인되지 않음을 포함) 전부 통과. 이번 라운드는 신규 모듈 1개만 production 코드이고 기존 파일은 변경하지 않았다. 지시대로 focused pytest만 실행했고 full pytest는 수행하지 않았다.
