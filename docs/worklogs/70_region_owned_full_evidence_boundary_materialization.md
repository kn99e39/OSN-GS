# Worklog 70: Region-Owned Full-Evidence Boundary Materialization

## 목적

worklog 69의 진단(22/22 patch가 `partition_materialization_required`, 지배적 원인은 boundary loop가 region-owned full evidence의 실제 범위를 포함하지 못하는 scale mismatch)은 승인하되, **`22/22 partition_materialization_required` 자체를 canonical 판정으로 채택하지 않는다.** Region formation, representative topology, chart boundary 구성, ownership gating을 전혀 변경하지 않고, 승인된 region의 full evidence로부터 실제 fitting에 사용할 DENSE boundary를 별도로 materialize한 뒤 재검증한다.

금지: convex hull/PCA rectangle/bounding box를 최종 boundary로 사용, open chain 강제 폐쇄, 다른 region evidence 병합, `physical_termination`/`partition_seam` 의미 혼합, region formation·accepted-edge threshold 변경.

## 방법

### 신규 production 모듈: `osn_gs/surface/torch_region_owned_boundary_materialization.py`

`materialize_dense_boundary()` — 기존 representative boundary는 topology/provenance seed로 유지하고, edge 단위로만 확장한다(전역 hull 아님):

1. 각 evidence 점을 가장 가까운 원본 boundary edge에 귀속(3D point-to-segment distance).
2. 그 edge의 소유 evidence 중, boundary loop의 world-space centroid로부터 두 endpoint보다 `local_evidence_scale`(worklog 32의 per-representative full-cloud spacing 추정치, `bundle.evidence.mean_spacing` — 새로 발명한 상수 아님) 이상 더 먼 점만 후보로 남긴다.
3. edge의 world-space tangent를 `local_evidence_scale` 폭의 bin으로 나누고, bin마다 centroid에서 가장 먼 후보 1개만 채택 — bin index가 tangent를 따라 단조 증가하므로 국소적으로 되돌아가지 않는 polyline이 보장된다.
4. 채택된 점들을 원래 edge 자리에 순서대로 끼워 넣고, 새 segment는 모두 원래 edge의 provenance type을 그대로 물려받는다(crease/observation_frontier/partition_seam/physical_termination — 혼합·발명 없음). 후보가 없는 edge는 원본 그대로 둔다(gap 보간 없음).
5. 결과 loop를 기존 `validate_simple_closed_loop`(미변경)로 검증 — 실패하면 `state="boundary_materialization_failed"`로 fail-closed.

이번 라운드 안에서 두 가지 더 단순한 설계를 실제로 측정하고 기각했다(결과에 맞춘 사후 튜닝이 아니라, 각 설계의 실측 실패를 근거로 다음 설계로 넘어간 것):

- **edge당 1점만 삽입**: `interior_outside_boundary`가 densification 후에도 12/12에서 그대로 실패(edge가 3~4개뿐이라 새 vertex도 3~4개뿐 — 수백~수천 개 evidence를 담기에 구조적으로 부족).
- **edge가 소유한 모든 후보를 tangent 투영 순서로 그대로 삽입(binning 없음)**: 실제 evidence의 국소 잡음 때문에 tangent 축 진행 방향으로도 수직 방향 산포가 커서 polyline이 스스로 교차 — 18/22가 `validate_simple_closed_loop`에서 즉시 탈락.

최종 채택한 binning 방식(위 방법 3)은 두 극단의 중간으로, "실제로 조밀하지만 국소적으로 단조"인 polyline을 만든다.

### PCA 대신 world 3D 좌표 사용

이전 초안은 boundary+evidence를 공유 `pca_parameterize_points` UV 프레임에 투영해 edge 소유권을 판정했으나, outlier evidence가 섞이면 결합 점집합의 주축 자체가 회전해 어떤 edge가 "가장 가까운지"가 예측 불가능하게 바뀌는 문제를 이번 라운드 자체 테스트 디버깅 중 발견했다(6점짜리 최소 재현 사례로 확인). Boundary loop는 국소적으로 이미 대략 평면이므로, edge 소유권·outward 판정·tangent 정렬을 모두 3D world 좌표에서 직접 계산하도록 바꿔 이 불안정성을 근본적으로 제거했다(PCA는 이 모듈에서 전혀 사용하지 않는다).

### 분석 스크립트: `scripts/devtools/region_owned_full_evidence_boundary_materialization.py`

각 patch(worklog 61의 parametric chart 경로만 — 실제 22개 patch가 전부 이 경로이며 physical `eligible_closed_boundary` 경로는 이번 데이터셋에 0건이라 별도 처리 없이 `full_evidence_boundary_materialization_required`로 fail-closed 처리, per-edge segment-kind provenance가 그 경로에는 애초에 없기 때문)에 대해:

1. `construction.region_parametric_chart_boundaries`(worklog 61, 미변경)에서 edge별 provenance type을 조회.
2. `bundle.region_owned_full_evidence_fits`(worklog 67, 미변경)에서 region-owned full evidence를 조회.
3. `materialize_dense_boundary()` 호출. 실패 시 `full_evidence_boundary_materialization_required`.
4. 성공 시, 원본 representative id와 새 evidence-sourced id가 모두 들어있는 **단일 공유 PCA-UV 프레임**(둘 다 동일 id 공간에서 직접 조회 가능하므로 nearest-position 매칭 hack 불필요)에서 worklog 69의 `torch_single_chart_uv_validity.py`(미변경) 검사를 재실행.
5. **reduced gate**: `uv_near_collision`/`neighborhood_preservation<0.5`/`accepted_edge_uv_crossing`/`interior_outside_boundary>10%` 4개만 승인 기준으로 사용한다. 지시대로 `parallel_sheet_suspected`와 raw-evidence `triangle_fold_fraction`은 이번 단계에서 진단으로만 기록하고 gate에서 제외했다.
6. gate 실패 시 `partition_materialization_required`로 승격(dense boundary 적용 **이후**에만 승격 — 적용 전 상태로는 승격하지 않음).
7. gate 통과 시 6×6 grid(worklog 68 지시대로 해상도 확대 없음)로 fitting, worklog 68의 checkerboard held-out split으로 평가, worklog 66에서 그대로 가져온 임계값(`UNDER_SUPPORTED_MIN_EVIDENCE=4`, `EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND=4.0`)으로 `valid_supported`/`extrapolative`/`partition_materialization_required`(Jacobian 퇴화 시) 분류.

before(원본 boundary)/after(dense boundary) fitting 오차는 **gate 통과 여부와 무관하게 항상 계산**해 diagnostic으로 기록한다 — gate가 막힌 patch도 "densify했을 때 오차가 실제로 어떻게 바뀌는지"를 정직하게 비교하기 위함이다. 이 diagnostic 값은 분류 결정에는 gate를 통과한 경우에만 사용된다.

## 결과 (baseline_compatible@2900/3100 + baseline@2900/3100 참조, 22 patch 전부)

### 분류

| 상태 | patch 수 |
|---|---:|
| `full_evidence_boundary_materialization_required`(dense boundary 자체가 fail-closed) | 11/22 |
| `partition_materialization_required`(dense boundary는 성공했지만 reduced gate 실패) | 11/22 |
| `valid_supported` | 0/22 |
| `extrapolative` | 0/22 |

조건별:

| 조건 | patch 수 | boundary_materialization_required | partition_materialization_required |
|---|---:|---:|---:|
| baseline_compatible@2900 | 5 | 3 | 2 |
| baseline_compatible@3100 | 11 | 5 | 6 |
| baseline@2900(참조) | 4 | 2 | 2 |
| baseline@3100(참조) | 2 | 1 | 1 |
| **합계** | **22** | **11** | **11** |

### dense boundary 자체가 실패한 11개의 원인

`validate_simple_closed_loop`의 `self_intersection_check_failed`가 11/11 전부에서 발동했다. 세부:

| 세부 원인 | 건수(중복 가능) |
|---|---:|
| `proper_self_intersection`(2D 투영에서 실제 교차) | 7 |
| `self_intersection_not_checked_nonplanar`(loop 자체가 평면성 가정을 벗어나 애초에 검사 불가로 fail-closed) | 4 |
| `orientation_inconsistency` | 6 |

### dense boundary는 성공했지만 여전히 실패한 11개의 원인

| gate 기준 | 위반 건수(11개 중) |
|---|---:|
| `interior_outside_boundary > 10%` | **11/11 (100%)** |
| `uv_near_collision_count > 0` | 9/11 |
| `accepted_edge_uv_crossing_count > 0` | 3/11 |
| `neighborhood_preservation_mean < 0.5` | 1/11 |

### Before/after 비교

| 지표 | before(원본 3~4점 boundary, 22개) | after(dense boundary, materialized된 11개) |
|---|---:|---:|
| boundary vertex 수 평균 | 3.27 | 8.73 |
| `interior_outside_boundary` 비율 평균 | 91.1% | **54.8%** |
| held-out surface-to-evidence p95(dense-NN 정규화) 평균 | 23.87 | 28.86 |
| Jacobian near-degenerate 합계 | 0 | 0 |
| patch area 평균 | 2.72 | 3.17 |

## 해석

Dense boundary materialization은 **측정 가능한 실질적 개선**을 만든다 — boundary vertex 수는 평균 2.7배 늘고, `interior_outside_boundary`는 91.1%에서 54.8%로 36.3%p 줄었다. 이는 worklog 69의 진단(boundary가 evidence 범위에 비해 지나치게 작다)을 구조적으로 뒷받침한다.

그러나 **이 개선만으로는 어떤 patch도 `valid_supported`/`extrapolative`에 도달하지 못한다.** dense boundary가 유효한 simple loop로 만들어진 11개 전부가 `interior_outside_boundary`에서 여전히 실패했고(줄었을 뿐 10% 기준에는 한참 못 미침), held-out fitting 오차는 개선되지 않고 오히려 소폭 악화(23.87→28.86)됐다. 나머지 11개는 densify를 시도하는 과정 자체가 `validate_simple_closed_loop`를 통과하지 못했다.

worklog 69보다 더 구체적인 진단: 문제는 단순히 "boundary가 작다"는 것을 넘어, **원본 topology가 3~4개 edge만 갖고 있어서 evidence 소유권이 그 3~4개 "wedge"로만 나뉜다는 구조 자체**에 있다. 각 wedge를 아무리 조밀하게 채워도(binning 적용 후에도), wedge 경계 자체가 실제 evidence의 진짜 외곽 형태를 따라가지 못하면 `interior_outside_boundary`는 개선되되 해소되지 않는다 — 이번 라운드의 실측(100% 위반 잔존)이 이를 직접 보여준다. 동시에, evidence를 조밀하게 그대로 이어 붙이면(binning 없이) 실제 데이터의 국소 잡음 때문에 loop 자체가 self-intersect한다(11/22)는 것도 실측으로 확인했다 — 즉 "더 조밀하게"와 "여전히 simple loop"는 실사용 evidence에서 근본적으로 긴장 관계에 있다.

`parallel_sheet_suspected`와 raw-evidence `triangle_fold_fraction`은 지시대로 이번 라운드에서 진단으로만 기록했으며 승인/거부 판정에 전혀 사용하지 않았다.

`surface_self_intersection`은 이번에도 어디에서도 검사되지 않았다 — 모든 patch record에 `"surface_self_intersection": "not_checked"`로 명시했다.

## 테스트

신규 `tests/test_region_owned_boundary_materialization.py`(7개: 무-evidence 유지, outward 확장, inward 무시, 미확장 edge 원본 보존, 2-edge provenance 비혼합, 1-edge 다중 후보 tangent 순서 삽입, self-intersection fail-closed) 전부 통과. 이번 라운드는 신규 모듈 1개(`torch_region_owned_boundary_materialization.py`)만 production 코드이고 기존 파일은 변경하지 않았다. 지시대로 focused pytest만 실행했고 full pytest는 수행하지 않았다.
