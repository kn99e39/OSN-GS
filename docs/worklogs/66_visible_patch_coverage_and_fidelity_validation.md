# Worklog 66: Visible Patch Coverage and Fidelity Validation

## 목적

worklog 65에서 정상화된 `baseline_compatible` 2900/3100 checkpoint에서 생성된 visible NURBS patch가 실제 observed surface를 충분하고 정확하게 재구성하는지 검증한다. `covariance_knn`은 과분할 artifact 비교용으로만, Graphdeco baseline은 동일 파이프라인을 실제 baseline PLY에 그대로 돌린 참조값으로 사용한다.

## 방법

`scripts/devtools/visible_patch_fidelity_validation.py`. 각 checkpoint에서 `TorchOSNGSPipeline._construct_canonical_with_full_evidence()`(기존 production 경로, 미변경)를 실행해 얻은 모든 materialized patch(physical + parametric chart)에 대해:

- **point-to-surface**: patch의 evidence(boundary+interior)에서 24×24 샘플링된 NURBS 표면까지 최근접 거리
- **surface-to-evidence**: 샘플 표면점에서 evidence까지 최근접 거리(역방향 — surface가 evidence 없이 뻗어나간 정도)
- 위 두 값을 patch 자신의 evidence 최근접 이웃 간격(local evidence scale)으로 정규화한 median/p90/p95/max
- patch 면적(샘플 그리드의 quad 면적 합), Jacobian 특이값/조건수(`compute_parametric_jacobian_metrics`, 기존 재사용), orientation consistency(`compute_orientation_consistency`, 기존 재사용), self-intersection(`validate_simple_closed_loop`, 기존 재사용 — 이미 materialization 단계의 admission gate이므로 통과한 patch는 원칙적으로 항상 만족)
- boundary segment provenance(parametric chart는 `RegionParametricChartBoundary.segment_kind_counts()`로 physical_termination/crease/observation_frontier/partition_seam 비율, physical closed-loop는 전부 physical_termination)
- scene 단위: 전체 accepted evidence 대비 covered/gap 비율, patch 쌍 간 evidence-ID Jaccard overlap(>0.3) 및 공간적 overlap(sampled point 50% 이상 근접) 탐지

5-way 분류(`valid_supported`/`under_supported`/`extrapolative`/`unsafe_geometry`/`duplicate_or_overlapping`)의 threshold는 전부 기존 코드 관례를 재사용했다(결과에 맞춘 조정 없음):
- `under_supported`: 지원 evidence < 4 — `RegionFormationConfig.core_region_typical_min_size`(기존)
- `extrapolative`: point-to-surface 또는 surface-to-evidence의 정규화 p95 > 4.0 — `RegionFormationConfig.local_backbone_max_normalized_distance`(기존)
- `unsafe_geometry`: self-intersection 또는 Jacobian degenerate cell 존재
- `duplicate_or_overlapping`: 위 오버랩 조건

## 구현 결함 발견 및 수정 (이번 분석 스크립트 자체, production 아님)

3-member 초소형 region은 `interior_points`와 `ordered_boundary_points`가 완전히 동일한 Gaussian 집합이다(별도 core/boundary 구분이 없음). 이를 그대로 concat하면 evidence에 정확히 중복된 행이 생겨 nearest-neighbor 기반 local scale이 0으로 collapse되고, 모든 정규화 거리가 수만 배로 폭발했다(예: `surface_to_evidence_p95_normalized=88042`). `torch.unique(evidence, dim=0)`로 evidence를 dedup하고 `supporting_evidence_count`도 dedup된 개수로 보고하도록 수정 후 재실행 — 이 발견 이전 초기 실행 결과(모든 patch가 `extrapolative`로 분류)는 폐기하고 이 worklog의 수치로 대체한다.

## 부수 성능 수정 (production, 이번 검증 과정에서 요청받아 별도로 진행)

이 분석을 실행하던 중 "GPU VRAM은 많이 먹는데 실제 연산은 느리다"는 지적을 받아 cProfile로 실제 병목을 추적했다. `osn_gs/surface/torch_density_preserving_representative_selection.py`의 `_boundary_evidence_swap_in`이 358초 중 316초(자기 자신의 bytecode 실행 시간, 하위 호출 아님)를 차지했다 — 원인은 pool adjacency graph를 만드는 이중 for-loop에서 `float(pool_pairwise[a, b])`로 CUDA tensor를 한 쌍씩 스칼라로 꺼내면서, 쌍마다 GPU 동기화(device-to-host sync)가 발생했기 때문이다(O(pool²) 개의 개별 GPU sync). 이게 "VRAM은 크게 잡고 있는데 GPU 사용률은 10%대"인 증상의 정확한 원인이었다.

수정: 동일한 boolean adjacency 관계를 `pool_pairwise <= edge_radius`로 한 번에 벡터화 계산하고, 결과 edge만 일괄로 Python으로 전송(`torch.triu_indices` + boolean masking). 반복 순서(row-major a<b)를 원본 이중 loop과 동일하게 유지해 adjacency list 순서 — 따라서 이후 BFS/component-count/eviction 판정 — 가 전혀 바뀌지 않도록 했다. 정렬 키에 있던 동일 패턴(`float(pool_nearest_distance[local])`)도 함께 일괄 추출로 수정.

**검증**: `git stash`로 수정 전/후를 A/B 대조 — 동일 checkpoint(`baseline_compatible@2900`)에서 region_count/patch_count/classification/patch_area(소수점 끝자리까지)/region_id가 **완전히 동일**, 소요시간만 338.22초→31.31초(10.8배). Focused pytest 86개(`test_density_preserving_representative_selection`, `test_full_cloud_continuation_shell`, `test_representative_graph_scale`, `test_gaussian_covariance_frame`, `test_gaussian_structural_reliability`, `test_gaussian_surface_region_formation`, `test_gaussian_manifold_affinity`, `test_surface_region_invariance`, `test_surface_region_validation`, `test_surface_region_adversarial_validation`, `test_local_evidence_scale`, `test_region_parametric_chart_boundary(_materialization)`) 전부 통과. 6개 조건 전체 재실행 시간도 이전 라운드의 약 40분에서 크게 단축됐다.

## 결과

### Scene-level

| 조건 | iter | region | patch | valid | under-supp | extrapol | unsafe | dup | gap | area |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_compatible | 2900 | 7 | 5 | 0 | 4 | 1 | 0 | 0 | 11(41%) | 0.507 |
| baseline_compatible | 3100 | 19 | 11 | 4 | 4 | 0 | 3 | 0 | 31(43%) | 2.04 |
| covariance_knn | 2900 | 159 | 90 | 24 | 41 | 3 | 21 | 1 | 392(55%) | 22.6 |
| covariance_knn | 3100 | 153 | 84 | 28 | 35 | 7 | 12 | 2 | 440(59%) | 142.2 |
| baseline(참조) | 2900 | 8 | 4 | 2 | 2 | 0 | 0 | 0 | 26(65%) | 0.303 |
| baseline(참조) | 3100 | 3 | 2 | 1 | 1 | 0 | 0 | 0 | 3(30%) | 0.202 |

**valid_supported 비율**: baseline_compatible 0%→36%, covariance_knn 27%→33%, baseline(참조) 50%(양쪽 다). `under_supported`(evidence<4)가 모든 조건에서 가장 큰 비중을 차지한다 — 3-member 최소 triangle region이 흔하기 때문이며(전체 accepted evidence 27개를 7개 region이 나눠 가지면 평균 3.9개), 이는 실제 3k 학습 규모에서 region 자체가 작다는 worklog 65의 관찰과 정확히 일치한다. `unsafe_geometry`(Jacobian degenerate)도 예외 없이 evidence=3인 최소 patch에서만 발생한다 — 3점짜리 경계로 NURBS를 피팅할 때의 구조적 취약성이지 구현 결함이 아니다.

시각화(대표 성공 2건, 실패 4건 — under_supported/unsafe_geometry/extrapolative/duplicate 각 1건)와 전체 표는 아티팩트 참고: **https://claude.ai/code/artifact/5831b65d-a3c8-4a87-8d93-6858a600c2cd**

## 완료 기준 대조

- **supporting region/accepted evidence 수, coverage 비율**: 확인함(위 표, region당 평균 3.9~4.6개 evidence).
- **point-to-surface / surface-to-evidence 오류, 정규화 median/p95/max**: 측정 완료(아티팩트 patch 카드). `valid_supported` patch는 예외 없이 정규화 p95가 4.0 미만(예: region 11 fwd 0.19/bwd 3.38).
- **patch 면적/전체 coverage, overlap/gap**: measured. covariance_knn은 patch 수는 많지만 gap 비율(55~59%)이 baseline_compatible(41~43%)보다 오히려 나쁘다 — 많은 patch가 실제로는 evidence를 잘 덮지 못하는 파편임을 보여준다.
- **normal consistency, Jacobian degeneracy, self-intersection**: 측정 완료. self-intersection 위반은 0건(애초에 materialization 단계에서 걸러짐, 예상대로). Jacobian degenerate는 3-evidence 최소 patch에서만 발생.
- **boundary provenance, partition seam 비율**: baseline_compatible 3%, covariance_knn 7%, baseline 0% — 전부 낮음(대부분 patch가 physical_termination 우세).
- **5-way 분류 + 대표 patch 시각화**: 완료(아티팩트).
- **scene 전체 coverage/중복률/누락률**: 완료 — covariance_knn의 duplicate_or_overlapping은 소수(1~2건)뿐이라 과분할의 본질은 "중복"이 아니라 "파편화"(작은 patch 대량 생산)임을 확인.

## 결론

baseline_compatible이 만드는 patch 수는 적지만(5~11개), 그중 `valid_supported` 비율은 3100에서 36%로 covariance_knn(33%)과 비슷하거나 오히려 근소하게 높고, gap 비율은 더 낮다(43% vs 55~59%) — **worklog 65의 over-segmentation 완화가 진짜 품질 개선과 함께 온다는 것을 뒷받침**한다. 다만 baseline(참조)의 50%에는 아직 못 미치고, 대다수 patch가 `under_supported`(evidence 3~4개)인 것은 두 OSN-GS 조건 모두에 공통된 근본적 한계로 남아있다 — region 자체가 실사용 3k 규모에서 매우 작다는, 이번 결과와 무관한 별도 문제(§2.2의 "candidate evidence density" 병목)로 보인다.

## 테스트

Production 코드 변경(`torch_density_preserving_representative_selection.py`, 순수 성능 최적화)에 대해 focused pytest 86개 실행, 전부 통과. 지시대로 full pytest는 실행하지 않았다. `visible_patch_fidelity_validation.py`(분석 스크립트, production 아님)의 dedup 수정에는 별도 pytest 없음.
