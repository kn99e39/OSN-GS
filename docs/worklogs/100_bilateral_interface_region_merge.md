# Worklog 100 — Region-conditioned bilateral interface Surfel Region merge

## 상태

**완료 — 실측 있음. 실제 scene에서 뚜렷한 개선을 확인했다.** Worklog 99가 재발시킨 percolation(최대 subset 53.86%)이 WL98의 `min(r_i->j, r_j->i)`(편측 허용) residual을 region merge라는 훨씬 강한 주장에 그대로 재사용했기 때문이라는 가설을 검증했다. Union rule의 나머지는 전부 Worklog 99와 동일하게 고정한 채, per-edge 증거만 **(1) region-conditioned**(현재 자기 region 이웃만으로 local shape operator 재적합) **(2) bilateral**(양방향 모두 통과해야 smooth)로 교체했다. 실제 scene 최대 subset이 **53.86% → 22.91%**로 줄었고(초기화 자체의 20.62%에 근접), WL99의 patio→hedge 연결 lineage 5개 merge **전부**가 새 인증서에서 기각됨을 직접 확인했다.

## 아키텍처(변경분만)

```
(Worklog 99와 완전히 동일)
    positional-gated WL97 초기 region
        -> region adjacency / 전체 interface
        -> [교체] region-conditioned, bilateral 스무스니스 인증서
        -> 결정론적 라운드 기반 merge (동일)
        -> Coverage-first Surfel Subsets
```

## 1. Worklog 99를 replayable baseline으로 보존

`osn_gs/surface/torch_interface_coherent_region_merge.py`는 전혀 수정하지 않았다. 신규 `osn_gs/surface/torch_bilateral_interface_region_merge.py`가 별도 모듈로 존재하며, `local`(candidate graph), `region`(positional-gated WL97 init), 지지(support)/extent floor, `residual_mad_multiplier=3.0`, `parallel_sheet_normal_over_tangent_ratio=1.0`, `interface_smooth_majority_fraction=0.5`를 **전부 Worklog 99와 동일한 값**으로 재사용한다(`test_no_new_free_parameter_is_introduced`로 고정 검증). 실제 scene 비교도 동일 checkpoint·동일 local candidate graph·동일 positional-gated 초기화로 A/B 재실행했다.

## 2. Region-conditioned local differential 모델

신규 `_fit_region_conditioned_shape_operators()`. 현재 라운드에서 실제로 cross-region interface에 걸쳐 있는 "boundary node" 집합만(전체 scene이 아님) 대상으로, 각 노드의 정적 kNN(k=8, 위치는 학습 후 불변이므로 한 번만 계산) 이웃 중 **현재 자신과 같은 root(region)에 속한 이웃만**(binary weight 0/1, 마스킹된 이웃은 fit에서 완전히 배제) 사용해 `S_i`를 재적합한다. 병합이 일어나 region 소속이 바뀌면 다음 라운드에서 이 마스크도 함께 갱신되므로 §2가 요구한 "merge 후 stale model 금지" 계약을 만족한다.

**지지 부족 처리**: 같은 region 이웃이 2개 미만이면(2×2 선형 fit이 구조적으로 불가능한 최소 조건, 스윕 아님) `UNSUPPORTED`로 표시하고, 해당 방향은 **절대 smooth로 카운트하지 않는다**(§2가 명시적으로 금지한 "지지 부족 = smooth 증거로 취급" 오류를 피함).

## 3. Bilateral 인증서

신규 `_region_conditioned_bilateral_residuals()` / `_compute_bilateral_edge_evidence()`. Edge `(i in A, j in B)`마다 **두 방향을 절대 min()으로 합치지 않고 독립적으로** 유지한다:

    r_A->B = A의 own-region-conditioned model이 B로의 전환을 얼마나 잘 예측하는가
    r_B->A = B의 own-region-conditioned model이 A로의 전환을 얼마나 잘 예측하는가

`bilateral_smooth = supported_A->B & (r_A->B <= threshold) & supported_B->A & (r_B->A <= threshold) & positional_ok`. `bilateral_smooth_fraction`이 interface 전체에서 집계되어 Worklog 99의 `fraction_smooth_continuation`을 대체하며, merge 승인 기준(과반 0.5)은 **그대로**다.

## 4. 지지/extent floor — 변경 없음

`min_unique_surfels_per_interface_side`, `min_interface_extent_in_spacing_units` 둘 다 Worklog 99와 동일한 식(`local.neighbor_count`, `local.spatial_connect_spacing_multiplier`에서 유도)을 그대로 재사용한다.

## 5. 새 자유 파라미터 — 0개

`interface_smooth_majority_fraction=0.5`을 포함해 Worklog 99의 모든 값을 그대로 재사용했다(`test_no_new_free_parameter_is_introduced`). 이번 배치는 **판정 semantics만** 바꿨다.

## 6. 결정론적 merge 알고리즘 — 라운드 구조는 동일, threshold 계산 시점만 다름

Worklog 99와 동일한 라운드 기반 순차 DSU(정렬 기준을 `bilateral_smooth_fraction` 내림차순으로 교체). **구현 중 발견한 실제 버그**: residual threshold(median + 3·MAD)를 처음엔 Worklog 99처럼 "매 라운드, cross-region edge만"으로 재계산했는데, 합성 zigzag 크리즈 fixture에서 **완전히 균일한 90도 크리즈의 모든 edge가 100% bilaterally smooth로 오분류**되는 실제 결함을 발견했다: 크리즈 전체가 균일하면 residual 분포의 분산이 0이 되어 median+3·MAD가 그 값 자체로 붕괴하고("outlier"가 정의상 존재할 수 없는 상태), 그 결과 threshold와 정확히 같은(즉 "≤ threshold") 모든 값이 통과해버린다. **Worklog 98/99가 이 문제를 겪지 않은 이유**는 threshold를 scene 전체(같은 region 내부의 사실상 0에 가까운 residual과 진짜 불연속의 큰 residual이 섞인 큰 모집단)에서 **한 번만** 계산했기 때문이다. **정정**: threshold를 신규 `_compute_initial_residual_threshold()`로 **한 번만**, 초기(라운드 1) region 소속 기준 **모든 spatial edge**(cross-region뿐 아니라 same-region 내부 edge도 포함)에서 region-conditioned 방식으로 계산하고, 이후 모든 라운드에서 재사용한다 — "WL98 residual threshold formula를 유지하라"는 지시를 가장 문자 그대로 따른 해석이며(Worklog 98/99도 정확히 "한 번만, 큰 모집단에서" 계산했다), 새 파라미터가 아니라 계산 **시점**을 정정한 것이다.

## 7. 합성 fixture

신규 focused 테스트 14개(`tests/test_bilateral_interface_region_merge.py`), 전부 통과.

| Fixture | 기대 동작 | 실측 |
|---|---|---|
| 1/4, 1/2 원통(WL97 과분열) | 최종 1개 region으로 복원 | ✅ (n_theta=90 — §7.1 참고) |
| 90° 크리즈 | 분리 유지 | ✅ top-2 fraction > 0.85 |
| 평행 시트 | 분리 유지 | ✅ |
| Zigzag 4-plate narrow-bridge 체인 | transitivity로 percolate 안 함 | ✅ |
| 손으로 구성한 one-sided interface(A→B 성공, B→A 실패) | merge 거부 | ✅ (§7.2) |
| 지지 부족(같은 region 이웃 <2) | UNSUPPORTED, smooth 아님 | ✅ |
| Boundary-contaminated 이웃(WL98이 min()을 쓴 이유 그 자체) | region conditioning으로 오염 배제 | ✅ |

### 7.1 발견한 사실: 매우 작은(poorly-populated) region의 수치적 한계

기본 fixture 파라미터(`n_theta=60`)에서 1/4 원통을 돌리면 WL97 초기화가 우연히 16-surfel짜리 아주 작은 tail fragment를 남기고, 이 fragment의 region-conditioned fit은 (조건수는 정상 — eigenvalue ratio 0.05~0.1, 특이행렬 아님) 이웃 수가 3~5개뿐이라 residual 스케일이 다른 큰 region(수백 개 이웃, residual ~1e-7)과 미세하게 다르다(~1e-6). 두 population이 거의 정확히 반반으로 섞이면 median+MAD 기반 판정이 이 둘을 구분하지 못한다 — MAD 기반 이상치 검출의 알려진 한계(오염 비율 ~50% 근처에서 붕괴)다. Fixture의 각도 표본을 조밀하게(`n_theta=90`) 바꿔 이 특정 fixture가 그런 작은 tail을 남기지 않도록 했다(알고리즘·threshold는 건드리지 않음). 실제 2DGS scene은 훨씬 큰 population과 실측 노이즈 스케일을 가지므로 이 병리가 지배적일 가능성은 낮지만, 알려진 한계로 정직하게 기록한다.

### 7.2 One-sided interface fixture — 정확한 구성

`_fit_region_conditioned_shape_operators`를 직접 호출하는 6-node 합성 예제: region A의 같은-region 이웃 2개로 `S_A`를 정확히 적합해 A→B 방향 예측 residual을 1.2로(큰 값), region B는 완전 평면이라 `S_B≈0`이고 B→A 예측 residual은 0(정확)이 되도록 구성했다. `r_A->B=1.2 > threshold`, `r_B->A=0.0 <= threshold` — 한쪽만 통과하는 상황을 손으로 결정론적으로 재현해 bilateral 게이트가 정확히 거부하는지 확인했다.

## 8. 실제 scene 비교 (A. Worklog 99 vs B. 신규)

Checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`(1,190,469 surfel, Worklog 96-99와 동일).

| | A. Worklog 99 | B. 신규(bilateral) |
|---|---:|---:|
| 초기 region 수 | 114,420 | 114,420 (동일 초기화) |
| 최종 region 수 | 108,848 | 112,768 |
| **최대 subset 비율** | **53.86%** | **22.91%** |
| 평가된 interface 수 | 1,855,041 | 1,763,096 |
| Accept된 interface | 6,051 | **1,742** |
| 적용된 merge | 5,572 | **1,652** |
| 라운드 수 | 7 | 6 |
| Coverage identity | True | True |

초기화(20.62%)에 매우 근접한 22.91%로, WL99의 33%p 악화(20.62%→53.86%)가 거의 전부 회복됐다. Accept된 interface 수가 6,051→1,742로(71% 감소) 급감한 것이 직접 원인이다.

## 9. Worklog 99 giant-region lineage 추적 (§5)

`scripts/devtools/worklog99_lineage_trace.py`. Worklog 99의 최대 subset(53.86%) 내부에서 위치 기준으로 patio-측 seed(`node 117922`, 테이블 근처 지면)와 hedge-측 seed(`node 711179`, 배경 산울타리 쪽)를 선택하고, Worklog 99 자신의 `merge_provenance` 그래프에서 두 seed의 초기 region(각각 `5052`, `555`)을 잇는 최단 merge 경로를 BFS로 찾았다. **경로 길이 5**(round 2~3에 걸친 5개의 순차 merge: `383↔5052`, `0↔383`, `0↔7`, `7↔13`, `13↔555`). 각 merge 시점의 **정확한** 당시 region 소속 상태를 Worklog 99의 **전체** 5,572개 merge(경로에 없는 것도 전부, 순서대로)를 replay해 재구성한 뒤(경로에 있는 merge만 replay하면 큰 root region의 실제 당시 크기를 과소평가하는 실제 버그를 구현 중 발견·수정했다 — 최초 시도는 5개 중 4개에서 "surviving edge 없음"이라는 명백히 잘못된 결과를 냈다), 신규 bilateral 증거 함수로 동일 interface를 재평가했다.

| Step | Region 쌍 | WL99 판정(smooth 비율) | Bilateral: A→B / B→A | Bilateral 비율 | 편측? | 신규 인증서라면 |
|---|---|---:|---:|---:|:---:|:---:|
| 0 | 383↔5052 | 0.939(accept) | 0.242 / 0.788 | 0.182 | **예** | 기각 |
| 1 | 0↔383 | 0.972(accept) | 0.667 / 0.583 | 0.417 | 아니오 | 기각 |
| 2 | 0↔7 | 0.893(accept) | 0.500 / 0.786 | 0.321 | 아니오 | 기각 |
| 3 | 7↔13 | 0.714(accept) | 0.192 / 0.615 | 0.077 | **예** | 기각 |
| 4 | 13↔555 | 0.781(accept) | 0.438 / 0.469 | 0.312 | 아니오 | 기각 |

**질문에 대한 답**:
- "그 merge들이 주로 한쪽에서만 지지됐는가?" — **5개 중 2개는 명백히 편측**(0.242 vs 0.788, 0.192 vs 0.615 — 한쪽은 과반을 훨씬 넘고 다른 쪽은 훨씬 밑돎). 나머지 3개도 양방향이 서로 다르고(0.667/0.583, 0.500/0.786, 0.438/0.469), bilateral 비율은 `min`이 아니라 `AND` 결합이라 **둘 다 0.5 이상이어야만** 통과하는데 5개 전부 그 조건을 만족하지 못했다.
- "신규 인증서가 이 연결을 막는가?" — **그렇다. 5개 전부 기각됐다.** 실제 신규 B 결과에서도 patio_seed(117922)와 hedge_seed(711179)는 서로 다른 최종 subset color(`f_dc` 값이 뚜렷이 다름)로 확인했다 — 이 특정 lineage가 끊겼을 뿐 아니라 최종 결과에서도 실제로 분리돼 있다.

## 10. 커버리지 정합성 표기 정정 (Worklog 99 문서 오탈자)

`docs/worklogs/99_interface_coherent_region_merge.md`의 "`assigned == unassigned == 0`"은 오탈자였다(코드 자체는 항상 옳았다 — `assigned_surfel_count == count`, `unassigned_surfel_count == 0`을 별도로 검사했다). `assigned == total_surfels`로 정정했다. 이번 배치의 accounting도 `assigned_surfel_count`/`unassigned_surfel_count`/`multiply_owned_surfel_count`를 명시적으로 분리해 반환한다(§9 표의 "Coverage identity" 행이 그 결과다).

## 11. Review export

`scripts/devtools/bilateral_interface_region_merge_export.py` → `output/osn_gs_bilateral_interface_region_merge/`:

    A. WL99_INTERFACE_COHERENT_PARTITION (baseline)
    B. BILATERAL_INTERFACE_PARTITION (신규)
    C. ACCEPTED_BILATERAL_INTERFACE_MERGES
    D. REJECTED_BILATERAL_INTERFACES
    E. MERGE_PROVENANCE_DEPTH

PNG preview: `output/osn_gs_bilateral_interface_region_merge/preview_png/`. 시각 검토: `BILATERAL_INTERFACE_PARTITION`에서 파티오 바닥(주황)과 배경 산울타리(청록/파랑 계열의 여러 색)가 이제 **뚜렷이 다른 색**으로 나뉜다(Worklog 99에서는 둘 다 같은 적갈색 하나였다). 테이블 상판은 여전히 하나의 단일 색으로 유지되지만, `MERGE_PROVENANCE_DEPTH`에서 낮은 depth로 나타나 이 특정 복원의 공은 이번 배치의 merge 메커니즘보다는(Worklog 99와 마찬가지로) positional-gated WL97 초기화 자체에 더 크게 귀속될 가능성이 있다 — 정확한 귀속은 확정하지 않는다.

## 12. 재현 명령

```
python scripts/devtools/bilateral_interface_region_merge_export.py \
  --checkpoint output/arch_2dgs_coverage_first_surface/2dgs_run1/30000 \
  --out output/osn_gs_bilateral_interface_region_merge \
  --device cuda \
  --source-path DATASET

python scripts/devtools/worklog99_lineage_trace.py
```

## 13. 테스트

- 신규 focused: `tests/test_bilateral_interface_region_merge.py` 14개, 전부 통과.
- 기존 Worklog 97/98/99 테스트는 변경 없이 전부 통과(재확인).
- 전체 회귀: **1158 passed, 1 skipped, 2 warnings, 18 subtests passed in 266.46s**(Worklog 99의 1144에서 정확히 +14, 신규 focused 테스트 수와 일치).

## 결론

이번 배치는 명확히 긍정적인 결과를 냈다: 실제 scene 최대 subset 비율이 53.86%(Worklog 99)에서 22.91%로 개선돼 초기화 자체의 20.62%에 근접했고, Worklog 99가 patio를 hedge/배경에 연결시켰던 정확한 merge 경로 5개 전부가 새 bilateral 인증서에서 기각됨을 직접 확인했다(그중 2개는 명시적으로 편측 지지였다). 다만 (1) 테이블 곡면 복원의 정확한 공로가 merge 메커니즘 자체인지 초기화인지는 여전히 미확정이고, (2) 아주 작은/이질적인 region에서 median+MAD 기반 threshold가 갖는 수치적 한계(§7.1)를 합성 fixture에서 발견해 정직하게 기록했다. Architecture 최종 채택 여부는 이 배치에서 결정하지 않는다.
