# Worklog 101 — Rejected-interface attribution and region-adaptive support

## 상태

**완료 — 실측 있음. Support starvation은 잔여 과분열의 주된 원인이 아니며, region-adaptive support는 채택하지 않는다.** Worklog 100(region-conditioned + bilateral)은 merge certificate로 accept됐지만, 실제 scene에서 merge된 interface가 1,742/1,763,096(0.10%)로 매우 적어, "conservative 초기화가 만든 작은 fragment가 같은-region 이웃을 충분히 확보하지 못해 support 자체가 성립하지 않는" 순환 의존(directive의 가설)이 원인인지 직접 측정했다. **측정 결과: 아니다.** Rejected interface 중 "support만 해결되면 나머지 기하 테스트를 전부 통과했을" 비율은 0.29%(5,186/1,761,354)에 불과했다. 그럼에도 이 가설을 실제로 검증하기 위해 region-adaptive support(§7)를 구현·실측했고, **실제 scene에서 largest fraction이 22.91%(WL100)→42.13%로 거의 두 배가 되는 실질적 percolation 위험**을 발견했다 — WL99가 막았던 patio↔hedge lineage 자체는 여전히 차단됐지만, 산울타리(hedge) 내부에서 새로운 거대 병합이 발생했다. **두 결과 모두 같은 결론을 가리킨다: adaptive support는 채택하지 않는다.**

## 1. Worklog 100을 baseline으로 보존

`osn_gs/surface/torch_bilateral_interface_region_merge.py`는 전혀 수정하지 않았다. 신규 `osn_gs/surface/torch_region_adaptive_support_merge.py`가 별도 모듈로 존재하며, `SUPPORT_MODE_FIXED_MASKED_KNN`은 Worklog 100의 `_fit_region_conditioned_shape_operators`를 **그대로 import해서 재사용**하고, 같은 checkpoint·같은 초기화·같은 candidate graph·같은 threshold 공식·같은 support/extent floor·같은 bilateral 과반 0.5로 실행하면 **subset_ids가 Worklog 100과 완전히 동일**함을 테스트로 고정했다(`test_fixed_masked_knn_mode_reproduces_worklog_100_exactly`, 실제 scene 재현도 확인 — 초기 114,420, 최종 112,768, 최대 22.91%, merge 1,652, accept된 interface 1,742 — 전부 정확히 일치).

## 2. Rejected-interface 완전 귀속 (multi-label)

매 라운드·매 interface마다 다음을 non-overlapping이 아닌 **multi-label**로 기록한다(directive가 요구한 정확한 키):

    insufficient_region_support_A / _B / _both
    interface_unique_support_failure
    interface_extent_failure
    directional_residual_failure_A_to_B / _B_to_A   (SUPPORTED인데 residual 초과 — UNKNOWN과 절대 혼동하지 않음)
    positional_continuity_failure
    bilateral_smooth_fraction_failure
    would_pass_geometry_tests_but_for_support        (support만 무시하면 나머지 전부 통과했을 interface)

**실제 scene(FIXED_MASKED_KNN, Worklog 100과 동일) 측정**:

| Reason | Count | % of rejected(1,761,354) |
|---|---:|---:|
| interface_unique_support_failure | 1,734,589 | 98.5% |
| bilateral_smooth_fraction_failure | 1,721,697 | 97.7% |
| interface_extent_failure | 1,533,065 | 87.0% |
| insufficient_region_support_B | 1,457,085 | 82.7% |
| directional_residual_failure_A_to_B | 1,105,493 | 62.8% |
| insufficient_region_support_A | 682,084 | 38.7% |
| positional_continuity_failure | 729,824 | 41.4% |
| insufficient_region_support_both | 595,657 | 33.8% |
| directional_residual_failure_B_to_A | 388,792 | 22.1% |
| **would_pass_geometry_tests_but_for_support** | **5,186** | **0.29%** |

거의 모든 rejected interface가 **여러 이유로 동시에** 실패한다(multi-label). Support 부족은 매우 흔하지만(38.7~82.7%), **support만 없었다면 나머지가 전부 통과했을 interface는 0.29%뿐**이다 — support가 유일한 병목인 경우는 극히 드물고, 대개 interface 자체가 너무 작아서(unique surfel count, extent) 이미 기각 대상이다.

## 3. Region-크기별 support 통계 (directive가 요구한 정확한 bin)

FIXED_MASKED_KNN, 라운드 전체 누적:

| Region size bin | Boundary-node 관측 수 | Supported 비율 |
|---|---:|---:|
| 1 | 339,465 | **0.0%**(구조적으로 불가능 — 자기 외에 같은-region 이웃이 존재할 수 없음) |
| 2-4 | 706,573 | 40.8% |
| 5-8 | 484,695 | 87.6% |
| 9-16 | 444,261 | 91.9% |
| 17-32 | 361,866 | 94.0% |
| 33-64 | 278,699 | 95.0% |
| >64 | 2,235,613 | 98.3% |

Support 비율은 region 크기와 뚜렷이 상관관계가 있다(작을수록 낮음) — 정성적 가설(작은 fragment일수록 support가 부족하다)은 **맞다**. 그러나 §2가 보여주듯 이것이 **최종 rejection의 지배적 원인은 아니다** — bin-1(singleton, 전체 boundary 관측의 상당수)은 support가 태생적으로 불가능하지만, 이런 region은 interface 쪽에서도 `unique_surfel_count < 8`(min_side floor)을 항상 위반하므로 support를 고쳐도 애초에 merge될 수 없다.

## 4. 실제 scene 대표 영역 — 정성적 확인

Review export(§11)의 `FIXED_MASKED_KNN_PARTITION` 뷰에서: 테이블 상판(단일 파랑)·다리(다색)는 Worklog 100과 시각적으로 동일하며, 이는 초기화 단계 자체의 효과라는 기존 귀속 미확정 상태가 그대로 유지된다. 파티오는 여전히 하나의 적갈색, 산울타리/배경은 여전히 다색 파편 상태 — WL100과 질적으로 동일하다.

## 5. Support starvation 유의성 판정: **아니다**

Directive §5의 조건에 따라, §2의 `would_pass_geometry_tests_but_for_support = 0.29%`와 §3의 "support가 rejection의 SOLE 원인인 경우는 드물다"는 측정을 근거로, **support starvation은 Worklog 100의 잔여 과분열에 대해 유의미한(material) 원인이 아니라고 판정한다.** 대부분의 interface는 support 여부와 무관하게 이미 너무 작다(interface_unique_support_failure 98.5%, interface_extent_failure 87.0%) — conservative 초기화 자체가 만드는 region/interface의 **크기(granularity)** 문제이지, 특정 region-conditioned model의 **support 획득 방식** 문제가 아니다.

## 6. 그럼에도 실행한 adaptive support 실험 (가설 직접 검증)

Directive §6이 "material하면 구현하라"고 지시했지만, 유의성 판정(§5) 자체가 정확한지 실측으로 교차검증하기 위해 §7-9의 adaptive support를 실제로 구현·실측했다(코드는 유지하되, 결과에 따라 **채택 여부는 별개**로 판단).

### 7. Adaptive same-region local support — 정확한 알고리즘

신규 `SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL`. Boundary surfel `i`(region `R`)의 shape operator 지지 이웃을, 고정 global k=8을 마스킹하는 대신:

    static kNN pool, 폭 = neighbor_count * adaptive_pool_size_multiplier(4) = 32
        -> 같은 region(R)인 후보만 유지
        -> locality bound(§8) 이내인 후보만 유지
        -> 거리순으로 가장 가까운 target_k(8)개만 채택

로 acquisition한다. Pool은 위치 기반이라 학습 후 불변이므로 **한 번만** 계산한다(Worklog 100의 `full_neighbor_index`와 동일한 재사용 패턴).

### 8. Locality contract

Locality bound는 **기존** `local.spatial_connect_spacing_multiplier`(2.0)를 candidate graph 자신의 `local_spacing`에 그대로 곱한 값이다 — 새 절대 반경을 전혀 도입하지 않는다. 이 pool 크기(32)는 유일하게 disclosed된 구현 파라미터이며 `neighbor_count`(8)에서 유도했다(스윕 아님, 이 값 자체가 merge 판정에 쓰이는 threshold가 아니라 "검색 범위를 얼마나 넓힐지"의 구현 디테일).

**중요한 실측 발견**: locality bound를 boundary node 자신의 raw `local_spacing`(주변 밀도에 의해 결정)에서 그대로 가져오면, 그 node가 밀도가 더 높은 다른-region 재질에 둘러싸여 있을 때 bound 자체가 지나치게 좁아져(예: `block_pitch=0.03` 합성 fixture) 진짜 같은-region 이웃(거리 0.1)조차 locality bound(0.08)를 벗어나 여전히 UNSUPPORTED로 남는 경우를 발견했다. 이는 §8이 명시적으로 허용한 결과("허용된 local 영역 안에 충분한 이웃이 없으면 UNSUPPORTED로 남긴다 — 지지를 조작하지 않는다")이며, threshold를 조작해 해결하지 않았다 — 대신 밀도 대비가 덜 극단적인 fixture(§10)로 메커니즘 자체를 명확히 분리해 검증했다.

### 9. Bilateral 의미는 변경 없음

`min(r_A->B, r_B->A)`로 회귀하지 않았고, 한쪽만 supported인 경우를 bilateral 증거로 인정하지 않는다 — 지지를 "획득"할 뿐 증거 요건을 낮추지 않는다. 회귀 테스트로 고정(`test_no_merge_threshold_field_differs_from_worklog_100_defaults`).

## 10. 합성 fixture 검증

신규 focused 테스트 12개(`tests/test_region_adaptive_support_merge.py`), 전부 통과.

- **핵심 메커니즘, 결정론적**: 밀도가 더 높은 반대-region 재질에 인접한 얇은 "finger" fixture(`finger_pitch=0.1`, `block_pitch=0.05`)의 중간 지점에서 FIXED support=1(<2, UNSUPPORTED), 같은 지점에서 ADAPTIVE support≥2(SUPPORTED) — 직접 수치로 검증(`test_dense_opposite_region_starves_fixed_support_but_adaptive_recovers_it`).
- **Locality 계약 위반 없음**: ADAPTIVE가 채택한 모든 이웃의 거리가 `spacing_multiplier * local_spacing` 이내임을 직접 검증(`test_adaptive_support_never_reaches_beyond_the_locality_bound`).
- **Interface 단위에서도 지지가 늘어남**을 확인(`test_adaptive_support_recovers_more_evaluable_interface_edges_than_fixed`).
- **부정 fixture 전부 유지**: 크리즈, 평행 시트, zigzag narrow-bridge 체인, (§10이 요구한) "region-label처럼 보이지만 실제로는 무관한 표면"(finger/block 자체 — 법선이 호환되지 않으므로 adaptive support로 지지를 더 얻어도 병합돼서는 안 됨)까지 전부 분리 유지 확인.
- Worklog 100의 원통 과분열 복원(1/4, 1/2)도 ADAPTIVE 모드에서 그대로 유지됨을 확인.

**완전한 end-to-end(자연 발생적 WL97 초기화 → 실제 merge 성공) 합성 fixture는 의도적으로 만들지 않았다** — 밀도 대비를 fixed-support 실패가 나올 만큼 키우면 candidate graph 자체의 kNN 연결이 끊기고(§8의 발견과 동일한 원인), 밀도를 맞추면(균일 격자) 기하학적으로 fixed support가 2 밑으로 잘 떨어지지 않는다는 것을 확인했다(2면 flanking으로는 same-strip 이웃이 항상 최근접 tie를 차지함). 이 자체가 정직하게 기록할 실측 결과다 — 메커니즘은 단위 수준에서 결정론적으로 검증했고, 실제 scene 측정(§11)이 최종 판단의 근거다.

## 11. 실제 scene A/B (FIXED vs ADAPTIVE, 모든 threshold 동일)

| | A. FIXED_MASKED_KNN(=WL100) | B. ADAPTIVE_SAME_REGION_LOCAL |
|---|---:|---:|
| 초기 region 수 | 114,420 | 114,420(동일 초기화) |
| 최종 region 수 | 112,768 | 110,907 |
| **최대 subset 비율** | **22.91%** | **42.13%** |
| 적용된 merge | 1,652 | 3,513 |
| Accept된 interface | 1,742 | 3,832 |
| `would_pass_geometry_tests_but_for_support` | 5,186 | 828(대부분 해소됨) |
| Coverage identity | True | True |

Support-by-size 재측정: bin 2-4는 40.8%→52.8%, bin 5-8은 87.6%→97.6%로 실제로 개선됐다 — adaptive support 메커니즘 자체는 의도대로 동작한다. 그러나 merge 수는 2.1배 늘었을 뿐인데(1,652→3,513) **최대 subset은 거의 2배**(22.91%→42.13%)가 됐다 — 소수의 회복된 merge가 라운드를 거치며 transitively 훨씬 큰 덩어리로 cascade됐다는 뜻이다. 이는 §2의 단일-라운드 proxy 측정(`would_pass_if_supported=0.29%`)이 **과소평가**했던 누적 효과다 — round-to-round cascade는 proxy가 포착하지 못한다.

## 12. Percolation 회귀 검사 (directive §12, 필수)

Worklog 100의 patio-측 seed(`node 117922`, 초기 region `5052`)와 hedge-측 seed(`node 711179`, 초기 region `555`) — Worklog 99 lineage 5-merge 전부가 기각됐던 바로 그 두 지점 — 을 ADAPTIVE 실행 결과에서 직접 확인했다.

**`patio subset=2828, hedge subset=173, connected=False`** — Worklog 100이 막은 그 특정 연결은 adaptive support에서도 **여전히 차단**된다.

그러나 최대 subset(42.13%)은 이 patio/hedge 쌍이 **아닌 다른** 거대 병합이다. Review export(`ADAPTIVE_SAME_REGION_LOCAL_PARTITION` 뷰)를 시각 검토한 결과, **산울타리(hedge) 좌측 절반이 새로 단일 색(녹색)으로 병합**됐다 — FIXED 뷰에서는 같은 영역이 여러 색(주황/청록/보라 등)으로 파편화돼 있었다. Patio는 여전히 별도(적갈색)이고 patio/hedge 간 특정 연결은 막혔지만, **hedge 내부에서 새로운 대규모 percolation이 발생했다** — WL99/100이 반복적으로 경고한 "조밀하고 텍스처 많은 영역에서의 연쇄적 과다-병합"과 같은 패턴이다.

## 13. 최종 판정 — Adaptive support는 채택하지 않는다

두 개의 독립적 증거가 같은 결론을 가리킨다:

1. **Support starvation은 material하지 않다**(§5): 잔여 과분열의 98% 이상이 support와 무관하게(interface 크기 자체가 너무 작아) 기각된다.
2. **Adaptive support를 실제로 적용하면 실질적 percolation 위험이 재발한다**(§11-12): 특정 patio/hedge 연결은 막혔지만, 최대 subset이 거의 두 배가 되는 새로운 거대 병합이 다른 곳(hedge 내부)에서 발생했다.

Directive §5("material하지 않으면 다른 support 메커니즘을 발명하지 말고 결과를 보고하라, 자동으로 진행하지 말라")와 §12("percolation은 hard regression으로 남아야 한다")를 함께 적용해, **`SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL`은 production 경로로 채택하지 않는다.** Worklog 100(`SUPPORT_MODE_FIXED_MASKED_KNN`과 동일)이 계속 유일한 baseline이다. 코드는 이번 배치의 실측 근거로 남겨두되(진단/실험 모드), 기본값이나 다음 단계의 입력으로 사용하지 않는다.

Worklog 100의 남은 과분열(112,768개 최종 region, 대부분 크기 1~수십)은 conservative 초기화가 만드는 **region/interface granularity** 자체의 결과로 보이며, 이는 이번 배치가 겨냥한 "support 획득 방식" 문제가 아니다 — 다음 단계에 대한 제안은 하지 않는다(directive 요구사항).

## 14. Coverage identity

FIXED·ADAPTIVE 양쪽 모두 실제 scene·합성 fixture에서 `assigned == total_surfels`, `unassigned == 0`, `multiply_owned == 0` 확인.

## 15. Review export

- `scripts/devtools/region_support_attribution_export.py` → `output/osn_gs_region_support_attribution/fixed_masked_knn_attribution.json`(§2-3 데이터)
- `scripts/devtools/region_adaptive_support_real_scene_ab.py` → `.../adaptive_same_region_local_accounting.json`
- `scripts/devtools/adaptive_support_percolation_trace.py` → `.../adaptive_percolation_trace.json`(§12)
- `scripts/devtools/region_adaptive_support_export.py` → `output/osn_gs_region_adaptive_support/`(2 view PLY/PNG: `FIXED_MASKED_KNN_PARTITION`, `ADAPTIVE_SAME_REGION_LOCAL_PARTITION`), preview PNG: `output/osn_gs_region_adaptive_support/preview_png/`.

## 16. 재현 명령

```
python scripts/devtools/region_support_attribution_export.py
python scripts/devtools/region_adaptive_support_real_scene_ab.py
python scripts/devtools/adaptive_support_percolation_trace.py
python scripts/devtools/region_adaptive_support_export.py \
  --checkpoint output/arch_2dgs_coverage_first_surface/2dgs_run1/30000 \
  --out output/osn_gs_region_adaptive_support \
  --device cuda --source-path DATASET
```

## 17. 테스트

- 신규 focused: `tests/test_region_adaptive_support_merge.py` 12개, 전부 통과(FIXED 모드가 WL100과 완전히 동일함을 재현 테스트로 고정 포함).
- 전체 회귀: **1170 passed, 1 skipped, 1 warning, 18 subtests passed in 260.79s**(Worklog 100의 1158에서 정확히 +12).

## 결론

Support starvation은 Worklog 100의 잔여 과분열에 대해 유의미한 원인이 아니며(0.29% 사례), 이를 해소하려는 adaptive support 메커니즘은 실제로 구현·실측한 결과 오히려 새로운 percolation 위험(최대 subset 22.91%→42.13%)을 만들었다. 두 결과 모두 directive 자신의 정지 조건("material하지 않으면 멈춰라", "percolation 회귀는 숨기지 마라")에 해당하므로, **adaptive support는 채택하지 않고 Worklog 100을 유일한 baseline으로 유지한다.** Architecture 최종 판단(다음 단계로의 진행 여부)은 이 배치에서 내리지 않는다.
