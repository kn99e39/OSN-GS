# Worklog 97 — Region-level anti-chaining Surfel Subset partition

## 상태

**완료 — 실측 있음.** Worklog 96이 밝힌 병리(intrinsic 2DGS normal을 써도 local pairwise connected component가 single-linkage chaining으로 74.70%짜리 거대 subset을 만듦)를 **partition union rule 하나만** 바꿔 다시 실측했다. Architecture 성공/실패 최종 판단은 여전히 이 배치에서 내리지 않는다 — 사용자가 review export를 시각 검토한 뒤 결정한다.

## 1. Region-orientation 표현

신규 `osn_gs/surface/torch_region_coherent_surfel_partition.py`.

각 region(성장 중이거나 이미 병합된 구조적 그룹)마다 **sign-invariant orientation scatter**를 유지한다:

    M_R = sum_i w_i * n_i n_i^T

`n n^T == (-n)(-n)^T`이므로 이 표현은 로컬 pairwise `|dot(n_i,n_j)|` 테스트와 **정확히 같은 방식으로** normal 부호에 불변이다 — 어떤 normal도 뒤집지 않는다. `w_i`는 이번 배치에서 요구한 대로 **균일 구조 가중치 1.0**(future Trust weight 아님, §14 요구대로 미구현)이다.

Region 전체의 방향 상태는:

    concentration C_R = lambda_max(M_R) / trace(M_R)   in [1/3, 1]

로 요약한다 — 1에 가까울수록 하나의 지배적 방향(평면에 가까움), 1/3에 가까울수록 등방적(방향 증거 없음). `dispersion = 1 - C_R`을 직접 반전 가능한 보조 통계로 함께 기록한다. Eigenvalue는 3x3 대칭행렬의 폐형식(Smith 1961 삼각함수법)으로 직접 계산하고(`_lambda_max_over_trace_batch`), `torch.linalg.eigvalsh`와 대조해 정확성을 확인했다(`tests/test_region_coherent_surfel_partition.py::test_lambda_max_over_trace_matches_numeric_eigensolver`). **이 eigenvalue 계산은 REGION scatter 진단 전용이며 개별 surfel의 normal을 유도하는 데는 전혀 쓰이지 않는다** — 그 사실을 정적 테스트(AST)로 강제한다.

## 2. Anti-chaining merge 규칙과 새 자유 파라미터 여부

**새 독립 파라미터를 도입하지 않았다.** Region-coherence floor는 기존 local `normal_compatibility_min_alignment = a = 0.85`에서 대수적으로 유도한다:

두 단위 normal `n_i, n_j`가 `|dot(n_i,n_j)| = a`일 때(균일 가중치), `M = n_i n_i^T + n_j n_j^T`의 공통 평면 위 고윳값은 `1+a`와 `1-a`, trace는 2이므로

    C_floor = (1 + a) / 2 = (1 + 0.85) / 2 = **0.925**

즉 "두 region의 합집합이 최소한 이미 승인된 pairwise edge 하나만큼은 concentration을 유지해야 병합을 허용한다"는 것이 병합 규칙이다. `a`를 바꾸면 `C_floor`도 같은 관계로 함께 바뀌므로, 코드베이스 전체에서 "normal compatibility"라는 숫자는 여전히 하나뿐이다. 이 유도를 해석적으로(두 normal 폐형식)와 수치적으로(랜덤 3x3 대칭행렬을 `torch.linalg.eigvalsh`와 대조) 모두 테스트로 고정했다.

병합 규칙(architecture directive §4):

    merge(region_i, region_j) 허용 조건 :=
        (A) 유효한 local spatial connectivity  (Worklog 96과 동일 kNN+spacing gate)
        AND (B) local normal 관계가 허용범위    (|dot| >= a, Worklog 96과 동일)
        AND (C) 병합된 region 전체가 여전히 orientation-coherent  (C_merged >= C_floor)  <- 신규

`RegionCoherenceConfig`에 새로 추가된 필드는 `structural_weight`(=1.0, 균일) **하나뿐**이다(`test_only_one_new_free_parameter_is_introduced`로 고정).

## 3. 결정론적 region 구성 알고리즘

Kruskal 스타일 순차 처리(architecture directive §7).

1. Worklog 96과 **완전히 동일한** local candidate graph를 재사용한다 — 신규 `build_candidate_graph()`(Worklog 96의 `partition_gaussian_subsets` 내부 로직을 그대로 추출, 동작 변경 없음을 기존 25개 테스트로 재확인)가 kNN spatial adjacency + local-spacing gate + `|dot|>=a` normal compatibility를 만든다. 두 arm(A/B)이 **같은 함수 호출**로 이 그래프를 얻으므로, 비교는 "동일 local evidence 위에서 union rule만 다른" 진짜 isolated comparison이다.
2. Accepted edge(공간적으로 인접 + normal 호환)를 **alignment 내림차순**(가장 확신도 높은 관계부터), 동률은 `(min_index, max_index)` 오름차순으로 결정론적 정렬한다.
3. 정렬된 순서대로 union-find를 진행한다: 두 root가 다르면 두 root의 현재 M_R을 합친 concentration을 계산해 `>= C_floor`면 병합(작은 index가 살아남는 root, Worklog 96의 min-label 관례와 동일), 아니면 REJECT로 기록한다(그 root쌍의 상태가 이후 바뀌지 않는 한 같은 root쌍을 잇는 다른 edge도 같은 이유로 계속 REJECT되며, 이는 버그가 아니라 boundary view가 보여주는 정보 그 자체다).
4. 순수 Python으로 구현했다(torch 벡터화 아님) — union-find는 root 상태가 매 병합마다 바뀌는 **본질적으로 순차적인** fixpoint라서, 라운드 기반 벡터화는 여러 후보가 동시에 같은 root를 두고 경쟁할 때 "병합 전 상태로 병렬 테스트"가 사실과 다른 결과(3-way 병합이 pairwise로는 통과하지만 합쳐 보면 비정합적인 경우)를 낼 위험이 있다. 실측 결과 순수 Python 순차 루프가 **4,015,325개 accepted edge를 76.4초**(RTX 5080 위 CPU 처리, GPU 병목 아님)에 처리해 벡터화 근사가 굳이 필요하지 않음을 확인했다.

Tie-breaking은 전부 명시적(최소 index가 root로 생존)이고 스레드 스케줄에 의존하지 않는다 — 같은 입력을 두 번 실행하면 `subset_ids`/`partition_role`/`rejected_merge_mask`가 bit-for-bit 동일함을 테스트로 확인했다.

## 4. Ownership propagation 규칙

병합 루프가 끝나면 두 종류의 root가 남는다: 크기 2 이상인 **structural region**(진짜로 병합이 일어난 곳)과 여전히 singleton인 root. 후자는 다음 원칙으로 처리한다(architecture directive §8):

- Singleton 노드의 **accepted local edge**(2단계에서 쓴 것과 동일 목록) 중 structural region 멤버로 이어지는 것이 있으면, **가장 alignment가 높은 edge**를 선택하고(동률이면 더 작은 대상 노드 index) 그 region의 subset id를 물려받는다 — `OWNERSHIP_PROPAGATED_MEMBER`.
  - 이 과정은 **region의 M_R을 절대 갱신하지 않는다** — propagated 멤버는 leaf일 뿐, 두 region을 잇는 다리가 될 수 없다.
  - 여러 서로 다른 structural region에 동시에 닿아 있으면 `ambiguous_multi_region_ownership`으로 기록하고, 그래도 결정론적으로 정확히 하나만 선택한다.
- Structural region에 닿는 accepted edge가 하나도 없으면(자신도 다른 singleton과만 이어져 있거나 accepted edge 자체가 없음) `ISOLATED_FALLBACK_MEMBER`로 자기 자신만의 최종 subset이 된다 — 조용히 사라지지 않는다.

이 propagation은 **완전히 벡터화**된 한 홉짜리 연산이다(union-find처럼 순차적이지 않음 — 각 singleton이 자신의 이웃만 보면 되므로 `scatter_reduce`(amax→amin) 두 패스로 정확하게 계산된다). 이 설계로 "rejected merge가 ownership-only surfel을 통해 간접적으로 재연결될 수 없다"와 "ownership-propagated surfel이 두 structural region을 병합할 수 없다"가 **구조적으로** 성립한다(우회 경로 자체가 코드에 없다) — 두 속성 모두 fixture 테스트로 재확인했다.

## 5. WL96 vs Region-coherent 비교표

checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`(Worklog 96과 동일, 1,197,331 visible surfel). **A와 B는 완전히 동일한 local candidate graph**(candidate 6,048,719 / spatial 5,156,342 / accepted 4,015,325) 위에서 union rule만 다르다.

| 지표 | A. WL96_PAIRWISE_CC | B. REGION_COHERENT |
|---|---:|---:|
| assigned/unassigned/multiply-owned | 1,197,331 / 0 / 0 | 1,197,331 / 0 / 0 |
| Subset 수 | 58,646 | **104,548** |
| Subset 크기 min/median/mean/p95/max | 1/1/20.42/9/894,378 | 1/2/11.45/20/253,853 |
| **최대 subset 비율** | **74.70%** | **21.20%** |
| Singleton subset 수(비율) | 40,410(68.90%) | 40,410(38.65%) |
| 크기 ≤8 subset 수(비율) | 55,390(94.45%) | 91,253(87.28%) |
| Spatially disconnected | 0 | 0(structural region 기준) |
| Local candidate edge | 6,048,719 | 6,048,719(동일) |
| Local normal-compatible edge | 4,015,325 | 4,015,325(동일) |
| **Region-coherence rejected merge** | 해당 없음 | **553,357** |
| Structural-core surfel | 해당 없음(개념 없음) | 1,122,895(93.79%) |
| Ownership-propagated surfel | 해당 없음 | 34,026(2.84%) |
| Isolated-fallback surfel | 40,410(fallback ownership과 동일 개념) | 40,410(3.38%) |
| Ambiguous multi-region ownership | 해당 없음 | 3,377 |

**최대 subset 비율이 74.70% → 21.20%로 3.5배 낮아졌다.** Worklog 105/106(3DGS)의 82.94%, Worklog 96(2DGS pairwise CC)의 74.70%와 비교해 이번이 처음으로 "하나의 subset이 scene 대부분을 잠식"하는 정도가 크게 완화된 실측이다.

## 6. 옛 894,378-surfel 거대 subset의 정확한 분해

| 지표 | 값 |
|---|---:|
| WL96 거대 subset 크기(비율) | 894,378(74.70%) |
| Region-coherent에서 나뉜 최종 subset 수 | **31,564개** |
| 최대 후손 subset 크기 | 253,853 |
| 최대 후손 / 원래 거대 subset 비율 | 28.38% |
| 최대 후손 / 전체 scene 비율 | 21.20%(= B의 최대 subset과 정확히 동일 — 즉 B의 최대 subset은 A의 거대 subset의 직접 후손이다) |
| 2~32번째로 큰 후손 크기 | 39,239 / 8,063 / 7,880 / 7,705 / 7,413 / 7,077 / 6,852 / 6,255 / 5,858 / 5,371 / 4,490 / 4,122 / 3,990 / 3,702 / 3,284 / 2,911 / 2,877 / 2,425 / 2,419 / 2,266 / 2,068 / 1,979 / 1,933 / 1,872 / 1,840 / 1,836 / 1,791 / 1,723 / 1,721 / 1,684 / 1,684 |

거대 subset 하나가 31,564개의 서로 다른 최종 subset으로 갈라졌다 — 급격한 크기 감쇠(2번째가 1번째의 15.5%, 3번째는 1번째의 3.2%)를 보인다. 시각 검토(§10) 결과 남은 최대 subset(21.20%)은 **평평한 patio 바닥 하나**이고, 이전에 그것과 합쳐져 있던 굴곡진 산울타리/배경은 수천 개의 작은 subset으로 흩어졌다.

## 7. WL96 singleton/fallback surfel의 정확한 행방

| 지표 | 값 |
|---|---:|
| WL96 singleton surfel 수 | 40,410 |
| → structural core로 전환 | **0** |
| → ownership-propagated로 전환 | **0** |
| → isolated fallback 유지 | **40,410**(100%) |

WL96에서 singleton이었던 surfel은 region-coherent partition에서도 **정확히 같은 40,410개가 그대로 isolated fallback으로 남는다.** 이는 버그가 아니라 두 arm이 **동일한 local candidate graph**를 쓴다는 사실의 직접적 결과다 — 이 surfel들은 애초에 WL96에서도 accepted local edge가 하나도 없었으므로(spatial 이웃이 없거나 normal이 전부 비호환), region-coherent 쪽의 ownership propagation도 참조할 accepted edge 자체가 없어 구제될 수 없다. 이 batch는 이 숫자를 "줄여야 할 목표"로 삼지 않았고(architecture directive §9), 실제로 강제 병합도 하지 않았다 — 정확한 행방만 보고한다.

## 8. Coverage identity 증명

`region_coherent_accounting()`의 `coverage_identity_holds`가 `assigned==count`, `in_range==count`, `subset_sizes_match_ownership_map`(독립적으로 유도된 두 값의 원소 단위 대조), `sum(subset_sizes)==count`를 전부 기계적으로 검증한다 — 실측 결과 **1,197,331개 전부 assigned, unassigned 0, multiply-owned 0.** `partition_role_counts`의 합(1,122,895+34,026+40,410=1,197,331)도 정확히 일치한다.

## 9. Region orientation-dispersion 통계

Structural region(64,138개, singleton은 제외) 대상, unweighted(region당 1개 데이터 포인트):

| 지표 | 값 |
|---|---:|
| Concentration median | 0.9577 |
| Concentration p05 | 0.9262(= floor 0.925 바로 위 — 병합 규칙이 실제로 경계에서 작동함을 보여준다) |
| Concentration p95 | 0.9966 |
| Dispersion median | 0.0423 |
| Dispersion p95 | 0.0738 |

Concentration p05가 floor(0.925)에 거의 붙어 있다는 것은 다수 region이 **정확히 허용 한계까지** 성장했다는 뜻이다 — 임의로 보수적인 값을 쓴 게 아니라 유도된 floor가 실제로 경계에서 결정을 내리고 있다는 확인이다.

## 10. Review export

`output/osn_gs_region_coherent_surfel_partition/`(전체 scene, crop 없음, Worklog 96과 동일 카메라·색상 계열):

| view | 경로 |
|---|---|
| A. 2DGS_ORIGINAL_SCENE | `2DGS_ORIGINAL_SCENE/{iteration_0000001/point_cloud.ply, render.ppm}` |
| B. WL96_PAIRWISE_CC_PARTITION | `WL96_PAIRWISE_CC_PARTITION/{...}` |
| C. REGION_COHERENT_PARTITION | `REGION_COHERENT_PARTITION/{...}` |
| D. REGION_ORIENTATION_DISPERSION_VIEW | `REGION_ORIENTATION_DISPERSION_VIEW/{...}` |
| E. OWNERSHIP_ROLE_VIEW | `OWNERSHIP_ROLE_VIEW/{...}`(초록=structural core, 노랑=ownership-propagated, 빨강=isolated fallback) |
| F. ANTI_CHAINING_BOUNDARY_VIEW | `ANTI_CHAINING_BOUNDARY_VIEW/{iteration_0000001/point_cloud.ply,nurbs_surface.json, render.ppm}` |
| 회계 | `region_coherent_partition_report.json` |

**시각 확인**: B(WL96)는 바닥+산울타리가 하나의 적갈색 subset으로 붙어 있다. C(region-coherent)는 바닥만 크게 남고 산울타리는 수백 가지 색으로 잘게 쪼개졌으며, 테이블 상판(평면)과 다리(곡률/방향 변화)도 분리됐다. F(anti-chaining boundary)는 rejected merge가 바닥-산울타리 전이 구간과 테이블 디테일 주변에 집중돼 있음을 보여준다 — 병합이 실제로 orientation이 바뀌는 곳에서 막혔다는 시각적 증거다. D(dispersion view)는 큰 region일수록 floor 근처(고분산)에 몰려 있어 대부분 주황으로 나타난다 — surfel 수 기준 가중 분포이므로 나온 그대로 보고하며, 렌더를 "보기 좋게" 만들려고 추가로 손대지 않았다(정규화 기준을 raw max에서 `region_orientation`의 p05/p95로 한 번 바꿨을 뿐 — 이는 보고되는 통계 자체를 색상 기준으로 쓴 것이지 결과를 보고 반복 조정한 것이 아니다).

## 11. 재현 명령

```
python scripts/devtools/region_coherent_surfel_partition_export.py \
    --checkpoint output/arch_2dgs_coverage_first_surface/2dgs_run1/30000 \
    --out output/osn_gs_region_coherent_surfel_partition \
    --device cuda --source-path DATASET --images images_8
```

런타임: A 72.1초 + B 76.4초(순차 Kruskal-gated merge 포함) + rendering, 총 약 150~180초(RTX 5080). kNN을 두 arm이 각자 재계산한다(캐시 공유 없음, 총 두 번) — 결과 동일성은 §3의 결정론 테스트로 보장되므로 재계산 비용은 안전성보다 우선하지 않았다.

## 12. WL96 CUDA 테스트 연대기 정정

Worklog 96 §5-A의 "빌드 직후 21개 CUDA 테스트가 pass로 바뀌었다(§8에서 재확인)"는 문구가 부정확했다 — §8의 `1077 passed, 22 skipped` 전체 회귀는 CUDA 확장 빌드 **이전**(최초 구현 커밋 시점)에 실행된 것이었고, 빌드 후에는 해당 2개 테스트 파일만 개별 실행했을 뿐 전체 회귀를 다시 돌리지 않았다. Worklog 96 §5-A에 정정 문구를 추가했다(역사를 조용히 다시 쓰지 않고, 정정 사실 자체를 남겼다). 이 배치(§13)에서 CUDA 확장이 빌드된 상태로 처음 전체 회귀를 실행해 정확한 현재 상태를 기록한다.

## 13. 검증

**Focused 테스트 15개 신규**(`tests/test_region_coherent_surfel_partition.py`), 전부 통과: concentration floor가 두 normal 폐형식과 정확히 일치 / 폐형식 eigenvalue 계산이 수치 eigensolver와 일치 / **누적 drift가 있는 chain은 하나의 region이 되지 않음**(15도씩 19단계, WL96 plain CC는 여전히 1개 subset이 되는 negative control로 재확인) / **완만한 곡률 sheet는 하나의 region으로 유지됨** / 순수 평면 sheet가 전부 structural core로 하나의 region / 부호 뒤집힌 normal이 동일한 partition을 만듦 / 반복 실행이 결정론적(subset_ids/role/rejected_mask 전부 bit-identical) / **rejected region merge가 ownership-only surfel로 간접 재연결되지 않음** / **ownership-propagated surfel이 두 structural region을 병합하지 않음** / 모든 surfel이 정확히 하나의 owner를 가짐 / structural region이 spatially connected / 원본 텐서 불변 / 빈 입력·단일 입력도 coverage 정확 / 이 모듈이 per-surfel normal을 유도하지 않음(AST) / 새 자유 파라미터가 `structural_weight` 하나뿐임을 고정.

**전체 회귀(CUDA surfel 확장 빌드된 상태로 처음 실행)**: `1113 passed, 1 skipped, 1 warning, 18 subtests passed in 251.92s`. Worklog 96 커밋 시점 기준선(`1077 passed, 22 skipped`)과 정확히 대조된다 — skip이 22개에서 **1개**로 줄었다(surfel CUDA 21개가 skip→pass로 전환, §12), 그리고 pass가 1077→1113으로 36개 늘었다(신규 region-coherent 테스트 15개 + 이번에 처음 pass로 집계된 CUDA surfel 테스트 21개 = 36, 정확히 일치). 실패·회귀 없음.

## 결론 없음

이 worklog는 region-level orientation coherence가 최종 architecture로 채택할 만한지 판단하지 않는다. 실측은 명확하다 — 최대 subset 비율이 74.70%에서 21.20%로 크게 줄었고, 남은 최대 subset은 시각적으로 진짜 하나의 평평한 표면(바닥)이지 chaining의 산물로 보이지 않는다. 그러나 이것이 "충분히 해결됐다"는 뜻인지, 남은 21.20%와 553,357개의 rejected merge가 다음 단계(subset-local Trust)에서 어떻게 다뤄져야 하는지는 사용자가 §10의 6개 view를 직접 검토한 뒤 결정한다.
