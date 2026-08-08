# Worklog 82: evidence-scale surface-consistency chart-unit 분해 — Region↔Chart 1:1 해체

## 목적

Worklog 81은 parameterization-only 가설을 닫았다: injectivity를 수학적으로 보장하는 대안(intrinsic Tutte embedding)조차 PCA-UV보다 나빴고, 원인은 region이 소유한 evidence 자신이 국소적으로도 평평하지 않다는 것(local normal 불일치 16.3~37.4%, 두께비 0.169~0.546)이었다. 이번 배치는 그 비평탄성을 진단이 아니라 **Region과 Parametric Chart의 암묵적 1:1 관계를 해체**하는 표현 재설계로 다룬다.

상류 계약 유지: Region은 ownership container로 그대로 두고, sparse representative topology는 macro-topology와 typed provenance로만 남긴다(단일 chart 분해 그래프로 쓰지 않음). normal-source architecture, connectivity scale, worklog 77 predicate, UV algorithm, NURBS capacity는 재검토하지 않는다.

## 구현

신규 `osn_gs/surface/torch_dense_surface_consistency_components.py`: **evidence-scale surface-consistency adjacency**.

- **Candidate adjacency**: bounded-degree kNN(기본 8, per-node cap 12) — raw radius graph도 dense clique도 아니다. `torch_gaussian_manifold_affinity.py`(worklog 111/113/114)의 `max_candidate_count_per_node` deterministic-cap 관례를 evidence 밀도에 그대로 재사용했다.
- **Edge acceptance**: normal alignment ≥0.85 AND mutual tangent residual ≤0.35일 때만 `same_surface`로 승인 — 같은 모듈의 `_classify_relation` same_surface 기준을 재사용(재조정 아님). residual은 각 점의 kNN spacing으로 정규화한다(아래 "정규화 선택 검증" 참고).
- **Typed crease veto**: 두 후보 edge의 양 끝점이 worklog 80의 이미 타입된 arc(perimeter arc의 `segment_kind`) 중 **서로 다른** arc에 가장 가깝게 배정되면, normal/residual이 아무리 잘 맞아도 edge를 거부한다 — 기존 provenance를 재사용할 뿐 새 separator를 만들지 않는다.
- **Component**: `same_surface` edge만으로 만든 connected component. crease-vetoed/ambiguous edge는 절대 합치지 않는다(`diagnose_same_surface_regions`와 동일 관례).
- **Fail-closed**: `same_surface` edge가 0개인 점은 미해결(unresolved)로 남긴다(강제 소속 없음). 한 component 내부에서 대표 normal 방향과 60도 넘게 어긋나는 멤버 비율이 15%를 넘으면 `non_manifold_suspected=True`로 표시하고 chart 재료로 쓰지 않는다 — 국소 pairwise 판정만으로는 못 보는 bow-tie/self-crossing을 사후에 잡는다.

`tests/test_dense_surface_consistency_components.py` 7개: 평탄한 sheet가 1개 component로 정확히 묶임, same_surface edge가 실제로 threshold를 만족, 직교하는 두 평탄 sheet가 정확히 2개 component로 분리(먼 kNN 거리로 서로 섞이지 않게 구성), 타입이 다른 두 crease arc가 기하적으로는 통과할 sheet를 실제로 분리, 고립점은 강제 배정되지 않고 unresolved, 빈 region 처리, 무작위 orientation은 30점 전체를 하나로 강제 병합하지 않음.

### Region↔Chart 파이프라인 재구성

신규 `scripts/devtools/dense_surface_consistency_replay.py`: region-owned evidence → **surface-consistency component(신규)** → component별 worklog 80 dense chart support(region 대신 component 단위로 적용, 변경 없음) → worklog 79 coverage 계약(변경 없음) → PCA-UV(worklog 81 확정, 대안 없음) → 6×6 NURBS(변경 없음) → held-out 평가. component의 로컬 tangent frame(PCA)은 worklog 80이 요구하는 arc-투영/UV용 frame으로만 쓰이고 chart 기하 자체를 결정하지 않는다.

## 정규화 선택 검증(구현 검증, threshold 재조정 아님)

region 1에서 residual 정규화를 kNN spacing(내 선택, median 0.056) 대신 대표-그래프 기본값인 `tangent_major_scale`(median 0.030)로 바꾸면 median residual이 0.34→0.63으로 **더 엄격**해진다 — 즉 내 정규화 선택은 두 합리적 옵션 중 이미 **더 관대한** 쪽이며, 분절(fragmentation)이 정규화 버그가 아님을 확인했다. raw normal-방향 변위(kNN pair 기준, median 0.019)는 `normal_thickness`(0.0058)의 3.3배, `tangent_major_scale`(0.030)의 0.63배로, worklog 81이 측정한 국소 불일치(16.3~37.4%)와 독립적으로 같은 결론을 재확인한다 — evidence 자체가 진짜로 국소 잡음이 크다.

## 실측 (real baseline_compatible@2900, 7개 region 전체)

| reg | evid | n_comp | unresolved% | before disagree% | before thick-ratio | valid | extrap | unsafe | no_chart |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 12 | 7.5% | 25.8% | 0.211 | 0 | 1 | 2 | 9 |
| 1 | 519 | 53 | 10.6% | 37.4% | 0.267 | **6** | 2 | 2 | 43 |
| 2 | 510 | 41 | 3.5% | 16.9% | 0.169 | **5** | 1 | 0 | 35 |
| 3 | 92 | 12 | 19.6% | 16.3% | 0.546 | 0 | 1 | 1 | 10 |
| 4 | 1035 | 117 | 13.4% | 35.9% | 0.519 | 0 | 0 | 0 | 117 |
| 5 | 375 | 19 | 6.7% | 28.3% | 0.207 | 0 | 0 | 0 | 19 |
| 6 | 902 | 110 | 17.3% | 34.3% | 0.321 | **5** | 3 | 4 | 98 |

**전 7-region 합계: 364개 component → valid_supported 16, extrapolative 8, unsafe_geometry 9, no_chart 331.**

worklog 79~81 전 라운드에서 valid_supported가 정확히 0이었던 것과 대비된다 — **evidence-scale componentization이 처음으로 사용 가능한 chart를 만들어낸다.** valid_supported held-out p95 평균: region 1 2.05, region 2 1.67, region 6 1.07(전부 EXTRAPOLATION_BOUND=4.0 이내).

그러나 component 크기 분포는 심하게 파편화돼 있다: 중앙값이 3~6점(region evidence 92~1035점 대비), 최대 component(24~157점, `non_manifold_suspected=False`로 국소 일관성 자체는 통과)도 대부분 `extrapolative`나 `no_chart`(worklog 80 dense chart support의 `dense_chart_support_self_intersecting` 또는 macro-topology 부재)로 끝난다. 전체 364개 component 중 **91%(331개)가 no_chart**다. region 4/5는(이미 worklog 80에서 ambiguous branching/open topology로 분류) 이번에도 유효 chart를 하나도 만들지 못했다.

## 원인 판별

파편화가 구현 결함(정규화 스케일 오류)인지 evidence 자체의 성질인지 직접 검증했다(위 "정규화 선택 검증" 참고) — **구현 결함이 아니다.** kNN 인접 evidence 쌍 사이의 raw normal-방향 변위가 `tangent_major_scale`의 0.63배, `normal_thickness`의 3.3배로 이미 크다는 것은 worklog 81의 국소 불일치 측정과 독립적으로 도달한 같은 결론이다 — real evidence는 실제로 국소적으로 잡음이 많다.

## 판정

**(B) evidence는 다중 chart 단위를 지지하지만, 그것을 안전하게 chart 스케일로 조립할 만한 topology 해상도가 아직 없다.**

세부적으로:

- **evidence-scale surface-consistency decomposition은 실재하는 신호를 잡아낸다.** typed crease veto와 normal/residual 기준으로 만든 component 중 16개가 처음으로 `valid_supported`에 도달했다(region 1/2/6). 이는 "evidence가 너무 volumetric/ambiguous해서 boundary-first parametric 가정 자체가 성립하지 않는다"는 (C)를 기각한다 — 최소한 일부 evidence는 진짜로 깨끗한 단일 chart를 이룬다.
- **그러나 현재의 evidence-scale kNN/degree-cap 해상도는 region 전체를 몇 개의 defensible chart 단위로 조립하기엔 너무 국소적이다.** 91%의 component가 no_chart로 끝나고, 중앙값 3~6점짜리 파편이 대부분이다. 즉 evidence는 여러 chart 단위를 지지한다는 신호(구조적 근거: valid_supported 16개, worklog 81의 국소 불일치 16~37%)는 있지만, 그 신호를 chart 하나 분량의 안전한 macro 단위로 묶어낼 topology가 없다 — (A)(그대로 canonical Region→Chart 계층으로 채택)는 지지되지 않는다: 대부분의 evidence가 no_chart로 버려지는 표현을 canonical로 삼을 수 없다.
- **region 4/5는 여전히 유효 chart를 하나도 만들지 못한다** — worklog 80이 이미 ambiguous branching(4)/open topology(5)로 분류한 것과 일치하며, 이번 evidence-scale 시도로도 바뀌지 않았다(강제 참여 없음, 지시대로).

## 검증

`tests/test_dense_surface_consistency_components.py` 신규 7개 전부 통과. 관련 기존 테스트(`test_dense_parametric_chart_support.py`, `test_region_owned_full_evidence.py`) 재실행 34개 전부 통과. **canonical Region→Chart 계층을 교체하지 않았으므로**(판정이 A가 아님) 지시대로 전체 regression은 실행하지 않았다. hull·PCA rectangle·bounding box·alpha shape·강제 분할·gap bridging·region merge·shape-specific fallback은 도입하지 않았고, raw radius graph/dense clique 없이 bounded-degree kNN만 사용했다. normal-source architecture, connectivity scale, worklog 77 predicate, UV algorithm, NURBS capacity는 재검토하지 않았으며 visible Gaussian photometric 학습과 상류 region ownership은 손대지 않았다. threshold(0.85/0.35/degree cap)는 전부 대표-그래프 기존값을 그대로 재사용했고 원하는 chart 개수를 향한 재조정은 하지 않았다.
