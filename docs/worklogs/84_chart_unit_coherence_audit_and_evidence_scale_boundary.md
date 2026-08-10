# Worklog 84: chart-unit coherence audit + evidence-scale boundary 표현

## 목적

Worklog 83는 micro-component 364개를 chart unit 178개로 조립했다(evidence의 75~93% 회수, gap bridging 0건). 그러나 no_chart 비율은 91%로 그대로였고, valid_supported는 16→4로 줄었다. 이번 배치는 threshold/kNN/normal/residual 재조정 없이 두 결합된 질문을 한 번에 닫는다: (1) assembled unit이 진짜 하나의 coherent chart인가 over-merge인가, (2) assembled unit이 3~7개 sparse representative node에 의존하지 않고 자기 evidence만으로 chart-scale boundary topology를 세울 수 있는가.

상류 계약 유지: worklog 83 assembly는 현재 제안으로 그대로 보존(fit 품질로 재조정 안 함), sparse representative topology는 macro provenance/typed frontier·crease 제약으로만 사용, worklog 79 coverage, 현재 PCA-UV/6×6 NURBS, visible Gaussian 학습 — 전부 미변경.

## 구현

### 1. Coherence audit (evidence-only, fit 품질 아님)

worklog 82가 이미 쓰던 `internal_normal_disagreement_fraction` 계산(0.15 bound, 동일 공식)을 함수로 분리해(`torch_dense_surface_consistency_components.py`에 추가, 기존 micro-component 호출부는 동일 값을 내도록 검증) **assembled unit 전체**에 재적용한다. micro-component 단계에서 이미 이 검사를 통과한 것들만 assembly에 들어가므로, unit 규모에서 다시 적용하는 것은 새 기준이 아니라 같은 계약의 반복 확인이다.

### 2. Evidence-scale boundary topology (신규 `torch_chart_unit_evidence_scale_boundary.py`)

**첫 설계는 실패로 끝났다**: `extract_dense_boundary_support`가 내부적으로 만드는 `_connect`의 closed-loop(상호 ±tangent half-line 선택)을 그대로 순서로 재사용하려 했으나, real 178개 unit에 직접 실행한 결과 **0/178이 materialize됐다**(107 not_closed, 71 no_dense_support). 이는 worklog 71이 이미 기록한 한계(seed component 282개 중 17개만 closed_loop_recovered, 그나마 대부분 퇴화된 삼각형)를 그대로 재현한 것이다 — `_connect`의 엄격한 상호 매칭은 이 배치의 결함이 아니라 real evidence에서 이미 알려진 한계였다.

**재설계**: `extract_dense_boundary_support`는 **candidate admission**(worklog 77의 보정된 predicate, 미변경 — 이 단계는 신뢰할 수 있음을 확인)에만 쓰고, 승인된 candidate를 **unit 자신의 best-fit tangent plane**(unit evidence의 SVD)에서 **자기 centroid 기준 각도순 정렬**한다. sparse macro node 의존이 전혀 없다. 이 순서는 두 단계로 검증된다(가정하지 않는다):

1. `evaluate_closed_loop_geometry`(worklog 71, 미변경)로 self-intersection 검사 — centroid 기준 star-shaped가 아닌 concave boundary는 `SELF_INTERSECTING`으로 fail-closed.
2. **`measure_edge_support_occupancy`(worklog 76, 원래 disclosure 전용, worklog 83에서 처음 acceptance 신호로 씀)를 loop의 모든 edge(wrap-around 포함)에 적용** — 각도순 정렬은 열린 arc라도 마지막-처음을 잇는 직선으로 항상 어떤 다각형을 닫아버리는데, 그 직선이 관측되지 않은 빈 공간을 가로지르면 self-intersection으로는 잡히지 않는다. 빈 interior bin이 하나라도 있는 edge가 있으면 `UNSUPPORTED_CLOSURE`로 fail-closed — **gap bridging 금지 계약을 실제로 강제하는 마지막 안전장치**다.

sparse macro topology는 이 순서가 이미 만들어진 **뒤에만** 쓰인다: 각 boundary segment를 가장 가까운 sparse arc의 `segment_kind`로 라벨링(기하 아님), 그리고 loop이 crease arc의 다른 쪽에서 들어온 경우를 disclosure(assembly 단계의 crease veto가 이미 막았어야 할 상황을 최종 boundary에서 다시 확인).

`tests/test_chart_unit_evidence_scale_boundary.py` 9개: 평탄한 disc는 coherent, 소수 orthogonal cluster(전체의 28.7%)가 섞이면 incoherent, coherent disc는 sparse arc 없이도 evidence만으로 boundary materialize, sparse arc가 있으면 segment에 라벨 부여, ambiguous/over-merged unit은 boundary 단계에 진입도 못함, 점 3개 미만은 no_dense_support, half-ring(열린 조각)은 자기교차 없이도 `UNSUPPORTED_CLOSURE`로 명시 실패(gap bridging 직접 검증).

### 3. 파이프라인 재구성

신규 `scripts/devtools/dense_chart_unit_boundary_replay.py`: region-owned evidence → worklog 82 micro-component(미변경) → worklog 83 assembly(미변경) → **coherence audit(신규)** → **evidence-scale boundary(신규)** → worklog 79 coverage(신규 모듈 내부에서 적용, 계약 동일) → PCA-UV(미변경) → 6×6 NURBS(미변경) → held-out. 7개 real region 전체 실행.

## 실측 (real baseline_compatible@2900, 7개 region 전체, evidence 3526점)

| reg | evid | units | assembled/coherent% | materialized% | valid% | extrap% | unsafe% | no_chart% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 3 | 92.5/92.5 | 0.0 | 0.0 | 0.0 | 0.0 | 92.5 |
| 1 | 519 | 24 | 89.4/89.4 | 2.5 | 1.3 | 0.0 | 1.2 | 86.9 |
| 2 | 510 | 22 | 96.5/96.5 | 0.8 | 0.8 | 0.0 | 0.0 | 95.7 |
| 3 | 92 | 9 | 80.4/80.4 | 13.0 | 7.6 | 0.0 | 5.4 | 67.4 |
| 4 | 1035 | 56 | 86.6/86.6 | 0.9 | 0.9 | 0.0 | 0.0 | 85.7 |
| 5 | 375 | 7 | 93.3/93.3 | 0.0 | 0.0 | 0.0 | 0.0 | 93.3 |
| 6 | 902 | 57 | 82.7/82.7 | 1.6 | 0.7 | 0.0 | 0.9 | 81.2 |

**전체 가중 합계(evidence 3526점 기준): assembled/coherent 88.1%, materialized 1.5%, valid_supported 0.9%, extrapolative 0.0%, unsafe 0.5%, no_chart 86.7%.**

**coherence audit: 178개 unit 전부(100%) coherent** — internal normal disagreement가 0.15를 넘는 unit이 **0개**다(최대 0.04). 즉 assembly 단계에서 만들어진 unit 중 orientation 불일치로 인한 over-merge는 이번 real 데이터에서 **하나도 발견되지 않았다**.

boundary 실패 원인(전체 168 no_chart): `unsupported_closure` 95건(56.5%), `no_dense_support` 71건(42.3%), `coverage_failed` 2건(1.2%). materialize 성공은 10건(6 valid_supported + 4 unsafe_geometry).

## worklog 83의 16→4(재기준 12→5) 손실 원인 판별

이번 배치의 boundary 방법으로 micro-component 각각을 개별 재평가하면(같은 evaluate_fit 체인, 같은 boundary 방법 적용) valid_supported가 **12개**다(worklog 82가 보고한 16개는 worklog 80의 옛 sparse-arc 방법 기준이라 직접 비교 불가 — 이번 배치는 새 boundary 방법 기준으로 재기준선을 잡았다). 이 12개 micro-component는 assembly 이후 11개 chart unit에 분산됐다(2개가 한 unit에 합쳐진 경우 1건).

11개 unit 전부를 직접 추적한 결과:

| region | unit | member 수 | boundary 결과 | 최종 분류 |
|---:|---:|---:|---|---|
| 1 | 9 | 9 | unsupported_closure | no_chart |
| 1 | 12 | 7 | materialized | **valid_supported** (p95 0.75) |
| 2 | 6 | 4 | materialized | **valid_supported** (p95 0.92) |
| 3 | 6 | 7 | materialized | **valid_supported** (p95 1.03) |
| 4 | 0 | 39 | unsupported_closure | no_chart |
| 4 | 16 | 88 | unsupported_closure | no_chart |
| 4 | 17 | 4 | materialized | **valid_supported** (p95 1.06) |
| 4 | 21 | 17 | unsupported_closure | no_chart |
| 4 | 33 | 5 | materialized | **valid_supported** (p95 0.90) |
| 6 | 4 | 239 | unsupported_closure | no_chart |
| 6 | 14 | 11 | unsupported_closure | no_chart |

**11개 중 5개는 valid_supported를 유지했고, 나머지 6개는 전부 `unsupported_closure`로 실패했다 — `ambiguous_or_over_merged`는 0건이다.**

이것이 판별의 핵심이다: 손실은 **over-merging이 아니다**(over-merge였다면 boundary_state가 `chart_unit_ambiguous_or_over_merged`로 나왔어야 하는데 11개 전부 coherent였다). 손실은 **전부 boundary 안전장치(occupancy gate)가 실제 관측 공백을 잡아낸 것**이다 — 특히 큰 unit일수록(239, 88, 39, 17 members) 실패가 몰려있다: assembled unit이 커질수록 perimeter도 커지는데, dense boundary-support candidate 밀도는 evidence 자체의 관측 밀도에 묶여 있어 큰 unit의 전체 둘레를 빈틈없이 감싸는 candidate 집합을 얻기 어렵다.

## 판정

**(B) chart unit은 조립 가능하지만, 관측된 evidence는 그 대부분에 대해 안전한 chart-scale boundary를 세우기에 아직 불충분하다.**

세부적으로:

- **(C) 기각**: coherence audit이 178개 unit 전부에서 통과했다(0개 ambiguous/over-merged). worklog 83의 aggregate 다중 신호 조립(occupancy 신호 포함)은 실제로 over-merge를 만들지 않았다 — "worklog 83 assembly가 너무 공격적으로 병합해 canonical이 될 수 없다"는 주장은 real 데이터로 직접 반박된다.
- **(A) 기각**: evidence의 88.1%가 assembled+coherent 상태까지 도달하지만, 최종 materialize는 **1.5%**뿐이다. valid_supported는 evidence 기준 0.9%에 불과하다. 이 상태로 Region→Charts를 canonical layer로 채택할 수 없다.
- **16→4(재기준 12→5) 손실은 명확히 "legitimate chart-scale assembly가 하류 boundary 한계를 드러낸 것"이다 — over-merge도 아니고, 혼합도 아니다.** 11개 추적 사례 전부가 `unsupported_closure` 아니면 유지였다. 원인은 assembly 로직의 결함이 아니라, **evidence-scale dense boundary-support candidate의 밀도가 unit이 커질수록 그 전체 perimeter를 gap 없이 감싸기에 부족해진다**는 것이다.

Sparse macro topology 의존을 없애려는 원래 목표(3~7 node로 수백 점 evidence의 perimeter를 감당하지 못한다)는 달성했다 — evidence-scale boundary는 sparse node 없이도 작동한다(9개 test로 확인). 그러나 그 자리를 대신한 evidence-scale candidate 자체가 큰 unit의 전체 둘레를 촘촘히 감싸지 못한다는 것이 새로운, 더 정확한 병목이다.

## 검증

`tests/test_chart_unit_evidence_scale_boundary.py` 신규 9개 전부 통과. 관련 기존 테스트(`test_dense_surface_consistency_components.py`, `test_dense_chart_unit_assembly.py`, `test_boundary_support_spacing.py`) 재실행 36개 전부 통과. **Region→Chart 계층을 canonical로 교체하지 않았으므로**(판정이 A가 아님) 지시대로 전체 regression은 실행하지 않았다. threshold/kNN/normal/residual ablation은 수행하지 않았고, 재사용한 값(0.15 disagreement bound, worklog 77 predicate, worklog 79 0.5 coverage bound)은 전부 기존 계약에서 그대로 가져왔다. hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·region merge·fit 주도 분할/병합은 도입하지 않았다 — 특히 gap bridging 금지는 새로 추가한 occupancy 안전장치가 half-ring fixture로 직접 검증했다. worklog 83 assembly는 fit 품질로 재조정하지 않고 그대로 보존했다. visible Gaussian photometric 학습과 상류 region ownership은 손대지 않았다.
