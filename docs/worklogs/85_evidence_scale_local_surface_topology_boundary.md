# Worklog 85: evidence-scale local surface topology로 chart-unit boundary 재구성

## 목적

Worklog 84는 centroid 기준 각도순 정렬로 boundary를 만들었다. 이는 일반적인 perimeter-topology 재구성이 아니다 — star-shaped가 아닌 concave 경계에서 정점을 잘못 재배열할 수 있고, unsupported_closure 안전장치가 실제로 자주 발동한 것 자체가 그 증거다. 이번 배치는 이 임시 순서 구성을 **evidence-scale local surface topology**로 완전히 대체한다.

상류 계약 유지: Region ownership, worklog 82 micro-component, worklog 83 assembly, coherence audit, worklog 77 boundary-support admission, sparse macro topology(provenance/typed frontier 전용), worklog 79 coverage, PCA-UV/6×6 NURBS, visible Gaussian 학습 — 전부 미변경.

## 구현

### 1. 재사용 가능한 adjacency builder로 리팩터

`torch_dense_surface_consistency_components.py`에서 kNN+same_surface+crease-veto edge 구성 로직을 `build_same_surface_adjacency()`로 분리했다(기존 `build_dense_surface_consistency_components`는 이 함수를 호출하도록 변경, 동작 불변 — worklog 82 테스트 7개로 확인).

### 2. Local 2-manifold adjacency graph로 boundary 재구성

`torch_chart_unit_evidence_scale_boundary.py`를 in-place로 재작성했다:

1. `extract_dense_boundary_support`는 여전히 **candidate admission만**(worklog 77 predicate, 미변경) 담당.
2. **`build_same_surface_adjacency`(worklog 82의 동일 consistency 기준, 0.85/0.35, 미변경)를 admitted candidate 자신들에게 적용**해 진짜 local adjacency graph를 만든다 — crease veto도 동일하게 적용.
3. 이 그래프에서 **모든 정점의 degree가 정확히 2인 connected component는 그래프 이론상 정확히 하나의 simple cycle이다** — 각도 계산도, 투영도, convex-hull에 준하는 연산도 전혀 없이 순서가 그래프 자체의 walk에서 그대로 나온다. degree≥3인 component는 `branch_detected`, degree≤1인 정점이 있으면 `open_fragment`로 명시(강제 해소 없음). **여러 개의 유효한 degree-2 component는 전부 독립된 boundary loop로 인정**한다(topology가 실제로 증명할 때만).
4. 모든 edge는 관측된 두 점 사이의 실제 local adjacency다 — 발명된 edge나 빈 공간을 가로지르는 chord가 아니다.

**구현 중 실측으로 발견·수정한 자체 결함**: 처음에는 worklog 82의 interior-mesh 기본값(k=8/cap=12)을 그대로 재사용했는데, 합성 ring fixture(24점)에 직접 적용해보니 **모든 후보가 서로 촘촘히 연결되어 degree 4~8의 dense-clique 구조가 되고 진짜 curve topology(degree 2)를 하나도 못 찾았다**(0/178 real unit materialize). 원인은 "탐색 후보 폭(search pool)"과 "최종 허용 degree(topological invariant)"를 혼동한 것 — curve는 정의상 degree>2를 가질 수 없으므로 **cap=2는 튜닝이 아니라 위상적 불변량**이지만, 탐색 자체를 k=2로 제한한 것은 진짜 curve-neighbor를 놓치는 별개의 문제였다(real unit에서 k=2/cap=2는 47개 candidate 중 degree-2가 3개뿐, k를 넓히면 개선되지만 여전히 다수가 미달— k=20/cap=2에서도 12개는 degree 1). **탐색 폭을 무제한(전체 candidate pool)으로 풀고 cap=2만 유지**하도록 수정했다 — cap=2 자체가 dense-clique을 막는 유일하고 충분한 장치이므로 탐색 폭은 제한할 필요가 없다(합성 ring에서 무제한 탐색+cap=2도 여전히 완벽한 degree-2 cycle을 복원함을 재확인).

두 단계의 독립 안전장치로 검증(가정하지 않음, 유효한 loop을 큰 순서로 시도):
- `evaluate_closed_loop_geometry`(worklog 71, 미변경) — self-intersection.
- **`measure_edge_support_occupancy`(worklog 76)를 topology가 만들지 않은 것을 발명하는 데 쓰지 않고, 독립적 최종 검사로만 유지** — 그래프가 만든 edge라도 무제한 탐색으로 인해 실제로는 관측되지 않은 먼 공간을 가로지르는 경우가 real 데이터에서 실제로 발생했다(29건, 아래 참고).

Sparse macro topology는 loop이 이미 만들어진 뒤 segment 라벨링과 crease 일관성 disclosure에만 쓰인다.

`tests/test_chart_unit_evidence_scale_boundary.py` 13개(9개 갱신 + 신규 4개): coherent flat disc가 sparse arc 없이 evidence만으로 materialize, sparse arc 라벨링, ambiguous/over-merged는 boundary 단계 진입 안 함, 점 3개 미만은 no_dense_support, half-ring은 진짜 fail-closed(고정된 실패 사유 하나를 강제하지 않고 유효한 fail-closed 상태 중 하나임을 검증 — 무제한 탐색이 먼 매칭을 만들 수 있음을 반영), **`_find_valid_loops`를 직접 검증하는 신규 4개**(disjoint 2-cycle은 독립된 2개 loop로 인식, degree-3 hub는 loop 0개+branch 1로 disclosure, path는 loop 0개+open 1로 disclosure, 하나의 유효 cycle과 하나의 open fragment가 섞여도 둘 다 올바르게 분리 보고).

### 3. 파이프라인 재실행

`scripts/devtools/dense_chart_unit_boundary_replay.py`를 갱신해 요구된 3-way 명시 분리(진짜 boundary-support evidence 부재 / evidence는 있으나 유효한 manifold topology 없음 / 성공적으로 복원된 supported perimeter)와 branch/open-fragment 집계를 evidence-weighted로 보고하도록 확장했다. 7개 real region 전체 재실행.

## 실측 (real baseline_compatible@2900, 7개 region 전체, evidence 3526점)

| reg | evid | units | no_support% | no_valid_topology% | recovered_perimeter% | valid p95 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 93 | 3 | 3.2 | 89.2 | 0.0 | — |
| 1 | 519 | 24 | 2.7 | 86.7 | 0.0 | — |
| 2 | 510 | 22 | 3.3 | 92.4 | 0.8 | 0.92 |
| 3 | 92 | 9 | 5.4 | 75.0 | 0.0 | — |
| 4 | 1035 | 56 | 5.0 | 81.2 | 0.4 | — |
| 5 | 375 | 7 | 0.5 | 92.8 | 0.0 | — |
| 6 | 902 | 57 | 7.9 | 73.6 | 1.2 | 1.00 |

**전체 가중 합계: 진짜 boundary-support 부재 4.7%, evidence는 있으나 유효한 manifold topology 없음 83.0%, 성공적으로 복원된 supported perimeter 0.5%.**

boundary 실패 원인 세부(전체 178 unit): `no_dense_support` 71, `no_valid_loop_topology`(branch/open) 42, `coverage_failed` 29, `unsupported_closure` 29, `self_intersecting` 3, `materialized` 4(2 valid_supported + 2 unsafe_geometry). `branch_detected`는 **0**이다 — cap=2가 구조적으로 degree>2를 원천 차단하므로 당연한 결과이며, 실패는 전부 open_fragment(연결 부족) 아니면 다운스트림 안전장치(coverage/occupancy/self-intersection)에서 발생했다.

worklog 84의(구조적으로 결함 있는) centroid 정렬 방식의 materialized 1.5%(6 valid + 4 unsafe)보다 이번 결과(0.5%, 2 valid + 2 unsafe)가 **오히려 더 낮다** — 이는 퇴보가 아니라 정직한 결과다: 각도 정렬은 위상적으로 검증되지 않은 순서를 종종 우연히 통과시켰고, 이번 그래프 기반 방법은 실제로 위상을 증명하는 경우에만 통과시킨다.

## 판정

**현재 관측된 Gaussian evidence는 evidence-scale local 2-manifold topology를 통한 boundary-first constructor에 불충분하다.**

- assembled+coherent evidence의 **83.0%**가 evidence는 있지만 유효한 manifold boundary topology를 형성하지 못한다 — sparse macro topology 의존(worklog 84까지의 병목)을 완전히 제거하고, 위상 불변량(degree≤2)만 강제하는 원칙적인 그래프 기반 방법으로도 마찬가지다.
- 이 실패는 알고리즘 결함이 아니라 evidence 자체의 성질임을 직접 검증했다: (1) 합성 ring fixture는 무제한 탐색+degree cap 2로 완벽하게 닫힌다(방법 자체는 건전함), (2) real assembled unit에서는 admitted boundary candidate들이 하나의 연속된 perimeter가 아니라 다수의 작은 조각/개별 조각으로 흩어져 있다(예: region 1 최대 unit 140점 중 admitted candidate 47개가 5+3+3+3+3+3+3=23개(작은 loop 7개)와 10개(open fragment 5개)로 파편화, 하나의 지배적 perimeter loop이 없다).
- `no_dense_support`(4.7%)까지 합치면 coherent evidence의 **87.7%**가 어떤 형태로도 supported perimeter에 도달하지 못한다. 성공(0.5%)은 예외적 사례일 뿐 일반적 경로가 아니다.

**이 배치는 boundary heuristic을 하나 더 추가하는 실험이 아니라, evidence-scale topology 표현 자체를 닫힌 결정으로 만든다.** 지시대로 별도의 고립된 boundary 실험을 새로 시작하지 않는다. worklog 84까지의 낮은 sparse-topology 의존은 해소됐지만, 그 자리를 대신한 evidence 자체가 chart-unit 규모의 완결된 perimeter를 이루기에 충분히 밀집·연속적이지 않다는 것이 최종 결론이다.

## 검증

`tests/test_chart_unit_evidence_scale_boundary.py` 13개 전부 통과(worklog 84의 9개 갱신 + 신규 4개). 관련 기존 테스트(`test_dense_surface_consistency_components.py`, `test_dense_chart_unit_assembly.py`, `test_boundary_support_spacing.py`) 재실행 40개 전부 통과. **evidence-scale topology가 materialized evidence를 유의미하게 늘리지 못했고(오히려 낮아짐) canonical Region→Chart 계층으로 채택할 근거가 없으므로**, 지시대로 전체 regression은 실행하지 않았다. hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·centroid-angle 정렬은 전혀 사용하지 않았다. worklog 77 predicate, worklog 82 same_surface consistency 기준(0.85/0.35)은 재조정하지 않았다 — 변경한 것은 그래프의 **탐색 폭**(search pool, 이번에 발견한 결함 수정)뿐이며 **최종 degree cap(2)는 위상적 불변량**으로 처음부터 끝까지 고정했다. visible Gaussian photometric 학습과 상류 region ownership은 손대지 않았다.
