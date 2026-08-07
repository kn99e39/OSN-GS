# Worklog 73: Dense Boundary Connectivity Closure Attribution

## 구현

Worklog 72 candidate extraction, scale, threshold, ownership, topology acceptance는 변경하지 않았다. `torch_dense_boundary_connectivity_diagnostics.py`가 각 dense candidate의 `+/- tangent` continuation을 동일 순서(distance/local-scale → reason → normal → tangent → ambiguity → mutuality)로 읽기 전용 재평가한다. 모든 half-line은 정확히 하나의 terminal outcome을 갖고, stage별 proposal/degree/component/cycle, reciprocity loss, candidate extent 측정만 report에 추가했다.

## baseline-compatible@2900

785 candidates, 1,570 half-lines에서 terminal accounting 합계는 정확히 1,570이다.

- `no_candidate_within_local_scale`: 1,068 (68.03%)
- `normal_incompatible`: 58 (3.69%)
- `tangent_incompatible`: 66 (4.20%)
- `ambiguous_competition`: 10 (0.64%)
- `valid_nonreciprocal_neighbor`: 40 (2.55%)
- `valid_reciprocal_neighbor`: 328 (20.89%)

방향 coverage는 both 58 / one 252 / neither 475이다. stage별 closed cycle은 distance 11 → normal 10 → tangent 3 → ambiguity 1 → mutuality 0으로 감소한다. mutuality만으로 제거된 otherwise-valid proposal은 40개다.

후보의 bbox span/evidence bbox span은 모든 region에서 축별 0.89--1.00이므로 support가 작은 sector에 국한된 coverage 실패는 아니다. 이 round의 지배 병목은 **full-evidence local sampling scale에서 continuation candidate가 없는 거리 certificate failure**이며, 그 다음은 tangent/normal compatibility이다. ambiguity와 mutuality는 주된 원인이 아니다. NURBS fitting, gap bridge, threshold 변경, geometric fallback은 수행하지 않았다.

## 검증

`tests/test_region_owned_dense_boundary_support.py`, `tests/test_dense_boundary_connectivity_diagnostics.py`: **3 passed**. Full pytest는 실행하지 않았다.
