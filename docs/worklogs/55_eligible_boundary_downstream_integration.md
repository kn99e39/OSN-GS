# Worklog 55: Eligible Boundary Downstream Integration

## 결과

`torch_visible_surface_construction.py`를 authoritative gate로 고정했다. status의 `eligible_component_ids`는 `find_inconsistent_eligible_component_ids()` 검사를 통과한 경우에만 materialization adapter로 전달된다.

- `VisibleBoundaryMaterializationInput`에 region status/reason, boundary role scope, source ID provenance를 전달한다.
- `eligible_materialized_surfaces()`를 downstream의 유일한 visible-surface 진입점으로 추가했다.
- outer/hole 판별은 아직 구현되지 않았으므로 scope를 `outer_boundary_only`로 명시한다. 여러 valid closed loop는 임의로 제거하지 않고 독립 materialize하며 진단에 노출한다.
- real cap 2048: 3k/10k eligible 0, 5k region 130/141 두 개만 eligible 및 materialized; inconsistency는 전부 0.

## 검증

```text
python -m pytest -q tests/test_visible_surface_construction.py tests/test_directed_boundary_ordering.py tests/test_visible_boundary_region_status.py tests/test_visible_boundary_materialization_adapter.py tests/test_boundary_topology_safety.py tests/test_full_cloud_continuation_shell.py tests/test_gaussian_surface_region_formation.py tests/test_surface_region_invariance.py tests/test_boundary_adjacency_semantics.py tests/test_cross_region_continuation.py
107 passed in 46.17s

python -m pytest -q
731 passed, 1 skipped, 1 warning, 8 subtests passed in 247.14s
```
