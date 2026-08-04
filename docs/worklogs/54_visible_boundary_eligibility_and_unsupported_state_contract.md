# Worklog 54: Visible Boundary Eligibility 및 Unsupported-State 확정

## 결과

region별 5-state production 계약을 `torch_visible_boundary_region_status.py`에 추가했다. `eligible_closed_boundary`, `open_observed_fragment`, `insufficient_observation`, `ambiguous_boundary`, `rejected_unsafe`를 명시하고, `construct_visible_nurbs_from_gaussians`는 eligible component만 materialize한다.

- status는 candidate count, component ordering state, source ID, reason을 보존한다.
- 2-cycle branch budget exhaustion, self-intersection closed loop, ordering capacity failure는 `rejected_unsafe`로 fail-closed한다.
- cap 2048 real replay: 3k=155 region(eligible 0), 5k=147(eligible 2: 130/141), 10k=136(eligible 0).
- Box/Cylinder/Sphere/Thin-slab의 기존 physical/closed/materialized 값은 51/6/6, 16/2/2, 14/0/0, 37/3/3으로 유지됐다.

## 검증

```text
python -m pytest -q tests/test_directed_boundary_ordering.py tests/test_visible_boundary_region_status.py tests/test_visible_surface_construction.py tests/test_boundary_topology_safety.py tests/test_full_cloud_continuation_shell.py tests/test_gaussian_surface_region_formation.py tests/test_surface_region_invariance.py tests/test_boundary_adjacency_semantics.py tests/test_cross_region_continuation.py
105 passed in 46.15s

python -m pytest -q
730 passed, 1 skipped, 1 warning, 8 subtests passed in 281.17s
```
