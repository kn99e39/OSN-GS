# Worklog 14 — Ordered Boundary to Boundary-first Visible NURBS Materialization Adapter

## 상태

완료(실험적 adapter 범위). Worklog 13의 review-only ordered boundary component를 admissibility 검토 후 canonical `TorchNURBSSurface`로 materialize하는 adapter를 추가했다.

## adapter 계약

- 입력은 `ordered_closed_loop + outer_boundary_candidate + no branch`인 reliable-core-only component와 observed boundary/interior points다.
- open, branch, ambiguous component는 synthetic rectangle closure 없이 `unsupported_topology`로 반환한다.
- phase-alias/nonlocal diagnostic edge, rejected/conflict evidence는 interior/closure support로 사용하지 않는다.
- 결과는 `VisibleBoundaryMaterializationResult`이며 실제 `TorchNURBSSurface`를 보유하고 canonical `evaluate(uv)`를 실행할 수 있다.

## fitting 및 검증

- observed boundary와 observed reliable interior point를 canonical `fit_torch_visible_surface_lsq`에 전달한다.
- LSQ 반환 `(surface, uv)` 계약과 `surface.evaluate(uv)` API를 사용한다.
- finite evaluate 검증과 interior residual을 기록하며, fit/evaluate 실패는 `fit_failed` 또는 `validation_failed`로 보존한다.
- planar closed outer-loop fixture는 materialized/evaluable이고, open component는 closure 없이 unsupported로 유지됨을 회귀로 고정했다.

## 범위

이 adapter는 builder/default dispatcher/trainer/renderer/checkpoint와 연결하지 않은 foundation이다. arbitrary multi-hole ownership, derived seam 생성, patch lifecycle, production readiness는 주장하지 않는다. 다음 Gate는 Gaussian-to-visible-NURBS end-to-end integration 검토다.

## 최종 회귀

- adapter 핵심 회귀(ordered graph, half-edge, phase-alias, central-cap, Boundary-first support 포함): 17 passed
- 전체 회귀 명령: .venv\Scripts\python.exe -B -m pytest -q
- 전체 회귀 결과: 577 passed, 1 skipped, 1 warning, 8 subtests passed in 149.17s
- warning은 기존 torch_voxel_hierarchy.py:108의 requires-grad tensor scalar conversion 경고다.

이 adapter는 reliable-core-only ordered outer loop를 canonical visible NURBS fitting으로 전달하는 실험적 vertical slice다. builder/default dispatcher/trainer/renderer/checkpoint에는 연결하지 않았으며, full surface coverage, arbitrary multi-hole ownership, derived seam 생성, patch lifecycle 또는 production readiness를 주장하지 않는다.