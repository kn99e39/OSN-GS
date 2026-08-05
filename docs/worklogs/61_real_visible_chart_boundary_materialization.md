# Worklog 61: Real Visible Chart Boundary Materialization

## Region별 chart-boundary 생성 근거

새 모듈 `osn_gs/surface/torch_region_parametric_chart_boundary.py`. 각 region의 **이미 승인된** accepted-edge topology(`SurfaceRegionCandidate.internal_accepted_edge_ids`, 기존 affinity/region-formation 결과, 미변경)만으로 chart boundary를 만든다. Physical termination candidate 유무와 무관하다.

1. Region 전체에 대해 canonical tangent frame(`construct_canonical_region_tangent_frames`, 기존, 미변경) 하나를 골라 2D (u,v) 평면에 투영한다.
2. `internal_accepted_edge_ids`로 만든 인접 그래프에서, extremal(최소 u) 정점을 시작점으로 **leftmost-turn walk**(기존 `torch_patch_boundary._trace_oriented_mask_loops`가 격자 그래프에 쓰는 turn-angle 기법을, 임의 그래프로 일반화한 것)로 outer face를 추적한다. Convex hull/PCA rectangle/bounding box는 전혀 사용하지 않는다 — concave shape과 interior chord가 있는 합성 그래프로 직접 검증했고, chord를 가로지르지 않고 실제 외곽만 따라감을 확인했다(`tests/test_region_parametric_chart_boundary.py`).
3. 단일 simple closed cycle, non-branching(각 정점 정확히 2개 boundary edge), self-intersection 없음(`validate_simple_closed_loop` 재사용)을 만족해야 `eligible_parametric_chart_boundary`. 위상이 3정점 미만/미접속, walk가 닫히지 않음/branch, self-intersection 중 하나면 각각 typed 상태(`parametric_chart_insufficient_topology`/`parametric_chart_topology_open_or_branching`/`parametric_chart_self_intersection_failed`)로 fail-closed.
4. 각 boundary edge는 **양 끝점이 이미 갖고 있는 typed evidence**(기존 `WorldSpaceBoundaryHalfEdgeCandidate.boundary_reason`, 미변경)만으로 분류한다: `observed_support_termination`→`physical_termination`, `crease_discontinuity`→`crease`, `reliability_frontier`/`unresolved_sampling_gap`/`parallel_sheet_conflict`/`ambiguous_continuation`→`observation_frontier`, 그 외(어떤 typed evidence도 없음)→`partition_seam`. Physical evidence로 위장하지 않는다.

`torch_visible_surface_construction.py`에 이 경로를 연결했다: `eligible_parametric_chart_boundary` region만 기존 `materialize_visible_boundary_component()`(변경 없음, 합성 `OrderedBoundaryComponent`로 재사용)로 materialize한다. 기존 `eligible_closed_boundary` → `materialized_visible_nurbs_surfaces` → `eligible_materialized_surfaces()` 경로는 완전히 그대로 두었고, 새 경로는 `materialized_parametric_chart_surfaces` / `eligible_parametric_chart_surfaces()`로 **분리 유지**한다(worklog 56의 continuation bridge 등 기존 소비자가 physical termination을 가정하므로 병합하지 않음).

## Before/after materialized visible NURBS 수

**Real 5k/10k (cap 2048)** — checkpoint가 이전 라운드 이후 완료된 새 학습 run으로 교체돼 physical baseline 자체가 달라졌음을 확인(5k eligible_closed_boundary 2→3, 10k 0→4 — 내 변경과 무관, 순수 학습 진행에 의한 변화이며 기존 physical 경로 코드는 전혀 건드리지 않았다):

| checkpoint | before(physical eligible/materialized) | after(parametric chart eligible/materialized) | combined | insufficient/open·branch/self-intersect |
|---|---:|---:|---:|---|
| 5k | 3 / 3 | 89 / 89 | 92 | 2 / 79 / 8 |
| 10k | 4 / 4 | 74 / 74 | 78 | 7 / 72 / 5 |
| 3k(cap 2048) | — (region formation의 사전 존재 결함으로 crash, 아래 참고) | — | — | — |
| 3k(cap 1024, 대체) | 1 / 1 | 46 / 46 | 47 | 5 / 34 / 3 |

3k는 cap=2048에서 `torch_gaussian_surface_region_formation.py`(worklog 111-123, 이번 라운드 미변경)의 사전 존재 `KeyError`로 crash한다 — 이번 라운드 코드 변경 전 스크립트(worklog 56의 `trace_eligible_boundary_continuation_bridge.py`, 미변경)로도 동일하게 재현되고 `git blame`으로 2026-07-30 코드임을 확인해, 이번 작업과 무관한 checkpoint-content 교체(완료된 새 학습 run)로 노출된 기존 결함임을 확정했다. Region formation은 이번 작업 범위 밖이라 수정하지 않았고, cap=1024로 우회해 real 데이터로 수치를 대체 보고한다.

**Negative control (cap 64)**: physical 경로 Box 6/6, Cylinder 2/2, Sphere 0/0, Thin-slab 3/3 — worklog 47-60 baseline과 완전 동일(byte-identical, 회귀 없음). Parametric chart 경로: Box 4/4 eligible/materialized(segment 전부 physical_termination), Cylinder 1/1(전부 physical_termination), **Sphere 1/1**(physical 0인데도 실제 materialize — arbitrary physical seam은 아니며 `region_status=eligible_parametric_chart_boundary`로 명확히 별도 provenance), Thin-slab 3/3(partition_seam 1건 포함, 명시 disclosure).

## 테스트 결과

```text
python -m pytest -q tests/test_visible_surface_construction.py tests/test_directed_boundary_ordering.py \
    tests/test_visible_boundary_region_status.py tests/test_visible_boundary_materialization_adapter.py \
    tests/test_boundary_topology_safety.py tests/test_full_cloud_continuation_shell.py \
    tests/test_gaussian_surface_region_formation.py tests/test_surface_region_invariance.py \
    tests/test_boundary_adjacency_semantics.py tests/test_cross_region_continuation.py \
    tests/test_region_parametric_chart_boundary.py tests/test_region_parametric_chart_boundary_materialization.py \
    tests/test_eligible_boundary_continuation_bridge.py tests/test_safe_uncertain_proposal_production.py \
    tests/test_safe_uncertain_append_production.py
136 passed, 6 subtests passed in 53.75s (+ 4 passed in region_parametric_chart_boundary_materialization)

python -m pytest -q
772 passed, 1 skipped, 1 warning, 18 subtests passed in 240.81s
```

(worklog 60의 760 passed에서 `test_region_parametric_chart_boundary.py`(8) + `test_region_parametric_chart_boundary_materialization.py`(4) 순증)
