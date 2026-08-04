# Worklog 13 — Ordered World-Space Boundary Graph and Boundary Role Recovery

## 상태

완료(실험적 review-only 범위). Worklog 12의 unordered half-edge evidence를 local compatibility graph로 묶어 deterministic boundary component와 역할 후보를 생성했다. builder adapter, chart parameterization, NURBS materialization은 구현하지 않았다.

## 구현된 boundary evidence

- accepted local topology 밖의 crease/parallel/ambiguous/rejected adjacency에 대해 directed half-edge review evidence를 유지한다.
- local tangent-sector continuation을 이용해 `observed_support_termination`과 `unresolved_sampling_gap`을 구분하는 별도 evidence extractor를 추가했다. global world axis, PCA rectangle, convex hull, raster contour는 사용하지 않는다.
- compatibility는 동일 region, local endpoint proximity, tangent/normal alignment, reason compatibility를 함께 요구한다. nearest-neighbor forced join은 하지 않는다.
- component state는 `ordered_closed_loop`, `ordered_open_chain`, `branching_boundary_graph`, `isolated_boundary_candidate`, `ambiguous_ordering`이며, open chain은 강제로 닫지 않는다.
- role candidate는 `outer_boundary_candidate`, `open_boundary_candidate`, `crease_boundary_candidate`, `unresolved_boundary_role`만 review-level로 기록한다.

## 검증 결과

- floor/wall fixture: `crease_discontinuity` candidate 생성.
- phase-alias curved sheet: pairwise shortcut은 존재할 수 있으나 accepted topology shortcut은 0이며 ordering 입력으로 쓰지 않는다.
- compatibility 과연결로 3-node open chain을 false closed loop로 읽던 결함을 발견해 수정했다. local proximity 기준을 보수적으로 적용한 뒤 open chain 유지 회귀를 고정했다.
- bundled targeted suite: **88 passed**.

## real-crop 및 coverage 한계

Worklog 11과 동일한 deterministic 4,000-Gaussian crop의 region 입력은 pairwise 413 components, consensus-aware 80 regions, 최대 29, `ambiguous_unassigned` 3,409, conflict 43, rejected 5이며 giant component는 없다. 이 boundary graph는 reliable-core-only evidence에 한정되고 full observed surface silhouette 또는 object boundary를 주장하지 않는다.

모든 ordered component는 `coverage_semantics = reliable_core_only`, `full_surface_coverage_claimed = false`로 해석해야 한다. `outer_boundary_candidate` 역시 reliable core 외곽 후보일 뿐 full scene outer boundary 확정이 아니다.

## builder adapter 입력 계약

다음 단계는 Visible NURBS materialization adapter이며, 입력으로 stable/core region, accepted local topology, phase-alias-filtered boundary evidence, ordered component state, role candidate, ambiguity/conflict/rejected provenance만 사용해야 한다. diagnostic nonlocal shortcut과 unresolved coverage를 interior/closure evidence로 사용해서는 안 된다.

## 비범위

ordered loop materialization, outer/inner ownership 확정, derived seam 생성, builder adapter, support curve/control grid/NURBS, dispatcher/trainer/renderer/checkpoint 및 production integration은 미착수다.
